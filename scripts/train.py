"""
Training script for KeyPilot model.

Usage:
    python scripts/train.py --config configs/model_config.yaml
"""

import os
import sys
import argparse
import yaml
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import get_cosine_schedule_with_warmup
from tqdm import tqdm

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from keypilot.models import KeyPilotVLM, KeyPilotLoss, create_keypilot_model
from keypilot.data.dataset import KeyPilotDataset
from loguru import logger

try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False
    logger.warning("wandb not installed, W&B logging will be disabled")


def parse_args():
    parser = argparse.ArgumentParser(description="Train KeyPilot model")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/model_config.yaml",
        help="Path to configuration file"
    )
    parser.add_argument(
        "--train_data",
        type=str,
        default="data/raw/data/train/all_samples_343.pkl",
        help="Path to training data pickle file"
    )
    parser.add_argument(
        "--val_data",
        type=str,
        default="data/raw/data/test/all_samples_343.pkl",
        help="Path to validation data pickle file"
    )
    parser.add_argument(
        "--output_train",
        type=str,
        default="results/train",
        help="Output directory for training results (checkpoints/, models/, log/)"
    )
    parser.add_argument(
        "--output_valid",
        type=str,
        default="results/valid",
        help="Output directory for validation results during training"
    )
    parser.add_argument(
        "--output_test",
        type=str,
        default="results/test",
        help="Output directory for test results (used by test.py script)"
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Maximum number of samples to use (for testing)"
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        default=True,
        help="Resume training from latest checkpoint (default: True). Use --no-resume to start fresh."
    )
    parser.add_argument(
        "--no-resume",
        dest="resume",
        action="store_false",
        help="Start fresh training and delete existing checkpoints"
    )
    parser.add_argument(
        "--use_wandb",
        action="store_true",
        help="Use Weights & Biases for logging"
    )
    parser.add_argument(
        "--gpu",
        type=int,
        default=None,
        help="GPU device id to use (e.g., 0, 1, 2). If not specified, uses all available GPUs or CPU"
    )
    parser.add_argument(
        "--local_rank",
        type=int,
        default=-1,
        help="Local rank for distributed training"
    )
    return parser.parse_args()


def setup_training(config, args):
    """Setup training environment."""
    # Setup device
    if args.local_rank == -1:
        if args.gpu is not None:
            # Use specified GPU
            if not torch.cuda.is_available():
                raise ValueError(f"CUDA is not available, cannot use GPU {args.gpu}")
            if args.gpu >= torch.cuda.device_count():
                raise ValueError(f"GPU {args.gpu} is not available. Available GPUs: 0-{torch.cuda.device_count()-1}")
            
            device = torch.device(f"cuda:{args.gpu}")
            torch.cuda.set_device(args.gpu)
            n_gpu = 1
            logger.info(f"Using GPU {args.gpu}: {torch.cuda.get_device_name(args.gpu)}")
        else:
            # Use all available GPUs or CPU
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            n_gpu = torch.cuda.device_count() if torch.cuda.is_available() else 0
    else:
        # Distributed training
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend="nccl")
        n_gpu = 1
    
    logger.info(f"Device: {device}, n_gpu: {n_gpu}")
    
    # Setup output directory structure for training
    train_dir = Path(args.output_train)
    checkpoints_dir = train_dir / "checkpoints"
    models_dir = train_dir / "models"
    log_dir = train_dir / "log"
    
    # Create training directories
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup output directory for validation/evaluation results
    valid_dir = Path(args.output_valid)
    valid_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup file logging
    log_file = log_dir / f"training_{Path(args.output_train).name}.log"
    logger.add(log_file, rotation="500 MB", level="INFO")
    logger.info(f"Logging to {log_file}")
    logger.info(f"Output structure:")
    logger.info(f"  Training outputs:")
    logger.info(f"    - Checkpoints: {checkpoints_dir}")
    logger.info(f"    - Best models: {models_dir}")
    logger.info(f"    - Logs: {log_dir}")
    logger.info(f"  Validation/Evaluation outputs:")
    logger.info(f"    - Validation results: {valid_dir}")
    
    # Setup W&B
    if args.use_wandb and HAS_WANDB and args.local_rank in [-1, 0]:
        wandb.init(
            project="keypilot",
            config=config,
            name=f"keypilot_{config['model']['name']}"
        )
        logger.info("Weights & Biases initialized")
    
    return device, n_gpu


def create_optimizer(model, config):
    """Create optimizer with layer-wise learning rates."""
    optimizer_config = config['training']['optimizer']
    
    # Separate encoder and decoder parameters
    encoder_params = []
    decoder_params = []
    
    for name, param in model.named_parameters():
        if param.requires_grad:
            if 'encoder' in name:
                encoder_params.append(param)
            else:
                decoder_params.append(param)
    
    optimizer = torch.optim.AdamW([
        {'params': encoder_params, 'lr': optimizer_config['lr'] * 0.5},  # Lower LR for encoder
        {'params': decoder_params, 'lr': optimizer_config['lr']}
    ],
    weight_decay=optimizer_config['weight_decay'],
    betas=optimizer_config['betas'],
    eps=optimizer_config['eps'])
    
    return optimizer


def create_scheduler(optimizer, config):
    """Create learning rate scheduler."""
    scheduler_config = config['training']['scheduler']
    
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=scheduler_config['warmup_steps'],
        num_training_steps=scheduler_config['total_steps']
    )
    
    return scheduler


def train_epoch(model, dataloader, criterion, optimizer, scheduler, device, config, epoch, args):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    total_task_correct = 0
    total_layout_correct = 0
    total_samples = 0
    
    gradient_accumulation_steps = config['training']['gradient_accumulation_steps']
    
    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch + 1}")
    
    for step, batch in enumerate(progress_bar):
        # Move batch to device
        image = batch['image'].to(device)
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        task_label = batch['task_label'].to(device)
        layout_label = batch['layout_label'].to(device)
        
        # Forward pass
        outputs = model(
            image=image,
            input_ids=input_ids,
            attention_mask=attention_mask,
            user_id=None,
            prev_layout=None,
            return_aux=True
        )
        
        # Prepare targets for loss computation
        targets = {
            'target_task': task_label,
            'target_layout': layout_label,
            'target_tokens': batch['target_token_ids'].to(device),
            'target_type': batch['target_type'].to(device),
        }
        
        # Compute multitask loss using KeyPilotLoss
        loss_dict = criterion(outputs, targets)
        loss = loss_dict['loss_total']
        loss = loss / gradient_accumulation_steps
        
        # Backward pass
        loss.backward()
        
        if (step + 1) % gradient_accumulation_steps == 0:
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                config['training']['max_grad_norm']
            )
            
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
        
        # Compute accuracies
        pred_task = outputs['task_probs'].argmax(dim=-1)
        pred_layout = outputs['layout_id']
        
        total_task_correct += (pred_task == task_label).sum().item()
        total_layout_correct += (pred_layout == layout_label).sum().item()
        total_samples += image.size(0)
        
        # Update metrics
        total_loss += loss.item() * gradient_accumulation_steps
        
        # Update progress bar (show text loss if available)
        progress_info = {
            'loss': total_loss / (step + 1),
            'task_acc': total_task_correct / total_samples,
            'layout_acc': total_layout_correct / total_samples,
            'lr': scheduler.get_last_lr()[0]
        }
        if 'loss_text' in loss_dict:
            progress_info['text_loss'] = loss_dict['loss_text'].item()
        
        progress_bar.set_postfix(progress_info)
        
        # Log to W&B
        if args.use_wandb and HAS_WANDB and step % config['training']['logging_steps'] == 0:
            # Prepare detailed loss logging
            log_dict = {
                'train/loss_total': loss.item() * gradient_accumulation_steps,
                'train/task_accuracy': total_task_correct / total_samples,
                'train/layout_accuracy': total_layout_correct / total_samples,
                'train/lr': scheduler.get_last_lr()[0],
                'step': epoch * len(dataloader) + step
            }
            
            # Add individual loss components if available
            for key, value in loss_dict.items():
                if key != 'loss_total' and isinstance(value, torch.Tensor):
                    log_dict[f'train/{key}'] = value.item()
            
            wandb.log(log_dict)
    
    metrics = {
        'loss': total_loss / len(dataloader),
        'task_accuracy': total_task_correct / total_samples,
        'layout_accuracy': total_layout_correct / total_samples
    }
    
    return metrics


def evaluate(model, dataloader, device, criterion, output_dir=None, epoch=None):
    """
    Evaluate model and optionally save results.
    
    Args:
        model: Model to evaluate
        dataloader: Validation/test dataloader
        device: Device to run evaluation on
        criterion: Loss criterion (KeyPilotLoss)
        output_dir: Optional directory to save evaluation results
        epoch: Optional epoch number for result naming
    """
    model.eval()
    total_loss = 0
    total_task_correct = 0
    total_layout_correct = 0
    total_text_correct = 0  # Token-level accuracy
    total_samples = 0
    total_tokens = 0
    
    # For detailed loss tracking
    loss_components = {
        'loss_text': 0,
        'loss_task': 0,
        'loss_layout': 0,
        'loss_encoder': 0,
        'loss_decoder': 0,
    }
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            # Move batch to device
            image = batch['image'].to(device)
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            task_label = batch['task_label'].to(device)
            layout_label = batch['layout_label'].to(device)
            target_token_ids = batch['target_token_ids'].to(device)
            target_length = batch['target_length'].to(device)
            
            # Forward pass
            outputs = model(
                image=image,
                input_ids=input_ids,
                attention_mask=attention_mask,
                user_id=None,
                return_aux=True
            )
            
            # Prepare targets for loss computation
            targets = {
                'target_task': task_label,
                'target_layout': layout_label,
                'target_tokens': target_token_ids,
                'target_type': batch['target_type'].to(device),
            }
            
            # Compute multitask loss using KeyPilotLoss
            loss_dict = criterion(outputs, targets)
            total_loss += loss_dict['loss_total'].item()
            
            # Accumulate loss components
            for key in loss_components.keys():
                if key in loss_dict:
                    loss_components[key] += loss_dict[key].item()
            
            # Compute task and layout accuracies
            pred_task = outputs['task_probs'].argmax(dim=-1)
            pred_layout = outputs['layout_id']
            
            total_task_correct += (pred_task == task_label).sum().item()
            total_layout_correct += (pred_layout == layout_label).sum().item()
            
            # Compute text prediction accuracy (token-level)
            # For auto-completion and error-correction: use top-N accuracy
            # For suggestion: use exact match accuracy
            if 'logits' in outputs:
                logits = outputs['logits']  # [B, seq_len, vocab_size] or [B, vocab_size]
                
                if logits.dim() == 3:
                    # Multi-token prediction (suggestion task)
                    # Use exact match for suggestion task
                    pred_tokens = logits.argmax(dim=-1)  # [B, seq_len]
                    pred_length = pred_tokens.size(1)  # Always use prediction length

                    # Compare up to prediction length for each sample
                    for i in range(pred_tokens.size(0)):
                        pred = pred_tokens[i, :pred_length]
                        # Always compare up to prediction length
                        target = target_token_ids[i, :pred_length]
                        total_text_correct += (pred == target).sum().item()
                        total_tokens += pred_length
                else:
                    # Single-token prediction (completion/error tasks)
                    # Use top-N accuracy: correct if ground truth is in top N candidates
                    num_candidates = 5  # Default N=5
                    target = target_token_ids[:, 0]  # Take first token [B]
                    
                    # Get top-N predictions
                    top_n_probs, top_n_tokens = torch.topk(logits, k=num_candidates, dim=-1)
                    # top_n_tokens: [B, N]
                    
                    # Check if target is in any of the N candidates
                    target_expanded = target.unsqueeze(1).expand_as(top_n_tokens)  # [B, N]
                    matches = (top_n_tokens == target_expanded).any(dim=1)  # [B]
                    total_text_correct += matches.sum().item()
                    total_tokens += target.size(0)
            
            total_samples += image.size(0)
    
    # Compute average metrics
    metrics = {
        'eval/loss': total_loss / len(dataloader),
        'eval/task_accuracy': total_task_correct / total_samples,
        'eval/layout_accuracy': total_layout_correct / total_samples,
        'eval/text_accuracy': total_text_correct / total_tokens if total_tokens > 0 else 0.0,
    }
    
    # Add loss components
    for key, value in loss_components.items():
        metrics[f'eval/{key}'] = value / len(dataloader)
    
    logger.info(f"Eval metrics: {metrics}")
    
    # Save evaluation results if output_dir is provided
    if output_dir is not None:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save metrics to JSON
        import json
        epoch_suffix = f"_epoch{epoch}" if epoch is not None else ""
        metrics_file = output_path / f"eval_metrics{epoch_suffix}.json"
        with open(metrics_file, 'w') as f:
            json.dump(metrics, f, indent=2)
        
        logger.info(f"Evaluation results saved to {metrics_file}")
    
    return metrics


def save_checkpoint(model, optimizer, scheduler, epoch, output_dir, config, is_best=False):
    """
    Save training checkpoint.
    
    Args:
        model: Model to save
        optimizer: Optimizer state
        scheduler: Scheduler state
        epoch: Current epoch number
        output_dir: Base output directory
        config: Model configuration
        is_best: Whether this is the best model so far
    """
    output_dir = Path(output_dir)
    
    # Save checkpoint to checkpoints/ directory
    checkpoint_dir = output_dir / "checkpoints" / f"checkpoint-epoch{epoch}"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    # Save full checkpoint with training state
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'config': config
    }, checkpoint_dir / "pytorch_model.bin")
    
    # Save config
    with open(checkpoint_dir / "config.yaml", 'w') as f:
        yaml.dump(config, f)
    
    logger.info(f"Checkpoint saved to {checkpoint_dir}")
    
    # Save best model to models/ directory
    if is_best:
        best_dir = output_dir / "models" / "best_model"
        best_dir.mkdir(parents=True, exist_ok=True)
        
        # Save model only (no optimizer/scheduler for inference)
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'config': config
        }, best_dir / "pytorch_model.bin")
        
        # Save config
        with open(best_dir / "config.yaml", 'w') as f:
            yaml.dump(config, f)
        
        logger.info(f"Best model saved to {best_dir}")


def main():
    args = parse_args()
    
    # Load configuration
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Setup training
    device, n_gpu = setup_training(config, args)
    
    # Create dataloaders
    logger.info("Loading datasets...")
    train_dataset = KeyPilotDataset(
        data_path=args.train_data,
        max_seq_length=config['data']['text']['max_length'],
        image_size=(config['data']['image']['height'], config['data']['image']['width']),
        max_samples=args.max_samples,
        use_bert_tokenizer=True
    )
    
    val_dataset = KeyPilotDataset(
        data_path=args.val_data,
        max_seq_length=config['data']['text']['max_length'],
        image_size=(config['data']['image']['height'], config['data']['image']['width']),
        max_samples=args.max_samples // 10 if args.max_samples else None,
        use_bert_tokenizer=True
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['training']['batch_size'] * 2,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    logger.info(f"Train dataset: {len(train_dataset)} samples")
    logger.info(f"Val dataset: {len(val_dataset)} samples")
    
    # Create model
    logger.info("Creating model...")
    model = create_keypilot_model(config['model'])
    model.to(device)
    
    # Print model summary
    summary = model.get_model_summary()
    logger.info(f"Model summary:")
    logger.info(f"  Total parameters: {summary['total_parameters']:,}")
    logger.info(f"  Model size (FP32): {summary['model_size_mb']:.2f} MB")
    logger.info(f"  Model size (INT8): {summary['model_size_mb_int8']:.2f} MB")
    
    # Create criterion
    criterion = KeyPilotLoss(**config['training']['loss'])
    
    # Create optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config['training']['optimizer']['lr'],
        weight_decay=config['training']['optimizer']['weight_decay'],
        betas=config['training']['optimizer']['betas'],
        eps=config['training']['optimizer']['eps']
    )
    
    # Create scheduler
    total_steps = len(train_loader) * config['training']['num_epochs']
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=config['training']['scheduler']['warmup_steps'],
        num_training_steps=total_steps
    )
    
    # Handle resume/fresh training
    start_epoch = 0
    train_dir = Path(args.output_train)
    
    if not args.resume:
        # Delete existing training files and start fresh
        if train_dir.exists():
            logger.warning(f"--no-resume specified. Deleting existing training directory: {train_dir}")
            import shutil
            shutil.rmtree(train_dir)
            logger.info(f"Deleted {train_dir}. Starting fresh training.")
            
            # Recreate directories
            checkpoints_dir = train_dir / "checkpoints"
            models_dir = train_dir / "models"
            log_dir = train_dir / "log"
            checkpoints_dir.mkdir(parents=True, exist_ok=True)
            models_dir.mkdir(parents=True, exist_ok=True)
            log_dir.mkdir(parents=True, exist_ok=True)
    else:
        # Try to resume from latest checkpoint
        checkpoints_dir = train_dir / "checkpoints"
        if checkpoints_dir.exists():
            # Find all checkpoint directories
            checkpoint_dirs = sorted([d for d in checkpoints_dir.iterdir() if d.is_dir() and d.name.startswith("checkpoint-epoch")])
            
            if checkpoint_dirs:
                # Get the latest checkpoint
                latest_checkpoint = checkpoint_dirs[-1] / "pytorch_model.bin"
                if latest_checkpoint.exists():
                    logger.info(f"Resuming from latest checkpoint: {latest_checkpoint}")
                    checkpoint = torch.load(latest_checkpoint, map_location=device)
                    model.load_state_dict(checkpoint['model_state_dict'])
                    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                    start_epoch = checkpoint['epoch'] + 1
                    logger.info(f"Resumed from epoch {checkpoint['epoch']}, starting at epoch {start_epoch}")
                else:
                    logger.warning(f"Checkpoint file not found: {latest_checkpoint}. Starting from scratch.")
            else:
                logger.info("No checkpoints found. Starting training from scratch.")
        else:
            logger.info("Checkpoints directory doesn't exist. Starting training from scratch.")
    
    # Training loop
    logger.info("Starting training...")
    best_eval_loss = float('inf')
    
    for epoch in range(start_epoch, config['training']['num_epochs']):
        logger.info(f"\n{'='*80}")
        logger.info(f"Epoch {epoch + 1}/{config['training']['num_epochs']}")
        logger.info(f"{'='*80}")
        
        # Train
        train_metrics = train_epoch(
            model, train_loader, criterion, optimizer, scheduler, device, config, epoch, args
        )
        
        logger.info(f"Train metrics: {train_metrics}")
        
        # Evaluate and save validation results
        eval_metrics = evaluate(model, val_loader, device, criterion, output_dir=args.output_valid, epoch=epoch)
        
        # Log to W&B
        if args.use_wandb and HAS_WANDB:
            wandb.log({
                **{f'train/{k}': v for k, v in train_metrics.items()},
                **eval_metrics,
                'epoch': epoch
            })
        
        # Save checkpoint
        is_best = eval_metrics['eval/loss'] < best_eval_loss
        if is_best:
            best_eval_loss = eval_metrics['eval/loss']
        
        # Save checkpoint every epoch or when best
        save_interval = max(1, config['training']['save_steps'] // len(train_loader))
        if (epoch + 1) % save_interval == 0 or is_best:
            save_checkpoint(model, optimizer, scheduler, epoch, args.output_train, config, is_best)
    
    logger.info("\n" + "="*80)
    logger.info("Training completed!")
    logger.info(f"Best eval loss: {best_eval_loss:.4f}")
    logger.info(f"Training outputs:")
    logger.info(f"  - Best model: {Path(args.output_train) / 'models' / 'best_model'}")
    logger.info(f"  - Checkpoints: {Path(args.output_train) / 'checkpoints'}")
    logger.info(f"  - Logs: {Path(args.output_train) / 'log'}")
    logger.info(f"Validation outputs:")
    logger.info(f"  - Validation results: {Path(args.output_valid)}")
    logger.info(f"Test outputs:")
    logger.info(f"  - Test results directory: {Path(args.output_test)}")
    logger.info("="*80)


if __name__ == "__main__":
    main()

