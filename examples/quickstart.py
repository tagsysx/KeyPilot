"""
QuickStart example for KeyPilot model.

This script demonstrates:
1. Model initialization
2. Forward pass
3. Prediction
4. Model summary
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import torch
from keypilot.models import create_keypilot_model


def main():
    print("=" * 80)
    print("KeyPilot QuickStart Example")
    print("=" * 80)
    
    # Configuration
    config = {
        'vocab_size': 32000,
        'd_model': 256,
        'num_tasks': 3,
        'num_layouts': 5,
        'num_experts': 5,
        'pretrained_backbone': False,  # Set to True to use pretrained MobileViT
        'user_emb_dim': 64
    }
    
    print("\n1. Creating KeyPilot model...")
    model = create_keypilot_model(config)
    model.eval()
    
    # Print model summary
    summary = model.get_model_summary()
    print("\nModel Summary:")
    print(f"  Total parameters: {summary['total_parameters']:,}")
    print(f"  Encoder parameters: {summary['encoder_parameters']['total']:,}")
    print(f"  Decoder parameters: {summary['decoder_parameters']['total']:,}")
    print(f"  Model size (FP32): {summary['model_size_mb']:.2f} MB")
    print(f"  Model size (INT8): {summary['model_size_mb_int8']:.2f} MB")
    
    # Create sample inputs
    print("\n2. Creating sample inputs...")
    batch_size = 2
    sample_inputs = {
        'image': torch.randn(batch_size, 3, 512, 256),
        'input_ids': torch.randint(0, 1000, (batch_size, 64)),
        'attention_mask': torch.ones(batch_size, 64, dtype=torch.long),
        'user_id': torch.tensor([0, 1]),
        'prev_layout': torch.tensor([0, 1])  # EN, ZH
    }
    
    print(f"  Image shape: {sample_inputs['image'].shape}")
    print(f"  Input IDs shape: {sample_inputs['input_ids'].shape}")
    print(f"  User IDs: {sample_inputs['user_id'].tolist()}")
    print(f"  Previous layouts: {[model.get_layout_name(i) for i in sample_inputs['prev_layout'].tolist()]}")
    
    # Forward pass
    print("\n3. Running forward pass...")
    with torch.no_grad():
        outputs = model(
            image=sample_inputs['image'],
            input_ids=sample_inputs['input_ids'],
            attention_mask=sample_inputs['attention_mask'],
            user_id=sample_inputs['user_id'],
            prev_layout=sample_inputs['prev_layout']
        )
    
    print("  Outputs:")
    print(f"    h_t shape: {outputs['h_t'].shape}")
    print(f"    Task probabilities shape: {outputs['task_probs'].shape}")
    print(f"    Layout IDs: {outputs['layout_id'].tolist()}")
    print(f"    Logits shape: {outputs['logits'].shape}")
    
    # Prediction mode
    print("\n4. Running prediction mode...")
    predictions = model.predict(
        image=sample_inputs['image'],
        input_ids=sample_inputs['input_ids'],
        attention_mask=sample_inputs['attention_mask'],
        user_id=sample_inputs['user_id'],
        prev_layout=sample_inputs['prev_layout'],
        temperature=0.9,
        top_k=50,
        top_p=0.9
    )
    
    print("\n  Predictions:")
    for i, pred in predictions.items():
        print(f"\n  Sample {i}:")
        print(f"    Task: {pred['task']}")
        print(f"    Layout: {pred['layout']}")
        print(f"    Task confidence: {pred['task_confidence']:.3f}")
        print(f"    Layout confidence: {pred['layout_confidence']:.3f}")
        
        if pred['task'] == 'suggestion':
            print(f"    Generated tokens: {pred['generated_tokens']}")
        else:
            print(f"    Predicted token: {pred['predicted_token']}")
            print(f"    Expert distribution: {[f'{p:.3f}' for p in pred['expert_distribution']]}")
    
    print("\n" + "=" * 80)
    print("KeyPilot QuickStart Complete!")
    print("=" * 80)
    
    # Additional info
    print("\nNext steps:")
    print("  1. Train tokenizer: python -m keypilot.utils.vocabulary")
    print("  2. Prepare dataset: Implement data loaders in src/keypilot/data/")
    print("  3. Start training: python scripts/train.py --config configs/model_config.yaml")
    print("  4. Run tests: pytest tests/")
    print("\nFor more details, see docs/module_selection.md")


if __name__ == "__main__":
    main()

