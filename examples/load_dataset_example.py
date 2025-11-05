"""
Example: Load and inspect KeyPilot dataset

This example demonstrates how to load the pickle-format dataset
and inspect samples from different task types.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import torch
from torch.utils.data import DataLoader

# Note: Requires dependencies
# pip install transformers torch torchvision pillow loguru

def main():
    print("=" * 80)
    print("KeyPilot Dataset Loading Example")
    print("=" * 80)
    
    # Import dataset (requires dependencies installed)
    try:
        from keypilot.data.dataset import KeyPilotDataset
    except ImportError as e:
        print(f"\n❌ Error: Missing dependencies")
        print(f"   {e}")
        print(f"\n   Please install required packages:")
        print(f"   pip install transformers torch torchvision pillow loguru")
        return
    
    # 1. Load training dataset
    print("\n1. Loading training dataset...")
    train_path = "data/raw/data/train/all_samples_343.pkl"
    
    if not Path(train_path).exists():
        print(f"   ❌ Dataset not found: {train_path}")
        print(f"   Please ensure the data is in the correct location")
        return
    
    train_dataset = KeyPilotDataset(
        data_path=train_path,
        max_seq_length=128,
        image_size=(512, 256),
        max_samples=1000,  # Load only 1000 samples for demo
        use_bert_tokenizer=True
    )
    
    print(f"   ✓ Loaded {len(train_dataset)} samples")
    
    # 2. Inspect sample from each task type
    print("\n2. Inspecting samples from each task type...")
    
    task_samples = {}
    for idx in range(len(train_dataset)):
        sample_data = train_dataset.data[idx]
        task_label = sample_data['task_label']
        if task_label not in task_samples:
            task_samples[task_label] = (idx, sample_data)
        if len(task_samples) == 3:
            break
    
    for task_label, (idx, sample_data) in sorted(task_samples.items()):
        task_name = train_dataset.get_task_name(task_label)
        print(f"\n   Task {task_label} ({task_name.upper()}):")
        print(f"   - Sample ID: {sample_data['id']}")
        print(f"   - Image: {sample_data['image_path']}")
        print(f"   - Input tokens: {len(sample_data['input_tokens'])} tokens")
        
        # Load processed sample
        processed = train_dataset[idx]
        print(f"   - Processed image shape: {processed['image'].shape}")
        print(f"   - Input IDs shape: {processed['input_ids'].shape}")
        print(f"   - Target token IDs: {processed['target_token_ids'].shape}")
        
        # Decode input
        input_text = train_dataset.tokenizer.decode(
            processed['input_ids'], 
            skip_special_tokens=True
        )
        print(f"   - Input text: {input_text[:60]}...")
    
    # 3. Create DataLoader
    print("\n3. Creating DataLoader...")
    train_loader = DataLoader(
        train_dataset,
        batch_size=8,
        shuffle=True,
        num_workers=0  # Set to 0 for Windows compatibility
    )
    
    print(f"   ✓ DataLoader created with batch_size=8")
    
    # 4. Iterate one batch
    print("\n4. Loading one batch...")
    batch = next(iter(train_loader))
    
    print(f"   Batch contents:")
    print(f"   - image: {batch['image'].shape}")
    print(f"   - input_ids: {batch['input_ids'].shape}")
    print(f"   - attention_mask: {batch['attention_mask'].shape}")
    print(f"   - task_label: {batch['task_label'].shape}")
    print(f"   - layout_label: {batch['layout_label'].shape}")
    print(f"   - target_token_ids: {batch['target_token_ids'].shape}")
    print(f"   - target_type: {batch['target_type'].shape}")
    
    # Show task distribution in batch
    task_labels = batch['task_label'].tolist()
    task_names = [train_dataset.get_task_name(t) for t in task_labels]
    print(f"\n   Task distribution in batch:")
    for name in set(task_names):
        count = task_names.count(name)
        print(f"   - {name}: {count}/{len(task_labels)}")
    
    # 5. Memory usage
    print(f"\n5. Memory usage estimate:")
    sample_size = sys.getsizeof(train_dataset.data[0])
    total_mb = (sample_size * len(train_dataset)) / (1024 * 1024)
    print(f"   - Approximate dataset size in memory: {total_mb:.1f} MB")
    
    print("\n" + "=" * 80)
    print("✅ Dataset loading example completed successfully!")
    print("=" * 80)
    
    print("\nNext steps:")
    print("  1. Load full dataset (remove max_samples limit)")
    print("  2. Implement weighted sampling for balanced training")
    print("  3. Integrate with KeyPilotVLM model")
    print("  4. Start training!")


if __name__ == "__main__":
    main()

