"""Test script to verify COCO dataset loader."""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.coco_dataset import build_coco_dataloader
from src.utils.tokenization import SimpleTokenizer


def test_dataset_loader():
    """Test that DataLoader yields batches correctly."""
    print("Testing COCO dataset loader...")

    # Use a very small subset for testing
    train_loader = build_coco_dataloader(
        annotation_file="images/annotations_trainval2017/annotations/captions_train2017.json",
        image_dir="images/train2017.1/train2017",
        batch_size=4,
        shuffle=True,
        num_workers=2,
        max_samples=20,  # Just 20 samples for testing
    )

    print(f"Dataset size: {len(train_loader.dataset)}")
    print(f"Number of batches: {len(train_loader)}")

    # Get a batch
    batch = next(iter(train_loader))
    print(f"\nBatch keys: {batch.keys()}")
    print(f"Image shape: {batch['image'].shape}")
    print(f"Number of captions: {len(batch['caption'])}")
    print(f"Sample caption: {batch['caption'][0]}")

    # Test tokenizer
    print("\nTesting tokenizer...")
    tokenizer = SimpleTokenizer(vocab_size=1000, min_freq=1)

    # Build vocab from captions
    all_captions = [batch['caption'][i] for i in range(len(batch['caption']))]
    tokenizer.build_vocab(all_captions)
    print(f"Vocabulary size: {len(tokenizer)}")

    # Encode a caption
    sample_caption = batch['caption'][0]
    encoded = tokenizer.encode(sample_caption, max_length=20)
    print(f"Original: {sample_caption}")
    print(f"Encoded: {encoded}")
    print(f"Decoded: {tokenizer.decode(encoded)}")

    print("\nDataset loader test passed!")


if __name__ == "__main__":
    test_dataset_loader()

