"""Script to download the Anthropic HH-RLHF dataset.

This script will be used in Week 7 for RAG implementation.
"""

import os
from pathlib import Path


def download_hh_rlhf_dataset(output_dir: str = "data/hh-rlhf", subset_size: int = None):
    """
    Download the Anthropic HH-RLHF dataset from Hugging Face.

    Args:
        output_dir: Directory to save the dataset
        subset_size: If specified, only download this many samples (for testing)
    """
    try:
        from datasets import load_dataset
    except ImportError:
        print("Error: 'datasets' package not installed.")
        print("Install it with: pip install datasets")
        return

    print("=" * 60)
    print("Downloading Anthropic HH-RLHF Dataset")
    print("=" * 60)
    print()
    print("Source: https://huggingface.co/datasets/Anthropic/hh-rlhf")
    print()

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    try:
        # Load the dataset
        print("Loading dataset from Hugging Face...")
        dataset = load_dataset("Anthropic/hh-rlhf")

        # Show dataset info
        print("\nDataset loaded successfully!")
        print(f"Available splits: {list(dataset.keys())}")

        for split_name, split_data in dataset.items():
            print(f"\n{split_name}: {len(split_data)} examples")

        # If subset_size is specified, take only a subset
        if subset_size:
            print(f"\nTaking subset of {subset_size} examples per split...")
            dataset = {
                split: split_data.select(range(min(subset_size, len(split_data))))
                for split, split_data in dataset.items()
            }

        # Save to disk
        print(f"\nSaving dataset to: {output_path}")
        dataset.save_to_disk(str(output_path))

        print("\n" + "=" * 60)
        print("Dataset downloaded successfully!")
        print("=" * 60)
        print(f"\nLocation: {output_path.absolute()}")

        # Show example
        print("\nExample conversation:")
        print("-" * 60)
        first_example = dataset["train"][0]
        for key, value in first_example.items():
            print(f"{key}:")
            print(f"  {value[:200]}..." if len(str(value)) > 200 else f"  {value}")
            print()

    except Exception as e:
        print(f"\nError downloading dataset: {e}")
        print("\nTroubleshooting:")
        print("1. Make sure you have internet connection")
        print("2. Install datasets: pip install datasets")
        print("3. Try again or download manually from Hugging Face")


def show_dataset_stats(dataset_dir: str = "data/hh-rlhf"):
    """
    Show statistics about the downloaded dataset.

    Args:
        dataset_dir: Path to the saved dataset
    """
    try:
        from datasets import load_from_disk
    except ImportError:
        print("Error: 'datasets' package not installed.")
        return

    dataset_path = Path(dataset_dir)
    if not dataset_path.exists():
        print(f"Dataset not found at: {dataset_path}")
        print("Run download_hh_rlhf_dataset() first.")
        return

    print("Loading dataset...")
    dataset = load_from_disk(str(dataset_path))

    print("\n" + "=" * 60)
    print("Dataset Statistics")
    print("=" * 60)

    for split_name, split_data in dataset.items():
        print(f"\n{split_name.upper()}:")
        print(f"  Total examples: {len(split_data)}")
        print(f"  Columns: {split_data.column_names}")

        # Sample data
        if len(split_data) > 0:
            example = split_data[0]
            print(f"\n  Example keys: {list(example.keys())}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Download Anthropic HH-RLHF dataset")
    parser.add_argument(
        "--output", "-o", default="data/hh-rlhf", help="Output directory for dataset"
    )
    parser.add_argument(
        "--subset",
        "-s",
        type=int,
        default=None,
        help="Download only a subset (for testing)",
    )
    parser.add_argument("--stats", action="store_true", help="Show dataset statistics")

    args = parser.parse_args()

    if args.stats:
        show_dataset_stats(args.output)
    else:
        download_hh_rlhf_dataset(args.output, args.subset)
