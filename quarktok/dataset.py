"""Download and save Wikipedia dataset with configurable size limit."""

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Optional

from datasets import load_dataset
from tqdm import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Default configuration
DEFAULT_TARGET_SIZE_MB = 1024  # 1 GB in MB
DEFAULT_DATASET_NAME = "wikimedia/wikipedia"
DEFAULT_CONFIG_NAME = "20231101.en"
DEFAULT_OUTPUT_DIR = "datasets"
DEFAULT_OUTPUT_FILENAME = "wikipedia-en-1gb.txt"


def validate_size(size_mb: float) -> int:
    """
    Validate and convert size from MB to bytes.

    Args:
        size_mb: Target size in megabytes

    Returns:
        Size in bytes

    Raises:
        ValueError: If size is invalid
    """
    if size_mb <= 0:
        raise ValueError(f"Size must be positive, got {size_mb} MB")
    if size_mb > 100000:  # 100 GB sanity check
        raise ValueError(f"Size {size_mb} MB seems too large (max 100000 MB)")
    return int(size_mb * 1024 * 1024)


def ensure_directory(path: Path) -> None:
    """
    Ensure directory exists, create if it doesn't.

    Args:
        path: Directory path to ensure exists

    Raises:
        OSError: If directory creation fails
    """
    try:
        path.mkdir(parents=True, exist_ok=True)
        logger.info(f"Output directory ready: {path}")
    except OSError as e:
        raise OSError(f"Failed to create directory {path}: {e}") from e


def download_limited_data(
    dataset_name: str = DEFAULT_DATASET_NAME,
    config_name: str = DEFAULT_CONFIG_NAME,
    output_path: Path = Path(DEFAULT_OUTPUT_DIR) / DEFAULT_OUTPUT_FILENAME,
    target_size_bytes: int = DEFAULT_TARGET_SIZE_MB * 1024 * 1024,
    text_field: str = "text"
) -> bool:
    """
    Download dataset articles until reaching the target size.

    Args:
        dataset_name: HuggingFace dataset identifier
        config_name: Dataset configuration (e.g., language and date)
        output_path: Path where the dataset will be saved
        target_size_bytes: Target file size in bytes
        text_field: Field name containing the text data

    Returns:
        True if successful, False otherwise
    """
    logger.info(f"Initializing stream for {dataset_name} ({config_name})...")
    logger.info(f"Target size: {target_size_bytes / (1024*1024):.2f} MB")

    # Ensure output directory exists
    try:
        ensure_directory(output_path.parent)
    except OSError as e:
        logger.error(f"Cannot create output directory: {e}")
        return False

    # Load dataset with streaming
    try:
        ds = load_dataset(
            dataset_name,
            config_name,
            split="train",
            streaming=True,
            trust_remote_code=False  # Security: don't execute remote code
        )
    except ValueError as e:
        logger.error(f"Invalid dataset configuration: {e}")
        logger.error(f"Check if '{config_name}' is valid for dataset '{dataset_name}'")
        return False
    except Exception as e:
        logger.error(f"Failed to load dataset: {e}")
        return False

    current_size = 0
    article_count = 0

    # Use context manager for progress bar to ensure cleanup
    try:
        with tqdm(
            total=target_size_bytes,
            unit='B',
            unit_scale=True,
            desc="Downloading"
        ) as pbar, open(output_path, "w", encoding="utf-8") as f:

            for example in ds:
                # Validate that the text field exists
                if text_field not in example:
                    logger.error(f"Field '{text_field}' not found in dataset")
                    logger.error(f"Available fields: {list(example.keys())}")
                    return False

                # Get text content
                text_content = example[text_field]
                if text_content is None:
                    continue  # Skip empty entries

                line = str(text_content) + "\n"
                line_size = len(line.encode('utf-8'))

                # Write to file
                try:
                    f.write(line)
                except IOError as e:
                    logger.error(f"Failed to write to file: {e}")
                    return False

                current_size += line_size
                article_count += 1
                pbar.update(line_size)

                # Stop condition
                if current_size >= target_size_bytes:
                    break

        # Success
        logger.info("Download complete!")
        logger.info(f"Total articles: {article_count:,}")
        logger.info(f"Total size: {current_size / (1024*1024):.2f} MB")
        logger.info(f"Saved to: {output_path.absolute()}")
        return True

    except KeyboardInterrupt:
        logger.warning("\nDownload interrupted by user")
        return False
    except Exception as e:
        logger.error(f"Unexpected error during download: {e}", exc_info=True)
        return False


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Download Wikipedia dataset with configurable size limit",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--dataset-name",
        default=DEFAULT_DATASET_NAME,
        help="HuggingFace dataset identifier"
    )

    parser.add_argument(
        "--config-name",
        default=DEFAULT_CONFIG_NAME,
        help="Dataset configuration (e.g., date and language)"
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(DEFAULT_OUTPUT_DIR),
        help="Output directory for the dataset"
    )

    parser.add_argument(
        "--output-filename",
        default=DEFAULT_OUTPUT_FILENAME,
        help="Output filename"
    )

    parser.add_argument(
        "--size-mb",
        type=float,
        default=DEFAULT_TARGET_SIZE_MB,
        help="Target dataset size in megabytes"
    )

    parser.add_argument(
        "--text-field",
        default="text",
        help="Name of the field containing text data"
    )

    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )

    return parser.parse_args()


def main() -> int:
    """
    Main entry point.

    Returns:
        Exit code (0 for success, 1 for failure)
    """
    args = parse_arguments()

    # Configure logging level
    if args.verbose:
        logger.setLevel(logging.DEBUG)

    # Validate and convert size
    try:
        target_size_bytes = validate_size(args.size_mb)
    except ValueError as e:
        logger.error(f"Invalid size: {e}")
        return 1

    # Construct output path
    output_path = args.output_dir / args.output_filename

    # Download dataset
    success = download_limited_data(
        dataset_name=args.dataset_name,
        config_name=args.config_name,
        output_path=output_path,
        target_size_bytes=target_size_bytes,
        text_field=args.text_field
    )

    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
