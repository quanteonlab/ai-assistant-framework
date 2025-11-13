#!/usr/bin/env python3
"""
PDF/EPUB to Chinese Flashcards - Specialized Pipeline
======================================================

This script provides a complete end-to-end pipeline for converting PDF or EPUB files
into Chinese language flashcards with difficulty rating for advanced learners.

Features:
- Generates advanced Chinese terminology flashcards (format: 彩带:::Cǎidài, Ribbon)
- Creates bilingual comprehension flashcards
- Rates difficulty for advanced Chinese learners (HSK 5-6 level)
- All content includes both English and Chinese

Pipeline:
1. Convert PDF/EPUB to chunked CSV using book2text.py
2. Generate Chinese flashcards from CSV using flashcard.py with chinese_flashcards prompt
3. Rate flashcard difficulty using chinese_difficulty_rating prompt

Usage:
    python chinese_flashcards.py input.pdf
    python chinese_flashcards.py input.epub --rating-threshold 7
    python chinese_flashcards.py input.pdf --chapters-per-file 2

Author: Generated for AI Assistant Framework
License: Same as parent project
"""

import os
import sys
import re
import argparse
import subprocess
import shutil
from pathlib import Path
from typing import Optional

# Color codes for terminal output
class Colors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

def print_header(text: str):
    """Print a formatted header."""
    print(f"\n{Colors.HEADER}{Colors.BOLD}{'='*70}")
    print(f"  {text}")
    print(f"{'='*70}{Colors.ENDC}\n")

def print_step(step_num: int, text: str):
    """Print a formatted step."""
    print(f"{Colors.OKCYAN}{Colors.BOLD}[Step {step_num}]{Colors.ENDC} {text}")

def print_success(text: str):
    """Print a success message."""
    print(f"{Colors.OKGREEN}[OK] {text}{Colors.ENDC}")

def print_error(text: str):
    """Print an error message."""
    print(f"{Colors.FAIL}[ERROR] {text}{Colors.ENDC}")

def print_warning(text: str):
    """Print a warning message."""
    print(f"{Colors.WARNING}[WARNING] {text}{Colors.ENDC}")

def check_dependencies():
    """Check if required scripts and dependencies are available."""
    script_dir = Path(__file__).parent.absolute()

    # Check if book2text.py exists
    book2text_path = script_dir / "book2text.py"
    if not book2text_path.exists():
        print_error(f"book2text.py not found at {book2text_path}")
        return False

    # Check if flashcard.py exists
    flashcard_path = script_dir / "flashcard.py"
    if not flashcard_path.exists():
        print_error(f"flashcard.py not found at {flashcard_path}")
        return False

    # Check if _config.yaml exists and has chinese prompts
    config_path = script_dir / "_config.yaml"
    if not config_path.exists():
        print_error(f"_config.yaml not found at {config_path}")
        return False

    # Verify chinese prompts exist in config
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config_content = f.read()
            if 'chinese_flashcards:' not in config_content:
                print_error("chinese_flashcards prompt not found in _config.yaml")
                return False
            if 'chinese_difficulty_rating:' not in config_content:
                print_error("chinese_difficulty_rating prompt not found in _config.yaml")
                return False
    except Exception as e:
        print_error(f"Error reading config: {e}")
        return False

    return True

def run_book2text(input_file: str, output_dir: str) -> Optional[str]:
    """
    Run book2text.py to convert PDF/EPUB to CSV.

    Args:
        input_file: Path to input PDF or EPUB file
        output_dir: Directory for output files

    Returns:
        Path to the processed CSV file, or None if failed
    """
    print_step(1, "Converting PDF/EPUB to chunked CSV...")

    script_dir = Path(__file__).parent.absolute()
    book2text_script = script_dir / "book2text.py"

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    try:
        # Run book2text.py
        result = subprocess.run(
            [sys.executable, str(book2text_script), input_file],
            cwd=script_dir,
            capture_output=True,
            text=True,
            timeout=600  # 10 minute timeout
        )

        if result.returncode != 0:
            print_error(f"book2text.py failed with return code {result.returncode}")
            print(f"STDOUT:\n{result.stdout}")
            print(f"STDERR:\n{result.stderr}")
            return None

        # Print output
        print(result.stdout)

        # Determine the expected output CSV filename (in processed subfolder)
        input_path = Path(input_file)
        base_name = input_path.stem.replace(" ", "-")
        base_name = re.sub(r'[^\w\-_]', '', base_name)
        expected_csv = script_dir / "out" / "processed" / f"{base_name}_processed.csv"

        if not expected_csv.exists():
            print_error(f"Expected output CSV not found: {expected_csv}")
            return None

        print_success(f"CSV created: {expected_csv}")
        return str(expected_csv)

    except subprocess.TimeoutExpired:
        print_error("book2text.py timed out after 10 minutes")
        return None
    except Exception as e:
        print_error(f"Error running book2text.py: {e}")
        return None

def run_chinese_flashcard_generation(csv_file: str, output_dir: str, args: argparse.Namespace) -> bool:
    """
    Run flashcard.py to generate Chinese flashcards from CSV.

    Args:
        csv_file: Path to input CSV file
        output_dir: Directory for flashcard output
        args: Command-line arguments

    Returns:
        True if successful, False otherwise
    """
    print_step(2, "Generating Chinese flashcards from CSV...")

    script_dir = Path(__file__).parent.absolute()
    flashcard_script = script_dir / "flashcard.py"

    # Build command - use chinese_flashcards prompt and chinese_difficulty_rating
    cmd = [
        sys.executable,
        str(flashcard_script),
        "-c", csv_file,
        "-o", output_dir,
        "-p", "chinese_flashcards",  # Use Chinese flashcard prompt
        "--rating-prompt", "chinese_difficulty_rating"  # Use Chinese difficulty rating
    ]

    # Add optional arguments
    if args.model:
        cmd.extend(["-m", args.model])

    if args.verbose:
        cmd.append("-v")

    if args.min_length:
        cmd.extend(["--min-length", str(args.min_length)])

    if args.chapters_per_file:
        cmd.extend(["--chapters-per-file", str(args.chapters_per_file)])

    if args.max_text_size:
        cmd.extend(["--max-text-size", str(args.max_text_size)])

    # Chinese flashcards should always have rating enabled
    cmd.append("--enable-rating")

    if args.rating_threshold:
        cmd.extend(["--rating-threshold", str(args.rating_threshold)])

    if args.rating_model:
        cmd.extend(["--rating-model", args.rating_model])

    if args.no_training_data:
        cmd.append("--no-training-data")

    try:
        # Run flashcard.py
        result = subprocess.run(
            cmd,
            cwd=script_dir,
            capture_output=True,
            text=True,
            timeout=10000  # Approx 3 hours timeout for flashcard generation
        )

        if result.returncode != 0:
            print_error(f"flashcard.py failed with return code {result.returncode}")
            print(f"STDOUT:\n{result.stdout}")
            print(f"STDERR:\n{result.stderr}")
            return False

        # Print output
        print(result.stdout)

        print_success("Chinese flashcards generated successfully!")
        return True

    except subprocess.TimeoutExpired:
        print_error("flashcard.py timed out")
        return False
    except Exception as e:
        print_error(f"Error running flashcard.py: {e}")
        return False

def run_flashcard_rating(output_dir: str, args: argparse.Namespace, book_basename: str = None) -> bool:
    """
    Run rate_flashcards.py to rate unrated Chinese flashcards using difficulty rating.

    Args:
        output_dir: Directory containing training data CSV
        args: Command-line arguments
        book_basename: Base name of the book (for finding the training data file)

    Returns:
        True if successful, False otherwise
    """
    print_step(3, "Rating unrated Chinese flashcards...")

    script_dir = Path(__file__).parent.absolute()
    rating_script = script_dir / "rate_flashcards.py"

    # Check if rating script exists
    if not rating_script.exists():
        print_warning(f"Rating script not found: {rating_script}")
        return False

    # Find training data file
    if book_basename:
        training_csv = Path(output_dir) / f"{book_basename}_training_data.csv"
    else:
        training_csv = Path(output_dir) / "flashcards_training_data.csv"

    if not training_csv.exists():
        # Try to find any *_training_data.csv file in the directory
        training_files = list(Path(output_dir).glob("*_training_data.csv"))
        if training_files:
            training_csv = training_files[0]
            print_warning(f"Using found training data: {training_csv.name}")
        else:
            print_warning(f"Training data not found: {training_csv}")
            return False

    # Build command with Chinese difficulty rating
    cmd = [
        sys.executable,
        str(rating_script),
        str(training_csv),
        "--threshold", str(args.rating_threshold),
        "--rating-prompt", "chinese_difficulty_rating"  # Use Chinese difficulty rating
    ]

    # Add optional arguments
    if args.rating_model:
        cmd.extend(["--model", args.rating_model])
    elif args.model:
        cmd.extend(["--model", args.model])

    if args.verbose:
        cmd.append("-v")

    cmd.extend(["--output", output_dir])

    try:
        # Run rate_flashcards.py
        result = subprocess.run(
            cmd,
            cwd=script_dir,
            capture_output=True,
            text=True,
            timeout=3600  # 1 hour timeout
        )

        if result.returncode != 0:
            print_error(f"rate_flashcards.py failed with return code {result.returncode}")
            print(f"STDOUT:\n{result.stdout}")
            print(f"STDERR:\n{result.stderr}")
            return False

        # Print output
        print(result.stdout)

        print_success("Chinese flashcard difficulty rating completed!")
        return True

    except subprocess.TimeoutExpired:
        print_error("rate_flashcards.py timed out after 1 hour")
        return False
    except Exception as e:
        print_error(f"Error running rate_flashcards.py: {e}")
        return False

def cleanup_intermediate_files(csv_file: str, keep_csv: bool):
    """
    Optionally clean up intermediate CSV files.

    Args:
        csv_file: Path to CSV file
        keep_csv: Whether to keep the CSV file
    """
    if not keep_csv:
        try:
            os.remove(csv_file)
            print_success(f"Cleaned up intermediate file: {csv_file}")
        except Exception as e:
            print_warning(f"Could not remove intermediate file {csv_file}: {e}")

def main():
    parser = argparse.ArgumentParser(
        description="Convert PDF/EPUB to Chinese language flashcards with difficulty rating for advanced learners",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage - PDF to Chinese flashcards (difficulty rating enabled)
  python chinese_flashcards.py chinese_textbook.pdf

  # Process EPUB with custom threshold
  python chinese_flashcards.py chinese_novel.epub --rating-threshold 7

  # Custom output directory
  python chinese_flashcards.py mybook.pdf -o chinese_flashcards/mybook

  # Smaller files: 2 chapters per file
  python chinese_flashcards.py mybook.pdf --chapters-per-file 2

  # Keep intermediate CSV for debugging
  python chinese_flashcards.py mybook.pdf --keep-csv

Output Structure:
  chinese_flashcards/
  |-- mybook_part01_Chapter_One.md        # Mixed terminology + comprehension flashcards
  |-- mybook_part02_Chapter_Two.md
  |-- high_quality/                        # Advanced flashcards rated >= threshold (default 6/10)
  |   |-- mybook_hq_part01.md              # Consolidated high-difficulty flashcards
  |   +-- mybook_hq_part02.md
  +-- mybook_training_data.csv             # All flashcards with difficulty ratings (1-10)

Flashcard Format Examples:
  Terminology: 彩带:::Cǎidài, Ribbon
  Terminology: 行驶的时间:::Xíngshǐ de shíjiān, Travel time
  Comprehension: Bilingual concept explanations with Q&A
        """
    )

    # Required arguments
    parser.add_argument('input_file', help='Input PDF or EPUB file (Chinese language content)')

    # Output options
    parser.add_argument('-o', '--output', default='chinese_flashcards',
                       help='Output directory for flashcards (default: chinese_flashcards/)')
    parser.add_argument('--keep-csv', action='store_true',
                       help='Keep intermediate CSV file (default: delete after processing)')

    # Model options
    parser.add_argument('-m', '--model',
                       help='Model name to use for flashcard generation (default from config)')

    # File splitting options
    parser.add_argument('--chapters-per-file', type=int, default=3,
                       help='Number of chapters per flashcard file (default: 3)')
    parser.add_argument('--max-text-size', type=int, default=50000,
                       help='Maximum text size per file in characters (default: 50000)')

    # Rating options (always enabled for Chinese flashcards)
    parser.add_argument('--rating-threshold', type=int, default=6,
                       help='Minimum difficulty rating for high-quality folder (default: 6 for advanced learners)')
    parser.add_argument('--rating-model',
                       help='Model to use for difficulty rating (default: same as generation model)')

    # Post-processing rating options
    parser.add_argument('--disable-post-rating', action='store_false', dest='enable_post_rating', default=True,
                       help='Disable post-processing rating of unrated flashcards (enabled by default)')

    # Content filtering options
    parser.add_argument('--min-length', type=int, default=200,
                       help='Minimum text length to process (default: 200)')

    # Other options
    parser.add_argument('--no-training-data', action='store_true',
                       help='Disable saving training data to CSV')
    parser.add_argument('-v', '--verbose', action='store_true',
                       help='Display flashcards as they are generated')

    args = parser.parse_args()

    # Print welcome header
    print_header("Chinese Language Flashcards Pipeline")

    # Validate input file
    if not os.path.exists(args.input_file):
        print_error(f"Input file not found: {args.input_file}")
        sys.exit(1)

    # Check file extension
    input_path = Path(args.input_file)
    if input_path.suffix.lower() not in ['.pdf', '.epub']:
        print_error("Input file must be PDF or EPUB")
        sys.exit(1)

    # Check dependencies
    if not check_dependencies():
        print_error("Missing required dependencies or Chinese prompts in config. Please check your installation.")
        sys.exit(1)

    print_success("All dependencies and Chinese prompts found")

    # Determine output directories
    script_dir = Path(__file__).parent.absolute()
    out_dir = script_dir / "out"

    # Step 1: Convert to CSV
    csv_file = run_book2text(args.input_file, str(out_dir))
    if not csv_file:
        print_error("Failed to convert PDF/EPUB to CSV")
        sys.exit(1)

    # Step 2: Generate Chinese flashcards
    success = run_chinese_flashcard_generation(csv_file, args.output, args)
    if not success:
        print_error("Failed to generate Chinese flashcards")
        sys.exit(1)

    # Extract book name for training data file reference
    book_basename = os.path.splitext(os.path.basename(csv_file))[0]

    # Step 3: Rate unrated flashcards (optional)
    if args.enable_post_rating:
        rating_success = run_flashcard_rating(args.output, args, book_basename)
        if not rating_success:
            print_warning("Chinese flashcard rating step had issues, but continuing...")

    # Step 4: Cleanup (optional)
    if not args.keep_csv:
        cleanup_intermediate_files(csv_file, args.keep_csv)
    else:
        print_success(f"Intermediate CSV kept: {csv_file}")

    # Final summary
    print_header("Chinese Flashcards Pipeline Complete!")

    print(f"[OK] Input: {args.input_file}")
    print(f"[OK] Output: {args.output}/")
    print(f"[OK] High-difficulty flashcards (>={args.rating_threshold}/10): {args.output}/high_quality/")
    print(f"[OK] Training data with difficulty ratings: {args.output}/{book_basename}_training_data.csv")
    print(f"\nFlashcard formats:")
    print(f"  - Advanced terminology: [Chinese]:::[Pinyin], [English]")
    print(f"  - Comprehension: Bilingual concept Q&A")
    print()

if __name__ == "__main__":
    main()
