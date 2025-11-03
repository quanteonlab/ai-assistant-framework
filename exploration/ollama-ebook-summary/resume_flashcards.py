#!/usr/bin/env python3
"""
Resume Flashcard Generation from Timeout
==========================================

This script resumes flashcard generation for books that timed out during processing.
It finds the last generated flashcard and continues from there.

Usage:
    python resume_flashcards.py "Game Engine Architecture.pdf"
    python resume_flashcards.py "Game Engine Architecture.pdf" --output flashcards
"""

import os
import re
import sys
import argparse
import subprocess
from pathlib import Path
from typing import Optional, Tuple


def find_last_generated_flashcard(flashcards_dir: Path, book_basename: str) -> Optional[int]:
    """
    Find the last generated flashcard part number for a book.

    Args:
        flashcards_dir: Root flashcards directory
        book_basename: Sanitized book base name (folder name)

    Returns:
        Last part number (e.g., 27 if last file is *_part27_*.md), or None if no flashcards found
    """
    book_folder = flashcards_dir / book_basename

    if not book_folder.exists():
        print(f"[INFO] No flashcards folder found: {book_folder}")
        return None

    # Find all flashcard markdown files
    flashcard_files = list(book_folder.glob("*.md"))

    if not flashcard_files:
        print(f"[INFO] No flashcard files found in: {book_folder}")
        return None

    # Extract part numbers from filenames
    # Expected format: {prefix}_part{num}_{chapter}.md
    part_numbers = []
    pattern = r'_part(\d+)_'

    for f in flashcard_files:
        match = re.search(pattern, f.name)
        if match:
            part_num = int(match.group(1))
            part_numbers.append(part_num)

    if not part_numbers:
        print(f"[WARN] No valid flashcard files with part numbers found")
        return None

    last_part = max(part_numbers)
    print(f"[INFO] Last generated flashcard: part {last_part}")
    print(f"[INFO] Found {len(flashcard_files)} flashcard files")

    return last_part


def find_processed_csv(out_dir: Path, book_basename: str) -> Optional[Path]:
    """
    Find the processed CSV file for a book.

    Args:
        out_dir: Output directory (typically 'out')
        book_basename: Sanitized book base name

    Returns:
        Path to processed CSV file, or None if not found
    """
    processed_dir = out_dir / "processed"

    if not processed_dir.exists():
        print(f"[ERROR] Processed directory not found: {processed_dir}")
        return None

    # Try different possible CSV filenames
    possible_names = [
        f"{book_basename}_processed.csv",
        f"{book_basename}.csv",
    ]

    for csv_name in possible_names:
        csv_path = processed_dir / csv_name
        if csv_path.exists():
            print(f"[INFO] Found processed CSV: {csv_path}")
            return csv_path

    print(f"[ERROR] No processed CSV found for: {book_basename}")
    return None


def sanitize_book_name(book_filename: str) -> str:
    """
    Sanitize book filename to match folder names.

    Args:
        book_filename: Original book filename (e.g., "Game Engine Architecture.pdf")

    Returns:
        Sanitized name (e.g., "Game-Engine-Architecture")
    """
    # Remove extension
    name = Path(book_filename).stem

    # Replace spaces with hyphens
    name = name.replace(" ", "-")

    # Remove problematic characters
    name = re.sub(r'[^\w\-]', '', name)

    return name


def resume_flashcard_generation(
    book_filename: str,
    output_folder: str = "flashcards",
    relevancy_target: str = None
) -> bool:
    """
    Resume flashcard generation from where it left off after timeout.

    Args:
        book_filename: Name of the book file
        output_folder: Output folder for flashcards
        relevancy_target: Relevancy target for focused generation

    Returns:
        True if successful, False otherwise
    """
    script_dir = Path(__file__).parent.absolute()

    # Sanitize book name
    book_basename = sanitize_book_name(book_filename)
    print(f"[INFO] Book basename: {book_basename}")

    # Find last generated flashcard
    flashcards_dir = script_dir / output_folder
    last_part = find_last_generated_flashcard(flashcards_dir, book_basename)

    if last_part is None:
        print(f"[ERROR] Cannot resume - no existing flashcards found")
        print(f"[INFO] Run the full pipeline instead: python pdf2flashcards.py '{book_filename}'")
        return False

    # Find processed CSV
    out_dir = script_dir / "out"
    csv_path = find_processed_csv(out_dir, book_basename)

    if csv_path is None:
        print(f"[ERROR] Cannot resume - processed CSV not found")
        return False

    # Calculate starting chapter (last_part + 1)
    # The flashcard.py script processes chapters sequentially into part files
    # Each part may contain multiple chapters, so we need to skip past all chapters
    # that were already processed into parts 1 through last_part
    start_chapter = last_part + 1

    print(f"\n{'='*70}")
    print(f"  RESUMING FLASHCARD GENERATION")
    print(f"{'='*70}")
    print(f"Book: {book_filename}")
    print(f"Last completed part: {last_part}")
    print(f"Resuming from part: {start_chapter}")
    print(f"Output: {output_folder}/{book_basename}/")
    print(f"{'='*70}\n")

    # Build command to continue flashcard generation
    flashcard_script = script_dir / "flashcard.py"

    cmd = [
        sys.executable,
        str(flashcard_script),
        "-c", str(csv_path),
        "-o", output_folder,
        "--start-from-part", str(start_chapter)
    ]

    # Add relevancy target if provided
    if relevancy_target:
        cmd.extend(["--relevancy-target", relevancy_target])

    print(f"[INFO] Running: {' '.join(cmd)}\n")

    try:
        # Run flashcard generation
        result = subprocess.run(
            cmd,
            cwd=script_dir,
            capture_output=False,  # Show output in real-time
            text=True
        )

        if result.returncode != 0:
            print(f"\n[ERROR] Flashcard generation failed with return code {result.returncode}")
            return False

        print(f"\n[SUCCESS] Resume complete!")
        return True

    except Exception as e:
        print(f"\n[ERROR] Error resuming flashcard generation: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Resume flashcard generation from timeout",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Resume for a book that timed out
  python resume_flashcards.py "Game Engine Architecture.pdf"

  # Resume with custom output folder
  python resume_flashcards.py "My Book.pdf" --output flashcards

  # Resume with relevancy target
  python resume_flashcards.py "My Book.pdf" --relevancy-target "programming techniques"
        """
    )

    parser.add_argument(
        'book_filename',
        help='Name of the book file (e.g., "Game Engine Architecture.pdf")'
    )

    parser.add_argument(
        '--output',
        default='flashcards',
        help='Output folder for flashcards (default: flashcards)'
    )

    parser.add_argument(
        '--relevancy-target',
        help='Relevancy target for focused generation'
    )

    args = parser.parse_args()

    success = resume_flashcard_generation(
        args.book_filename,
        args.output,
        args.relevancy_target
    )

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
