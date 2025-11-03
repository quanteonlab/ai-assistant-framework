#!/usr/bin/env python3
"""
Script to reorganize flashcards into the new folder structure.

Old structure:
  flashcards/{long_book_name}_part01_chapter.md
  flashcards/high_quality/{long_book_name}_hq_part01_chapter.md

New structure:
  flashcards/{full_book_name}/{short_prefix}_part01_chapter.md
  flashcards/{full_book_name}/{short_prefix}_hq_part01_chapter.md
"""

import os
import re
import shutil
from pathlib import Path
from typing import Dict, List, Tuple
from flashcard import generate_book_prefix, sanitize_filename


def extract_book_info(filename: str) -> Tuple[str, str, str, str]:
    """
    Extract book name, part number, and chapter from filename.

    Args:
        filename: Original filename like "Book-Name_part01_Chapter.md"
                  or "Book-Name_hq_part01_Chapter.md"

    Returns:
        Tuple of (book_name, is_hq, part_num, chapter_name)
    """
    # Remove .md extension
    name = filename.replace('.md', '')

    # Check if it's a high-quality file
    is_hq = '_hq_part' in name

    # Extract parts
    if is_hq:
        # Format: Book-Name_hq_part01_Chapter
        match = re.match(r'(.+?)_hq_part(\d+)_(.+)', name)
    else:
        # Format: Book-Name_part01_Chapter or Book-Name_processed_part01_Chapter
        match = re.match(r'(.+?)_part(\d+)_(.+)', name)

    if match:
        book_name = match.group(1)
        part_num = match.group(2)
        chapter_name = match.group(3)

        # Clean up book name - remove _processed suffix if present
        book_name = re.sub(r'_processed$', '', book_name)

        return (book_name, is_hq, part_num, chapter_name)

    return (None, None, None, None)


def group_files_by_book(flashcards_dir: str) -> Dict[str, List[str]]:
    """
    Group flashcard files by book name.

    Args:
        flashcards_dir: Path to flashcards directory

    Returns:
        Dictionary mapping book names to lists of files
    """
    books = {}

    # Process regular flashcards
    for filename in os.listdir(flashcards_dir):
        if not filename.endswith('.md'):
            continue

        filepath = os.path.join(flashcards_dir, filename)
        if os.path.isfile(filepath):
            book_name, is_hq, part_num, chapter = extract_book_info(filename)
            if book_name:
                if book_name not in books:
                    books[book_name] = []
                books[book_name].append(('regular', filename, part_num, chapter))

    # Process high_quality flashcards
    hq_dir = os.path.join(flashcards_dir, 'high_quality')
    if os.path.exists(hq_dir):
        for filename in os.listdir(hq_dir):
            if not filename.endswith('.md'):
                continue

            filepath = os.path.join(hq_dir, filename)
            if os.path.isfile(filepath):
                book_name, is_hq, part_num, chapter = extract_book_info(filename)
                if book_name:
                    if book_name not in books:
                        books[book_name] = []
                    books[book_name].append(('hq', filename, part_num, chapter))

    return books


def reorganize_flashcards(flashcards_dir: str, dry_run: bool = True):
    """
    Reorganize flashcards into new folder structure.

    Args:
        flashcards_dir: Path to flashcards directory
        dry_run: If True, only print what would be done without making changes
    """
    books = group_files_by_book(flashcards_dir)

    print(f"Found {len(books)} unique books")
    print()

    for book_name, files in books.items():
        print(f"\n{'='*80}")
        print(f"Book: {book_name}")
        print(f"Files: {len(files)}")

        # Generate short prefix
        book_prefix = generate_book_prefix(book_name, max_length=5)
        print(f"Prefix: {book_prefix}")

        # Create book folder
        book_folder = os.path.join(flashcards_dir, book_name)

        if dry_run:
            print(f"[DRY RUN] Would create folder: {book_folder}")
        else:
            os.makedirs(book_folder, exist_ok=True)
            print(f"Created folder: {book_folder}")

        # Process each file
        for file_type, filename, part_num, chapter in files:
            # Determine source path
            if file_type == 'hq':
                src_path = os.path.join(flashcards_dir, 'high_quality', filename)
            else:
                src_path = os.path.join(flashcards_dir, filename)

            # Generate new filename with shortened chapter (max 15 chars)
            short_chapter = sanitize_filename(chapter, max_length=15)

            if file_type == 'hq':
                new_filename = f"{book_prefix}_hq_part{part_num}_{short_chapter}.md"
            else:
                new_filename = f"{book_prefix}_part{part_num}_{short_chapter}.md"

            dst_path = os.path.join(book_folder, new_filename)

            print(f"  {filename}")
            print(f"    -> {new_filename}")

            if dry_run:
                print(f"    [DRY RUN] Would move: {src_path} -> {dst_path}")
            else:
                # Move file
                shutil.move(src_path, dst_path)
                print(f"    Moved to: {dst_path}")

        # Also move training data CSV if it exists
        csv_files = [
            f"{book_name}_training_data.csv",
            f"{book_name}_processed_training_data.csv"
        ]

        for csv_name in csv_files:
            csv_path = os.path.join(flashcards_dir, csv_name)
            if os.path.exists(csv_path):
                dst_csv = os.path.join(book_folder, csv_name)

                if dry_run:
                    print(f"  [DRY RUN] Would move CSV: {csv_path} -> {dst_csv}")
                else:
                    shutil.move(csv_path, dst_csv)
                    print(f"  Moved CSV: {dst_csv}")

    if not dry_run:
        # Clean up empty high_quality folder
        hq_dir = os.path.join(flashcards_dir, 'high_quality')
        if os.path.exists(hq_dir) and not os.listdir(hq_dir):
            os.rmdir(hq_dir)
            print(f"\nRemoved empty folder: {hq_dir}")

    print(f"\n{'='*80}")
    if dry_run:
        print("\n*** DRY RUN COMPLETE - No files were actually moved ***")
        print("Run with --execute to perform the reorganization")
    else:
        print("\n*** REORGANIZATION COMPLETE ***")


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Reorganize flashcards into new folder structure"
    )
    parser.add_argument(
        'flashcards_dir',
        nargs='?',
        default='flashcards',
        help='Path to flashcards directory (default: flashcards)'
    )
    parser.add_argument(
        '--execute',
        action='store_true',
        help='Actually perform the reorganization (default is dry run)'
    )

    args = parser.parse_args()

    if not os.path.exists(args.flashcards_dir):
        print(f"Error: Directory not found: {args.flashcards_dir}")
        return 1

    reorganize_flashcards(args.flashcards_dir, dry_run=not args.execute)

    return 0


if __name__ == '__main__':
    exit(main())
