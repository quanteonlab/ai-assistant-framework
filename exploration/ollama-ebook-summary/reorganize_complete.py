#!/usr/bin/env python3
"""
Complete reorganization script for flashcards.

Old structure:
  flashcards/{long_book_name}_part01_chapter.md
  flashcards/high_quality/{long_book_name}_hq_part01_chapter.md

New structure:
  flashcards/{full_book_name}/{short_prefix}_part01_chapter.md
  refined_docs/book_notes/{full_book_name}/{short_prefix}_part01_chapter.md (high quality only)
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


def reorganize_flashcards(base_dir: str, dry_run: bool = True):
    """
    Reorganize flashcards into new folder structure.

    Regular flashcards: flashcards/{book_name}/{prefix}_part{num}_{chapter}.md
    High quality: refined_docs/book_notes/{book_name}/{prefix}_part{num}_{chapter}.md

    Args:
        base_dir: Base directory (should be ollama-ebook-summary)
        dry_run: If True, only print what would be done without making changes
    """
    flashcards_dir = os.path.join(base_dir, 'flashcards')
    refined_docs_dir = os.path.join(base_dir, 'refined_docs', 'book_notes')

    # Create refined_docs/book_notes if it doesn't exist
    if not dry_run:
        os.makedirs(refined_docs_dir, exist_ok=True)

    books = group_files_by_book(flashcards_dir)

    print(f"Found {len(books)} unique books")
    print(f"\nRegular flashcards will go to: flashcards/{{book}}/")
    print(f"High-quality flashcards will go to: refined_docs/book_notes/{{book}}/")
    print()

    for book_name, files in books.items():
        print(f"\n{'='*80}")
        print(f"Book: {book_name}")
        print(f"Files: {len(files)}")

        # Generate short prefix
        book_prefix = generate_book_prefix(book_name, max_length=5)
        print(f"Prefix: {book_prefix}")

        # Create book folders
        regular_book_folder = os.path.join(flashcards_dir, book_name)
        hq_book_folder = os.path.join(refined_docs_dir, book_name)

        if dry_run:
            print(f"[DRY RUN] Would create folder: {regular_book_folder}")
            print(f"[DRY RUN] Would create folder: {hq_book_folder}")
        else:
            os.makedirs(regular_book_folder, exist_ok=True)
            os.makedirs(hq_book_folder, exist_ok=True)
            print(f"Created folder: {regular_book_folder}")
            print(f"Created folder: {hq_book_folder}")

        # Process each file
        for file_type, filename, part_num, chapter in files:
            # Determine source path
            if file_type == 'hq':
                src_path = os.path.join(flashcards_dir, 'high_quality', filename)
                # High quality goes to refined_docs
                dst_folder = hq_book_folder
            else:
                src_path = os.path.join(flashcards_dir, filename)
                # Regular goes to flashcards
                dst_folder = regular_book_folder

            # Generate new filename with shortened chapter (max 15 chars)
            short_chapter = sanitize_filename(chapter, max_length=15)

            # Remove _hq_ from filename since it's already separated by folder
            new_filename = f"{book_prefix}_part{part_num}_{short_chapter}.md"

            dst_path = os.path.join(dst_folder, new_filename)

            print(f"  {filename}")
            print(f"    -> {new_filename} ({'refined_docs' if file_type == 'hq' else 'flashcards'})")

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
                # Training data goes with regular flashcards
                dst_csv = os.path.join(regular_book_folder, csv_name)

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
        print(f"\nStructure:")
        print(f"  flashcards/{{book_name}}/{{prefix}}_part{{num}}_{{chapter}}.md")
        print(f"  refined_docs/book_notes/{{book_name}}/{{prefix}}_part{{num}}_{{chapter}}.md")


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Reorganize flashcards into new folder structure"
    )
    parser.add_argument(
        '--execute',
        action='store_true',
        help='Actually perform the reorganization (default is dry run)'
    )

    args = parser.parse_args()

    # Get the base directory (ollama-ebook-summary)
    base_dir = os.path.dirname(os.path.abspath(__file__))

    reorganize_flashcards(base_dir, dry_run=not args.execute)

    return 0


if __name__ == '__main__':
    exit(main())
