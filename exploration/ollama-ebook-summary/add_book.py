#!/usr/bin/env python3
"""
Add Book to Orchestration
==========================

Helper script to add books to the orchestration CSV.

Usage:
    python add_book.py mybook.pdf
    python add_book.py mybook.epub --pipeline SUMMARIZEDFLASHCARDS
    python add_book.py *.pdf  # Add multiple books
    python add_book.py --scan  # Scan in/ folder and add all new books
"""

import csv
import sys
import argparse
from pathlib import Path


def get_existing_books(orchestration_csv: str):
    """Get a set of books already in orchestration CSV."""
    csv_path = Path(orchestration_csv)
    existing_books = set()

    if csv_path.exists():
        with open(csv_path, 'r', encoding='utf-8', newline='') as f:
            reader = csv.DictReader(f)
            for row in reader:
                existing_books.add(row['book_filename'])

    return existing_books


def add_book_to_orchestration(book_filename: str, pipeline_type: str, output_folder: str, orchestration_csv: str, force: bool = False):
    """Add a book to the orchestration CSV."""

    # Read existing rows
    rows = []
    fieldnames = ['book_filename', 'pipeline_type', 'status', 'started_at', 'completed_at', 'output_folder', 'error_message']

    csv_path = Path(orchestration_csv)
    if csv_path.exists():
        with open(csv_path, 'r', encoding='utf-8', newline='') as f:
            reader = csv.DictReader(f)
            for row in reader:
                rows.append(row)

    # Check if book already exists
    book_exists = False
    for idx, row in enumerate(rows):
        if row['book_filename'] == book_filename:
            book_exists = True
            if not force:
                print(f"[SKIP] Already in orchestration: {book_filename} (status: {row.get('status', 'PENDING')})")
                return False
            else:
                # Update existing row
                rows[idx]['pipeline_type'] = pipeline_type
                rows[idx]['output_folder'] = output_folder
                print(f"[UPDATE] Updated: {book_filename}")
                break

    # Add new book if not exists
    if not book_exists:
        new_row = {
            'book_filename': book_filename,
            'pipeline_type': pipeline_type,
            'status': 'PENDING',
            'started_at': '',
            'completed_at': '',
            'output_folder': output_folder,
            'error_message': ''
        }
        rows.append(new_row)
        print(f"[ADD] Added: {book_filename}")

    # Write back
    with open(csv_path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    if not book_exists:
        print(f"  Pipeline: {pipeline_type}")
        print(f"  Output: {output_folder}")
    return True


def scan_and_add_books(input_folder: str, pipeline_type: str, output_folder: str, orchestration_csv: str):
    """
    Scan the input folder for PDF/EPUB files and add any that aren't in orchestration CSV.

    Args:
        input_folder: Folder to scan for books
        pipeline_type: Default pipeline type
        output_folder: Default output folder
        orchestration_csv: Path to orchestration CSV

    Returns:
        Number of books added
    """
    input_path = Path(input_folder)

    # Create input folder if it doesn't exist
    if not input_path.exists():
        print(f"[ERROR] Input folder not found: {input_path}")
        return 0

    # Get existing books
    existing_books = get_existing_books(orchestration_csv)

    # Scan for PDF and EPUB files
    pdf_files = list(input_path.glob("*.pdf")) + list(input_path.glob("*.PDF"))
    epub_files = list(input_path.glob("*.epub")) + list(input_path.glob("*.EPUB"))
    all_books = pdf_files + epub_files

    if not all_books:
        print(f"[INFO] No PDF or EPUB files found in {input_folder}")
        return 0

    print(f"[INFO] Found {len(all_books)} books in {input_folder}")
    print(f"[INFO] {len(existing_books)} books already in orchestration")

    # Add new books
    added_count = 0
    for book_path in sorted(all_books):
        book_filename = book_path.name

        if book_filename in existing_books:
            print(f"[SKIP] Already tracked: {book_filename}")
            continue

        # Add the book
        if add_book_to_orchestration(book_filename, pipeline_type, output_folder, orchestration_csv, force=False):
            added_count += 1

    return added_count


def main():
    parser = argparse.ArgumentParser(
        description="Add books to orchestration CSV",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Add a single book
  python add_book.py mybook.pdf

  # Add with specific pipeline
  python add_book.py mybook.epub --pipeline SUMMARIZEDFLASHCARDS

  # Add with custom output folder
  python add_book.py mybook.pdf --output flashcards/mybook

  # Add multiple books
  python add_book.py book1.pdf book2.pdf book3.epub

  # Scan in/ folder and add all new books
  python add_book.py --scan

  # Scan custom folder
  python add_book.py --scan --input-folder my_pdfs/

  # Scan and set pipeline type
  python add_book.py --scan --pipeline ONLYFLASHCARDS
        """
    )

    parser.add_argument(
        'books',
        nargs='*',
        help='PDF or EPUB files to add (not needed with --scan)'
    )

    parser.add_argument(
        '--scan',
        action='store_true',
        help='Scan input folder and add all PDF/EPUB files not yet in orchestration'
    )

    parser.add_argument(
        '--input-folder',
        default='in',
        help='Input folder to scan (default: in/)'
    )

    parser.add_argument(
        '--pipeline',
        choices=['ONLYFLASHCARDS', 'SUMMARIZEDFLASHCARDS'],
        default='ONLYFLASHCARDS',
        help='Pipeline type (default: ONLYFLASHCARDS)'
    )

    parser.add_argument(
        '--output',
        default='flashcards',
        help='Output folder (default: flashcards)'
    )

    parser.add_argument(
        '--orchestration',
        default='orchestration.csv',
        help='Orchestration CSV file (default: orchestration.csv)'
    )

    parser.add_argument(
        '--force',
        action='store_true',
        help='Force update existing books'
    )

    args = parser.parse_args()

    # Handle scan mode
    if args.scan:
        print(f"[INFO] Scanning {args.input_folder} for PDF/EPUB files...")
        added_count = scan_and_add_books(
            args.input_folder,
            args.pipeline,
            args.output,
            args.orchestration
        )
        print(f"\n[OK] Added {added_count} new books to orchestration")
        return

    # Handle individual book mode
    if not args.books:
        print("[ERROR] No books specified. Use --scan to scan a folder or provide book filenames.")
        parser.print_help()
        sys.exit(1)

    added_count = 0
    for book in args.books:
        if add_book_to_orchestration(book, args.pipeline, args.output, args.orchestration, args.force):
            added_count += 1

    print(f"\n[OK] Added {added_count}/{len(args.books)} books to orchestration")


if __name__ == "__main__":
    main()
