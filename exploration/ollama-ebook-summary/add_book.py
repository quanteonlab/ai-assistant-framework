#!/usr/bin/env python3
"""
Add Book to Orchestration
==========================

Helper script to add books to the orchestration CSV.

Usage:
    python add_book.py mybook.pdf
    python add_book.py mybook.epub --pipeline SUMMARIZEDFLASHCARDS
    python add_book.py *.pdf  # Add multiple books
"""

import csv
import sys
import argparse
from pathlib import Path


def add_book_to_orchestration(book_filename: str, pipeline_type: str, output_folder: str, orchestration_csv: str):
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
    for row in rows:
        if row['book_filename'] == book_filename:
            print(f"[WARNING] Book already in orchestration: {book_filename}")
            print(f"  Current status: {row.get('status', 'PENDING')}")
            print(f"  Use --force to update")
            return False

    # Add new book
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

    # Write back
    with open(csv_path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"[OK] Added: {book_filename}")
    print(f"  Pipeline: {pipeline_type}")
    print(f"  Output: {output_folder}")
    return True


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
        """
    )

    parser.add_argument(
        'books',
        nargs='+',
        help='PDF or EPUB files to add'
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

    args = parser.parse_args()

    added_count = 0
    for book in args.books:
        if add_book_to_orchestration(book, args.pipeline, args.output, args.orchestration):
            added_count += 1

    print(f"\n[OK] Added {added_count}/{len(args.books)} books to orchestration")


if __name__ == "__main__":
    main()
