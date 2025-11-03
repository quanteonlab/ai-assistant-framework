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
import os
import re
import requests
from pathlib import Path


def generate_relevancy_target(book_filename: str, api_base: str = None, model: str = None) -> str:
    """
    Generate a relevancy target description for a book using local LLM.

    Args:
        book_filename: Name of the book file
        api_base: Ollama API base URL (default from env or http://localhost:11434/api)
        model: Model to use (default from env or qwen2.5:latest)

    Returns:
        String description without ending punctuation
    """
    if api_base is None:
        api_base = os.getenv('OLLAMA_API_BASE', 'http://localhost:11434/api')

    if model is None:
        model = os.getenv('OLLAMA_MODEL', 'qwen2.5:latest')

    # Extract book title from filename (remove extension and clean up)
    book_title = Path(book_filename).stem
    # Clean up common patterns in filenames
    book_title = re.sub(r'^\d+[A-Z]?\d*\s*-+\s*', '', book_title)  # Remove prefixes like "2A001 -"
    book_title = re.sub(r'_', ' ', book_title)  # Replace underscores with spaces

    prompt = f"""Based on the book title: "{book_title}"

Generate a brief relevancy target description (5-10 words) that describes the main technical focus areas or topics this book covers.

Focus on: key technical concepts, programming techniques, methodologies, or domain-specific knowledge.

Format: Return ONLY the description without any punctuation at the end (no periods, commas, etc.)

Examples:
- "LLM implementation techniques and practical applications"
- "Distributed systems concepts and design patterns"
- "Python programming techniques and idioms"

Description:"""

    try:
        # Make request to Ollama API
        response = requests.post(
            f"{api_base}/generate",
            json={
                "model": model,
                "prompt": prompt,
                "stream": False
            },
            timeout=30
        )

        if response.status_code == 200:
            result = response.json()
            relevancy_target = result.get('response', '').strip()

            # Remove any trailing punctuation
            relevancy_target = re.sub(r'[.,;:!?]+$', '', relevancy_target)

            # If response is too long, truncate
            if len(relevancy_target) > 100:
                relevancy_target = ' '.join(relevancy_target.split()[:10])

            return relevancy_target if relevancy_target else ''
        else:
            print(f"[WARN] LLM API returned status {response.status_code}, skipping relevancy generation")
            return ''

    except requests.RequestException as e:
        print(f"[WARN] Failed to generate relevancy target: {e}")
        return ''
    except Exception as e:
        print(f"[WARN] Unexpected error generating relevancy target: {e}")
        return ''


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


def add_book_to_orchestration(book_filename: str, pipeline_type: str, output_folder: str, orchestration_csv: str, force: bool = False, auto_generate_relevancy: bool = True):
    """Add a book to the orchestration CSV."""

    # Read existing rows
    rows = []
    fieldnames = ['book_filename', 'pipeline_type', 'status', 'started_at', 'completed_at', 'output_folder', 'error_message', 'relevancy_target']

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

                # Generate relevancy target if empty and auto-generation is enabled
                if auto_generate_relevancy and not rows[idx].get('relevancy_target', '').strip():
                    print(f"[INFO] Generating relevancy target for: {book_filename}")
                    relevancy = generate_relevancy_target(book_filename)
                    if relevancy:
                        rows[idx]['relevancy_target'] = relevancy
                        print(f"  Relevancy: {relevancy}")

                print(f"[UPDATE] Updated: {book_filename}")
                break

    # Add new book if not exists
    if not book_exists:
        # Generate relevancy target if auto-generation is enabled
        relevancy_target = ''
        if auto_generate_relevancy:
            print(f"[INFO] Generating relevancy target for: {book_filename}")
            relevancy_target = generate_relevancy_target(book_filename)
            if relevancy_target:
                print(f"  Relevancy: {relevancy_target}")

        new_row = {
            'book_filename': book_filename,
            'pipeline_type': pipeline_type,
            'status': 'PENDING',
            'started_at': '',
            'completed_at': '',
            'output_folder': output_folder,
            'error_message': '',
            'relevancy_target': relevancy_target
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


def scan_and_add_books(input_folder: str, pipeline_type: str, output_folder: str, orchestration_csv: str, auto_generate_relevancy: bool = True):
    """
    Scan the input folder for PDF/EPUB files and add any that aren't in orchestration CSV.

    Args:
        input_folder: Folder to scan for books
        pipeline_type: Default pipeline type
        output_folder: Default output folder
        orchestration_csv: Path to orchestration CSV
        auto_generate_relevancy: Whether to automatically generate relevancy targets

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
        if add_book_to_orchestration(book_filename, pipeline_type, output_folder, orchestration_csv, force=False, auto_generate_relevancy=auto_generate_relevancy):
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

    parser.add_argument(
        '--no-auto-relevancy',
        action='store_true',
        help='Disable automatic relevancy target generation (LLM will not be called)'
    )

    args = parser.parse_args()

    # Determine if auto-relevancy generation should be enabled
    auto_generate_relevancy = not args.no_auto_relevancy

    # Handle scan mode
    if args.scan:
        print(f"[INFO] Scanning {args.input_folder} for PDF/EPUB files...")
        added_count = scan_and_add_books(
            args.input_folder,
            args.pipeline,
            args.output,
            args.orchestration,
            auto_generate_relevancy
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
        if add_book_to_orchestration(book, args.pipeline, args.output, args.orchestration, args.force, auto_generate_relevancy):
            added_count += 1

    print(f"\n[OK] Added {added_count}/{len(args.books)} books to orchestration")


if __name__ == "__main__":
    main()
