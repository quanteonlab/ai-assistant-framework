#!/usr/bin/env python3
"""
Flashcard Rating Script
=======================

This script reviews existing flashcards in the training data CSV and rates
unrated flashcards for technical usefulness and long-term value.

Features:
- Adds 'usefulness_rating' column if it doesn't exist
- Rates only flashcards that don't have a rating yet
- Creates a high_quality/ folder for flashcards above the threshold
- Consolidates high-quality flashcards into single markdown files per book
- Automatically splits files when they exceed 2000 lines

Usage:
    python rate_flashcards.py flashcards/flashcards_training_data.csv
    python rate_flashcards.py flashcards/flashcards_training_data.csv --threshold 7
    python rate_flashcards.py flashcards/flashcards_training_data.csv --model qwen2.5:latest

"""

import os
import sys
import csv
import time
import argparse
from pathlib import Path
from typing import Optional
import requests
import json
from urllib.parse import urljoin
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Increase CSV field size limit to handle large text fields
try:
    max_int = sys.maxsize
    while True:
        try:
            # Attempt to set the maximum field size limit
            csv.field_size_limit(max_int)
            break  # Break the loop if successful
        except OverflowError:
            # Reduce max_int and retry
            max_int = int(max_int / 10)
except Exception as e:
    print(f"Warning: Could not set CSV field size limit: {e}")

# Import Config and helper functions from flashcard.py
# We'll use absolute imports to access the existing functions
try:
    from flashcard import Config, make_api_request, sanitize_filename
except ImportError:
    print("Error: Could not import from flashcard.py")
    print("Make sure flashcard.py is in the same directory")
    sys.exit(1)


def rate_flashcard(api_base: str, model: str, flashcard: str, rating_prompt: str) -> int:
    """
    Rate a flashcard's technical usefulness on a scale of 1-10.

    Args:
        api_base: API base URL
        model: Model name to use for rating
        flashcard: The flashcard content to rate
        rating_prompt: The rating prompt template

    Returns:
        Integer rating from 1-10, or 0 if rating fails
    """
    # Format the prompt with the flashcard content
    full_prompt = rating_prompt.replace("{flashcard}", flashcard)

    payload = {
        "model": model,
        "prompt": full_prompt,
        "stream": False
    }

    result = make_api_request(api_base, "generate", payload)
    if result:
        response = result.get("response", "").strip()
        # Extract the first number from the response
        import re
        match = re.search(r'\b([1-9]|10)\b', response)
        if match:
            return int(match.group(1))

    # Return 0 if rating fails
    return 0


def consolidate_high_quality_flashcards(rows: list, threshold: int, high_quality_dir: str, max_lines: int = 2000):
    """
    Consolidate high-quality flashcards into single markdown files per source.

    Args:
        rows: List of all CSV rows
        threshold: Minimum rating for inclusion
        high_quality_dir: Output directory
        max_lines: Maximum lines per file before starting a new part

    Returns:
        Total number of high-quality flashcards consolidated
    """
    # Group flashcards by source
    by_source = {}

    for row in rows:
        rating = row.get('usefulness_rating', '').strip()
        try:
            rating_int = int(rating)
        except (ValueError, TypeError):
            continue

        if rating_int >= threshold:
            source = row.get('source_file', 'unknown')
            if source not in by_source:
                by_source[source] = []
            by_source[source].append(row)

    # Create consolidated markdown files
    total_hq_flashcards = 0

    print(f"\nConsolidating high-quality flashcards by source...")

    for source, flashcards in by_source.items():
        if not flashcards:
            continue

        print(f"\n  Source: {source} ({len(flashcards)} high-quality flashcards)")

        safe_source = sanitize_filename(source, max_length=30)
        part_num = 1
        line_count = 0
        current_file = None

        for idx, row in enumerate(flashcards):
            flashcard = row.get('flashcard_content', row.get('flashcards_output', '')).strip()
            if not flashcard:
                continue

            # Calculate lines in this flashcard
            flashcard_lines = flashcard.count('\n') + 3  # +3 for separators

            # Check if we need to start a new file
            if current_file is None or line_count + flashcard_lines > max_lines:
                if current_file:
                    current_file.close()
                    print(f"    Created: {safe_source}_part{part_num-1:02d}.md ({line_count} lines)")

                filename = f"{safe_source}_part{part_num:02d}.md"
                filepath = os.path.join(high_quality_dir, filename)
                current_file = open(filepath, 'w', encoding='utf-8')

                # Write header
                current_file.write(f"# High-Quality Flashcards: {source} (Part {part_num})\n\n")
                current_file.write("---\n\n")

                line_count = 4  # Header lines
                part_num += 1

            # Write flashcard content (without metadata)
            current_file.write(flashcard)
            current_file.write("\n\n---\n\n")

            line_count += flashcard_lines + 2
            total_hq_flashcards += 1

        if current_file:
            current_file.close()
            print(f"    Created: {safe_source}_part{part_num-1:02d}.md ({line_count} lines)")

    return total_hq_flashcards

def process_training_data(csv_file: str, config: Config, api_base: str, model: str,
                         threshold: int = 8, output_dir: str = None, verbose: bool = False):
    """
    Process training data CSV and rate unrated flashcards.

    Args:
        csv_file: Path to flashcards_training_data.csv
        config: Configuration object
        api_base: API base URL
        model: Model name to use for rating
        threshold: Minimum rating for high-quality folder (default: 6)
        output_dir: Output directory for high-quality flashcards (default: same as CSV)
        verbose: Print detailed information
    """
    if not os.path.exists(csv_file):
        print(f"Error: CSV file not found: {csv_file}")
        sys.exit(1)

    # Determine output directory
    if output_dir is None:
        output_dir = os.path.dirname(csv_file) or "."

    high_quality_dir = os.path.join(output_dir, "high_quality")
    os.makedirs(high_quality_dir, exist_ok=True)

    # Get rating prompt
    rating_prompt = config.get_prompt('flashcard_rating')

    # Read the CSV
    rows = []
    fieldnames = []
    has_rating_column = False

    with open(csv_file, 'r', encoding='utf-8', newline='') as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames

        # Check if usefulness_rating column exists
        if 'usefulness_rating' in fieldnames:
            has_rating_column = True
        else:
            # Add the column
            fieldnames = list(fieldnames) + ['usefulness_rating']

        # Read all rows
        for row in reader:
            # Ensure the row has the usefulness_rating key (even if empty)
            if 'usefulness_rating' not in row:
                row['usefulness_rating'] = ''
            rows.append(row)

    print(f"Loaded {len(rows)} flashcard entries from {csv_file}")
    print(f"Rating column exists: {has_rating_column}")
    print(f"Rating threshold: {threshold}/10")
    print(f"High-quality output: {high_quality_dir}/\n")

    # Count how many need rating
    unrated_count = 0
    for row in rows:
        rating = row.get('usefulness_rating', '').strip()
        if not rating or rating == '0':
            unrated_count += 1

    print(f"Flashcards needing rating: {unrated_count}\n")

    if unrated_count == 0:
        print("All flashcards are already rated!")
        return

    # Process unrated flashcards
    rated_count = 0
    high_quality_count = 0
    skipped_count = 0
    error_count = 0

    for idx, row in enumerate(rows):
        rating = row.get('usefulness_rating', '').strip()

        # Skip if already rated (and rating is not 0 or empty)
        if rating and rating != '0':
            try:
                rating_int = int(rating)
                if rating_int > 0:
                    skipped_count += 1
                    if verbose:
                        print(f"[Skipped] Already rated ({rating}/10): {row.get('title', '')[:50]}...")

                    # Track high-quality count (will be consolidated at the end)
                    if rating_int >= threshold:
                        high_quality_count += 1

                    continue
            except ValueError:
                # Invalid rating, will re-rate
                pass

        # Initialize rating to 0 if empty or missing
        if not rating:
            row['usefulness_rating'] = '0'

        # Get flashcard content (support both old and new CSV structures)
        flashcard = row.get('flashcard_content', row.get('flashcards_output', '')).strip()
        title = row.get('flashcard_title', row.get('title', f'flashcard_{idx}'))
        chapter_title = row.get('chapter_title', row.get('title', ''))
        source_file = row.get('source_file', 'unknown')

        if not flashcard:
            if verbose:
                print(f"[Skipped] No flashcard content: {title[:50]}...")
            continue

        print(f"[{rated_count + 1}/{unrated_count}] Rating: {title[:50]}...")

        try:
            # Rate the flashcard
            start_time = time.time()
            usefulness_rating = rate_flashcard(api_base, model, flashcard, rating_prompt)
            elapsed = time.time() - start_time

            # Update the row
            row['usefulness_rating'] = str(usefulness_rating)
            rated_count += 1

            print(f"  Rating: {usefulness_rating}/10 (took {elapsed:.2f}s)")

            if verbose and usefulness_rating > 0:
                print(f"  AI response: {usefulness_rating}")

            # Track high-quality flashcards (will be consolidated at the end)
            if usefulness_rating >= threshold:
                high_quality_count += 1

        except Exception as e:
            error_count += 1
            print(f"  ✗ Error rating flashcard: {e}")
            # Set rating to 0 on error so it can be retried later
            row['usefulness_rating'] = '0'
            continue

    # Write updated CSV back (always write, even if there were errors)
    print(f"\nWriting updated CSV...")
    try:
        with open(csv_file, 'w', encoding='utf-8', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        print(f"CSV updated successfully")
    except Exception as e:
        print(f"Error writing CSV: {e}")
        return

    # Consolidate high-quality flashcards into consolidated markdown files
    total_consolidated = consolidate_high_quality_flashcards(rows, threshold, high_quality_dir)

    print(f"\n{'='*60}")
    print(f"Rating complete!")
    print(f"  Total flashcards: {len(rows)}")
    print(f"  Already rated (skipped): {skipped_count}")
    print(f"  Newly rated: {rated_count}")
    if error_count > 0:
        print(f"  Errors: {error_count}")
    print(f"  High-quality consolidated: {total_consolidated}")
    print(f"  Updated CSV: {csv_file}")
    print(f"  High-quality folder: {high_quality_dir}/")
    print(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser(
        description="Rate unrated flashcards in training data CSV",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Rate with default threshold (6/10)
  python rate_flashcards.py flashcards/flashcards_training_data.csv

  # Custom threshold - only save 8-10 rated flashcards
  python rate_flashcards.py flashcards/flashcards_training_data.csv --threshold 8

  # Use specific model for rating
  python rate_flashcards.py flashcards/flashcards_training_data.csv --model qwen2.5:latest

  # Verbose output
  python rate_flashcards.py flashcards/flashcards_training_data.csv -v

Output:
  - Updates the CSV file with ratings in the 'usefulness_rating' column
  - Creates consolidated .md files per book in high_quality/ folder
  - Automatically splits into multiple parts when exceeding 2000 lines
  - Flashcards are saved without metadata headers for cleaner reading
        """
    )

    # Required arguments
    parser.add_argument('csv_file', help='Path to flashcards_training_data.csv')

    # Optional arguments
    parser.add_argument('-t', '--threshold', type=int, default=8,
                       help='Minimum rating for high-quality folder (default: 6)')
    parser.add_argument('-m', '--model',
                       help='Model to use for rating (default from config)')
    parser.add_argument('-o', '--output',
                       help='Output directory (default: same as CSV file directory)')
    parser.add_argument('-v', '--verbose', action='store_true',
                       help='Verbose output')

    args = parser.parse_args()

    # Load config
    config = Config()

    # Get model (use provided or default)
    model = args.model or config.defaults.get('summary', 'llama3.2:latest')

    # Get API base
    api_base = os.getenv('OLLAMA_API_BASE', 'http://localhost:11434/api')

    # Process the training data
    process_training_data(
        csv_file=args.csv_file,
        config=config,
        api_base=api_base,
        model=model,
        threshold=args.threshold,
        output_dir=args.output,
        verbose=args.verbose
    )


if __name__ == "__main__":
    main()
