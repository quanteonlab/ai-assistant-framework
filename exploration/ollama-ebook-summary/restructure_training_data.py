#!/usr/bin/env python3
"""
Restructure Training Data CSV
==============================

This script restructures the flashcards_training_data.csv to have one row per
individual flashcard instead of one row per chapter with multiple flashcards.

This allows for individual rating of each flashcard.

Usage:
    python restructure_training_data.py flashcards/flashcards_training_data.csv
"""

import os
import sys
import csv
import re
from pathlib import Path

# Increase CSV field size limit
try:
    max_int = sys.maxsize
    while True:
        try:
            csv.field_size_limit(max_int)
            break
        except OverflowError:
            max_int = int(max_int / 10)
except Exception as e:
    print(f"Warning: Could not set CSV field size limit: {e}")


def extract_individual_flashcards(flashcards_output: str) -> list:
    """
    Extract individual flashcards from the combined output.

    Flashcards are separated by --- and have the format:
    #### Title
    Background context: ...

    :p Question
    ??x
    Answer
    x??

    Args:
        flashcards_output: Combined flashcard output string

    Returns:
        List of individual flashcard strings
    """
    # Split by --- separator
    flashcards = flashcards_output.split('\n---\n')

    # Clean up and filter empty entries
    individual_flashcards = []
    for fc in flashcards:
        fc = fc.strip()
        if fc and '####' in fc:  # Valid flashcard should have a title
            individual_flashcards.append(fc)

    return individual_flashcards


def extract_flashcard_title(flashcard: str) -> str:
    """Extract the title from a flashcard."""
    match = re.search(r'####\s+(.+?)(?:\n|$)', flashcard)
    if match:
        return match.group(1).strip()
    return "Untitled"


def restructure_csv(input_file: str, output_file: str = None):
    """
    Restructure the CSV to have one row per flashcard.

    Args:
        input_file: Path to original training data CSV
        output_file: Path to output CSV (default: input_file with '_per_flashcard' suffix)
    """
    if output_file is None:
        base = os.path.splitext(input_file)[0]
        output_file = f"{base}_per_flashcard.csv"

    print(f"Reading from: {input_file}")
    print(f"Writing to: {output_file}")

    # Read original CSV
    rows = []
    with open(input_file, 'r', encoding='utf-8', newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)

    print(f"Loaded {len(rows)} chapter entries")

    # New structure: one row per flashcard
    new_fieldnames = [
        'source_file',
        'chapter_title',
        'flashcard_title',
        'flashcard_content',
        'flashcard_length',
        'input_text_excerpt',
        'model',
        'timestamp',
        'usefulness_rating'
    ]

    new_rows = []
    total_flashcards = 0

    for row in rows:
        source_file = row.get('source_file', '')
        chapter_title = row.get('title', '')
        input_text = row.get('input_text', '')
        model = row.get('model', '')
        timestamp = row.get('timestamp', '')
        flashcards_output = row.get('flashcards_output', '')

        # Extract individual flashcards
        individual_flashcards = extract_individual_flashcards(flashcards_output)

        # Create input excerpt (first 200 chars)
        input_excerpt = input_text[:200] + '...' if len(input_text) > 200 else input_text

        for flashcard in individual_flashcards:
            flashcard_title = extract_flashcard_title(flashcard)

            new_row = {
                'source_file': source_file,
                'chapter_title': chapter_title,
                'flashcard_title': flashcard_title,
                'flashcard_content': flashcard,
                'flashcard_length': len(flashcard),
                'input_text_excerpt': input_excerpt,
                'model': model,
                'timestamp': timestamp,
                'usefulness_rating': ''  # Empty, ready to be filled
            }
            new_rows.append(new_row)
            total_flashcards += 1

    print(f"Extracted {total_flashcards} individual flashcards")

    # Write new CSV
    with open(output_file, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=new_fieldnames)
        writer.writeheader()
        writer.writerows(new_rows)

    print(f"[OK] Restructured CSV saved to: {output_file}")
    print(f"\nNew structure:")
    print(f"  - Chapters: {len(rows)}")
    print(f"  - Individual flashcards: {total_flashcards}")
    print(f"  - Average flashcards per chapter: {total_flashcards / len(rows):.1f}")


def main():
    if len(sys.argv) < 2:
        print("Usage: python restructure_training_data.py <input_csv> [output_csv]")
        sys.exit(1)

    input_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else None

    if not os.path.exists(input_file):
        print(f"Error: File not found: {input_file}")
        sys.exit(1)

    restructure_csv(input_file, output_file)


if __name__ == "__main__":
    main()
