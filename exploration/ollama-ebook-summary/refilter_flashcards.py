#!/usr/bin/env python3
"""
Re-filter Existing Flashcards

This script re-evaluates existing flashcards with a new relevancy target and
re-filters them into refined_docs/book_notes/ based on updated usefulness scores.

Usage:
    python refilter_flashcards.py "Book Name.pdf" --relevancy-target "new focus area"
    python refilter_flashcards.py "Book Name.pdf" --threshold 8
"""

import os
import re
import sys
import json
import time
import argparse
import requests
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


def sanitize_book_name(book_filename: str) -> str:
    """Sanitize book filename to match folder names."""
    name = Path(book_filename).stem
    name = name.replace(" ", "-")
    name = re.sub(r'[^\w\-]', '', name)
    return name


def generate_book_prefix(book_name: str, max_length: int = 5) -> str:
    """Generate a short prefix from book name for use in filenames."""
    name = book_name.lower()
    name = re.sub(r'[<>:"/\\|?*\-_]', ' ', name)

    # Remove common filler words
    remove_words = ['the', 'an', 'a', 'and', 'or', 'for', 'to', 'of', 'in', 'processed', 'part']
    words = name.split()
    words = [w for w in words if w not in remove_words and len(w) > 0]

    if not words:
        return book_name[:max_length].lower()

    # Create meaningful abbreviation
    prefix = ''.join(w[0] for w in words[:max_length])

    # If prefix is too short, add more characters from first word
    if len(prefix) < max_length and len(words[0]) > 1:
        prefix += words[0][1:max_length - len(prefix)]

    return prefix[:max_length]


def parse_flashcards_from_markdown(md_file: Path) -> List[Dict[str, str]]:
    """
    Parse individual flashcards from a markdown file.

    Returns:
        List of flashcard dictionaries with 'content' and 'title' keys
    """
    try:
        with open(md_file, 'r', encoding='utf-8') as f:
            content = f.read()
    except Exception as e:
        print(f"  [ERROR] Failed to read {md_file}: {e}")
        return []

    # Split by separator (---)
    parts = re.split(r'\n---\n', content)

    flashcards = []
    for part in parts:
        part = part.strip()
        if not part or part.startswith('#'):
            continue

        # Extract title (if present) - usually starts with ####
        title_match = re.search(r'^####\s+(.+)$', part, re.MULTILINE)
        title = title_match.group(1) if title_match else "Flashcard"

        flashcards.append({
            'content': part,
            'title': title
        })

    return flashcards


def rate_flashcard_usefulness(
    flashcard_content: str,
    book_title: str,
    relevancy_target: str,
    api_base: str,
    model: str,
    rating_prompt: str
) -> Optional[int]:
    """
    Rate a flashcard's usefulness based on relevancy target.

    Returns:
        Rating from 1-10, or None if rating fails
    """
    # Build the rating prompt with relevancy context
    if relevancy_target:
        full_prompt = f"Focus area for this book: {relevancy_target}\n\n{rating_prompt}"
    else:
        full_prompt = rating_prompt

    # Replace placeholders
    full_prompt = full_prompt.replace("{book_title}", book_title)
    full_prompt = full_prompt.replace("{flashcard_content}", flashcard_content)

    try:
        response = requests.post(
            f"{api_base}/generate",
            json={
                "model": model,
                "prompt": full_prompt,
                "stream": False
            },
            timeout=30
        )

        if response.status_code == 200:
            result = response.json()
            rating_text = result.get('response', '').strip()

            # Extract numeric rating
            match = re.search(r'\b([1-9]|10)\b', rating_text)
            if match:
                return int(match.group(1))

        return None

    except Exception as e:
        print(f"  [WARN] Rating request failed: {e}")
        return None


def refilter_flashcards(
    book_filename: str,
    relevancy_target: str = None,
    rating_threshold: int = 8,
    flashcards_dir: str = "flashcards",
    output_dir: str = "refined_docs/book_notes",
    verbose: bool = False
) -> bool:
    """
    Re-filter existing flashcards based on new relevancy target.

    Args:
        book_filename: Name of the book file
        relevancy_target: New relevancy target for filtering
        rating_threshold: Minimum rating for high-quality flashcards
        flashcards_dir: Directory containing existing flashcards
        output_dir: Output directory for refined notes
        verbose: Enable verbose output

    Returns:
        True if successful, False otherwise
    """
    script_dir = Path(__file__).parent.absolute()

    # Get configuration
    api_base = os.getenv('OLLAMA_API_BASE', 'http://localhost:11434/api')
    rating_model = os.getenv('OLLAMA_RATING_MODEL', 'qwen2.5:latest')

    # Load rating prompt from config
    import yaml
    config_path = script_dir / "_config.yaml"
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
            rating_prompt = config.get('prompts', {}).get('flashcard_usefulness_rating', {}).get('prompt', '')
            if not rating_prompt:
                print("[ERROR] Rating prompt not found in _config.yaml")
                return False
    except Exception as e:
        print(f"[ERROR] Failed to load config: {e}")
        return False

    # Sanitize book name
    book_basename = sanitize_book_name(book_filename)
    book_prefix = generate_book_prefix(book_basename, max_length=5)
    book_title = Path(book_filename).stem

    print(f"\n{'='*70}")
    print(f"  RE-FILTERING FLASHCARDS")
    print(f"{'='*70}")
    print(f"Book: {book_filename}")
    print(f"Relevancy target: {relevancy_target or '(none)'}")
    print(f"Rating threshold: >= {rating_threshold}/10")
    print(f"Rating model: {rating_model}")
    print(f"{'='*70}\n")

    # Find flashcards directory
    flashcards_path = script_dir / flashcards_dir / book_basename
    if not flashcards_path.exists():
        print(f"[ERROR] Flashcards directory not found: {flashcards_path}")
        return False

    # Get all markdown files
    md_files = sorted(flashcards_path.glob("*.md"))
    if not md_files:
        print(f"[ERROR] No flashcard files found in: {flashcards_path}")
        return False

    print(f"[INFO] Found {len(md_files)} flashcard file(s)")

    # Parse all flashcards
    all_flashcards = []
    for md_file in md_files:
        if verbose:
            print(f"  Reading: {md_file.name}")
        flashcards = parse_flashcards_from_markdown(md_file)
        all_flashcards.extend(flashcards)

    print(f"[INFO] Total flashcards to re-evaluate: {len(all_flashcards)}\n")

    if len(all_flashcards) == 0:
        print("[WARN] No flashcards found to filter")
        return False

    # Re-rate and filter flashcards
    high_quality_flashcards = []
    rating_counts = {i: 0 for i in range(1, 11)}

    print("[INFO] Re-rating flashcards...")
    for idx, flashcard in enumerate(all_flashcards, 1):
        if verbose or idx % 10 == 0:
            print(f"  Progress: {idx}/{len(all_flashcards)}", end='\r')

        # Rate the flashcard
        rating = rate_flashcard_usefulness(
            flashcard['content'],
            book_title,
            relevancy_target or '',
            api_base,
            rating_model,
            rating_prompt
        )

        if rating is not None:
            rating_counts[rating] += 1

            if rating >= rating_threshold:
                flashcard['rating'] = rating
                high_quality_flashcards.append(flashcard)

        # Small delay to avoid overwhelming API
        time.sleep(0.1)

    print(f"\n\n[INFO] Rating distribution:")
    for rating in range(10, 0, -1):
        count = rating_counts.get(rating, 0)
        bar = '█' * (count // 2) if count > 0 else ''
        print(f"  {rating:2d}/10: {count:3d} {bar}")

    print(f"\n[INFO] High-quality flashcards (>= {rating_threshold}/10): {len(high_quality_flashcards)}")

    if len(high_quality_flashcards) == 0:
        print("[WARN] No flashcards met the quality threshold")
        return True  # Not an error, just nothing to save

    # Create output directory
    output_path = script_dir / output_dir / book_basename
    output_path.mkdir(parents=True, exist_ok=True)

    # Clear existing refined notes
    for old_file in output_path.glob("*.md"):
        old_file.unlink()
        if verbose:
            print(f"  Removed old file: {old_file.name}")

    # Write high-quality flashcards to new files
    # Group by approximately 3 flashcards per file or 50k chars
    chapters_per_file = 3
    max_text_size = 50000

    current_file_num = 1
    current_file_flashcards = []
    current_file_text_size = 0

    for flashcard in high_quality_flashcards:
        # Check if we should start a new file
        if (len(current_file_flashcards) >= chapters_per_file or
            current_file_text_size >= max_text_size):
            # Write current file
            _write_refined_file(
                output_path,
                book_prefix,
                book_basename,
                current_file_num,
                current_file_flashcards
            )

            # Reset for new file
            current_file_num += 1
            current_file_flashcards = []
            current_file_text_size = 0

        current_file_flashcards.append(flashcard)
        current_file_text_size += len(flashcard['content'])

    # Write remaining flashcards
    if current_file_flashcards:
        _write_refined_file(
            output_path,
            book_prefix,
            book_basename,
            current_file_num,
            current_file_flashcards
        )

    print(f"\n[SUCCESS] Re-filtering complete!")
    print(f"[INFO] Output: {output_path}")
    print(f"[INFO] Created {current_file_num} refined note file(s)")

    return True


def _write_refined_file(
    output_path: Path,
    book_prefix: str,
    book_basename: str,
    file_num: int,
    flashcards: List[Dict[str, str]]
):
    """Write a batch of flashcards to a refined notes file."""
    # Use first flashcard title for filename
    first_title = flashcards[0]['title']
    short_title = _sanitize_filename(first_title, max_length=15)

    filename = f"{book_prefix}_hq_part{file_num:02d}_{short_title}.md"
    filepath = output_path / filename

    with open(filepath, 'w', encoding='utf-8') as f:
        # Write header
        f.write(f"# High-Quality Flashcards: {book_basename} (Part {file_num})\n\n")
        f.write(f"**Starting Chapter:** {flashcards[0]['title']}\n\n")
        f.write("---\n\n")

        # Write flashcards (no rating metadata)
        for flashcard in flashcards:
            f.write(flashcard['content'])
            f.write("\n\n---\n\n")

    print(f"  [WRITE] {filename} ({len(flashcards)} flashcards)")


def _sanitize_filename(filename: str, max_length: int = 50) -> str:
    """Sanitize a string for use in filenames."""
    # Remove or replace problematic characters
    filename = re.sub(r'[<>:"/\\|?*]', '_', filename)

    # Replace spaces with underscores
    filename = filename.replace(' ', '_')

    # Remove multiple consecutive underscores
    filename = re.sub(r'_+', '_', filename)

    # Trim to max length
    if len(filename) > max_length:
        filename = filename[:max_length]

    # Remove trailing underscores or dots
    filename = filename.rstrip('_.')

    return filename


def main():
    parser = argparse.ArgumentParser(
        description="Re-filter existing flashcards with new relevancy target",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Re-filter with new relevancy target
  python refilter_flashcards.py "My Book.pdf" --relevancy-target "advanced techniques"

  # Re-filter with different threshold
  python refilter_flashcards.py "My Book.pdf" --threshold 9

  # Re-filter with both
  python refilter_flashcards.py "My Book.pdf" --relevancy-target "specific topic" --threshold 7
        """
    )

    parser.add_argument(
        'book_filename',
        help='Name of the book file (e.g., "My Book.pdf")'
    )

    parser.add_argument(
        '--relevancy-target',
        help='New relevancy target for filtering (from orchestration.csv)'
    )

    parser.add_argument(
        '--threshold',
        type=int,
        default=8,
        help='Minimum rating for high-quality flashcards (default: 8)'
    )

    parser.add_argument(
        '--flashcards-dir',
        default='flashcards',
        help='Input directory with existing flashcards (default: flashcards)'
    )

    parser.add_argument(
        '--output-dir',
        default='refined_docs/book_notes',
        help='Output directory for refined notes (default: refined_docs/book_notes)'
    )

    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='Enable verbose output'
    )

    args = parser.parse_args()

    success = refilter_flashcards(
        args.book_filename,
        args.relevancy_target,
        args.threshold,
        args.flashcards_dir,
        args.output_dir,
        args.verbose
    )

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
