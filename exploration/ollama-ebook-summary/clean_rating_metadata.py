#!/usr/bin/env python3
"""
Clean Rating Metadata from Refined Book Notes

This script removes rating metadata from existing high-quality flashcard markdown files.
Removes lines like:
- **Rating threshold:** >= 8/10
- **Rating: 8/10**
"""

import os
import re
from pathlib import Path
from typing import List, Tuple


def clean_rating_lines(content: str) -> Tuple[str, int]:
    """
    Remove rating metadata lines from markdown content.

    Args:
        content: Original markdown content

    Returns:
        Tuple of (cleaned_content, num_lines_removed)
    """
    lines = content.split('\n')
    cleaned_lines = []
    removed_count = 0

    # Patterns to match and remove
    patterns = [
        r'^\*\*Rating threshold:\*\*\s*>=\s*\d+/10\s*$',  # **Rating threshold:** >= 8/10
        r'^\*\*Rating:\s*\d+/10\*\*\s*$',                 # **Rating: 8/10**
    ]

    for line in lines:
        # Check if line matches any removal pattern
        should_remove = False
        for pattern in patterns:
            if re.match(pattern, line.strip()):
                should_remove = True
                removed_count += 1
                break

        if not should_remove:
            cleaned_lines.append(line)

    cleaned_content = '\n'.join(cleaned_lines)

    # Clean up excessive blank lines (more than 2 consecutive)
    cleaned_content = re.sub(r'\n{4,}', '\n\n\n', cleaned_content)

    return cleaned_content, removed_count


def clean_markdown_file(file_path: Path, dry_run: bool = False) -> Tuple[bool, int]:
    """
    Clean a single markdown file by removing rating metadata.

    Args:
        file_path: Path to markdown file
        dry_run: If True, don't actually modify files

    Returns:
        Tuple of (was_modified, num_lines_removed)
    """
    try:
        # Read file
        with open(file_path, 'r', encoding='utf-8') as f:
            original_content = f.read()

        # Clean content
        cleaned_content, removed_count = clean_rating_lines(original_content)

        # Check if content changed
        if cleaned_content != original_content:
            if not dry_run:
                # Write cleaned content back
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(cleaned_content)
            return True, removed_count

        return False, 0

    except Exception as e:
        print(f"  [ERROR] Failed to process {file_path}: {e}")
        return False, 0


def clean_all_refined_notes(base_dir: str = ".", dry_run: bool = False) -> None:
    """
    Clean all markdown files in refined_docs/book_notes/.

    Args:
        base_dir: Base directory (where script is located)
        dry_run: If True, only show what would be changed
    """
    refined_docs_dir = Path(base_dir) / "refined_docs" / "book_notes"

    if not refined_docs_dir.exists():
        print(f"[INFO] Directory not found: {refined_docs_dir}")
        print("[INFO] No refined book notes to clean")
        return

    print(f"\n{'='*70}")
    print(f"  CLEANING RATING METADATA FROM REFINED BOOK NOTES")
    print(f"{'='*70}")
    print(f"Directory: {refined_docs_dir}")
    print(f"Mode: {'DRY RUN (no changes)' if dry_run else 'LIVE (modifying files)'}")
    print(f"{'='*70}\n")

    # Find all markdown files
    md_files = list(refined_docs_dir.rglob("*.md"))

    if not md_files:
        print("[INFO] No markdown files found")
        return

    print(f"[INFO] Found {len(md_files)} markdown files\n")

    # Process each file
    total_modified = 0
    total_lines_removed = 0

    for md_file in md_files:
        relative_path = md_file.relative_to(refined_docs_dir)

        was_modified, removed_count = clean_markdown_file(md_file, dry_run=dry_run)

        if was_modified:
            total_modified += 1
            total_lines_removed += removed_count
            status = "[DRY RUN]" if dry_run else "[CLEANED]"
            print(f"{status} {relative_path} - Removed {removed_count} rating line(s)")

    # Summary
    print(f"\n{'='*70}")
    print(f"  SUMMARY")
    print(f"{'='*70}")
    print(f"Total files scanned: {len(md_files)}")
    print(f"Files modified: {total_modified}")
    print(f"Rating lines removed: {total_lines_removed}")

    if dry_run:
        print(f"\n[INFO] This was a DRY RUN - no files were modified")
        print(f"[INFO] Run with --execute to actually clean the files")
    else:
        print(f"\n[SUCCESS] Cleanup complete!")

    print(f"{'='*70}\n")


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Clean rating metadata from refined book notes",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Dry run (see what would be changed)
  python clean_rating_metadata.py

  # Actually clean the files
  python clean_rating_metadata.py --execute
        """
    )

    parser.add_argument(
        '--execute',
        action='store_true',
        help='Actually modify files (default is dry run)'
    )

    parser.add_argument(
        '--base-dir',
        default='.',
        help='Base directory (default: current directory)'
    )

    args = parser.parse_args()

    # Run cleanup (dry run by default unless --execute is passed)
    clean_all_refined_notes(
        base_dir=args.base_dir,
        dry_run=not args.execute
    )


if __name__ == "__main__":
    main()
