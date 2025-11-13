#!/usr/bin/env python3
r"""
Fix Math Spacing Issues in Markdown Files
==========================================

This script fixes spacing issues around inline math delimiters where the
conversion removed necessary spaces around the delimiters.

Examples of fixes:
- `-$f(t)$is` → `- $f(t)$ is`
- `parameter$\alpha$can` → `parameter $\alpha$ can`
- `$$equation$$Where:` → `$$equation$$\n\nWhere:`

Usage:
    python fix_math_spacing.py
    python fix_math_spacing.py --dry-run
    python fix_math_spacing.py --dir custom_dir

Author: AI Assistant Framework
License: Same as parent project
"""

import os
import re
import argparse
from pathlib import Path
from typing import Tuple, List

# Color codes for terminal output
class Colors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'

def print_header(text: str):
    """Print a formatted header."""
    print(f"\n{Colors.HEADER}{Colors.BOLD}{'='*70}")
    print(f"  {text}")
    print(f"{'='*70}{Colors.ENDC}\n")

def print_success(text: str):
    """Print a success message."""
    print(f"{Colors.OKGREEN}[OK] {text}{Colors.ENDC}")

def print_warning(text: str):
    """Print a warning message."""
    print(f"{Colors.WARNING}[WARNING] {text}{Colors.ENDC}")

def print_info(text: str):
    """Print an info message."""
    print(f"{Colors.OKBLUE}[INFO] {text}{Colors.ENDC}")

def fix_math_spacing(content: str) -> Tuple[str, int]:
    """
    Fix spacing issues around math delimiters.

    Args:
        content: The file content to process

    Returns:
        Tuple of (fixed_content, number_of_fixes)
    """
    fixes = 0
    original = content

    # Fix: word$math$ (no space before) → word $math$
    # Match: letter/digit followed immediately by $
    pattern1 = r'([a-zA-Z0-9])(\$[^\$\n]+?\$)'
    content = re.sub(pattern1, r'\1 \2', content)

    # Fix: $math$word (no space after) → $math$ word
    # Match: $ followed immediately by letter/digit
    pattern2 = r'(\$[^\$\n]+?\$)([a-zA-Z])'
    content = re.sub(pattern2, r'\1 \2', content)

    # Fix: -$math$ (dash followed by no space) → - $math$
    # This handles list items specifically
    pattern3 = r'(^|\n)([-*+])(\$[^\$\n]+?\$)'
    content = re.sub(pattern3, r'\1\2 \3', content, flags=re.MULTILINE)

    # Fix: $$block$$Word → $$block$$\n\nWord
    # Add double newline after block math if followed immediately by capital letter
    pattern4 = r'(\$\$[^\$]+?\$\$)([A-Z])'
    content = re.sub(pattern4, r'\1\n\n\2', content)

    # Fix: :$$block → :\n$$block
    # Add newline before block math if preceded by colon without space
    pattern5 = r'(:)(\$\$)'
    content = re.sub(pattern5, r'\1\n\2', content)

    # Count fixes
    if content != original:
        fixes = 1  # At least one fix was made

    return content, fixes

def process_file(file_path: Path, dry_run: bool = False) -> Tuple[bool, int]:
    """
    Process a single markdown file.

    Args:
        file_path: Path to the markdown file
        dry_run: If True, don't modify the file, just report what would change

    Returns:
        Tuple of (was_modified, number_of_fixes)
    """
    try:
        # Read the file
        with open(file_path, 'r', encoding='utf-8') as f:
            original_content = f.read()

        # Fix spacing
        fixed_content, fixes = fix_math_spacing(original_content)

        # Check if anything changed
        if fixes == 0 or fixed_content == original_content:
            return False, 0

        # Write back if not dry run
        if not dry_run:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(fixed_content)
            print_success(f"Fixed math spacing in: {file_path.name}")
        else:
            print_info(f"Would fix math spacing in: {file_path.name}")

        return True, fixes

    except Exception as e:
        print_warning(f"Error processing {file_path}: {e}")
        return False, 0

def find_markdown_files(directories: List[Path]) -> List[Path]:
    """
    Find all markdown files in the specified directories.

    Args:
        directories: List of directories to search

    Returns:
        List of markdown file paths
    """
    markdown_files = []

    for directory in directories:
        if not directory.exists():
            print_warning(f"Directory not found: {directory}")
            continue

        # Recursively find all .md files
        for md_file in directory.rglob("*.md"):
            markdown_files.append(md_file)

    return markdown_files

def main():
    parser = argparse.ArgumentParser(
        description="Fix spacing issues around math delimiters in .md files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Fix all files in refined_docs and flashcards
  python fix_math_spacing.py

  # Dry run to preview changes
  python fix_math_spacing.py --dry-run

  # Fix a specific directory
  python fix_math_spacing.py --dir custom_folder

Fixes Applied:
  word$math$     → word $math$
  $math$word     → $math$ word
  -$math$        → - $math$
  $$block$$Word  → $$block$$\\n\\nWord
  :$$block       → :\\n$$block
        """
    )

    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Preview changes without modifying files'
    )

    parser.add_argument(
        '--dir',
        action='append',
        dest='directories',
        help='Directory to process (can be specified multiple times)'
    )

    args = parser.parse_args()

    # Determine directories to process
    script_dir = Path(__file__).parent.absolute()

    if args.directories:
        # Use custom directories
        directories = [Path(d) if Path(d).is_absolute() else script_dir / d
                      for d in args.directories]
    else:
        # Default directories
        directories = [
            script_dir / "refined_docs",
            script_dir / "flashcards"
        ]

    # Print header
    print_header("Math Spacing Fixer")

    if args.dry_run:
        print_warning("DRY RUN MODE - No files will be modified")
        print()

    # Find all markdown files
    print_info("Searching for markdown files...")
    markdown_files = find_markdown_files(directories)

    if not markdown_files:
        print_warning("No markdown files found")
        return

    print_success(f"Found {len(markdown_files)} markdown file(s)")
    print()

    # Process each file
    total_files_modified = 0

    for file_path in markdown_files:
        was_modified, _ = process_file(file_path, args.dry_run)

        if was_modified:
            total_files_modified += 1

    # Print summary
    print()
    print_header("Summary")

    print(f"Total markdown files processed: {len(markdown_files)}")
    print(f"Files with spacing fixes: {total_files_modified}")

    if args.dry_run:
        print()
        print_info("This was a dry run. Use without --dry-run to apply changes.")
    elif total_files_modified > 0:
        print()
        print_success("Math spacing successfully fixed!")
    else:
        print()
        print_info("No spacing issues found to fix.")

if __name__ == "__main__":
    main()
