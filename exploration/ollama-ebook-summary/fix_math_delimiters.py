#!/usr/bin/env python3
r"""
Fix LaTeX Math Delimiters in Markdown Files
============================================

This script converts LaTeX math delimiters to standard markdown math format:
- Block math: \[ ... \] → $$ ... $$
- Inline math: \( ... \) → $ ... $

It processes all .md files in the refined_docs and flashcards directories.

Usage:
    python fix_math_delimiters.py
    python fix_math_delimiters.py --dry-run  # Preview changes without modifying files
    python fix_math_delimiters.py --dir custom_dir  # Process a specific directory

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

def convert_math_delimiters(content: str) -> Tuple[str, int]:
    """
    Convert LaTeX math delimiters to markdown format and fix spacing issues.

    Args:
        content: The file content to process

    Returns:
        Tuple of (converted_content, number_of_replacements)
    """
    replacements = 0

    # Replace block math: \[ ... \] → $$ ... $$
    # Use a non-greedy match to handle multiple equations on the same line
    # Keep the content as-is (with any spaces/newlines)
    block_pattern = r'\\\[(.*?)\\\]'
    matches = re.findall(block_pattern, content, re.DOTALL)
    replacements += len(matches)
    content = re.sub(block_pattern, r'$$\1$$', content, flags=re.DOTALL)

    # Replace inline math: \( ... \) → $...$
    # Strip spaces from inline math to ensure tight formatting: $f(t)$ not $ f(t) $
    inline_latex_pattern = r'\\\((.*?)\\\)'

    def replace_inline_latex(match):
        # Strip leading/trailing whitespace from captured content
        return f'${match.group(1).strip()}$'

    matches = re.findall(inline_latex_pattern, content, re.DOTALL)
    replacements += len(matches)
    content = re.sub(inline_latex_pattern, replace_inline_latex, content, flags=re.DOTALL)

    # Fix existing inline math with spaces immediately inside delimiters: $ ... $ → $...$
    # Pattern 1: Match $ + spaces + content + spaces + $ (spaces on both sides)
    # Use negative lookahead/lookbehind to avoid matching $$ (block math)
    pattern1 = r'(?<!\$)\$\s+([^\$\n]+?)\s+\$(?!\$)'

    def replace_inline_both_spaces(match):
        return f'${match.group(1).strip()}$'

    matches = re.findall(pattern1, content)
    replacements += len(matches)
    content = re.sub(pattern1, replace_inline_both_spaces, content)

    # Pattern 2: Match $ + content + spaces + $ (trailing space only)
    pattern2 = r'(?<!\$)\$([^\$\s][^\$\n]*?)\s+\$(?!\$)'

    def replace_inline_trailing_space(match):
        return f'${match.group(1).strip()}$'

    matches = re.findall(pattern2, content)
    replacements += len(matches)
    content = re.sub(pattern2, replace_inline_trailing_space, content)

    # Pattern 3: Match $ + spaces + content + $ (leading space only)
    pattern3 = r'(?<!\$)\$\s+([^\$\n]*?[^\$\s])\$(?!\$)'

    def replace_inline_leading_space(match):
        return f'${match.group(1).strip()}$'

    matches = re.findall(pattern3, content)
    replacements += len(matches)
    content = re.sub(pattern3, replace_inline_leading_space, content)

    return content, replacements

def process_file(file_path: Path, dry_run: bool = False) -> Tuple[bool, int]:
    """
    Process a single markdown file.

    Args:
        file_path: Path to the markdown file
        dry_run: If True, don't modify the file, just report what would change

    Returns:
        Tuple of (was_modified, number_of_replacements)
    """
    try:
        # Read the file
        with open(file_path, 'r', encoding='utf-8') as f:
            original_content = f.read()

        # Convert delimiters
        converted_content, replacements = convert_math_delimiters(original_content)

        # Check if anything changed
        if replacements == 0:
            return False, 0

        # Write back if not dry run
        if not dry_run:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(converted_content)
            print_success(f"Fixed {replacements} math delimiter(s) in: {file_path.name}")
        else:
            print_info(f"Would fix {replacements} math delimiter(s) in: {file_path.name}")

        return True, replacements

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
        description="Convert LaTeX math delimiters to markdown format in .md files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process all files in refined_docs and flashcards
  python fix_math_delimiters.py

  # Dry run to preview changes
  python fix_math_delimiters.py --dry-run

  # Process a specific directory
  python fix_math_delimiters.py --dir custom_folder

  # Process multiple directories
  python fix_math_delimiters.py --dir folder1 --dir folder2

Conversions:
  Block math:  \\[ equation \\]  →  $$ equation $$
  Inline math: \\( equation \\)  →  $equation$ (spaces stripped)
  Fix spacing: $ equation $      →  $equation$ (spaces stripped)
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
    print_header("LaTeX Math Delimiter Converter")

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
    total_replacements = 0

    for file_path in markdown_files:
        was_modified, replacements = process_file(file_path, args.dry_run)

        if was_modified:
            total_files_modified += 1
            total_replacements += replacements

    # Print summary
    print()
    print_header("Summary")

    print(f"Total markdown files processed: {len(markdown_files)}")
    print(f"Files with math delimiters: {total_files_modified}")
    print(f"Total replacements: {total_replacements}")

    if args.dry_run:
        print()
        print_info("This was a dry run. Use without --dry-run to apply changes.")
    elif total_files_modified > 0:
        print()
        print_success("Math delimiters successfully converted!")
    else:
        print()
        print_info("No math delimiters found to convert.")

if __name__ == "__main__":
    main()
