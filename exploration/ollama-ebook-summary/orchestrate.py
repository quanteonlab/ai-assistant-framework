#!/usr/bin/env python3
"""
Book Processing Orchestrator
=============================

This script orchestrates the processing of multiple books from an orchestration CSV.
It tracks status, supports resume capability, and processes books sequentially.

Orchestration CSV Format:
    book_filename,pipeline_type,status,started_at,completed_at,output_folder,error_message

Pipeline Types:
    - ONLYFLASHCARDS: Direct PDF/EPUB to flashcards
    - SUMMARIZEDFLASHCARDS: Book summary + flashcard generation (future)

Status Values:
    - PENDING or empty: Ready to process
    - PROCESSING: Currently being processed
    - COMPLETED: Successfully processed
    - ERROR: Failed with error

Usage:
    python orchestrate.py
    python orchestrate.py --orchestration custom_orchestration.csv
    python orchestrate.py --continue  # Resume from where it left off
    python orchestrate.py --reset-processing  # Reset stuck PROCESSING rows to PENDING

Author: AI Assistant Framework
License: Same as parent project
"""

import os
import sys
import csv
import time
import argparse
import subprocess
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, List

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


class Colors:
    """Terminal color codes."""
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


def print_status(text: str, status: str = "INFO"):
    """Print a status message with color."""
    color = Colors.OKBLUE
    if status == "SUCCESS":
        color = Colors.OKGREEN
    elif status == "ERROR":
        color = Colors.FAIL
    elif status == "WARNING":
        color = Colors.WARNING

    print(f"{color}[{status}]{Colors.ENDC} {text}")


class BookOrchestrator:
    """Orchestrates processing of multiple books from CSV."""

    def __init__(self, orchestration_csv: str, input_folder: str = "in"):
        """
        Initialize the orchestrator.

        Args:
            orchestration_csv: Path to orchestration CSV file
            input_folder: Folder containing input PDF/EPUB files
        """
        self.orchestration_csv = Path(orchestration_csv)
        self.input_folder = Path(input_folder)
        self.script_dir = Path(__file__).parent.absolute()
        self.fieldnames = [
            'book_filename',
            'pipeline_type',
            'status',
            'started_at',
            'completed_at',
            'output_folder',
            'error_message'
        ]

        # Ensure input folder exists
        self.input_folder.mkdir(exist_ok=True)

        # Ensure orchestration CSV exists
        if not self.orchestration_csv.exists():
            self._create_empty_csv()

    def _create_empty_csv(self):
        """Create an empty orchestration CSV with headers."""
        with open(self.orchestration_csv, 'w', encoding='utf-8', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=self.fieldnames)
            writer.writeheader()
        print_status(f"Created orchestration CSV: {self.orchestration_csv}", "SUCCESS")

    def read_orchestration(self) -> List[Dict]:
        """Read the orchestration CSV and return all rows."""
        rows = []
        with open(self.orchestration_csv, 'r', encoding='utf-8', newline='') as f:
            reader = csv.DictReader(f)
            for row in reader:
                rows.append(row)
        return rows

    def write_orchestration(self, rows: List[Dict]):
        """Write all rows back to the orchestration CSV."""
        with open(self.orchestration_csv, 'w', encoding='utf-8', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=self.fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    def update_row_status(self, rows: List[Dict], index: int, status: str,
                          started_at: str = None, completed_at: str = None,
                          error_message: str = None):
        """
        Update the status of a specific row.

        Args:
            rows: All rows from CSV
            index: Index of row to update
            status: New status (PENDING, PROCESSING, COMPLETED, ERROR)
            started_at: Timestamp when processing started
            completed_at: Timestamp when processing completed
            error_message: Error message if failed
        """
        rows[index]['status'] = status

        if started_at:
            rows[index]['started_at'] = started_at

        if completed_at:
            rows[index]['completed_at'] = completed_at

        if error_message:
            rows[index]['error_message'] = error_message

        # Write immediately to persist status
        self.write_orchestration(rows)

    def find_next_pending_row(self, rows: List[Dict]) -> Optional[int]:
        """
        Find the next row that needs processing.

        Returns:
            Index of next pending row, or None if all done
        """
        for idx, row in enumerate(rows):
            status = row.get('status', '').strip().upper()
            if not status or status == 'PENDING':
                return idx
        return None

    def reset_processing_rows(self):
        """Reset any rows stuck in PROCESSING status back to PENDING."""
        rows = self.read_orchestration()
        reset_count = 0

        for row in rows:
            if row.get('status', '').strip().upper() == 'PROCESSING':
                row['status'] = 'PENDING'
                row['error_message'] = 'Reset from stuck PROCESSING state'
                reset_count += 1

        if reset_count > 0:
            self.write_orchestration(rows)
            print_status(f"Reset {reset_count} rows from PROCESSING to PENDING", "SUCCESS")
        else:
            print_status("No rows in PROCESSING state found", "INFO")

    def process_book(self, book_filename: str, pipeline_type: str, output_folder: str) -> bool:
        """
        Process a single book using the specified pipeline.

        Args:
            book_filename: Name of the PDF/EPUB file
            pipeline_type: Type of pipeline (ONLYFLASHCARDS, SUMMARIZEDFLASHCARDS)
            output_folder: Output directory for results

        Returns:
            True if successful, False otherwise
        """
        # Find the book file
        book_path = self.input_folder / book_filename

        if not book_path.exists():
            print_status(f"Book file not found: {book_path}", "ERROR")
            return False

        print_status(f"Processing: {book_filename}", "INFO")
        print_status(f"Pipeline: {pipeline_type}", "INFO")
        print_status(f"Output: {output_folder}", "INFO")

        # Build command based on pipeline type
        if pipeline_type == 'ONLYFLASHCARDS':
            cmd = [
                sys.executable,
                str(self.script_dir / "pdf2flashcards.py"),
                str(book_path),
                "-o", output_folder
            ]
        elif pipeline_type == 'SUMMARIZEDFLASHCARDS':
            # Future: Add summary pipeline
            print_status("SUMMARIZEDFLASHCARDS pipeline not yet implemented", "ERROR")
            return False
        else:
            print_status(f"Unknown pipeline type: {pipeline_type}", "ERROR")
            return False

        # Execute the pipeline
        try:
            print_status(f"Running command: {' '.join(cmd)}", "INFO")

            result = subprocess.run(
                cmd,
                cwd=self.script_dir,
                capture_output=True,
                text=True,
                timeout=7200  # 2 hour timeout
            )

            if result.returncode != 0:
                print_status(f"Pipeline failed with return code {result.returncode}", "ERROR")
                print("STDOUT:", result.stdout[-1000:] if len(result.stdout) > 1000 else result.stdout)
                print("STDERR:", result.stderr[-1000:] if len(result.stderr) > 1000 else result.stderr)
                return False

            print_status("Pipeline completed successfully", "SUCCESS")
            print(result.stdout[-500:] if len(result.stdout) > 500 else result.stdout)
            return True

        except subprocess.TimeoutExpired:
            print_status("Pipeline timed out after 2 hours", "ERROR")
            return False
        except Exception as e:
            print_status(f"Error running pipeline: {e}", "ERROR")
            return False

    def run(self, continuous: bool = True):
        """
        Run the orchestrator to process all pending books.

        Args:
            continuous: If True, process all pending books. If False, process only one.
        """
        print_header("Book Processing Orchestrator")

        processed_count = 0
        total_count = 0

        while True:
            # Read current state
            rows = self.read_orchestration()
            total_count = len(rows)

            # Find next pending row
            next_idx = self.find_next_pending_row(rows)

            if next_idx is None:
                print_status("No pending books to process", "SUCCESS")
                break

            # Get book info
            row = rows[next_idx]
            book_filename = row['book_filename']
            pipeline_type = row.get('pipeline_type', 'ONLYFLASHCARDS').upper()
            output_folder = row.get('output_folder', 'flashcards')

            print_header(f"Processing Book {processed_count + 1}/{total_count}")

            # Mark as PROCESSING
            started_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            self.update_row_status(rows, next_idx, 'PROCESSING', started_at=started_at)

            # Process the book
            success = self.process_book(book_filename, pipeline_type, output_folder)

            # Re-read rows (they may have changed during processing)
            rows = self.read_orchestration()

            # Update status
            completed_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            if success:
                self.update_row_status(rows, next_idx, 'COMPLETED', completed_at=completed_at)
                print_status(f"Successfully completed: {book_filename}", "SUCCESS")
            else:
                self.update_row_status(
                    rows, next_idx, 'ERROR',
                    completed_at=completed_at,
                    error_message="Pipeline execution failed"
                )
                print_status(f"Failed to process: {book_filename}", "ERROR")

            processed_count += 1

            # If not continuous mode, stop after one book
            if not continuous:
                break

        print_header("Orchestration Complete")
        print(f"Total books in CSV: {total_count}")
        print(f"Books processed this run: {processed_count}")


def main():
    parser = argparse.ArgumentParser(
        description="Orchestrate processing of multiple books from CSV",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process all pending books
  python orchestrate.py

  # Use custom orchestration CSV
  python orchestrate.py --orchestration my_books.csv

  # Process only one book then stop
  python orchestrate.py --single

  # Reset stuck PROCESSING rows
  python orchestrate.py --reset-processing

  # Continue from where it left off (default behavior)
  python orchestrate.py --continue

Orchestration CSV Format:
  book_filename: Name of PDF/EPUB file in the 'in/' folder
  pipeline_type: ONLYFLASHCARDS or SUMMARIZEDFLASHCARDS
  status: PENDING, PROCESSING, COMPLETED, or ERROR
  started_at: Timestamp when processing started
  completed_at: Timestamp when processing finished
  output_folder: Where to save output (default: flashcards)
  error_message: Error details if failed
        """
    )

    parser.add_argument(
        '--orchestration',
        default='orchestration.csv',
        help='Path to orchestration CSV file (default: orchestration.csv)'
    )

    parser.add_argument(
        '--input-folder',
        default='in',
        help='Folder containing input PDF/EPUB files (default: in/)'
    )

    parser.add_argument(
        '--single',
        action='store_true',
        help='Process only one book then stop'
    )

    parser.add_argument(
        '--continue',
        action='store_true',
        dest='continue_processing',
        help='Continue processing from last pending book (default behavior)'
    )

    parser.add_argument(
        '--reset-processing',
        action='store_true',
        help='Reset stuck PROCESSING rows to PENDING'
    )

    args = parser.parse_args()

    # Create orchestrator
    orchestrator = BookOrchestrator(args.orchestration, args.input_folder)

    # Reset processing if requested
    if args.reset_processing:
        orchestrator.reset_processing_rows()
        return

    # Run orchestration
    continuous = not args.single
    orchestrator.run(continuous=continuous)


if __name__ == "__main__":
    main()
