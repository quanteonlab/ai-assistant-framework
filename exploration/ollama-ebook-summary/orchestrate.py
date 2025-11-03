#!/usr/bin/env python3
"""
Book Processing Orchestrator
=============================

This script orchestrates the processing of multiple books from an orchestration CSV.
It tracks status, supports resume capability, and processes books sequentially.

Orchestration CSV Format:
    book_filename,pipeline_type,status,started_at,completed_at,output_folder,error_message,relevancy_target

Pipeline Types:
    - ONLYFLASHCARDS: Direct PDF/EPUB to flashcards
    - SUMMARIZEDFLASHCARDS: Book summary + flashcard generation (future)

Status Values:
    - PENDING or empty: Ready to process
    - PROCESSING: Currently being processed
    - COMPLETED: Successfully processed
    - TIMEOUTCOMPLETED: Timed out but may have partial results (will skip on re-run)
    - ERROR: Failed with error (will skip on re-run)

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

    def __init__(self, orchestration_csv: str, input_folder: str = "in", timeout_hours: float = 4.0):
        """
        Initialize the orchestrator.

        Args:
            orchestration_csv: Path to orchestration CSV file
            input_folder: Folder containing input PDF/EPUB files
            timeout_hours: Timeout in hours for each book processing (default: 2.0 hours)
        """
        self.orchestration_csv = Path(orchestration_csv)
        self.input_folder = Path(input_folder)
        self.script_dir = Path(__file__).parent.absolute()
        self.timeout_seconds = int(timeout_hours * 60 * 60)
        self.fieldnames = [
            'book_filename',
            'pipeline_type',
            'status',
            'started_at',
            'completed_at',
            'output_folder',
            'error_message',
            'relevancy_target'
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

        Skips: COMPLETED, TIMEOUTCOMPLETED, ERROR, PROCESSING

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

    def process_book(self, book_filename: str, pipeline_type: str, output_folder: str, relevancy_target: str = None) -> str:
        """
        Process a single book using the specified pipeline.

        Args:
            book_filename: Name of the PDF/EPUB file
            pipeline_type: Type of pipeline (ONLYFLASHCARDS, SUMMARIZEDFLASHCARDS)
            output_folder: Output directory for results
            relevancy_target: Target focus for relevancy filtering (e.g., "programming techniques")

        Returns:
            Status string: 'COMPLETED', 'TIMEOUTCOMPLETED', or 'ERROR'
        """
        # Find the book file
        book_path = self.input_folder / book_filename

        if not book_path.exists():
            print_status(f"Book file not found: {book_path}", "ERROR")
            return 'ERROR'

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
            # Add relevancy target if provided
            if relevancy_target:
                cmd.extend(["--relevancy-target", relevancy_target])
        elif pipeline_type == 'SUMMARIZEDFLASHCARDS':
            # Future: Add summary pipeline
            print_status("SUMMARIZEDFLASHCARDS pipeline not yet implemented", "ERROR")
            return 'ERROR'
        else:
            print_status(f"Unknown pipeline type: {pipeline_type}", "ERROR")
            return 'ERROR'

        # Execute the pipeline with real-time output
        timeout_seconds = self.timeout_seconds
        timeout_minutes = timeout_seconds // 60

        try:
            print_status(f"Running command: {' '.join(cmd)}", "INFO")
            print_status(f"Timeout: {timeout_minutes} minutes ({timeout_seconds // 3600} hours)", "INFO")
            print(f"{Colors.OKCYAN}{'='*70}")
            print(f"  REAL-TIME OUTPUT")
            print(f"{'='*70}{Colors.ENDC}\n")

            # Use Popen for real-time output streaming
            process = subprocess.Popen(
                cmd,
                cwd=self.script_dir,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,  # Merge stderr into stdout
                text=True,
                bufsize=1,  # Line buffered
                universal_newlines=True
            )

            # Stream output in real-time
            start_time = time.time()
            output_lines = []

            try:
                while True:
                    # Check for timeout
                    elapsed = time.time() - start_time
                    if elapsed > timeout_seconds:
                        process.kill()
                        process.wait()
                        print_status(f"Pipeline timed out after {timeout_minutes} minutes", "WARNING")
                        print_status("Partial results may have been generated", "INFO")
                        return 'TIMEOUTCOMPLETED'

                    # Read line with timeout
                    line = process.stdout.readline()

                    if not line:
                        # Process has finished
                        break

                    # Print line in real-time
                    print(line, end='', flush=True)
                    output_lines.append(line)

                # Wait for process to complete
                return_code = process.wait()

            except Exception as e:
                process.kill()
                process.wait()
                raise e

            print(f"\n{Colors.OKCYAN}{'='*70}")
            print(f"  END OF OUTPUT")
            print(f"{'='*70}{Colors.ENDC}\n")

            if return_code != 0:
                print_status(f"Pipeline failed with return code {return_code}", "ERROR")
                return 'ERROR'

            print_status("Pipeline completed successfully", "SUCCESS")
            return 'COMPLETED'

        except subprocess.TimeoutExpired:
            print_status(f"Pipeline timed out after {timeout_minutes} minutes", "WARNING")
            print_status("Partial results may have been generated", "INFO")
            return 'TIMEOUTCOMPLETED'
        except Exception as e:
            print_status(f"Error running pipeline: {e}", "ERROR")
            return 'ERROR'

    def run(self, continuous: bool = True):
        """
        Run the orchestrator to process all pending books.

        Args:
            continuous: If True, process all pending books. If False, process only one.
        """
        print_header("Book Processing Orchestrator")

        # Show initial status summary
        rows = self.read_orchestration()
        total_count = len(rows)

        status_counts = {
            'PENDING': 0,
            'COMPLETED': 0,
            'TIMEOUTCOMPLETED': 0,
            'ERROR': 0,
            'PROCESSING': 0
        }

        for row in rows:
            status = row.get('status', '').strip().upper()
            if not status:
                status = 'PENDING'
            if status in status_counts:
                status_counts[status] += 1
            else:
                status_counts[status] = 1

        print(f"Total books: {total_count}")
        print(f"  PENDING: {status_counts.get('PENDING', 0)}")
        print(f"  COMPLETED: {status_counts.get('COMPLETED', 0)} (will skip)")
        print(f"  TIMEOUTCOMPLETED: {status_counts.get('TIMEOUTCOMPLETED', 0)} (will skip - partial results may exist)")
        print(f"  ERROR: {status_counts.get('ERROR', 0)} (will skip)")
        print(f"  PROCESSING: {status_counts.get('PROCESSING', 0)} (use --reset-processing if stuck)")
        print()

        processed_count = 0

        while True:
            # Read current state
            rows = self.read_orchestration()

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
            relevancy_target = row.get('relevancy_target', '')

            print_header(f"Processing Book {processed_count + 1}/{total_count}")

            # Mark as PROCESSING
            started_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            self.update_row_status(rows, next_idx, 'PROCESSING', started_at=started_at)

            # Process the book
            status = self.process_book(book_filename, pipeline_type, output_folder, relevancy_target)

            # Re-read rows (they may have changed during processing)
            rows = self.read_orchestration()

            # Update status
            completed_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            if status == 'COMPLETED':
                self.update_row_status(rows, next_idx, 'COMPLETED', completed_at=completed_at)
                print_status(f"Successfully completed: {book_filename}", "SUCCESS")
            elif status == 'TIMEOUTCOMPLETED':
                self.update_row_status(
                    rows, next_idx, 'TIMEOUTCOMPLETED',
                    completed_at=completed_at,
                    error_message="Pipeline timed out - partial results may exist"
                )
                print_status(f"Timed out (partial results): {book_filename}", "WARNING")
            else:  # ERROR
                self.update_row_status(
                    rows, next_idx, 'ERROR',
                    completed_at=completed_at,
                    error_message="Pipeline execution failed"
                )
                print_status(f"Failed to process: {book_filename}", "ERROR")

            processed_count += 1

            # Always continue to next book in continuous mode
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

  # Set custom timeout (3 hours instead of default 2 hours)
  python orchestrate.py --timeout 3.0

Orchestration CSV Format:
  book_filename: Name of PDF/EPUB file in the 'in/' folder
  pipeline_type: ONLYFLASHCARDS or SUMMARIZEDFLASHCARDS
  status: PENDING, PROCESSING, COMPLETED, TIMEOUTCOMPLETED, or ERROR
  started_at: Timestamp when processing started
  completed_at: Timestamp when processing finished
  output_folder: Where to save output (default: flashcards)
  error_message: Error details if failed or timed out
  relevancy_target: Target focus for relevancy evaluation (e.g., "programming techniques")
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

    parser.add_argument(
        '--timeout',
        type=float,
        default=4.0,
        help='Timeout in hours for each book processing (default: 4.0 hours)'
    )

    args = parser.parse_args()

    # Create orchestrator
    orchestrator = BookOrchestrator(args.orchestration, args.input_folder, args.timeout)

    # Reset processing if requested
    if args.reset_processing:
        orchestrator.reset_processing_rows()
        return

    # Run orchestration
    continuous = not args.single
    orchestrator.run(continuous=continuous)


if __name__ == "__main__":
    main()
