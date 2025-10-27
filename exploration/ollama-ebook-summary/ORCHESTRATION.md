# Book Processing Orchestration

Automated batch processing of multiple PDF/EPUB books with status tracking and resume capability.

## Overview

The orchestration system allows you to:
- Process multiple books automatically
- Track processing status for each book
- Resume from where it left off if interrupted
- Support different processing pipelines
- Handle errors gracefully

## Quick Start

### 1. Setup

Create the input folder and orchestration CSV:
```bash
mkdir in/
```

The orchestration CSV will be created automatically on first run.

### 2. Add Books

Place your PDF/EPUB files in the `in/` folder:
```bash
cp mybook.pdf in/
```

Add them to the orchestration queue:
```bash
# Add a single book
python add_book.py mybook.pdf

# Add multiple books
python add_book.py book1.pdf book2.epub book3.pdf

# Add with specific pipeline
python add_book.py mybook.pdf --pipeline ONLYFLASHCARDS
```

### 3. Run Orchestration

Process all pending books:
```bash
python orchestrate.py
```

The script will:
1. Find the first book with status PENDING (or empty)
2. Mark it as PROCESSING
3. Run the specified pipeline
4. Update status to COMPLETED or ERROR
5. Move to the next PENDING book
6. Repeat until all books are processed

## Orchestration CSV Format

The `orchestration.csv` file tracks all books:

| Column | Description | Example |
|--------|-------------|---------|
| `book_filename` | Name of PDF/EPUB in `in/` folder | `mybook.pdf` |
| `pipeline_type` | Processing pipeline | `ONLYFLASHCARDS` |
| `status` | Current status | `PENDING`, `PROCESSING`, `COMPLETED`, `ERROR` |
| `started_at` | When processing started | `2025-01-26 10:30:00` |
| `completed_at` | When processing finished | `2025-01-26 11:45:00` |
| `output_folder` | Where output is saved | `flashcards` |
| `error_message` | Error details if failed | Error text |

### Example CSV

```csv
book_filename,pipeline_type,status,started_at,completed_at,output_folder,error_message
book1.pdf,ONLYFLASHCARDS,COMPLETED,2025-01-26 10:00:00,2025-01-26 10:45:00,flashcards,
book2.epub,ONLYFLASHCARDS,PENDING,,,flashcards,
book3.pdf,SUMMARIZEDFLASHCARDS,ERROR,2025-01-26 11:00:00,2025-01-26 11:05:00,flashcards,Pipeline not implemented
```

## Pipeline Types

### ONLYFLASHCARDS
Direct PDF/EPUB to flashcards conversion:
- Splits PDF/EPUB by chapters
- Generates flashcards for each chapter
- Rates flashcards for technical usefulness
- Creates high-quality consolidated markdown files

### SUMMARIZEDFLASHCARDS (Future)
Book summary + flashcard generation:
- Creates book summary first
- Generates flashcards from summary
- (Not yet implemented)

## Status Values

| Status | Meaning |
|--------|---------|
| `PENDING` or empty | Ready to process |
| `PROCESSING` | Currently being processed |
| `COMPLETED` | Successfully processed |
| `ERROR` | Failed with error |

## Commands

### Process All Books
```bash
python orchestrate.py
```

### Process Single Book
```bash
python orchestrate.py --single
```

### Resume After Interruption
```bash
python orchestrate.py --continue
```
(This is the default behavior)

### Reset Stuck Processing
If a book gets stuck in PROCESSING status:
```bash
python orchestrate.py --reset-processing
```

### Use Custom Orchestration CSV
```bash
python orchestrate.py --orchestration my_books.csv
```

### Use Custom Input Folder
```bash
python orchestrate.py --input-folder my_pdfs/
```

## Adding Books to Queue

### Method 1: Using add_book.py (Recommended)

```bash
# Add single book
python add_book.py mybook.pdf

# Add with pipeline type
python add_book.py mybook.pdf --pipeline ONLYFLASHCARDS

# Add with custom output folder
python add_book.py mybook.pdf --output flashcards/mybook

# Add multiple books
python add_book.py *.pdf
```

### Method 2: Manually Edit CSV

Edit `orchestration.csv` and add a row:
```csv
mybook.pdf,ONLYFLASHCARDS,PENDING,,,flashcards,
```

## Resume Capability

The orchestrator automatically resumes from where it left off:

1. **Graceful Shutdown**: Press Ctrl+C to stop
2. **Status Tracking**: Current book is marked as PROCESSING
3. **Resume**: Run `python orchestrate.py` again
4. **Reset if Stuck**: Use `--reset-processing` if needed

## Error Handling

If a book fails:
- Status is set to ERROR
- Error message is recorded in CSV
- Processing continues with next book
- You can review errors in the CSV
- Fix the issue and reset status to PENDING

## Directory Structure

```
ollama-ebook-summary/
├── in/                          # Input PDF/EPUB files
│   ├── book1.pdf
│   ├── book2.epub
│   └── book3.pdf
├── orchestration.csv            # Status tracking
├── orchestrate.py              # Main orchestration script
├── add_book.py                 # Helper to add books
├── pdf2flashcards.py           # Pipeline script
└── flashcards/                 # Output folder
    ├── book1_part01.md
    ├── book2_part01.md
    ├── high_quality/
    │   ├── book1_part01.md
    │   └── book2_part01.md
    └── flashcards_training_data.csv
```

## Example Workflow

```bash
# 1. Place books in input folder
cp ~/Downloads/*.pdf in/

# 2. Add them to orchestration
python add_book.py in/*.pdf

# 3. Check what will be processed
cat orchestration.csv

# 4. Run orchestration (processes all pending books)
python orchestrate.py

# 5. If interrupted, just run again to resume
python orchestrate.py

# 6. Check results
ls flashcards/high_quality/
```

## Monitoring Progress

While running, the orchestrator displays:
- Current book being processed
- Pipeline type
- Real-time output from pdf2flashcards.py
- Success/error messages
- Final summary

Example output:
```
======================================================================
  Book Processing Orchestrator
======================================================================

[INFO] Processing: book1.pdf
[INFO] Pipeline: ONLYFLASHCARDS
[INFO] Output: flashcards

[Step 1] Converting PDF/EPUB to chunked CSV...
[Step 2] Generating flashcards from CSV...
[Step 3] Rating unrated flashcards...

[SUCCESS] Successfully completed: book1.pdf

======================================================================
  Processing Book 2/5
======================================================================
...
```

## Best Practices

1. **Test First**: Process one book with `--single` before batch processing
2. **Monitor Logs**: Keep an eye on output for errors
3. **Incremental Adds**: Add books in batches to monitor progress
4. **Backup CSV**: Keep a backup of orchestration.csv
5. **Check Disk Space**: Ensure enough space for output files
6. **Use Resume**: Don't worry about interruptions - just restart

## Troubleshooting

### Book Stuck in PROCESSING
```bash
python orchestrate.py --reset-processing
```

### Want to Reprocess a Book
Edit `orchestration.csv` and change status from COMPLETED to PENDING

### Check What's Pending
```bash
grep "PENDING" orchestration.csv
```

### Clear All Status (Start Fresh)
```bash
# Backup first!
cp orchestration.csv orchestration.csv.backup

# Reset all to PENDING
sed -i 's/,COMPLETED,/,PENDING,/g' orchestration.csv
sed -i 's/,ERROR,/,PENDING,/g' orchestration.csv
```

## Advanced Usage

### Custom Orchestration for Different Projects

```bash
# Project A
python orchestrate.py --orchestration projectA.csv --input-folder projectA/books/

# Project B
python orchestrate.py --orchestration projectB.csv --input-folder projectB/books/
```

### Filter by Status

```bash
# Count pending books
grep -c "PENDING" orchestration.csv

# List completed books
grep "COMPLETED" orchestration.csv | cut -d',' -f1

# List failed books
grep "ERROR" orchestration.csv
```

## Future Enhancements

- [ ] Parallel processing of multiple books
- [ ] Email notifications on completion
- [ ] Web dashboard for monitoring
- [ ] SUMMARIZEDFLASHCARDS pipeline implementation
- [ ] Priority queue for books
- [ ] Retry failed books automatically
