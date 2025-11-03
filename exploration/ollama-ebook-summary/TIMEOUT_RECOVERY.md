# Timeout Recovery Feature

## Overview

The system now supports resuming flashcard generation from where it left off after a timeout. This prevents you from having to restart large books from the beginning.

## How It Works

When a book times out during flashcard generation, you can now:

1. Set the status to `COMPLETE_TIMEOUT` in `orchestration.csv`
2. Run the orchestrator again - it will automatically resume from the last generated flashcard
3. The system finds the last flashcard part number and continues from there

## Status Values

### TIMEOUTCOMPLETED
- **Behavior**: Skip on next run (partial results exist but won't continue)
- **Use when**: You want to keep partial results but not complete the book

### COMPLETE_TIMEOUT
- **Behavior**: Resume from last flashcard on next run
- **Use when**: You want to complete the book from where it stopped

## How to Resume

### Method 1: Using Orchestrator (Recommended)

1. **Edit `orchestration.csv`** - Change status from `TIMEOUTCOMPLETED` to `COMPLETE_TIMEOUT`:

```csv
book_filename,pipeline_type,status,...
Game Engine Architecture.pdf,ONLYFLASHCARDS,COMPLETE_TIMEOUT,...
```

2. **Run orchestrator**:

```bash
python orchestrate.py
```

The orchestrator will:
- Detect the `COMPLETE_TIMEOUT` status
- Find the last generated flashcard (e.g., `geaa_part27_...md`)
- Resume generation starting from part 28

### Method 2: Manual Resume

You can also manually resume a specific book:

```bash
python resume_flashcards.py "Game Engine Architecture.pdf"
```

With options:

```bash
# Custom output folder
python resume_flashcards.py "My Book.pdf" --output flashcards

# With relevancy target
python resume_flashcards.py "My Book.pdf" --relevancy-target "programming techniques"
```

## Example Workflow

### Scenario: Book Times Out

```
[2025-11-02 15:30] Processing: Game Engine Architecture.pdf
[2025-11-02 19:30] Pipeline timed out after 240 minutes
[INFO] Generated flashcards up to part 27
[WARN] Status: TIMEOUTCOMPLETED
```

### Step 1: Check Progress

```bash
ls flashcards/Game-Engine-Architecture/
```

Output:
```
geaa_part01_Introduction.md
geaa_part02_Foundations.md
...
geaa_part27_Audio_Engine.md  ← Last generated
```

### Step 2: Set Resume Status

Edit `orchestration.csv`:

```csv
book_filename,pipeline_type,status,output_folder
Game Engine Architecture.pdf,ONLYFLASHCARDS,COMPLETE_TIMEOUT,flashcards
```

### Step 3: Resume

```bash
python orchestrate.py
```

Output:
```
[INFO] Book: Game Engine Architecture.pdf
[INFO] Mode: RESUME from timeout
[INFO] Last generated flashcard: part 27
[INFO] Resuming from part 28
[INFO] Skipping already processed parts 1 through 27
[INFO] Starting generation from part 28

Processing entry 450: Game Loop Architecture...
Processing entry 451: Memory Management...
...
```

## Technical Details

### Resume Process

1. **Find Last Flashcard**:
   - Scans `flashcards/{book_name}/` for `*_part{num}_*.md` files
   - Extracts part numbers using regex pattern: `_part(\d+)_`
   - Finds maximum part number (e.g., 27)

2. **Locate CSV**:
   - Finds processed CSV in `out/processed/{book_name}_processed.csv`
   - This contains all the extracted chapters

3. **Resume Generation**:
   - Calls `flashcard.py` with `--start-from-part 28`
   - Skips processing until reaching the start part
   - Continues generating flashcards from there

### File Naming

Flashcards follow this pattern:
```
{prefix}_part{num}_{chapter}.md
```

Examples:
- `geaa_part01_Introduction.md` (part 1)
- `geaa_part27_Audio_Engine.md` (part 27)
- `geaa_part28_Game_Loop.md` (part 28, resumed)

### Part Numbering

Parts are numbered sequentially based on:
- Chapters processed per file (default: 3)
- OR text size limit (default: 50,000 chars)

Whichever limit is hit first triggers a new part file.

## Scripts

### resume_flashcards.py

Standalone script to resume a single book:

```python
#!/usr/bin/env python3
# Resume flashcard generation from timeout

python resume_flashcards.py "Book Name.pdf"
```

**Features**:
- Automatically finds last flashcard
- Locates processed CSV
- Resumes from correct part number

### Updated flashcard.py

Added `--start-from-part` argument:

```bash
python flashcard.py -c book.csv --start-from-part 28
```

This skips processing until reaching part 28, then continues normally.

### Updated orchestrate.py

Now handles `COMPLETE_TIMEOUT` status:
- Detects status in orchestration CSV
- Calls `resume_flashcards.py` instead of `pdf2flashcards.py`
- Passes through relevancy target and output folder

## Comparison: TIMEOUTCOMPLETED vs COMPLETE_TIMEOUT

| Feature | TIMEOUTCOMPLETED | COMPLETE_TIMEOUT |
|---------|-----------------|------------------|
| **Keeps partial results** | ✅ Yes | ✅ Yes |
| **Skips on next run** | ✅ Yes | ❌ No |
| **Resumes automatically** | ❌ No | ✅ Yes |
| **Use case** | Accept partial results | Complete the book |

## Troubleshooting

### "Cannot resume - no existing flashcards found"

**Cause**: No flashcards were generated before timeout

**Solution**:
- Run the full pipeline instead
- Check if flashcards exist in `flashcards/{book_name}/`

### "Cannot resume - processed CSV not found"

**Cause**: The CSV file in `out/processed/` is missing

**Solution**:
- Verify CSV exists: `ls out/processed/`
- If missing, extract it again from the PDF/EPUB: `python book2text.py book.pdf`

### Resume starts from wrong part

**Cause**: Part numbering doesn't match expectations

**Solution**:
- Check last flashcard manually: `ls flashcards/{book_name}/ | tail -5`
- Note the highest part number
- Manually run: `python flashcard.py -c book.csv --start-from-part {num}`

### Duplicate flashcards after resume

**Cause**: Started from wrong part or files weren't properly detected

**Solution**:
- Delete duplicates manually
- Check file naming pattern matches `{prefix}_part{num}_{chapter}.md`

## Best Practices

### 1. Regular Status Checks

Monitor long-running books:
```bash
# Check progress
tail -f orchestration.csv
watch 'ls -l flashcards/Book-Name/'
```

### 2. Adjust Timeout

For very large books, increase timeout:
```bash
python orchestrate.py --timeout 8.0  # 8 hours instead of 4
```

### 3. Resume Promptly

Resume timed-out books soon after they stop:
- Keeps context fresh
- Reduces risk of file system changes

### 4. Backup Before Resume

```bash
# Backup existing flashcards
cp -r flashcards/Book-Name/ flashcards/Book-Name-backup/

# Then resume
python orchestrate.py
```

## Examples

### Example 1: Quick Resume

```bash
# 1. Book times out
python orchestrate.py
# [INFO] Game Engine Architecture.pdf: TIMEOUTCOMPLETED

# 2. Change status
sed -i 's/TIMEOUTCOMPLETED/COMPLETE_TIMEOUT/' orchestration.csv

# 3. Resume
python orchestrate.py
# [INFO] Resuming from part 27
# [SUCCESS] Completed: Game Engine Architecture.pdf
```

### Example 2: Manual Resume with Options

```bash
# Resume specific book with custom settings
python resume_flashcards.py \
  "Operating Systems.pdf" \
  --output flashcards \
  --relevancy-target "operating system concepts and low-level techniques"
```

### Example 3: Multiple Books

orchestration.csv:
```csv
book_filename,pipeline_type,status,output_folder
Book1.pdf,ONLYFLASHCARDS,COMPLETE_TIMEOUT,flashcards
Book2.pdf,ONLYFLASHCARDS,COMPLETE_TIMEOUT,flashcards
Book3.pdf,ONLYFLASHCARDS,PENDING,flashcards
```

Run orchestrator:
```bash
python orchestrate.py
```

Will process in order:
1. Resume Book1 from timeout
2. Resume Book2 from timeout
3. Process Book3 from start

## Architecture

```
orchestrate.py
    ↓ (detects COMPLETE_TIMEOUT)
    ↓
resume_flashcards.py
    ↓ (finds last flashcard: part 27)
    ↓
flashcard.py --start-from-part 28
    ↓ (skips parts 1-27, generates 28+)
    ↓
flashcards/{book}/*.md
```

## Configuration

No configuration needed! The feature works automatically with:
- Existing `orchestration.csv` format
- Existing folder structure
- Existing naming conventions

Just change status to `COMPLETE_TIMEOUT` and run.

---

## Summary

**To resume a timed-out book:**

1. Edit `orchestration.csv`: Change status to `COMPLETE_TIMEOUT`
2. Run: `python orchestrate.py`
3. System automatically resumes from last flashcard

**Status values:**
- `TIMEOUTCOMPLETED` = Skip on next run
- `COMPLETE_TIMEOUT` = Resume on next run

That's it! The system handles everything else automatically.
