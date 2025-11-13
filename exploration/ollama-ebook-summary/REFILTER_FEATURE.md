# REFILTER Status - Re-evaluate Flashcards with New Relevancy Target

## Overview

The `REFILTER` status allows you to re-evaluate existing flashcards with a new relevancy target without regenerating them from the original PDF/EPUB. This is useful when you want to:

- Apply a different focus area to existing flashcards
- Re-filter with a different quality threshold
- Update book notes after changing the relevancy target

## How It Works

When you set a book's status to `REFILTER` in `orchestration.csv`, the system:

1. **Reads existing flashcards** from `flashcards/{book}/`
2. **Parses individual flashcards** from the markdown files
3. **Re-evaluates each flashcard** using the LLM with the new `relevancy_target`
4. **Filters by quality** based on the rating threshold (default: 8/10)
5. **Writes high-quality flashcards** to `refined_docs/book_notes/{book}/`

## Usage

### Step 1: Update Relevancy Target

Edit `orchestration.csv` and:
1. Change the `relevancy_target` field to your new focus area
2. Set `status` to `REFILTER`

```csv
book_filename,pipeline_type,status,output_folder,relevancy_target
Game Engine Architecture.pdf,ONLYFLASHCARDS,REFILTER,flashcards,game loop architecture and memory management
```

### Step 2: Run Orchestrator

```bash
python orchestrate.py
```

The orchestrator will:
- Detect the `REFILTER` status
- Call `refilter_flashcards.py` with the new relevancy target
- Re-evaluate all flashcards and update `refined_docs/book_notes/`

### Manual Re-filtering

You can also manually re-filter a specific book:

```bash
# Re-filter with new relevancy target
python refilter_flashcards.py "My Book.pdf" --relevancy-target "specific techniques"

# Re-filter with different threshold
python refilter_flashcards.py "My Book.pdf" --threshold 9

# Re-filter with both
python refilter_flashcards.py "My Book.pdf" \
  --relevancy-target "advanced concepts" \
  --threshold 7
```

## Example Workflow

### Scenario: Change Focus Area

**Initial Setup:**
```csv
book_filename,pipeline_type,status,relevancy_target
Operating Systems.pdf,ONLYFLASHCARDS,COMPLETED,operating system fundamentals
```

You have 1000 flashcards, 300 in book_notes focused on "fundamentals"

**Update Focus:**
```csv
book_filename,pipeline_type,status,relevancy_target
Operating Systems.pdf,ONLYFLASHCARDS,REFILTER,memory management and scheduling algorithms
```

**Run Re-filter:**
```bash
python orchestrate.py
```

**Result:**
```
[INFO] Re-rating flashcards...
[INFO] Total flashcards to re-evaluate: 1000

[INFO] Rating distribution:
  10/10:  45 ██████████████████████
   9/10:  89 ████████████████████████████████████████████
   8/10: 156 ████████████████████████████████████████████████████████████████████████████████
   7/10: 234 █████████████...
   ...

[INFO] High-quality flashcards (>= 8/10): 290

[SUCCESS] Re-filtering complete!
[INFO] Output: refined_docs/book_notes/Operating-Systems
[INFO] Created 97 refined note file(s)
```

Now you have 290 flashcards in book_notes focused on "memory management and scheduling algorithms"

## Command Options

### refilter_flashcards.py

```bash
python refilter_flashcards.py <book_filename> [OPTIONS]
```

**Required Arguments:**
- `book_filename` - Name of the book file (e.g., "My Book.pdf")

**Optional Arguments:**
- `--relevancy-target TEXT` - New relevancy target for filtering
- `--threshold INT` - Minimum rating for high-quality (default: 8)
- `--flashcards-dir DIR` - Input directory (default: flashcards)
- `--output-dir DIR` - Output directory (default: refined_docs/book_notes)
- `-v, --verbose` - Enable verbose output

## Technical Details

### Flashcard Parsing

The script parses flashcards from existing markdown files:

```python
# Finds flashcards separated by ---
parts = re.split(r'\n---\n', content)

# Extracts title (if present)
title_match = re.search(r'^####\s+(.+)$', part, re.MULTILINE)
```

### Re-Rating Process

Each flashcard is re-evaluated:

1. **Build rating prompt** with new relevancy target
2. **Call LLM** to rate usefulness (1-10 scale)
3. **Extract numeric rating** from response
4. **Filter** flashcards >= threshold

### Output Structure

High-quality flashcards are grouped into files:
- **~3 flashcards per file** (configurable)
- **Or max 50,000 chars** (whichever comes first)
- **Clean format** (no rating metadata)

Example output structure:
```
refined_docs/book_notes/
└── Operating-Systems/
    ├── ostp_hq_part01_Process_API.md
    ├── ostp_hq_part02_Scheduling.md
    ├── ostp_hq_part03_Memory_Virt.md
    └── ...
```

## Comparison: REFILTER vs Full Regeneration

| Feature | REFILTER | Full Regeneration |
|---------|----------|-------------------|
| **Speed** | Fast (~1-2 min/1000 cards) | Slow (~hours) |
| **Uses existing flashcards** | ✅ Yes | ❌ No |
| **Requires original PDF** | ❌ No | ✅ Yes |
| **Changes flashcard content** | ❌ No | ✅ Yes (may differ) |
| **Updates filtering only** | ✅ Yes | N/A |
| **Use case** | Change focus area | Regenerate content |

## Use Cases

### 1. Refine Focus Area

**Scenario:** You initially generated flashcards with a broad focus but now want to narrow down to specific topics.

```csv
# Before
relevancy_target,programming fundamentals

# After
relevancy_target,concurrent programming patterns and synchronization
status,REFILTER
```

### 2. Increase Quality Threshold

**Scenario:** You want only the highest-quality flashcards (9/10 or higher).

```bash
python refilter_flashcards.py "My Book.pdf" --threshold 9
```

### 3. Different Relevancy for Different Needs

**Scenario:** You're studying for an exam and want to focus on specific topics.

```csv
# For data structures exam
relevancy_target,data structure implementation and complexity analysis
status,REFILTER
```

### 4. Batch Re-filtering

Set multiple books to REFILTER in orchestration.csv:

```csv
book_filename,pipeline_type,status,relevancy_target
Book1.pdf,ONLYFLASHCARDS,REFILTER,advanced algorithms
Book2.pdf,ONLYFLASHCARDS,REFILTER,system design patterns
Book3.pdf,ONLYFLASHCARDS,REFILTER,distributed systems
```

Run orchestrator to re-filter all:
```bash
python orchestrate.py
```

## Status Flow

```
PENDING → PROCESSING → COMPLETED
                          ↓
                      (change relevancy_target)
                          ↓
                      REFILTER → PROCESSING → COMPLETED
```

## Architecture

```
orchestrate.py
    ↓ (detects REFILTER status)
    ↓
refilter_flashcards.py
    ↓ (reads flashcards/{book}/*.md)
    ↓
    ↓ (parses individual flashcards)
    ↓
    ↓ (re-rates with new relevancy_target)
    ↓
    ↓ (filters >= threshold)
    ↓
refined_docs/book_notes/{book}/*.md
```

## Configuration

### Environment Variables

```bash
# API configuration
OLLAMA_API_BASE=http://localhost:11434/api
OLLAMA_RATING_MODEL=qwen2.5:latest
```

### Config File (_config.yaml)

The rating prompt is loaded from `_config.yaml`:

```yaml
prompts:
  flashcard_usefulness_rating:
    prompt: |
      Evaluate this flashcard for the book "{book_title}"...
      [Your rating prompt here]
```

## Troubleshooting

### "No flashcards found to filter"

**Cause:** The flashcards directory doesn't exist or is empty

**Solution:**
```bash
# Check if flashcards exist
ls flashcards/Book-Name/

# If not, generate flashcards first
python orchestrate.py  # (with status=PENDING)
```

### "Re-filtering without relevancy target"

**Cause:** The `relevancy_target` field is empty in orchestration.csv

**Solution:**
- Add a relevancy target to the CSV
- Or provide it via command line: `--relevancy-target "your focus"`

### Rating distribution shows all low scores

**Cause:** The relevancy target may be too specific or unrelated to flashcard content

**Solution:**
- Review your relevancy target - make it broader
- Check if flashcards actually cover that topic
- Try a different threshold (e.g., --threshold 6)

### Re-filtering takes too long

**Cause:** Large number of flashcards (1000+) with API delays

**Solution:**
- The script includes 0.1s delay between ratings to avoid overwhelming API
- For very large books, run overnight or during breaks
- Consider filtering on a subset first

## Performance

Typical re-filtering performance:

| Flashcards | Time | API Calls | Output Files |
|------------|------|-----------|--------------|
| 100 | ~30 sec | 100 | ~10 |
| 500 | ~2 min | 500 | ~50 |
| 1000 | ~5 min | 1000 | ~100 |
| 5000 | ~25 min | 5000 | ~500 |

*Based on qwen2.5:latest with 0.1s delay between calls*

## Best Practices

1. **Test on one book first** - Verify the new relevancy target produces desired results
2. **Review rating distribution** - Check if the threshold is appropriate
3. **Keep original flashcards** - Re-filtering doesn't modify `flashcards/`, only `refined_docs/book_notes/`
4. **Backup before batch re-filtering** - In case you want to revert
5. **Use specific relevancy targets** - "memory management patterns" vs "operating systems"

## Examples

### Example 1: Change from General to Specific

```bash
# Original: "Python programming"
# New: "Python decorators and metaprogramming"

python refilter_flashcards.py "Fluent Python.pdf" \
  --relevancy-target "Python decorators and metaprogramming"
```

### Example 2: Increase Quality Bar

```bash
# Only keep exceptional flashcards (9+)
python refilter_flashcards.py "My Book.pdf" --threshold 9
```

### Example 3: Different Focus for Exam Prep

```bash
# Original: "C++ programming techniques"
# Exam focus: "C++ templates and STL implementation"

python refilter_flashcards.py "C++ Software Design.pdf" \
  --relevancy-target "C++ templates and STL implementation" \
  --threshold 8
```

---

## Summary

**To re-filter existing flashcards:**

1. Edit `orchestration.csv`:
   - Update `relevancy_target` field
   - Set `status` to `REFILTER`

2. Run: `python orchestrate.py`

3. System automatically:
   - Reads existing flashcards
   - Re-evaluates with new relevancy target
   - Filters and writes to book_notes

**Key benefits:**
- ✅ Fast (minutes vs hours)
- ✅ No PDF required
- ✅ Change focus without regenerating
- ✅ Experiment with different thresholds
- ✅ Original flashcards preserved
