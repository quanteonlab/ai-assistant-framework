# Auto-Relevancy Target Generation

## Overview

The `add_book.py` script now automatically generates the `relevancy_target` field when adding books to `orchestration.csv` using your local LLM (Ollama).

## How It Works

When you add a book, the script:

1. **Extracts the book title** from the filename
   - Removes common prefixes (e.g., "2A001 -", "10A005 -")
   - Replaces underscores with spaces
   - Example: `2A001 - Build a Large Language Model.pdf` → `Build a Large Language Model`

2. **Calls your local LLM** (Ollama) with a specialized prompt
   - Asks for a brief 5-10 word technical description
   - Focuses on key concepts, techniques, and domain knowledge

3. **Cleans the response**
   - Removes all trailing punctuation (periods, commas, semicolons, etc.)
   - Truncates if too long (max 100 chars)
   - Example output: `LLM architecture and implementation techniques`

4. **Saves to orchestration.csv**
   - Populates the `relevancy_target` field automatically
   - Also works when updating existing books with `--force`

## Usage

### Basic Usage (Auto-generates relevancy)

```bash
# Add a single book
python add_book.py "Machine Learning Patterns.pdf"
```

**Output:**
```
[INFO] Generating relevancy target for: Machine Learning Patterns.pdf
  Relevancy: ML design patterns and production deployment techniques
[ADD] Added: Machine Learning Patterns.pdf
  Pipeline: ONLYFLASHCARDS
  Output: flashcards
```

### Scan Mode

```bash
# Scan folder and auto-generate for all new books
python add_book.py --scan
```

**Output:**
```
[INFO] Scanning in/ for PDF/EPUB files...
[INFO] Found 5 books in in/
[INFO] 2 books already in orchestration
[INFO] Generating relevancy target for: Deep Learning Book.pdf
  Relevancy: Deep learning architectures and training techniques
[ADD] Added: Deep Learning Book.pdf
...
```

### Disable Auto-Generation

```bash
# Add without generating relevancy (leaves field empty)
python add_book.py mybook.pdf --no-auto-relevancy
```

### Update Existing Books

```bash
# Fill in missing relevancy targets for existing books
python add_book.py "Some Book.pdf" --force
```

If the book exists and has no relevancy target, it will be generated automatically.

## Configuration

The script uses environment variables or sensible defaults:

- **API Base URL**: `OLLAMA_API_BASE` (default: `http://localhost:11434/api`)
- **Model**: `OLLAMA_MODEL` (default: `qwen2.5:latest`)

Set in your `.env` file:
```bash
OLLAMA_API_BASE=http://localhost:11434/api
OLLAMA_MODEL=qwen2.5:latest
```

## Example Relevancy Targets

Here are examples of generated relevancy targets:

| Book Title | Generated Relevancy Target |
|------------|---------------------------|
| Building Microservices | Microservices architecture patterns and practices |
| Designing Data-Intensive Applications | Data system design patterns and distributed architecture concepts |
| Python for Data Analysis | Python data manipulation and analysis techniques |
| Kubernetes Up and Running | Container orchestration and Kubernetes deployment patterns |
| Financial Data Engineering | Financial data modeling and quantitative techniques |
| Reinforcement Learning | RL algorithms and implementation techniques |

## Error Handling

The script gracefully handles errors:

- **LLM API unavailable**: Prints warning, continues with empty relevancy
- **Timeout**: 30-second timeout, continues with empty relevancy
- **Invalid response**: Continues with empty relevancy

The book is still added successfully even if relevancy generation fails.

## Command Reference

### Arguments

```
python add_book.py [OPTIONS] [books...]
```

**Options:**
- `--scan`: Scan input folder and add all PDF/EPUB files
- `--input-folder DIR`: Folder to scan (default: `in/`)
- `--pipeline TYPE`: Pipeline type (default: `ONLYFLASHCARDS`)
- `--output DIR`: Output folder (default: `flashcards`)
- `--orchestration FILE`: Orchestration CSV (default: `orchestration.csv`)
- `--force`: Force update existing books
- `--no-auto-relevancy`: Disable automatic relevancy generation

### Examples

```bash
# Add single book with auto-relevancy
python add_book.py mybook.pdf

# Add multiple books
python add_book.py book1.pdf book2.pdf book3.pdf

# Scan folder
python add_book.py --scan

# Scan custom folder
python add_book.py --scan --input-folder my_books/

# Add without auto-relevancy
python add_book.py mybook.pdf --no-auto-relevancy

# Update existing book and fill missing relevancy
python add_book.py existing_book.pdf --force
```

## Benefits

1. **Automated Workflow**: No manual editing of CSV needed
2. **Consistent Format**: LLM ensures consistent style and length
3. **Time Saving**: Generates descriptions instantly
4. **Better Flashcards**: Relevancy targets help focus flashcard generation on key topics
5. **Flexible**: Can disable if you prefer manual entry

## Integration with Flashcard Generation

The `relevancy_target` is used during flashcard generation to focus the LLM on specific technical areas:

```python
# In flashcard.py
relevancy_target = getattr(args, 'relevancy_target', None)
if relevancy_target:
    rating_prompt = f"Focus area for this book: {relevancy_target}\n\n{rating_prompt}"
```

This helps generate more targeted, relevant flashcards based on the book's main focus areas.

## Technical Details

### Prompt Template

```
Based on the book title: "{book_title}"

Generate a brief relevancy target description (5-10 words) that describes
the main technical focus areas or topics this book covers.

Focus on: key technical concepts, programming techniques, methodologies,
or domain-specific knowledge.

Format: Return ONLY the description without any punctuation at the end
(no periods, commas, etc.)

Examples:
- "LLM implementation techniques and practical applications"
- "Distributed systems concepts and design patterns"
- "Python programming techniques and idioms"

Description:
```

### Punctuation Removal

Uses regex to remove trailing punctuation:
```python
relevancy_target = re.sub(r'[.,;:!?]+$', '', relevancy_target)
```

This ensures clean output without any trailing punctuation marks.

## Troubleshooting

### LLM Not Responding

If the LLM doesn't respond:
```
[WARN] Failed to generate relevancy target: Connection refused
```

**Solution**: Make sure Ollama is running:
```bash
ollama serve
```

### Wrong Model

If using a different model:
```bash
export OLLAMA_MODEL=mistral:latest
python add_book.py mybook.pdf
```

### Disable for Specific Books

```bash
# Add without relevancy for sensitive/unclear titles
python add_book.py unclear_book.pdf --no-auto-relevancy
```

Then manually edit `orchestration.csv` to add the relevancy target.

---

**Note**: The auto-relevancy feature is enabled by default. Use `--no-auto-relevancy` to disable it if needed.
