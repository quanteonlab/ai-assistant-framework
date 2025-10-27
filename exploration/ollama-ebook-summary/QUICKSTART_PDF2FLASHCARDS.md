# Quick Start: PDF to Flashcards

## Single Command Pipeline

The `pdf2flashcards.py` script provides a complete end-to-end pipeline to convert PDF or EPUB files into flashcards with optional quality rating.

## Installation

Make sure you have all dependencies installed:
```bash
pip install -r requirements.txt
```

Ensure Ollama is running with the required models (see main README.md).

## Basic Usage

### Simplest Command
```bash
python pdf2flashcards.py mybook.pdf
```

This will:
1. Convert PDF to chunked CSV
2. Generate flashcards for each section
3. Split flashcards into multiple files (3 chapters per file by default)
4. Save to `flashcards/` directory

## Common Use Cases

### 1. Process EPUB File
```bash
python pdf2flashcards.py mybook.epub
```

### 2. Enable Quality Rating (Recommended!)
```bash
python pdf2flashcards.py mybook.pdf --enable-rating
```

This will:
- Rate each flashcard 1-10 for technical usefulness
- Save high-quality flashcards (rated >= 8/10) to `flashcards/high_quality/`
- Add rating column to training data CSV

### 3. Custom Output Directory
```bash
python pdf2flashcards.py mybook.pdf -o flashcards/mybook
```

### 4. Strict Quality Filter
```bash
python pdf2flashcards.py mybook.pdf --enable-rating --rating-threshold 9
```

Only saves flashcards rated 9-10 to the high-quality folder.

### 5. Smaller Flashcard Files
```bash
python pdf2flashcards.py mybook.pdf --chapters-per-file 2 --max-text-size 30000
```

Creates more, smaller flashcard files for easier review.

### 6. Keep Intermediate CSV for Inspection
```bash
python pdf2flashcards.py mybook.pdf --keep-csv
```

Useful for debugging or manual inspection of the conversion process.

## Full Command Reference

```bash
python pdf2flashcards.py --help
```

### Key Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `input_file` | (required) | PDF or EPUB file to process |
| `-o, --output` | `flashcards` | Output directory |
| `--enable-rating` | False | Enable flashcard quality rating |
| `--rating-threshold` | 8 | Minimum rating for high-quality folder |
| `--chapters-per-file` | 3 | Chapters per flashcard file |
| `--max-text-size` | 50000 | Max characters per file |
| `--min-length` | 200 | Minimum text length to process |
| `--keep-csv` | False | Keep intermediate CSV file |
| `-v, --verbose` | False | Print flashcards during generation |

## Output Structure

### Without Rating
```
flashcards/
|-- mybook_part01_Introduction.md
|-- mybook_part02_Chapter_Four.md
|-- mybook_part03_Chapter_Seven.md
+-- flashcards_training_data.csv
```

### With Rating Enabled
```
flashcards/
|-- mybook_part01_Introduction.md           # All flashcards
|-- mybook_part02_Chapter_Four.md
|-- mybook_part03_Chapter_Seven.md
|-- high_quality/                            # Only rated >= threshold
|   |-- mybook_hq_part01_Introduction.md
|   +-- mybook_hq_part02_Chapter_Four.md
+-- flashcards_training_data.csv             # Includes rating column
```

## What Gets Filtered Out

The script automatically skips non-relevant sections:
- Front matter (dedication, copyright, table of contents)
- Back matter (index, bibliography, references)
- Very short sections (< 200 characters by default)
- Administrative pages

## Processing Time

Typical processing times:
- **PDF to CSV**: 1-5 minutes (depends on book size)
- **CSV to Flashcards**: 5-30 minutes (depends on content and model speed)
- **With Rating**: Add 50-100% more time (each flashcard is rated)

For a typical 200-page technical book:
- Without rating: ~15-20 minutes total
- With rating: ~25-35 minutes total

## Troubleshooting

### "book2text.py failed"
- Make sure the PDF has proper text extraction (not scanned images)
- Try with EPUB if available (better structure preservation)

### "flashcard.py failed"
- Check that Ollama is running: `ollama list`
- Verify models are installed (see main README.md)
- Check `_config.yaml` has correct model names

### Rating Takes Too Long
- Use a faster model for rating: `--rating-model qwen2.5:latest`
- Or disable rating for faster processing

### Intermediate CSV Not Found
- The CSV should be in `out/` directory
- Use `--keep-csv` to inspect it
- Check console output for errors from book2text.py

## Tips for Best Results

1. **Use EPUB when available** - Better structure preservation than PDF
2. **Enable rating for study materials** - Helps focus on high-value content
3. **Adjust chapter splitting** - Use `--chapters-per-file 2` for more granular files
4. **Review training data** - Check the CSV to see input/output quality
5. **Start with defaults** - Fine-tune parameters after seeing initial results

## Examples

### Example 1: Quick Processing
```bash
# Fast, no rating, default settings
python pdf2flashcards.py programming_book.pdf
```

### Example 2: High-Quality Study Material
```bash
# Enable rating, strict threshold, organized output
python pdf2flashcards.py programming_book.pdf \
  --enable-rating \
  --rating-threshold 9 \
  -o flashcards/programming
```

### Example 3: Large Book, Smaller Files
```bash
# 2 chapters per file, smaller text chunks
python pdf2flashcards.py large_textbook.pdf \
  --chapters-per-file 2 \
  --max-text-size 30000
```

### Example 4: Debug Mode
```bash
# Keep CSV, verbose output, see everything
python pdf2flashcards.py mybook.pdf \
  --keep-csv \
  --verbose
```

## Integration with Existing Workflow

If you already have CSV files from `book2text.py`, you can still use just the flashcard generation:

```bash
# Just generate flashcards from existing CSV
python flashcard.py -c out/mybook_processed.csv -o flashcards/mybook --enable-rating
```

The `pdf2flashcards.py` script is a convenience wrapper that chains both steps together.

## Next Steps

1. Review generated flashcards in your favorite markdown viewer
2. Import into spaced repetition software (Anki, RemNote, etc.)
3. Check training data CSV for model performance analysis
4. Adjust parameters based on output quality
5. Fine-tune rating prompts in `_config.yaml` if needed

## Support

For issues or questions:
- Check the main README.md for detailed documentation
- Review FLASHCARD_RATING_FEATURE.md for rating system details
- Inspect intermediate CSV files with `--keep-csv` for debugging
