# Flashcard Rating Feature

## Overview
The flashcard generation script now includes an optional **usefulness rating system** that evaluates each flashcard on a scale of 1-10 based on its technical value and long-term retention importance for engineers.

## Key Features

### 1. **Automated Rating System**
- Each flashcard is evaluated by an LLM for technical usefulness
- Rating scale: 1 (not useful) to 10 (critical foundational knowledge)
- Considers:
  - Technical depth and accuracy
  - Practical applicability in real-world engineering
  - Fundamental concepts vs trivial facts
  - Long-term career development value

### 2. **High-Quality Flashcards Folder**
- Flashcards meeting the rating threshold are automatically saved to a separate folder
- Default threshold: 8/10 (configurable)
- Location: `output_dir/high_quality/`
- Same chapter-based file structure as main output
- Each flashcard includes its rating inline

### 3. **Enhanced Training Data**
- New column added to CSV: `usefulness_rating`
- Tracks rating for each flashcard generation
- Useful for analyzing model performance and fine-tuning

## Usage

### Basic Usage (Rating Disabled)
```bash
# No changes to existing behavior
python flashcard.py -c input.csv
```

### Enable Rating with Defaults
```bash
# Rates flashcards, saves those rated ≥8/10 to high_quality folder
python flashcard.py -c input.csv --enable-rating
```

### Custom Rating Threshold
```bash
# Only save flashcards rated 9-10
python flashcard.py -c input.csv --enable-rating --rating-threshold 9

# Save all flashcards rated 6 or higher
python flashcard.py -c input.csv --enable-rating --rating-threshold 6
```

### Use Different Model for Rating
```bash
# Use a faster/different model for rating
python flashcard.py -c input.csv --enable-rating --rating-model qwen2.5:latest
```

## Command-Line Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--enable-rating` | flag | False | Enable flashcard usefulness rating |
| `--rating-threshold` | int | 8 | Minimum rating for high-quality folder (1-10) |
| `--rating-model` | string | (same as generation) | Model to use for rating |

## Output Structure

### Without Rating
```
flashcards/
├── ebook_part01_Chapter_One.md
├── ebook_part02_Chapter_Four.md
└── flashcards_training_data.csv
```

### With Rating Enabled
```
flashcards/
├── ebook_part01_Chapter_One.md
├── ebook_part02_Chapter_Four.md
├── high_quality/
│   ├── ebook_hq_part01_Chapter_One.md     # Only highly-rated flashcards
│   └── ebook_hq_part02_Chapter_Four.md
└── flashcards_training_data.csv           # Includes usefulness_rating column
```

## High-Quality Flashcard Format

Each flashcard in the `high_quality` folder includes its rating:

```markdown
**Rating: 9/10**

#### Concept Title
Background context...
:p Question?
??x
Answer...
x??

---
```

## Configuration

The rating prompt is defined in `_config.yaml`:

```yaml
prompts:
  flashcard_rating:
    prompt: |
      Rate the following flashcard on a scale of 1-10 based on how technically
      useful and important it is for long-term retention as an engineer...
```

You can customize this prompt to adjust rating criteria.

## Performance Considerations

- Rating adds ~1-3 seconds per flashcard (depending on model speed)
- Consider using a faster model for rating if processing many flashcards
- Rating is done sequentially after generation to ensure accuracy

## Training Data CSV

When rating is enabled, the CSV includes:

| Column | Description |
|--------|-------------|
| source_file | Input file name |
| title | Chapter/section title |
| input_text | Source text |
| input_length | Character count |
| flashcards_output | Generated flashcard |
| output_length | Output character count |
| model | Model used for generation |
| timestamp | Generation time |
| elapsed_time_seconds | Generation duration |
| **usefulness_rating** | **1-10 technical usefulness score** |

## Examples

### Example 1: Default Settings
```bash
python flashcard.py -c mybook.csv --enable-rating
```
- Rates all flashcards
- Saves flashcards rated ≥8 to `flashcards/high_quality/`
- Uses same model for both generation and rating

### Example 2: Strict Quality Filter
```bash
python flashcard.py -c mybook.csv --enable-rating --rating-threshold 9
```
- Only saves flashcards rated 9-10 to high_quality folder
- More selective curation

### Example 3: Fast Rating
```bash
python flashcard.py -c mybook.csv --enable-rating --rating-model qwen2.5:latest
```
- Uses a potentially faster model for rating
- Reduces processing time

## Tips

1. **Start with defaults**: Use `--rating-threshold 8` initially to see the distribution of ratings
2. **Adjust threshold**: Based on output quality, adjust threshold up (stricter) or down (more inclusive)
3. **Model selection**: Use the same model for consistency, or a faster model for speed
4. **Review ratings**: Check the training CSV to see how ratings correlate with your judgment
5. **Iterative refinement**: Use rating data to fine-tune the rating prompt in `_config.yaml`
