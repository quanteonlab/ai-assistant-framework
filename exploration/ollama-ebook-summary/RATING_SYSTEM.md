# Flashcard Rating System

## Overview

All flashcards are automatically rated on a **1-10 scale** for technical usefulness and long-term learning value.

## Single Threshold System

We use **one rating threshold** for all quality filtering:

```bash
python pdf2flashcards.py mybook.pdf --rating-threshold 8
```

This threshold is used for:
1. ✅ **During generation** - Real-time rating as flashcards are created
2. ✅ **Post-processing** - Rating any unrated flashcards after generation
3. ✅ **Consolidation** - Selecting which flashcards to include in high-quality markdown files

## Default Threshold: 8/10

By default, only flashcards rated **8-10** are saved to the `high_quality/` folder.

### Rating Scale

| Rating | Quality | Saved to Markdown? |
|--------|---------|-------------------|
| 10 | Exceptional - Must memorize | ✅ Yes |
| 9 | Excellent - Very useful | ✅ Yes |
| 8 | Very good - Worth learning | ✅ Yes |
| 7 | Good - Moderately useful | ❌ No (CSV only) |
| 6 | Decent - Some value | ❌ No (CSV only) |
| 1-5 | Low priority | ❌ No (CSV only) |

## Output Files

### CSV (All Flashcards)
- **File**: `flashcards/flashcards_training_data.csv`
- **Contains**: ALL flashcards with their ratings (1-10)
- **Purpose**: Training data, analytics, review

### Markdown (High-Quality Only)
- **Folder**: `flashcards/high_quality/`
- **Contains**: Only flashcards rated ≥ threshold (default: 8-10)
- **Format**: Consolidated files per book, split every 2000 lines
- **Purpose**: Study and review

## Customizing the Threshold

### Strict Quality (9-10 only)
```bash
python pdf2flashcards.py mybook.pdf --rating-threshold 9
```

### Inclusive Quality (6-10)
```bash
python pdf2flashcards.py mybook.pdf --rating-threshold 6
```

### Very Strict (10 only)
```bash
python pdf2flashcards.py mybook.pdf --rating-threshold 10
```

## Disabling Rating

### Disable All Rating
```bash
python pdf2flashcards.py mybook.pdf --disable-rating --disable-post-rating
```

### Disable During Generation (Only Post-Process)
```bash
python pdf2flashcards.py mybook.pdf --disable-rating
```

### Disable Post-Processing (Only During Generation)
```bash
python pdf2flashcards.py mybook.pdf --disable-post-rating
```

## Rating Process

### Stage 1: Real-Time Rating (During Generation)
As each flashcard is generated, it's immediately rated by AI:

```
[INFO] Processing entry 42: Chapter 5...
  Generated in 2.3s
  Usefulness rating: 9/10 (rated in 1.2s)
```

**Result**: Flashcard is saved with rating in CSV

### Stage 2: Post-Processing Rating (After Generation)
After all flashcards are generated, any unrated flashcards are rated:

```
[Step 3] Rating unrated flashcards...
[1/50] Rating: Technical Debt Concept...
  Rating: 8/10 (took 1.5s)
```

**Result**: Previously unrated flashcards now have ratings

### Stage 3: Consolidation
All flashcards meeting the threshold are consolidated into markdown files:

```
Consolidating high-quality flashcards by source...
  Source: mybook (142 high-quality flashcards)
    Created: mybook_part01.md (1856 lines)
    Created: mybook_part02.md (1243 lines)
```

**Result**: Clean, consolidated markdown files in `high_quality/` folder

## What Gets Rated?

The AI rates flashcards based on:
- ✅ **Technical depth** - Is this concept important?
- ✅ **Practical value** - Will I use this in real projects?
- ✅ **Long-term retention** - Is this worth memorizing?
- ✅ **Fundamentals vs trivia** - Core concept vs minor detail?

## Example Ratings

### Rating 10/10 - Exceptional
```markdown
#### Big O Notation Fundamentals
:p What is the time complexity of binary search?
??x O(log n) - divides the search space in half each iteration x??
```
**Why 10**: Fundamental CS concept, used constantly, critical for interviews

### Rating 8/10 - Very Good
```markdown
#### Python List Comprehension Syntax
:p How do you create a list of squares using list comprehension?
??x [x**2 for x in range(10)] x??
```
**Why 8**: Very useful Python pattern, saves time, common in real code

### Rating 5/10 - Low Priority
```markdown
#### Book Publication Date
:p When was this book published?
??x 2023 x??
```
**Why 5**: Not technically valuable, purely informational

## Tips

1. **Start with default (8)**: Good balance of quality vs quantity
2. **Adjust based on book**: Technical books may need lower threshold (6-7)
3. **Review CSV for gems**: Sometimes good flashcards get rated 7 - review them!
4. **Trust the AI**: The rating model is trained on technical usefulness
5. **Iterate**: Process book, review ratings, adjust threshold if needed

## Orchestration

When using orchestration, set the threshold for all books:

```bash
# In orchestrate.py, the default threshold (8) is used
python orchestrate.py

# To use a custom threshold, modify pdf2flashcards.py call
# or set it as an environment variable
```
