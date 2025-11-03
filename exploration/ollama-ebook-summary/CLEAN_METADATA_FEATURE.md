# Clean Rating Metadata Feature

## Overview

The refined book notes (high-quality flashcards) now save only flashcard content without rating metadata. This keeps the documents clean and focused on the learning material.

## What Changed

### Before
```markdown
# High-Quality Flashcards: Book Name (Part 1)

**Rating threshold:** >= 8/10

**Starting Chapter:** Introduction

---

**Rating: 9/10**

#### Flashcard Title
Content here...

---

**Rating: 8/10**

#### Another Flashcard
More content...
```

### After
```markdown
# High-Quality Flashcards: Book Name (Part 1)

**Starting Chapter:** Introduction

---

#### Flashcard Title
Content here...

---

#### Another Flashcard
More content...
```

## Removed Metadata

The following lines are **no longer included** in refined book notes:

1. `**Rating threshold:** >= X/10` - Removed from file header
2. `**Rating: X/10**` - Removed before each flashcard

The `**Starting Chapter:**` line is **kept** as it provides useful context.

## Implementation

### 1. Updated flashcard.py (Lines 674, 683)

**Line 674 - File Header:**
```python
# BEFORE
hq_md_out.write(f"**Rating threshold:** >= {rating_threshold}/10\n\n")

# AFTER
# Rating threshold removed - keep only flashcard content
```

**Line 683 - Individual Ratings:**
```python
# BEFORE
hq_md_out.write(f"**Rating: {usefulness_rating}/10**\n\n")
hq_md_out.write(flashcard_content)

# AFTER
# Write to high-quality file (no rating metadata)
hq_md_out.write(flashcard_content)
```

### 2. Created clean_rating_metadata.py

A cleanup script to remove rating metadata from existing files:

**Features:**
- Scans all markdown files in `refined_docs/book_notes/`
- Removes rating threshold and individual rating lines
- Supports dry run mode to preview changes
- Cleans up excessive blank lines after removal

**Usage:**
```bash
# Dry run (preview changes)
python clean_rating_metadata.py

# Actually clean files
python clean_rating_metadata.py --execute

# Custom base directory
python clean_rating_metadata.py --execute --base-dir /path/to/project
```

## Cleanup Results

**Execution Summary:**
- **Total files scanned:** 944
- **Files modified:** 944
- **Rating lines removed:** 4,525

All existing refined book notes have been cleaned and are now metadata-free.

## Why This Change?

1. **Cleaner Content**: Focuses on learning material, not internal metrics
2. **Better Readability**: Removes visual clutter
3. **Export-Friendly**: Cleaner files for sharing or exporting
4. **Simplified Format**: Easier to parse and process externally

## Rating Data Preservation

While ratings are removed from markdown files, they are still:
- **Tracked in CSV files**: Training data CSVs contain rating information
- **Used during generation**: Rating system still filters low-quality flashcards
- **Stored in orchestration logs**: Processing logs maintain rating metrics

Only the **final output markdown** is clean - internal processing still uses ratings.

## For Future Flashcards

All new flashcards generated after this change will automatically:
- ✅ Skip rating threshold header
- ✅ Skip individual rating lines
- ✅ Keep starting chapter information
- ✅ Include only flashcard content and separators

## Code References

### flashcard.py
- **Line 674:** Removed rating threshold from file header
- **Line 683:** Removed individual rating from flashcard output

### clean_rating_metadata.py
- **Main function:** `clean_rating_lines()` - Removes rating patterns with regex
- **Patterns matched:**
  - `r'^\*\*Rating threshold:\*\*\s*>=\s*\d+/10\s*$'`
  - `r'^\*\*Rating:\s*\d+/10\*\*\s*$'`

## Testing

Verified cleanup on sample file:
```bash
# Check a cleaned file
cat "refined_docs/book_notes/Angie-Papple-Johnston---ASVAB-2023-_-2024-For-Dummies/apja2_part03_Getting_to_the.md"
```

Results:
- ✅ No rating threshold in header
- ✅ No individual ratings before flashcards
- ✅ Flashcard content preserved intact
- ✅ Structure maintained correctly

## Rollback

If you need to restore ratings:
1. They're preserved in CSV training data files
2. Revert changes to `flashcard.py` lines 674 and 683
3. Regenerate flashcards from processed CSVs

---

**Note:** This change applies only to refined book notes in `refined_docs/book_notes/`. Regular flashcards in `flashcards/` were never affected by rating metadata.
