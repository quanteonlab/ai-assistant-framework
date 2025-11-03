# Flashcard Reorganization Guide

## Summary of Changes

The flashcard generation system has been updated to use a more organized folder structure with shorter, more manageable filenames.

### Old Structure
```
flashcards/
├── Long-Book-Name-Here_part01_Very_Long_Chapter_Name.md
├── Long-Book-Name-Here_part02_Another_Long_Chapter.md
└── high_quality/
    ├── Long-Book-Name-Here_hq_part01_Chapter.md
    └── Long-Book-Name-Here_hq_part02_Chapter.md
```

### New Structure
```
flashcards/
└── Long-Book-Name-Here/
    ├── short_part01_chapter_name.md
    ├── short_part02_another_chap.md
    ├── short_hq_part01_chapter_name.md
    ├── short_hq_part02_another_chap.md
    └── Long-Book-Name-Here_training_data.csv
```

## Key Changes

### 1. Book-Specific Folders
- Each book now gets its own folder named with the full book name
- All flashcards for a book are contained within its folder
- Training data CSV is also stored in the book folder

### 2. Shortened Filenames
Filenames now follow this pattern: `{prefix}_part{num}_{chapter}.md`

- **Prefix**: Max 5 characters, auto-generated from book name
  - Example: "Designing-data-intensive-applications" → "ddiai"
  - Example: "Understanding-Distributed-Systems" → "0uds2"

- **Part Number**: Zero-padded 2 digits (part01, part02, etc.)

- **Chapter Name**: Max 15 characters
  - Example: "Practical_considerations" → "Practical_consi"
  - Example: "Network_load_balancing" → "Network_load_ba"

### 3. High-Quality Flashcards
- No longer in a separate `high_quality/` folder
- Now in the same book folder with `_hq_` in the filename
- Example: `short_hq_part01_chapter.md`

## How to Use

### For New Flashcard Generation

Simply run the flashcard generation script as usual:

```bash
python flashcard.py -c input.csv --enable-rating
```

The script will automatically:
1. Create a folder named after the input CSV file (without extension)
2. Generate a short prefix for filenames
3. Save flashcards with the new naming scheme
4. Save high-quality flashcards in the same folder with `_hq_` prefix

### Reorganizing Existing Flashcards

To reorganize your existing flashcards into the new structure:

#### Step 1: Dry Run (Preview)
First, see what would be changed without actually moving files:

```bash
python reorganize_flashcards.py flashcards
```

This will show you:
- Which books were found
- What short prefix will be used for each book
- What the new filenames will be
- Where files will be moved

#### Step 2: Execute Reorganization
If the dry run looks good, execute the actual reorganization:

```bash
python reorganize_flashcards.py flashcards --execute
```

This will:
- Create book-specific folders
- Move all flashcard files into their book folders
- Rename files with shortened names
- Move training data CSVs into book folders
- Remove the empty `high_quality/` folder

## Examples

### Example 1: Concurrency Book

**Input CSV**: `ConcurrencyNetModern_processed.csv`

**Generated Structure**:
```
flashcards/
└── ConcurrencyNetModern/
    ├── cnm_part01_intro.md
    ├── cnm_part02_threads.md
    ├── cnm_hq_part01_intro.md
    └── ConcurrencyNetModern_training_data.csv
```

### Example 2: Data Intensive Applications

**Input CSV**: `Designing-data-intensive-applications_processed.csv`

**Generated Structure**:
```
flashcards/
└── Designing-data-intensive-applications/
    ├── ddiai_part01_reliability.md
    ├── ddiai_part02_scalability.md
    ├── ddiai_hq_part01_reliability.md
    └── Designing-data-intensive-applications_training_data.csv
```

## Benefits

1. **Better Organization**: Each book's flashcards are in a dedicated folder
2. **Shorter Paths**: Reduced filename length prevents path length issues on Windows
3. **Easier Navigation**: Browse flashcards by book folder
4. **Simplified Structure**: No separate high_quality folder to manage
5. **Consistent Naming**: Predictable naming pattern across all books

## Code Changes

The following files were modified:

### `flashcard.py`
- Added `generate_book_prefix()` function to create short prefixes
- Updated `process_csv_for_flashcards()` to:
  - Create book-specific folders
  - Use shortened filenames (max 15 chars for chapter names)
  - Save high-quality flashcards in same folder with `_hq_` prefix
  - Store training data CSV in book folder

### `reorganize_flashcards.py` (New)
- Script to reorganize existing flashcards into new structure
- Supports dry-run mode to preview changes
- Automatically generates short prefixes for consistency
- Moves both regular and high-quality flashcards

## Troubleshooting

### File Path Too Long Errors (Windows)
The new structure with shorter filenames should prevent this issue. If you still encounter it:
1. Move the `flashcards` directory closer to the drive root
2. Use even shorter chapter names by reducing `max_length` in code

### Duplicate Files
If you run the reorganization twice, files may get duplicated. To fix:
1. Delete the book folders
2. Restore files from the root `flashcards/` directory
3. Run reorganization again with `--execute`

### Missing High-Quality Flashcards
If high-quality flashcards weren't generated:
1. Make sure you used `--enable-rating` flag
2. Check that flashcards met the rating threshold (default: 8/10)
3. Look for files with `_hq_` in the filename

## Migration Checklist

- [ ] Backup your current flashcards folder
- [ ] Run dry-run to preview changes: `python reorganize_flashcards.py flashcards`
- [ ] Review the output and verify book names and prefixes look correct
- [ ] Execute reorganization: `python reorganize_flashcards.py flashcards --execute`
- [ ] Verify all files were moved correctly
- [ ] Test generating new flashcards with the updated script
- [ ] Delete backup after confirming everything works
