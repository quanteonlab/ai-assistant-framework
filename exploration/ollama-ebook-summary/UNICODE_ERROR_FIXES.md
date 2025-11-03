# Unicode Error Handling Improvements

## Problem
The flashcard generation pipeline was failing with Unicode encoding errors when trying to print flashcard titles containing special characters (like ellipsis `…`) to the Windows console, which uses CP1252 encoding.

## Error Details
```
Traceback (most recent call last):
  File "E:\Documents\GitHub\ai-assistant-framework\exploration\ollama-ebook-summary\flashcard.py", line 787, in <module>
    main()
  File "E:\Documents\GitHub\ai-assistant-framework\exploration\ollama-ebook-summary\flashcard.py", line 782, in main
    process_csv_for_flashcards(input_file, config, api_base, model, output_dir, args.verbose,     
  File "E:\Documents\GitHub\ai-assistant-framework\exploration\ollama-ebook-summary\flashcard.py", line 536, in process_csv_for_flashcards
    print(f"    Flashcard '{flashcard_title[:50]}...' rating: {usefulness_rating}/10 (rated in {rating_time:.2f}s)")
  File "C:\Users\Nelson Wang\anaconda3\envs\windows_base_1\Lib\encodings\cp1252.py", line 19, in encode
    return codecs.charmap_encode(input,self.errors,encoding_table)[0]
UnicodeEncodeError: 'charmap' codec can't encode character '\u2026' in position 49: character maps to <undefined>
```

## Solutions Implemented

### 1. Safe Print Function
Added `safe_print()` and `safe_format_title()` functions to handle Unicode encoding errors gracefully:

```python
def safe_print(text: str) -> None:
    """Safely print text, handling Unicode encoding errors on Windows."""
    try:
        print(text)
    except UnicodeEncodeError:
        # Fallback: encode to ASCII with error replacement
        safe_text = text.encode('ascii', 'replace').decode('ascii')
        print(safe_text)
    except Exception as e:
        # Ultimate fallback: print a simple message
        print(f"[Print Error] Could not display text: {type(e).__name__}")

def safe_format_title(title: str, max_length: int = 50) -> str:
    """Safely format a title for display, handling Unicode issues."""
    try:
        if len(title) > max_length:
            return title[:max_length] + "..."
        return title
    except Exception:
        # Fallback for any string processing errors
        return "[Title with special characters]"
```

### 2. Comprehensive Error Handling
Wrapped critical sections in try-except blocks to ensure the pipeline continues processing even when individual flashcards fail:

- **Entry-level error handling**: Each CSV row processing is wrapped in try-catch
- **Flashcard-level error handling**: Each individual flashcard rating and processing
- **File operation error handling**: High-quality file writing operations
- **Print operation error handling**: All print statements use `safe_print()`

### 3. Graceful Continuation
Instead of stopping the entire pipeline, the system now:
- Logs the specific error that occurred
- Continues with the next flashcard/entry
- Writes default values to training CSV for failed entries
- Maintains pipeline state for successful entries

### 4. Improved Logging
All print statements now use `safe_print()` to prevent encoding issues:
- Progress messages
- Error messages  
- Summary statistics
- File completion notifications

## Status Reset
Reset "Parallel and High Performance Computing.pdf" from ERROR to NOT_STARTED status in orchestration.csv to allow reprocessing with the new error handling.

## Test Results
The `safe_print()` function correctly handles Unicode characters by replacing them with ASCII equivalents:
- Input: `Algorithm Analysis for Parallel Computing Applicat…`
- Output: `Algorithm Analysis for Parallel Computing Applicat?`

## Impact
- **Resilience**: Pipeline no longer crashes on Unicode encoding errors
- **Continuity**: Processing continues with remaining flashcards when one fails
- **Visibility**: Better error reporting with specific error types and messages
- **Compatibility**: Works consistently across different console encodings