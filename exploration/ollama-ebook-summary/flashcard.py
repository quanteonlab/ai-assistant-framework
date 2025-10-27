import os, sys, csv, time, re, json, yaml
import requests, argparse, traceback
from typing import Dict, Any, Tuple, Optional, List
from urllib.parse import urljoin
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class Config:
    """Centralized access to configuration parameters."""

    def __init__(self, config_path: str = None):
        # Use Path for cross-platform path handling
        script_dir = Path(__file__).parent.absolute()

        # Default config path using Path for proper path joining
        if config_path is None:
            config_path = script_dir / "_config.yaml"
        else:
            config_path = Path(config_path)

        self.config = self.load_config(config_path)
        self.prompts = self.config.get('prompts', {})
        self.defaults = self.config.get('defaults', {})

    @staticmethod
    def load_config(config_path: Path) -> dict:
        """Load configuration from a YAML file."""
        try:
            with config_path.open('r', encoding='utf-8') as file:
                return yaml.safe_load(file)
        except FileNotFoundError:
            print(f"Configuration file {config_path} not found.")
            sys.exit(1)
        except yaml.YAMLError as e:
            print(f"Error parsing the configuration file: {e}")
            sys.exit(1)
        except PermissionError:
            print(f"Permission denied when accessing {config_path}")
            sys.exit(1)

    def get_prompt(self, alias: str) -> str:
        """Retrieve prompt by alias from the configuration."""
        prompt = self.prompts.get(alias, {}).get('prompt')
        if not prompt:
            print(f"Prompt alias '{alias}' not found in configuration.")
            sys.exit(1)
        return prompt

# -----------------------------
# Error Handling
# -----------------------------

def handle_error(message: str, details: Dict[str, Any] = None, exit: bool = True):
    """
    Handle errors by printing a detailed message and optionally exiting.

    Args:
        message: Main error message
        details: Dictionary containing additional error details
        exit: Whether to exit the program
    """
    print("\n=== ERROR DETAILS ===")
    print(f"Error: {message}")

    if details:
        print("\n--- Additional Details ---")
        for key, value in details.items():
            print(f"{key}: {value}")

    print("=====================\n")

    if exit:
        sys.exit(1)

# -----------------------------
# API Interaction
# -----------------------------

def make_api_request(api_base: str, endpoint: str, payload: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Make a POST request to the specified API endpoint with detailed error handling."""
    full_url = urljoin(api_base + "/", endpoint)

    try:
        response = requests.post(full_url, json=payload)
        response.raise_for_status()
        return response.json()

    except requests.RequestException as e:
        error_details = {
            "Request URL": full_url,
            "Request Method": "POST",
            "Request Headers": dict(response.request.headers),
            "Request Payload": payload,
            "Response Status": getattr(response, 'status_code', None),
            "Response Headers": getattr(response, 'headers', {}),
            "Response Body": getattr(response, 'text', ''),
            "Exception Type": type(e).__name__,
            "Exception Message": str(e)
        }
        handle_error("API request failed", error_details, exit=False)

    except json.JSONDecodeError as e:
        error_details = {
            "Request URL": full_url,
            "Request Method": "POST",
            "Request Headers": dict(response.request.headers),
            "Request Payload": payload,
            "Response Status": response.status_code,
            "Response Headers": dict(response.headers),
            "Raw Response": response.text,
            "JSON Error": str(e),
            "JSON Error Position": f"line {e.lineno}, column {e.colno}"
        }
        handle_error("Failed to parse JSON response", error_details, exit=False)

    except Exception as e:
        error_details = {
            "Request URL": full_url,
            "Request Method": "POST",
            "Request Payload": payload,
            "Exception Type": type(e).__name__,
            "Exception Message": str(e),
            "Traceback": traceback.format_exc()
        }
        handle_error("Unexpected error during API request", error_details, exit=False)

    return None

# -----------------------------
# Text Sanitization
# -----------------------------

def sanitize_text(text: str) -> str:
    """Sanitize the input text by replacing unwanted characters."""
    return text.strip()

# -----------------------------
# Flashcard Generation
# -----------------------------

def generate_flashcards(api_base: str, model: str, clean_text: str, flashcard_prompt: str) -> Optional[str]:
    """Generate flashcards using the specified API."""
    payload = {
        "model": model,
        "prompt": f"```{clean_text}```\n\n{flashcard_prompt}",
        "stream": False
    }

    result = make_api_request(api_base, "generate", payload)
    if result:
        return result.get("response", "").strip()
    return None

def rate_flashcard(api_base: str, model: str, flashcard: str, rating_prompt: str) -> int:
    """
    Rate a flashcard's technical usefulness on a scale of 1-10.

    Args:
        api_base: API base URL
        model: Model name to use for rating
        flashcard: The flashcard content to rate
        rating_prompt: The rating prompt template

    Returns:
        Integer rating from 1-10, or 0 if rating fails
    """
    # Format the prompt with the flashcard content
    full_prompt = rating_prompt.replace("{flashcard}", flashcard)

    payload = {
        "model": model,
        "prompt": full_prompt,
        "stream": False
    }

    result = make_api_request(api_base, "generate", payload)
    if result:
        response = result.get("response", "").strip()
        # Extract the first number from the response
        import re
        match = re.search(r'\b([1-9]|10)\b', response)
        if match:
            return int(match.group(1))

    # Return 0 if rating fails
    return 0

# -----------------------------
# Processing Logic
# -----------------------------

def is_relevant_for_flashcards(title: str, text: str, min_text_length: int = 200) -> bool:
    """
    Determine if a section is relevant for flashcard generation.
    Filters out front matter, back matter, and non-technical sections.

    Args:
        title: The title of the section
        text: The text content of the section
        min_text_length: Minimum text length to consider (default: 200)

    Returns:
        True if the section should have flashcards generated, False otherwise
    """
    # Normalize title for checking
    title_lower = title.lower().strip()
    text_lower = text.lower().strip()

    # List of non-relevant section patterns (front matter and back matter)
    non_relevant_patterns = [
        'half title', 'half-title',
        'dedication',
        'contents', 'table of contents', 'toc',
        'copyright', 'copyright page',
        'acknowledgment', 'acknowledgement',
        'about the author', 'about author',
        'preface',
        'foreword',
        'introduction',  # Often just overview, not technical
        'index',
        'bibliography',
        'references',
        'glossary',  # Usually just definitions
        'appendix a', 'appendix b', 'appendix c',  # Often supplementary
        'back cover', 'front cover',
        'praise for',
        'also by',
        'colophon',
        'title page',
        'blank page',
        'intentionally left blank'
    ]

    # Check if title matches any non-relevant pattern
    for pattern in non_relevant_patterns:
        if pattern in title_lower:
            return False

    # Check if text contains common front/back matter indicators
    front_back_matter_keywords = [
        'this page intentionally left blank',
        'copyright ©',
        'all rights reserved',
        'isbn',
        'published by',
        'library of congress',
        'for information about buying this title',
        'praise for',
        'visit us on the web'
    ]

    for keyword in front_back_matter_keywords:
        if keyword in text_lower:
            return False

    # Filter out very short sections (likely not substantive)
    if len(text.strip()) < min_text_length:
        return False

    # Check if text is mostly administrative (high ratio of capitalized words)
    words = text.split()
    if len(words) > 10:
        capitalized_count = sum(1 for word in words if word and word[0].isupper())
        cap_ratio = capitalized_count / len(words)
        # If more than 50% capitalized, likely a title page or credits
        if cap_ratio > 0.5:
            return False

    return True

def sanitize_filename(text: str, max_length: int = 50) -> str:
    """
    Sanitize text for use in a filename.

    Args:
        text: The text to sanitize
        max_length: Maximum length of the resulting filename

    Returns:
        A safe filename string
    """
    # Remove or replace problematic characters
    text = re.sub(r'[<>:"/\\|?*]', '', text)
    text = re.sub(r'\s+', '_', text)
    text = text.strip('_')

    # Truncate if too long
    if len(text) > max_length:
        text = text[:max_length].strip('_')

    return text

def process_csv_for_flashcards(input_file: str, config: Config, api_base: str,
                               model: str, output_dir: str, verbose: bool = False,
                               min_length: int = 200, save_training_data: bool = True,
                               chapters_per_file: int = 3, max_text_size: int = 50000,
                               enable_rating: bool = False, rating_threshold: int = 8,
                               rating_model: str = None):
    """
    Process CSV input files and generate flashcards split across multiple files.

    Files are split based on either:
    - Number of chapters processed (chapters_per_file)
    - Total text size accumulated (max_text_size)

    Args:
        input_file: Path to input CSV file
        config: Configuration object
        api_base: API base URL
        model: Model name to use
        output_dir: Output directory for flashcards
        verbose: Print flashcards to terminal
        min_length: Minimum text length to process
        save_training_data: Save training data to CSV
        chapters_per_file: Number of chapters per file (default: 3)
        max_text_size: Maximum total text size per file in characters (default: 50000)
        enable_rating: Enable flashcard rating (default: False)
        rating_threshold: Minimum rating for high-quality folder (default: 8)
        rating_model: Model to use for rating (default: same as generation model)
    """

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Get flashcard prompt
    flashcard_prompt = config.get_prompt('flashcards')

    # Get rating prompt and model if rating is enabled
    rating_prompt = None
    actual_rating_model = rating_model or model
    if enable_rating:
        rating_prompt = config.get_prompt('flashcard_rating')
        high_quality_dir = os.path.join(output_dir, "high_quality")
        os.makedirs(high_quality_dir, exist_ok=True)
        print(f"Rating enabled: High-quality flashcards (>= {rating_threshold}/10) will be saved to: {high_quality_dir}")

    # Output files
    base_name = os.path.splitext(os.path.basename(input_file))[0]
    training_csv_file = os.path.join(output_dir, "flashcards_training_data.csv")

    # Check if training CSV exists to determine if we need to write headers
    training_csv_exists = os.path.exists(training_csv_file)

    with open(input_file, "r", encoding='utf-8') as csv_in:
        reader = csv.DictReader(csv_in)

        # Open training data CSV in append mode
        training_csv_out = None
        training_writer = None
        if save_training_data:
            training_csv_out = open(training_csv_file, "a", newline="", encoding='utf-8')
            training_writer = csv.writer(training_csv_out)
            # Write header only if file is new
            if not training_csv_exists:
                header = [
                    "source_file", "title", "input_text", "input_length",
                    "flashcards_output", "output_length", "model",
                    "timestamp", "elapsed_time_seconds"
                ]
                if enable_rating:
                    header.append("usefulness_rating")
                training_writer.writerow(header)

        try:
            processed_count = 0
            skipped_count = 0
            chapter_count = 0
            current_file_num = 1
            previous_title = ""
            chapters_in_current_file = []
            current_file_text_size = 0  # Track accumulated text size
            flashcards_buffer = []

            md_out = None
            current_output_file = None

            # Track high-quality flashcards
            hq_md_out = None
            hq_current_output_file = None
            hq_current_file_num = 1
            hq_chapters_in_current_file = []
            hq_current_file_text_size = 0

            for idx, row in enumerate(reader):
                text = next((row[key] for key in row if key.lower() == "text"), "").strip()
                clean = sanitize_text(text)
                title = next((row[key] for key in row if key.lower() == "title"), "").strip()

                if not clean:
                    continue

                # Check if this section is relevant for flashcards
                if not is_relevant_for_flashcards(title, text, min_text_length=min_length):
                    skipped_count += 1
                    print(f"Skipping entry {idx + 1}: {title[:50]}... (non-relevant section)")
                    continue

                # Detect new chapter (title changed from previous)
                is_new_chapter = title != previous_title and previous_title != ""

                if is_new_chapter:
                    chapter_count += 1

                # Check if we need to start a new file based on:
                # 1. Number of chapters reached the limit
                # 2. Text size exceeded the threshold
                should_create_new_file = False
                if is_new_chapter and chapter_count > 0:
                    if chapter_count % chapters_per_file == 0:
                        should_create_new_file = True
                        split_reason = f"chapter limit ({chapters_per_file})"
                    elif current_file_text_size >= max_text_size:
                        should_create_new_file = True
                        split_reason = f"text size limit ({current_file_text_size:,} chars)"

                if should_create_new_file:
                    # Close current file if open
                    if md_out:
                        md_out.close()
                        print(f"\n  Completed file: {current_output_file}")
                        print(f"  Reason: Reached {split_reason}")
                        print(f"  Chapters included: {', '.join(chapters_in_current_file)}")

                    # Reset for next file
                    current_file_num += 1
                    chapters_in_current_file = []
                    current_file_text_size = 0
                    md_out = None

                # Open new file if needed
                if md_out is None:
                    # Track the first chapter title for this file
                    if title not in chapters_in_current_file:
                        chapters_in_current_file.append(title)

                    # Generate descriptive filename with first chapter
                    first_chapter = sanitize_filename(chapters_in_current_file[0])
                    current_output_file = os.path.join(
                        output_dir,
                        f"{base_name}_part{current_file_num:02d}_{first_chapter}.md"
                    )

                    md_out = open(current_output_file, "w", encoding='utf-8')
                    # Write header with chapter range info
                    md_out.write(f"# Flashcards: {base_name} (Part {current_file_num})\n\n")
                    md_out.write(f"**Starting Chapter:** {chapters_in_current_file[0]}\n\n")
                    md_out.write("---\n\n")
                    print(f"\nStarting new file: {os.path.basename(current_output_file)}")

                print(f"Processing entry {idx + 1}: {title[:50]}...")
                processed_count += 1

                # Track chapters for filename generation
                if title not in chapters_in_current_file:
                    chapters_in_current_file.append(title)

                # Accumulate text size
                current_file_text_size += len(clean)

                # Generate flashcards
                start_time = time.time()
                flashcards = generate_flashcards(api_base, model, clean, flashcard_prompt)
                elapsed_time = time.time() - start_time

                if flashcards:
                    # Write to markdown output file
                    md_out.write(flashcards)
                    md_out.write("\n\n")

                    if verbose:
                        print(flashcards)

                    print(f"  Generated in {elapsed_time:.2f}s")

                    # Rate the flashcard if rating is enabled
                    usefulness_rating = 0
                    if enable_rating and rating_prompt:
                        rating_start = time.time()
                        usefulness_rating = rate_flashcard(api_base, actual_rating_model, flashcards, rating_prompt)
                        rating_time = time.time() - rating_start
                        print(f"  Usefulness rating: {usefulness_rating}/10 (rated in {rating_time:.2f}s)")

                        # If rating meets threshold, save to high-quality folder
                        if usefulness_rating >= rating_threshold:
                            # Check if we need to start a new high-quality file
                            should_create_new_hq_file = False
                            is_new_hq_chapter = title not in hq_chapters_in_current_file

                            if is_new_hq_chapter and len(hq_chapters_in_current_file) > 0:
                                if len(hq_chapters_in_current_file) >= chapters_per_file:
                                    should_create_new_hq_file = True
                                elif hq_current_file_text_size >= max_text_size:
                                    should_create_new_hq_file = True

                            if should_create_new_hq_file:
                                if hq_md_out:
                                    hq_md_out.close()
                                hq_current_file_num += 1
                                hq_chapters_in_current_file = []
                                hq_current_file_text_size = 0
                                hq_md_out = None

                            # Open new high-quality file if needed
                            if hq_md_out is None:
                                if title not in hq_chapters_in_current_file:
                                    hq_chapters_in_current_file.append(title)

                                first_chapter = sanitize_filename(hq_chapters_in_current_file[0])
                                hq_current_output_file = os.path.join(
                                    high_quality_dir,
                                    f"{base_name}_hq_part{hq_current_file_num:02d}_{first_chapter}.md"
                                )

                                hq_md_out = open(hq_current_output_file, "w", encoding='utf-8')
                                hq_md_out.write(f"# High-Quality Flashcards: {base_name} (Part {hq_current_file_num})\n\n")
                                hq_md_out.write(f"**Rating threshold:** >= {rating_threshold}/10\n\n")
                                hq_md_out.write(f"**Starting Chapter:** {hq_chapters_in_current_file[0]}\n\n")
                                hq_md_out.write("---\n\n")

                            # Track chapter
                            if title not in hq_chapters_in_current_file:
                                hq_chapters_in_current_file.append(title)

                            # Write to high-quality file
                            hq_md_out.write(f"**Rating: {usefulness_rating}/10**\n\n")
                            hq_md_out.write(flashcards)
                            hq_md_out.write("\n\n")
                            hq_current_file_text_size += len(clean)

                    # Save to training data CSV
                    if save_training_data and training_writer:
                        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
                        row_data = [
                            base_name,
                            title,
                            clean,
                            len(clean),
                            flashcards,
                            len(flashcards),
                            model,
                            timestamp,
                            f"{elapsed_time:.2f}"
                        ]
                        if enable_rating:
                            row_data.append(usefulness_rating)
                        training_writer.writerow(row_data)
                else:
                    print(f"  Failed to generate flashcards")

                previous_title = title

            # Close the last file
            if md_out:
                md_out.close()
                print(f"\n  Completed file: {current_output_file}")

            # Close high-quality file if it was opened
            if hq_md_out:
                hq_md_out.close()
                print(f"\n  Completed high-quality file: {hq_current_output_file}")

            print(f"\n--- Summary ---")
            print(f"Processed: {processed_count} sections")
            print(f"Skipped: {skipped_count} non-relevant sections")
            print(f"Files created: {current_file_num}")
            if enable_rating:
                print(f"High-quality files created: {hq_current_file_num if hq_md_out or hq_current_file_num > 1 else 0}")

        finally:
            if training_csv_out:
                training_csv_out.close()

    if save_training_data:
        print(f"Training data appended to: {training_csv_file}")

# -----------------------------
# Help Display
# -----------------------------

def display_help():
    """Display help message."""
    help_message = """
    Usage: python flashcard.py [OPTIONS] input_file

    Options:
    -c, --csv                Process a CSV file. Expected columns: Title, Text
    -m, --model              Model name to use for generation (default from config)
    -o, --output             Output directory for flashcards (default: flashcards/)
    -v, --verbose            Print flashcards to terminal as they're generated
    --min-length             Minimum text length to process (default: 200)
    --chapters-per-file      Number of chapters per output file (default: 3)
    --max-text-size          Maximum text size per file in characters (default: 50000)
    --enable-rating          Enable flashcard usefulness rating (1-10 scale)
    --rating-threshold       Minimum rating for high-quality folder (default: 8)
    --rating-model           Model to use for rating (default: same as generation)
    --no-training-data       Disable saving training data to CSV
    --help                   Show this help message and exit.

    For CSV input:
    - Ensure your CSV has 'Title' and 'Text' columns.
    - The script will generate flashcards for each row.
    - Non-relevant sections (dedication, copyright, TOC, etc.) are automatically skipped.

    File Splitting:
    - Flashcards are automatically split into multiple markdown files
    - Files are split based on EITHER:
      * Number of chapters processed (--chapters-per-file, default: 3)
      * Total text size accumulated (--max-text-size, default: 50000 chars)
    - Each file is named with the starting chapter title for easy reference
    - Example: ebook_part01_Introduction_to_Python.md

    Usefulness Rating (Optional):
    - When --enable-rating is used, each flashcard is rated 1-10 for technical value
    - Rating considers:
      * Technical depth and accuracy
      * Practical applicability in real-world engineering
      * Long-term value for career development
      * Fundamental vs. trivial concepts
    - High-quality flashcards (>= --rating-threshold, default 8) are saved to:
      * output_dir/high_quality/ folder
      * Organized in the same chapter-based file structure
      * Each flashcard includes its rating in the output
    - The usefulness_rating column is added to training data CSV

    Output:
    - Flashcards will be saved in markdown format
    - Each flashcard follows the format: #### Header, :p prompt, ??x answer x??
    - Flashcards are separated by ---
    - Training data is saved to flashcards_training_data.csv (aggregable across runs)

    Training Data Export:
    - By default, inputs and outputs are saved to flashcards_training_data.csv
    - This CSV is aggregable - multiple runs append to the same file
    - Useful for fine-tuning models in the future
    - Columns: source_file, title, input_text, input_length, flashcards_output,
               output_length, model, timestamp, elapsed_time_seconds
    - When rating is enabled, adds: usefulness_rating (1-10)

    Filtering:
    - The script automatically filters out non-technical sections like:
      * Front matter: dedications, copyright, table of contents, preface
      * Back matter: index, bibliography, references
      * Very short sections (< min-length characters)

    Examples:
    # Default: 3 chapters or 50000 chars per file
    python flashcard.py -c input.csv

    # Custom: 2 chapters or 30000 chars per file
    python flashcard.py -c input.csv --chapters-per-file 2 --max-text-size 30000

    # Larger files: 5 chapters or 100000 chars per file
    python flashcard.py -c input.csv --chapters-per-file 5 --max-text-size 100000

    # Enable rating with default threshold (8/10)
    python flashcard.py -c input.csv --enable-rating

    # Custom rating threshold: only save flashcards rated 9-10
    python flashcard.py -c input.csv --enable-rating --rating-threshold 9

    # Use different model for rating
    python flashcard.py -c input.csv --enable-rating --rating-model qwen2.5:latest
    """
    print(help_message)

# -----------------------------
# Main Function
# -----------------------------

def main():
    config = Config()
    parser = argparse.ArgumentParser(description="Generate flashcards from CSV files using a specified model.", add_help=False)

    # Optional Arguments
    parser.add_argument('-m', '--model', default=config.defaults.get('summary', 'DEFAULT_SUMMARY_MODEL'), help='Model name to use for generation')
    parser.add_argument('-c', '--csv', action='store_true', help='Process a CSV file')
    parser.add_argument('-o', '--output', default='flashcards', help='Output directory for flashcards')
    parser.add_argument('--min-length', type=int, default=200, help='Minimum text length to process (default: 200)')
    parser.add_argument('--chapters-per-file', type=int, default=3, help='Number of chapters per flashcard file (default: 3)')
    parser.add_argument('--max-text-size', type=int, default=50000, help='Maximum text size per file in characters (default: 50000)')
    parser.add_argument('--enable-rating', action='store_true', help='Enable flashcard usefulness rating (1-10 scale)')
    parser.add_argument('--rating-threshold', type=int, default=8, help='Minimum rating for high-quality folder (default: 8)')
    parser.add_argument('--rating-model', help='Model to use for rating (default: same as generation model)')
    parser.add_argument('--no-training-data', action='store_true', help='Disable saving training data to CSV')
    parser.add_argument('--help', action='store_true', help='Show help message and exit')
    parser.add_argument('-v', '--verbose', action='store_true', help='Display flashcards as they are generated')
    parser.add_argument('input_file', nargs='?', help='Input file path')

    args = parser.parse_args()

    if args.help:
        display_help()
        sys.exit(0)

    if not args.input_file:
        handle_error("Error: Input file is required when not using --help.")

    if not args.csv:
        handle_error("Error: You must specify --csv for CSV input.")

    model = args.model
    input_file = args.input_file
    api_base = os.getenv('OLLAMA_API_BASE', 'http://localhost:11434/api')
    output_dir = args.output
    min_length = args.min_length
    save_training_data = not args.no_training_data
    chapters_per_file = args.chapters_per_file
    max_text_size = args.max_text_size
    enable_rating = args.enable_rating
    rating_threshold = args.rating_threshold
    rating_model = args.rating_model

    # Process the CSV
    process_csv_for_flashcards(input_file, config, api_base, model, output_dir, args.verbose,
                              min_length, save_training_data, chapters_per_file, max_text_size,
                              enable_rating, rating_threshold, rating_model)

if __name__ == "__main__":
    main()
