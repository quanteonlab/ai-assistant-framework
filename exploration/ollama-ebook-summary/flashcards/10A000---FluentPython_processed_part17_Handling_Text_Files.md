# Flashcards: 10A000---FluentPython_processed (Part 17)

**Starting Chapter:** Handling Text Files

---

#### BOM (Byte Order Mark) and UTF-8-SIG

Background context: The Byte Order Mark (BOM) is a special character at the start of a file to indicate its encoding, often used with UTF-8. However, using a BOM can break certain conventions like making Python scripts executable on Unix systems. The UTF-8-SIG codec in Python is designed to handle files with or without a BOM and avoids returning the BOM itself.

:p What is the significance of using UTF-8-SIG instead of regular UTF-8 when reading text files?

??x
Using UTF-8-SIG ensures that your code can correctly read files regardless of whether they contain a BOM. It is particularly useful in scenarios where you want to maintain compatibility with various file formats, such as those containing the BOM or not.

```python
# Example of using UTF-8-SIG to read a file with or without BOM
import codecs

with codecs.open('example.txt', 'r', encoding='utf_8_sig') as file:
    content = file.read()
```

x??

---

#### Unicode Sandwich for Text Processing in Python 3

Background context: The "Unicode sandwich" is the recommended practice for handling text files and strings in Python 3. It emphasizes decoding input to `str` objects early, processing data exclusively on `str` objects within your program, and encoding output to bytes late.

:p What does the term "Unicode sandwich" refer to?

??x
The "Unicode sandwich" refers to a best practice for handling text I/O in Python 3. It involves decoding input to `str` as early as possible, processing data exclusively on `str` objects within your program's logic, and encoding output to bytes as late as possible.

```python
# Example of the Unicode Sandwich
with open('example.txt', 'r', encoding='utf_8') as file:
    text = file.read()  # Decoding happens here

# Business logic using str objects
# ...

# Encoding back to bytes when writing
with open('output.txt', 'w', encoding='utf_8') as file:
    file.write(text)
```

x??

---

#### Handling Text Files with Default Encodings in Python 3

Background context: When working with text files, it is important to specify the correct encoding explicitly to avoid issues related to platform defaults. By default, Python uses the system's locale settings for encoding when opening a file without specifying an encoding.

:p What is the consequence of relying on default encodings when handling text files in Python?

??x
Relying on default encodings can lead to bugs and compatibility issues. Different machines or environments may have different default encodings, which can result in incorrect character decoding or encoding, as seen in Example 4-8.

```python
# Example of the problem with default encodings
with open('cafe.txt', 'w') as file:
    file.write('café')  # Default encoding is used here

with open('cafe.txt') as file:  # No explicit encoding specified, defaults to platform's locale
    content = file.read()  # Content might be incorrect due to different encoding
```

x??

---

#### Writing and Reading Files with Explicit Encoding

Background context: To ensure consistent behavior across platforms, it is crucial to explicitly specify the encoding when opening text files. Using `open` with an explicit encoding argument guarantees that the file is read or written in a specific encoding.

:p How does specifying an explicit encoding help avoid platform-specific issues?

??x
Specifying an explicit encoding when opening a file ensures consistent behavior across different platforms and environments, avoiding reliance on potentially inconsistent default encodings. For example, using UTF-8 explicitly can prevent issues related to character decoding on systems with different default settings.

```python
# Example of specifying an explicit encoding
with open('example.txt', 'w', encoding='utf_8') as file:
    file.write('café')

with open('example.txt', encoding='utf_8') as file:  # Explicitly specifies UTF-8
    content = file.read()  # Content is read correctly with the specified encoding
```

x??

---

#### Open Function and TextIOWrapper

Background context: The `open` function in Python 3 provides a `TextIOWrapper` object when working with text files. This object handles encoding and decoding based on the specified mode and encoding.

:p What is the role of the `TextIOWrapper` object returned by `open`?

??x
The `TextIOWrapper` object returned by `open` manages the conversion between bytes and strings, providing a layer for handling encodings. It allows you to read and write text files while abstracting away the details of byte-level operations.

```python
# Example of using TextIOWrapper
fp = open('example.txt', 'w', encoding='utf_8')
print(fp)  # <_io.TextIOWrapper name='example.txt' mode='w' encoding='utf_8'>
fp.write('café')
fp.close()

import os
os.stat('example.txt').st_size  # 5 bytes, due to UTF-8 encoding

fp2 = open('example.txt', 'r')  # Default encoding might differ from the write operation
print(fp2.encoding)  # Shows the current default encoding
content = fp2.read()
```

x??

---
#### Opening Files Correctly
Background context: The text discusses opening files correctly in Python, emphasizing the importance of using the correct encoding to avoid issues. It explains that binary mode should be used only for binary files and not for text files unless you need to determine the encoding.

:p What is the problem with opening a text file in binary mode?
??x
Opening a text file in binary mode can lead to incorrect decoding of characters, especially when dealing with non-ASCII characters. Binary mode treats the file as raw bytes, which may result in garbled text if the file contains special character encodings.
```python
my_file = open('example.txt', 'rb')
```
x??

---
#### Encoding Defaults in Python
Background context: The passage highlights how different operating systems have different default encoding settings. It shows that UTF-8 is used on GNU/Linux and macOS, whereas Windows uses CP1252 as a preferred encoding.

:p What are the encoding defaults for I/O operations in Python?
??x
The encoding defaults vary depending on the operating system:
- On GNU/Linux and macOS: `UTF-8`
- On Windows: `CP1252`

These defaults can be accessed using various methods, such as `locale.getpreferredencoding()`, `sys.stdout.encoding`, etc.
```python
import locale
print(locale.getpreferredencoding())
```
x??

---
#### Text File Reading Example
Background context: The example explains how the default encoding can lead to issues when reading text files. It mentions that using the correct encoding is crucial for displaying characters correctly.

:p Why should you avoid opening a text file in binary mode unless necessary?
??x
Opening a text file in binary mode treats it as raw bytes, and Python will not automatically apply the appropriate encoding to interpret these bytes into readable text. This can result in incorrect character decoding and display issues, especially for non-ASCII characters.

Example of reading a text file incorrectly:
```python
with open('example.txt', 'rb') as my_file:
    content = my_file.read()
print(content)
```
x??

---
#### Locale and Encoding in Different Environments
Background context: The example demonstrates how the `locale.getpreferredencoding()` function returns different values on different operating systems, highlighting the importance of considering the environment when dealing with file encodings.

:p What does `locale.getpreferredencoding()` return?
??x
`locale.getpreferredencoding()` returns the preferred encoding for the current locale. This can vary depending on the operating system and its configuration.
```python
import locale
print(locale.getpreferredencoding())
```
x??

---
#### File Writing in Python
Background context: The text mentions that the `open('dummy', 'w')` method opens a file for writing, but this does not specify the encoding. By default, it uses UTF-8 on most systems.

:p What happens if you do not specify an encoding when opening a file for writing?
??x
When opening a file for writing without specifying an encoding, Python defaults to using UTF-8 encoding. This is generally safe and works well with text files containing non-ASCII characters.
```python
with open('dummy.txt', 'w') as my_file:
    my_file.write('café')
```
x??

---

#### Locale's Preferred Encoding Setting
Locale settings determine how text files are encoded by default. This setting is crucial for handling Unicode characters correctly.

:p What does `locale.getpreferredencoding()` return?
??x
`locale.getpreferredencoding()` returns the preferred encoding for the current locale, which is often used when saving or reading text files without explicit specification of an encoding.
```python
import locale
print(locale.getpreferredencoding())
```
x??

---

#### Console Encoding vs. File Encoding on Windows
Windows has evolved to support UTF-8 encoding both in its console and filesystem since Python 3.6, due to PEPs 528 and 529.

:p How does the encoding of `sys.stdout` change when output is redirected to a file?
??x
When output is redirected to a file, `sys.stdout.isatty()` returns False. In this case, `sys.stdout.encoding` changes to the value returned by `locale.getpreferredencoding()`. For example, it might be 'cp1252' on some Windows machines.
```python
import sys
print(sys.stdout.isatty())  # Should return False if output is redirected
print(sys.stdout.encoding)  # Returns the preferred encoding from locale
```
x??

---

#### Unicode Support in Python and Windows Console
Python’s handling of Unicode improved with changes to `sys.stdout`’s encoding, making it more consistent across different environments.

:p Why might a script that works on the console produce errors when output is redirected?
??x
A script may work correctly on the console because `sys.stdout.isatty()` returns True, and its encoding matches the system's default (UTF-8). However, when output is redirected to a file, `sys.stdout.isatty()` becomes False, and the encoding switches to the locale’s preferred one. This can cause issues if the target file expects UTF-8 but receives a different encoding.
```python
import sys

# Redirecting stdout to a file would change its encoding behavior
with open('output.txt', 'w') as f:
    sys.stdout = f  # Redirect output
    print('\N{INFINITY}')
```
x??

---

#### Example of Encoding Behavior in Scripts
Example 4-12 demonstrates how different characters are handled based on the system's preferred encoding.

:p How does Example 4-12 test character handling?
??x
Example 4-12 tests how the script handles and prints different Unicode characters. It checks if a specific character exists in `cp1252` or `cp437` encodings by attempting to print them.
```python
test_chars = [
    '\N{HORIZONTAL ELLIPSIS} ',  # Exists in cp1252, not in cp437
    '\N{INFINITY} ',             # Exists in cp437, not in cp1252
    '\N{CIRCLED NUMBER FORTY TWO}',  # Not in cp1252 or cp437
]

for char in test_chars:
    print(f'Trying to output {name(char)}:')
    print(char)
```
x??

---

#### Using Unicode Escapes with \N{}
Using `\N{}` for Unicode literals ensures that the correct character is used, even when redirected to a file.

:p What advantage does using `\N{}` have over hexadecimal escapes?
??x
`\N{}` allows specifying the official name of the character, ensuring correctness. If the name doesn't exist, Python raises a `SyntaxError`. This method avoids potential issues with incorrect hex codes.
```python
print('\N{HORIZONTAL ELLIPSIS}')  # Outputs: …
```
x??

---

#### Consistency in Encoding Across Streams
The encoding of `sys.stdout`, `stdin`, and `stderr` changed with Python 3.6 to UTF-8, but redirection can affect this.

:p How do the encodings of different streams change when redirected?
??x
In normal console usage, all three streams (`stdout`, `stdin`, `stderr`) use UTF-8 encoding due to PEPs 528 and 529. However, when output is redirected to a file, only `sys.stdout` changes its behavior based on the locale's preferred encoding.
```python
print(sys.stdin.encoding)  # Always utf-8 in Python 3.6+
print(sys.stdout.encoding)  # May change if stdout is redirected
print(sys.stderr.encoding)  # Always utf-8 in Python 3.6+
```
x??

---

#### Code Page and Encoding Conflicts
Background context: The text discusses how different code pages (e.g., 437, 1252) affect character representation and display. It highlights issues with encoding when using the console or redirecting output to a file.

:p What is the issue described by the character "CIRCLED NUMBER FOR TWO" being replaced by a rectangle in the console?

??x
The issue arises because the active code page (437) does not have a glyph for the CIRCLED NUMBER FOR TWO. When this character is encountered, the console font fails to display it properly and replaces it with a placeholder like a rectangle.

```python
# Example Python code showing encoding issues
import sys

def print_special_chars():
    # Attempting to print characters from different code pages
    cp437_char = chr(0x85)  # Code page 437 character 'à'
    infinity_symbol = '\u221e'  # Unicode for infinity symbol
    
    print(cp437_char, file=sys.stdout)  # Expected to be 'à' but might not display correctly
    print(infinity_symbol, file=sys.stdout)  # Expected to be ∞

print_special_chars()
```
x??

---

#### Default Encoding and Redirection
Background context: The text explains that when output is redirected to a file (e.g., stdout_check.py), the encoding can change from UTF-8 to something else like 'cp1252', leading to issues with characters not being displayed correctly.

:p What happens when you redirect `stdout_check.py` to a file?

??x
When `stdout_check.py` is redirected to a file, it might use an encoding other than UTF-8 (like 'cp1252'). This can cause issues because the file uses a different character mapping for certain byte values. As a result, characters that are valid in one code page may appear incorrectly or not at all when read with another application.

For example:
```python
import sys

def redirect_output():
    # Redirecting output to a file
    with open('out.txt', 'w') as f:
        print('\u221e', file=f)  # Infinity symbol in Unicode

redirect_output()
```

When reading `out.txt` using Windows editors or the command line, characters may be displayed incorrectly due to encoding differences.

```bash
type out.txt
```
x??

---

#### Environment Variables and Encoding Settings
Background context: The text describes how environment variables like `PYTHONIOENCODING` affect the default encoding for standard I/O. It also mentions that Python 3.6+ ignores this variable unless `PYTHONLEGACYWINDOWSSTDIO` is set.

:p How does setting `PYTHONIOENCODING` and `PYTHONLEGACYWINDOWSSTDIO` influence encoding settings?

??x
Setting `PYTHONIOENCODING` can override the default encoding for standard I/O in Python 3.6+ if `PYTHONLEGACYWINDOWSSTDIO` is not empty. If this environment variable is set, it specifies the desired encoding and errors handling (e.g., 'strict', 'ignore', 'replace').

```python
import os

def check_io_encoding():
    # Setting PYTHONIOENCODING to override default encoding
    os.environ['PYTHONIOENCODING'] = 'utf-8:strict'
    
    print('This text should be displayed correctly with UTF-8 encoding')

check_io_encoding()
```

If `PYTHONLEGACYWINDOWSSTDIO` is set, it reverts the behavior of Python 3.6+ to use `PYTHONIOENCODING`. If this variable is not set (empty), standard I/O defaults to 'UTF-8' for interactive I/O and to a system default encoding when output/input is redirected.

```python
import os

def check_legacy_io_encoding():
    # Setting PYTHONLEGACYWINDOWSSTDIO to revert to previous behavior in Python 3.6+
    os.environ['PYTHONLEGACYWINDOWSSTDIO'] = '1'
    
    print('This text should be displayed correctly with system default encoding')

check_legacy_io_encoding()
```
x??

---

#### File Encodings and Locale Preferences
Background context: The text explains how the preferred encoding is determined by `locale.getpreferredencoding()`. This setting can vary based on the operating environment (e.g., US vs Brazil).

:p How does `locale.getpreferredencoding()` determine the default file encoding?

??x
`locale.getpreferredencoding()` returns the preferred locale-specific encoding for file operations. In Windows, this is often 'cp1252', but it can differ depending on the system's regional settings.

```python
import locale

def check_file_encoding():
    # Getting the preferred encoding for file operations
    preferred_encoding = locale.getpreferredencoding()
    print(f"Preferred Encoding: {preferred_encoding}")

check_file_encoding()
```

This function retrieves and prints the default encoding used by Python when opening files without specifying an encoding explicitly.

```python
# Example of opening a file with and without specifying encoding
def open_files():
    # Without specifying encoding (uses getpreferredencoding())
    with open('example.txt', 'r') as f:
        print(f.read())
    
    # With explicit UTF-8 encoding
    with open('example.txt', 'r', encoding='utf-8') as f:
        print(f.read())

open_files()
```
x??

---

#### Unicode and Character Encodings
Background context: Understanding character encodings is crucial for handling text data, especially when working with non-ASCII characters. On Windows, different code pages like 'cp850' or 'cp1252' are commonly used. These support only ASCII with 127 additional characters that vary between encodings.

On Unix-based systems (GNU/Linux and MacOS), the default encoding is UTF-8 for several years. This means I/O handles all Unicode characters, making it easier to work with text data in these environments.
:p What are some key differences between Windows and Unix-based systems regarding character encodings?
??x
Windows uses different code pages like 'cp850' or 'cp1252', which support only ASCII with 127 additional characters that vary between encodings, whereas Unix-based systems default to UTF-8, supporting all Unicode characters.
x??

---

#### `locale.getpreferredencoding()`
Background context: The function `locale.getpreferredencoding()` returns the encoding used for text data according to user preferences. However, these preferences may not be available programmatically on some systems and are only a guess.

Using this function as a default can lead to encoding errors if the system's actual encoding settings differ from the guessed one.
:p What does `locale.getpreferredencoding()` return, and why should it not be relied upon?
??x
`locale.getpreferredencoding()` returns an encoding based on user preferences. However, these preferences are not always available programmatically and might only return a guess. Therefore, relying solely on this function can lead to encoding errors.
x??

---

#### String Comparisons in Unicode
Background context: Unicode introduces combining characters like diacritics that can complicate string comparisons. For example, "café" can be represented as two or five code points but appear visually identical.

Python treats these sequences differently when comparing strings, leading to unexpected results unless normalized.
:p How do combining characters affect string comparisons in Python?
??x
Combining characters like diacritics can lead to different representations of the same text. For example, "café" can be represented as 'cafe\u0301' (4 code points) or 'café' (5 code points). When comparing these strings directly in Python, they are not equal because Python sees them as distinct sequences of code points.
x??

---

#### Normalizing Unicode with `unicodedata.normalize()`
Background context: To ensure reliable comparisons and sorting, normalizing Unicode text is essential. The `unicodedata.normalize()` function can be used to convert text into a uniform representation.

Normalization forms 'NFC' and 'NFD' are commonly used:
- NFC composes code points to produce the shortest equivalent string.
- NFD decomposes composed characters into base characters and separate combining characters.

Using these forms makes comparisons consistent, as demonstrated in the example.
:p How do normalization forms 'NFC' and 'NFD' differ, and why are they useful?
??x
Normalization form NFC composes code points to produce the shortest equivalent string. NFD decomposes composed characters into base characters and separate combining characters.

These forms are useful because they ensure consistent representations of text, making comparisons reliable. For example, 'cafe\u0301' (NFD) becomes 'café' (NFC), ensuring that these strings compare as equal.
x??

---

#### Other Normalization Forms: NFKC and NFD
Background context: In addition to NFC and NFD, there are two other normalization forms:
- NFKC and NFD which handle compatibility characters.

NFKC and NFD replace compatibility characters with a preferred representation, potentially losing some formatting information but providing more consistent results for searching and indexing.
:p What are the differences between NFKC and NFD in handling compatibility characters?
??x
NFKC (Normalization Form KC) and NFD (Normalization Form KD) handle compatibility characters differently:
- NFKC replaces each compatibility character with a "compatibility decomposition" of one or more characters that have a preferred representation.
- This process can lose some formatting information, as the goal is to provide a consistent canonical form.

For example, '½' (U+00BD VULGAR FRACTION ONE HALF) decomposes into '1⁄2' and 'µ' (U+00B5 MICRO SIGN) decomposes into 'μ' (U+03BC GREEK SMALL LETTER MU).
x??

---

