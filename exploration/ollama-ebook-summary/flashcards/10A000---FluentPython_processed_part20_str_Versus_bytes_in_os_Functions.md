# Flashcards: 10A000---FluentPython_processed (Part 20)

**Starting Chapter:** str Versus bytes in os Functions

---

#### Regular Expressions on Strings and Bytes
Background context: This concept discusses how regular expressions can be used with both `str` (string) and `bytes` data types. It highlights that `str` regular expressions treat non-ASCII bytes as nondigit/nonword characters, whereas the `re.ASCII` flag makes them perform ASCII-only matching.

:p What is the difference between using regular expressions on strings versus bytes?
??x
When using regular expressions with `str`, non-ASCII bytes are treated as nondigit and nonword characters. However, by adding the `re.ASCII` flag, these patterns will only match ASCII characters. In contrast, when used with `bytes`, the default behavior matches any character, including those outside the ASCII range.

For example:
```python
import re

# Using str
pattern = b'\d'  # Matches digit bytes
print(re.findall(pattern, b'1234π'))  # Output: [b'1', b'2', b'3', b'4']

# Using re.ASCII flag for str
pattern_str = r'\w'
print(re.findall(pattern_str, 'abc.txtπ.txt', flags=re.ASCII))  # Output: ['a', 'b', 'c']
```
x??

---

#### os Module and File Path Handling
Background context: The `os` module in Python provides functions that accept filenames or pathnames as either `str` (string) or `bytes`. This is necessary because the underlying operating system may use non-Unicode filenames. The `os` module automatically converts strings to bytes using the filesystem encoding, and vice versa.

:p What does the `os.listdir` function do when given a directory path?
??x
The `os.listdir` function lists all entries in the specified directory (both files and subdirectories). If a string path is provided, it will be encoded to bytes using the filesystem's encoding. Conversely, if bytes are passed, they will remain as bytes.

For example:
```python
import os

# Using str
print(os.listdir('.'))  # Output: ['abc.txt', 'digits-of-π.txt']

# Using bytes
print(os.listdir(b'.'))
# Output: [b'abc.txt', b'digits-of-\xcf\x80.txt']
```
x??

---

#### fsencode and fsdecode Functions in os Module
Background context: The `os` module includes functions like `os.fsencode` and `os.fsdecode` to help handle filenames or pathnames as either `str` or `bytes`. These functions are useful for manual handling of file paths, especially when dealing with non-Unicode filenames.

:p What do the `os.fsencode` and `os.fsdecode` functions do?
??x
The `os.fsencode` function encodes a filename or path into bytes using the filesystem's encoding. The `os.fsdecode` function decodes byte-encoded paths back to strings, ensuring that file paths are handled correctly regardless of their original format.

For example:
```python
import os

# Encoding a string to bytes
path_str = 'abc.txt'
encoded_path = os.fsencode(path_str)
print(encoded_path)  # Output: b'abc.txt'

# Decoding bytes back to string
decoded_path = os.fsdecode(encoded_path)
print(decoded_path)  # Output: abc.txt
```
x??

---

#### Concept: Character Encoding and Text Strings

Background context explaining that in Python 3, text strings are separate from binary sequences, unlike in earlier versions of Python where `str` was essentially a byte sequence. Unicode introduced the concept of encoding text into bytes.

:p What is the main difference between `str` and `bytes` in Python 3?

??x
In Python 3, `str` represents text strings using Unicode characters, while `bytes` represent binary data as sequences of bytes. The key distinction is that `str` objects are immutable and represented internally by their code points (Unicode characters), whereas `bytes` are mutable and contain raw byte values.

```python
# Example showing the difference between str and bytes

text = "Hello, world!"
binary_data = b"Hello, binary!"

print(type(text))  # <class 'str'>
print(type(binary_data))  # <class 'bytes'>
```

x??

---

#### Concept: Binary Sequence Data Types

Background context explaining that `bytes`, `bytearray`, and `memoryview` are used to handle binary data. Each type has its use cases, with `bytes` being immutable and `bytearray` being mutable.

:p What is the difference between `bytes` and `bytearray`?

??x
The key difference lies in mutability:
- `bytes` objects are immutable sequences of integers (8-bit values).
- `bytearray` objects are mutable sequences of integers, similar to lists but with elements restricted to 0 ≤ x < 256.

```python
# Example showing the difference between bytes and bytearray

b = b"Hello"
ba = bytearray(b)

print(type(b))  # <class 'bytes'>
print(type(ba))  # <class 'bytearray'>

# Modifying bytearray
ba[0] = ord('h')

print(b)  # b'Hello'
print(ba)  # bytearray(b'hello')
```

x??

---

#### Concept: Encoding and Decoding

Background context explaining that encoding converts text into bytes, while decoding converts bytes back into text. Important codecs such as UTF-8, UTF-16, and ASCII are covered.

:p What is the process of encoding a string to bytes in Python?

??x
The process involves converting a `str` object (text) to a sequence of bytes using an encoding scheme. In Python 3, this is done with the `.encode()` method on strings.

```python
# Example of encoding

text = "Hello, world!"
encoded_bytes = text.encode('utf-8')

print(encoded_bytes)  # b'Hello,\x2c \xe2\x80\x98orld!'
```

Here, `utf-8` is the codec used to convert the string into bytes. The encoded bytes can be decoded back to a string using `.decode()`.

```python
# Example of decoding

decoded_text = encoded_bytes.decode('utf-8')
print(decoded_text)  # Hello, world!
```

x??

---

#### Concept: Handling Encoding Errors

Background context explaining common encoding errors like `UnicodeEncodeError` and `UnicodeDecodeError`. The chapter discusses how to handle these errors using different strategies.

:p What are some common ways to prevent `UnicodeEncodeError`?

??x
To prevent `UnicodeEncodeError`, you can use the `.encode()` method with an error handler such as `'backslashreplace'`, `'ignore'`, or `'xmlcharrefreplace'`. These handlers provide alternatives when a specific character cannot be encoded.

```python
# Example of handling UnicodeEncodeError

try:
    text = "Hello, 你好！"
    print(text.encode('ascii'))  # Raises UnicodeEncodeError
except UnicodeEncodeError as e:
    print(e)
    encoded_text = text.encode('ascii', 'backslashreplace')
    print(encoded_text)  # b'Hello, \x4e16\x529b!'
```

x??

---

#### Concept: Opening Text Files with Specified Encoding

Background context explaining that opening a text file in Python without specifying the encoding can lead to `UnicodeDecodeError`. The chapter emphasizes the importance of explicitly setting the encoding.

:p What is the potential consequence of not specifying the encoding when opening a text file?

??x
If you do not specify the encoding when opening a text file, Python uses its default encoding (which varies by operating system), which can lead to `UnicodeDecodeError` if the file's encoding does not match. For example, on Windows, the default encoding is often 'cp1252', while on Unix-based systems, it is usually 'utf-8'.

```python
# Example of opening a text file without specifying encoding

try:
    with open('example.txt') as f:  # No explicit encoding specified
        print(f.read())
except UnicodeDecodeError as e:
    print(e)
```

To avoid this issue, always specify the correct encoding when opening files.

```python
# Example of correctly specifying encoding

with open('example.txt', 'r', encoding='utf-8') as f:  # Explicitly specify encoding
    print(f.read())
```

x??

---

#### Concept: Default Encoding Settings in Python

Background context explaining that default encoding settings vary by operating system and can be detected using the `sys` module.

:p How can you determine the current default encoding used in Python?

??x
You can use the `sys` module to find out the current default encoding:

```python
import sys

print(sys.getdefaultencoding())  # Output depends on OS and Python version
```

On Windows, this typically outputs `'cp1252'`, while on Unix-based systems like GNU/Linux or macOS, it often returns `'utf-8'`.

x??

---

#### Concept: Normalization in Unicode Text

Background context explaining that normalization is essential for text matching because Unicode can represent the same character in multiple ways (e.g., composed vs. decomposed forms). The chapter covers normalization and case folding.

:p What is normalization in the context of Unicode?

??x
Normalization ensures consistent representation of characters, especially important for text matching. It involves converting characters to a standard form, such as combining marks being replaced with precomposed characters or different ways of representing the same character being unified.

```python
import unicodedata

text = "café"  # Contains a decomposed 'e' with a combining acute accent
print(unicodedata.normalize('NFC', text))  # Precomposed form: café
```

Normalization forms include NFC (compatibility decomposition followed by canonical composition), NFD, NFKC, and NFKD.

x??

---

#### Unicode and Python

Background context: The passage discusses various resources and tools for understanding and working with Unicode in Python. It highlights the importance of character encoding, the challenges faced when migrating from Python 2 to Python 3, and the need for a comprehensive understanding of Unicode.

:p What are some key differences between Python 2 and Python 3 related to handling text and bytes?
??x
The main differences include that `str` in Python 2 corresponds to bytes, while `str` in Python 3 represents text (Unicode), necessitating the use of `bytes` for raw binary data. This change affects how strings are handled and processed.

```python
# Example Code in Python 2
text = u'Hello, world!'
binary_data = text.encode('utf-8')

# Example Code in Python 3
text = 'Hello, world!'
binary_data = text.encode('utf-8')
```
x??

---

#### PyUCA and External Packages

Background context: The passage mentions the use of external packages like `PyUCA` to handle Unicode data more effectively. It emphasizes leveraging the Unicode database for complex operations such as searching characters by name.

:p What is the purpose of the `PyUCA` package in Python?
??x
The `PyUCA` package provides a way to sort and compare strings according to their Unicode collation rules, which are essential for locale-specific sorting. This can be particularly useful when dealing with non-ASCII characters or ensuring proper sorting across different languages.

```python
from pyuca import Collator

text = ['apple', 'banana', 'Äpfel', 'Orange']
collator = Collator('de_DE')
sorted_text = sorted(text, key=collator.sort_key)
```
x??

---

#### Character Encoding and Text Handling

Background context: The passage emphasizes the importance of understanding that "Humans use text. Computers speak bytes." This highlights the distinction between human-readable text (text) and machine-readable binary data (bytes).

:p How do you write a byte sequence to a text file in Python 3?
??x
In Python 3, you need to explicitly handle the conversion from text to bytes before writing to a file. The `encode()` method is used to convert Unicode strings to bytes.

```python
text = 'Hello, world!'
binary_data = text.encode('utf-8')
with open('output.txt', 'wb') as file:
    file.write(binary_data)
```
x??

---

#### Lennart Regebro's UMMU

Background context: The passage introduces Lennart Regebro's "Useful Mental Model of Unicode" (UMMU) as a helpful tool for understanding the complexities of Unicode. This model aids in grasping how characters are represented and processed.

:p What is the key idea behind Lennart Regebro’s UMMU?
??x
The core concept of the UMMU is to view Unicode as a mapping from code points to abstract characters, which can then be transformed into concrete representations for display or storage. This helps in understanding how different systems handle and process Unicode characters.

```python
# Example with UMMU in mind
char = 'A'
code_point = ord(char)  # Get the Unicode code point of 'A'
print(f"The code point for {char} is {code_point}")
```
x??

---

#### PyCon Talks on Unicode

Background context: The passage references several notable talks and resources, such as Ned Batchelder’s “Pragmatic Unicode” and Esther Nam and Travis Fischer’s talk on character encoding. These discussions provide valuable insights into practical approaches to working with Unicode in Python.

:p What is the key takeaway from Ned Batchelder’s 2012 PyCon US talk?
??x
The main takeaway from Ned's talk is that handling text in Python involves understanding that `str` and `bytes` are distinct types, with `str` representing text (Unicode) and `bytes` representing binary data. He also emphasizes the importance of using proper encoding when dealing with text.

```python
# Example Code
text = 'Hello, world!'
binary_data = text.encode('utf-8')
decoded_text = binary_data.decode('utf-8')
```
x??

---

#### Standard Encodings in Python
Background context: The text mentions that a list of encodings supported by Python is available at the `Standard Encodings` section in the `codecs` module documentation. This list can be retrieved programmatically using the script `/Tools/unicode/listcodecs.py` which comes with the CPython source code.
:p What command or script allows you to get a list of standard encodings supported by Python?
??x
You can use the following Python code snippet to retrieve and print a list of standard encodings from the `codecs` module:
```python
import codecs

# Retrieve available encodings
encodings = codecs.lookup('utf-8').name  # This is just an example, you need to loop through all encodings
for encoding in encodings:
    print(encoding)
```
x??

---

#### Programming with Unicode by Victor Stinner
Background context: The text mentions that "Programming with Unicode" by Victor Stinner is a free book covering Unicode concepts and tools across different operating systems and programming languages, including Python.
:p What does the book "Programming with Unicode" cover?
??x
The book covers general Unicode concepts as well as tools and APIs for working with Unicode in various operating systems and programming environments, including Python. It provides insights into how to handle text data using Unicode, including discussions on normalization forms and encoding/decoding mechanisms.
x??

---

#### Case Folding: An Introduction
Background context: The W3C page "Case Folding: An Introduction" is mentioned as a gentle introduction to the concept of case folding in Unicode. This is useful for understanding how different cases (lowercase, uppercase) of characters are treated and converted within Unicode text processing.
:p What does the term "case folding" refer to?
??x
Case folding refers to the process of converting all letters in a string to either lowercase or uppercase according to specific rules defined by the Unicode standard. This is useful for comparing strings without regard to case differences, ensuring that "Hello" and "hello" are considered equivalent.
x??

---

#### NFC FAQ by Mark Davis
Background context: The text mentions that Mark Davis' NFC FAQ, which covers normalization forms in detail, is a good resource for understanding how Unicode characters should be normalized for consistency and correctness.
:p What information can you find in the NFC FAQ?
??x
The NFC FAQ by Mark Davis provides detailed information on Unicode normalization forms, specifically focusing on Normalization Form C (NFC). It explains how to normalize text to ensure that all equivalent sequences of characters are represented using the same form. This is crucial for maintaining consistent and predictable behavior when working with Unicode text.
x??

---

#### The Original Emoji
Background context: The text mentions that The Museum of Modern Art added "The Original Emoji" to its collection, showcasing 176 emojis designed by Shigetaka Kurita in 1999. This provides historical context about the early development and design of emojis.
:p What is "The Original Emoji"?
??x
"The Original Emoji" refers to a set of 176 emojis created by Shigetaka Kurita in 1999 for NTT DOCOMO, a Japanese mobile carrier. This collection represents one of the earliest known sets of emojis and provides historical context about the early development and design of these icons.
x??

---

#### Emojitracker
Background context: The text mentions that Matthew Rothenberg's emojitracker.com is a live dashboard showing counts of emoji usage on Twitter, updated in real time. This can be useful for tracking trends and popular emojis over time.
:p What does the emojitracker show?
??x
The emojitracker shows real-time counts of emoji usage on Twitter. It provides data on which emojis are currently trending or being used most frequently at any given moment, allowing users to monitor the popularity of different emojis in near-real-time.
x??

---

#### Non-ASCII Names in Source Code
Background context: The text discusses Python 3's support for non-ASCII identifiers in source code and argues that using such names can make the code more readable for its intended audience. It also mentions that leaving out accents could negatively impact readability if the code is meant to be used by a broader audience.
:p What are the arguments for using non-ASCII names in Python source code?
??x
The arguments for using non-ASCII names in Python source code include making the code more readable and easier to write for developers whose native language uses such characters. However, it's important to consider the intended audience of the code: if the code is meant for a multinational corporation or open-source projects, sticking with ASCII identifiers might be better for broader compatibility.
x??

#### Plain Text Definition and Characteristics
Background context explaining that plain text refers to computer-encoded text consisting only of a sequence of code points from a given standard, with no other formatting or structural information. However, this definition can be disputed as HTML is an example of plain text carrying some formatting.

:p What is the primary characteristic distinguishing plain text according to its definition?
??x
Plain text typically consists solely of characters represented by their Unicode code points, and no bytes have non-text meanings. This means that in a plain text file like AsciiDoc, numbers are represented as sequences of digit characters rather than binary values.
x??

---

#### HTML as Plain Text
Background context explaining that despite carrying formatting information, HTML can still be considered plain text since every byte represents a character or structural tag.

:p Why is HTML classified as plain text?
??x
HTML files contain tags and content in such a way that they represent text characters. The structure and formatting are part of the text itself, with all bytes representing either character data or structural information.
x??

---

#### Unicode and AsciiDoc Usage
Background context explaining how AsciiDoc is used for writing books but is actually UTF-8 encoded, not ASCII.

:p How does AsciiDoc relate to Unicode?
??x
AsciiDoc uses UTF-8 encoding, allowing it to include a wide range of characters beyond the limitations of ASCII. This makes it more versatile for modern text processing.
x??

---

#### Representation of Characters in RAM (str)
Background context explaining how Python 3 stores strings as sequences of code points using variable-length memory layouts.

:p How does Python 3 store strings in memory?
??x
Python 3 stores strings using a flexible representation, where each string is stored with the minimum number of bytes per code point to allow efficient direct access. Python checks characters and chooses the most economical memory layout based on the range of characters used.
x??

---

#### Example of Memory Usage for Strings
Background context explaining how certain Unicode characters can significantly increase the memory usage of strings.

:p How does the presence of specific Unicode characters affect string memory usage in Python?
??x
The presence of certain Unicode characters, like the RA T (U+1F400), can inflate the memory usage because they require more bytes per character. For instance, a single RA T character will make an otherwise all-ASCII text use 4 bytes per character instead of just one.
x??

---

#### Summary
This set of flashcards covers key concepts like plain text definitions, Unicode encoding, and Python's flexible string representation in memory. Each card provides context and explanations to help with understanding rather than pure memorization.

#### Unicode Character Retrieval and Slicing Issues
Background context explaining the challenges associated with retrieving characters by position and slicing Unicode text. These issues are exacerbated as emojis become more popular, leading to problems like mojibake.
:p How do retrieval of arbitrary characters and slicing of Unicode text pose challenges?
??x
Retrieving an arbitrary character by position is considered overrated due to the complex ways in which Unicode characters can combine. Slicing from Unicode text is often wrong and can produce mojibake, especially as emojis gain popularity.
```python
# Example of incorrect slicing
text = "😊👍"
slice_result = text[1:3]
print(slice_result)  # This might not give the expected result due to emoji encoding issues
```
x??

---

#### Python 2.6 and 2.7 Byte Types
Background context explaining that in Python 2.6 and 2.7, `bytes` was just an alias for `str`. This is relevant when discussing binary data handling.
:p What is the relationship between `bytes` and `str` in Python versions 2.6 and 2.7?
??x
In Python 2.6 and 2.7, `bytes` was essentially a synonym for `str`, meaning that both types represented sequences of bytes without any distinction between text and binary data.
```python
# Example of alias relationship
text = b'hello'
print(type(text))  # <class 'str'>
```
x??

---

#### Unicode Delimiters in Python
Background context explaining the use of ASCII characters as delimiters in Python, with a specific focus on the apostrophe character. Provide information about its Unicode representation.
:p What is the default string delimiter used by Python and what is its Unicode name?
??x
The default string delimiter in Python is the ASCII “single quote” character (U+0027), which is named APOSTROPHE in the Unicode standard. In contrast, the real single quotes are asymmetric: left ('U+2018) and right ('U+2019).
```python
# Example of string delimiter usage
text = 'hello'
print(ord("'"))  # Output: 39 (ASCII code for apostrophe)
```
x??

---

#### Byte-to-String Conversion in Python 3.0 to 3.4
Background context explaining the issues with byte-to-string conversion in Python versions 3.0 to 3.4, causing pain among developers dealing with binary data.
:p What problems did developers face due to internal conversions from bytes to str in Python 3.0 to 3.4?
??x
Developers faced significant difficulties due to the lack of proper byte-to-string conversion methods in Python versions 3.0 to 3.4, which led to numerous issues when handling binary data. This problem was addressed and documented in PEP 461, which introduced `%` formatting for `bytes` and `bytearray`.
```python
# Example of incorrect usage before PEP 461
data = b'\x01\x02'
result = "%s" % data  # Incorrect string conversion
```
x??

---

#### "Unicode Sandwich" Term Origin
Background context explaining the origin of the term “Unicode sandwich” and its relevance to handling strings and bytes in Python.
:p Where did the term “Unicode sandwich” originate, and what does it refer to?
??x
The term “Unicode sandwich” was first used by Ned Batchelder in his talk "Pragmatic Unicode" at US PyCon 2012. It refers to the situation where developers are forced to handle both text (strings) and binary data (bytes) within the same program, creating a complex layering or “sandwich” of data types.
```python
# Example of a "Unicode sandwich"
text = 'hello'
binary_data = b'data'
combined = text + binary_data  # Potential issues if not handled correctly
```
x??

---

#### Windows Command-Line and Unicode Output
Background context explaining the challenges with Unicode output in Windows command-line environments, particularly regarding UTF-8 encoding.
:p What are the common issues with Unicode and UTF-8 output in the Windows command line?
??x
Common issues include improper rendering of Unicode characters when using UTF-8 encoding. For example, the CIRCLED NUMBER FOR TY TWO character might display incorrectly as a black circular outline with the number 42 inside.
```python
# Example of potential rendering issue
print(chr(0x3242))  # Circled Number Two, U+3242
```
x??

---

#### Python 2 `sys.setdefaultencoding` Function
Background context explaining why the `sys.setdefaultencoding` function was misused in Python 2 and its removal in Python 3.
:p Why is the `sys.setdefaultencoding` function no longer documented or used in Python 3?
??x
The `sys.setdefaultencoding` function was misused by developers to set a default encoding for Python, which led to several issues. In Python 3, it was removed and is not documented because CPython now uses 'utf-8' as the default encoding. Calling this function directly in user code is strongly discouraged.
```python
# Example of misuse in Python 2
import sys
sys.setdefaultencoding('ascii')  # Incorrect usage
```
x??

---

#### Diacritics and Sorting in Unicode
Background context explaining how diacritics affect sorting in Unicode, underlining the rare occurrence of their impact on sorting.
:p How do diacritics influence sorting in Unicode?
??x
Diacritics generally do not affect sorting unless they are the only difference between two words. In such cases, the word with a diacritic is sorted after the plain word.
```python
# Example of sorting behavior
words = ['café', 'cafe']
sorted_words = sorted(words)
print(sorted_words)  # Output: ['cafe', 'café']
```
x??

---

#### Micro Sign and Ohm Symbol in Unicode
Background context explaining the handling of specific characters like micro sign and ohm symbol under NFC, NFKC, and NFKD normalization.
:p How are the micro sign (µ) and ohm symbol (∩) handled by different normalization forms?
??x
The micro sign (µ) is treated as a compatibility character, while the ohm symbol (∩) is not. As a result, NFC does not change the micro sign but converts the ohm symbol to the capital omega (Ω). NFKC and NFKD further normalize both characters into Greek characters.
```python
# Example of normalization behavior
import unicodedata

micro = '\u00b5'  # Micro sign
ohm = '\u2126'    # Ohm symbol

print(unicodedata.normalize('NFC', micro))  # Output: µ
print(unicodedata.normalize('NFKC', ohm))  # Output: Ω
```
x??

---

