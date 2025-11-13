# Flashcards: 10A000---FluentPython_processed (Part 16)

**Starting Chapter:** Understanding EncodeDecode Problems. Coping with UnicodeEncodeError

---

---
#### Surrogate Pairs and UTF-16 Encoding
UTF-16 superseded UCS-2 in 1996, supporting more than just U+FFFF. This change was necessary to accommodate modern characters like emojis.

:p What is surrogate pairs?
??x
Surrogate pairs are a method used by Unicode when a single code point exceeds the original 16-bit limit of UCS-2, allowing for encoding of characters above U+FFFF using two 16-bit code units.
x??

---
#### UTF-16 vs. UCS-2
UTF-16 was introduced to handle more than just U+FFFF, while UCS-2 only supports up to this point. Many systems still use UCS-2 despite its limitations.

:p Why was UTF-16 introduced?
??x
UTF-16 was introduced to support a larger range of Unicode characters beyond the 16-bit limit of UCS-2, ensuring compatibility with modern characters and emojis.
x??

---
#### Handling Encoding Errors in Python
Python provides specific exceptions for encoding and decoding errors: `UnicodeEncodeError` and `UnicodeDecodeError`. These are part of the broader `UnicodeError`.

:p What types of exceptions does Python raise when dealing with Unicode issues?
??x
Python raises `UnicodeEncodeError` during text-to-binary sequence conversion, and `UnicodeDecodeError` during binary-to-text sequence reading. Other errors might include a `SyntaxError` if the source encoding is unexpected.
x??

---
#### Handling `UnicodeEncodeError`
When converting to bytes using codecs that do not support all Unicode characters, Python raises a `UnicodeEncodeError`. Special handling can be provided with an `errors` argument.

:p What are error handlers in encoding and their behavior?
??x
Error handlers include:
- 'ignore': Skips unencodable characters.
- 'replace': Replaces unencodable characters with '?'.
- 'xmlcharrefreplace': Replaces unencodable characters with XML character references, e.g., '&#227;'.
:x??

---
#### Example of Encoding Errors

```python
city = 'São Paulo'
print(city.encode('utf_8'))     # b'S\xc3\xa3o Paulo'
print(city.encode('utf_16'))    # b'\xff\xfeS\x00\xe3\x00o\x00 \x00P\x00a\x00u\x00l\x00o\x00'
print(city.encode('iso8859_1'))# b'S\xc3\xa3o Paulo'
try:
    print(city.encode('cp437'))
except UnicodeEncodeError as e:
    print(e)                      # 'charmap' codec can't encode character '\xe3' in position 1: character maps to <undefined>
print(city.encode('cp437', errors='ignore'))   # b'So Paulo'
print(city.encode('cp437', errors='replace'))  # b'S?o Paulo'
print(city.encode('cp437', errors='xmlcharrefreplace'))# b'S&#227;o Paulo'
```

:p What does this code demonstrate?
??x
This code demonstrates encoding a string using different codecs and error handlers. It shows how `UnicodeEncodeError` is raised for unsupported characters and how to handle such errors.
x??

---

#### XML Character Replacement Mechanism
XML characters that cannot be encoded are replaced by an XML entity. This is useful when you can't use UTF-8 and can't afford to lose data.

:p What does 'xmlcharrefreplace' do?
??x
The 'xmlcharrefreplace' error handling mechanism replaces unencodable characters with their corresponding XML entities, ensuring no data loss while encoding. It's a fallback option if other encodings fail.
x??

---

#### Error Handling in Python Encodings
Python provides flexible error handling for codecs, allowing you to register custom functions for different errors.

:p How can one add custom error handlers for codecs?
??x
You can use the `codecs.register_error` function to register a name and an error handling function. This allows you to define custom behavior when encoding or decoding fails.
```python
import codecs

def my_custom_handler(exc):
    # Custom logic here
    return ('replacement', len(exc.object))

codecs.register_error('my_custom_error', my_custom_handler)
```
x??

---

#### ASCII as a Subset of Encodings
ASCII is a common subset that all encodings support, meaning that encoding text to ASCII will always work if the text contains only ASCII characters.

:p Why should you prefer using ASCII for encoding?
??x
Using ASCII ensures compatibility with many systems and simplifies error handling. Since ASCII includes only the first 128 Unicode code points, any text made exclusively of these characters can be safely encoded without issues.
x??

---

#### Python's str.isascii() Method
Python 3.7 introduced `str.isascii()` to check if a string contains only ASCII characters.

:p How does `str.isascii()` work?
??x
The `str.isascii()` method checks if all the characters in a string are ASCII (i.e., within the range of \u0000-\u007F). It returns `True` if the string is 100% pure ASCII, and `False` otherwise.
```python
>>> s = "Hello, World!"
>>> s.isascii()
True

>>> t = "Café"
>>> t.isascii()
False
```
x??

---

#### Dealing with UnicodeDecodeErrors
UnicodeDecodeErrors occur when byte sequences are not valid for the assumed encoding.

:p What causes a `UnicodeDecodeError`?
??x
A `UnicodeDecodeError` is raised when byte sequences are encountered that cannot be decoded using the specified encoding. For instance, trying to decode non-UTF-8 data as UTF-8 will result in this error.
```python
octets = b'Montr\xe9al'
octets.decode('utf_8')  # Raises UnicodeDecodeError
```
x??

---

#### Using Error Handling with Decoding
You can handle `UnicodeDecodeError` by specifying the 'replace' option, which replaces unencodable characters with a placeholder.

:p How does using 'replace' error handling work?
??x
Using 'replace' as an error handler in decoding operations will replace any unencodable character with the Unicode REPLACEMENT CHARACTER (U+FFFD). This ensures that no data is lost.
```python
octets = b'Montr\xe9al'
decoded = octets.decode('utf_8', errors='replace')
print(decoded)  # Output: Montr � al
```
x??

---

#### SyntaxError When Loading Modules with Unexpected Encoding

:p What happens if you load a .py module containing non-UTF-8 data without an encoding declaration?
??x
If you load a Python module that contains non-UTF-8 data and has no encoding declaration, Python will raise a `SyntaxError` because it assumes the source code is UTF-8. To fix this, add a magic comment like `# coding: cp1252` at the top of your file.
x??

---

#### Detecting Encoding of Byte Sequences
Detecting the exact encoding of byte sequences can be challenging as Python does not automatically determine it.

:p How do you detect the encoding of an unknown byte sequence?
??x
Unfortunately, there is no direct way to detect the encoding of a byte sequence without prior knowledge or hints. Some protocols like HTTP and XML provide headers that explicitly state the encoding.
However, if you know certain characteristics of the data (e.g., frequent null bytes suggest 16-bit encodings), heuristics can sometimes help in guessing the correct encoding.
x??

---

#### Guessing UTF-8 Encoding
UTF-8 is designed such that random byte sequences are unlikely to decode correctly as garbage.

:p Why is it difficult for a sequence of bytes to be decoded as garbage when using UTF-8?
??x
The design of UTF-8 makes accidental decoding as garbage very unlikely. This is because:
1. UTF-8 escape sequences never use ASCII characters.
2. The bit patterns in UTF-8 sequences make it improbable for random data to accidentally decode as valid UTF-8.

Therefore, if you can decode a byte sequence with codes > 127 as UTF-8 without errors, it's likely that the encoding is indeed UTF-8.
x??

---

#### BOM (Byte Order Mark)
Background context explaining the BOM and its role in UTF-16 encoding. The BOM is a special invisible character used to denote the byte ordering of the encoded text, typically U+FEFF but also can be b'\xff\xfe' for little-endian systems.

:p What is the Byte Order Mark (BOM) and why is it important in UTF-16 encoding?
??x
The BOM is a special invisible character used to denote the byte ordering of the encoded text. For example, in little-endian systems like Intel CPUs, the BOM is represented as b'\xff\xfe'. This helps determine whether the data should be interpreted in big-endian or little-endian order.

In UTF-16 encoding, without a BOM, it can be ambiguous on which byte ordering to use. The presence of U+FEFF (ZERO WIDTH NO-BREAK SPACE) at the beginning of a file is used as an indicator for little-endian UTF-16. 

For example, if you encode "El Niño" in UTF-16 and see b'\xff\xfe', this indicates it's little-endian.
??x
The BOM helps avoid confusion about byte ordering when working with binary sequences of encoded text.

```python
u16 = 'El Niño'.encode('utf_16')
print(list(u16))  # Output: [255, 254, 69, 0, 108, 0, 32, 0, 78, 0, 105, 0, 241, 0, 111, 0]
```

The output shows that the BOM is b'\xff\xfe' for little-endian encoding.
x??

---

#### UTF-16LE and UTF-16BE
Background context explaining the differences between UTF-16LE (little-endian) and UTF-16BE (big-endian). These are variations of UTF-16 that specify whether the byte ordering is in little or big-endian.

:p What are UTF-16LE and UTF-16BE, and how do they differ from regular UTF-16?
??x
UTF-16LE and UTF-16BE are variants of UTF-16 encoding that specify the byte order for multi-byte code points. Regular UTF-16 can be ambiguous without a BOM to indicate whether it's little-endian or big-endian.

UTF-16LE is explicitly little-endian, meaning bytes are ordered as (least significant byte, most significant byte). For example:
```python
u16le = 'El Niño'.encode('utf_16le')
print(list(u16le))  # Output: [69, 0, 108, 0, 32, 0, 78, 0, 105, 0, 241, 0, 111]
```

UTF-16BE is explicitly big-endian, meaning bytes are ordered as (most significant byte, least significant byte). For example:
```python
u16be = 'El Niño'.encode('utf_16be')
print(list(u16be))  # Output: [0, 69, 0, 108, 0, 32, 0, 78, 0, 105, 0, 241, 0, 111]
```

The difference lies in the byte order of multi-byte code points. The BOM is not needed for UTF-16LE and UTF-16BE as the encoding type itself specifies the endianness.
??x
UTF-16LE and UTF-16BE are used to explicitly denote the byte ordering when encoding text, ensuring clarity in the representation of multi-byte code points.

```python
# Example for UTF-16LE
u16le = 'El Niño'.encode('utf_16le')
print(list(u16le))  # Output: [69, 0, 108, 0, 32, 0, 78, 0, 105, 0, 241, 0, 111]

# Example for UTF-16BE
u16be = 'El Niño'.encode('utf_16be')
print(list(u16be))  # Output: [0, 69, 0, 108, 0, 32, 0, 78, 0, 105, 0, 241, 0, 111]
```

The output shows the differences in byte ordering for 'El Niño' between UTF-16LE and UTF-16BE.
x??

---

#### BOM in UTF-8
Background context explaining how BOM is used in UTF-8 encoding. The Unicode standard recommends using no BOM for UTF-8, but some Windows applications add a BOM to UTF-8 files.

:p How does the BOM work with UTF-8 encoding?
??x
In UTF-8 encoding, the BOM (Byte Order Mark) is not used as it produces the same byte sequence regardless of machine endianness. However, some Windows applications like Notepad add a BOM to UTF-8 files anyway, and Excel relies on this BOM to detect UTF-8 encoded text.

This additional BOM in UTF-8 encoding is referred to as "UTF-8-SIG" in Python's codec registry.
??x
The BOM (Byte Order Mark) in UTF-8 is not used by the standard because UTF-8 does not have byte ordering issues. However, some Windows applications like Notepad add a BOM to UTF-8 files for consistency or compatibility reasons.

In Python, this specific encoding with a BOM is called "UTF-8-SIG".

```python
# Example of adding a BOM in UTF-8-SIG
text = 'El Niño'.encode('utf_8_sig')
print(text)  # Output: b'\xef\xbb\xbfE\x00l\x00 \x00N\x00i\x00\xf1\x00o\x00'
```

The BOM in UTF-8-SIG is represented as b'\xef\xbb\xbf', and it helps Excel recognize the file as UTF-8 encoded.
x??

---

#### Chardet Package
Background context explaining how the Chardet package works to detect character encodings. Chardet can guess one or more than 30 supported encodings, including UTF-16LE.

:p How does the Chardet package work to detect character encodings?
??x
The Chardet package is a Python library designed to automatically detect character encoding of text files by analyzing their content. It supports over 30 different encodings and uses heuristics and byte frequency analysis to determine the most likely encoding.

Chardet includes a command-line utility, `chardetect`, which can be used to check the encoding of text files. For example:

```sh
$chardetect 04-text-byte.asciidoc
04-text-byte.asciidoc: utf-8 with confidence 0.99
```

This indicates that the file "04-text-byte.asciidoc" is likely encoded in UTF-8, according to Chardet's analysis.
??x
Chardet works by analyzing byte frequency and patterns within a text file to determine its encoding. The `chardetect` command-line utility can be used to automatically detect the character encoding of files.

```sh$ chardetect 04-text-byte.asciidoc
04-text-byte.asciidoc: utf-8 with confidence 0.99
```

This output indicates that Chardet is highly confident (with a confidence score of 0.99) that the file "04-text-byte.asciidoc" is encoded in UTF-8.
x??

---

#### Endianness and Byte Ordering
Background context explaining endianness, its impact on encodings like UTF-16, and how it affects byte ordering.

:p What is endianness and how does it affect encodings?
??x
Endianness refers to the order in which bytes are stored in multi-byte code points. Big-endian systems store the most significant byte first, while little-endian systems store the least significant byte first.

In UTF-16 encoding, the BOM (Byte Order Mark) helps determine the endianness of the data. For example:
```python
# Example for little-endian UTF-16
u16 = 'El Niño'.encode('utf_16')
print(list(u16))  # Output: [255, 254, 69, 0, 108, 0, 32, 0, 78, 0, 105, 0, 241, 0, 111, 0]
```

Here, the BOM is b'\xff\xfe', indicating little-endian byte ordering.

In big-endian UTF-16:
```python
# Example for big-endian UTF-16
u16be = 'El Niño'.encode('utf_16be')
print(list(u16be))  # Output: [0, 69, 0, 108, 0, 32, 0, 78, 0, 105, 0, 241, 0, 111]
```

The BOM is b'\xff\xfe' (big-endian) in this case.
??x
Endianness affects the order of bytes within multi-byte code points. Big-endian systems store the most significant byte first, while little-endian systems store the least significant byte first.

For example:
```python
# Example for little-endian UTF-16
u16 = 'El Niño'.encode('utf_16')
print(list(u16))  # Output: [255, 254, 69, 0, 108, 0, 32, 0, 78, 0, 105, 0, 241, 0, 111, 0]

# Example for big-endian UTF-16
u16be = 'El Niño'.encode('utf_16be')
print(list(u16be))  # Output: [0, 69, 0, 108, 0, 32, 0, 78, 0, 105, 0, 241, 0, 111]
```

The BOM in UTF-16 is used to denote the byte ordering. For little-endian, it starts with b'\xff\xfe'.
x??

---

