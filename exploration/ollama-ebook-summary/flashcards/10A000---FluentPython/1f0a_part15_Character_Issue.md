# Flashcards: 10A000---FluentPython_processed (Part 15)

**Starting Chapter:** Character Issues

---

#### Characters, Code Points, and Byte Representations
Background context explaining the concept. In Python 3, strings are Unicode characters, not raw bytes as in Python 2. The identity of a character is its code point, which ranges from U+0000 to U+10FFFF (4 to 6 hex digits with "U+" prefix). Code points are converted to byte sequences using encodings like UTF-8 and UTF-16LE.
:p What is the range of valid code points in Unicode?
??x
The range of valid code points in Unicode is from U+0000 to U+10FFFF, which corresponds to 0 to 1,114,111 in decimal. This wide range allows for a vast variety of characters including symbols and scripts used globally.
x??

---
#### Encoding and Decoding
Explanation: The process of converting code points to byte sequences is called encoding, while the reverse operation (converting bytes back to code points) is decoding. In Example 4-1, the string 'café' is encoded in UTF-8 and decoded back into a string.
:p What are the steps involved in encoding and decoding using an example?
??x
In Example 4-1, the process of encoding and decoding works as follows:
```python
s = 'café'
b = s.encode('utf8')    # Encoding: Convert str to bytes
print(b)                # Output: b'caf\xc3\xa9'
len(b)                  # Number of bytes in encoded string

# Decoding: Convert bytes back to str
decoded_str = b.decode('utf8')
print(decoded_str)      # Output: café
```
The original string 'café' is first converted into bytes using the UTF-8 encoding, resulting in `b'caf\xc3\xa9'`, which has five bytes. Then, these bytes are decoded back into the string 'café'.
x??

---
#### The Unicode Standard and Code Points
Explanation: Each character in a string is represented by its code point, which is a number from 0 to 1,114,111 (base 10). For example, U+0041 represents the letter 'A', while U+20AC stands for the Euro sign. About 13% of valid code points have actual characters assigned.
:p What is a code point and what range does it belong to?
??x
A code point is a number that uniquely identifies each character in the Unicode standard, ranging from 0 to 1,114,111 (base 10). For instance, U+0041 corresponds to 'A', while U+20AC represents the Euro sign.
x??

---
#### Encoding Differences Between UTF-8 and UTF-16LE
Explanation: Different encodings represent the same character using different byte sequences. The letter 'A' is represented as a single byte (\x41) in UTF-8, but it requires two bytes (\x41\x00) in UTF-16LE.
:p How does the encoding of the letter 'A' differ between UTF-8 and UTF-16LE?
??x
The letter 'A' is encoded differently in UTF-8 and UTF-16LE:
- In UTF-8, it is represented as a single byte: `\x41`
- In UTF-16LE (Little Endian), it requires two bytes: `\x41\x00`

This difference highlights how the same character can be represented in various ways depending on the encoding.
x??

---
#### Converting Characters to Bytes and Back
Explanation: The `encode` method converts a string into its byte representation, while the `decode` method reverses this process. For instance, encoding 'café' with UTF-8 results in b'caf\xc3\xa9', which is then decoded back to the original string.
:p How does the encoding and decoding of the string 'café' work using UTF-8?
??x
The string 'café' can be encoded into bytes using UTF-8:
```python
s = 'café'
b = s.encode('utf8')    # Encoding: Convert str to bytes
print(b)                # Output: b'caf\xc3\xa9'
len(b)                  # Number of bytes in encoded string

# Decoding: Convert bytes back to str
decoded_str = b.decode('utf8')
print(decoded_str)      # Output: café
```
Encoding 'café' with UTF-8 results in `b'caf\xc3\xa9'`, which consists of five bytes. These bytes can then be decoded back into the string 'café'.
x??

---
#### The Concept of Encoding and Decoding
Explanation: Understanding encoding and decoding is crucial for handling text data in Python, especially when dealing with different character sets or transmitting data over networks. Errors may occur if the wrong encoding is assumed.
:p What are the key steps involved in encoding and decoding text?
??x
The key steps in encoding and decoding text involve:
1. **Encoding**: Converting a string (text) into its byte representation using an appropriate encoding method, such as `encode('utf8')`.
2. **Decoding**: Reversing the process by converting bytes back into a string using the corresponding decode method, like `decode('utf8')`.

For example:
```python
s = 'café'
b = s.encode('utf8')    # Encoding: Convert str to bytes
print(b)                # Output: b'caf\xc3\xa9'
len(b)                  # Number of bytes in encoded string

# Decoding: Convert bytes back to str
decoded_str = b.decode('utf8')
print(decoded_str)      # Output: café
```
This process ensures that text data can be correctly interpreted and transmitted.
x??

---

#### Overview of Python 3 Binary Sequence Types

Background context: In Python 3, binary sequence types like `bytes` and `bytearray` were introduced to handle binary data. These types differ from their Python 2 counterparts (`str` for Unicode and `str` for byte strings). The `bytes` type is immutable, while `bytearray` is mutable. Both support many string methods but not those that depend on Unicode.

:p What are the key differences between `bytes` and `bytearray` in Python 3?
??x
- `bytes`: An immutable sequence of integers representing bytes.
- `bytearray`: A mutable sequence of integers representing bytes, similar to a list of integers where each integer is in the range 0 to 255.

Both types support slicing and provide methods like `endswith`, `replace`, `strip`, `translate`, and `upper` but exclude those that deal with Unicode data or formatting.
x??

---

#### Building `bytes` from String

Background context: In Python 3, you can create a `bytes` object by specifying a string along with an encoding. The resulting bytes represent the encoded version of the string.

:p How do you build a `bytes` object from a string?
??x
Use the constructor of `bytes` and provide the string and its encoding as arguments.
```python
my_bytes = bytes('café', encoding='utf_8')
```
x??

---

#### Slicing Behavior in `bytes`

Background context: Slicing behavior differs between immutable types like `bytes` and mutable types like `bytearray`. Even slicing a single byte produces a sequence of the same type.

:p What is the result of slicing a `bytes` object?
??x
Slicing a `bytes` object always returns another `bytes` object, even if the slice length is 1.
```python
cafe = bytes('café', encoding='utf_8')
print(cafe[:1])  # b'c'
```
x??

---

#### Literal Syntax for `bytearray`

Background context: There is no literal syntax for creating a `bytearray`. You can create one by assigning a `bytes` object to a variable.

:p How do you create a `bytearray`?
??x
You cannot use a literal like `b'...'` directly. Instead, initialize it from a `bytes` object.
```python
cafe_arr = bytearray(bytes('café', encoding='utf_8'))
```
x??

---

#### Displaying Byte Sequences

Background context: Python uses different display formats for byte sequences depending on their values.

:p What are the four ways to display bytes in Python?
??x
1. ASCII characters (space to tilde).
2. Escape sequences for special characters like tab, newline, carriage return.
3. Single and double quotes within a sequence are escaped with backslashes.
4. Hexadecimal escape sequences for other values.
```python
cafe = bytes('café', encoding='utf_8')
print(cafe)  # b'caf\xc3\xa9'
```
x??

---

#### Methods Supported by `bytes` and `bytearray`

Background context: Both `bytes` and `bytearray` support most string methods, excluding those related to Unicode or formatting.

:p What methods do `bytes` and `bytearray` share in common?
??x
- Common methods include `endswith`, `replace`, `strip`, `translate`, and `upper`.
- They exclude methods like `format`, `format_map`, `casefold`, `isdecimal`, `isidentifier`, `isnumeric`, and `encode`.
```python
cafe_bytes = bytes('café', encoding='utf_8')
print(cafe_bytes.upper())  # b'CAF\xc3\x89'
```
x??

---

#### Building `bytearray` from Buffers

Background context: You can construct a `bytearray` or `bytes` object from other buffer-like objects like another sequence of bytes, strings, etc.

:p How do you build a `bytearray` from another `bytes` object?
??x
You can create a `bytearray` by calling its constructor with a `bytes` object.
```python
cafe_bytes = bytes('café', encoding='utf_8')
cafe_arr = bytearray(cafe_bytes)
print(cafe_arr)  # bytearray(b'caf\xc3\xa9')
```
x??

---

#### The `fromhex` Method

Background context: Starting from Python 3.5, the `bytes.fromhex()` class method was introduced to build a binary sequence from hexadecimal strings.

:p How do you use the `fromhex` method?
??x
Use `bytes.fromhex('string')` to create a `bytes` object from a hexadecimal string.
```python
cafe_bytes = bytes.fromhex('31 4B CE A9')
print(cafe_bytes)  # b'1K\xce\xa9'
```
x??

---

#### Array Initialization from Raw Data
Background context: The `array` module allows creating arrays of fixed types. In Python, this is useful for handling binary data more efficiently than using lists.

:p How does the `array.array()` function work with raw data?
??x
The `array.array()` function takes a type code and an iterable (like a list or tuple) to create an array of elements in that specified type. For example, 'h' represents signed short integers (16-bit), and the provided list `[-2, -1, 0, 1, 2]` is converted into its binary representation.

```python
import array

numbers = array.array('h', [-2, -1, 0, 1, 2])
octets = bytes(numbers)
```

The resulting `octets` would be a byte sequence representing the integer values.
x??

---

#### Bytes and Memoryview Objects
Background context: While `bytes()` creates a new copy of the data, `memoryview()` allows sharing memory with other binary data structures. This is useful for large datasets to avoid unnecessary memory duplication.

:p How does `memoryview` differ from `bytes` when handling binary data?
??x
The key difference between `bytes` and `memoryview` is that `memoryview` shares the underlying memory with the original object, whereas `bytes` creates a new copy. This means modifications to the memory through a `memoryview` will affect the original object.

```python
import array

numbers = array.array('h', [-2, -1, 0, 1, 2])
octets = bytes(numbers)  # Creates a new byte sequence
shared_memory = memoryview(octets)
```

Using `memoryview` can be more efficient for large binary data.
x??

---

#### String Encoding and Decoding with Codecs
Background context: Python provides numerous codecs to encode strings into byte sequences and decode them back. This is essential for handling different character encodings, especially when dealing with international text.

:p What are the differences between using `encode()` and `decode()` methods in Python?
??x
The `encode()` method converts a string into bytes based on the specified encoding. Conversely, the `decode()` method converts byte sequences back into strings.

```python
text = "El Niño"
for codec in ['latin_1', 'utf_8', 'utf_16']:
    print(codec, text.encode(codec))
```

`latin_1` encodes characters using 8 bits, while `utf_8` and `utf_16` are variable-length encodings that can handle a wider range of Unicode characters.

Example output:
- `latin_1`: b'El Ni\xf1o'
- `utf_8`: b'El Ni\xc3\xb1o'
- `utf_16`: b'\xff\xfeE\x00l\x00 \x00N\x00i\x00\xf1\x00o\x00'

Note that not all encodings can represent every Unicode character, as shown by the presence of asterisks in Figure 4-1.
x??

---

#### Common Character Encodings
Background context: Different encodings handle different sets and ranges of characters. Some common ones include ASCII, UTF-8, GB2312, and various Microsoft-specific encodings like cp1252.

:p What is the importance of the 'utf_8' encoding in web development?
??x
The `utf_8` encoding is crucial for web development due to its wide adoption and ability to handle a broad range of Unicode characters. As of July 2021, W3Techs reports that 97% of websites use UTF-8.

```python
text = "El Niño"
encoded_text = text.encode('utf_8')
print(encoded_text)  # Output: b'El Ni\xc3\xb1o'
```

UTF-8 is particularly advantageous because it is backward compatible with ASCII, making it a safe choice for many applications.
x??

---

#### Memory Management and `memoryview`
Background context: Using `memoryview` can be more memory-efficient when working with large binary data structures. It allows multiple views to share the same memory without copying.

:p How does `memoryview` benefit performance in handling large binary datasets?
??x
`memoryview` benefits performance by allowing multiple views (like slices or transformations) of a single memory block. This avoids unnecessary copying, which is crucial for large datasets.

```python
import array

numbers = array.array('h', [-2, -1, 0, 1, 2])
octets = bytes(numbers)
shared_memory = memoryview(octets)

# Example usage: Accessing a portion of the data without copying
first_half = shared_memory[:5]
print(first_half.tobytes())  # Output: b'\xfe\xff\xff\xff\x00'
```

Using `memoryview` ensures that any changes to the view affect the original memory, reducing memory overhead.
x??

---

