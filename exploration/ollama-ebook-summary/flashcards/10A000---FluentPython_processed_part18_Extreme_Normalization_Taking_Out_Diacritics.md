# Flashcards: 10A000---FluentPython_processed (Part 18)

**Starting Chapter:** Extreme Normalization Taking Out Diacritics

---

#### Unicode Normalization and Search Challenges
Background context: When dealing with Unicode, normalization is crucial because different representations of the same character can exist. For example, the VULGAR FRACTION ONE HALF could be represented as "1⁄2" or "1/2", but NFKC normalization turns it into "1\u0035\/\u0032". This means that searching for "1/2" will not find this normalized sequence.
:p Why is normalization important when working with Unicode strings?
??x
Normalization is essential because it ensures consistent representation of characters, making comparisons and searches more reliable. Different Unicode sequences might represent the same character or string in various ways, which can lead to mismatches during operations like searching or comparison if not handled properly.
x??

---

#### Case Folding Basics
Background context: Case folding converts all text to lowercase with some additional transformations. It is different from simple `lower()` conversion and helps in case-insensitive comparisons across languages. The core difference lies in how certain characters are transformed, such as the micro sign 'µ' becoming 'μ', and 'ß' becoming 'ss'.
:p What does case folding do differently compared to regular lowercase conversion?
??x
Case folding converts all text to lowercase but includes additional transformations for specific characters that `lower()` might not handle. For instance, 'µ' is converted to 'μ', and 'ß' becomes 'ss'. This ensures more accurate case-insensitive comparisons across different languages.
x??

---

#### NFKC vs NFC Normalization
Background context: NFC (Normalization Form C) is generally safe for most applications as it combines characters where possible but does not decompose them. NFD (Normalization Form D), on the other hand, decomposes characters into their base elements and combining marks. However, NFKC can cause data loss by removing certain character compositions.
:p What are the differences between NFC and NFKC normalization?
??x
NFC combines characters where possible but does not decompose them, making it suitable for most applications due to its stability and safety. In contrast, NFD decomposes characters into their base elements and combining marks, which can be useful in some scenarios. NFKC, while more aggressive in decomposing characters, may cause data loss by removing certain character compositions.
x??

---

#### Utility Functions for Text Comparison
Background context: To handle Unicode normalization properly, utility functions like `nfc_equal` and `fold_equal` are often used. These functions ensure that strings are compared in a normalized form suitable for the task at hand—either case-sensitive or insensitive.
:p How do you create a function to compare two strings using NFC normalization?
??x
To compare two strings using NFC normalization, you can use the following utility function:
```python
from unicodedata import normalize

def nfc_equal(str1, str2):
    return normalize('NFC', str1) == normalize('NFC', str2)
```
This function normalizes both strings to NFC form before comparing them.
x??

---

#### Case-Folding Comparison with `casefold()`
Background context: To perform case-insensitive comparisons across different languages, you can use the `casefold()` method. This method is more aggressive than `lower()` and handles special cases like 'µ' becoming 'μ' and 'ß' becoming 'ss'.
:p How does the `str.casefold()` function work?
??x
The `str.casefold()` function converts a string to lowercase but also performs additional transformations for certain characters, ensuring case-insensitive comparisons are more accurate across different languages. For example:
```python
micro = 'µ'
micro_cf = micro.casefold()  # μ
eszett = 'ß'
eszett_cf = eszett.casefold()  # ss
```
This function is particularly useful when working with text in multiple languages that have special case-sensitive characters.
x??

---

#### Combining Functions for Normalized String Comparison
Background context: To develop utility functions for normalized string comparison, you can combine `normalize` and `casefold` to ensure accurate comparisons. The following example shows how to create such functions:
:p How do you create a function that uses both NFC normalization and case folding?
??x
To create a function that uses both NFC normalization and case folding, you can use the following utility function:
```python
def fold_equal(str1, str2):
    return (normalize('NFC', str1).casefold() == normalize('NFC', str2).casefold())
```
This function first normalizes both strings to NFC form and then applies `casefold()` to ensure a case-insensitive comparison.
x??

---

#### Removing Diacritics: General Approach
Background context explaining how diacritics affect word meaning and search efficiency. Discussing scenarios where removing diacritics can be useful, such as improving URL readability.

:p What is the purpose of removing diacritics in text processing?
??x
The purpose of removing diacritics is to standardize spellings and improve search functionality by ignoring accents, cedillas, and other diacritical marks. This helps handle spelling inconsistencies and changing rules in languages, making searches more flexible but potentially leading to false positives.

For example:
- "café" becomes "cafe"
- "São Paulo" becomes "Sao Paulo"

This approach can be useful for making URLs cleaner and more readable.
x??

---

#### Removing Diacritics: General Function Implementation
Explanation of the `shave_marks` function provided, which removes all diacritical marks from text.

:p How does the `shave_marks` function work?
??x
The `shave_marks` function works by decomposing characters into their base and combining components using Unicode normalization. It then filters out any combining marks and recomposes the string to remove them.

Code example:
```python
import unicodedata

def shave_marks(txt):
    """Remove all diacritic marks"""
    norm_txt = unicodedata.normalize('NFD', txt)
    shaved = ''.join(c for c in norm_txt if not unicodedata.combining(c))
    return unicodedata.normalize('NFC', shaved)

# Example usage
print(shave_marks("Herr Voß: • ½ cup of Œtker™ caffè latte • bowl of açaí."))
```

Explanation:
- `unicodedata.normalize('NFD', txt)` decomposes the text into base characters and combining marks.
- A generator expression filters out all combining marks, preserving only the base characters.
- The result is recomposed using `unicodedata.normalize('NFC', shaved)`.
x??

---

#### Removing Diacritics: Latin Character Focus
Explanation of the modified function that only removes diacritics from Latin characters.

:p What does the `shave_marks_latin` function do differently?
??x
The `shave_marks_latin` function focuses specifically on Latin characters, preserving non-Latin characters with their diacritical marks. It decomposes characters and skips combining marks if they are attached to a non-Latin base character.

Code example:
```python
def shave_marks_latin(txt):
    """Remove all diacritic marks from Latin base characters"""
    norm_txt = unicodedata.normalize('NFD', txt)
    latin_base = False
    preserve = []
    for c in norm_txt:
        if unicodedata.combining(c) and latin_base:
            continue  # ignore combining chars on Latin base char
        preserve.append(c)
        if not unicodedata.combining(c):
            latin_base = c in string.ascii_letters
    shaved = ''.join(preserve)
    return unicodedata.normalize('NFC', shaved)

# Example usage
print(shave_marks_latin("Ζέφυρος, Zéfiro"))
```

Explanation:
- The function decomposes the text into base and combining marks.
- It preserves combining marks if they are attached to a non-Latin character (e.g., Greek letters).
- Only Latin characters have their diacritics removed.

This approach ensures that non-Latin characters retain their accents while Latin characters do not.
x??

---

#### Concept of Character Mapping and Replacement
Background context: The provided text discusses how to transform Western typographical symbols into their ASCII equivalents. This involves building mapping tables for character-to-character and character-to-string replacements, then merging these mappings.

:p What is the purpose of creating a `single_map` and `multi_map` in the provided code?
??x
The purpose of creating `single_map` and `multi_map` is to define specific character replacements. The `single_map` contains one-to-one character translations, such as curly quotes into straight quotes. The `multi_map` includes more complex transformations like the trademark symbol (™) into a string representation.

```python
# Example creation of single and multi mapping tables
single_map = str.maketrans("‚ƒ„ˆ‹‘’“”•–—˜›", "'f\"^<''\"\"---~>")
multi_map = {
    '€': 'EUR',
    '…': '...',
    'Æ': 'AE',
    'æ': 'ae',
    'Œ': 'OE',
    'œ': 'oe',
    '™': '(TM)',
    '‰': '<per mille>',
    '†': '**',
    '‡': '***'
}
```

x??

---

#### Concept of Merging Mapping Tables
Background context: After defining `single_map` and `multi_map`, the next step is to merge these two mappings. This ensures that all possible replacements are covered, whether they involve single characters or strings.

:p How do you merge the `single_map` and `multi_map` in Python?
??x
To merge the `single_map` and `multi_map` in Python, you can use the `update()` method of dictionaries. The `update()` method adds dictionary entries to an existing dictionary, updating it with the key-value pairs from another dictionary.

```python
# Merging single and multi mappings
multi_map.update(single_map)
```

This line of code effectively combines all the replacements defined in both maps into a single mapping table called `multi_map`.

x??

---

#### Concept of `dewinize` Function
Background context: The `dewinize` function is responsible for replacing Windows 1252-specific symbols with their ASCII equivalents. This function only affects symbols added to Latin1 by Microsoft and does not alter ASCII or Latin1 characters.

:p What does the `dewinize` function do in Python?
??x
The `dewinize` function replaces specific Win1252 symbols with their ASCII representations. It uses a mapping table defined in `multi_map` to perform these replacements, ensuring that curly quotes, bullets, and other non-ASCII characters are transformed into their ASCII equivalents.

```python
# Dewinize function definition
def dewinize(txt):
    """
    Replace Win1252 symbols with ASCII chars or sequences
    """
    return txt.translate(multi_map)
```

The `translate()` method is used here to replace the characters defined in `multi_map` within the input text `txt`.

x??

---

#### Concept of Removing Diacritical Marks and Eszett Replacement
Background context: After using `dewinize`, the next step involves removing diacritical marks (accents) from Latin1 characters. The text then replaces the German 'ß' with "ss".

:p What is the purpose of the function that removes diacritical marks in Python?
??x
The purpose of the function that removes diacritical marks is to strip non-essential accents and other marks from characters, making them more suitable for ASCII or simplified text representation. This step ensures that only basic Latin characters remain.

```python
# Removing diacritics example
no_marks = shave_marks_latin(dewinize(txt))
```

The function `shave_marks_latin` is assumed to be a placeholder here, representing the actual logic of stripping diacritical marks from Latin1 characters. The result is stored in the variable `no_marks`.

x??

---

#### Concept of Applying Normalization and Final Text Transformation
Background context: After removing diacritics, the text undergoes normalization using `unicodedata.normalize('NFKC', no_marks)`. This step applies Unicode normalization to compose characters with their compatibility code points. Finally, 'ß' is replaced with "ss".

:p What does the final transformation in Python involve?
??x
The final transformation involves several steps:
1. Removing diacritical marks from the text.
2. Replacing the German letter 'ß' with "ss".
3. Applying NFKC normalization to the processed text.

```python
# Final asciize function definition
def asciize(txt):
    no_marks = shave_marks_latin(dewinize(txt))
    no_marks = no_marks.replace('ß', 'ss')
    return unicodedata.normalize('NFKC', no_marks)
```

The `unicodedata.normalize()` method applies NFKC normalization, which composes characters with their compatibility code points, making the text more uniform and easier to process.

x??

---

#### Locale-Aware String Sorting in Python
Background context explaining the concept. In Python, sorting strings by default compares their Unicode code points directly, which can lead to unexpected results when dealing with non-ASCII characters. Different locales have different rules for string comparison, especially regarding accents and diacritics.

The provided example shows that without locale-aware sorting, the fruits are not sorted correctly according to Portuguese rules:
```python
fruits = ['caju', 'atemoia' , 'cajá', 'açaí', 'acerola']
sorted(fruits)  # Output: ['acerola', 'atemoia', 'açaí', 'caju', 'cajá']
```
:p How does the default string sorting in Python handle non-ASCII characters?
??x
The default string sorting in Python compares Unicode code points directly, which may not follow locale-specific rules for sorting strings that contain accented or special characters. For example, "cajá" should come before "caju" in Portuguese but is sorted after it due to the direct comparison of code points.
x??

---
#### Using `locale.strxfrm` for Locale-Aware Sorting
Background context explaining the concept. The `locale.strxfrm` function can be used as a key function in sorting operations to apply locale-specific string transformations, ensuring that non-ASCII characters are sorted according to the rules defined by the chosen locale.

The example demonstrates setting up and using the `locale.strxfrm` function for sorting strings:
```python
import locale

my_locale = locale.setlocale(locale.LC_COLLATE, 'pt_BR.UTF-8')
print(my_locale)  # Output: 'pt_BR.UTF-8'
fruits = ['caju', 'atemoia' , 'cajá', 'açaí', 'acerola']
sorted_fruits = sorted(fruits, key=locale.strxfrm)
print(sorted_fruits)  # Output: ['açaí', 'acerola', 'atemoia', 'cajá', 'caju']
```
:p How do you use `locale.strxfrm` for locale-aware sorting in Python?
??x
You use the `locale.strxfrm` function as a key in the `sorted()` method to sort strings according to the rules of the specified locale. This requires setting up the desired locale using `locale.setlocale(locale.LC_COLLATE, 'your_locale')`. For example, with Portuguese Brazil ('pt_BR.UTF-8'), "açaí" should come before "caju", and this is achieved by transforming each string into a form that can be compared according to the rules of the locale.
x??

---
#### Caveats of Using `locale.strxfrm`
Background context explaining the concept. While using `locale.strxfrm` provides a solution for locale-aware sorting, there are several limitations and caveats:
- Locale settings are global, which means changing them can affect other parts of your application or framework.
- The required locale must be installed on the operating system; otherwise, `setlocale()` will raise an exception.
- You need to know how to spell the correct locale name for your specific use case.

The example shows that setting up and using the function might work in some environments but not others:
```python
fruits = ['caju', 'atemoia' , 'cajá', 'açaí', 'acerola']
sorted(fruits, key=locale.strxfrm)  # Might produce incorrect results if locale is not set correctly
```
:p What are the limitations of using `locale.strxfrm` for sorting?
??x
The limitations include:
- Locale settings affect all parts of the application globally, making it risky to change them in a library.
- The required locale must be installed on the operating system; otherwise, `setlocale()` will raise an exception.
- You need to know the exact locale name that corresponds to your use case.
- Even if the correct locale is set, there's no guarantee that the sorting rules are correctly implemented by the OS developers.
x??

---
#### PyUCA Library for Locale-Aware Sorting
Background context explaining the concept. The `PyUCA` library provides a simpler and more portable solution for locale-aware string comparison in Python without relying on system locales.

The example suggests using `PyUCA` as an alternative:
```python
# Install PyUCA using pip: pip install pyuca

from pyuca import Collator

collator = Collator('pt_BR')
sorted_fruits = sorted(fruits, key=collator.sort_key)
print(sorted_fruits)  # Output: ['açaí', 'acerola', 'atemoia', 'cajá', 'caju']
```
:p What is the `PyUCA` library used for?
??x
The `PyUCA` library provides a simple and portable solution for locale-aware string comparison in Python. It uses the Unicode Collation Algorithm (UCA) to sort strings according to specific rules, making it independent of system locales and reducing deployment headaches.

This approach avoids the global state issues associated with changing locale settings and does not require knowing the exact locale name.
x??

---

#### PyUCA Overview
PyUCA is a Python implementation of the Unicode Collation Algorithm (UCA), which helps in sorting strings according to their linguistic rules, ensuring that characters are sorted correctly based on locale-specific rules. This is particularly important for languages with complex character compositions and ordering requirements.
:p What is PyUCA used for?
??x
PyUCA is used to sort strings according to the Unicode Collation Algorithm (UCA), which respects the specific sorting rules of different languages and scripts. It provides a way to perform locale-aware string sorting in Python without relying on system-specific locale settings.
x??

---

#### Using PyUCA for Sorting
PyUCA simplifies the process of sorting text data by using the `sort_key` method, which generates a sorting key based on UCA rules. This ensures that strings are sorted correctly according to their linguistic properties.
:p How can you use PyUCA to sort a list of fruits?
??x
You can use PyUCA to sort a list of fruits as follows:

```python
import pyuca

coll = pyuca.Collator()
fruits = ['caju', 'atemoia', 'cajá', 'açaí', 'acerola']
sorted_fruits = sorted(fruits, key=coll.sort_key)
```

This code creates a `Collator` object and uses its `sort_key` method to generate sorting keys for each fruit in the list. The fruits are then sorted based on these keys.
x??

---

#### Customizing Sorting with PyUCA
PyUCA allows you to provide a custom collation table, which can be useful if you need to follow specific sorting rules that differ from the default UCA settings. This is particularly useful for languages where standard UCA behavior might not suffice.
:p How can you use a custom collation table in PyUCA?
??x
You can use a custom collation table by passing its path to the `Collator` constructor. Here’s an example:

```python
import pyuca

custom_coll = pyuca.Collator(path_to_custom_collation_table)
```

This code initializes a `Collator` with a custom collation table, allowing you to define specific sorting rules for your application.
x??

---

#### PyICU as Alternative
PyICU is an alternative to PyUCA that offers more flexibility in sorting behavior. It mimics the locale settings used by operating systems but doesn’t change them globally. This makes it suitable for applications where changing the system locale might not be desirable.
:p What does PyICU offer compared to PyUCA?
??x
PyICU provides a way to perform locale-aware string operations without altering the system's locale settings. It uses ICU (International Components for Unicode) and can handle more complex sorting scenarios, such as case folding in Turkish where 'ı' and 'i' need special handling.
x??

---

#### PyICU Installation Considerations
PyICU requires an extension that must be compiled, making its installation potentially more challenging on certain systems compared to PyUCA, which is purely Python-based.
:p Why might PyICU be harder to install than PyUCA?
??x
PyICU is harder to install than PyUCA because it includes a C extension that needs to be compiled. This can pose challenges on systems where compiling C code is difficult or not supported, whereas PyUCA can be installed easily as a pure Python package.
x??

---

