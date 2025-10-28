# Flashcards: 10A006---Java-Cookbook---Ian-F-Darwin_processed (Part 13)

**Starting Chapter:** Problem. Discussion

---

---
#### Checksum Calculation for Text Files
Background context: A checksum is a numeric quantity that confirms and verifies the integrity of data. In this example, we're creating a simple checksum by summing up the numeric values of each character in a file's content.

Relevant code snippet:
```java
public static int process(BufferedReader  is) {
    int sum = 0;
    try {
        String inputLine ;
        while ((inputLine  = is.readLine ()) != null) { // Note: There was a typo in the original snippet; it should be "!="
            for (char c : inputLine .toCharArray ()) {
                sum += c;
            }
        }
    } catch (IOException  e) {
        throw new RuntimeException ("IOException: " + e);
    }
    return sum;
}
```
:p How does the provided code calculate a checksum?
??x
The provided code calculates a checksum by iterating over each line of text read from a `BufferedReader`. For every character in these lines, it adds up their numeric values. This simple method works for plain text but might not be suitable for binary files due to its simplistic approach.
```java
// Example usage:
int checksum = process(new BufferedReader(new FileReader("example.txt")));
```
x??

---
#### Aligning Strings using StringAlign Class
Background context: The `StringAlign` class is designed to center or align strings according to specified lengths and justifications. This utility can be particularly useful in creating formatted output, such as reports.

Relevant code snippet:
```java
public class StringAlignSimple {
    public static void main(String[] args) {
        // Construct a "formatter" to center strings.
        StringAlign formatter = new StringAlign(70, StringAlign.Justify.CENTER);
        
        // Try it out, for page "i".
        System.out.println(formatter.format("- i -"));
        
        // Try it out, for page 4. Since this formatter is
        // optimized for Strings, not specifically for page numbers,
        // we have to convert the number to a String.
        System.out.println(formatter.format(Integer.toString(4)));
    }
}
```
:p How can you use `StringAlign` to center strings?
??x
The `StringAlign` class uses the `format` method to center or align strings based on specified lengths and justifications. The provided example demonstrates creating an instance with a max length of 70 characters, centered alignment, and then formatting two different strings.
```java
// Example usage:
StringAlign formatter = new StringAlign(70, StringAlign.Justify.CENTER);
System.out.println(formatter.format("Centered String"));
```
x??

---
#### StringJustification Logic in `StringAlign` Class
Background context: The `StringAlign` class implements a custom justification logic to align strings. This involves calculating padding spaces based on the string length and desired alignment.

Relevant code snippet:
```java
@Override
public StringBuffer format(Object input, StringBuffer where, FieldPosition ignore) {
    String s = input.toString();
    String wanted = s.substring(0, Math.min(s.length(), maxChars));
    
    // Get the spaces in the right place.
    switch (just) {
        case RIGHT: 
            pad(where, maxChars - wanted.length());
            where.append(wanted);
            break;
        case CENTER:
            int toAdd = maxChars - wanted.length();
            pad(where, toAdd / 2);
            where.append(wanted);
            pad(where, toAdd - toAdd / 2);
    }
}
```
:p How does the `StringAlign` class handle right justification?
??x
For right justification, the `StringAlign` class first calculates the necessary padding spaces by subtracting the length of the trimmed string from the maximum allowed character count. It then appends the appropriate number of spaces before the input string.
```java
// Example implementation:
void pad(StringBuffer where, int numSpaces) {
    for (int i = 0; i < numSpaces; i++) {
        where.append(' ');
    }
}
```
x??

---
#### StringJustification Logic for Centered Strings in `StringAlign` Class
Background context: The `StringAlign` class also supports centering strings, which requires calculating padding spaces on both sides of the string.

Relevant code snippet:
```java
@Override
public StringBuffer format(Object input, StringBuffer where, FieldPosition ignore) {
    // ... (previous logic)
    
    case CENTER:
        int toAdd = maxChars - wanted.length();
        pad(where, toAdd / 2);
        where.append(wanted);
        pad(where, toAdd - toAdd / 2);
    }
}
```
:p How does the `StringAlign` class handle centering strings?
??x
For centering strings, the `StringAlign` class calculates the total number of padding spaces needed by subtracting the length of the trimmed string from the maximum allowed character count. It then evenly distributes these spaces on both sides of the input string.
```java
// Example implementation:
void pad(StringBuffer where, int numSpaces) {
    for (int i = 0; i < numSpaces; i++) {
        where.append(' ');
    }
}
```
x??

---

#### Indent Method Introduction
Background context: Java 12 introduced a new method `public String indent(int n)` that prepends `n` spaces to the string, treating it as a sequence of lines with line separators. This works well in conjunction with the Java 11 Stream<String> lines() method.
If applicable, add code examples with explanations:
:p What is the purpose of the `indent` method introduced in Java 12?
??x
The `indent` method introduces spaces at the beginning of a string to indent it. This is useful for aligning text or adding a margin before content.

```java
// Example usage
String original = "abc def";
String indented = original.indent(30);
```
x??

---

#### Lines Method and Indentation
Background context: The `lines()` method in Java can be used to split a string into lines. When combined with the `indent` method, it can indent each line of a multi-line string.
:p How does combining the `indent` and `lines` methods work?
??x
When you call `indent` on a string, it adds spaces at the beginning of each character in the string. If this string contains multiple lines (newlines are present), calling `lines()` will split these into individual strings. The `indent` method is applied to each line separately.

```java
// Example usage
String original = "abc def\nghi jkl";
String indented = original.indent(30);
indented.lines().forEach(System.out::println);
```
x??

---

#### Line Indentation with Streams and Indent Method
Background context: The `indent` method can be used in conjunction with the Java 11 `Stream<String>` lines() method to apply indentation to each line of a string.
:p How can you use the `indent` method along with `lines()` to indent multiple lines?
??x
You can use the `indent` method on a multi-line string, and then call `lines()` to split it into individual strings. Each line is indented independently.

```java
// Example usage
String original = "abc def\nghi jkl";
String indented = original.indent(30).lines().collect(Collectors.joining("\n"));
```
x??

---

#### Using the `indent` Method with Negative Values
Background context: The `indent` method can also handle negative values, which will remove spaces from the beginning of each line.
:p What happens if you use a negative value in the `indent` method?
??x
Using a negative value in the `indent` method will remove spaces from the beginning of each line. If the number is too large (more than the current indentation), it effectively removes all leading spaces.

```java
// Example usage
String original = "   abc def\n   ghi jkl";
String indented = original.indent(-10);
indented.lines().forEach(System.out::println);
```
x??

---

#### Indentation and Formatting
Background context: The `indent` method can be used in conjunction with the `format` method to create aligned text.
:p How does the `indent` method interact with the `format` method?
??x
The `indent` method can be used to add or remove spaces at the beginning of a string. This can be useful when combined with formatting methods like `format`, which is used to create neatly aligned output.

```java
// Example usage
String original = "abc def";
String indented = original.indent(30).lines().collect(Collectors.joining("\n"));
System.out.println(indented);
```
x??

---
#### Unicode and Character Encoding
Background context: Unicode is an international standard for representing characters from all languages, including emojis and historical scripts. It uses a 16-bit character set to accommodate a wide range of characters, but over time has expanded to include more than 1 million code points.

Java's `char` type naturally supports Unicode, being 16 bits wide. However, as the number of characters grew beyond 65,525 (the limit for a 16-bit encoding), UTF-16 was introduced as a standard way to handle these extended characters.
:p What is the primary reason Java's `char` type uses 16 bits?
??x
The primary reason Java's `char` type uses 16 bits is because Unicode originally aimed to represent all known characters using a 16-bit encoding. This was done to ensure that it could accommodate a wide variety of languages and symbols.
x??

---
#### Surrogate Pairs in UTF-16
Background context: To handle the growth in the number of Unicode characters, UTF-16 uses surrogate pairs for characters beyond the Basic Multilingual Plane (BMP). These pairs consist of two 16-bit code units where each unit represents a half of the character.
:p What are surrogate pairs used for in UTF-16?
??x
Surrogate pairs are used in UTF-16 to represent Unicode characters that exceed the range of a single 16-bit value. They allow encoding these characters by using two 16-bit code units, each representing half of the character.
x??

---
#### String Class and Unicode Characters
Background context: The `String` class in Java provides methods for dealing with both Unicode code points (individual characters) and code units (the underlying 16-bit values). This is crucial when working with extended Unicode characters that require more than one `char`.
:p How can you determine the raw value of a character in a `String`?
??x
To determine the raw value of a character in a `String`, you can convert the `char` to its integer representation using casting or by directly calling methods like `Character.codePointAt()`. Here is an example:
```java
String str = "abc¥ǼΑΩ";
for (int i = 0; i < str.length(); i++) {
    System.out.printf("Character # %d (%04x) is %c%n", 
                      i, (int)str.charAt(i), str.charAt(i));
}
```
This code iterates over each character in the `String` and prints its integer value.
x??

---
#### Arithmetic on Characters
Background context: Although arithmetic operations on characters are not commonly used in Java due to the availability of high-level methods provided by the `Character` class, it can be useful for certain programming tasks. However, such operations should be approached with caution as they might lead to unexpected results if not handled correctly.
:p Can you use arithmetic operations directly on characters?
??x
Yes, you can perform arithmetic operations directly on characters in Java because a `char` is essentially an integer type representing Unicode code points. Here's an example of using arithmetic to control a loop and append characters to a `StringBuilder`:
```java
StringBuilder b = new StringBuilder();
for (char c = 'a'; c < 'd'; c++) {
    b.append(c);
}
b.append('\u00a5'); // Japanese Yen symbol
b.append('\u01fc'); // Roman AE with acute accent
b.append('\u0391'); // GREEK Capital Alpha
b.append('\u03a9'); // GREEK Capital Omega

for (int i = 0; i < b.length(); i++) {
    System.out.printf("Character # %d (%04x) is %c%n", 
                      i, (int)b.charAt(i), b.charAt(i));
}
```
This example demonstrates appending characters to a `StringBuilder` and printing their Unicode code points.
x??

---

