# Flashcards: 10A006---Java-Cookbook---Ian-F-Darwin_processed (Part 23)

**Starting Chapter:** 5.5 Formatting Numbers. Problem. Solution

---

#### Rounding Method Implementation

Background context: The provided Java code demonstrates a custom rounding method and compares it with the built-in `Math.round` function. The custom rounding logic is based on adding a specific threshold to the input number before applying the floor function.

:p How does the custom rounding method work?
??x
The custom rounding method works by adding a value (1.0 - THRESHOLD) to the input double `d`. This adjustment pushes numbers just above the threshold towards the next integer, effectively rounding them up if their fractional part is greater than 0.54.

```java
public static int round(double d) {
    return (int)Math.floor(d + 1.0 - THRESHOLD);
}
```

x??

---

#### Rounding Method Comparison

Background context: The provided code snippet includes a comparison between the custom rounding method and the built-in `Math.round` function.

:p What does this code do?
??x
This code compares the results of a custom rounding method with the built-in `Math.round` function for various double values from 0.1 to 1.0 in increments of 0.05.

```java
for (double d = 0.1; d<=1.0; d+=0.05) {
    System.out.println("My way:  " + d + " -> " + round(d));
    System.out.println("Math way:" + d + " -> " + Math.round(d));
}
```

x??

---

#### NumberFormat Class Overview

Background context: The provided text discusses the `NumberFormat` class in Java, which is part of the `java.text` package. This class provides flexible and general-purpose number formatting capabilities.

:p What is the purpose of the `NumberFormat` class?
??x
The `NumberFormat` class is used for flexible and internationalized number formatting in Java. It can be customized to format numbers, currencies, or percentages according to various locales.

x??

---

#### Locale-Specific Number Formatting

Background context: The example demonstrates how to use the `NumberFormat` class with different locale settings to achieve custom number formatting patterns.

:p How is a locale-specific number format created and applied?
??x
To create a locale-specific number format, you can use the `NumberFormat.getInstance()` method. This method returns an instance of `NumberFormat` configured according to the default or specified locale.

```java
// Get a format instance
NumberFormat form = NumberFormat.getInstance();

// Set it to look like 999.99[99]
form.setMinimumIntegerDigits(3);
form.setMinimumFractionDigits(2);
form.setMaximumFractionDigits(4);

// Now print using it
for (int i=0; i<data.length; i++) {
    System.out.println(data[i] + "\tformats as " +
                       form.format(data[i]));
}
```

x??

---

#### Custom DecimalFormat Pattern

Background context: The example shows how to use `DecimalFormat` with custom patterns and manipulate the format using set methods.

:p How does the `DecimalFormat` class allow for customization?
??x
The `DecimalFormat` class allows for customization of number formatting through its constructor and method calls. You can apply a specific pattern or modify existing patterns dynamically.

```java
NumberFormat defForm = NumberFormat.getInstance();
NumberFormat ourForm = new DecimalFormat("##0.##");

// toPattern() reveals the pattern used by this Locale
System.out.println(defForm.getFormat().toPattern());

System.out.println(intlNumber + " formats as " +
                   ((DecimalFormat)defForm).format(intlNumber));
```

x??

---

#### Human-Readable Number Formatting

Background context: The example demonstrates how to use `CompactNumberFormat` for human-readable number formatting, commonly used in Unix/Linux systems.

:p How does the `CompactNumberFormat` class format numbers?
??x
The `CompactNumberFormat` class is used to print numbers in a more readable and compact form, often used in Unix/Linux environments. It automatically handles large numbers by appending appropriate suffixes like K (kilo), M (mega), etc.

```java
NumberFormat cnf = NumberFormat.getCompactNumberInstance();
System.out.println(n + ": " + cnf.format(n));
```

x??

---

#### Roman Numeral Formatting

Background context: The example shows how to use the `RomanNumberFormat` class for formatting numbers into Roman numerals.

:p How does the `RomanNumberFormat` class format numbers as Roman numerals?
??x
The `RomanNumberFormat` class formats numbers as Roman numerals. It can be used to convert integers to their Roman numeral representation, which is particularly useful for displaying dates or other text-based representations.

```java
RomanNumberFormat nf = new RomanNumberFormat();
int year = LocalDate.now().getYear();
System.out.println(year + " -> " + nf.format(year));
```

x??

---

#### Converting Between Bases Using Integer Class
The `Integer` class provides methods to convert numbers between different bases, which is useful for various applications such as hardware interactions and data manipulation. The `parseInt()` method converts a string representation of a number into an integer value with a specified radix (base), while the `toString()` method converts an integer value into its string representation in a given base.

:p How can you convert a binary string to an integer using the `Integer` class?
??x
To convert a binary string to an integer, use the `parseInt()` method from the `Integer` class. This method takes two parameters: the string representation of the number and the radix (base) in which the number is represented.

Example code:
```java
String binaryString = "101010";
int decimalValue = Integer.parseInt(binaryString, 2);
System.out.println(decimalValue); // Output: 42
```
x??

---

#### Converting an Integer to a String in Different Bases
The `Integer` class provides the `toString()` method to convert an integer value into its string representation in various bases (binary, octal, decimal, hexadecimal). This is useful for displaying or storing numbers in different formats.

:p How can you display an integer as a series of bits using the `Integer` class?
??x
To display an integer as a series of bits, use the `toString()` method from the `Integer` class. This method takes two parameters: the integer value and the radix (base) in which the number should be represented.

Example code:
```java
int i = 42;
for (int radix : new int[] { 2, 8, 10, 16, 36 }) {
    System.out.println(i + " formatted in base " + radix + " is " + Integer.toString(i, radix));
}
```
This will output the integer `42` in binary, octal, decimal, hexadecimal, and base-36 formats.

Example output:
```
42 formatted in base 2 is 101010
42 formatted in base 8 is 52
42 formatted in base 10 is 42
42 formatted in base 16 is 2a
42 formatted in base 36 is 16
```
x??

---

#### Parsing a String to an Integer with Specified Radix
The `Integer.parseInt()` method allows you to parse a string into an integer value based on the specified radix (base). This method is particularly useful when dealing with non-decimal number systems.

:p How do you use `parseInt()` to convert a hexadecimal value into an integer?
??x
To convert a hexadecimal value into an integer, use the `Integer.parseInt()` method and specify the string representation of the number along with the radix (16 for hexadecimal).

Example code:
```java
String hexValue = "2a";
int decimalValue = Integer.parseInt(hexValue, 16);
System.out.println(decimalValue); // Output: 42
```
x??

---

#### Using `toString()` and `parseInt()` Together
The `Integer.toString()` method converts an integer value to a string in the specified base, while `parseInt()` does the opposite—it converts a string representation of a number into its integer equivalent with a given radix.

:p How can you convert a binary string "101010" to decimal and back using both methods?
??x
To convert a binary string "101010" to decimal, use `parseInt()`:

```java
String binaryString = "101010";
int decimalValue = Integer.parseInt(binaryString, 2);
System.out.println(decimalValue); // Output: 42
```

To convert the resulting integer back to a binary string, use `toString()`:

```java
String binaryStringBack = Integer.toString(decimalValue, 2);
System.out.println(binaryStringBack); // Output: 101010
```
x??

---

#### Range and For Loops for Integer Sequences
In Java, to process a range of integers, you can use `IntStream::range` or `rangeClosed`, or traditional for loops. These methods provide flexibility in iterating over sequences.

:p How do you iterate over a contiguous set of integers using `IntStream`?
??x
You can use the `IntStream.rangeClosed(start, endInclusive)` method to generate an integer stream from a starting number (inclusive) to an ending number (inclusive). This is useful for operations that need to process each integer in a given range.

Example:
```java
IntStream.rangeClosed(1, 12).forEach(i -> System.out.println("Month #" + i));
```

x??

---
#### Discontinuous Ranges using BitSet
For discontinuous ranges of numbers, Java provides the `BitSet` class. This class can be used to set specific bits (representing integers) and then iterate over those set bits.

:p How do you use a `BitSet` for iterating over discontiguous sets of integers?
??x
You create a `BitSet`, set specific bits using the `set(int i)` method, and then iterate over the set bits. This is particularly useful when you need to process a non-sequential range of numbers.

Example:
```java
BitSet b = new BitSet();
b.set(0);    // January
b.set(3);    // April
b.set(8);    // September

for (int i = 0; i < months.length; i++) {
    if (b.get(i)) {
        System.out.println("Month " + months[i]);
    }
}
```

x??

---
#### For-Each Loop for Arrays and Collections
Java's `for-each` loop can be used to iterate over the elements of an array or collection, making it a convenient method for processing data.

:p How do you use a `for-each` loop to iterate over an array?
??x
You can use the `for-each` loop to iterate over each element in an array. This is particularly useful when you want to process all elements without needing index-based access.

Example:
```java
String[] months = {"January", "February", "March", "April", "May", "June",
                   "July", "August", "September", "October", "November", "December"};

for (String month : months) {
    System.out.println(month);
}
```

x??

---
#### Contiguous Ranges with `range` and `rangeClosed`
Java's `IntStream` class provides methods like `range(start, endExclusive)` and `rangeClosed(start, endInclusive)` to generate a range of integers. These methods are useful for processing contiguous sequences.

:p How do you use `IntStream::range` to iterate over a range of numbers?
??x
The `IntStream.range(start, endExclusive)` method generates an integer stream from the starting number (inclusive) up to but not including the ending number (exclusive). This is ideal for processing a sequence where the upper bound is not included.

Example:
```java
IntStream.range(0, months.length).forEach(i -> System.out.println("Month " + months[i]));
```

x??

---
#### Counting by Increments with For Loops
For incrementing sequences or custom ranges, you can use traditional for loops. This provides flexibility in controlling the loop's logic and can be used to count by specific increments.

:p How do you use a for loop to count by 3 from 11 to 27?
??x
You can use a for loop with an increment statement to count by a specific number, such as 3. This example demonstrates counting from 11 to 27 in steps of 3.

Example:
```java
for (int i = 11; i <= 27; i += 3) {
    System.out.println("i = " + i);
}
```

x??

---

