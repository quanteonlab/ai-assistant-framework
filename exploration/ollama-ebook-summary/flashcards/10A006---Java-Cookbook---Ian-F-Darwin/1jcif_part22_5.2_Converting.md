# Flashcards: 10A006---Java-Cookbook---Ian-F-Darwin_processed (Part 22)

**Starting Chapter:** 5.2 Converting Numbers to Objects and Vice Versa. Problem. Solution. 5.3 Taking a Fraction of an Integer Without Using Floating Point

---

#### Number Parsing and Handling
Background context: The provided snippet discusses how to handle string input that might or might not represent a valid integer, and converting it accordingly. It mentions using `Integer.parseInt()` for parsing integers and handling exceptions with `NumberFormatException`.

:p What is the primary method used for parsing an integer from a string in Java?
??x
The `Integer.parseInt()` method is primarily used to convert a string representing an integer into an actual integer value.

Example code:
```java
int iValue;
try {
    iValue = Integer.parseInt(s);
} catch (NumberFormatException e2) {
    // Handle the case where s is not a valid integer
}
```

x??

---

#### Auto-Boxing and Unboxing in Java
Background context: The provided text discusses auto-boxing and unboxing, which are automatic conversions between primitive types and their corresponding wrapper classes. This is particularly useful for passing values to methods that expect `Object` parameters.

:p What is the term used for converting a primitive type to its corresponding object wrapper class?
??x
The process of converting a primitive type to its corresponding object wrapper class is called auto-boxing.

Example code:
```java
public static Integer foo(Integer i) {
    return i + 1; // Example method that takes an Integer parameter and returns an Integer
}
```

x??

---

#### Using Auto-Boxing and Unboxing in Practice
Background context: The example provided shows how Java automatically handles conversions between primitive types and their wrapper classes, making code more concise and readable.

:p In the given `AutoboxDemo` class, what is the result of the method call `foo(i)`?
??x
The method call `foo(i)` will return an Integer object that represents the value 43. Here's a detailed explanation:

```java
public static void main(String[] args) {
    int i = 42; // Primitive integer
    int result = foo(i); // Auto-boxing occurs here, converting 'i' to an Integer
    System.out.println(result); // Prints the value of the returned Integer object
}

public static Integer foo(Integer i) {
    return i + 1; // Unboxing occurs implicitly when adding 1 to the Integer
}
```

x??

---

#### Exception Handling in Number Parsing
Background context: The text mentions handling cases where a string might not represent a valid number using `NumberFormatException`. This is an important part of robust input validation.

:p What exception is caught when attempting to parse a non-numeric string as an integer?
??x
The `NumberFormatException` is caught when the provided string cannot be parsed into an integer. Here's how it works in the context given:

```java
try {
    int iValue = Integer.parseInt(s); // This might throw NumberFormatException if s is not a valid number
} catch (NumberFormatException e2) {
    System.out.println("Not a number: " + s); // Handle the error case
    return Double.NaN; // Return NaN to indicate invalid input
}
```

x??

---

#### More Advanced Parsing with `DecimalFormat`
Background context: The text suggests using `DecimalFormat` for more complex parsing needs, which is not covered in detail here. However, it is mentioned as an alternative or additional method.

:p What class is suggested for more complex number formatting and parsing?
??x
The `DecimalFormat` class is suggested for more complex number formatting and parsing. It allows for finer control over the way numbers are formatted and parsed compared to basic methods like `Integer.parseInt()`.

Example usage:
```java
DecimalFormat df = new DecimalFormat("###,##0.00");
String result = df.format(123456789); // Formats the number as "123,456,789.00"
```

x??

---

#### Auto-boxing and Unboxing
Background context: In Java, auto-boxing is the automatic conversion from a primitive type to its corresponding wrapper class. Similarly, unboxing converts an object of a wrapper class back into its primitive form. This mechanism simplifies working with both primitives and their objects.

:p What are the examples of auto-boxing and unboxing mentioned in the text?
??x
Auto-boxing is demonstrated by converting `int 42` to `Integer(42)`. Unboxing occurs when an `Integer` object returned from a method like `foo()` is automatically converted back into an `int`, which is then assigned to `result`.

Example of auto-unboxing:
```java
int result = foo();
```
where `foo()` returns an `Integer` value.

Auto-boxing example where it would be explicitly mentioned:
```java
Integer result = Integer.valueOf(123);
```

x??

---

#### Wrapper Class Methods for Boxing and Unboxing
Background context: Java provides wrapper classes for each of its primitive data types. These classes have methods like `valueOf()` to convert between the primitive type and its object form, as well as methods like `intValue()` to retrieve the value back in its primitive form.

:p How do you explicitly convert an `int` to an `Integer` and vice versa using wrapper class methods?
??x
To convert an `int` to an `Integer`, use the `valueOf(int)` method:
```java
Integer i1 = Integer.valueOf(42);
```
To get the primitive value back from an `Integer`, use the `intValue()` method:
```java
int i2 = i1.intValue();
```

Explanation: The `Integer` class methods allow for explicit type conversion, which can be useful in scenarios where clarity and control are important.

x??

---

#### Multiplying Integer by a Fraction Without Using Floating Point
Background context: When you need to multiply an integer by a fraction without using floating-point numbers, one approach is to multiply the integer by the numerator of the fraction and then divide the result by the denominator. This avoids the potential loss of precision that can occur with floating-point arithmetic.

:p How do you perform multiplication of an integer by a fraction without using floating point?
??x
To multiply an integer by a fraction without using floating point, follow these steps:
1. Multiply the integer by the numerator of the fraction.
2. Divide the result by the denominator.

Example code in Java:
```java
public class FractionMultiplication {
    public static void main(String[] args) {
        int integer = 10;
        int numerator = 3;
        int denominator = 5;

        // Multiply and divide to simulate fraction multiplication
        int result = (integer * numerator) / denominator;
        System.out.println(result);  // Output: 6
    }
}
```

Explanation: This method ensures that the operation remains within the integer data type, avoiding floating-point arithmetic. However, be cautious with division as it can lead to truncation if not handled carefully.

x??

---

#### Floating-Point Multiplication and Division
In floating-point arithmetic, direct multiplication or division can lead to inaccuracies due to the representation of numbers as fractions. It is crucial to understand how to handle these operations properly to avoid unexpected results.

:p How does direct multiplication of a fraction with an integer in Java result in incorrect answers?
??x
Direct multiplication between a fraction like 2/3 and an integer, such as `5`, can lead to an incorrect answer because the division `2/3` is performed first using integer arithmetic, which results in `0`. Then, multiplying this by `5` gives `0`. To get the correct result, you need to ensure that at least one of the operands is a floating-point number.

Example:
```java
double d1 = 2 / 3 * 5; // Incorrect: Result will be 0.0
```
This can be fixed by making sure the division operation involves a floating-point number:
```java
double d3 = 2d / 3d * 5; // Correct: Result will be approximately 3.3333333333333335
```
x??

---

#### Handling Division by Zero in Floating-Point Arithmetic
Division by zero in floating-point arithmetic does not result in an exception but instead returns the constant `POSITIVE_INFINITY` for positive numbers, `NEGATIVE_INFINITY` for negative numbers, and `NaN` (Not a Number) for other invalid results.

:p What happens when you divide a number by zero using floating-point arithmetic?
??x
When dividing a number by zero with floating-point arithmetic in Java, the following outcomes can occur:
- If the dividend is positive: The result will be `POSITIVE_INFINITY`.
- If the dividend is negative: The result will be `NEGATIVE_INFINITY`.
- For other invalid operations (e.g., 0/0): The result will be `NaN`.

Example code to check these constants:
```java
public static void main(String[] argv) {
    double d = 123;
    double e = 0;
    if (d / e == Double.POSITIVE_INFINITY)
        System.out.println("Check for POSITIVE_INFINITY works");
    double s = Math.sqrt(-1);
    if (s == Double.NaN)
        System.out.println("Comparison with NaN incorrectly returns true");
    if (Double.isNaN(s))
        System.out.println("Double.isNaN() correctly returns true");
}
```
x??

---

#### Comparing Floating-Point Numbers
Direct comparison of floating-point numbers can lead to inaccuracies due to the inherent imprecision in their representation. It is recommended to compare with a small epsilon value.

:p How do you compare two floating-point numbers for equality?
??x
To accurately compare two floating-point numbers, it's not sufficient to use the `==` operator because of the precision issues. Instead, you should define an acceptable difference (epsilon) and check if the absolute difference is less than this value.

Example:
```java
public static boolean almostEqual(double a, double b, double epsilon) {
    return Math.abs(a - b) < epsilon;
}
```
For instance, to compare `a` and `b`, you might use an epsilon of `1e-9`.

x??

---

#### Rounding Floating-Point Numbers
Rounding floating-point numbers can be done using the built-in `Math.round()` method or by creating custom logic.

:p How do you round a floating-point number in Java?
??x
You can round a floating-point number using the `Math.round()` method, which rounds to the nearest integer. For example:

```java
double num = 3.5;
int roundedNum = (int) Math.round(num);
System.out.println(roundedNum); // Output: 4
```

Alternatively, you can create custom logic for rounding based on specific requirements.

x??

---

#### Heron's Formula and Floating-Point Arithmetic Differences
Background context: The example demonstrates how using `float` versus `double` can affect the accuracy of calculations, particularly with large numbers. Heron’s formula is used to calculate the area of a triangle given its side lengths.

Formula:
$$\text{Area} = \sqrt{s(s-a)(s-b)(s-c)}$$where $ s = \frac{a+b+c}{2}$and $ a, b, c$ are the sides of the triangle.

:p How does using `float` versus `double` in Heron's formula affect the result?
??x
Using `float` can lead to significant rounding errors due to its lower precision compared to `double`. In this example, the area calculation with `float` results in 0.0 because of these errors.

```java
public class Heron {
    public static void main(String[] args) {
        // Sides for triangle in float
        float af = 12345679.0f;
        float bf = 12345678.0f;
        float cf = 1.01233995f;
        float sf = (af + bf + cf) / 2.0f;
        float areaf = (float)Math.sqrt(sf * (sf - af) * (sf - bf) * (sf - cf));
        
        // Area of triangle in double
        double ad = 12345679.0;
        double bd = 12345678.0;
        double cd = 1.01233995;
        double sd = (ad + bd + cd) / 2.0;
        double aread = Math.sqrt(sd * (sd - ad) * (sd - bd) * (sd - cd));
        
        System.out.println("Single precision: " + areaf); // 0.0
        System.out.println("Double precision: " + aread); // 972730.0557076167
    }
}
```
x??

---
#### Strict-FP Keyword in Java
Background context: The `strictfp` keyword ensures that floating-point operations are performed consistently across different Java implementations, particularly for large-magnitude computations.

:p What does the `strictfp` keyword ensure?
??x
The `strictfp` keyword ensures that all floating-point operations adhere to strict IEEE 754 standards, which means operations will be consistent even on different hardware and software platforms. This is crucial for computations near the limits of representable values in a double.

```java
public class StrictFPExample {
    @strictfp
    public void compute() {
        float f1 = 12345679.0f;
        float f2 = 12345678.0f;
        float f3 = 1.01233995f;
        
        // Strict-FP ensures consistent results
    }
}
```
x??

---
#### Floating-Point Comparison with `equals()`
Background context: In Java, comparing floating-point numbers directly using `==` is unreliable due to precision issues. The `Float.equals()` and `Double.equals()` methods compare the values bit-for-bit.

:p How does `equals()` handle NaN?
??x
The `equals()` method for `Float` and `Double` returns true if both arguments are NaN. However, comparing object identities using `==` will return false because they are distinct objects.

```java
Float f1 = Float.valueOf(Float.NaN);
Float f2 = Float.valueOf(Float.NaN);

System.out.println(f1 == f2); // false (object identity)
System.out.println(f1.equals(f1)); // true (bitwise comparison of values)
```
x??

---

#### Comparing Floating-Point Numbers for Equality
When comparing floating-point numbers, directly using `==` can lead to inaccuracies due to rounding errors. A more reliable method is to compare them within a small tolerance or epsilon value.

:p How do you compare two floating-point numbers within an epsilon?
??x
To compare two floating-point numbers within an epsilon, use the following approach:

```java
public static boolean equals(double a, double b, double eps) {
    if (a == b) return true; // Direct comparison for exact equality
    return Math.abs(a - b) < eps; // Check if their difference is less than the epsilon value
}
```

This method first checks if `a` and `b` are exactly equal. If not, it then compares the absolute difference between them against a predefined tolerance (epsilon).

x??

---

#### Handling NaN in Floating-Point Numbers
NaN (Not-a-Number) values are special floating-point numbers that indicate an undefined or unrepresentable value resulting from an operation. Direct comparison with `==` on NaN will always return false, making it unreliable for checking NaN.

:p How do you handle NaN values during comparisons?
??x
For handling NaN values in comparisons, use the appropriate methods provided by the Double class:

```java
double nan1 = Double.NaN;
double nan2 = Double.NaN;

if (nan1 == nan2) {
    System.out.println("Comparing two NaNs incorrectly returns true.");
} else {
    System.out.println("Comparing two NaNs correctly reports false.");
}

// Using Double.equals() method to handle NaN
if (Double.valueOf(nan1).equals(Double.valueOf(nan2))) {
    System.out.println("Double(NaN).equals(NaN) correctly returns true.");
} else {
    System.out.println("Double(NaN).equals(NaN) incorrectly returns false.");
}
```

The `Double.equals()` method is preferred over direct comparison with `==` for NaN values because it treats two NaN values as not equal, which is the correct behavior.

x??

---

#### Rounding Floating-Point Numbers
When casting floating-point numbers to integers, Java simply truncates the decimal part. To round a floating-point number properly, use the `Math.round()` method. This method returns an `int` when given a `double` and an `long` when given a `float`.

:p How do you round a double value correctly?
??x
To round a double value correctly, use the `Math.round()` method:

```java
double value = 3.999999;
int roundedValue = Math.round(value); // This will return 4
```

The `Math.round()` method works by adding 0.5 to the number and then taking the floor of the result, effectively rounding it to the nearest integer.

x??

---

#### Custom Rounding Method for Floating-Point Numbers
You can create a custom rounding method if you need to use different rounding rules than the standard `Math.round()` method provided by Java.

:p How do you implement a custom round() method?
??x
To implement a custom round() method, you can follow this logic:

```java
public static long customRound(double value) {
    return (long)(value + 0.5); // Add 0.5 and truncate to get the rounded value
}
```

This method adds 0.5 to the input value before casting it to a `long`, effectively rounding it according to your specific requirements.

x??

---

