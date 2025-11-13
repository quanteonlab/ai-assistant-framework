# Flashcards: 10A006---Java-Cookbook---Ian-F-Darwin_processed (Part 25)

**Starting Chapter:** See Also. 5.11 Using Complex Numbers. Problem. Solution

---

#### Using Complex Numbers in Java

Background context: Java does not provide native support for complex numbers, but you can use libraries or create your own classes to handle them. The Apache Commons Math library is a recommended choice as it provides comprehensive implementations.

If applicable, add code examples with explanations:

:p How do you represent and manipulate complex numbers using the Apache Commons Math library?
??x
To work with complex numbers in Java using the Apache Commons Math library, you can use the `Complex` class provided by this library. Here is a basic example of how to create and manipulate complex numbers:

```java
import org.apache.commons.math3.complex.Complex;

public class ComplexDemoACM {
    public static void main(String[] args) {
        // Create two complex numbers
        Complex c = new Complex(3, 5);
        Complex d = new Complex(2, -2);

        // Print the complex number and its real part
        System.out.println(c);
        System.out.println("c.getReal() = " + c.getReal());

        // Perform addition of two complex numbers
        System.out.println(c.add(d));

        // Perform multiplication of two complex numbers
        System.out.println(c.multiply(d));
    }
}
```

This code demonstrates creating complex numbers, accessing their real and imaginary parts, performing addition and multiplication.

x??

---
#### Creating a Custom Complex Number Class

Background context: If you need to implement complex number operations in Java without using external libraries, you can create your own `Complex` class. This is useful for understanding the underlying mechanisms or when no third-party library is available.

If applicable, add code examples with explanations:

:p How would you design a custom `Complex` class in Java?
??x
To design a custom `Complex` class in Java, you need to encapsulate the real and imaginary parts as instance variables. You should also provide methods for basic operations like addition, subtraction, multiplication, division, and magnitude calculation.

Here is an example of how you might implement such a class:

```java
public class Complex {
    /** The real part */
    private double r;
    /** The imaginary part */
    private double i;

    // Constructor to initialize the complex number
    public Complex(double rr, double ii) {
        r = rr;
        i = ii;
    }

    // Method to return a string representation of the complex number
    public String toString() {
        StringBuilder sb = new StringBuilder().append(r);
        if (i > 0) {
            sb.append('+'); // else append(i) appends - sign
        }
        return sb.append(i).append('i').toString();
    }

    // Method to get the real part of the complex number
    public double getReal() {
        return r;
    }

    // Method to get the imaginary part of the complex number
    public double getImaginary() {
        return i;
    }

    // Method to calculate the magnitude of the complex number
    public double magnitude() {
        return Math.sqrt(r * r + i * i);
    }

    // Static method to add two complex numbers
    public static Complex add(Complex c1, Complex c2) {
        return new Complex(c1.r + c2.r, c1.i + c2.i);
    }

    // Non-static method to add a complex number to itself
    public Complex add(Complex other) {
        return add(this, other);
    }

    // Static method to subtract two complex numbers
    public static Complex subtract(Complex c1, Complex c2) {
        return new Complex(c1.r - c2.r, c1.i - c2.i);
    }

    // Non-static method to subtract a complex number from itself
    public Complex subtract(Complex other) {
        return subtract(this, other);
    }

    // Static method to multiply two complex numbers
    public static Complex multiply(Complex c1, Complex c2) {
        return new Complex(c1.r * c2.r - c1.i * c2.i,
                           c1.r * c2.i + c1.i * c2.r);
    }

    // Non-static method to multiply a complex number by another
    public Complex multiply(Complex other) {
        return multiply(this, other);
    }

    // Static method to divide one complex number by another
    public static Complex divide(Complex c1, Complex c2) {
        double denominator = c2.r * c2.r + c2.i * c2.i;
        return new Complex(
                (c1.r * c2.r + c1.i * c2.i) / denominator,
                (c1.i * c2.r - c1.r * c2.i) / denominator);
    }
}
```

This class provides methods for basic arithmetic operations and a custom string representation.

x??

---
#### Complex Number Operations

Background context: Understanding the operations on complex numbers, such as addition, subtraction, multiplication, division, and magnitude calculation, is crucial in many scientific and engineering applications. These operations can be implemented using both static and non-static methods.

:p How do you add two complex numbers?
??x
To add two complex numbers, you simply add their real parts together and their imaginary parts together. Here's how you can implement this:

```java
public class Complex {
    // Non-static method to add a complex number to itself
    public Complex add(Complex other) {
        return add(this, other);
    }

    // Static method to add two complex numbers
    public static Complex add(Complex c1, Complex c2) {
        return new Complex(c1.r + c2.r, c1.i + c2.i);
    }
```

The `add` method takes a `Complex` object and adds it to another `Complex` object by summing their real and imaginary parts. The static version of the method achieves the same result but requires both complex numbers as parameters.

x??

---
#### Subtracting Complex Numbers

Background context: Similar to addition, subtraction of two complex numbers involves subtracting their corresponding real and imaginary parts. This operation can also be implemented using both static and non-static methods for flexibility.

:p How do you subtract one complex number from another?
??x
To subtract one complex number from another, you subtract the real part of the second number from the first's real part, and similarly for the imaginary parts. Here’s how to implement this:

```java
public class Complex {
    // Non-static method to subtract a complex number from itself
    public Complex subtract(Complex other) {
        return subtract(this, other);
    }

    // Static method to subtract two complex numbers
    public static Complex subtract(Complex c1, Complex c2) {
        return new Complex(c1.r - c2.r, c1.i - c2.i);
    }
```

The `subtract` method takes a `Complex` object and subtracts another `Complex` object by subtracting their real and imaginary parts. The static version of the method allows you to pass two complex numbers as parameters.

x??

---
#### Multiplying Complex Numbers

Background context: Multiplication of complex numbers involves using the distributive property (FOIL method) on the real and imaginary components. This operation is essential for many applications in engineering and science.

:p How do you multiply two complex numbers?
??x
To multiply two complex numbers, you use the FOIL method to distribute terms and combine like parts. Here’s how to implement this:

```java
public class Complex {
    // Non-static method to multiply a complex number by another
    public Complex multiply(Complex other) {
        return multiply(this, other);
    }

    // Static method to multiply two complex numbers
    public static Complex multiply(Complex c1, Complex c2) {
        return new Complex(
                c1.r * c2.r - c1.i * c2.i,
                c1.r * c2.i + c1.i * c2.r);
    }
```

The `multiply` method takes a `Complex` object and multiplies it with another `Complex` object by applying the FOIL method. The static version of the method allows you to pass two complex numbers as parameters.

x??

---
#### Dividing Complex Numbers

Background context: Division of complex numbers involves multiplying by the conjugate of the denominator to simplify the expression into a standard form. This is a fundamental operation in many mathematical and engineering applications.

:p How do you divide one complex number by another?
??x
To divide one complex number by another, you multiply the numerator and the denominator by the conjugate of the denominator. Here’s how to implement this:

```java
public class Complex {
    // Static method to divide one complex number by another
    public static Complex divide(Complex c1, Complex c2) {
        double denominator = c2.r * c2.r + c2.i * c2.i;
        return new Complex(
                (c1.r * c2.r + c1.i * c2.i) / denominator,
                (c1.i * c2.r - c1.r * c2.i) / denominator);
    }
```

The `divide` method takes two `Complex` objects and divides the first by the second by multiplying both by the conjugate of the denominator. The result is returned as a new `Complex` object.

x??

---

#### Handling Very Large Numbers: BigInteger and BigDecimal Overview
Background context explaining the need for handling very large numbers. Mention that `Long.MAX_VALUE` and `Double.MAX_VALUE` are limits within Java's standard numeric types, and sometimes these are insufficient.

:p What is the main reason for using `BigInteger` and `BigDecimal` in Java?
??x
The main reason for using `BigInteger` and `BigDecimal` in Java is to handle integer numbers larger than `Long.MAX_VALUE` or floating-point values larger than `Double.MAX_VALUE`. These classes provide methods to perform arithmetic operations on arbitrarily large integers and arbitrary precision decimals, respectively.

Example code demonstrating the use of `BigInteger`:
```java
System.out.println("Here's Long.MAX_VALUE: " + Long.MAX_VALUE);
BigInteger bInt = new BigInteger("3419229223372036854775807");
System.out.println("Here's a bigger number: " + bInt);
System.out.println("Here it is as a double: " + bInt.doubleValue());
```
x??

---

#### BigInteger Class Constructor
Background context explaining the constructor of `BigInteger`. Mention that the constructor takes a string to represent very large numbers, which cannot be represented using primitive types like `long`.

:p How does the `BigInteger` constructor work?
??x
The `BigInteger` constructor works by taking a string as an argument. This allows for representing and operating on integers larger than can fit into a `long`. For example:
```java
new BigInteger("3419229223372036854775807")
```
This constructor converts the string representation of the number into a `BigInteger` object.

x??

---

#### BigDecimal Class Usage
Background context explaining how to use `BigDecimal` for handling floating-point numbers. Mention that `BigDecimal` is immutable, meaning once constructed, it always represents the same value, and methods return new objects with mutated values.

:p What are some key features of the `BigDecimal` class?
??x
Key features of the `BigDecimal` class include:
- It is immutable, so any operation returns a new object.
- Supports arithmetic operations such as addition, subtraction, multiplication, division, etc., via corresponding methods.
- Provides methods for rounding and handling precision issues.

Example code demonstrating basic usage of `BigDecimal` in an arithmetic expression:
```java
public BigDecimal calculate(Object[] input) {
    BigDecimal tmp;
    for (int i = 0; i < input.length; i++) {
        Object o = input[i];
        if (o instanceof BigDecimal) {
            stack.push((BigDecimal) o);
        } else if (o instanceof String) {
            switch (((String) o).charAt(0)) {
                case '+':
                    stack.push((stack.pop()).add(stack.pop()));
                    break;
                case '*':
                    stack.push((stack.pop()).multiply(stack.pop()));
                    break;
                // - and /, order does matter
                case '-':
                    tmp = (BigDecimal) stack.pop();
                    stack.push((stack.pop()).subtract(tmp));
            }
        }
    }
    return stack.pop(); // Ensure the result is on top of the stack
}
```
x??

---

#### Stack-Based Calculator Example with BigDecimal
Background context explaining a simple stack-based calculator using `BigDecimal` as its numeric data type. Mention that such calculators are useful for parsing and evaluating arithmetic expressions.

:p How does the provided stack-based calculator work?
??x
The provided stack-based calculator works by using a `Stack<BigDecimal>` to hold operands and perform operations based on input operators. It supports basic arithmetic operations like addition, multiplication, subtraction, and handles operator precedence through a stack approach:
```java
public class BigNumCalc {
    public static Object[] testInput = {
        new BigDecimal("3419229223372036854775807.23343"),
        new BigDecimal("2.0"), 
        "*",
    };

    public static void main(String[] args) {
        BigNumCalc calc = new BigNumCalc();
        System.out.println(calc.calculate(testInput));
    }

    Stack<BigDecimal> stack = new Stack<>();

    public BigDecimal calculate(Object[] input) {
        BigDecimal tmp;
        for (int i = 0; i < input.length; i++) {
            Object o = input[i];
            if (o instanceof BigDecimal) {
                stack.push((BigDecimal) o);
            } else if (o instanceof String) {
                switch (((String) o).charAt(0)) {
                    case '+':
                        stack.push(stack.pop().add(stack.pop()));
                        break;
                    case '*':
                        stack.push(stack.pop().multiply(stack.pop()));
                        break;
                    // - and /, order does matter
                    case '-':
                        tmp = (BigDecimal) stack.pop();
                        stack.push(stack.pop().subtract(tmp));
                }
            }
        }
        return stack.pop(); // Ensure the result is on top of the stack
    }
}
```
x??

---

#### BigNumCalc Program
Background context: The provided Java code implements a basic calculator that can perform division operations on large numbers using `BigDecimal`. This is useful for handling very precise and large numerical values, which are common in financial or scientific applications.

:p What is the main function of the `BigNumCalc` program?
??x
The main function of the `BigNumCalc` program is to process a string containing an arithmetic expression with division operations. It evaluates the expression by using a stack to manage numbers and ensures that all operators are correctly applied, particularly focusing on handling division with precision.

```java
switch (operator) {
    case '/':
        tmp = stack.pop();
        stack.push((stack.pop()).divide(tmp, BigDecimal.ROUND_HALF_UP));
        break;
}
```
The `divide` method uses `BigDecimal.ROUND_HALF_UP` to round the result of the division operation to the nearest value with the halfway cases rounded away from zero.

x??

---

#### TempConverter Program
Background context: The first version of the `TempConverter` program prints a table of Fahrenheit temperatures and their corresponding Celsius conversions using basic arithmetic operations in Java. However, it outputs numbers with many decimal places that are not necessary for practical use.

:p How does the original `TempConverter` handle temperature conversion?
??x
The original `TempConverter` handles temperature conversion by using simple arithmetic formulas:

- To convert from Fahrenheit (F) to Celsius (C), the formula used is:
  $$C = \left( F - 32 \right) \times \frac{5}{9}$$- To convert from Celsius (C) to Fahrenheit (F), the formula used is:
$$

F = C \times \frac{9}{5} + 32$$

However, it outputs these values with many decimal places that are not practical for everyday use.

x??

---

#### TempConverter2 Program
Background context: The second version of `TempConverter` improves the formatting of output by using `printf` to control the number of decimal places displayed. This ensures that the printed temperatures are more readable and suitable for common usage.

:p How does `TempConverter2` improve the display of temperature values?
??x
`TempConverter2` improves the display of temperature values by controlling the precision with which they are printed using `printf`. The method `print` in `TempConverter2` formats both Fahrenheit (F) and Celsius (C) temperatures to 2 decimal places:

```java
System.out.printf("%6.2f %6.2f\n", f, c);
```
This formatting ensures that the output is more readable and practical for everyday use.

x??

---

