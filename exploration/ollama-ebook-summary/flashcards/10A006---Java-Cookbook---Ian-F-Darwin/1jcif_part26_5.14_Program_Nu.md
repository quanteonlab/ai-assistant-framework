# Flashcards: 10A006---Java-Cookbook---Ian-F-Darwin_processed (Part 26)

**Starting Chapter:** 5.14 Program Number Palindromes

---

#### Temperature Conversion Concept

Temperature conversion from Fahrenheit to Centigrade involves a linear transformation. The formula for converting $F $(Fahrenheit) to $ C$ (Centigrade) is:
$$C = \frac{5}{9} (F - 32)$$:p What is the formula for converting Fahrenheit to Centigrade?
??x
The formula for converting Fahrenheit to Centigrade is $C = \frac{5}{9} (F - 32)$.
x??

---

#### Palindrome Generation Concept

A palindrome is a sequence that reads the same forwards and backwards. For numbers, this means the number remains the same when its digits are reversed.

To generate a palindrome from any positive integer:
1. Reverse the digits of the number.
2. Add the reversed number to the original number.
3. Repeat until the result is a palindrome.

:p What is the basic approach for generating palindromes?
??x
The basic approach involves repeatedly adding a number to its reverse until the sum becomes a palindrome. The method `findPalindrome` uses recursion, where:
- If the current number is already a palindrome, return it.
- Otherwise, add the number to its reversed version and try again.

```java
public class Palindrome {
    public static boolean verbose = true;
    
    public static long findPalindrome(long num) {
        if (num < 0)
            throw new IllegalStateException("negative");
        
        if (isPalindrome(num))
            return num;
        
        if (verbose)
            System.out.println("Trying " + num);
        
        return findPalindrome(num + reverseNumber(num));
    }
    
    private static long reverseNumber(long num) {
        // Reversing logic here
    }
    
    private static boolean isPalindrome(long num) {
        // Checking palindrome logic here
    }
}
```
x??

---

#### Palindrome Check Concept

To check if a number is a palindrome:
1. Convert the number to an array of its digits.
2. Compare the array with its reverse.

:p How do you check if a number is a palindrome?
??x
To check if a number is a palindrome, convert it into an array of its digits and compare this array with its reversed version. The `isPalindrome` method performs this by:
- Initializing a digit array.
- Extracting each digit from the number.
- Storing these digits in the array.
- Comparing the array to its reverse.

```java
static boolean isPalindrome(long num) {
    if (num >= 0 && num <= 9)
        return true;
    
    int nDigits = 0;
    while (num > 0) {
        digits[nDigits++] = num % 10;
        num /= 10;
    }
    
    for (int i = 0; i < nDigits / 2; ++i) {
        if (digits[i] != digits[nDigits - 1 - i])
            return false;
    }
    
    return true;
}
```
x??

---

#### Reverse Number Concept

Reversing a number involves extracting each digit and forming a new number with those digits in reverse order.

:p How do you reverse a number?
??x
To reverse a number, extract its digits one by one from the least significant to the most significant. Store these digits in an array or list. Then construct a new number using these digits but in reversed order.

```java
private static long reverseNumber(long num) {
    int nDigits = 0;
    while (num > 0) {
        digits[nDigits++] = num % 10;
        num /= 10;
    }
    
    long reversedNum = 0;
    for (int i = 0; i < nDigits; ++i) {
        reversedNum = reversedNum * 10 + digits[i];
    }
    
    return reversedNum;
}
```
x??

---

#### Checking Palindrome Using Array Operations
Background context: This section describes an approach to checking whether a number is a palindrome using array operations. A number is a palindrome if it reads the same backward as forward.

:p How can you check if a number is a palindrome using array operations?
??x
To check if a number is a palindrome, we first need to break down the number into its individual digits and store them in an array. We then compare the digits from both ends towards the center. If all corresponding pairs of digits match, the number is a palindrome.

Here's the step-by-step process:

1. Initialize an array `digits` to store each digit.
2. Use a loop to extract each digit of the number and store it in the array.
3. Compare the first half of the digits with the second half by iterating from both ends towards the center.
4. If any pair of corresponding digits does not match, return false; otherwise, return true.

Here is the C/Java code snippet for this approach:

```java
public class PalindromeChecker {
    static boolean isPalindrome(long num) {
        int nDigits = 0;
        // Extract each digit and store in array `digits`
        while (num > 0) {
            digits[nDigits++] = num % 10; // Get the last digit
            num /= 10;                   // Remove the last digit
        }
        
        // Check if the number is a palindrome by comparing pairs of digits
        for (int i = 0; i < nDigits / 2; i++) {
            if (digits[i] != digits[nDigits - i - 1]) { // Compare pairs from both ends
                return false;
            }
        }
        return true;
    }
}
```

x??

---

#### Reversing a Number Using Array Operations
Background context: This section explains how to reverse a number by breaking it down into its individual digits, storing them in an array, and then reconstructing the reversed number.

:p How can you reverse a number using array operations?
??x
To reverse a number using array operations, follow these steps:

1. Initialize an array `digits` to store each digit.
2. Use a loop to extract each digit of the number and store it in the array.
3. Reconstruct the reversed number by iterating through the `digits` array from left to right, appending each digit to the result.

Here is the C/Java code snippet for this approach:

```java
public class NumberReverser {
    static long reverseNumber(long num) {
        int nDigits = 0;
        // Extract each digit and store in array `digits`
        while (num > 0) {
            digits[nDigits++] = num % 10; // Get the last digit
            num /= 10;                   // Remove the last digit
        }
        
        long ret = 0;
        // Reconstruct the reversed number from the `digits` array
        for (int i = 0; i < nDigits; i++) {
            ret *= 10;                  // Shift left to make space for next digit
            ret += digits[i];           // Append current digit
        }
        
        return ret;
    }
}
```

x??

---

#### Using StringBuilder for Reversing a Number
Background context: This section describes an alternative approach using `StringBuilder` to reverse a number, which is more concise and readable.

:p How can you reverse a number using `StringBuilder`?
??x
To reverse a number using `StringBuilder`, follow these steps:

1. Convert the number to a string.
2. Use `StringBuilder` to reverse the string representation of the number.
3. Parse the reversed string back into a long integer.

Here is the C/Java code snippet for this approach:

```java
public class PalindromeChecker {
    private static long reverseNumber(long num) {
        StringBuilder stringBuilder = new StringBuilder();
        stringBuilder.append(num); // Convert number to string
        return Long.parseLong(stringBuilder.reverse().toString()); // Reverse and convert back to long
    }
    
    public static boolean isPalindrome(long num) {
        long result = reverseNumber(num);
        return num == result; // Check if the original number is a palindrome
    }
}
```

x??

---

#### Finding Today’s Date: Using LocalDate, LocalTime, and LocalDateTime Classes
Background context: To find today's date or time, you can use the `LocalDate`, `LocalTime`, and `LocalDateTime` classes from the Java Time API. These classes provide methods like `now()` to get the current date and time, but they do not have public constructors; instead, they offer factory methods.

:p How do you find today’s date using the LocalDate class?
??x
To find today's date using the `LocalDate` class, you can call its `now()` method, which returns the current date based on the system clock. This method does not require any arguments and provides a simple way to get the current date.

```java
LocalDate dNow = LocalDate.now();
System.out.println(dNow);
```
x??

---

#### Finding Today’s Time: Using LocalTime Class
Background context: To find today's time, you can use the `LocalTime` class. Similar to `LocalDate`, it also provides a `now()` method to get the current local time.

:p How do you find today’s time using the LocalTime class?
??x
To find today's time using the `LocalTime` class, you call its `now()` method, which returns the current local time based on the system clock. This method does not require any arguments and provides a simple way to get the current time.

```java
LocalTime tNow = LocalTime.now();
System.out.println(tNow);
```
x??

---

#### Finding Today’s Date and Time: Using LocalDateTime Class
Background context: To find today's date and time, you can use the `LocalDateTime` class. This class combines both date and time information.

:p How do you find today’s date and time using the LocalDateTime class?
??x
To find today's date and time using the `LocalDateTime` class, you call its `now()` method, which returns the current local date and time based on the system clock. This method does not require any arguments and provides a simple way to get the current date and time.

```java
LocalDateTime now = LocalDateTime.now();
System.out.println(now);
```
x??

---

#### Passing a Clock Instance for Testing: Testable DateTime Class
Background context: In full-scale applications, it is recommended to pass a `Clock` instance into the `now()` methods. The `Clock` class is used internally to find the current time and can be overridden in tests to provide known dates or times.

:p How does the TestableDateTime class allow for passing a Clock instance?
??x
The `TestableDateTime` class allows you to pass a `Clock` instance by providing a setter method. This enables test code to plug in a fixed clock, making it easy to use a known date and time during testing.

```java
package datetime;

import java.time.Clock;
import java.time.LocalDateTime;

public class TestableDateTime {
    private static Clock clock = Clock.systemDefaultZone();

    public static void main(String[] args) {
        System.out.println("It is now " + LocalDateTime.now(clock));
    }

    public static void setClock(Clock clock) {
        TestableDateTime.clock = clock;
    }
}
```

To use a fixed clock in tests, you would call the `setClock()` method with an instance obtained from `Clock.fixed(Instant fixedInstant, ZoneId zone)`.

x??

---

#### Using Clock for Testing: Fixed Instant Clock
Background context: In testing, you might want to have a known date or time used so that you can compare against known output. The `Clock` class makes this easy by allowing the use of a fixed clock that does not tick.

:p How do you set up a fixed instant clock in TestableDateTime?
??x
To set up a fixed instant clock in the `TestableDateTime` class, you would call the `setClock()` method with an instance obtained from `Clock.fixed(Instant fixedInstant, ZoneId zone)`. This allows you to use a known date and time for testing.

```java
public static void main(String[] args) {
    Instant testDate = Instant.parse("2023-10-01T08:00:00Z");
    ZoneId zone = ZoneId.of("America/New_York");

    setClock(Clock.fixed(testDate, zone));
    System.out.println("It is now " + LocalDateTime.now());
}
```

This sets the clock to a fixed date and time that can be used in your tests.

x??

