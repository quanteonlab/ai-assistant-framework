# Flashcards: Angie-Papple-Johnston---ASVAB-2023-_-2024-For-Dummies_processed (Part 79)

**Starting Chapter:** Dealing with more than one operation in a sequence. Working on Both Sides of the Line Fractions

---

#### Finding Patterns in Sequences

Background context explaining the importance of identifying patterns in sequences. This involves understanding how numbers change relative to each other and recognizing common operations like addition, subtraction, multiplication, or division.

:p What is the pattern for the sequence 6, 13, 27, 55, ...?
??x
The answer with detailed explanations.
To find the pattern in the sequence 6, 13, 27, 55, ..., observe the differences between consecutive terms:
- $13 - 6 = 7 $-$27 - 13 = 14 $-$55 - 27 = 28 $ Notice that each difference is double the previous one:$7, 14, 28$. This suggests a pattern of doubling the increase:
- Next term should be $55 + 56 = 111 $(since $28 \times 2 = 56$).

Thus, the next number in the sequence is 111.
x??

---

#### Handling Multiple Operations in Sequences

Background context explaining that a single operation might not be sufficient to describe the entire sequence. The sequence may involve multiple operations like "add 1, subtract 1, add 2, subtract 2."

:p What pattern does the sequence 2, 3, 2, 4, ... follow?
??x
The answer with detailed explanations.
To identify the pattern in the sequence 2, 3, 2, 4, ..., observe the operations:
- From 2 to 3: add 1
- From 3 to 2: subtract 1
- From 2 to 4: add 2

This suggests an alternating pattern of adding and subtracting consecutive numbers. The next operation would be to add 2 (as it was the last subtraction, the next should be addition).

Thus, the next number in the sequence is 6.
x??

---

#### Averaging Mean, Median, and Mode

Background context explaining what averages are and why they matter in sequences or data sets. The arithmetic mean involves summing all numbers and dividing by the count of numbers.

:p What is the arithmetic mean (average) of the scores 35, 35, and 50?
??x
The answer with detailed explanations.
To find the arithmetic mean:
1. Sum the values: $35 + 35 + 50 = 120 $2. Divide by the number of data points (3):$\frac{120}{3} = 40$

Thus, the arithmetic mean is 40.
x??

---

#### Multiple Average Types

Background context explaining that besides the mean, the ASVAB may also ask for median and mode.

:p How do you find the median of a set of numbers?
??x
The answer with detailed explanations.
To find the median:
1. Arrange the numbers in ascending order: $35, 35, 50$
2. Identify the middle number: Since there are three numbers, the middle one is the second number.

Thus, the median of the set $35, 35, 50$ is 35.
x??

---

#### Multiple Average Types (Mode)

Background context explaining that besides mean and median, mode refers to the most frequently occurring value in a data set.

:p What is the mode of the scores 35, 35, and 50?
??x
The answer with detailed explanations.
To find the mode:
1. Identify the number(s) that occur most often: In this case, both 35 appear twice, while 50 appears only once.

Thus, the modes of the set $35, 35, 50$ are 35 and 35 (dual mode).

In practice, if a data set has multiple modes or no clear mode, it is called multimodal or having no mode.
x??

#### Median Definition
In statistics, the median is the middle value in a set of ordered numbers. It's particularly useful when dealing with skewed distributions or outliers that might affect other measures like the mean.

To find the median:
1. Order your data from smallest to largest.
2. Identify the middle number; if there’s an even number of values, take the average of the two middle numbers.

Example: For the dataset 47, 56, 58, 63, 100, order them and find the median.

:p How do you determine the median in a given set of data?
??x
The median is found by ordering the numbers from smallest to largest. In the provided dataset (47, 56, 58, 63, 100), after ordering, the middle number is 58, making it the median.
x??

---

#### Mode Definition
The mode in statistics is the value that appears most frequently in a set of numbers. A dataset can have more than one mode or no mode at all.

Example: For the scores 35, 35, and 50, the number 35 occurs twice, making it the mode since no other number repeats as much.

:p How do you determine the mode in a given set of data?
??x
The mode is identified by finding which value appears most frequently. In the example with scores (35, 35, and 50), the number 35 occurs twice, making it the mode.
x??

---

#### Fraction Representation
A fraction represents parts of a whole. The top number (numerator) indicates how many parts are considered, while the bottom number (denominator) shows how many equal parts the whole is divided into.

Example: In the fraction $\frac{3}{5}$, 3 slices out of 5 equal parts are taken.

:p What do the numerator and denominator represent in a fraction?
??x
In a fraction, the numerator represents the number of parts considered (top number), while the denominator indicates how many equal parts the whole is divided into (bottom number). For example, in $\frac{3}{5}$, 3 slices are taken from a pizza cut into 5 equal pieces.
x??

---

#### Common Denominators and Adding Fractions
To add or subtract fractions, they need to have the same denominator. This shared denominator is called a common denominator.

If the denominators don't match:
1. Divide the larger denominator by the smaller one.
2. If the result is an integer (no remainder), use that multiplier for both numerators and denominators of the fraction with the smaller denominator.

Example: To add $\frac{3}{5}$ and $\frac{3}{10}$:
- 10 can be divided evenly by 5, giving a quotient of 2.
- Multiply both numerator and denominator of $\frac{3}{5}$ by 2 to get $\frac{6}{10}$.

:p How do you find a common denominator when adding fractions?
??x
To add or subtract fractions with different denominators, first determine if one denominator can be divided evenly by the other. If so, use that division result as a multiplier for both the numerator and denominator of the fraction being adjusted. For example, to add $\frac{3}{5} + \frac{3}{10}$, since 10 can be evenly divided by 5 (quotient is 2), multiply both numerator and denominator of $\frac{3}{5}$ by 2 to get $\frac{6}{10}$.
x??

---

#### Pseudocode for Finding Common Denominators
Here's a simple pseudocode example to illustrate the process:

```plaintext
function findCommonDenominator(fraction1, fraction2):
    // Assume fraction1 and fraction2 are in the form of (numerator, denominator)
    
    if fraction2.denominator % fraction1.denominator == 0:
        // If second denominator is divisible by first
        commonDenominator = fraction2.denominator
    else:
        // Otherwise, find the least common multiple (LCM) manually or use a function to get it.
        LCM = lcm(fraction1.denominator, fraction2.denominator)
        commonDenominator = LCM
    
    return commonDenominator

// Example usage
commonDenom = findCommonDenominator((3, 5), (3, 10))
print("Common denominator is", commonDenom)
```

:p How does the pseudocode determine a common denominator?
??x
The pseudocode determines if one denominator can be evenly divided by the other. If so, it uses that division result to create a common denominator for adding fractions. Otherwise, it calculates the least common multiple (LCM) of the two denominators. In our example with $\frac{3}{5}$ and $\frac{3}{10}$, since 10 is divisible by 5, we use 10 as the common denominator.
x??

---

#### Finding Common Denominators for Fractions
Background context: To add fractions, you need a common denominator. This is because the denominators represent the total number of parts into which something (like a pizza) has been divided, and to add these parts accurately, they must be in terms of the same size.
:p How do you find a common denominator for two or more fractions?
??x
To find a common denominator for two or more fractions, start with the largest denominator. Multiply this by whole numbers (1, 2, 3, etc.) until it is divisible by all other denominators. In some cases, using the least common multiple (LCM) of the denominators can be faster.
??x
For example:
```java
// Example for finding LCM
public class LCMExample {
    public static int gcd(int a, int b) {
        if (b == 0) return a;
        return gcd(b, a % b);
    }

    public static int lcm(int a, int b) {
        return (a / gcd(a, b)) * b; // Using the formula: LCM(a,b) = |a*b| / GCD(a,b)
    }
}
```
x??

---

#### Adding Fractions with Different Denominators
Background context: When adding fractions that have different denominators, you must first find a common denominator. This is done by multiplying the original denominators if they don't divide into each other evenly.
:p How do you add 3/5 and 1/6?
??x
First, multiply the denominators (5 * 6 = 30) to get the common denominator. Then, adjust the numerators:
- For $\frac{3}{5}$, multiply both numerator and denominator by 6:$\frac{3*6}{5*6} = \frac{18}{30}$.
- For $\frac{1}{6}$, multiply both numerator and denominator by 5:$\frac{1*5}{6*5} = \frac{5}{30}$.

Now add the adjusted fractions:
$$\frac{18}{30} + \frac{5}{30} = \frac{23}{30}$$??x
```java
public class FractionAddition {
    public static void main(String[] args) {
        int numerator1 = 3, denominator1 = 5;
        int numerator2 = 1, denominator2 = 6;

        int lcmDenominator = lcm(denominator1, denominator2);
        int newNumerator1 = (lcmDenominator / denominator1) * numerator1;
        int newNumerator2 = (lcmDenominator / denominator2) * numerator2;

        System.out.println("The sum is: " + (newNumerator1 + newNumerator2) + "/" + lcmDenominator);
    }

    public static int gcd(int a, int b) {
        if (b == 0) return a;
        return gcd(b, a % b);
    }

    public static int lcm(int a, int b) {
        return (a / gcd(a, b)) * b;
    }
}
```
x??

---

#### Multiplying and Reducing Fractions
Background context: To multiply fractions, you simply multiply the numerators and denominators. Sometimes, this results in large fractions that can be simplified by dividing both the numerator and denominator by their common factors.
:p How do you simplify 6/10?
??x
You can divide both the numerator (6) and the denominator (10) by 2:
$$\frac{6}{10} = \frac{3}{5}$$

The fraction $\frac{3}{5}$ is in its simplest form because only 1 divides evenly into both numbers.
??x
```java
public class FractionSimplification {
    public static void main(String[] args) {
        int numerator = 6, denominator = 10;

        // Finding the GCD to simplify the fraction
        int gcdValue = gcd(numerator, denominator);
        System.out.println("The simplified form is: " + (numerator / gcdValue) + "/" + (denominator / gcdValue));
    }

    public static int gcd(int a, int b) {
        if (b == 0) return a;
        return gcd(b, a % b);
    }
}
```
x??

---

#### Dividing Fractions
Background context: To divide fractions, you multiply the first fraction by the reciprocal of the second fraction. The reciprocal is obtained by flipping the numerator and denominator.
:p How do you divide 1/3 by 2/5?
??x
To divide $\frac{1}{3}$ by $\frac{2}{5}$, follow these steps:
- Find the reciprocal of $\frac{2}{5}$, which is $\frac{5}{2}$.
- Multiply the first fraction by this reciprocal: $\frac{1}{3} \times \frac{5}{2} = \frac{5}{6}$.

The result is $\frac{5}{6}$.
??x
```java
public class FractionDivision {
    public static void main(String[] args) {
        int numerator1 = 1, denominator1 = 3;
        int numerator2 = 2, denominator2 = 5;

        // Finding the reciprocal and multiplying
        System.out.println("The division result is: " + (numerator1 * denominator2) + "/" + (denominator1 * numerator2));
    }
}
```
x??

---

