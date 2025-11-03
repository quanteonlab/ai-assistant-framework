# Flashcards: Angie-Papple-Johnston---ASVAB-2023-_-2024-For-Dummies_processed (Part 12)

**Starting Chapter:** Dealing with more than one operation in a sequence. Working on Both Sides of the Line Fractions

---

#### Finding Patterns in Sequences
Background context explaining the importance of recognizing patterns in sequences. This involves understanding how numbers change and whether operations like addition or multiplication are involved.

:p What is a sequence where numbers both increase and decrease, indicating more than one operation?
??x
A sequence that shows both increasing and decreasing values suggests multiple operations such as "add 1, subtract 1, add 2, subtract 2." For example:
2, 3, 2, 4.
x??

---
#### Predicting Next Numbers in Sequences
Background context on predicting the next numbers in a sequence by identifying the pattern. This requires careful observation and sometimes manual effort.

:p What are the next numbers for the sequence: 6, 13, 27, 55, . . . ?
??x
The sequence increases rapidly, indicating multiplication or another significant operation. Observing that each term is roughly double the previous plus an increment (6+7=13, 13*2-1=25~27, 27*2-1=53~55), suggests a pattern of multiplying by 2 and subtracting 1.

The next number would be:
??x
111
x??

---
#### Another Sequence Problem
Background on identifying patterns in sequences. This example involves simpler operations but requires careful observation.

:p What are the next numbers for the sequence: 5, 16, 27, 38, . . . ?
??x
This sequence increases by a consistent amount each time (16-5=11, 27-16=11, 38-27=11), indicating an addition of 11.

The next number would be:
??x
49
x??

---
#### Third Sequence Problem
Background on identifying patterns in sequences. This example involves a combination of operations that may not be immediately obvious.

:p What are the next numbers for the sequence: 8, 14, 22, 36, 58, . . . ?
??x
This sequence increases by increasingly larger amounts each time (14-8=6, 22-14=8, 36-22=14, 58-36=22), suggesting a pattern where the increment itself increases.

The next number would be:
??x
94
x??

---
#### Averaging Mean, Median, and Mode in a Range
Background on averages: mean (arithmetic average) is typically used to find the typical value. It involves adding all numbers and dividing by their count.

:p What is the arithmetic mean of 35, 35, and 50?
??x
To calculate the mean, add the numbers and divide by their count:
(35 + 35 + 50) / 3 = 120 / 3 = 40.

The answer is:
??x
40
x??

---
#### Additional Averages on ASVAB
Background on different types of averages, including median (middle value in a sorted list) and mode (most frequent number).

:p What is the median of scores 80, 85, and 90?
??x
To find the median, sort the numbers: 80, 85, 90. The middle number is:
85.

The answer is:
??x
85
x??

---
#### Mode in ASVAB Scores
Background on mode, which is the most frequent score among a set of data.

:p What is the mode of scores 35, 35, and 50?
??x
The mode is the number that appears most frequently. In this case:
35.

The answer is:
??x
35
x??

---

#### Median Concept
Background context explaining the median. The median is the middle value in a set of ordered numbers, and it can be found by arranging numbers from smallest to largest.
:p How do you find the median in a set of numbers?
??x
To find the median, first, order the numbers from smallest to largest. Then, identify the number that splits the list into two halves; this is the median.
For example, for the data set 47, 56, 58, 63, 100, arrange them in order: 47, 56, 58, 63, 100. The middle number, 58, is the median.
x??

---

#### Mode Concept
Background context explaining the mode. The mode is the value that appears most frequently in a list of numbers. If no values are repeated, there's no mode.
:p What does the mode represent?
??x
The mode represents the most frequent number or value in a set of data. For instance, if you have test scores 35, 35, and 50, the mode is 35 because it appears more frequently than any other score.
x??

---

#### Fractions Concept
Background context explaining fractions using pizza as an example. A fraction represents parts of a whole (like slices of a pizza). The top number (numerator) indicates how many parts are considered, and the bottom number (denominator) shows the total number of equal parts the whole is divided into.
:p What are the components of a fraction?
??x
A fraction consists of two parts: the numerator (the top number, indicating how many parts are being considered) and the denominator (the bottom number, showing the total number of equal parts in the whole).
For example, in 3/5, the numerator is 3, meaning three slices out of five.
x??

---

#### Common Denominators: Finding a Common Denominator
Background context explaining how to find a common denominator when adding or subtracting fractions. This involves ensuring that both fractions have the same bottom number (denominator) so they can be combined properly.
:p How do you find a common denominator for two fractions?
??x
To find a common denominator, divide the larger denominator by the smaller one. If the result is an integer (no remainder), multiply the fraction with the smaller denominator by this quotient to get a new equivalent fraction with the same denominator as the other fraction.
For example, if you want to add 3/5 and 3/10:
- Divide 10 by 5: The result is 2.
- Multiply both the numerator and the denominator of 3/5 by 2: This gives you 6/10.
Now, both fractions have a common denominator (10).
x??

---

#### Example Pseudocode for Finding a Common Denominator
:p Write pseudocode to find a common denominator.
??x
```
function findCommonDenominator(num1, denom1, num2, denom2):
    if denom1 > denom2:
        largerDenom = denom1
        smallerDenom = denom2
    else:
        largerDenom = denom2
        smallerDenom = denom1
    
    quotient = largerDenom / smallerDenom  # Perform integer division to get the quotient

    if remainder(largerDenom, smallerDenom) == 0:  # If there's no remainder
        commonDenominator = largerDenom
    else:
        commonDenominator = lcm(denom1, denom2)  # Use least common multiple if necessary
    
    return commonDenominator

function multiplyFractionByQuotient(numerator, denominator, quotient):
    newNumerator = numerator * quotient
    newDenominator = denominator * quotient
    return (newNumerator, newDenominator)
```
x??

#### Adding Fractions with Different Denominators
Background context: When adding fractions with different denominators, you need to find a common denominator that both denominators can divide into evenly. This involves multiplying the denominators together or finding the least common multiple (LCM) of the denominators.

:p How do you add two fractions with different denominators?
??x
To add two fractions with different denominators, first find a common denominator by either multiplying the denominators or using the LCM method. Then convert each fraction to an equivalent fraction with this common denominator and finally add the numerators.

For example:
- Adding 3/5 and 1/6 involves finding a common denominator (which is 30 in this case). Convert 3/5 to 18/30 by multiplying both numerator and denominator by 6, and convert 1/6 to 5/30 by multiplying both numerator and denominator by 5. Then add the numerators: 18 + 5 = 23.

??x
To verify this:
```java
public class FractionAddition {
    public static void main(String[] args) {
        int num1 = 3, denom1 = 5;
        int num2 = 1, denom2 = 6;

        // Finding common denominator (LCM)
        int lcm = denom1 * denom2; // In this case, it is 30

        // Converting fractions to equivalent forms
        int numerator1 = num1 * (lcm / denom1); // 18
        int numerator2 = num2 * (lcm / denom2); // 5

        // Adding the numerators
        int resultNumerator = numerator1 + numerator2; // 23
        System.out.println("The sum is " + resultNumerator + "/" + lcm);
    }
}
```
x??

---

#### Multiplying Fractions and Reducing Them to Lowest Terms
Background context: Multiplying fractions involves multiplying the numerators together and then multiplying the denominators. After multiplication, you may need to reduce (simplify) the resulting fraction by dividing both the numerator and the denominator by their greatest common divisor (GCD).

:p How do you multiply and reduce fractions?
??x
To multiply and reduce fractions:
1. Multiply the numerators.
2. Multiply the denominators.
3. Simplify the resulting fraction by finding the GCD of the numerator and the denominator, and dividing both by this number.

For example:
- To multiply 3/4 and 2/5: 
  - Numerator: \(3 \times 2 = 6\)
  - Denominator: \(4 \times 5 = 20\)
  - The fraction is now 6/20, which can be reduced by dividing both the numerator and denominator by their GCD (which is 2):
    - Reduced form: \(6/20\) becomes \(3/10\).

??x
To verify this:
```java
public class FractionMultiplication {
    public static void main(String[] args) {
        int num1 = 3, denom1 = 4;
        int num2 = 2, denom2 = 5;

        // Multiply numerators and denominators
        int numeratorResult = num1 * num2; // 6
        int denominatorResult = denom1 * denom2; // 20

        // Simplifying the fraction by GCD (which is 2 in this case)
        int gcd = gcd(numeratorResult, denominatorResult);
        System.out.println("The simplified result is " + (numeratorResult / gcd) + "/" + (denominatorResult / gcd));
    }

    public static int gcd(int a, int b) {
        if (b == 0) return a;
        return gcd(b, a % b);
    }
}
```
x??

---

#### Dividing Fractions Using Reciprocals
Background context: To divide fractions, you multiply the first fraction by the reciprocal of the second fraction. The reciprocal of a number is obtained by flipping the number (i.e., swapping its numerator and denominator).

:p How do you divide fractions using reciprocals?
??x
To divide fractions:
1. Take the reciprocal of the divisor (the fraction following the division symbol).
2. Change the division operation to multiplication.
3. Multiply the numerators together.
4. Multiply the denominators together.

For example:
- To divide 5/6 by 3/4, first find the reciprocal of 3/4 which is 4/3.
- Now multiply: \( \frac{5}{6} \times \frac{4}{3} = \frac{5 \times 4}{6 \times 3} = \frac{20}{18} \).
- Simplify the fraction if possible (in this case, 20/18 can be simplified to 10/9).

??x
To verify this:
```java
public class FractionDivision {
    public static void main(String[] args) {
        int num1 = 5, denom1 = 6;
        int num2 = 3, denom2 = 4;

        // Reciprocal of the second fraction
        int reciprocalNum2 = denom2; // 4
        int reciprocalDenom2 = num2; // 3

        // Multiplying the first fraction by the reciprocal of the second
        int resultNumerator = num1 * reciprocalNum2; // 5 * 4 = 20
        int resultDenominator = denom1 * reciprocalDenom2; // 6 * 3 = 18

        // Simplifying the fraction if possible (GCD is 2 here)
        int gcd = gcd(resultNumerator, resultDenominator);
        System.out.println("The simplified result is " + (resultNumerator / gcd) + "/" + (resultDenominator / gcd));
    }

    public static int gcd(int a, int b) {
        if (b == 0) return a;
        return gcd(b, a % b);
    }
}
```
x??

---

