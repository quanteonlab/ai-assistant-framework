# Flashcards: Angie-Papple-Johnston---ASVAB-2023-_-2024-For-Dummies_processed (Part 80)

**Starting Chapter:** Adding and subtracting decimals

---

#### Converting Improper Fractions to Mixed Numbers
Background context: An improper fraction is a fraction where the numerator (top number) is larger than or equal to its denominator. To convert an improper fraction into a mixed number, you divide the numerator by the denominator and express any remainder as a new fraction with the same denominator.
:p How do you convert 7/3 into a mixed number?
??x
To convert 7/3 into a mixed number:
1. Divide the numerator (7) by the denominator (3). The quotient is 2, and the remainder is 1.
2. The result can be expressed as 2 with a remainder of 1 over the original denominator: $2\frac{1}{3}$.
??x
This process breaks down as follows:
- Divide 7 by 3 to get a quotient (2) and a remainder (1).
- Express this as a mixed number: $2 + \frac{1}{3} = 2\frac{1}{3}$.

---

#### Converting Mixed Numbers to Improper Fractions
Background context: A mixed number is a combination of a whole number and a fraction. To convert a mixed number back into an improper fraction, you combine the whole number with the fraction.
:p How do you convert $7\frac{2}{3}$ to an improper fraction?
??x
To convert $7\frac{2}{3}$ to an improper fraction:
1. Multiply the denominator (3) by the whole number (7):$3 \times 7 = 21$.
2. Add this result to the numerator of the fractional part: $21 + 2 = 23$.
3. The resulting fraction is $\frac{23}{3}$.

The process can be explained with the following steps:
- Multiply 7 (whole number) by 3 (denominator): $7 \times 3 = 21$.
- Add this to the numerator of the fractional part: $21 + 2 = 23$.
- The improper fraction is then $\frac{23}{3}$.

---

#### Expressing Fractions as Decimals
Background context: A decimal is a way of representing numbers in base ten. To convert a fraction to a decimal, divide the numerator by the denominator.
:p How do you convert $3/5$ into a decimal?
??x
To convert $3/5$ into a decimal:
1. Divide 3 (the numerator) by 5 (the denominator).
2. The result is 0.6.

The division can be performed as follows in pseudocode:

```pseudocode
decimal = 3 / 5
print(decimal)
```

This gives the answer:$3/5 = 0.6$.

---

#### Expressing Fractions as Percents
Background context: A percent is a ratio expressed per hundred. To convert a fraction to a percent, first convert it to a decimal and then move the decimal point two places to the right.
:p How do you convert 0.6 into a percent?
??x
To convert 0.6 into a percent:
1. Move the decimal point two places to the right: $0.6 \rightarrow 60$.
2. Add the percentage sign (%).

The process can be described as follows in code:

```java
double decimal = 0.6;
String percent = String.format("%.2f%%", decimal * 100);
System.out.println(percent);
```

This will output: $60\%$.

---

#### Converting to Repeating Decimals
Background context: Some fractions result in repeating decimals, where one digit or a sequence of digits repeats infinitely. These are often rounded for practical use.
:p How do you convert 2/3 into a decimal and what is its rounded form?
??x
To convert $2/3$ into a decimal:
1. Perform the division:$2 \div 3 = 0.6666...$(with the sixes repeating indefinitely).

The decimal representation of $2/3 $ is$0.\overline{6}$, which can be rounded to $0.67$.

This process involves recognizing that the digits after the decimal point repeat and rounding them appropriately.

---

---
#### Adding and Subtracting Decimals
When adding or subtracting decimals, you align the numbers by their decimal points. Perform the arithmetic as if they were whole numbers and then place the decimal point directly below where it is in the original numbers.

:p How do you add 0.55 to 14.583?
??x
Align the numbers by the decimal point:
```
  14.583
+   0.550
-------
  15.133
```
The answer is 15.133.
x??

---
#### Multiplying Decimals
To multiply decimals, follow these steps:

1. Multiply the numbers as if they were whole numbers.
2. Count and add the total number of decimal places in the factors.
3. Place the decimal point in the product so that it has the same total number of decimal places.

:p How do you multiply 3.77 by 2.8?
??x
First, ignore the decimals:
```
   377
 x  28
------
 3016 (377 * 28)
  7540 (377 * 20, shifted one place to the left)
-------
 10556
```
Count the total number of decimal places: there are two in 3.77 and one in 2.8, so three in total.
Place the decimal point in 10556, moving it three places from right to left:
The answer is 10.556.
x??

---
#### Dividing Decimals by Whole Numbers
To divide a decimal by a whole number, move the decimal point in the dividend so that it becomes a whole number. Perform the division on this new number and then place the decimal point in the quotient according to how many places you moved the decimal.

:p How do you divide 1254 by 4?
??x
Move the decimal point two places to the right in 1254, making it 125. Perform the division:
```
 31.25
------
 125   4
 -120
 -----
    50
 -48
 ----
     2 (remainder)
```
Move the decimal point two places back to the left, making the answer 0.3125.
x??

---
#### Dividing Decimals by Decimals with Same Decimal Places
To divide decimals where both numbers have the same number of decimal places, treat them as whole numbers after moving their decimal points right until they are integers.

:p How do you divide 0.15 by 0.25?
??x
Move the decimal point in both the divisor and dividend two places to the right:
```
   15
-----
  25
------
   60 (with the decimal points aligned)
```
The answer is 0.6.
x??

---
#### Dividing Decimals with Different Decimal Places
When dividing decimals where the divisor has more or fewer decimal places, move the decimal point in both the dividend and divisor to align them.

:p How do you divide 0.42 by 2.394?
??x
Move the decimal points two places to the right:
```
   4200
-------
 23940
------
   178 (with the decimal points aligned)
```
The answer is approximately 0.178.
x??

---

#### Converting Percents to Decimals and Fractions
Background context: When dealing with percentages on the ASVAB, understanding how to convert percents to decimals or fractions is essential. This helps in performing arithmetic operations such as addition, subtraction, multiplication, and division accurately.

:p How do you convert a percent to a decimal?
??x
To convert a percent to a decimal, simply remove the percentage sign (%) and move the decimal point two places to the left, adding zeros if necessary. For example, 45% becomes 0.45.
```java
public class PercentToDecimal {
    public static double convertPercentToDecimal(int percent) {
        return (double) percent / 100;
    }
}
```
x??

---

#### Calculating Percents: Examples and Operations
Background context: The ASVAB often asks for calculations involving percents, such as "15% off" or "an increase of 25%." To solve these problems, convert the percent to a decimal or fraction first.

:p What is 20% of 300?
??x
To find 20% of 300:
- Convert 20% to a decimal: 20 / 100 = 0.20.
- Multiply 0.20 by 300: 0.20 * 300 = 60.

Therefore, 20% of 300 is 60.
```java
public class PercentCalculation {
    public static int calculatePercent(int percent, int total) {
        double decimal = (double) percent / 100;
        return (int) (decimal * total);
    }
}
```
x??

---

#### Understanding and Using Ratios
Background context: Ratios are used to compare two quantities. They can be expressed as fractions or in the form of "a to b." In problems like computing gas mileage, you use ratios to determine how one quantity relates to another.

:p How do you compute your car's gas mileage if you drive 240 miles on 15 gallons of gas?
??x
To compute gas mileage:
- Use the ratio: miles per gallon (mpg) = total miles driven / total gallons used.
- Plug in the numbers: 240 miles / 15 gallons = 16 miles per gallon.

Therefore, your car's gas mileage is 16 miles per gallon.
```java
public class GasMileageCalculation {
    public static double calculateGasMileage(int totalMiles, int totalGallons) {
        return (double) totalMiles / totalGallons;
    }
}
```
x??

---

#### Working with Scale Drawings and Maps
Background context: Scale drawings are used to represent real-world objects or areas in a reduced form. The scale is often given as a ratio, such as 1 inch = 4 miles. You can use this information to find the actual distance represented by a drawing.

:p If the scale on a map is 1 inch : 250 miles, how many inches would represent 1,250 miles?
??x
To solve this:
- The scale tells you that 1 inch represents 250 miles.
- To find out how many inches represent 1,250 miles, set up the proportion: $\frac{1 \text{ inch}}{250 \text{ miles}} = \frac{x \text{ inches}}{1,250 \text{ miles}}$.

Solving for x:
$$x = \frac{1 \times 1,250}{250} = 5$$

Therefore, 5 inches on the map represent 1,250 miles.
```java
public class ScaleDrawing {
    public static int calculateMapDistance(int scaleMiles, int actualMiles) {
        return (int) (actualMiles / scaleMiles);
    }
}
```
x??

---

