# Flashcards: Angie-Papple-Johnston---ASVAB-2023-_-2024-For-Dummies_processed (Part 67)

**Starting Chapter:** Subtest 2 Arithmetic Reasoning Answers

---

#### Cost Calculation for Donuts
Background context explaining how to calculate costs and profits. The problem involves multiplying the cost of making donuts by a factor and then finding the profit.

:p How do you determine the profit made on five dozen donuts if each dozen costs $0.45 to make and sells for $3.99?

??x
To find the profit, first calculate the cost to make one dozen donuts: \( 0.45 \times 4 = 1.80 \). The profit per dozen is then calculated by subtracting this from the selling price: \( 3.99 - 1.80 = 2.19 \). Since the baker sold five dozen, multiply the profit per dozen by five: \( 2.19 \times 5 = 10.95 \).

```java
// Pseudocode for calculating donut profit
public class DonutProfit {
    public static void main(String[] args) {
        double costPerDozen = 0.45 * 4; // Cost to make a dozen donuts
        double sellingPrice = 3.99;
        double profitPerDozen = sellingPrice - costPerDozen; // Profit per dozen
        int dozensSold = 5;
        double totalProfit = profitPerDozen * dozensSold; // Total profit

        System.out.println("Total profit: " + totalProfit);
    }
}
```
x??

---

#### Equation for Dimes and Quarters
Background context explaining how to set up equations involving different coin values.

:p How do you determine the number of dimes if the total value is $19.75, knowing that each dime is worth 10 cents and each quarter is worth 25 cents?

??x
Let \( x \) be the number of dimes. Then \( 100x \) represents the number of quarters because there are 100 cents in a dollar.

The equation based on their values would be:
\[ 0.10x + 0.25(100x) = 19.75 \]

Simplifying this, we get:
\[ 0.10x + 25x = 19.75 \]
\[ 25.10x = 19.75 \]
\[ x = \frac{19.75}{25.10} \approx 35 \]

Thus, the number of dimes is approximately 35.

```java
// Pseudocode for solving the coin problem
public class CoinValue {
    public static void main(String[] args) {
        double totalValue = 19.75;
        double dimeValue = 0.10;
        double quarterValue = 0.25;

        // Calculate number of dimes and quarters
        double x = totalValue / (dimeValue + 10 * quarterValue);
        System.out.println("Number of dimes: " + x);
    }
}
```
x??

---

#### Bricklayer's Fee Calculation
Background context explaining how to calculate fees based on area and cost per unit.

:p How do you determine the total fee a bricklayer charges if he charges $8 per square foot for a 12 by 16 feet patio?

??x
The area of the patio is calculated as:
\[ \text{Area} = 12 \times 16 = 192 \, \text{square feet} \]

Given that the bricklayer charges $8 per square foot, the total fee would be:
\[ \text{Total Fee} = 192 \times 8 = 536 \, \$ \]

```java
// Pseudocode for calculating the bricklayer's fee
public class BricklayerFee {
    public static void main(String[] args) {
        int length = 12;
        int width = 16;
        double costPerSquareFoot = 8;

        // Calculate area and total fee
        double area = length * width;
        double totalFee = area * costPerSquareFoot;
        System.out.println("Total Fee: " + totalFee);
    }
}
```
x??

---

#### Hourly Wage Calculation for Workers
Background context explaining how to set up equations involving different hourly wages.

:p How do you determine the hourly wage of Angie if Tim’s hourly wage is \( x^2 \) and Terry’s hourly wage is 32 times Angie's, given that Angie earns $58 per hour?

??x
Let \( x \) be Angie’s hourly wage. Then:
\[ x = 58 \]

Since Tim’s hourly wage is \( x^2 \), we calculate it as:
\[ x^2 = 58^2 = 3364 \]

And since Terry’s hourly wage is 32 times Angie's, the equation would be:
\[ 32x = 1792 \]

Thus, Terry’s hourly wage is \( 1792 \).

```java
// Pseudocode for calculating the workers' wages
public class WorkersWages {
    public static void main(String[] args) {
        double angieHourlyWage = 58;
        double timHourlyWage = Math.pow(angieHourlyWage, 2);
        double terryHourlyWage = 32 * angieHourlyWage;

        System.out.println("Tim's hourly wage: " + timHourlyWage);
        System.out.println("Terry's hourly wage: " + terryHourlyWage);
    }
}
```
x??

---

#### Machine Operation by Number of People
Background context explaining how to determine the number of machines that can be run based on people.

:p How do you determine how many machines two people can run if four people can run 8 machines?

??x
Two people is half as many as four people. Therefore, multiply the number of machines that four people can run by \( \frac{1}{2} \):

\[ \text{Number of Machines for Two People} = 8 \times \frac{1}{2} = 4 \]

Thus, two people can run 4 machines.

```java
// Pseudocode for calculating machine operation based on number of people
public class MachineOperation {
    public static void main(String[] args) {
        int machinesFourPeople = 8;
        double factor = 0.5;

        // Calculate number of machines for two people
        int machinesTwoPeople = (int)(machinesFourPeople * factor);
        System.out.println("Machines run by two people: " + machinesTwoPeople);
    }
}
```
x??

---

#### Season Ticket vs Daily Ticket Calculation
Background context explaining how to determine the cost-effectiveness of a season ticket.

:p How do you determine if it is cheaper to use a daily ticket for more than 62.3 days or buy a season ticket?

??x
To determine whether buying a season ticket is cheaper, calculate the total cost of using a daily ticket over 62.3 days and compare it to the cost of a season ticket.

Assume the daily ticket costs $240:
\[ \text{Cost of Daily Ticket for 62.3 Days} = 240 \times 62.3 \]

If this cost is higher than the season ticket price, using the season ticket would be cheaper. For example, if a season ticket costs $1975:
\[ 240 \times 62.3 > 1975 \]
\[ 14952 > 1975 \]

Thus, it is not cheaper to use the daily ticket for more than 62.3 days.

```java
// Pseudocode for calculating season ticket vs daily ticket cost
public class TicketCost {
    public static void main(String[] args) {
        double dailyTicketCost = 240;
        int days = (int)(62.3);
        double seasonTicketCost = 1975;

        // Calculate total cost of daily tickets for the number of days
        double totalDailyTicketCost = dailyTicketCost * days;
        
        if (totalDailyTicketCost > seasonTicketCost) {
            System.out.println("Season ticket is cheaper.");
        } else {
            System.out.println("Daily ticket is cheaper.");
        }
    }
}
```
x??

---

#### Pipe Length Conversion and Calculation
Background context explaining how to convert pipe lengths from feet and inches.

:p How do you determine the total number of feet of pipe needed if each pipe is 3 feet, 6 inches long and 43 pipes are required?

??x
First, convert the length of one pipe into feet. Since 1 foot = 12 inches:
\[ \text{Length in Feet} = 3 + \frac{6}{12} = 3.5 \, \text{feet} \]

Then multiply this by the number of pipes needed:
\[ \text{Total Length} = 43 \times 3.5 = 150.5 \, \text{feet} \]

Thus, you need 150.5 feet of pipe.

```java
// Pseudocode for calculating total pipe length
public class PipeLength {
    public static void main(String[] args) {
        int pipesRequired = 43;
        double lengthInFeet = 3 + (6 / 12); // Convert inches to feet

        // Calculate total length of pipe needed
        double totalPipeLength = pipesRequired * lengthInFeet;
        System.out.println("Total Pipe Length: " + totalPipeLength);
    }
}
```
x??

---

#### Consecutive Odd Numbers Product
Background context explaining how to find the product of consecutive odd numbers.

:p How do you determine the pair of consecutive odd numbers that have a product of 399?

??x
To solve this, start by checking pairs of consecutive odd numbers around the square root of 399. The square root of 399 is approximately 19.97, so we check 19 and 21:

\[ 19 \times 21 = 399 \]

Alternatively, set up an algebraic equation:
Let \( x \) be the first number and \( x + 2 \) be the second number (since they are consecutive odd numbers):
\[ x(x + 2) = 399 \]
\[ x^2 + 2x - 399 = 0 \]

Solve this quadratic equation using factoring or the quadratic formula:
\[ (x - 19)(x + 21) = 0 \]
Thus, \( x = 19 \) or \( x = -21 \).

Since we are dealing with positive numbers, the pair is 19 and 21.

```java
// Pseudocode for finding consecutive odd number product
public class OddNumberProduct {
    public static void main(String[] args) {
        int targetProduct = 399;
        
        // Check pairs of consecutive odd numbers around the square root of the target product
        for (int x = (int)Math.sqrt(targetProduct); ; x += 2) {
            if (x * (x + 2) == targetProduct) {
                System.out.println("The pair is: " + x + ", " + (x + 2));
                break;
            }
        }
    }
}
```
x??

---

#### Diagonal of a Rectangle

The formula for the length of the diagonal \(d\) of a rectangle is given by:

\[ d = \sqrt{l^2 + w^2} \]

where \(l\) and \(w\) are the sides of the rectangle.

:p Calculate the diagonal of a rectangle with side lengths 5 and 12.
??x
The diagonal can be calculated using the Pythagorean theorem. Given the sides \(l = 5\) and \(w = 12\):

\[ d = \sqrt{5^2 + 12^2} = \sqrt{25 + 144} = \sqrt{169} = 13 \]

So, the diagonal is 13 units.

x??

---

#### Length of a Rectangle

Given the perimeter \(P\) and one side length \(l\), you can find the other side length \(w\) using the formula for the perimeter of a rectangle:

\[ P = 2(l + w) \]

:p Given a rectangle with perimeter 100 inches and length 15 inches, what is the width?
??x
Given:
\[ P = 100 \]
\[ l = 15 \]

Using the formula for the perimeter of a rectangle:

\[ 100 = 2(15 + w) \]
\[ 100 = 30 + 2w \]
\[ 70 = 2w \]
\[ w = 35 \]

So, the width is 35 inches.

x??

---

#### Distance Conversion

If a map scale shows that 2 inches represent 3 miles, then 1 inch represents \(1.5\) miles.

:p Convert 9.5 inches on the map to actual distance.
??x
Given:
\[ \text{Scale} = 2 \text{ inches} : 3 \text{ miles} \]

So, 1 inch represents:

\[ \frac{3}{2} = 1.5 \text{ miles} \]

Therefore, 9.5 inches on the map represent:

\[ 9.5 \times 1.5 = 14.25 \text{ miles} \]

The actual distance is approximately 14.25 miles.

x??

---

#### Area of a Rectangle

To find the area of a rectangle, you use the formula:

\[ \text{Area} = l \times w \]

:p Calculate the area of a canvas that is 10 feet by 14 feet.
??x
Given:
\[ l = 10 \text{ feet} \]
\[ w = 14 \text{ feet} \]

Using the formula for the area:

\[ \text{Area} = 10 \times 14 = 140 \text{ square feet} \]

The area of the canvas is 140 square feet.

x??

---

#### Percentage Calculation

To find a percentage, use the formula:

\[ \text{Percentage} = \left( \frac{\text{Part}}{\text{Whole}} \right) \times 100 \]

:p Calculate the percentage profit if the sale price is $530 and the profit is $30.
??x
Given:
\[ \text{Sale Price} = \$530 \]
\[ \text{Profit} = \$30 \]

Using the formula for the percentage:

\[ \text{Percentage Profit} = \left( \frac{30}{530} \right) \times 100 \approx 5.6\% \]

The profit is approximately 5.6%.

x??

---

#### Division and Fraction

To find how many of a certain fraction fit into a whole number, you divide the whole by the denominator.

:p Determine how much ribbon was used for each dress if 34 yards were used to make 4 dresses.
??x
Given:
\[ \text{Total Ribbon} = 3.4 \text{ yards} \]
\[ \text{Number of Dresses} = 4 \]

Using division:

\[ \text{Ribbon per Dress} = \frac{3.4}{4} = 0.85 \text{ yards} \]

Three-quarters (0.75) of a yard is used for each dress.

x??

---

#### Time and Distance

Use the formula for time when distance and speed are known:

\[ t = \frac{\text{Distance}}{\text{Speed}} \]

:p If two bikes start at different times, calculate the meeting point.
??x
Given:
- First bike travels at 12 mph with a 48-minute head start (0.8 hours).
- Second bike travels at 14 mph.

First, find the distance between them when the second bike starts:

\[ \text{Distance} = 12 \times 0.8 = 9.6 \text{ miles} \]

Their combined speed is:

\[ 12 + 14 = 26 \text{ mph} \]

Using the formula for time:

\[ t = \frac{9.6}{26} \approx 0.365 \text{ hours} \]

Convert to minutes (0.365 * 60 ≈ 22 minutes).

So, they meet at \(2:09 + 21 \text{ minutes} = 2:30\).

x??

---

#### Sales Amount Calculation
Background context: This problem involves adding up sales amounts and then performing a multiplication to find out how much money is left. The relevant operation here is basic arithmetic.

:p What are the total sales, and what percentage of this amount does the vendor have left?
??x
To solve this, first add the sales amounts together:
\[
0.25 + 0.32 + 0.318 = 0.898
\]

Next, multiply the total sales by \( \frac{3}{4} \):
\[
0.898 \times \frac{3}{4} = 0.6735
\]

So, the vendor has $0.6735 left.
x??

---

#### Pair of Socks Calculation
Background context: This problem involves dividing a total amount by a unit price to determine how many pairs of socks can be bought. It requires understanding whole number division and estimation for quick calculation.

:p How many pairs of socks can the recruit buy?
??x
To find out, divide $30.00 by $3.95:
\[
30 \div 3.95 = 7.5847...
\]

Since we need a whole number of pairs, the recruit can buy 7 pairs of socks.
x??

---

#### Weight Calculation
Background context: This problem involves subtracting two weights and converting units to perform the calculation. The task is to find the difference in weight between a crate and a puppy.

:p What is the difference in weight between the crate and the puppy?
??x
First, convert 60 pounds, 5 ounces to an equivalent value:
\[
60 \text{ pounds} + 5 \text{ ounces} = 59 \text{ pounds} + (5/16) \text{ pounds}
\]

Now subtract the weight of the puppy from this total:
\[
(59 \text{ pounds} + 5 \text{ ounces}) - 43 \text{ pounds} - 7 \text{ ounces} = 16 \text{ ounces}
\]

So, the difference is 16 ounces.
x??

---

#### Probability Calculation
Background context: This problem involves calculating the probability that all five computers are defective. The key here is understanding how to multiply probabilities for independent events.

:p What is the probability that all five computers will be defective?
??x
The probability of one computer being defective is \( \frac{150}{200} = 0.75 \). For all five computers, multiply this probability:
\[
0.75 \times 0.75 \times 0.75 \times 0.75 \times 0.75 = 0.2373
\]

Rounding to four decimal places gives \( 0.000076 \).
x??

---

#### Area and Square Yards Calculation
Background context: This problem involves finding the total area of a house by summing up the areas of its bedrooms and then converting square feet to square yards.

:p What is the total number of square yards needed for carpeting?
??x
First, calculate the area of each bedroom:
\[
12 \times 14 = 168 \text{ sq ft}
\]
\[
12 \times 10 = 120 \text{ sq ft}
\]
\[
8 \times 12 = 96 \text{ sq ft}
\]

Add these areas together:
\[
168 + 120 + 96 = 384 \text{ sq ft}
\]

Convert square feet to square yards (since \( 9 \) sq ft make a sq yard):
\[
384 \div 9 = 42.67 \approx 43 \text{ sq yd}
\]
x??

---

#### Typing Pages Calculation
Background context: This problem involves dividing the total number of pages to be typed by the typing rate per hour to find out how many hours it will take.

:p How many hours does it take Rafael to type 126 pages if he can type 9 pages per hour?
??x
To determine this, divide the total number of pages by the number of pages Rafael can type per hour:
\[
126 \div 9 = 14 \text{ hours}
\]
x??

---

#### Time Allocation Calculation
Background context: This problem involves dividing a group of students into smaller groups for playing and then determining how much time each student plays.

:p How many minutes does each student get to play?
??x
First, determine the number of groups by dividing the total number of students by those who can play at once:
\[
48 \div 12 = 4 \text{ groups}
\]

Next, divide the total time available (60 minutes) by the number of groups:
\[
60 \div 4 = 15 \text{ minutes per group}
\]
x??

---

#### Late Fee Calculation
Background context: This problem involves subtracting the initial late charge from the total and then dividing to find out how many additional days the movie was overdue.

:p How many days is the movie overdue?
??x
First, subtract the first day’s late charge:
\[
8.25 - 0.625 = 7.625 \text{ dollars}
\]

Then divide by $1.25 to find additional days:
\[
7.625 \div 1.25 = 6.1 \approx 6 \text{ days (since partial days are not considered)}
\]

Add this to the first day of being late:
\[
6 + 1 = 7 \text{ days}
\]
x??

---

#### Calorie Calculation
Background context: This problem involves comparing calorie content between pudding and broccoli, and then determining how much broccoli can be eaten based on the same number of calories.

:p How many cups of broccoli can be consumed for 150 calories if a person can eat 0.5 cup of pudding?
??x
First, find out how many times more calories are in broccoli compared to pudding:
\[
150 \div 60 = 2.5 \text{ times}
\]

Then multiply the amount of pudding (0.5 cups) by this factor:
\[
2.5 \times 0.5 = 1.25 \text{ cups}
\]
x??

---

#### Barking Calculation
Background context: This problem involves determining how many barks a dog makes over a specific time period, given that it barks every 15 minutes.

:p How many times does the dog bark between 10 p.m. and 2 a.m.?
??x
Calculate the total hours in this period:
\[
4 \text{ hours}
\]

Since the dog barks every 15 minutes, calculate the number of barks per hour (4 times):
\[
4 \times 4 = 16 \text{ barks}
\]

Add one bark for the initial bark at the beginning of the period:
\[
16 + 1 = 17 \text{ barks}
\]
x??