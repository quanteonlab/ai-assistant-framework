# Flashcards: Angie-Papple-Johnston---ASVAB-2023-_-2024-For-Dummies_processed (Part 84)

**Starting Chapter:** All Math Isnt Created Equal Solving Inequalities

---

#### Factoring a Three-Term Equation (Trinomial)
Background context: Factoring a trinomial involves breaking down an expression with three terms into simpler factors. This process is essential for solving quadratic equations and simplifying algebraic expressions.

:p What are the steps to factor the trinomial $x^2 - 12x + 20$?
??x
The first step is to identify the factors of the first term, which in this case is $x^2 $. The factors are $ x \cdot x$.

Next, determine the signs for the parentheses. Since the third term (20) has a plus sign, the signs in the parentheses must be either both plus or both minus. However, since the second term ($-12x$) is negative, we need two negative numbers that multiply to 20 and add up to -12.

The factors of 20 that satisfy this condition are $-2 $ and$-10$, because:
- $(-2) \cdot (-10) = 20 $-$(-2) + (-10) = -12 $ Thus, the factored form is$(x - 2)(x - 10)$.

Therefore, the factors of $x^2 - 12x + 20 $ are$(x - 2)(x - 10)$.
??x

---

#### Solving Quadratic Equations
Background context: A quadratic equation is an equation that includes the square of a variable. The general form is $ax^2 + bx + c = 0 $, where $ a, b,$and $ c$are constants and $ a \neq 0$. Simple quadratic equations can be solved using the square root rule.

:p What is the process for solving a simple quadratic equation like $y^2 - 16 = 0$?
??x
To solve $y^2 - 16 = 0$, first, isolate the squared term:
- Add 16 to both sides: $y^2 = 16$ Next, take the square root of both sides. Remember that taking the square root gives both a positive and a negative solution:
- $\sqrt{y^2} = \pm \sqrt{16}$- Therefore,$ y = \pm 4 $So, the solutions are$ y = 4 $or$ y = -4$.
??x

---

#### Solving Complex Quadratic Equations
Background context: For complex quadratic equations like $x^2 - 5x - 10 = 0$, you can convert them into standard form and solve by factoring.

:p How do you factor the equation $x^2 - 5x - 10 = 0$?
??x
First, identify the factors of $-10 $ that add up to$-5 $. The factors are $-2 $ and$-3$, because:
- $(-2) \cdot (-3) = 6$(incorrect, need -10)
- Adjusting:$x^2 - 7x + 2x - 10$ Rewriting the equation:
$$x(x - 5) - 2(x - 5) = 0$$

Factor out the common term:
$$(x - 5)(x - 2) = 0$$

Thus, set each factor equal to zero:
- $x - 5 = 0 $ or$x - 2 = 0 $- Therefore,$ x = 5 $or$ x = 2 $The solutions are$ x = 5 $and$ x = 2$.
??x

---

#### Solving Inequalities
Background context: Inequalities involve expressions that are not equal. Common inequality symbols include <, >, ≤, ≥.

:p How do you solve the inequality $3x - 4 < 8$?
??x
To solve $3x - 4 < 8$, follow these steps:
1. Add 4 to both sides of the inequality:
   $$3x - 4 + 4 < 8 + 4$$

Simplifying, we get:
$$3x < 12$$2. Divide both sides by 3:
$$x < \frac{12}{3}$$

Simplifying further, we get:
$$x < 4$$

Therefore, the solution is $x < 4$.
??x

---

#### Number Line Representations of Inequalities
Background context: Inequalities can be represented on a number line. Common symbols used are <, >, ≤, ≥.

:p How do you represent the inequality $x \geq -1$ on a number line?
??x
To represent $x \geq -1$ on a number line:
1. Draw a horizontal line.
2. Place a closed circle at $-1 $ to indicate that$-1$ is included in the solution set.
3. Shade the region to the right of $-1 $, indicating all values greater than or equal to $-1$.

The representation looks like this:
```
---[-----]
  -2 -1 0
```

Where a closed circle at $-1 $ indicates that$-1 $ is part of the solution, and the shading to the right shows all numbers greater than$-1$.
??x

---
---

#### Simplifying Algebraic Expressions
Background context: Simplification of algebraic expressions involves combining like terms and reducing complex expressions to simpler forms. This is a fundamental skill for solving more advanced algebraic problems.

:p How do you simplify the expression $8x^3 + 4x - 9x^2 + x$?
??x
To simplify the expression, combine the like terms:

- The $x^2 $ term:$-9x^2 $- The $ x $term:$4x + x = 5x$ So, the simplified form is:
$$-9x^2 + 8x^3 + 5x$$

The code below demonstrates how to handle this in a simple pseudocode:

```pseudocode
function simplifyExpression(expression) {
    let terms = expression.split(' ');
    let coefficients = [];
    let variables = [];

    for (let term of terms) {
        if (term.includes('x')) {
            // Extract coefficient and variable parts
            let [coefficient, variable] = extractCoefficientAndVariable(term);
            coefficients.push(coefficient);
            variables.push(variable);
        }
    }

    // Combine like terms
    let simplifiedTerms = combineLikeTerms(coefficients, variables);

    return simplifyTerms(simplifiedTerms);
}

function extractCoefficientAndVariable(term) {
    if (term.includes('x')) {
        let coefficient = parseInt(term.split('x')[0]);
        let variable = 'x';
        return [coefficient, variable];
    } else {
        // Handle constant terms
        return [parseInt(term), 1]; 
    }
}

function combineLikeTerms(coefficients, variables) {
    // Logic to find and combine like terms
    // This is a placeholder function for demonstration purposes.
}
```

x??

---

#### Evaluating Expressions
Background context: Evaluating algebraic expressions involves substituting given values into the expression and simplifying it to find the result. This skill is crucial for understanding how variables affect mathematical outcomes.

:p Evaluate $3x^2 + 2x - 5 $ when$x = 4$.
??x
Substitute $x = 4$ into the expression:
$$3(4)^2 + 2(4) - 5$$

First, calculate each term:
- $3(16) = 48 $-$2(4) = 8$ Now combine these results:
$$48 + 8 - 5 = 51$$

The answer is 51.

:x??

---

#### Solving Linear Equations
Background context: Solving linear equations involves finding the value of a variable that makes the equation true. The process typically includes isolating the variable on one side of the equation.

:p Solve for $x $ in the equation$2x + 3 = 7$.
??x
To solve the equation $2x + 3 = 7$, follow these steps:

1. Subtract 3 from both sides:
$$2x + 3 - 3 = 7 - 3$$
$$2x = 4$$2. Divide both sides by 2 to isolate $ x$:
$$x = \frac{4}{2}$$
$$x = 2$$

The solution is $x = 2$.

:p
??x
The value of $x $ that satisfies the equation$2x + 3 = 7$ is 2.

:x??

---

#### Word Problems Involving Algebraic Expressions
Background context: Word problems often require setting up and solving equations based on real-life situations. These problems test your ability to translate verbal descriptions into mathematical expressions.

:p Jesse has taken four tests in his English class and scored 89, 83, 78, and 92 percent. His final exam is worth two test grades. What score does he need on the final exam to get a 90 percent average for the semester?
??x
To solve this problem, we can set up an equation where $y$ represents the score Jesse needs on his final exam.

The formula for the average score over five tests is:
$$\frac{89 + 83 + 78 + 92 + y}{5} = 90$$

First, sum the scores of the four tests:
$$89 + 83 + 78 + 92 = 342$$

Now, set up the equation and solve for $y$:
$$\frac{342 + y}{5} = 90$$

Multiply both sides by 5 to clear the fraction:
$$342 + y = 450$$

Subtract 342 from both sides to isolate $y$:
$$y = 450 - 342$$
$$y = 108$$

Jesse needs a score of 108 on his final exam to achieve an average of 90%.

:x??

---

#### Age Problems
Background context: Age problems often involve setting up equations based on relationships between the ages of individuals. These problems test your ability to translate verbal descriptions into mathematical expressions and solve for unknowns.

:p Angie is 6 years older than Heather. Six years ago, Angie was twice as old as Heather was. How old is Heather now?
??x
Let $h $ represent Heather's current age. Then Angie's current age is$h + 6$.

Six years ago:
- Heather’s age: $h - 6 $- Angie’s age:$(h + 6) - 6 = h$ According to the problem, six years ago Angie was twice as old as Heather:
$$h = 2(h - 6)$$

Solve for $h$:

$$h = 2h - 12$$

Subtract $2h$ from both sides:
$$-h = -12$$

Multiply by -1 to isolate $h$:
$$h = 12$$

Heather is currently 12 years old.

:x??

---

#### System of Equations
Background context: Solving systems of equations involves finding the values that satisfy all given equations simultaneously. These problems often involve setting up two or more related equations and solving them together.

:p Charles has $60 and is saving $7 per week. David has $120 and is saving$5 per week. How long will it be before David and Charles have the same amount of money?
??x
Let $t$ represent the number of weeks from now until they have the same amount of money.

Charles's total savings after $t$ weeks:
$$60 + 7t$$

David's total savings after $t$ weeks:
$$120 + 5t$$

Set these two expressions equal to each other because their savings will be the same:
$$60 + 7t = 120 + 5t$$

Subtract $5t$ from both sides:
$$60 + 2t = 120$$

Subtract 60 from both sides:
$$2t = 60$$

Divide by 2:
$$t = 30$$

It will take 30 weeks for David and Charles to have the same amount of money.

:x??

---

#### Money and Saving Problems
Background context: These problems involve setting up equations based on financial situations. They test your ability to translate verbal descriptions into mathematical expressions and solve them.

:p Bianca's cellular carrier charges a monthly rate of $12.95 and $0.25 per minute for international calls. If the bill is $21.20, how many minutes did Bianca spend on international calls?
??x
Let $m$ represent the number of minutes spent on international calls.

The total cost can be expressed as:
$$12.95 + 0.25m = 21.20$$

Subtract $12.95 from both sides to isolate the term with $ m$:
$$0.25m = 8.25$$

Divide by 0.25:
$$m = \frac{8.25}{0.25}$$
$$m = 33$$

Bianca spent 33 minutes on international calls.

:x??

---

#### Ratio and Proportion Problems
Background context: These problems involve setting up equations based on proportional relationships between quantities. They test your ability to translate verbal descriptions into mathematical expressions and solve for unknowns.

:p Chief Hall and Sergeant First Class Aziz both donated money to the Army Emergency Relief Fund. Chief Hall gave three times as much as Sergeant First Class Aziz did. Between the two of them, they donated$280. How much money did Chief Hall give?
??x
Let $a$ represent the amount donated by Sergeant First Class Aziz.

Then, Chief Hall donated:
$$3a$$

The total donation is:
$$a + 3a = 4a$$

We know the total donation is$280, so we can set up the equation:
$$4a = 280$$

Divide by 4:
$$a = 70$$

Chief Hall donated three times this amount:
$$3 \times 70 = 210$$

Chief Hall gave$210.

:x??

---

#### Distance, Rate, and Time Problems
Background context: These problems often involve setting up equations based on the relationship between distance, rate (speed), and time. They test your ability to translate verbal descriptions into mathematical expressions and solve for unknowns.

:p A car travels 300 miles at a constant speed of 60 miles per hour. How long does the trip take?
??x
To find the time taken, use the formula:
$$\text{Time} = \frac{\text{Distance}}{\text{Speed}}$$

Given that the distance is 300 miles and the speed is 60 mph:
$$\text{Time} = \frac{300}{60}$$
$$\text{Time} = 5$$

The trip takes 5 hours.

:x??

---

#### Combination of Multiple Concepts
Background context: Some problems combine multiple algebraic concepts, such as solving equations and working with variables in expressions. These problems test your ability to apply various algebraic skills together.

:p Solve for $x $ in the equation$2(x - 3) + 4 = 10$.
??x
First, simplify the left side of the equation:
$$2(x - 3) + 4 = 10$$

Distribute the 2:
$$2x - 6 + 4 = 10$$

Combine like terms:
$$2x - 2 = 10$$

Add 2 to both sides to isolate the term with $x$:
$$2x = 12$$

Divide by 2 to solve for $x$:
$$x = \frac{12}{2}$$
$$x = 6$$

The solution is $x = 6$.

:x??

--- 
--- 

Each flashcard covers a different aspect of the algebra problems described in your text. These cards should help you practice and understand various algebraic concepts. ---

#### Solve for x: xy $\frac{xy}{5} + 31 = 2 $ Background context explaining the concept. The problem involves solving a linear equation with one variable,$x $. We need to isolate$ x$ by performing algebraic operations.

:p What is the value of $x$ that satisfies this equation?
??x
To solve for $x$, we start by simplifying and rearranging the given equation:

$$\frac{xy}{5} + 31 = 2$$

First, subtract 31 from both sides to isolate the term with $x$:

$$\frac{xy}{5} = 2 - 31$$
$$\frac{xy}{5} = -29$$

Next, multiply both sides by 5 to eliminate the fraction:
$$xy = -29 \times 5$$
$$xy = -145$$

Since the equation is in terms of $x $ and we are given no specific value for$y$, we can conclude that there might be a misunderstanding. Given the options, it seems the problem intended to be simpler or had an error.

Assuming the simplified form should have been:

$$x = -145 \div y$$

However, based on the provided choices and typical algebraic simplifications, let's assume $y$ is 1 (as a common simplification step):
$$x = -145 \div 1$$
$$x = -145$$

Given the choices, the closest reasonable answer from the options would be:

??x $x = -145$ x??

---

#### Simplify:$78 \cdot 827^{\frac{5}{y}} \div 5y$ Background context explaining the concept. This problem involves simplifying an expression with exponents and division.

:p What is the simplified form of this expression?
??x
The given expression is:
$$78 \cdot 827^{\frac{5}{y}} \div 5y$$

To simplify, we first handle the exponentiation and then perform the division:

1. Evaluate $827^{\frac{5}{y}}$. This term cannot be simplified further without a specific value for $ y$.
2. Divide by $5y$:

$$\frac{78 \cdot 827^{\frac{5}{y}}}{5y}$$

This is the most simplified form given the current information.

??x$$\frac{78 \cdot 827^{\frac{5}{y}}}{5y}$$x??

---

#### One number is 10 more than another. The sum of twice the smaller number plus three times the larger number is 55.
Background context explaining the concept. This problem involves setting up and solving a system of linear equations.

:p What are the two numbers?
??x
Let's denote:
- $x$ as the smaller number,
- $y$ as the larger number.

From the problem, we have two conditions:

1. One number is 10 more than another:
$$y = x + 10$$2. The sum of twice the smaller number plus three times the larger number is 55:
$$2x + 3y = 55$$

Substitute $y = x + 10$ into the second equation:
$$2x + 3(x + 10) = 55$$
$$2x + 3x + 30 = 55$$
$$5x + 30 = 55$$

Subtract 30 from both sides:
$$5x = 25$$

Divide by 5:
$$x = 5$$

Now, substitute $x = 5 $ back into the equation for$y$:

$$y = 5 + 10$$
$$y = 15$$

So, the smaller number is 5 and the larger number is 15.

??x
The two numbers are:
- Smaller number: 5
- Larger number: 15
x??

---

#### Price of a pair of jeans increased by 15% from$20.
Background context explaining the concept. This problem involves calculating a percentage increase and finding the new price.

:p What is the new price?
??x
The original price of the jeans was $20, and it has increased by 15%.

To find the amount of the increase:

$$\text{Increase} = 20 \times \frac{15}{100}$$
$$\text{Increase} = 20 \times 0.15$$
$$\text{Increase} = 3$$

Now, add this increase to the original price:
$$\text{New Price} = 20 + 3$$
$$\text{New Price} = 23$$??x
The new price is$23.
x??

---

#### Wrapping Burritos: A chef wraps 3 burritos in 2 minutes.
Background context explaining the concept. This problem involves setting up a rate equation to find the relationship between time and number of items.

:p What equation expresses this rate?
??x
Let's denote:
- $x$ as the number of minutes,
- $y$ as the number of burritos.

The chef wraps 3 burritos in 2 minutes, so we can set up a proportion:

$$\frac{y}{x} = \frac{3}{2}$$

To express this rate equation directly with $x $ and$y$:

$$y = \frac{3}{2}x$$??x
The equation that expresses the rate is:
$$y = \frac{3}{2}x$$x??

---

#### Abby's Age Problem: Abby is one-third her mom’s age, in 12 years she will be half.
Background context explaining the concept. This problem involves setting up and solving a system of linear equations to find ages.

:p How old is Abby now?
??x
Let $A $ be Abby's current age and$M$ be her mom's current age.

From the first condition:
$$A = \frac{1}{3}M$$

From the second condition in 12 years:
$$

A + 12 = \frac{1}{2}(M + 12)$$

Substitute $A = \frac{1}{3}M$ into the second equation:
$$\frac{1}{3}M + 12 = \frac{1}{2}(M + 12)$$

Multiply both sides by 6 to clear the fractions:
$$2M + 72 = 3(M + 12)$$
$$2M + 72 = 3M + 36$$

Subtract $2M$ from both sides:
$$72 = M + 36$$

Subtract 36 from both sides:
$$

M = 36$$

Now, substitute $M = 36$ back into the first equation:
$$A = \frac{1}{3} \times 36$$
$$

A = 12$$

So, Abby is currently 12 years old.

??x
Abby is 12 years old.
x??

---

#### Coins Problem: You have 25 coins in nickels and dimes that total$1.65.
Background context explaining the concept. This problem involves setting up a system of linear equations with two variables to find the number of each type of coin.

:p How many nickels do you have?
??x
Let $n $ be the number of nickels and$d$ be the number of dimes.

From the total number of coins:
$$n + d = 25$$

From the total value of the coins:
$$0.05n + 0.10d = 1.65$$

We can multiply the second equation by 100 to clear decimals:
$$5n + 10d = 165$$
$$n + d = 25$$

Now, we have a system of equations:
1.$n + d = 25 $2.$5n + 10d = 165$ We can simplify the second equation by dividing by 5:
$$n + 2d = 33$$

Subtract the first equation from this new equation:
$$(n + 2d) - (n + d) = 33 - 25$$
$$d = 8$$

Now, substitute $d = 8$ back into the first equation:
$$n + 8 = 25$$
$$n = 17$$

So, you have 17 nickels.

??x
You have 17 nickels.
x??

---

#### Vending Machine Coins:$17 in$1 and quarter coins with a total of 26 coins.
Background context explaining the concept. This problem involves setting up and solving a system of linear equations to find the number of each type of coin.

:p How many $1 coins does the vending machine contain?
??x
Let $d $ be the number of quarters and$o $ be the number of$1 coins.

From the total number of coins:
$$d + o = 26$$

From the total value of the coins:
$$0.25d + 1o = 17$$

Multiply by 100 to clear decimals:
$$25d + 100o = 1700$$

We can simplify the second equation by dividing by 25:
$$d + 4o = 68$$

Now, we have a system of equations:
1.$d + o = 26 $2.$ d + 4o = 68$ Subtract the first equation from this new equation:
$$(d + 4o) - (d + o) = 68 - 26$$
$$3o = 42$$
$$o = 14$$

So, there are 14 one-dollar coins.

??x
The vending machine contains 14$1 coins.
x??

---

#### Joe Snuffy at the Post Exchange: $35 was$7 less than three times what he spent at the commissary.
Background context explaining the concept. This problem involves setting up and solving a linear equation to find the amount spent.

:p How much did he spend at the commissary?
??x
Let $c$ be the amount Joe spent at the commissary.

From the given information:

$$3c - 7 = 35$$

Add 7 to both sides:
$$3c = 42$$

Divide by 3:
$$c = 14$$

So, Joe spent$14 at the commissary.

??x
Joe spent $14 at the commissary.
x??

---

#### Sum of Two Numbers: The sum of two numbers is 72 and one number is five times the other.
Background context explaining the concept. This problem involves setting up a system of linear equations to find the values of both numbers.

:p What are the two numbers?
??x
Let $x $ be the smaller number, then the larger number will be$5x$.

From the given information:

$$x + 5x = 72$$
$$6x = 72$$

Divide by 6:
$$x = 12$$

So, the smaller number is 12 and the larger number is:
$$5 \times 12 = 60$$

Thus, the two numbers are 12 and 60.

??x
The two numbers are:
- Smaller number: 12
- Larger number: 60
x??

---

---
#### Substitution and Solving Equations
Background context: The problem involves substituting a value into an expression and solving simple algebraic equations. This is foundational for understanding more complex algebraic manipulations.

:p What is the substitution method used to solve the equation $3x + 2 $ at$x = -2$?
??x
To substitute $-2 $ for$x$, we perform the following calculation: 
$$3(-2) + 2$$

First, calculate $3 \times (-2)$:
$$3 \times (-2) = -6$$

Then add 2:
$$-6 + 2 = -4$$

The answer is $-4$.
??x
The process involves direct substitution and basic arithmetic. No additional code examples are needed for this simple arithmetic.
x??

---
#### Solving Inequalities with Variables
Background context: The problem demonstrates solving inequalities by isolating the variable, similar to solving equations but paying attention to inequality signs.

:p Solve the inequality $4x - 27 > 8$.
??x
To solve $4x - 27 > 8 $, isolate $ x$ by following these steps:
1. Add 27 to both sides of the inequality:
$$4x - 27 + 27 > 8 + 27$$
$$4x > 35$$2. Divide both sides by 4:
$$x > \frac{35}{4}$$
$$x > 8.75$$

The answer is $x > 8.75$.
??x
The process involves basic algebraic manipulation, ensuring that the inequality sign remains correct when performing operations.

---
#### Number Line Representation of Inequalities
Background context: This problem requires understanding how to represent a solution on a number line. The key here is recognizing which interval satisfies the inequality $a^2$.

:p Determine which number line corresponds with the solution set for $a^2 \leq 16$.
??x
The inequality $a^2 \leq 16$ can be solved by taking the square root of both sides:
$$|a| \leq 4$$

This means that $a$ is between -4 and 4, inclusive.

The corresponding number line would show a closed interval from -4 to 4. Therefore, the correct choice is (D).

The answer is Choice D.
??x
The process involves solving for the absolute value of the variable and interpreting it as an interval on the number line.

---
#### Combining Like Terms and Solving Equations
Background context: This problem requires combining like terms and then isolating a variable. It demonstrates basic algebraic manipulation skills, which are essential in more complex equations.

:p Solve $32y + 5 = 145$.
??x
To solve the equation $32y + 5 = 145$, follow these steps:
1. Subtract 5 from both sides to isolate the term with $y$:
$$32y + 5 - 5 = 145 - 5$$
$$32y = 140$$2. Divide both sides by 32 to solve for $ y$:
$$y = \frac{140}{32}$$

Simplify the fraction:
$$y = \frac{70}{16} = \frac{35}{8}$$

The answer is $y = \frac{35}{8}$.
??x
This problem involves basic algebraic operations: subtraction and division, leading to a simplified fractional solution.

---
#### Weighted Average Problem
Background context: This problem deals with weighted averages and requires setting up an equation based on given weights and scores. It tests understanding of how different elements contribute to an overall average.

:p Jesse needs to score 90 in the class. Given test grades, calculate the required test grade for the final exam.
??x
Let $x$ be the score needed on the final exam (which counts as two test grades). The equation is:
$$89 + 83 + 78 + 92 + \frac{2x}{6} = 90$$

First, sum the known scores:
$$89 + 83 + 78 + 92 = 342$$

The total score needed for a 90 average over six tests is:
$$90 \times 6 = 540$$

Set up the equation and solve for $x$:
$$342 + \frac{2x}{6} = 540$$

Subtract 342 from both sides:
$$\frac{2x}{6} = 198$$

Multiply by 6:
$$2x = 1,188$$

Divide by 2:
$$x = 594 / 2 = 297 / 3 = 99$$

The answer is $x = 99$.
??x
This involves setting up an equation for the weighted average and solving it step-by-step, ensuring each operation is clearly explained.

---
#### Counting Principle in Problem Solving
Background context: This problem uses the counting principle to find the total number of students with specific attributes (sibling AND pet) in a class. It tests understanding of set theory basics.

:p Ms. Burton’s students are categorized by having both a sibling and a pet, or only one attribute.
??x
Let $x$ represent the number of students who have both a sibling and a pet. The total number of students is:
$$14 - 22 + x = 32$$

Rearrange to solve for $x$:
$$x = 32 - (14 - 22)$$
$$x = 32 - (-8) = 32 + 8 = 40$$

The answer is $x = 40$.
??x
This involves setting up and solving an equation based on the given conditions, ensuring all students are accounted for.

---
#### Linear Equations with Variables Representing Real Quantities
Background context: This problem uses variables to represent real-world quantities (in this case, time in weeks) and sets up linear equations to solve for those variables. It helps build understanding of how algebra applies to practical scenarios.

:p Charles and David save money over a number of weeks.
??x
Let $x$ represent the number of weeks they will save their money. The equation representing Charles's savings is:
$$7x + 60$$

The equation for David’s savings is:
$$5x + 120$$

Setting these equal to find when their savings are the same:
$$7x + 60 = 5x + 120$$

Solve for $x$:
Subtract $5x$ from both sides:
$$2x + 60 = 120$$

Subtract 60 from both sides:
$$2x = 60$$

Divide by 2:
$$x = 30$$

The answer is $x = 30$.
??x
This involves setting up and solving a linear equation, ensuring each step in the process is clear.

---
#### Age Problems with Variables
Background context: This problem uses variables to represent ages of individuals and sets up equations based on relationships between their ages. It tests understanding of basic algebraic manipulation.

:p Heather’s age $x$ and Angie’s age now are represented by a linear equation.
??x
Let $x $ represent Heather's current age, then Angie's age is$x + 6$.

The sum of the ages is given as:
$$x + (x + 6) = 12 + 6$$

Solve for $x$:
Combine like terms:
$$2x + 6 = 18$$

Subtract 6 from both sides:
$$2x = 12$$

Divide by 2:
$$x = 6$$

Heather is currently 6 years old, and Angie is $6 + 6 = 12$ years old.

The answer is $x = 6$.
??x
This involves setting up and solving a simple linear equation based on age relationships.

#### Representing Variables and Equations for Jayden's Van Rental
Background context: This concept involves setting up an equation to represent a real-world scenario where Jayden rents a van. The problem states that there are rental fees, which include two days of rental at $30 each.

:p Let $m $ represent the number of miles Jayden drove. Create an equation and solve for$m$.
??x
To solve this, we need to form an equation based on the given information:

- Two days of van rental cost: $2 \times 30 = 60 $- Additional mileage charge (assuming it's a part of the equation):$5m + 360 $ The total cost is$600, so we can set up the following equation:
$$60 + 5m + 360 = 600$$

Solving for $m$:
$$60 + 5m + 360 = 600$$

Subtract 420 from both sides:
$$5m = 180$$

Divide by 5:
$$m = 36$$

The equation can be broken down as follows:

- Initial rental fee for two days:$60 $- Additional cost based on miles driven:$5m + 360 $- Total cost:$60 + 5m + 360 = 600$ Thus, the number of miles Jayden drove is:
$$m = 36$$x??

---

#### Equation for Chief Hall's Donation
Background context: This problem involves setting up an equation based on donations from Sergeant First Class Aziz and Chief Hall. The total donation amount is given as$280.

:p Let $x $ represent how much money Sergeant First Class Aziz donated. Create the equation and solve for$x$.
??x
Sergeant First Class Aziz's donation: $x $ Chief Hall's donation (three times that of Aziz):$3x$

Total donations:
$$x + 3x = 280$$

Combine like terms:
$$4x = 280$$

Divide by 4:
$$x = 70$$

So, Sergeant First Class Aziz donated $70 and Chief Hall's donation is $3x$:
$$3 \times 70 = 210$$x??

---

#### Cost of Lessons and Equipment Rental
Background context: This problem involves calculating the cost for three people taking lessons and renting equipment, with a total cost provided.

:p Let $x$ represent the total cost per person for lessons. Create the equation and solve.
??x
Let $3x $ be the total cost of lessons for three people. The rental cost for each person is$15:
$$45 = 3x + 45$$

Simplify:
$$3x + 45 - 45 = 120 - 45$$
$$3x = 75$$

Divide by 3:
$$x = 25$$

The cost per person for lessons is$25, and the total cost for three people is:
$$3x = 75 + 45 = 120$$x??

---

#### Finding How Much Ava Spent on Skate Shoes
Background context: This problem involves setting up an equation to determine how much Ava spent on skate shoes. The total amount spent and the remaining money are provided.

:p Let $x$ represent the cost of the skate shoes. Create the equation and solve.
??x
Let $21 - 4 = 17$ be the remaining amount after buying two other items, which is:
$$2x + (17) = 28$$

Simplify and solve for $x$:
$$2x + 17 - 17 = 28 - 17$$
$$2x = 11$$

Divide by 2:
$$x = 5.50$$

Thus, the cost of the skate shoes is$14.
x??

---

#### Number of Cats in the Daycare Center
Background context: This problem involves setting up an equation based on a relationship between the number of cats and dogs in a daycare center.

:p Let $c $ represent the number of cats. Create the equation and solve for$c$.
??x
Given that there are 31 dogs, which is three more than four times the number of cats:
$$4c + 3 = 31$$

Solve for $c$:
$$4c + 3 - 3 = 31 - 3 \\
4c = 28 \\
c = 7$$

Thus, there are 7 cats in the daycare center.
x??

---

#### Sum of Consecutive Numbers
Background context: This problem involves finding two consecutive numbers whose sum is given.

:p Let $x $ represent the first number. Create an equation and solve for$x$.
??x
Let $x $ be the first number, and the next number is$x + 1$. The sum of these two numbers is 37:
$$x + (x + 1) = 37$$

Solve for $x$:
$$2x + 1 = 37 \\
2x + 1 - 1 = 37 - 1 \\
2x = 36 \\
x = 18$$

Thus, the two numbers are 18 and 19.
x??

---

#### Amount of Money for Simon, Alberto, and Paloma
Background context: This problem involves setting up an equation to determine how much money each person has based on their contributions.

:p Let $x $ represent the amount of money Simon gets. Create the equations and solve for$x$.
??x
Let:
- $x$ be the amount of money Simon gets,
- $2x$ be Alberto's money, and
- $2x - 5$ be Paloma's money.

The total sum is $80:
$$x + 2x + (2x - 5) = 80$$

Solve for $x$:
$$x + 2x + 2x - 5 = 80 \\
5x - 5 = 80 \\
5x - 5 + 5 = 80 + 5 \\
5x = 85 \\
x = 17$$

Thus, Simon has$17. Paloma's money:
$$2(17) - 5 = 34 - 5 = 29$$x??

---

#### Substitution in a System of Equations
Background context: This problem involves using substitution to solve for variables in a system of equations.

:p Let $y = x + 5$. Substitute and simplify the equation:
$$xy = 12, y = x + 5$$??x
Given the equations:
$$y = x + 5$$
$$xy = 12$$

Substitute $y$ in the second equation:
$$x(x + 5) = 12$$

Simplify and solve for $x$:
$$x^2 + 5x - 12 = 0$$

Factorize or use the quadratic formula to find $x$. Let's factorize:

Possible factors of 12 that add up to 5 are 3 and 4:
$$(x - 3)(x + 4) = 0$$

Thus,$x = 3 $ or$x = -4$.

If $x = 3$:
$$y = x + 5 = 3 + 5 = 8$$

If $x = -4$:
$$y = x + 5 = -4 + 5 = 1$$

The solutions are:
$$(x, y) = (3, 8) \text{ or } (-4, 1)$$x??

---

#### Simplifying Exponential Expressions
Background context: This problem involves simplifying expressions with exponents.

:p Simplify the expression $3y^2 \cdot 5y^3$.
??x
To simplify the expression:
$$3y^2 \cdot 5y^3$$

Combine like terms by multiplying coefficients and adding exponents of $y$:
$$(3 \cdot 5) \cdot (y^2 \cdot y^3) = 15y^{2+3} = 15y^5$$x??

---

#### Finding the Larger Number Based on a Difference
Background context: This problem involves setting up an equation to find two numbers where one is 10 more than the other.

:p Let $x $ represent the smaller number. Create the equation and solve for$x$.
??x
Let $x $ be the smaller number, and the larger number be$x + 10$.

Create an equation based on the given information:
$$2(x + 10) - 3 = 30$$

Solve for $x$:
$$2x + 20 - 3 = 30 \\
2x + 17 = 30 \\
2x + 17 - 17 = 30 - 17 \\
2x = 13 \\
x = 6.5$$

Thus, the smaller number is $x$ and the larger number:
$$x + 10 = 6.5 + 10 = 16.5$$x??

---

#### Finding 15% of a Price
Background context: To find 15% of a price, you can convert 15% into a decimal and multiply it by the original amount. Alternatively, you can add 15% to the original amount using multiplication by 1.15.
:p How do you calculate 15% of$20?
??x
To find 15% of $20, you can convert 15% into a decimal (0.15) and multiply it by$20:
$$0.15 \times 20 = 3$$

Alternatively, you can use the formula $x \times 1.15$:
$$20 \times 1.15 = 23$$

Both methods give you the correct answer of$23.
x??

---

#### Equation for Burritos per Minute
Background context: This problem requires setting up an equation to determine how many burritos a chef can wrap in a given number of minutes, based on their wrapping rate.
:p Write an equation relating the number of burritos wrapped in 3 minutes.
??x
Let $x $ represent the number of minutes and$y$ represent the number of burritos. Given that the chef wraps 2 burritos per minute, we can set up the equation as:
$$y = \frac{2}{3} x$$

Simplifying further:
$$y = \frac{2x}{3}$$
This equation relates the number of burritos ($y $) to the number of minutes ($ x$).
x??

---

#### Abby’s Future Age
Background context: The problem involves setting up and solving an equation based on the relationship between Abby's age and her mom's age, and predicting their ages 12 years in the future.
:p Calculate Abby’s future age given that she will be half her mom’s age in 12 years.
??x
Let $a$ represent Abby’s current age. Her mom is three times as old as Abby:
$$\text{Mom's age} = 3a$$

In 12 years, Abby will be $a + 12 $ and her mom will be$3a + 12$. According to the problem, in 12 years, Abby will be half her mom’s age:
$$a + 12 = \frac{1}{2} (3a + 12)$$

Solving this equation step-by-step:
$$a + 12 = \frac{3a + 12}{2}$$

Multiply both sides by 2 to clear the fraction:
$$2(a + 12) = 3a + 12$$
$$2a + 24 = 3a + 12$$

Subtract $2a$ from both sides:
$$24 = a + 12$$

Subtract 12 from both sides:
$$a = 12$$

Abby is currently 12 years old, and her mom is 36. In 12 years, Abby will be 24 and her mom will be 48.
x??

---

#### Coins in a Jar
Background context: This problem involves setting up equations to determine the number of nickels and dimes in a jar given their total value and quantity.
:p Determine how many nickels you have if you have 25 coins in total, including both nickels (worth $0.05) and dimes (worth$0.10).
??x
Let $n $ represent the number of nickels and$d$ represent the number of dimes. You know that:
$$n + d = 25$$

The total value in cents is:
$$5n + 10d = 2500$$(since$2.50 = 2500 cents)
From $n + d = 25 $, solve for $ d$:
$$d = 25 - n$$

Substitute $d$ into the value equation:
$$5n + 10(25 - n) = 2500$$

Simplify and solve for $n$:
$$5n + 250 - 10n = 2500$$
$$-5n + 250 = 2500$$

Subtract 250 from both sides:
$$-5n = 2250$$

Divide by -5:
$$n = -450 / -5 = 17$$

Thus, you have 17 nickels. You can verify this by finding $d$:
$$d = 25 - 17 = 8$$

So, there are 17 nickels and 8 dimes.
x??

---

#### Machine Coins
Background context: This problem requires determining the number of quarters in a machine that contains a total of 26 coins with a combined value of$17.00, using algebraic equations to solve for the unknowns.
:p Determine how many quarters are in the machine.
??x
Let $q $ represent the number of quarters and$d $ represent the number of dollar coins (worth$1 each). You know that:
$$q + d = 26$$

The total value equation is:
$$25q + 100d = 1700$$(since$17.00 = 1700 cents)
From $q + d = 26 $, solve for $ d$:
$$d = 26 - q$$

Substitute $d$ into the value equation:
$$25q + 100(26 - q) = 1700$$

Simplify and solve for $q$:
$$25q + 2600 - 100q = 1700$$
$$-75q + 2600 = 1700$$

Subtract 2600 from both sides:
$$-75q = -900$$

Divide by -75:
$$q = 12$$

Thus, there are 12 quarters. The remaining coins are dollar coins:
$$d = 26 - 12 = 14$$

So, the machine contains 12 quarters and 14 dollar coins.
x??

---

#### Snuffy’s Spending
Background context: This problem involves setting up an equation to determine how much a soldier spent at the post exchange based on his expenditure at the commissary.
:p Determine how much more Snuffy spent at the post exchange than he did at the commissary.
??x
Let $x $ represent the amount of money Spuffy spent at the commissary. The problem states that he spent three times as much at the post exchange, less$7:
$$3x - 7 = x + 14$$

Solve for $x$:
$$3x - 7 = x + 14$$

Subtract $x$ from both sides:
$$2x - 7 = 14$$

Add 7 to both sides:
$$2x = 21$$

Divide by 2:
$$x = 10.5$$

So, Snuffy spent$10.50 at the commissary. At the post exchange, he spent:
$$3(10.5) - 7 = 31.5 - 7 = 24.5$$

Thus, Snuffy spent$24.50 more at the post exchange than at the commissary.
x??

---

