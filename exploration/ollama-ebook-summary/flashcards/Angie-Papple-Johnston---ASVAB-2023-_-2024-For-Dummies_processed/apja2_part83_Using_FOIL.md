# Flashcards: Angie-Papple-Johnston---ASVAB-2023-_-2024-For-Dummies_processed (Part 83)

**Starting Chapter:** Using FOIL

---

#### Solving One-Step Equations Involving Addition and Subtraction
Background context: This concept involves solving equations that require only one operation to isolate a variable. The inverse operation is used to move numbers from one side of the equation to the other.

Example problem: $x + 47,432 = 50,000 $:p How do you solve for$ x$ in this equation?
??x
To solve for $x $, subtract 47,432 from both sides of the equation. This isolates $ x$ and gives us the solution.

```java
// Pseudocode to illustrate solving an equation
int x = 50000 - 47432; // Subtract 47432 from 50000
```
x??

---

#### Solving Problems with Multiplication and Division
Background context: In multiplication and division equations, the sign of the result depends on whether both numbers are positive or negative. The inverse operation of multiplication is division.

Example problem: $6x = -36 $:p How do you solve for$ x$ in this equation?
??x
To solve for $x $, divide both sides by 6, the coefficient of $ x $. This isolates$ x$ and gives us the solution.

```java
// Pseudocode to illustrate solving an equation
int x = -36 / 6; // Divide -36 by 6
```
x??

---

#### Solving Multistep Equations with Variables on Both Sides
Background context: Multistep equations can have variables on both sides of the equal sign. The goal is to isolate the variable by moving terms around and using inverse operations.

Example problem: $3x + 9 = x - 3 $:p How do you solve for$ x$ in this equation?
??x
First, subtract $x $ from both sides to get all$x$-terms on one side:
$$2x + 9 = -3$$

Then, subtract 9 from both sides:
$$2x = -12$$

Finally, divide by 2 to isolate $x$:
$$x = -6$$```java
// Pseudocode to illustrate solving an equation
int x = (-3 - 9) / 2; // First step: move terms and then solve for x
```
x??

---

#### Simplifying Algebraic Expressions
Background context: Simplifying expressions involves removing parentheses, using exponent rules, combining like terms, and combining constants. This makes the expression easier to evaluate or solve.

Example problem:$(3x + 4)(2x - 1)$:p How do you simplify this algebraic expression?
??x
First, use the distributive property:
$$6x^2 - 3x + 8x - 4$$

Combine like terms:
$$6x^2 + 5x - 4$$```java
// Pseudocode to illustrate simplifying an algebraic expression
String simplifiedExpression = "6*x^2 + 5*x - 4"; // Simplified form of the expression
```
x??

---
---

#### FOIL Method for Binomials
FOIL is a mnemonic used to multiply two binomials. It stands for First, Outer, Inner, Last, which are the pairs of terms that need to be multiplied and then combined. The formula for multiplying two binomials $(a + b) \times (c + d)$ using FOIL would look like this:
- **First**: Multiply the first terms in each binomial:$a \times c $- **Outer**: Multiply the outermost terms:$ a \times d $- **Inner**: Multiply the innermost terms:$ b \times c $- **Last**: Multiply the last terms in each binomial:$ b \times d$:p What is FOIL and how does it work?
??x
FOIL (First, Outer, Inner, Last) is a method for distributing two binomials. It involves multiplying the first terms of both binomials, then the outermost terms, followed by the innermost terms, and finally the last terms in each binomial.
For example:
$$(x + 2)(x + 5)$$- **First**:$ x \times x = x^2 $- **Outer**:$ x \times 5 = 5x $- **Inner**:$2 \times x = 2x $- **Last**:$2 \times 5 = 10$ Combining like terms, we get:
$$x^2 + 5x + 2x + 10 = x^2 + 7x + 10$$

The final expression is simplified to:
$$x^2 + 7x + 10$$??x

---

#### Tackling Two-Variable Equations
When dealing with two-variable equations, the goal is often to find a solution that satisfies both equations. If you encounter multiple solutions or complex systems, substitution and combining equations are common methods.

Substitution involves solving one equation for one variable and plugging that value into the other equation. Combining equations means adding or subtracting the equations to eliminate one of the variables.

:p What is the difference between using substitution and combining equations in two-variable systems?
??x
In a two-variable system, **substitution** works when you can easily solve one equation for one variable and substitute that expression into the other equation. For example:
$$xy = 3$$
$$x + y = 7$$

First, solve $x + y = 7 $ for$y$:
$$y = 7 - x$$

Then substitute this expression into $xy = 3$:
$$x(7 - x) = 3$$
$$7x - x^2 = 3$$
$$x^2 - 7x + 3 = 0$$

Solve the quadratic equation for $x$.

On the other hand, **combining equations** is used when you have a more complex system where substitution might not be straightforward. You add or subtract the equations to eliminate one of the variables. For example:
$$12x + 9y = 73$$
$$8x + 2y = 69$$

You can multiply the second equation by 4.5 to make the coefficients of $y$ match:
$$12x + 9y = 73$$
$$36x + 9y = 310.5$$

Subtract the first equation from the modified second equation to eliminate $y$:
$$(36x + 9y) - (12x + 9y) = 310.5 - 73$$
$$24x = 237.5$$
$$x = \frac{237.5}{24} \approx 9.896$$

Substitute this value back into one of the original equations to solve for $y$.
??x

---

#### Solving Linear Systems Using Substitution
Linear systems with exponents of 1 (first-degree polynomials) can be solved using substitution when they are relatively simple. The process involves solving one equation for one variable and then substituting that expression into the other equation.

:p How do you solve a linear system using substitution?
??x
To solve a linear system using substitution, follow these steps:
1. Solve one of the equations for one of the variables.
2. Substitute the expression from step 1 into the other equation.
3. Simplify and solve the resulting equation to find the value of the remaining variable.
4. Use this value to find the value of the first variable by substituting back.

Example:
$$xy = 3$$
$$x + y = 7$$

First, solve $x + y = 7 $ for$y$:
$$y = 7 - x$$

Substitute this expression into $xy = 3$:
$$x(7 - x) = 3$$
$$7x - x^2 = 3$$
$$x^2 - 7x + 3 = 0$$

Solve the quadratic equation for $x$. Use the quadratic formula:
$$x = \frac{-b \pm \sqrt{b^2 - 4ac}}{2a}$$

Where $a = 1 $, $ b = -7 $, and$ c = 3$:
$$x = \frac{7 \pm \sqrt{49 - 12}}{2}$$
$$x = \frac{7 \pm \sqrt{37}}{2}$$

So the solutions for $x$ are:
$$x_1 = \frac{7 + \sqrt{37}}{2}$$
$$x_2 = \frac{7 - \sqrt{37}}{2}$$

Substitute each value of $x $ back into$y = 7 - x$:
For $x_1$:
$$y_1 = 7 - \left( \frac{7 + \sqrt{37}}{2} \right)$$
$$y_1 = \frac{14 - (7 + \sqrt{37})}{2}$$
$$y_1 = \frac{7 - \sqrt{37}}{2}$$

For $x_2$:
$$y_2 = 7 - \left( \frac{7 - \sqrt{37}}{2} \right)$$
$$y_2 = \frac{14 - (7 - \sqrt{37})}{2}$$
$$y_2 = \frac{7 + \sqrt{37}}{2}$$

So the solutions are:
$$\left( \frac{7 + \sqrt{37}}{2}, \frac{7 - \sqrt{37}}{2} \right)$$and$$\left( \frac{7 - \sqrt{37}}{2}, \frac{7 + \sqrt{37}}{2} \right)$$

The solutions are:
$$(x_1, y_1) = \left( \frac{7 + \sqrt{37}}{2}, \frac{7 - \sqrt{37}}{2} \right)$$
$$(x_2, y_2) = \left( \frac{7 - \sqrt{37}}{2}, \frac{7 + \sqrt{37}}{2} \right)$$

The solutions are:
??x
The solutions to the system of equations $xy = 3 $ and$x + y = 7$ are:
$$\left( \frac{7 + \sqrt{37}}{2}, \frac{7 - \sqrt{37}}{2} \right)$$and$$\left( \frac{7 - \sqrt{37}}{2}, \frac{7 + \sqrt{37}}{2} \right)$$

These are the points where both equations intersect.
??x

---

#### Solving Linear Systems Using Combining Equations
When solving linear systems with more complex expressions, combining or eliminating one of the variables by adding or subtracting the equations is a common method. This technique can be useful when substitution would result in complicated algebraic manipulations.

:p How do you solve a system of linear equations using combining equations?
??x
To solve a system of linear equations using combining equations, follow these steps:

1. **Identify the coefficients**: Look at the coefficients of the variables in both equations.
2. **Eliminate one variable**: Multiply one or both equations by constants to make the coefficients of one variable the same (but with opposite signs) so that they can be added together and eliminated.

Example:
$$12x + 9y = 73$$
$$8x + 2y = 69$$

First, multiply the second equation by 4.5 to align coefficients of $y$:
$$12x + 9y = 73$$
$$36x + 9y = 310.5$$

Next, subtract the first equation from the modified second equation to eliminate $y$:
$$(36x + 9y) - (12x + 9y) = 310.5 - 73$$
$$24x = 237.5$$
$$x = \frac{237.5}{24} \approx 9.896$$

Substitute this value of $x $ back into one of the original equations to solve for$y$:
Using the second equation:
$$8x + 2y = 69$$
$$8(9.896) + 2y = 69$$
$$79.168 + 2y = 69$$
$$2y = 69 - 79.168$$
$$2y = -10.168$$
$$y = \frac{-10.168}{2} \approx -5.084$$

So, the solution to the system is:
$$x \approx 9.896$$
$$y \approx -5.084$$??x

---

#### Solving Complex Two-Variable Systems
For more complex two-variable systems, combining equations may be necessary when substitution results in complicated algebraic manipulations.

:p How do you handle a system of linear equations that is not easily solved by simple substitution?
??x
To solve a system of linear equations that is not easily solved by simple substitution, use the method of combining or eliminating variables. This involves adding or subtracting the equations to eliminate one variable and then solving for the remaining variable.

Example:
$$71 - 36 = 0$$
$$6x + 29y = 65$$

First, multiply the bottom equation by 10 to align coefficients of $x$:
$$20 - 90 = 0$$
$$60x + 290y = 650$$

Next, add the first equation to the modified second equation to eliminate $y$:
$$(20 - 90) + (60x + 290y) = 0 + 650$$
$$60x - 70 = 650$$
$$60x = 650 + 70$$
$$60x = 720$$
$$x = \frac{720}{60}$$
$$x = 12$$

Substitute $x = 12 $ back into one of the original equations to solve for$y$:
Using the second equation:
$$6(12) + 29y = 65$$
$$72 + 29y = 65$$
$$29y = 65 - 72$$
$$29y = -7$$
$$y = \frac{-7}{29}$$

So, the solution to the system is:
$$x = 12$$
$$y = \frac{-7}{29}$$??x

---

#### Explanation of Exponents in Algebra
Exponents are a shorthand way to show repeated multiplication. In algebra, an exponent indicates how many times a base is multiplied by itself. For example,$5^2 $ means$5 \times 5 $, and $ y^3 $means$ y \times y \times y$.
:p What do exponents represent in algebra?
??x
Exponents represent repeated multiplication of the same number or variable. The base is the number being multiplied, while the exponent indicates how many times it is to be used as a factor.
x??

---

#### Rule 1: Any Base Raised to the Power of One Equals Itself
A fundamental rule in algebra states that any base raised to the power of one equals itself. This means $x^1 = x$.
:p According to the rules, what does $x$ equal when raised to the first power?
??x
When a number or variable is raised to the first power, it simply equals the original number or variable:$x^1 = x$.
x??

---

#### Rule 2: Any Base Raised to the Zero Power Equals One (Except for Zero)
Another important rule states that any base except zero raised to the power of zero equals one. Thus, $x^0 = 1 $ where$x \neq 0$.
:p What does $x^0$ equal according to this rule?
??x
According to this rule, any non-zero number or variable raised to the power of zero equals one:$x^0 = 1$.
x??

---

#### Rule 3: Multiplying Terms with the Same Base by Adding Exponents
When multiplying terms with the same base, you add the exponents. For example, $x^2 \times x^3 = x^{2+3} = x^5$.
:p How do you multiply two terms with the same base in algebra?
??x
To multiply two terms with the same base, add their exponents: $x^a \times x^b = x^{a+b}$.
x??

---

#### Rule 4: Dividing Terms with the Same Base by Subtracting Exponents
When dividing terms with the same base, you subtract the exponents. For example, $\frac{x^5}{x^2} = x^{5-2} = x^3$.
:p How do you divide two terms with the same base in algebra?
??x
To divide two terms with the same base, subtract their exponents: $\frac{x^a}{x^b} = x^{a-b}$.
x??

---

#### Rule 5: Negative Exponents and Reciprocals
A negative exponent means the reciprocal of the base raised to a positive exponent. For example, $x^{-3} = \frac{1}{x^3}$.
:p What does a negative exponent indicate in algebra?
??x
A negative exponent indicates that the term is the reciprocal of the base with a positive exponent: $x^{-n} = \frac{1}{x^n}$.
x??

---

#### Rule 6: Distributing Exponents to Products
When an expression has an exponent, each factor within the product must be raised to that power. For example, $(xy)^3 = x^3 y^3$.
:p How do you handle exponents in products?
??x
To distribute an exponent over a product, raise each factor to the given exponent: $(xy)^n = x^n y^n$.
x??

---

#### Factoring Algebra Expressions
Factoring is the process of breaking down a complex expression into its simpler components (factors). This helps in simplifying expressions and solving equations.
:p What does factoring mean in algebra?
??x
Factoring means expressing a number or polynomial as a product of its factors, which are simpler terms that multiply together to form the original expression.
x??

---

#### Pulling Out the Greatest Common Factor (GCF)
To factor an expression by pulling out the greatest common factor (GCF), identify the highest number and variable that divide all parts of the expression evenly. For example, $4x^2y = 2x \times 2xy$.
:p How do you find the greatest common factor (GCF) of terms in algebra?
??x
To find the GCF, determine the highest number or variable that divides into each term without a remainder. In $4x^2y $, the GCF is $2x $ because 2 is the highest number dividing both 4 and$x$.
x??

---

#### Factoring Steps Example
Steps to factor out the GCF from terms like $4x^2y + 2xy$:
1. Identify the GCF: The GCF of 4, 2, $x^2 $, and $ x$ is 2x.
2. Divide each term by the GCF:$\frac{4x^2y}{2x} = 2xy $, $\frac{2xy}{2x} = y$.
3. Multiply the result by the GCF to verify: $2x(2xy + y) = 4x^2y + 2xy$.
:p How do you factor out the GCF from an expression?
??x
To factor out the GCF, follow these steps:
1. Identify the highest common number and variable that divide all terms.
2. Divide each term by this GCF.
3. Multiply the result of Step 2 by the GCF to ensure it matches the original expression.
Example:
```java
public class Factorization {
    public static void main(String[] args) {
        int a = 4, b = 2;
        String x = "x", y = "xy";
        System.out.println("Expression: 4" + x + "^2" + y + " + 2" + x + y);
        String gcf = "2" + x; // GCF
        int div1 = a / (gcf.contains(x) ? Integer.parseInt(gcf.substring(0, gcf.indexOf(x))) : 1);
        int div2 = b / (gcf.contains(y) ? Integer.parseInt(gcf.substring(gcf.indexOf(x)+1)) : 1);
        String factoredTerm1 = div1 + x + "^" + y;
        String factoredTerm2 = div2 + y;
        System.out.println("Factored Terms: " + gcf + "(" + factoredTerm1 + " + " + factoredTerm2 + ")");
    }
}
```
x??

