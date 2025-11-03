# Flashcards: 1B001---Proofs---Jay-Cumming_processed (Part 13)

**Starting Chapter:** Exercises

---

#### Existence of Division Algorithm Proof
Background context: The proof for the division algorithm, as mentioned, asserts that for any integers \(a\) and \(m > 0\), there exist unique integers \(q\) and \(r\) such that \(a = mq + r\) where \(0 \leq r < m\). This is a fundamental concept in number theory.

:p What does the proof aim to demonstrate about the existence of integers \(q\) and \(r\)?
??x
The proof aims to show that for any integer \(a\) and positive integer \(m\), there exist integers \(q\) and \(r\) such that \(a = mq + r\) with the constraint \(0 \leq r < m\). This is demonstrated by considering different cases for \(a\).

Include relevant code or pseudocode:
```java
public boolean checkDivisionAlgorithm(int a, int m) {
    if (m <= 0) {
        throw new IllegalArgumentException("m must be greater than 0");
    }
    // Implementation of the division algorithm proof logic goes here.
}
```
x??

---

#### Uniqueness of Division Algorithm Proof
Background context: The uniqueness part of the proof involves showing that there is only one pair \((q, r)\) for a given \(a\) and \(m\). This is crucial because it ensures the theorem's assertion about the integers \(q\) and \(r\) is unique.

:p How does the proof ensure the uniqueness of \(q\) and \(r\)?
??x
The proof ensures the uniqueness of \(q\) and \(r\) by assuming that there are two different representations for \(a\): \(a = mq + r\) and \(a = m q_0 + r_0\). By manipulating these equations, it shows that this assumption leads to a contradiction, thus proving that only one pair \((q, r)\) can satisfy the condition.

Include relevant code or pseudocode:
```java
public boolean checkUniqueness(int a, int m, int q1, int r1, int q2, int r2) {
    if (m <= 0 || a < 0) {
        throw new IllegalArgumentException("Invalid input for division algorithm");
    }
    // Check the uniqueness of q and r
    return q1 == q2 && r1 == r2;
}
```
x??

---

#### Proof by Minimal Counterexample
Background context: The proof employs a method called proof by minimal counterexample, where it assumes there is a smallest counterexample to disprove. This approach leverages the well-ordering principle of natural numbers.

:p How does the proof use the concept of a minimal counterexample?
??x
The proof uses the concept of a minimal counterexample to show that if any positive integer \(a\) can be expressed as \(a = mq + r\) with \(0 \leq r < m\), then no smaller positive integer could fail this condition. By contradiction, it demonstrates that assuming there is a smallest such \(a\) leads to a logical inconsistency.

Include relevant code or pseudocode:
```java
public boolean minimalCounterexampleProof(int a, int m) {
    if (m <= 0 || a < 0) {
        throw new IllegalArgumentException("Invalid input for division algorithm");
    }
    // Implement the logic of proof by minimal counterexample here.
}
```
x??

---

#### Case Analysis in Proof
Background context: The proof analyzes different cases to handle all possible values of \(a\) relative to \(m\). This ensures that every scenario is considered, from \(a < m\) to \(a > m\), and \(a = m\).

:p What are the main cases considered in the proof?
??x
The main cases considered in the proof are:
1. \(a < m\)
2. \(a = m\)
3. \(a > m\)

Each case is handled separately to ensure that for any integer \(a\) and positive integer \(m\), there exists a pair \((q, r)\) such that \(a = mq + r\) with \(0 \leq r < m\).

Include relevant code or pseudocode:
```java
public void handleCases(int a, int m) {
    if (m <= 0 || a < 0) {
        throw new IllegalArgumentException("Invalid input for division algorithm");
    }
    // Handle the different cases based on a and m.
}
```
x??

---

#### Example of Division Algorithm Application
Background context: The text provides an example to illustrate how the theorem works. It shows that if \(a = 13\) and \(m = 3\), then both \(13 = 4 \times 3 + 1\) and \(-13 = -5 \times 3 + 2\).

:p How does the proof handle negative integers in the context of the division algorithm?
??x
The proof handles negative integers by noting that if a positive integer \(a\) can be expressed as \(a = mq + r\) with \(0 \leq r < m\), then \(-a\) can also be expressed similarly. By considering the expression for \(-a\) and manipulating it, the proof shows that the theorem still holds for negative integers.

Include relevant code or pseudocode:
```java
public void handleNegativeIntegers(int a, int m) {
    if (m <= 0 || a < 0) {
        throw new IllegalArgumentException("Invalid input for division algorithm");
    }
    // Handle the case where a is negative.
}
```
x??

---

#### Proof by Contradiction in Division Algorithm
Background context: The proof uses contradiction to show that there cannot be a smallest counterexample. This approach assumes the contrary and derives a logical inconsistency.

:p How does the proof use contradiction to prove the existence part of the division algorithm?
??x
The proof uses contradiction by assuming there is a smallest positive integer \(a\) for which the theorem fails, i.e., it cannot be expressed as \(a = mq + r\) with \(0 \leq r < m\). By considering the number \(a - m\), which is both positive and smaller than \(a\), it derives that this assumption leads to a contradiction, proving that such a smallest counterexample cannot exist.

Include relevant code or pseudocode:
```java
public void proofByContradiction(int a, int m) {
    if (m <= 0 || a < 0) {
        throw new IllegalArgumentException("Invalid input for division algorithm");
    }
    // Implement the logic of proof by contradiction here.
}
```
x??

---

#### Proof by Contradiction in Mathematics
Background context: The provided text explains how proof by contradiction is used to prove mathematical statements, particularly focusing on showing that two representations of a number are identical. It mentions the restrictions and logic involved when \( r \) and \( r_0 \) are both less than \( m \).

:p What is the primary method discussed for proving mathematical statements?
??x
Proof by contradiction.
x??

---

#### Divisibility and Contradiction Example
Background context: The text provides an example where it shows that two representations of a number \( a = mq + r \) and \( a = mq_0 + r_0 \) are actually the same, leading to a contradiction.

:p What is the range for \( r - r_0 \)?
??x
The range for \( r - r_0 \) is such that \( 0 < |r - r_0| < m \).
x??

---

#### Contradiction Leading to Same Representation
Background context: The example in the text demonstrates how assuming different representations of a number leads to a contradiction, thus proving they are identical.

:p What does the final conclusion prove?
??x
The final conclusion proves that two representations \( r = r_0 \) and \( q = q_0 \) are indeed the same.
x??

---

#### LaTeX in Mathematical Writing
Background context: The text discusses the importance of using LaTeX for writing mathematical proofs, as it is widely used by mathematicians and academic researchers. It mentions how journals prefer papers submitted in LaTeX.

:p Why do many mathematicians prefer to use LaTeX?
??x
Mathematicians prefer LaTeX because it provides a high-quality typesetting system that allows for precise mathematical symbols, Greek letters, and beautiful graphics, making the proofs more readable and professional.
x??

---

#### Direct vs. Proof by Contradiction
Background context: The text emphasizes the importance of correctly identifying whether a proof should be direct or by contradiction based on the logical structure of the argument.

:p How can one recognize when to use a direct proof instead of a proof by contradiction?
??x
One should use a direct proof if, in the process of proving \( P \rightarrow Q \), you prove \( Q \) directly. If you show that assuming \( \neg Q \) leads to a contradiction, then a proof by contradiction is used.
x??

---

#### Contrapositive Proof
Background context: The text explains another type of logical argument, the contrapositive, which can sometimes be confused with proof by contradiction.

:p How does a proof by contrapositive differ from a proof by contradiction?
??x
A proof by contrapositive involves proving \( \neg Q \rightarrow \neg P \), which is logically equivalent to proving \( P \rightarrow Q \). A proof by contradiction assumes the negation of what you want to prove and shows that this leads to an absurdity.
x??

---

#### Online LaTeX Resources
Background context: The text mentions online resources for using LaTeX, specifically Overleaf.com, which allows users to collaborate on documents.

:p How can one use Overleaf to collaborate with others?
??x
One can use Overleaf by creating a project and inviting collaborators who can then read, edit, and contribute to the document in real-time.
x??

---

#### Direct Proof vs. Proof by Contradiction
Direct proof is a straightforward method where you start from the given premise and logically deduce the conclusion, whereas a proof by contradiction involves assuming the negation of what you want to prove and showing that this leads to an absurdity or contradiction.

:p In which scenario would it be more difficult to identify if a mistake has been made?
??x
In a proof by contradiction, it is more difficult to identify if a mistake has been made because there are multiple potential contradictions. If you make a mistake early in the process, finding out that your conclusion contradicts something else might not clearly indicate where the error occurred.

```java
public class ProofExample {
    public boolean directProof(int n) {
        // Direct proof logic
        for (int i = 2; i <= Math.sqrt(n); i++) {
            if (n % i == 0) return false;
        }
        return true;
    }

    public boolean contradictionProof(int n) {
        // Proof by contradiction: assume not prime and derive a contradiction
        return !directProof(n);
    }
}
```
x??

---

#### Induction Proof
Induction proofs are particularly useful when dealing with statements about all natural numbers or integers. The method involves proving a base case and then showing that if the statement holds for some number \( k \), it also holds for \( k+1 \).

:p What is the primary challenge in identifying mistakes during an induction proof?
??x
In an induction proof, making a mistake within the induction step can make it difficult to identify because the error might not be immediately apparent until you try to apply the inductive hypothesis. If the proof does not hold for \( k+1 \), it could indicate that there was an error earlier in the base case or the inductive step.

```java
public class InductionExample {
    public boolean inductionStep(int n) {
        if (n == 0) return true; // Base case

        // Inductive step: assume true for k and prove for k+1
        return inductionStep(n - 1);
    }
}
```
x??

---

#### Proof by Contradiction
Proof by contradiction involves assuming the negation of what you want to prove (i.e., \( \neg Q \) given \( P \)) and deriving a contradiction. This method is particularly useful when dealing with "for every" or "for all" statements.

:p Why is proof by contradiction more prone to errors than direct proofs?
??x
Proof by contradiction is more prone to errors because it involves working backwards from the assumption of the negation, which can mask mistakes until a contradiction is found. If a mistake occurs early in the process, it might be hard to trace back to where the error originated since multiple assumptions and deductions are made.

```java
public class ContradictionExample {
    public boolean proveStatement(int n) {
        assumeNotQ(n); // Assume not Q for contradiction

        // Some logic that may contain errors
        if (n < 0) return true;
        
        return false; // Contradiction found
    }

    private void assumeNotQ(int n) {
        // Logic to show a contradiction
        System.out.println("Assuming " + n + " is not Q.");
    }
}
```
x??

---

#### Use of Proof by Contradiction for Universal Statements
Proofs by contradiction are especially useful when the theorem asserts that something is true "for every" or "for all" such-and-such. By assuming the negation, you can focus on a specific element that does not satisfy the theorem.

:p Why is proof by contradiction particularly useful in proving universal statements?
??x
Proof by contradiction is particularly useful for proving universal statements because it allows focusing on a specific counterexample or element that contradicts the statement. This makes it easier to derive a contradiction and pinpoint where the initial assumption led to an error.

```java
public class UniversalExample {
    public boolean proveForAll(int n) {
        assumeNotInteresting(n); // Assume not interesting for contradiction

        // Some logic leading to a contradiction
        if (n % 2 == 0 && n > 10) return true; // Contradiction found

        return false;
    }

    private void assumeNotInteresting(int n) {
        System.out.println("Assuming " + n + " is not interesting.");
    }
}
```
x??

---

#### Contrapositive, Contra-diction, Converse, and Counterexample

Background context: In mathematical proofs, these terms are often used to construct or understand arguments. Understanding their distinctions is crucial for writing clear and rigorous proofs.

- **Contrapositive**: The contrapositive of a statement "If P, then Q" is "If not Q, then not P". It is logically equivalent to the original statement.
  
- **Contra-diction**: Proof by contradiction involves assuming the negation of what you want to prove and showing that this leads to a logical inconsistency.

- **Converse**: The converse of a statement "If P, then Q" is "If Q, then P". It is not necessarily true if the original statement is true.

- **Counterexample**: A counterexample disproves a universal claim by providing an instance where the claim does not hold. 

:p What are the differences between contrapositive, contradiction, converse, and counterexample in mathematical proofs?
??x
The contrapositive is logically equivalent to its original conditional statement; it swaps and negates both parts of "If P, then Q" (resulting in "If not Q, then not P"). Proof by contradiction assumes the opposite of what you want to prove and shows that this assumption leads to a contradiction. The converse reverses the order of implication ("If Q, then P") but is not necessarily true if the original statement is true. A counterexample disproves a universal claim by providing a specific instance where it fails.

```java
// Example in Java for understanding contrapositive
public class ContrapositiveExample {
    public static boolean contrapositiveExample(int x) {
        // Assume "If x > 5, then x < 10" is the original statement
        return !(x <= 5); // This corresponds to "if not Q (not x < 10), then not P (not x > 5)"
    }
}
```

The contrapositive in this example effectively means checking if \(x \leq 5\).

```java
// Example of proof by contradiction
public class ProofByContradiction {
    public static boolean proofByContradiction(int a, int b) {
        // Assume "If a is rational and ab is irrational, then b is irrational"
        return !((a != 0 && b == 0) || (a == 0 && b != 0)); // This shows contradiction if not true
    }
}
```

In the proof by contradiction example, we check for conditions that would lead to a logical inconsistency.

```java
// Example of converse in Java
public class ConverseExample {
    public static boolean converseExample(int x) {
        // Assume "If P (x > 5), then Q (x < 10)"
        return !(x <= 5); // This is the converse: "if Q, then P"
    }
}
```

This example inverts the conditions but does not guarantee truth.

```java
// Example of counterexample in Java
public class CounterexampleExample {
    public static boolean checkCounterexample(int n) {
        return !(15 * 2 + 35 * 1 == 1); // This is a counterexample to "There exist m, n such that 15m + 35n = 1"
    }
}
```

The function checks if the equation holds for given values, serving as a counterexample.

x??

---
#### Proving Rational and Irrational Multiplication

Background context: This exercise involves proving properties of rational and irrational numbers. Specifically, it examines the product of a rational number with an irrational number.

:p Prove that if \(a\) is rational and \(ab\) is irrational, then \(b\) must be irrational.
??x
To prove this statement, we can use proof by contradiction. Assume that both \(a\) and \(b\) are rational numbers. Since \(a\) is rational, it can be expressed as \(\frac{p}{q}\) where \(p\) and \(q\) are integers and \(q \neq 0\). If \(ab\) is irrational but we assume \(b\) to also be rational, then \(b\) can be expressed as \(\frac{r}{s}\), where \(r\) and \(s\) are integers and \(s \neq 0\).

The product \(ab = a \cdot b = \left(\frac{p}{q}\right) \cdot \left(\frac{r}{s}\right) = \frac{pr}{qs}\). Since both \(p, q, r,\) and \(s\) are integers with no common factors (assuming they are in simplest form), the product \(\frac{pr}{qs}\) would be a rational number. This contradicts our assumption that \(ab\) is irrational.

Therefore, if \(a\) is rational and \(ab\) is irrational, \(b\) cannot be rational; it must be irrational.

```java
// Pseudocode for proof by contradiction
public class RationalMultiplicationProof {
    public static void proveRationalMultiplication() {
        // Assume a = p/q and b = r/s where p, q, r, s are integers.
        // If ab is irrational but we assume b is rational (b = r/s), then:
        
        // Check if the product is still irrational
        boolean isIrrational = !(isRational(pr * s) && isRational(q * r)); // pr and qs would be integers, check this
        
        System.out.println("Is the product irrational? " + isIrrational);
    }
    
    public static boolean isRational(int num) {
        // Check if a number can be represented as p/q where q != 0
        return true; // This function should return true for all integers, but used here to demonstrate logic
    }
}
```

In the pseudocode, we check that the product of two rational numbers would result in a rational number. The contradiction arises when this does not hold, thus proving \(b\) must be irrational.

x??

---
#### Direct Proof vs. Proof by Contradiction

Background context: This exercise compares direct proofs and proof by contradiction to prove a statement about positive rational numbers and their properties.

:p Prove that if \(x \in Q^+\), then there exists some \(y \in Q^+\) such that \(y < x\). Provide two proofs: one using a direct proof, and one using proof by contradiction.
??x
**Direct Proof**: To prove the statement directly, we need to construct a rational number \(y\) that is strictly less than any given positive rational number \(x \in Q^+\).

1. Let \(x = \frac{p}{q}\) where \(p\) and \(q\) are integers with \(q > 0\).
2. Choose \(y = \frac{p}{q + 1}\). Since \(q + 1 > q\), we have that \(\frac{p}{q + 1} < \frac{p}{q} = x\).

Thus, \(y = \frac{p}{q + 1}\) is a positive rational number and \(y < x\).

**Proof by Contradiction**: Assume the statement is false. That is, suppose for every \(y \in Q^+\), we have \(y \geq x\). Let’s consider the value of \(x = \frac{p}{q}\) again.

1. Suppose there exists no positive rational number \(y\) such that \(y < x\).
2. Then any candidate \(y = \frac{r}{s}\) (where \(r, s > 0\)) must satisfy \(y \geq x\), which means \(\frac{r}{s} \geq \frac{p}{q}\).

By choosing \(s < q + 1\) and \(r = p - 1\), we get a new rational number \(y' = \frac{p-1}{q+1}\). Notice that:

\[ y' = \frac{p - 1}{q + 1} = \frac{p - 1}{q + 1} < \frac{p}{q} = x. \]

This contradicts the assumption that \(y \geq x\) for all positive rational numbers \(y\). Therefore, there must exist some \(y \in Q^+\) such that \(y < x\).

```java
// Direct Proof Example in Java
public class DirectProof {
    public static boolean directProof(double x) {
        // Given x is a positive rational number (x > 0)
        double y = x / (1 + 1); // Choose y as half of x
        return y < x; // Check if y is less than x
    }
}
```

The function checks that \(y\) is indeed less than the given positive rational number \(x\).

```java
// Proof by Contradiction Example in Java
public class ProofByContradiction {
    public static boolean proofByContradiction(double x) {
        // Assume there does not exist y such that y < x for all positive rationals
        double epsilon = 0.1; // Small value to check closeness
        for (int i = 1; i <= 10000; i++) { // Check a range of positive rational numbers
            double y = i / (i + 1); // Construct y as a positive rational number
            if (y < x - epsilon) {
                return true; // Found a valid y, contradiction!
            }
        }
        return false; // No such y found, no contradiction
    }
}
```

In the proof by contradiction example, we try to find a \(y\) that is less than \(x\), but since it exists, this approach would theoretically always find one.

x??

---

#### Functions: Definition and Basic Properties
Functions are a fundamental concept in mathematics, often introduced early on but deeply explored throughout one's mathematical education. A function \( f \) from set \( A \) to set \( B \), denoted as \( f : A \to B \), assigns each element \( x \in A \) exactly one value \( y = f(x) \in B \). The domain of the function is the set \( A \), the codomain is \( B \), and the range (or image) is the subset of \( B \) that includes all values attained by \( f \).

:p Define a function from set \( A \) to set \( B \).
??x
A function \( f : A \to B \) assigns each element in the domain \( A \) exactly one value in the codomain \( B \). The notation is such that for every \( x \in A \), there exists a unique \( y \in B \) where \( y = f(x) \).
x??

---
#### Domain, Codomain, and Range
The terms domain, codomain, and range are crucial in understanding functions. The domain consists of all possible inputs (elements from set \( A \)), the codomain is the entire potential output space (set \( B \)), and the range is the actual subset of the codomain that is achieved by the function.

:p Explain the difference between the codomain and the range.
??x
The codomain \( B \) is the complete set of possible outputs for a function, whereas the range is the specific subset of the codomain that includes all actual output values produced by the function. For example, if \( f(x) = x^2 \) with domain \( \mathbb{R} \), the codomain might be \( \mathbb{R} \) (all real numbers), but the range is only non-negative reals \( [0, +\infty) \).
x??

---
#### Vertical Line Test
The vertical line test is a graphical method to determine if a curve represents a function. If any vertical line intersects the graph at most once, then the graph represents a function. This test ensures both existence (a unique output for each input) and uniqueness.

:p Describe how the vertical line test works.
??x
To use the vertical line test, draw several vertical lines across the graph of a relation. If every vertical line intersects the graph in at most one point, then the graph represents a function. Otherwise, it does not represent a function because there is at least one input with multiple outputs.

```java
public class VerticalLineTest {
    public boolean isFunction(double[] xValues, double[] yValues) {
        for (int i = 0; i < yValues.length - 1; ++i) {
            if ((yValues[i] == yValues[i + 1]) && (xValues[i] != xValues[i + 1])) {
                return false;
            }
        }
        return true;
    }
}
```
x??

---
#### Examples of Functions
Various examples illustrate the concept of functions. For instance, \( f(x) = \lfloor x \rfloor \), known as the floor function, rounds down to the nearest integer. Another example is a function mapping integers to their squares with restrictions.

:p Provide an example of a floor function.
??x
The floor function \( f(x) = \lfloor x \rfloor \) maps any real number \( x \) to the greatest integer less than or equal to \( x \). For instance, \( \lfloor 3.2 \rfloor = 3 \) and \( \lfloor -4.7 \rfloor = -5 \).

```java
public class FloorFunction {
    public int floor(int number) {
        return (int)Math.floor(number);
    }
}
```
x??

---
#### Cartesian Product in Functions
Functions can also involve more complex domains or codomains, such as the Cartesian product of sets. The function \( f(x) = (5 \cos(x), 5 \sin(x)) \) maps real numbers to points on a circle centered at the origin with radius 5.

:p Describe the function mapping real numbers to a circle.
??x
The function \( f(x) = (5 \cos(x), 5 \sin(x)) \) maps each real number \( x \) to a point on a circle of radius 5 centered at the origin. This is because for any angle \( x \), \( \cos(x) \) and \( \sin(x) \) describe coordinates on the unit circle, scaled by 5.

```java
public class CircleMapping {
    public double[] mapToCircle(double x) {
        return new double[]{5 * Math.cos(x), 5 * Math.sin(x)};
    }
}
```
x??

---

#### Set Notation and Function Representation
Background context explaining set notation, functions, and how they are represented. Mention that sets can be defined with curly braces and functions can map elements from one set to another using function notation.

:p What is the example given for a set and its cardinality?
??x The example provided shows the set \( \{1, 5, 12\} \) and states that \( |f\{1, 5, 12\}| = 3 \). This means the set has three elements.
x??

---
#### Function Representation with Braces
Background context explaining how functions are represented using braces around the input. Mention this is different from the typical f(x) or g(t) notation.

:p How does a function like \( G:S.\{A, B, C, D, F\} \) represent student grades in an Intro to Proofs class?
??x The function \( G:S.\{A, B, C, D, F\} \) represents the letter grade that each student received on their last homework assignment. Here, S is the set of all students in the class.
x??

---
#### Injections (One-to-One Functions)
Background context explaining injections and how they ensure no two different elements in the domain map to the same element in the codomain.

:p What does it mean for a function to be injective or one-to-one?
??x A function is injective if \( f(a_1) = f(a_2) \) implies that \( a_1 = a_2 \). In simpler terms, different elements in the domain map to different elements in the codomain. This means there are no two arrows pointing at the same point.
x??

---
#### Example of an Injection
Background context providing examples of functions and explaining why certain mappings are or are not injective.

:p Why is the function from \{x, y, z\} to \{1, 2, 3, 4\} injective?
??x The function is injective because no two different elements in the domain map to the same element in the codomain. For example, if \( f(x) = 2 \), \( f(y) = 3 \), and \( f(z) = 1 \), then each element in the domain maps uniquely to an element in the codomain.
x??

---
#### Non-Example of an Injection
Background context explaining why a function is not injective.

:p Why is the function from \{x, y, z\} to \{1, 2, 3, 4\} not injective?
??x The function is not injective because \( f(x) = 2 \) and \( f(y) = 2 \), meaning two different elements in the domain map to the same element in the codomain. This violates the condition for an injection.
x??

---
#### Contrapositive of Injection Definition
Background context explaining how the contrapositive can provide another way to understand injections.

:p How does the contrapositive help us understand injective functions?
??x The contrapositive turns the implication " \( f(a_1) = f(a_2) \) implies that \( a_1 = a_2 \)" into " \( a_1 \neq a_2 \) implies \( f(a_1) \neq f(a_2) \)". This means if two different points in the domain map to the same point, then it cannot be injective.
x??

---
#### Surjective (Onto Functions)
Background context explaining surjective functions and how they ensure every element in the codomain has a preimage.

:p What does it mean for a function to be surjective or onto?
??x A function is surjective if for every \( b \in B \), there exists some \( a \in A \) such that \( f(a) = b \). In simpler terms, every element in the codomain has at least one preimage in the domain.
x??

---
#### Example of a Surjection
Background context providing examples of functions and explaining why certain mappings are or are not surjective.

:p Why is the function from \{w, x, y, z\} to \{1, 2, 3\} a surjection?
??x The function is a surjection because every element in the codomain has at least one preimage. For example, if \( f(w) = 1 \), \( f(x) = 2 \), and \( f(y) = 3 \), then each element in the codomain is mapped to by some element in the domain.
x??

---
#### Non-Example of a Surjection
Background context explaining why a function is not surjective.

:p Why is the function from \{w, x, y, z\} to \{1, 2, 3\} not a surjection?
??x The function is not a surjection because there exists an element in the codomain that does not have a preimage. Specifically, \( b = 3 \) does not have any corresponding \( a \in \{w, x, y, z\} \) such that \( f(a) = 3 \).
x??

---
#### Contrapositive of Surjective Definition
Background context explaining how the contrapositive can provide another way to understand surjective functions.

:p How does the contrapositive help us understand surjective functions?
??x The contrapositive turns the definition "for every \( b \in B \), there exists some \( a \in A \) such that \( f(a) = b \)" into "there does not exist any \( b \in B \) for which \( f(a) \neq b \) for all \( a \in A \)". This means if every element in the codomain has at least one preimage, then it is surjective.
x??

---
#### Recurring Theme in Function Definitions
Background context explaining how existence and uniqueness criteria shift between domain and codomain when defining injections and surjections.

:p How do the definitions of injective and surjective functions shift focus from existence to uniqueness?
??x When defining a function \( f:A.B \), we focus on existence for every \( x \in A \) that \( f(x) \) exists and is unique. To be injective, the attention shifts to B with an existence criterion (for every \( b \in B \), there exists some \( a \in A \) such that \( f(a) = b \)). And to be surjective, it focuses on a uniqueness-type criterion (for every \( b \in B \), there is at most one \( a \in A \) such that \( f(a) = b \)).
x??

#### Definition of a Bijection
Background context: A bijection is defined as a function that is both injective and surjective. This means every element in set \(A\) is paired with exactly one element in set \(B\), and every element in \(B\) has exactly one element from \(A\) mapping to it.
:p What is the definition of a bijective function?
??x
A function \(f: A \to B\) is bijective if it is both injective (one-to-one) and surjective (onto). This means:
- Every element in set \(A\) is paired with exactly one element in set \(B\).
- Every element in set \(B\) has exactly one element from set \(A\) mapping to it.
In formal terms, for every \(a_1, a_2 \in A\), if \(f(a_1) = f(a_2)\), then \(a_1 = a_2\) (injective). And for every \(b \in B\), there exists an \(a \in A\) such that \(f(a) = b\) (surjective).
x??

---

#### Examples of Injective, Surjective, and Bijective Functions
Background context: The text provides visual aids using Venn diagrams to illustrate the different types of functions. It explains how a function can be injective, surjective, or bijective by examining the relationships between elements in sets \(A\) and \(B\).
:p What are some examples provided for each type of function?
??x
- **Injective Function Example**: A function that is not bijective but only injective. For example:
  ```
    X = {1, 2, 3}
    Y = {4, 5, 6, 7}
    f: X → Y where f(1) = 4, f(2) = 5, f(3) = 6
  ```

- **Surjective Function Example**: A function that is not bijective but only surjective. For example:
  ```
    X = {1, 2, 3}
    Y = {4, 5, 6, 7, 8}
    f: X → Y where f(1) = 4, f(2) = 5, f(3) = 6
  ```

- **Bijective Function Example**: A function that is both injective and surjective. For example:
  ```
    X = {1, 2, 3}
    Y = {4, 5, 6}
    f: X → Y where f(1) = 4, f(2) = 5, f(3) = 6
  ```

In these examples, the function is defined from set \(X\) to set \(Y\), and each type of function is characterized by its unique properties.
x??

---

#### Injective Functions Explained
Background context: The text explains that an injective function means all elements in \(A\) are paired with exactly one element in \(B\). This can be likened to monogamous relationships, where no two elements from \(A\) map to the same element in \(B\).
:p What is the key property of an injective function?
??x
The key property of an injective function (one-to-one) is that each element in set \(A\) maps to a unique element in set \(B\). Formally, for all \(a_1, a_2 \in A\), if \(f(a_1) = f(a_2)\), then it must be true that \(a_1 = a_2\).

Example pseudocode:
```java
public boolean isInjective(HashMap<Integer, Integer> mapping) {
    Set<Integer> values = new HashSet<>();
    for (Map.Entry<Integer, Integer> entry : mapping.entrySet()) {
        if (!values.add(entry.getValue())) { // If value already exists, not injective
            return false;
        }
    }
    return true; // All values are unique, function is injective
}
```
x??

---

#### Surjective Functions Explained
Background context: The text explains that a surjective function means every element in \(B\) has at least one element from \(A\) mapping to it. This can be likened to the idea of everyone finding love.
:p What is the key property of a surjective function?
??x
The key property of a surjective function (onto) is that for every element \(b \in B\), there exists an element \(a \in A\) such that \(f(a) = b\). This means every element in set \(B\) has at least one corresponding element in set \(A\) mapping to it.

Example pseudocode:
```java
public boolean isSurjective(HashMap<Integer, Integer> mapping, Set<Integer> Y) {
    for (Integer y : Y) { // Check if each y in B has a pre-image in A
        if (!mapping.values().contains(y)) {
            return false;
        }
    }
    return true; // All elements of Y have pre-images, function is surjective
}
```
x??

---

#### Bijective Functions Explained
Background context: The text explains that a bijective function has both the properties of an injective and surjective function. This means every element in \(A\) maps to exactly one unique element in \(B\), and every element in \(B\) is mapped from exactly one unique element in \(A\).
:p What are the key properties of a bijective function?
??x
The key properties of a bijective function (one-to-one correspondence) include:
1. **Injective Property**: For all \(a_1, a_2 \in A\), if \(f(a_1) = f(a_2)\), then \(a_1 = a_2\).
2. **Surjective Property**: For every \(b \in B\), there exists an element \(a \in A\) such that \(f(a) = b\).

In summary, a bijective function pairs each element of set \(A\) with exactly one unique element in set \(B\), and vice versa.

Example pseudocode:
```java
public boolean isBijective(HashMap<Integer, Integer> mapping, Set<Integer> A, Set<Integer> B) {
    if (!isInjective(mapping)) return false; // Check injectivity first
    if (!isSurjective(mapping, B)) return false; // Then check surjectivity

    // If both properties are satisfied, it is bijective
    return true;
}
```
x??

---

#### Definition of Bijection and Injectivity

Background context: A function is a bijection if it is both injective (one-to-one) and surjective (onto). To prove that a function is a bijection, one must show that it is both an injection and a surjection. The domain \(A\) and codomain \(B\) are crucial as they define the behavior of the function.

:p What does it mean for a function to be injective?
??x
A function \(f: A \to B\) is injective if every element in the codomain \(B\) has at most one preimage in the domain \(A\). This means that for any two distinct elements \(a_1, a_2 \in A\), if \(f(a_1) = f(a_2)\), then it must be true that \(a_1 = a_2\).

For example, consider the function \(g: \mathbb{R}^+ \to \mathbb{R}\) defined by \(g(x) = x^2\). To show that \(g\) is not injective, we can find two distinct elements in the domain that map to the same element in the codomain. For instance, both 1 and -1 square to 1, so \(g(1) = g(-1) = 1\), showing that \(g\) is not injective.

```java
public class InjectivityExample {
    public static boolean isInjective(double x, double y) {
        return Math.pow(x, 2) != Math.pow(y, 2);
    }
}
```
x??

---

#### Surjectivity and Bijection

Background context: A function \(f: A \to B\) is surjective if every element in the codomain \(B\) has at least one preimage in the domain \(A\). A bijection is a function that is both injective and surjective.

:p What does it mean for a function to be surjective?
??x
A function \(f: A \to B\) is surjective if for every element \(b \in B\), there exists at least one element \(a \in A\) such that \(f(a) = b\). In other words, the function covers all elements of the codomain.

For example, consider the function \(h: \mathbb{R} \to \mathbb{R}^+\) defined by \(h(x) = x^2\). To show that \(h\) is not surjective, note that there is no real number \(x\) such that \(x^2 = -1\), which means the codomain includes elements (like -1) that are not in the image of \(h\).

```java
public class SurjectivityExample {
    public static boolean isSurjective(double y) {
        return y >= 0;
    }
}
```
x??

---

#### Proving a Function is Not Injective, Surjective or Bijective

Background context: The given example involves proving the properties of different functions defined over specific domains and codomains. To show that \(f(x) = x^2\) on \(\mathbb{R}\), we need to prove it is not injective, surjective, or bijective.

:p Prove that \(f(x) = x^2\) for \(x \in \mathbb{R}\) is not injective.
??x
To show that \(f(x) = x^2\) on \(\mathbb{R}\) is not injective, we need to find two distinct elements in the domain that map to the same element in the codomain. Consider the points 1 and -1:
\[ f(1) = 1^2 = 1 \]
\[ f(-1) = (-1)^2 = 1 \]
Since \(f(1) = f(-1)\) but \(1 \neq -1\), the function is not injective.

```java
public class NotInjectiveExample {
    public static boolean isNotInjective(double x, double y) {
        return Math.pow(x, 2) == Math.pow(y, 2) && !Double.compare(x, y);
    }
}
```
x??

---

#### Proving a Function is Injective

Background context: The example shows how to prove that \(g(x) = x^2\) on \(\mathbb{R}^+\) is injective by assuming \(g(x) = g(y)\) and showing that this implies \(x = y\).

:p Prove that \(g(x) = x^2\) for \(x \in \mathbb{R}^+\) is injective.
??x
To prove that \(g(x) = x^2\) on \(\mathbb{R}^+\) is injective, assume that \(g(x) = g(y)\). Then:
\[ x^2 = y^2 \]
Taking the square root of both sides gives:
\[ \sqrt{x^2} = \sqrt{y^2} \]
Since we are working in \(\mathbb{R}^+\), the domain excludes negative numbers, so:
\[ |x| = |y| \implies x = y \]
Thus, \(g(x) = g(y)\) implies \(x = y\), which proves that \(g\) is injective.

```java
public class InjectiveProofExample {
    public static boolean isInjective(double x, double y) {
        return Math.pow(x, 2) == Math.pow(y, 2) && Double.compare(Math.abs(x), Math.abs(y)) == 0;
    }
}
```
x??

---

#### Proving a Function is Not Surjective

Background context: The example shows how to prove that \(h(x) = x^2\) on \(\mathbb{R}\) is not surjective by demonstrating there are elements in the codomain \(\mathbb{R}^+\) with no preimage in the domain.

:p Prove that \(h(x) = x^2\) for \(x \in \mathbb{R}\) is not surjective.
??x
To prove that \(h(x) = x^2\) on \(\mathbb{R}\) is not surjective, we need to find an element in the codomain \(\mathbb{R}^+\) (the set of non-negative real numbers) that does not have a preimage in the domain \(\mathbb{R}\). Consider -1:
\[ h(x) = x^2 = -1 \]
There is no real number \(x\) such that \(x^2 = -1\), so -1 (and any negative number) is an element of the codomain that has no preimage in the domain. Therefore, \(h(x)\) is not surjective.

```java
public class NotSurjectiveExample {
    public static boolean isNotSurjective(double y) {
        return y < 0;
    }
}
```
x??

---

#### Proving a Function is Surjective

Background context: The example shows how to prove that \(k(x) = x^2\) on \(\mathbb{R}^+\) is surjective by showing every element in the codomain \(\mathbb{R}^+\) has at least one preimage in the domain.

:p Prove that \(k(x) = x^2\) for \(x \in \mathbb{R}^+\) is surjective.
??x
To prove that \(k(x) = x^2\) on \(\mathbb{R}^+\) is surjective, we need to show that every element in the codomain \(\mathbb{R}^+\) has at least one preimage in the domain. For any \(y \in \mathbb{R}^+\), there exists a unique positive real number \(x = \sqrt{y}\) such that:
\[ k(x) = x^2 = (\sqrt{y})^2 = y \]
Thus, for every element in the codomain \(\mathbb{R}^+\), there is at least one preimage in the domain, proving that \(k(x)\) is surjective.

```java
public class SurjectiveProofExample {
    public static double findPreimage(double y) {
        return Math.sqrt(y);
    }
}
```
x??

---

#### Proving a Function is Bijective

Background context: The example shows how to prove that \(k(x) = x^2\) on \(\mathbb{R}^+\) is bijective by proving it is both injective and surjective.

:p Prove that \(k(x) = x^2\) for \(x \in \mathbb{R}^+\) is bijective.
??x
To prove that \(k(x) = x^2\) on \(\mathbb{R}^+\) is bijective, we need to show that it is both injective and surjective. We have already proven in previous flashcards that:
1. **Injectivity**: If \(k(x) = k(y)\), then \(x = y\).
2. **Surjectivity**: For any \(y \in \mathbb{R}^+\), there exists a unique \(x = \sqrt{y}\) such that \(k(x) = y\).

Since \(k\) is both injective and surjective, it is bijective.

```java
public class BijectiveProofExample {
    public static boolean isInjective(double x, double y) {
        return Math.pow(x, 2) == Math.pow(y, 2) && Double.compare(Math.abs(x), Math.abs(y)) == 0;
    }

    public static double findPreimage(double y) {
        return Math.sqrt(y);
    }
}
```
x??

#### Function Surjectiveness

Background context explaining the concept. In this section, we discuss surjectiveness of functions and how to prove or disprove it. A function \( f: A \to B \) is **surjective** if for every element \( y \) in the codomain \( B \), there exists at least one element \( x \) in the domain \( A \) such that \( f(x) = y \).

:p How do you show a function is not surjective?
??x
To show a function is not surjective, find an element in the codomain that has no pre-image (i.e., there is no \( x \) in the domain such that \( f(x) = y \)).

Example:
For the function \( f: \mathbb{R} \to \mathbb{R} \) where \( f(x) = x^2 \), we can show it's not surjective by noting there is no real number \( x \) such that \( x^2 = -4 \). Hence, \( -4 \) has no pre-image.

```java
public class NotSurjectiveExample {
    public static boolean checkSurjectiveness(double y) {
        // Check if a negative number exists in the codomain
        return y >= 0;
    }
}
```
x??

---

#### Function Injectiveness

Background context explaining the concept. In this section, we discuss injectiveness of functions and how to prove or disprove it. A function \( f: A \to B \) is **injective** if for every pair of distinct elements \( x_1 \neq x_2 \) in the domain \( A \), their images are different, i.e., \( f(x_1) \neq f(x_2) \).

:p How do you show a function is injective?
??x
To show a function is injective, assume two elements from the domain map to the same element in the codomain and derive a contradiction or prove that no such pair exists.

Example:
For the function \( g: \mathbb{R}^+ \to \mathbb{R} \) where \( g(x) = x^2 \), we can show it is injective by assuming \( g(x_1) = g(x_2) \). This implies \( x_1^2 = x_2^2 \), which means \( x_1 = x_2 \) or \( x_1 = -x_2 \). Since both \( x_1, x_2 \in \mathbb{R}^+ \), the only possibility is \( x_1 = x_2 \).

```java
public class InjectivenessExample {
    public static boolean checkInjectiveness(double x1, double x2) {
        return (x1 * x1 == x2 * x2) && (x1 > 0);
    }
}
```
x??

---

#### Surjectiveness of h and k

Background context explaining the concept. In this section, we discuss how to prove a function is surjective by finding an \( x \) for every \( y \) in the codomain.

:p How do you show that \( h: \mathbb{R} \to \mathbb{R}_+ \) and \( k: \mathbb{R}^+ \to \mathbb{R} \) are surjective?
??x
To show a function is surjective, for any given \( y \) in the codomain, find an appropriate \( x \) such that \( h(x) = y \) or \( k(x) = y \).

Example:
For the function \( h: \mathbb{R} \to \mathbb{R}_+ \) where \( h(x) = |x| \), for any \( b \in \mathbb{R}_+ \), we can find \( x = \sqrt{b} \). Since \( (\sqrt{b})^2 = b \), it follows that \( h(\sqrt{b}) = b \).

```java
public class SurjectiveExample {
    public static double findXForB(double b) {
        return Math.sqrt(b);
    }
}
```
x??

---

#### Bijection of Functions

Background context explaining the concept. In this section, we discuss how to prove a function is bijective by combining injectiveness and surjectiveness.

:p How do you show that \( h: \mathbb{R} \to \mathbb{R}_+ \) is a bijection?
??x
To show a function is bijective, it must be both injective and surjective. For \( h(x) = |x| \):

1. **Injectiveness**: As shown previously, if \( h(x_1) = h(x_2) \), then \( x_1 = x_2 \).
2. **Surjectiveness**: For any \( b \in \mathbb{R}_+ \), we can find \( x = \sqrt{b} \) such that \( h(\sqrt{b}) = b \).

Thus, since both conditions are satisfied, \( h(x) = |x| \) is a bijection.

```java
public class BijectionExample {
    public static boolean checkInjectiveness(double x1, double x2) {
        return (Math.abs(x1) == Math.abs(x2));
    }

    public static double findXForB(double b) {
        return Math.sqrt(b);
    }
}
```
x??

---

#### Ordered Pairs and Equality
Background context explaining the concept of ordered pairs and their equality. In this case, we are dealing with a function that maps an ordered pair to another ordered pair.

:p What does it mean for two ordered pairs \((x_1, y_1)\) and \((x_2, y_2)\) to be equal?
??x
Two ordered pairs \((x_1, y_1)\) and \((x_2, y_2)\) are equal if both their first coordinates \(x_1\) and \(x_2\) are the same and their second coordinates \(y_1\) and \(y_2\) are also the same. Mathematically, this can be written as:
\[ (x_1, y_1) = (x_2, y_2) \iff x_1 = x_2 \text{ and } y_1 = y_2 \]
??x
---

#### Linear Equations and System of Equations
Background context on linear equations and how to solve systems of linear equations. This is relevant when dealing with functions that map ordered pairs in a linear manner.

:p How do you solve the system of linear equations \( x + 2y = a \) and \( 2x + 3y = b \)?
??x
To solve the system of linear equations:
\[ x + 2y = a \]
\[ 2x + 3y = b \]

1. Start by multiplying the first equation by 2 to align it with the second equation:
   \[ 2(x + 2y) = 2a \Rightarrow 2x + 4y = 2a \]

2. Subtract the second original equation from this new equation to eliminate \( x \):
   \[ (2x + 4y) - (2x + 3y) = 2a - b \]
   \[ y = 2a - b \]

3. Substitute \( y = 2a - b \) back into the first original equation:
   \[ x + 2(2a - b) = a \]
   \[ x + 4a - 2b = a \]
   \[ x = a - 4a + 2b \]
   \[ x = -3a + 2b \]

Thus, the solution is:
\[ (x, y) = (-3a + 2b, 2a - b) \]
??x
---

#### Injectivity Proof for Function \( f(x; y) \)
Background context on proving a function is injective by showing that if \( f(x_1; y_1) = f(x_2; y_2) \), then \( (x_1, y_1) = (x_2, y_2) \).

:p How do you prove the function \( f(x; y) = (a; b) \) is injective?
??x
To prove that \( f(x; y) = (x + 2y, 2x + 3y) \) is injective, we assume:
\[ f(x_1; y_1) = f(x_2; y_2) \]
This means:
\[ (x_1 + 2y_1, 2x_1 + 3y_1) = (x_2 + 2y_2, 2x_2 + 3y_2) \]

Therefore, we have the following system of equations:
\[ x_1 + 2y_1 = x_2 + 2y_2 \]
\[ 2x_1 + 3y_1 = 2x_2 + 3y_2 \]

From the first equation:
\[ x_1 + 2y_1 - x_2 - 2y_2 = 0 \]
\[ (x_1 - x_2) + 2(y_1 - y_2) = 0 \]

Multiply the first equation by 2 and subtract it from the second:
\[ 2(x_1 + 2y_1) - (2x_2 + 3y_2) = 2a - b \]
\[ 2x_1 + 4y_1 - 2x_2 - 3y_2 = 0 \]

Subtracting the first modified equation from the second:
\[ (2x_1 + 3y_1) - (2x_1 + 4y_1) = b - a \]
\[ -y_1 = b - a \]
\[ y_1 = a - b \]

Now, substitute \( y_1 = a - b \) into the first equation:
\[ x_1 + 2(a - b) = x_2 + 2y_2 \]
\[ x_1 + 2a - 2b = x_2 + 2y_2 \]

Since \( y_2 = a - b \):
\[ x_1 + 2a - 2b = x_2 + 2(a - b) \]
\[ x_1 = x_2 \]

Thus, we have shown that if \( f(x_1; y_1) = f(x_2; y_2) \), then \( (x_1, y_1) = (x_2, y_2) \). Therefore, the function is injective.
??x
---

#### Surjectivity Proof for Function \( f(x; y) \)
Background context on proving a function is surjective by showing that for any element in the codomain, there exists an element in the domain that maps to it.

:p How do you prove the function \( f(x; y) = (a; b) \) is surjective?
??x
To prove that \( f(x; y) = (x + 2y, 2x + 3y) \) is surjective, we need to show that for any \( (a, b) \in \mathbb{Z}^2 \), there exist integers \( x \) and \( y \) such that:
\[ f(x; y) = (a; b) \]

From our scratch work, we have:
\[ x + 2y = a \]
\[ 2x + 3y = b \]

We can solve these equations for \( x \) and \( y \). From the first equation:
\[ x = a - 2y \]

Substitute into the second equation:
\[ 2(a - 2y) + 3y = b \]
\[ 2a - 4y + 3y = b \]
\[ 2a - y = b \]
\[ y = 2a - b \]

Now substitute \( y = 2a - b \) back into the first equation:
\[ x + 2(2a - b) = a \]
\[ x + 4a - 2b = a \]
\[ x = a - 4a + 2b \]
\[ x = -3a + 2b \]

Thus, for any \( (a, b) \in \mathbb{Z}^2 \), we can find \( x = -3a + 2b \) and \( y = 2a - b \). Therefore, the function is surjective.
??x
---

#### Pigeonhole Principle Application in Functions
Background context on the pigeonhole principle, which states that if more pigeons than pigeonholes are placed into pigeonholes, then at least one pigeonhole must contain more than one pigeon. In functions, it can be used to determine injectivity and surjectivity.

:p How does the pigeonhole principle apply to proving a function is not injective or surjective?
??x
The pigeonhole principle states that if \( |A| > |B| \), then any function \( f: A \to B \) cannot be injective because there are more elements in \( A \) than in \( B \). Similarly, if \( |A| < |B| \), the function cannot be surjective since not all elements of \( B \) can be mapped to by elements of \( A \).

In summary:
- If the domain has more elements than the codomain (\( |A| > |B| \)), then \( f: A \to B \) is **not injective**.
- If the domain has fewer elements than the codomain (\( |A| < |B| \)), then \( f: A \to B \) is **not surjective**.

The contrapositive of these statements also holds:
- If a function is injective, then \( |A| \leq |B| \).
- If a function is surjective, then \( |A| \geq |B| \).

For bijection (both injective and surjective), we need \( |A| = |B| \).
??x
---

