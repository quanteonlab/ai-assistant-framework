# Flashcards: 1B001---Proofs---Jay-Cumming_processed (Part 11)

**Starting Chapter:** Exercises

---

#### Writing Proofs with Caution
Background context: The passage discusses the importance of caution when writing proofs, especially as they become more complex. It references Joan Didion’s insight that writing is both sharing and discovering, highlighting the risk of convincing oneself of false ideas.
:p Why does the author emphasize being a critic of yourself while writing proofs?
??x
The author emphasizes this because proof writing requires rigorous self-testing and critical examination of one's own ideas to ensure they are not falling into fallacies. This process helps in maintaining accuracy and correctness in mathematical arguments.
x??

---
#### Identifying Contrapositive Proof Technique
Background context: The passage explains that recognizing certain characteristics of a proposition can indicate the suitability of using the contrapositive proof technique. It mentions that if a statement involves many "not" conditions, this might suggest trying the contrapositive approach.
:p How can you identify when to use the contrapositive in a proof?
??x
You should consider using the contrapositive when the proposition contains multiple negations (indicated by words like “not” or symbols such as “¬”). The contrapositive transforms these statements into simpler, more manageable forms, making the proof easier to construct.
x??

---
#### Contrapositive Proof for Proposition 6.5
Background context: The passage presents a specific proposition and its proof using the contrapositive method. It explains how transforming the original statement can simplify the proof process by breaking down complex conditions into simpler ones.
:p How does the contrapositive help in proving Proposition 6.5?
??x
The contrapositive helps by converting the original implication "If $36a \equiv 36b \pmod{n}$, then $ n \nmid 36$" into its equivalent form: "If $ n \mid 36$, then $36a \not\equiv 36b \pmod{n}$". This makes the proof more straightforward because it involves working with divisibility rather than congruence, which simplifies the algebraic manipulations.
x??

---
#### Contrapositive Proof Steps
Background context: The passage provides a detailed proof of Proposition 6.5 using the contrapositive method. It breaks down each step and explains how to use definitions and algebraic manipulation to reach the conclusion.
:p What are the key steps in proving $n \nmid 36 $ given that$36a \equiv 36b \pmod{n}$ via contrapositive?
??x
The key steps involve assuming $n \mid 36 $, then showing that this assumption leads to $36a \equiv 36b \pmod{n}$. Specifically:
1. Assume $n \mid 36 $; by definition, there exists $ k_1 \in \mathbb{Z}$such that $36 = nk_1$.
2. Show that this leads to $36a - 36b = n(ka - kb)$, implying $ n \mid (36a - 36b)$.
3. Conclude that $36a \equiv 36b \pmod{n}$ because of the above step.
x??

---
#### Using Quadratic Formula and Lemmas
Background context: The passage briefly mentions using the quadratic formula and lemmas in proofs, indicating that these tools can be crucial for solving complex problems. It suggests stating lemmas upfront to make the main proof more manageable.
:p How do you use a lemma in a larger proof?
??x
You use a lemma by first formally stating it as an auxiliary result. Then, within your main proof, you invoke this lemma to simplify or break down parts of the argument that would otherwise be complex. This modular approach helps in managing the complexity of proofs.
x??

---
#### Lemma Statement and Usage
Background context: The passage introduces a lemma with two parts, indicating its role as an intermediate result that can aid in proving more complex propositions. Lemmas are often used to break down larger problems into smaller, more manageable pieces.
:p What is the purpose of stating a lemma before using it?
??x
The purpose of stating a lemma before using it is to establish a smaller, self-contained result that can be utilized within the main proof. This modular approach helps in organizing complex proofs and making them easier to understand and verify step-by-step.
x??

---

#### Lemma 6.6 Part (i)
Background context: This lemma deals with the property of sums involving even numbers. If $m \in \mathbb{Z}$, then $ m^2 + m$ is always an even number.

:p What does this lemma state about the sum of a square and itself for any integer?
??x
If $m $ is an integer, then$m^2 + m$ is always even.
This is because:
- If $m $ is even, both$m^2 $ and$m$ are even, so their sum is even.
- If $m $ is odd, both$m^2 $ and$m$ are odd, and the sum of two odds is even.

No code examples are needed for this lemma:
```java
// No code example needed as it's a mathematical property
```
x??

---

#### Lemma 6.6 Part (ii)
Background context: This part of the lemma states that if $a \in \mathbb{Z}$ and $ a^2 $ is even, then $a$ must also be even.

:p What does this lemma state about an integer squared being even?
??x
If $a \in \mathbb{Z}$ and $ a^2 $ is even, then $a$ is even.
This follows from the properties of even numbers: if the square of a number is even, the original number must be even.

No code examples are needed for this lemma:
```java
// No code example needed as it's a mathematical property
```
x??

---

#### Proposition 6.7
Background context: This proposition states that if $a $ is an odd integer, then the quadratic equation$x^2 + x - a^2 = 0$ has no integer solutions.

:p What does this proposition claim about the equation $x^2 + x - a^2 = 0$?
??x
If $a $ is an odd integer, then the equation$x^2 + x - a^2 = 0$ has no integer solutions.
This is proven using the contrapositive method and properties of even and odd numbers.

No code examples are needed for this proposition:
```java
// No code example needed as it's a proof concept
```
x??

---

#### Proof Strategy: Contrapositive
Background context: The proof uses the contrapositive to show that if there were an integer solution, then $a$ would not be odd. This involves several steps, including using the quadratic formula and properties of even and odd numbers.

:p What is the main strategy used in this proof?
??x
The main strategy is to use the contrapositive: assuming there is an integer solution for $x^2 + x - a^2 = 0 $ implies that$a$ cannot be odd.
This involves several steps:
1. Assume $m \in \mathbb{Z}$ and $m^2 + m - a^2 = 0$.
2. Show $m^2 + m$ is even (by Lemma 6.6 part i).
3. Conclude $a^2$ must be even.
4. Therefore, $a$ must be even.

No code examples are needed for this proof strategy:
```java
// No code example needed as it's a logical reasoning concept
```
x??

---

#### Contrapositive and Theorems in General
Background context: This section discusses the equivalence of statements expressed as $P \rightarrow Q $ and$\neg Q \rightarrow \neg P$, and how to choose between direct proof or contrapositive when writing proofs.

:p What is the relationship between $P \rightarrow Q$ and its contrapositive?
??x
The statement $P \rightarrow Q $ is logically equivalent to$\neg Q \rightarrow \neg P $. Therefore, if a theorem is proven by proving the contrapositive, it can also be stated as $\neg Q \rightarrow \neg P$.

No code examples are needed for this general concept:
```java
// No code example needed as it's a logical reasoning concept
```
x??

---

#### Proof by Contradiction (Future Topic)
Background context: The text mentions that there is also a proof of the proposition by contradiction, which will be covered in the next chapter.

:p What other method can be used to prove the same proposition?
??x
The proposition can also be proved by contradiction. This involves assuming the opposite of what we want to prove and showing it leads to a contradiction.
In this case, assume $x^2 + x - a^2 = 0$ has an integer solution, and derive a contradiction based on properties of even and odd numbers.

No code examples are needed for this future topic:
```java
// No code example needed as it's a future topic concept
```
x??

#### Contrapositive, Converse, and Counterexample
Background context: Understanding these logical concepts is crucial for constructing valid proofs. The contrapositive of a statement "If P, then Q" is "If not Q, then not P," which is logically equivalent to the original statement. The converse of "If P, then Q" is "If Q, then P," and it is not necessarily true if the original statement is true. A counterexample disproves a general statement by providing an instance where the statement does not hold.

:p Explain the difference between the contrapositive, the converse, and a counterexample.
??x
The contrapositive of "If P, then Q" is "If not Q, then not P," which is logically equivalent to the original statement. The converse is "If Q, then P," and it is not necessarily true even if the original statement is true. A counterexample disproves a general statement by providing an instance where the statement does not hold.

Example:
- Original: If n² - 4n + 7 is even, then n is odd.
  - Contrapositive: If n is even, then n² - 4n + 7 is odd.
  - Converse: If n is odd, then n² - 4n + 7 is even.

:p Give an example of a real-world implication and its contrapositive.
??x
A real-world implication could be "If it rains, the ground gets wet." The contrapositive would be "If the ground does not get wet, then it did not rain."

Example:
- Implication: If it rains, the ground gets wet.
  - Contrapositive: If the ground does not get wet, then it did not rain.

:p Give an example of a mathematical implication and its contrapositive.
??x
A mathematical implication could be "If m and n are integers, and mn is odd, then both m and n are odd." The contrapositive would be "If either m or n is even, then mn is even."

Example:
- Implication: If mn is odd, then m and n are both odd.
  - Contrapositive: If either m or n is even, then mn is even.

:p Prove the statement using its contrapositive.
??x
Prove "If $n^2 - 4n + 7 $ is even, then$n $ is odd" by proving its contrapositive: "If$ n $ is even, then $n^2 - 4n + 7$ is odd."

Example:
- Given n is even, write $n = 2k$.
- Then $n^2 - 4n + 7 = (2k)^2 - 4(2k) + 7 = 4k^2 - 8k + 7$.
- Notice that $4k^2 - 8k$ is even, so the expression becomes an odd number plus 1.
- Therefore,$n^2 - 4n + 7$ is odd.

??x
The contrapositive of "If $n^2 - 4n + 7 $ is even, then$n $ is odd" is "If$n $ is even, then$n^2 - 4n + 7 $ is odd." We assumed$n $ is even and wrote$n = 2k $. Substituting this into the expression $ n^2 - 4n + 7$gives:
$$n^2 - 4n + 7 = (2k)^2 - 4(2k) + 7 = 4k^2 - 8k + 7.$$

Since $4k^2 - 8k$ is even, adding 1 results in an odd number. Therefore, the contrapositive is true, proving the original statement.

---

---

#### Reductio ad Absurdum (Proof by Contradiction)
Background context: The method of reductio ad absurdum, also known as proof by contradiction, is a common logical argument where one assumes the opposite of what is to be proven and then shows that this assumption leads to a contradiction or an absurdity. This contradiction implies that the original assumption must be false, thus proving the statement in question.
:p What does reductio ad absurdum (proof by contradiction) involve?
??x
Proof by contradiction involves assuming the negation of what you want to prove and then showing that this assumption leads to a logical contradiction or an absurdity. If such a contradiction is found, it means the initial assumption was false, thereby proving the original statement.
```
// Example in Java:
public class ProofByContradictionExample {
    public static void main(String[] args) {
        boolean assumption = true; // Assume we are dating
        if (assumption) { // If assumption is true
            System.out.println("You would have told your mom.");
        }
        // Since you didn't tell her, the assumption must be false.
    }
}
```
x??

---

#### The Truth Table of P → Q
Background context: The implication $P \rightarrow Q $(read as "P implies Q") is a logical connective that forms a compound statement from two simpler statements. It evaluates to true unless $ P $ is true and $ Q$ is false. The truth table for this relationship provides a clear way to understand when the implication is valid.
:p What is the truth table for $P \rightarrow Q$?
??x
The truth table for $P \rightarrow Q$ looks like this:
```
P  |  Q  | P → Q
---|-----|------
T  |  T  |   T
T  |  F  |   F
F  |  T  |   T
F  |  F  |   T
```
The only case where $P \rightarrow Q $ is false is when$P $ is true and$Q$ is false. This relationship is crucial in understanding proof by contradiction.
x??

---

#### Applying Proof by Contradiction to Prove a Statement
Background context: In mathematical proofs, we often use the technique of proof by contradiction to establish the validity of a statement. We assume the negation of the statement we want to prove and show that this assumption leads to a logical contradiction. This contradiction implies that our initial assumption was false, thereby proving the original statement.
:p How does one apply proof by contradiction?
??x
To apply proof by contradiction:
1. Assume the opposite (negation) of what you want to prove.
2. Derive a contradiction from this assumption.
3. Conclude that the original statement must be true.

For example, if we want to prove "There is no largest integer," we assume there is one and call it $N $. We then derive that $ N+1 $is also an integer larger than$ N $, leading to a contradiction since$ N$ was assumed to be the largest.
```
// Example in Java:
public class ProofByContradictionExample {
    public static void main(String[] args) {
        int N = 100; // Assume there exists a largest integer, say 100
        if (isLargestInteger(N)) { // Check if N is the largest integer
            System.out.println("N is the largest integer.");
        } else {
            System.out.println("There must be no largest integer.");
        }
    }

    public static boolean isLargestInteger(int N) {
        int nextInt = N + 1;
        return !isInteger(nextInt); // Assume there's a larger integer
    }

    public static boolean isInteger(int x) { // Placeholder for actual check
        return true; // For simplicity, assume all numbers are integers
    }
}
```
x??

---

#### Proof by Contradiction and Logical Implication
Background context: In formal logic, the implication $P \rightarrow Q $ means that if$P $ is true, then$Q$ must also be true. The truth table for this relationship can help us understand how to use proof by contradiction effectively.
:p How does the truth table of $P \rightarrow Q$ relate to proof by contradiction?
??x
The truth table for $P \rightarrow Q$ helps in understanding that if we want to prove a statement using proof by contradiction:
- We assume $P $ is true and$Q$ is false.
- If this assumption leads to a logical contradiction, then the original statement must be true.

This relationship shows that when assuming the negation of what you want to prove ($P \rightarrow Q$), if it results in a contradiction, the implication itself is valid.
```
// Example in Java:
public class ContradictionExample {
    public static void main(String[] args) {
        boolean P = true; // Assume something that leads to a contradiction
        boolean Q = false; // Assume its negation

        if (P && !Q) { // Check the assumption
            System.out.println("This should lead to a contradiction.");
        } else {
            System.out.println("No contradiction, so the statement is true.");
        }
    }
}
```
x??

---

#### Proof by Contradiction in Everyday Reasoning
Background context: The concept of proof by contradiction can be applied not only in formal logic but also in everyday reasoning. For example, when your mom asks if you are dating someone and you respond that if you were, she would know, this is an informal application of proof by contradiction.
:p How does the mom’s question relate to proof by contradiction?
??x
When your mom asks "Are you dating anyone right now?" and you respond "No, if I were I would have told you," this is a form of reductio ad absurdum or proof by contradiction. You are assuming that you are dating (P) and then showing that this assumption leads to the absurdity that you would have already told your mom (Q), which contradicts the reality that she has not been told.

This argument follows the structure:
- Assume P is true.
- If P is true, then Q must also be true.
- Since Q is false, our initial assumption P must be false.
```
// Example in Java:
public class EverydayProofByContradiction {
    public static void main(String[] args) {
        boolean dating = false; // Assume you are not dating
        boolean toldMom = false; // Assume she hasn't been told

        if (dating && !toldMom) { // Check the assumption
            System.out.println("This should lead to a contradiction.");
        } else {
            System.out.println("No contradiction, so you must not be dating.");
        }
    }
}
```
x??

---

#### Proposition 7.1 - Largest Natural Number
Background context: This proposition states that there does not exist a largest natural number. We will prove this by contradiction, assuming there is one and then finding a larger number.

:p How do we start the proof of Proposition 7.1?
??x
We assume for a contradiction that there is a largest element of $\mathbb{N}$, and call this number $ N$. This means $ N$has the property that $ N \geq m$for all $ m \in \mathbb{N}$.

By assuming $N $ is the largest, we deduce that$N+1 $ must be larger. If it weren't, then$0 \geq 1$, which is clearly a contradiction.
x??

---

#### Proof by Contradiction - Largest Natural Number
Background context: The proof involves showing that if $N $ is assumed to be the largest natural number, we can find a number larger than$N$, leading to a contradiction.

:p What is the logical step in proving Proposition 7.1?
??x
We assume for a contradiction that there exists a largest element of $\mathbb{N}$ and call this number $N$. By assumption,$ N $is larger than every natural number, meaning$ N \geq m $for all$ m \in \mathbb{N}$.

Since $N \in \mathbb{N}$, we also have $ N+1 \in \mathbb{N}$ and thus:
$$N \geq N + 1.$$

Subtracting $N$ from both sides, we get:
$$0 \geq 1.$$

This is a contradiction since $0 < 1$.

Therefore, there cannot be a largest natural number.
x??

---

#### Proposition 7.2 - Smallest Positive Rational Number
Background context: This proposition states that there does not exist the smallest positive rational number. We will prove this by contradiction, assuming such a number exists and finding a smaller one.

:p How do we start the proof of Proposition 7.2?
??x
We assume for a contradiction that there is a smallest positive rational number, and call this number $q $. By definition, $ q = \frac{a}{b}$where $ a, b \in \mathbb{Z}$,$ b > 0 $, and both are positive since$ q > 0$.

We need to show that there exists a smaller rational number than $q$.
x??

---

#### Proof by Contradiction - Smallest Positive Rational Number
Background context: The proof involves showing that if $q$ is assumed to be the smallest positive rational number, we can find a smaller rational number, leading to a contradiction.

:p What is the logical step in proving Proposition 7.2?
??x
We assume for a contradiction that there exists a smallest positive rational number and call this number $q $. Then, since $ q $is rational,$ q = \frac{a}{b}$where $ a, b \in \mathbb{Z}$,$ b > 0$, and both are positive.

We can now find a smaller such number. For example, consider $\frac{a}{2b}$. If $\frac{a}{b}$ is rational and positive, then $\frac{a}{2b}$ will also be too. And why is $\frac{a}{2b}$ smaller?

Let's do some scratch work:
$$\frac{a}{2b} < \frac{a}{b}.$$

Multiplying both sides by $2b$:
$$a < 2a.$$

Subtracting $a$ from both sides:
$$0 < a.$$

Since we know $a > 0 $, if we do this same scratch work in reverse, we start with something we know ($0 < a $) and conclude with the statement we want ($\frac{a}{2b} < \frac{a}{b}$). This concluding inequality gives us our contradiction.

Therefore, there cannot be a smallest positive rational number.
x??

---

#### Proof Techniques - Contradiction Form
Background context: Both Proposition 7.1 and Proposition 7.2 use the method of proof by contradiction to show that certain propositions do not hold. The general form involves assuming the negation of what we want to prove, then deriving a contradiction.

:p How does the proof-by-contradiction work in both cases?
??x
In both cases, we assume the opposite of what we need to prove:
1. For Proposition 7.1: Assume there is a largest natural number $N$.
2. For Proposition 7.2: Assume there is a smallest positive rational number $q = \frac{a}{b}$.

We then show that this assumption leads to a contradiction, thus proving the original statement.

The logical steps involve:
- Starting with an assumption.
- Deriving logical consequences from the assumption.
- Showing that these consequences lead to a known falsehood (e.g., $0 \geq 1 $, or $ a > 2a$).

This process demonstrates that our initial assumption was incorrect, and thus the original statement must be true.
x??

---

#### Contradiction Method - Application
Background context: Both proofs use the method of contradiction to show the non-existence of certain elements in sets. The core idea is to assume the existence of such an element and derive a logical inconsistency.

:p How do you apply proof by contradiction in these specific propositions?
??x
In both Propositions 7.1 and 7.2, we use proof by contradiction:
- **Proposition 7.1**: Assume there exists a largest natural number $N $. Show that this assumption leads to $0 \geq 1$, which is false.
- **Proposition 7.2**: Assume there exists the smallest positive rational number $q = \frac{a}{b}$. Show that you can find a smaller rational number, leading to $ a > 2a$, which is false.

By showing these contradictions, we prove the original propositions.
x??

---

#### Poetic Interpretation of Proof by Contradiction
Background context: The text includes a poem about proof by contradiction. It describes it as a method where one assumes something false and then finds that assumption leads to a logical inconsistency.

:p What is the essence of the poem on proof by contradiction?
??x
The essence of the poem highlights the strangeness and effectiveness of proof by contradiction:
- You assume something false (a crooked person in prison).
- Then, you look for any logical friction or inconsistency.
- If you find such a contradiction, it means your initial assumption was incorrect.

This method is considered one of the most powerful yet bizarre techniques in mathematical proofs.
x??

---

