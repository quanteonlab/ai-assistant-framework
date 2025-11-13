# Flashcards: 1B001---Proofs---Jay-Cumming_processed (Part 14)

**Starting Chapter:** The Composition

---

#### Composition of Functions
Composition of functions is a fundamental concept where the output of one function serves as the input to another. Given two functions $g: A \to B $ and$f: B \to C $, their composition, denoted by$ f \circ g$, is defined such that:
$$(f \circ g)(a) = f(g(a))$$

This means we first apply the function $g $ to an element in set$A $ and then use the result as input for the function $ f$.

If we have a practical example, imagine you are dealing with a process in which one step transforms data using function $g $, and the next step uses that transformed data through another transformation defined by $ f $. The overall effect is the same as applying the combined function$ f \circ g$.
:p How does composition of functions work?
??x
Composition of functions combines two functions where the output of one serves as input to the other. For example, if you have a function $g $ that maps from set$A $ to set$ B $, and another function $ f$that maps from set $ B$to set $ C$, their composition is defined as:
$$(f \circ g)(a) = f(g(a))$$

This process can be visualized in a diagram where the output of $g $ flows into the input of$f$.
x??

---

#### Example with Functions
Consider the following functions:
- $g: \mathbb{R} \to \mathbb{R}$ defined by $g(x) = x + 1$-$ f: \mathbb{R} \to \mathbb{R}_+$defined by $ f(x) = x^2$ The composition of these functions is:
$$(f \circ g)(x) = f(g(x)) = f(x+1) = (x+1)^2$$:p What are the definitions and compositions for given functions $ g $ and $ f$?
??x
For the provided functions, we have:
- $g: \mathbb{R} \to \mathbb{R}$, where $ g(x) = x + 1$-$ f: \mathbb{R} \to \mathbb{R}_+$, where $ f(x) = x^2$The composition of these functions is:
$$(f \circ g)(x) = f(g(x)) = f(x+1) = (x+1)^2$$

This means we first apply $g $ to any real number, resulting in that number plus one. Then, we take the result and square it using function$f$.
x??

---

#### Injectivity of Composition
If both functions are injective, their composition is also injective. This can be proven by showing that if the output of the composed function for two inputs is equal, then those inputs must have been the same initially.
:p How does injectivity work in the context of function composition?
??x
Injectivity means that each element in the domain maps to a unique element in the codomain. For functions $g: A \to B $ and$f: B \to C$, if both are injective, then:
- If $g(a_1) = g(a_2)$ implies $a_1 = a_2$- If $ f(b_1) = f(b_2)$implies $ b_1 = b_2$Thus, for the composition $ f \circ g: A \to C$, if:
$$(f \circ g)(a_1) = (f \circ g)(a_2)$$

Then:
$$f(g(a_1)) = f(g(a_2))$$

Since $f$ is injective, this implies:
$$g(a_1) = g(a_2)$$

And since $g$ is also injective, we conclude:
$$a_1 = a_2$$

Therefore, the composition $f \circ g$ is injective.
x??

---

#### Surjectivity of Composition
If both functions are surjective, their composition is also surjective. This can be proven by showing that for any element in the codomain $C $, there exists an element in the domain $ A$ such that the composed function maps to it.
:p How does surjectivity work in the context of function composition?
??x
Surjectivity means every element in the codomain is mapped to by at least one element in the domain. For functions $g: A \to B $ and$f: B \to C$, if both are surjective, then:
- For any $b \in B $, there exists some $ a \in A $such that$ g(a) = b $- For any$ c \in C $, there exists some$ b \in B $such that$ f(b) = c $Thus, for the composition$ f \circ g: A \to C$, if:
$$c \in C$$

Then we can find $b \in B $ such that$f(b) = c $ and subsequently find $ a \in A $ such that $ g(a) = b $. This shows that for any$ c \in C$:
$$(f \circ g)(a) = f(g(a)) = c$$

Therefore, the composition $f \circ g$ is surjective.
x??

---

#### Corollary of Composition
From the previous two theorems, we can conclude:
- If both functions are injective, their composition is injective.
- If both functions are surjective, their composition is surjective.

These results follow directly from the definitions and properties of injectivity and surjectivity.
:p What corollary can be derived from the previous two theorems?
??x
From the previous theorems:
- If $g: A \to B $ and$f: B \to C $ are both injective, then their composition$f \circ g: A \to C$ is also injective.
- If $g: A \to B $ and$f: B \to C $ are both surjective, then their composition$f \circ g: A \to C$ is also surjective.

These results can be combined into a corollary:
If $g $ and$f$ are either both injective or both surjective, the resulting composition function will retain that property.
x??

---

#### Function Composition and Bijectivity

Background context: The text discusses how function composition works, specifically when functions $g:A \to B $ and$f:B \to C $ are involved. It explains that if both functions are bijective (both one-to-one and onto), then their composition$f \circ g$ is also bijective.

:p What does the text say about function composition involving bijections?
??x
The text states that if $g:A \to B $ and$f:B \to C $ are both bijective functions, then their composition$f \circ g: A \to C$ is also a bijection. This conclusion follows from the fact that:

1. The composition of two injective (one-to-one) functions results in an injective function.
2. The composition of two surjective (onto) functions results in a surjective function.

This can be seen by using Theorem 8.13 for injectivity and Theorem 8.14 for surjectivity, which together imply the bijectivity of $f \circ g$.

??x
```java
// Pseudocode to demonstrate composition of functions in Java
public class FunctionComposition {
    public static int composeFunctions(int a) {
        // Suppose g(x) = x + 1 and f(y) = y * 2
        int b = g(a);
        int c = f(b);
        return c;
    }

    private static int g(int a) { return a + 1; } // Example function g: Z -> Z
    private static int f(int b) { return b * 2; } // Example function f: Z -> Z

    public static void main(String[] args) {
        System.out.println(composeFunctions(5)); // Expected output: 12 (since 5 + 1 = 6, then 6 * 2 = 12)
    }
}
```
x??

---

#### Identity Function and Invertibility

Background context: The text introduces the identity function $i_A:A \to A $ where$i_A(x) = x $ for every$ x \in A $. It then discusses how a function $ f:A \to B$can have an inverse $ f^{-1}:B \to A$, provided that both functions are bijective (one-to-one and onto).

:p What is the definition of an inverse function according to the text?
??x
According to Definition 8.16, a function $f:A \to B$ has an inverse if it exists such that:

- The composition $f^{-1} \circ f = i_A $, where $ i_A$ is the identity function on set A.
- The composition $f \circ f^{-1} = i_B $, where $ i_B$ is the identity function on set B.

In simpler terms, applying a function followed by its inverse (or vice versa) should result in the original input value.

??x
```java
// Pseudocode to demonstrate finding an inverse of a function in Java
public class InverseFunction {
    public static double findInverse(double x) {
        if (x < 0 || x > 1) {
            throw new IllegalArgumentException("Input must be between 0 and 1");
        }
        return 2 * x; // Example: f(x) = 2x -> f^(-1)(x) = x/2
    }

    public static double checkInverse(double x) {
        return findInverse(findInverse(x)); // Check if applying the function twice returns the original value
    }

    public static void main(String[] args) {
        System.out.println(checkInverse(0.5)); // Expected output: 0.5 (since f(f^(-1)(0.5)) = 2 * 0.25 = 0.5)
    }
}
```
x??

---

#### Invertibility and Bijection

Background context: The text explains that for a function $f:A \to B$ to be invertible, it must be both injective (one-to-one) and surjective (onto). This means every element in the domain A maps uniquely to an element in the codomain B, and every element in B is mapped from at least one element in A.

:p Why does a function need to be bijective to have an inverse?
??x
A function needs to be both injective and surjective (bijective) to ensure it has an inverse for several reasons:

1. **Injectivity**: Ensures that each input maps to a unique output, meaning $f(x_1) = f(x_2) \implies x_1 = x_2$. This property is necessary because if the same output could be produced by different inputs, then there would be ambiguity when trying to determine which input corresponds to a given output.
   
2. **Surjectivity**: Ensures that every element in the codomain B is mapped from at least one element in the domain A, meaning for any $b \in B $, there exists an $ a \in A $ such that $ f(a) = b$. This property ensures that no output is "left out," so each output can be traced back to its corresponding input.

These conditions together ensure that the function pairs every element in A with exactly one unique element in B, allowing for a well-defined inverse function. If either condition fails, the inverse cannot be defined unambiguously or might not exist at all.

??x
```java
// Pseudocode to check if a function is bijective and thus invertible
public class BijectivityCheck {
    public static boolean isInjective(int[] domain, int[] codomain) {
        Set<Integer> seen = new HashSet<>();
        for (int x : domain) {
            int y = f(x); // Assume f is the function mapping from domain to codomain
            if (!seen.add(y)) return false; // If we see the same y twice, it's not injective
        }
        return true;
    }

    public static boolean isSurjective(int[] domain, int[] codomain) {
        Set<Integer> seen = new HashSet<>();
        for (int x : domain) {
            seen.add(f(x)); // Map from domain to codomain and check coverage of codomain
        }
        return seen.size() == codomain.length; // Check if every element in codomain is covered
    }

    public static boolean isBijective(int[] domain, int[] codomain) {
        return isInjective(domain, codomain) && isSurjective(domain, codomain);
    }

    public static void main(String[] args) {
        int[] domain = {1, 2, 3, 4};
        int[] codomain = {5, 6, 7, 8};

        System.out.println(isBijective(domain, codomain)); // Expected output: true
    }
}
```
x??

---

#### Definition of Injection and Bijection
Background context explaining that an injection is a function where each element of the domain maps to a unique element in the codomain, ensuring no two different elements in the domain map to the same element in the codomain. A bijection is both an injection and a surjection.
:p What does it mean for a function $f$ to be an injection?
??x
A function $f$ is an injection if each element of its domain maps to a unique element in its codomain, meaning no two different elements in the domain map to the same element in the codomain. Formally:
$$\forall x_1, x_2 \in A : f(x_1) = f(x_2) \implies x_1 = x_2$$

For example, consider a function $f: A \rightarrow B$:
```java
public class InjectionExample {
    public boolean isInjection(int[] domain, int[] codomain) {
        Map<Integer, Integer> map = new HashMap<>();
        for (int i = 0; i < domain.length; i++) {
            if (!map.put(domain[i], codomain[i]).equals(null)) {
                return false;
            }
        }
        return true;
    }
}
```
x??

---

#### Proof of Bijection Using Inverse Function
Background context explaining that a bijection is both an injection and a surjection, and how the inverse function $f^{-1}$ can be used to prove this. The proof involves applying $f^{-1}$ to both sides of the equation when given $f(a_1) = f(a_2)$.
:p How do you prove that a function $f$ is bijective using its inverse?
??x
To prove that a function $f $ is bijective, we first show it is an injection and then a surjection. Given$f:A \rightarrow B $, assume$ f(a_1) = f(a_2)$. Since $ f^{-1} : B \rightarrow A$, applying $ f^{-1}$ to both sides gives:
$$f(a_1) = f(a_2) \implies f^{-1}(f(a_1)) = f^{-1}(f(a_2)) \implies a_1 = a_2$$

This proves that $f $ is an injection. For surjection, for every$b \in B $, there exists some$ a \in A $such that$ f(a) = b$.

For example:
```java
public class BijectionProof {
    public boolean isBijection(Map<Integer, Integer> map) {
        Set<Integer> codomainElements = new HashSet<>();
        for (Map.Entry<Integer, Integer> entry : map.entrySet()) {
            if (!codomainElements.add(entry.getValue())) {
                return false;
            }
        }
        return true;
    }
}
```
x??

---

#### Universal Lossless Compression Algorithm
Background context explaining that a universal lossless compression algorithm is one which can take any file and compress it without losing data, and the proof by contradiction showing no such algorithm exists.
:p Why does there not exist a universal lossless compression algorithm?
??x
There cannot exist a universal lossless compression algorithm because if such an algorithm $f $ existed, applying it to all messages of length at most$n $ would result in messages of length at most$n-1$. This implies that the set of possible compressed files is smaller than the set of original files, which violates the pigeonhole principle since there are more original files than compressed ones.

Formally:
$$|A| > |B|$$where $ A $ is the set of all messages of length at most $ n $, and$ B $ is the set of all messages of length at most $ n-1$.

Thus, by the pigeonhole principle, $f $ cannot be injective. Since a bijection must be both an injection and surjection, and$f$ is not injective, it cannot be bijective, hence no universal lossless compression algorithm can exist.

Example code:
```java
public class CompressionExample {
    public boolean canCompressAllFiles() {
        // Assume A has more elements than B
        Set<String> originalFiles = new HashSet<>(Arrays.asList("file1", "file2", "file3"));
        Set<String> compressedFiles = new HashSet<>(Arrays.asList("compressedFile1", "compressedFile2"));

        return originalFiles.size() > compressedFiles.size();
    }
}
```
x??

---

#### Finding Inverses of Functions
Background context explaining how to find the inverse of a function using algebraic manipulation, specifically for $f(x) = \frac{1}{x+1}$.
:p How do you find the inverse of the function $f(x) = \frac{1}{x+1}$?
??x
To find the inverse of the function $f(x) = \frac{1}{x+1}$, we switch $ x$and $ y$:
$$y = \frac{1}{x+1} \implies x = \frac{1}{y+1}$$

Solving for $y$ gives:
$$y = \frac{1}{x-1}$$

Thus, the inverse function is:
$$f^{-1}(x) = \frac{1}{x-1}$$

Example code:
```java
public class InverseFunction {
    public double findInverse(double x) {
        return 1.0 / (x - 1);
    }
}
```
x??

---

#### Practical Example of Bijection Proof
Background context explaining the steps to prove a function is bijective by showing it is both injective and surjective.
:p Prove that $f(x) = \frac{1}{x+1}$ for $x \in (0, 1)$ is a bijection.
??x
To prove that $f: (0,1) \rightarrow (0,1)$ where $f(x) = \frac{1}{x+1}$ is a bijection:

**Injective:**
Assume $x_1, x_2 \in (0, 1)$ and $f(x_1) = f(x_2)$. Then:
$$\frac{1}{x_1 + 1} = \frac{1}{x_2 + 1}$$

This implies:
$$x_1 + 1 = x_2 + 1 \implies x_1 = x_2$$**Surjective:**
For any $y \in (0,1)$, we need to find an $ x \in (0,1)$such that $ f(x) = y$. Solving:
$$y = \frac{1}{x+1} \implies x + 1 = \frac{1}{y} \implies x = \frac{1}{y} - 1$$

Since $y \in (0,1)$,$\frac{1}{y} > 1 $, so $\frac{1}{y} - 1 > 0 $. Also, since $ y < 1 $,$\frac{1}{y} > 1 \implies x = \frac{1}{y} - 1 < 1$.

Thus, $x \in (0,1)$, and we have:
$$f\left(\frac{1}{y} - 1\right) = y$$

Therefore,$f(x)$ is both injective and surjective, making it a bijection.

```java
public class BijectionExample {
    public boolean isBijection(double x) {
        return (x > 0 && x < 1);
    }

    public double findInverse(double y) {
        return 1.0 / y - 1;
    }
}
```
x??

--- 

This covers the key concepts from the provided text with detailed explanations and examples. Each flashcard focuses on one specific question to ensure clarity and understanding. --- 

Note: The `??x` and `x??" format is used as placeholders for questions and answers, respectively, within the context of the flashcards. The code examples are provided to illustrate the concepts in a practical manner.

