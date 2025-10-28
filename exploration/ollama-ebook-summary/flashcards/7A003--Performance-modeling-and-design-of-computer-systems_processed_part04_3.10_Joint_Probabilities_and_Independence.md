# Flashcards: 7A003--Performance-modeling-and-design-of-computer-systems_processed (Part 4)

**Starting Chapter:** 3.10 Joint Probabilities and Independence

---

#### Bernoulli Distribution Variance
Background context: The variance of a Bernoulli random variable \(X\) with parameter \(p\) is given by \(\text{Var}(X) = p(1 - p)\). This formula arises because a Bernoulli random variable can take on only two values, 0 and 1.
:p What is the variance of a Bernoulli distributed random variable?
??x
The variance of a Bernoulli random variable \(X\) with parameter \(p\) is \(\text{Var}(X) = p(1 - p)\).

This comes from calculating the expected value squared minus the square of the expected value:
\[ E[X^2] = 0^2 \cdot (1 - p) + 1^2 \cdot p = p \]
And we know \(E[X] = p\), so
\[ \text{Var}(X) = E[X^2] - (E[X])^2 = p - p^2 = p(1 - p). \]

No code is needed for this concept.
x??

---

#### Uniform Distribution Variance
Background context: The variance of a continuous uniform random variable \(X\) over the interval \([a, b]\) is given by \(\text{Var}(X) = \frac{(b - a)^2}{12}\). This formula arises because the mean and variance for such distributions have simple closed-form expressions.
:p What is the variance of a uniform distribution?
??x
The variance of a continuous uniform random variable \(X\) over the interval \([a, b]\) is \(\text{Var}(X) = \frac{(b - a)^2}{12}\).

This formula can be derived by first finding the mean and then using it to find the variance. The mean (or expected value) of a uniform distribution on \([a, b]\) is:
\[ E[X] = \frac{a + b}{2} \]

Then, we calculate the second moment \(E[X^2]\):
\[ E[X^2] = \int_a^b x^2 \cdot \frac{1}{b - a} \, dx = \frac{1}{b - a} \left[ \frac{x^3}{3} \right]_a^b = \frac{b^3 - a^3}{3(b - a)} = \frac{(b - a)(b^2 + ab + a^2)}{3(b - a)} = \frac{b^2 + ab + a^2}{3}. \]

Finally, the variance is:
\[ \text{Var}(X) = E[X^2] - (E[X])^2 = \frac{b^2 + ab + a^2}{3} - \left(\frac{a + b}{2}\right)^2 = \frac{4(b^2 + ab + a^2) - 3(a + b)^2}{12} = \frac{4b^2 + 4ab + 4a^2 - 3a^2 - 6ab - 3b^2}{12} = \frac{(b^2 - 2ab + a^2)}{12} = \frac{(b - a)^2}{12}. \]

No code is needed for this concept.
x??

---

#### Joint Probability Mass Function
Background context: The joint probability mass function (PMF) \(p_{X,Y}(x, y)\) of two discrete random variables \(X\) and \(Y\) gives the probability that both \(X = x\) and \(Y = y\). For continuous random variables, the joint probability density function (PDF) \(f_{X,Y}(x, y)\) is similarly defined.
:p What does the joint probability mass function represent?
??x
The joint probability mass function \(p_{X,Y}(x, y)\) represents the probability that two discrete random variables \(X\) and \(Y\) take on specific values simultaneously. Specifically,
\[ P(X = x \text{ and } Y = y) = p_{X,Y}(x, y). \]

For continuous random variables, this concept is analogous with the joint probability density function:
\[ P(a < X < b \text{ and } c < Y < d) = \int_c^d \int_a^b f_{X,Y}(u, v) \, du \, dv. \]
x??

---

#### Independence of Random Variables
Background context: Two random variables \(X\) and \(Y\) are said to be independent if the occurrence of one does not affect the probability of the other. For discrete random variables, this means:
\[ p_{X,Y}(x, y) = p_X(x) \cdot p_Y(y). \]
For continuous random variables, independence is defined similarly using PDFs:
\[ f_{X,Y}(x, y) = f_X(x) \cdot f_Y(y), \text{ for all } x, y. \]

Independence allows us to simplify the calculation of joint probabilities and expectations.
:p How do we define independence between two random variables?
??x
Two random variables \(X\) and \(Y\) are said to be independent if:
- For discrete random variables: 
\[ p_{X,Y}(x, y) = p_X(x) \cdot p_Y(y). \]
- For continuous random variables:
\[ f_{X,Y}(x, y) = f_X(x) \cdot f_Y(y), \text{ for all } x, y. \]

This means that the joint distribution factors into the product of their marginal distributions.
x??

---

#### Expected Value and Independence
Background context: If two random variables \(X\) and \(Y\) are independent, then the expected value of their product is equal to the product of their individual expected values:
\[ E[XY] = E[X] \cdot E[Y]. \]
This property can be used in various probability calculations.
:p What theorem connects independence with the expected value of a product?
??x
Theorem 3.20 states that if two random variables \(X\) and \(Y\) are independent, then:
\[ E[XY] = E[X] \cdot E[Y]. \]

This is derived from the definition of independence in terms of joint probability mass or density functions.
:p Can you provide a proof for this theorem?
??x
Proof: For discrete random variables,
\[ E[XY] = \sum_x \sum_y xy \cdot P(X = x, Y = y). \]
Using the definition of independence \(P(X = x, Y = y) = P(X = x) \cdot P(Y = y)\),
\[ E[XY] = \sum_x \sum_y xy \cdot P(X = x) \cdot P(Y = y) = \left( \sum_x x \cdot P(X = x) \right) \left( \sum_y y \cdot P(Y = y) \right) = E[X] \cdot E[Y]. \]

For continuous random variables, the proof follows similarly by integrating over the joint density function:
\[ E[XY] = \int_{-\infty}^{\infty} \int_{-\infty}^{\infty} xy \cdot f_{X,Y}(x, y) \, dx \, dy. \]
Using \(f_{X,Y}(x, y) = f_X(x) \cdot f_Y(y)\),
\[ E[XY] = \left( \int_{-\infty}^{\infty} x \cdot f_X(x) \, dx \right) \left( \int_{-\infty}^{\infty} y \cdot f_Y(y) \, dy \right) = E[X] \cdot E[Y]. \]

No code is needed for this concept.
x??

#### Conditional Probabilities and Expectations for Discrete Random Variables

Background context: Conditional probabilities extend the concept of conditional events to random variables. This is particularly useful when we want to understand how the probability distribution of one random variable changes given some information about another.

Formula: The conditional probability mass function (p.m.f.) of a discrete r.v. \(X\) given an event \(A\) is defined as:
\[ p_X|A(x) = \frac{P(X=x, A)}{P(A)} = \frac{P((X=x) \cap A)}{P(A)} \]

:p What is the definition of the conditional probability mass function for a discrete random variable given an event?
??x
The conditional probability mass function \(p_X|A(x)\) is defined as the probability that the random variable \(X\) takes on the value \(x\), given that the event \(A\) has occurred. It can be expressed as:
\[ p_X|A(x) = \frac{P(X=x, A)}{P(A)} = \frac{P((X=x) \cap A)}{P(A)} \]
This essentially normalizes the joint probability of \(X=x\) and \(A\) by the probability of \(A\).

---

#### Conditional Expectation for Discrete Random Variables

Background context: The conditional expectation of a discrete random variable \(X\) given an event \(A\) is similar to the unconditional expectation but takes into account additional information.

Formula: For a discrete r.v. \(X\), the conditional expectation of \(X\) given event \(A\) is:
\[ E[X|A] = \sum_x x \cdot p_X|A(x) = \sum_x x \cdot P(X=x, A) / P(A) \]

:p How do you compute the conditional expectation for a discrete random variable?
??x
To compute the conditional expectation \(E[X|A]\) for a discrete random variable \(X\) given an event \(A\), we use the formula:
\[ E[X|A] = \sum_x x \cdot p_X|A(x) = \sum_x x \cdot P(X=x, A) / P(A) \]
This means that you sum up the product of each value \(x\) and its conditional probability given \(A\).

---

#### Example: Hair Color

Background context: This example uses a discrete random variable to categorize people based on their hair color. The goal is to find probabilities and expectations related to hair colors.

:p What are the steps to compute the conditional expectation of a discrete r.v. in this hair color example?
??x
To compute the conditional expectation of a discrete r.v. \(X\) (hair color) given event \(A\) (light-colored or dark-colored hair), follow these steps:
1. Define the events and their probabilities.
2. Calculate the conditional probability mass function \(p_X|A(x)\).
3. Use the formula for conditional expectation:
\[ E[X|A] = \sum_x x \cdot p_X|A(x) \]

For instance, if we define "light" as Blonde or Red-haired (values 1 and 2), and "dark" as Brown or Black-haired (values 3 and 4):
- \(p_{X}(1) = P\{Blonde\} = 5/38\)
- \(p_{X}(2) = P\{Red\} = 2/38\)
- \(p_{X}(3) = P\{Brown\} = 17/38\)
- \(p_{X}(4) = P\{Black\} = 14/38\)

The conditional expectation when a person has light-colored hair is:
\[ E[X|A] = 1 \cdot p_X|A(1) + 2 \cdot p_X|A(2) \]

---

#### Continuous Random Variables and Conditional Expectation

Background context: For continuous r.v.'s, the concept of conditional expectation uses probability density functions (p.d.f.) to describe how the distribution changes given additional information.

Formula: The conditional p.d.f. of a continuous r.v. \(X\) given an event \(A\), where \(A\) is a subset of real numbers with positive probability, is defined as:
\[ f_{X|A}(x) = \frac{f_X(x)}{P(X \in A)} \text{ if } x \in A; 0 \text{ otherwise} \]

The conditional expectation of a continuous r.v. \(X\) given an event \(A\) is:
\[ E[X|A] = \int_{-\infty}^{\infty} x f_{X|A}(x) dx = \frac{\int_A x f_X(x) dx}{P(X \in A)} \]

:p How do you compute the conditional expectation for a continuous random variable?
??x
To compute the conditional expectation \(E[X|A]\) for a continuous r.v. \(X\) given an event \(A\), use the following steps:
1. Define the subset of real numbers \(A\) with positive probability.
2. Compute the p.d.f. of \(X\) given \(A\):
\[ f_{X|A}(x) = \frac{f_X(x)}{P(X \in A)} \text{ if } x \in A; 0 \text{ otherwise} \]
3. Integrate the product of \(x\) and the conditional p.d.f. over the subset \(A\):
\[ E[X|A] = \int_{-\infty}^{\infty} x f_{X|A}(x) dx = \frac{\int_A x f_X(x) dx}{P(X \in A)} \]

---

#### Example: Pittsburgh Supercomputing Center (Continuous Case)

Background context: This example demonstrates how to compute the conditional expectation in a real-world scenario using continuous random variables and exponential distributions.

:p How would you calculate the expected job duration given that it is sent to bin 1?
??x
To calculate the expected job duration given that it is sent to bin 1, follow these steps:
1. Define the event \(A\) as the condition that the job is sent to bin 1.
2. Use the conditional p.d.f. of the duration \(X\) given \(A\):
\[ f_{X|Y}(t) = \frac{f_X(t)}{P(Y=1)} = \frac{\lambda e^{-\lambda t}}{P(Y=1)} \text{ if } t < 500; 0 \text{ otherwise} \]
3. Integrate the product of \(t\) and the conditional p.d.f. over the range of \(A\):
\[ E[X|Y=1] = \int_{-\infty}^{\infty} t f_{X|Y}(t) dt = \int_0^{500} t \cdot \frac{\lambda e^{-\lambda t}}{P(Y=1)} dt \]

For an Exponential distribution with mean 1000 hours:
\[ E[X|Y=1] = \frac{\int_0^{500} t \lambda e^{-\lambda t} dt}{P(Y=1)} \]
Given \(P(Y=1) \approx 0.39\) and \(\lambda = \frac{1}{1000}\):
\[ E[X|Y=1] \approx \frac{\int_0^{500} t e^{-t/1000} dt}{0.39} \approx 229 \]

---

#### Example: Uniform Distribution

Background context: This example shows how the expected job duration changes when the distribution is uniform instead of exponential.

:p How does the answer change if the job durations are uniformly distributed between 0 and 2000 hours?
??x
If the job durations are uniformly distributed between 0 and 2000 hours, given that the job is in bin 1 (i.e., less than 500 hours), the expected duration is:
\[ E[X|Y=1] = \frac{500}{2} = 250 \]

This result makes sense because the uniform distribution has a linear density function, and the midpoint of the range [0, 500) gives the expected value.

---

#### Expected Value of Exponential Distribution

Background context: The example demonstrates why the expected size of jobs in bin 1 is less than the midpoint of its range when the underlying distribution is exponential.

:p Why is the expected job duration in bin 1 less than 250 hours?
??x
The expected value for an exponentially distributed random variable \(X\) with mean \(\mu = 1000\) hours is:
\[ E[X] = \mu = 1000 \]

When we consider only the values between 0 and 500, the distribution of \(X|Y=1\) (where \(Y=1\) means the job duration is less than 500) has a truncated exponential distribution. The expected value for such a truncated distribution is:
\[ E[X|Y=1] = \int_0^{500} x f_{X|Y}(x) dx / P(Y=1) \]
Since the exponential distribution gives more weight to smaller values, the expected value of the truncated distribution will be less than 250.

---

#### Comparison of Distributions

Background context: This example highlights how different distributions can lead to different expected values even if they have the same mean.

:p How would the answer change for a uniform distribution with the same mean as an exponential distribution?
??x
For a uniform distribution between 0 and 2000 hours, given that the job duration is less than 500, the expected value is:
\[ E[X|Y=1] = \frac{500}{2} = 250 \]

This result shows that even though both distributions have the same mean (2500/2 = 1000), the shape of the distribution affects the conditional expectation. The uniform distribution, being symmetric and linear, gives a straightforward midpoint as the expected value.

--- 
These flashcards cover key concepts in conditional probabilities and expectations for discrete and continuous random variables. Each card provides a clear explanation and context to aid understanding.

#### Law of Total Probability for Discrete Random Variables
Background context: The Law of Total Probability is a fundamental concept that extends to random variables. It states that the probability of an event E can be computed by conditioning on all possible outcomes of another event F1, ..., F_n that partition the sample space Ω. For discrete random variables, it's expressed as:
\[ P\{X = k\} = \sum_{y} P\{X = k | Y = y\}P\{Y = y\} \]

:p How is the Law of Total Probability applied to discrete random variables?
??x
The law allows us to break down the probability calculation by conditioning on all possible values of another random variable \( Y \). This can simplify complex probability calculations into a series of simpler problems.
```java
public class Example {
    // Assuming we have two discrete random variables X and Y, where Y's value is known
    public double lawOfTotalProbability(double k) {
        double totalProb = 0;
        for (double y : possibleValuesY) { // iterate over all possible values of Y
            totalProb += conditionalProbability(k | y) * probabilityOfY(y);
        }
        return totalProb;
    }

    private double conditionalProbability(int k, double y) {
        // Logic to calculate P(X = k | Y = y)
        return /* some value */;
    }

    private double probabilityOfY(double y) {
        // Logic to calculate P(Y = y)
        return /* some value */;
    }
}
```
x??

---

#### Probability of One Exponential Random Variable Happening Before Another
Background context: We need to derive \( P\{X_1 < X_2\} \) for two independent exponential random variables \( X_1 \sim \text{Exp}(\lambda_1) \) and \( X_2 \sim \text{Exp}(\lambda_2) \).

:p How can we use conditioning to find the probability that one exponential random variable happens before another?
??x
By conditioning on the value of \( X_2 \), we can break down the problem into simpler sub-problems. The key step is recognizing that if \( X_1 < X_2 \) given \( X_2 = k \), it occurs with probability \( 1 - e^{-\lambda_1 k} \).

```java
public class ExponentialComparison {
    public double probabilityX1BeforeX2() {
        double lambda1 = /* some value */;
        double lambda2 = /* some value */;
        double totalProbability = 0;
        
        // Integrate over all possible values of X2 (from 0 to infinity)
        for (double k = 0; k <= 1000; k += 0.01) { // Use a loop or integral
            totalProbability += (1 - Math.exp(-lambda1 * k)) * lambda2 * Math.exp(-lambda2 * k);
        }
        
        return totalProbability;
    }
}
```
x??

---

#### Linearity of Expectation for Discrete Random Variables
Background context: The theorem states that the expected value of a sum of random variables is equal to the sum of their individual expected values. This holds true even if the random variables are not independent.

:p What does the Linearity of Expectation state and how can it simplify proofs?
??x
The theorem states that for any two random variables \( X \) and \( Y \), \( E[X + Y] = E[X] + E[Y] \). This holds true without needing independence, which makes it a powerful tool in simplifying many probability-related derivations.

```java
public class LinearityOfExpectation {
    public double expectedValueSum(int n) {
        double expectedValue = 0;
        
        // Summing the expected values of individual variables
        for (int i = 1; i <= n; i++) {
            expectedValue += i * binomialCoefficient(n, i) * Math.pow(0.5, i) * Math.pow(0.5, n - i);
        }
        
        // Simplified using linearity of expectation: E[X] = n * p
        return n * 0.5;
    }
    
    private int binomialCoefficient(int n, int k) {
        // Calculate the binomial coefficient
        return /* some value */;
    }
}
```
x??

---

#### Binomial Distribution as a Sum of Indicator Random Variables
Background context: The Binomial distribution can be seen as the sum of indicator random variables. Each trial results in either success (1) or failure (0).

:p How does Linearity of Expectation simplify the calculation of \( E[X] \) for a binomially distributed variable?
??x
Using linearity of expectation, we can express \( X \), the number of successes in \( n \) trials, as the sum of indicator random variables. Since each trial is independent and has an expected value of \( p \), the overall expectation is simply \( np \).

```java
public class BinomialExpectation {
    public double binomialExpectedValue(int n, double p) {
        // Using linearity of expectation directly
        return n * p;
    }
}
```
x??

---

#### Linearity of Expectation for Variance Calculation
Background context: While linearity of expectation is powerful, it does not always hold for variance. However, under certain conditions, we can still use it effectively.

:p How does linearity of expectation relate to the calculation of \( E[X^2 + Y^2] \)?
??x
Linearity of expectation states that \( E[X^2 + Y^2] = E[X^2] + E[Y^2] \). This is true regardless of whether X and Y are independent or not. However, it does not imply that the variance can be calculated in a similar manner without additional conditions.

```java
public class VarianceExample {
    public double expectedValueOfSquares(int n) {
        // Summing the squares' expectations
        return n * 1 + (n - 1) * 0; // Simplified for demonstration, actual logic may vary
    }
}
```
x??

---

#### Linearity of Expectation and Indicator Random Variables in Permutations
Background context: In problems involving permutations or allocations, indicator random variables can be used to represent events. The linearity of expectation simplifies the calculation of expected values.

:p How does linearity of expectation help in calculating the number of people who get their own hat back?
??x
By using indicator random variables \( I_i \), where \( I_i = 1 \) if the \( i \)-th person gets their own hat and \( 0 \) otherwise, we can leverage linearity of expectation. The expected value is simply the sum of individual expectations.

```java
public class HatExample {
    public double expectedValueOfPeopleWithTheirOwnHat(int n) {
        // Summing over all people's indicators
        return n * (1 / n + 0); // Simplified for demonstration, actual logic may vary
    }
}
```
x??

---

