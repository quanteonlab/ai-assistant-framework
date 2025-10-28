# Flashcards: 7A003--Performance-modeling-and-design-of-computer-systems_processed (Part 5)

**Starting Chapter:** 3.14 Normal Distribution

---

#### Var(X+Y) for Independent Random Variables

Background context: This concept deals with finding the variance of the sum of two independent random variables \(X\) and \(Y\). The formula given is a derivation of the properties of variance, specifically when \(X \perp Y\).

:p What is the formula used to derive the variance of the sum of two independent random variables?
??x
The formula used to derive the variance of the sum of two independent random variables \(X\) and \(Y\) is:
\[ \text{Var}(X+Y) = E\left[(X+Y)^2\right] - (E[X+Y])^2 = E\left[X^2\right] + E\left[Y^2\right] + 2E[XY] - (E[X])^2 - (E[Y])^2 - 2E[X]E[Y] \]

This simplifies to:
\[ \text{Var}(X+Y) = \text{Var}(X) + \text{Var}(Y) + 2\text{Cov}(X, Y) \]
If \(X\) and \(Y\) are independent, then \(\text{Cov}(X, Y) = 0\), hence:
\[ \text{Var}(X+Y) = \text{Var}(X) + \text{Var}(Y) \]

The derivation shows that the covariance term is zero when the random variables are independent.
x??

---

#### Definition of Normal Distribution

Background context: The Normal distribution, also known as the Gaussian distribution, is a continuous probability distribution characterized by its bell-shaped curve. It is defined by two parameters: the mean \(\mu\) and the variance \(\sigma^2\).

:p What does the Probability Density Function (PDF) of a Normal distribution look like?
??x
The PDF of a Normal distribution with mean \(\mu\) and variance \(\sigma^2\) is given by:
\[ f_X(x) = \frac{1}{\sqrt{2\pi}\sigma} e^{-\frac{(x-\mu)^2}{2\sigma^2}} \]

This formula describes the probability density at each point \(x\) in the domain of the distribution.
x??

---

#### Linear Transformation Property of Normal Distribution

Background context: The Linear Transformation Property states that if a random variable \(X\) follows a normal distribution, then any linear transformation of \(X\), such as \(Y = aX + b\) (where \(a > 0\)), will also follow a normal distribution.

:p How is the mean and variance of the transformed random variable \(Y\) calculated?
??x
The mean and variance of the transformed random variable \(Y = aX + b\) are:
- Mean: \(E[Y] = E[aX + b] = aE[X] + b = a\mu + b\)
- Variance: \(\text{Var}(Y) = \text{Var}(aX + b) = a^2 \text{Var}(X) = a^2\sigma^2\)

Thus, if \(X\) is normally distributed with mean \(\mu\) and variance \(\sigma^2\), then \(Y\) will be normally distributed with mean \(a\mu + b\) and variance \(a^2\sigma^2\).
x??

---

#### Central Limit Theorem (CLT)

Background context: The Central Limit Theorem states that the sum of a large number of independent and identically distributed random variables, each with finite mean \(\mu\) and variance \(\sigma^2\), will be approximately normally distributed. This theorem is crucial because it allows us to approximate the distribution of sums even when the underlying distributions are not normal.

:p What is the formula for \(Z_n\) in the context of the Central Limit Theorem?
??x
In the context of the Central Limit Theorem, \(Z_n\) is defined as:
\[ Z_n = \frac{S_n - n\mu}{\sigma \sqrt{n}} \]
where \(S_n = X_1 + X_2 + \cdots + X_n\) and \(X_i\) are i.i.d. random variables with mean \(\mu\) and variance \(\sigma^2\).

The formula for \(Z_n\) standardizes the sum of the random variables, making it approximately normal as \(n\) becomes large.
x??

---

#### Standardizing a Normal Distribution

Background context: To find probabilities related to non-standard Normal distributions, we can use the Linear Transformation Property to convert them into standard Normal distributions.

:p How is the probability that a Normally distributed random variable \(X\) with mean \(\mu\) and variance \(\sigma^2\) is less than \(k\), denoted as \(P(X < k)\), calculated?
??x
To find the probability that a Normally distributed random variable \(X\) with mean \(\mu\) and variance \(\sigma^2\) is less than \(k\), we can use the standard Normal distribution. The transformation used is:
\[ P(X < k) = P\left( \frac{X - \mu}{\sigma} < \frac{k - \mu}{\sigma} \right) = \Phi\left( \frac{k - \mu}{\sigma} \right) \]
where \(\Phi(z)\) is the cumulative distribution function of the standard Normal distribution.

This transformation standardizes \(X\) to a standard Normal variable, making it easier to look up probabilities in standard normal tables.
x??

---

#### IQ Testing and Normal Distribution

Background context: The concept of IQ testing often involves using the Normal distribution with mean 100 and standard deviation 15. This example demonstrates how to calculate the probability that an individual's IQ score falls within a certain range.

:p What is the fraction of people having an IQ greater than 130?
??x
To find the fraction of people having an IQ greater than 130, we first determine how many standard deviations 130 is from the mean (100):
\[ \frac{130 - 100}{15} = 2 \]

Thus, we need to calculate \(P(X > 130) = P(Z > 2)\), where \(Z\) is a standard Normal variable. Using the properties of the normal distribution:
\[ P(Z > 2) = 1 - \Phi(2) \]
From standard normal tables or using a calculator, we find that \(\Phi(2) \approx 0.9772\). Therefore:
\[ P(Z > 2) \approx 1 - 0.9772 = 0.0228 \]

So, only about 2% of people have an IQ above 130.
x??

---

#### Central Limit Theorem in Practice

Background context: The Central Limit Theorem is a fundamental concept that explains why the sum or average of a large number of independent and identically distributed random variables tends to be normally distributed.

:p What does the CLT say about the distribution of \(Z_n\) as \(n \to \infty\)?
??x
The Central Limit Theorem states that for a sequence of i.i.d. random variables \(X_1, X_2, \ldots, X_n\) with common mean \(\mu\) and variance \(\sigma^2\), the standardized sum:
\[ Z_n = \frac{S_n - n\mu}{\sigma \sqrt{n}} \]
where \(S_n = X_1 + X_2 + \cdots + X_n\), converges in distribution to a standard normal random variable as \(n\) approaches infinity. Formally, this means:
\[ \lim_{n \to \infty} P(Z_n \leq z) = \Phi(z) \]
where \(\Phi(z)\) is the cumulative distribution function of the standard Normal distribution.

This theorem provides a powerful tool for approximating the distribution of sums or averages of large numbers of random variables, even when their individual distributions are not normal.
x??

--- 
--- 
Note: Each flashcard covers one specific concept from the provided text, with clear explanations and relevant formulas. The answers include detailed context to ensure understanding beyond pure memorization. Code examples have been used where applicable to illustrate logical processes.

#### Linear Transformation Property of Normal Distributions
Background context: The provided text discusses the application of the linear transformation property to normal distributions. This property is crucial for understanding how scaling and shifting affect a normally distributed random variable.

:p What is the distribution of \( S_{100} \) given that each noise source produces an amount uniformly distributed between -1 and 1?
??x
The distribution of \( S_{100} \), which is the sum of 100 independent and identically distributed (i.i.d.) random variables, each with a uniform distribution on \([-1, 1]\), can be transformed into a normal distribution using the Linear Transformation Property. Specifically:

- Each \( X_i \) has mean \( \mu_{X_i} = 0 \).
- The variance of \( X_i \) is given by:
\[ \sigma^2_{X_i} = \frac{(b-a)^2}{12} = \frac{(-1 - 1)^2}{12} = \frac{4}{12} = \frac{1}{3} \]
- Therefore, the standard deviation \( \sigma_{X_i} = \sqrt{\frac{1}{3}} = \frac{1}{\sqrt{3}} \).

By the Linear Transformation Property:
\[ S_{100} \sim N(100 \cdot 0, 100 \cdot \frac{1}{3}) = N(0, \frac{100}{3}) \]

To find \( P(|S_{100}| < 10) \):
\[ P\left(\left| S_{100} - 0 \right| < 10 \sqrt{\frac{100}{3}} \right) = P\left( -10 \sqrt{\frac{100}{3}} < S_{100} - 0 < 10 \sqrt{\frac{100}{3}} \right) \approx 2 \Phi\left(\frac{10}{\sqrt{\frac{100}{3}}}\right) - 1 = 2 \Phi(3.464) - 1 \]

Using the standard normal distribution table or a calculator, \( \Phi(3.464) \approx 0.9997 \):
\[ P(|S_{100}| < 10) \approx 2 (0.9997) - 1 = 0.9994 - 1 = 0.9894 \]

Thus, the approximate probability that the absolute value of the total amount of noise from the 100 signals is less than 10 is about 99%.

x??

---

#### Conditional Expectation for Sum of Random Variables
Background context: The text discusses how to derive quantities such as \( E[S] \) and \( E[S^2] \) when summing a random number of i.i.d. random variables, where the number of these variables is itself a random variable.

:p Can we directly apply linearity of expectation in this scenario?
??x
No, we cannot directly apply the linearity of expectation here because \( N \), which represents the number of i.i.d. random variables being summed, is not a constant but a random variable itself. The standard linearity of expectation applies only when all involved variables are constants or if they are independent from each other.

To handle this situation, we condition on the value of \( N \) and then apply the linearity of expectation:

\[ E[S] = E\left[\sum_{i=1}^{N} X_i\right] = \sum_{n} E\left[\sum_{i=1}^n X_i | N=n\right] P(N=n) \]
\[ = \sum_{n} n E[X_1] P(N=n) = E[N] E[X_1] \]

The key idea is to break down the expectation based on different possible values of \( N \).

x??

---

#### Variance Calculation for Sum of Random Variables
Background context: The text explains how to derive the second moment \( E[S^2] \) using conditional expectations and variance properties.

:p Can we use the same trick to get \( E[S^2] \)?
??x
No, directly conditioning on \( N \) would lead to a complicated sum that needs to be squared, which is not straightforward. Instead, it's better to first derive the conditional variance \( Var(S|N=n) \).

From Theorem 3.27:
\[ Var(S|N=n) = nVar(X_1) \]

And since:
\[ Var(S|N=n) = E\left[S^2 | N=n\right] - (E[S|N=n])^2 \]
\[ = E\left[S^2 | N=n\right] - (nE[X_1])^2 \]

Thus:
\[ E\left[S^2 | N=n\right] = nVar(X_1) + (nE[X_1])^2 \]

To find \( E[S^2] \):
\[ E[S^2] = \sum_{n} E\left[S^2 | N=n\right] P(N=n) \]
\[ = \sum_{n} \left(nVar(X_1) + (nE[X_1])^2\right) P(N=n) \]

This provides a structured way to calculate the second moment.

x??

---

#### Poisson Distribution and Normal Approximation
Background context: The text mentions that the Poisson distribution can be approximated by a normal distribution when \( n \) (the parameter of the Poisson distribution) is high. This approximation helps in simplifying complex sums involving many independent events.

:p Why does the Poisson(λ) distribution approximate a Normal distribution?
??x
The Poisson distribution with parameter \( \lambda \), denoted as \( \text{Poisson}(\lambda) \), can be approximated by a normal distribution when \( \lambda \) is large. This approximation simplifies the analysis of sums involving many independent events.

For a Poisson-distributed random variable \( X \):
\[ P(X = k) = \frac{\lambda^k e^{-\lambda}}{k!} \]

When \( \lambda \) is large, the Central Limit Theorem (CLT) can be applied to show that:
\[ S_n = X_1 + X_2 + \cdots + X_n \]
where each \( X_i \sim \text{Poisson}(1) \), approximates a normal distribution with mean \( n\lambda \) and variance \( n\lambda \).

Thus, for large \( \lambda \):
\[ S_n \approx N(n\lambda, n\lambda) \]

This approximation is particularly useful in practical applications where exact Poisson calculations are complex.

x??

---

#### Sum of a Random Number of Random Variables
Background context: The text introduces the concept of summing a random number of i.i.d. random variables, where the number \( N \) of these variables themselves follows some distribution. This scenario is common in various applications like signal processing and queuing theory.

:p How do we derive \( E[S] \) when \( S = \sum_{i=1}^{N} X_i \)?
??x
To derive the expected value \( E[S] \), where \( S = \sum_{i=1}^{N} X_i \) and \( N \) is a non-negative integer-valued random variable, we use conditional expectations:

\[ E[S] = E\left[\sum_{i=1}^{N} X_i \right] = \sum_{n} E\left[\sum_{i=1}^n X_i | N=n\right] P(N=n) \]

Given that \( X_1, X_2, \ldots \) are i.i.d. and have mean \( E[X_i] \), we can simplify:
\[ E[S] = \sum_{n} n E[X_i] P(N=n) = E[N] E[X_i] \]

This result comes from breaking down the expectation based on different possible values of \( N \).

x??

---

#### Expectation Brainteaser
Background context: The scenario involves understanding how two different measures of central tendency (mean and minimum value) can coexist. Here, we are given that a friend claims he was never in a class with less than 90 students during his first year in school, yet the dean states that the mean freshman class size is 30.

:p How can it be possible for both statements to be true?
??x
The key here is understanding the difference between the minimum value and the mean. The fact that no class had less than 90 students means the smallest class size was always 90, but this does not preclude the possibility of having many smaller classes overall, leading to a lower average.

For example, consider a scenario with three classes:
- Class A: 80 students
- Class B: 90 students
- Class C: 90 students

The mean class size is \((80 + 90 + 90) / 3 = 86.7\) which rounds to approximately 30, while the smallest class size was still 90.

```java
// Java code for calculating mean and minimum class sizes
public class ClassSizes {
    public static void main(String[] args) {
        int[] classes = {80, 90, 90};
        double mean = calculateMean(classes);
        System.out.println("Mean class size: " + mean); // Should be close to 30
        int min = findMin(classes);
        System.out.println("Minimum class size: " + min); // 80
    }

    public static double calculateMean(int[] classes) {
        int sum = 0;
        for (int cls : classes) {
            sum += cls;
        }
        return (double) sum / classes.length;
    }

    public static int findMin(int[] classes) {
        int min = Integer.MAX_VALUE;
        for (int cls : classes) {
            if (cls < min) {
                min = cls;
            }
        }
        return min;
    }
}
```
x??

---

#### Probability of Getting a Girlfriend
Background context: This problem involves understanding the probability distribution and using it to calculate the likelihood of an event occurring over multiple trials. Here, Ned asks out a new girl every day with two possible outcomes (yes or no), each with different probabilities.

:p What is the probability that it takes more than 100 days for Ned to get a girlfriend?
??x
We need to find the probability that Ned does not get a girlfriend in the first 100 days. The probability of getting rejected on any given day is \(\frac{99}{100}\), so the probability of being rejected for 100 consecutive days is \(\left(\frac{99}{100}\right)^{100}\).

The answer can be calculated as:
\[
P(\text{more than 100 days}) = \left(\frac{99}{100}\right)^{100} \approx 0.366
\]

```java
// Java code for calculating the probability
public class Girlfriends {
    public static void main(String[] args) {
        double probRejected = 99.0 / 100;
        int days = 100;
        double probMoreThan100Days = Math.pow(probRejected, days);
        System.out.println("Probability of more than 100 days: " + probMoreThan100Days);
    }
}
```
x??

---

#### Variance Proof
Background context: This problem involves using the linearity of expectation to prove a key property of variance. The goal is to show that \(Var(X) = E[X^2] - (E[X])^2\).

:p Prove that Var(X) = E[X^2] − E[X]^2.
??x
To prove this, we start with the definition of variance:
\[ \text{Var}(X) = E[(X - E[X])^2] \]

Expanding the square inside the expectation:
\[
E[(X - E[X])^2] = E[X^2 - 2XE[X] + (E[X])^2]
\]

Using linearity of expectation, we get:
\[
E[X^2 - 2XE[X] + (E[X])^2] = E[X^2] - 2E[XE[X]] + E[(E[X])^2]
\]

Since \(E[E[X]] = E[X]\) and \((E[X])^2\) is a constant, we have:
\[
= E[X^2] - 2(E[X])^2 + (E[X])^2
\]

Simplifying this expression gives us the desired result:
\[
= E[X^2] - (E[X])^2
\]

Thus, we prove that \(\text{Var}(X) = E[X^2] - (E[X])^2\).

```java
// Java code for demonstrating variance calculation
public class VarianceCalculation {
    public static void main(String[] args) {
        double mean = 5.0;
        double x1 = 3.0, x2 = 7.0; // Example values
        double meanSquared = mean * mean;
        double xsquaredSum = (x1 * x1 + x2 * x2);
        
        double variance = (xsquaredSum / 2) - meanSquared;
        System.out.println("Variance: " + variance);
    }
}
```
x??

---

#### Chain Rule for Conditioning
Background context: The chain rule for conditioning is a fundamental concept in probability theory that allows us to compute the joint probability of multiple events. It states that:
\[ P\left(\bigcap_{i=1}^n E_i \right) = P(E_1) \cdot P(E_2|E_1) \cdot P(E_3|E_1 \cap E_2) \cdots P\left( E_n | \bigcap_{i=1}^{n-1} E_i \right) \]

:p Prove the chain rule for conditioning.
??x
The proof of the chain rule for conditioning is based on the definition of conditional probability. We start with the basic definition and extend it to multiple events.

Given any sequence of events \(E_1, E_2, ..., E_n\), we want to show:
\[ P(E_1 \cap E_2 \cap ... \cap E_n) = P(E_1) \cdot P(E_2|E_1) \cdot P(E_3|E_1 \cap E_2) \cdots P(E_n|\bigcap_{i=1}^{n-1} E_i) \]

Using the definition of conditional probability, we know:
\[ P(A \cap B) = P(A) \cdot P(B|A) \]

We can apply this recursively to multiple events. For three events \(E_1, E_2, E_3\):
\[ P(E_1 \cap E_2 \cap E_3) = P(E_1) \cdot P(E_2|E_1) \cdot P(E_3|E_1 \cap E_2) \]

This can be extended to any number of events by induction. For \(n\) events, the statement holds because:
\[ P\left(\bigcap_{i=1}^n E_i \right) = P\left(E_1 \cap E_2 \cap ... \cap E_n \right) = P(E_1) \cdot P(E_2|E_1) \cdot P(E_3|E_1 \cap E_2) \cdots P(E_n|\bigcap_{i=1}^{n-1} E_i) \]

```java
// Java code for demonstrating conditional probability
public class ConditionalProbability {
    public static void main(String[] args) {
        double pE1 = 0.5; // Probability of event E1
        double pE2_given_E1 = 0.4; // Probability of E2 given E1
        double pE3_given_E1_and_E2 = 0.3; // Probability of E3 given both E1 and E2

        // Calculate joint probability using chain rule
        double jointProbability = pE1 * pE2_given_E1 * pE3_given_E1_and_E2;
        System.out.println("Joint probability: " + jointProbability);
    }
}
```
x??

---

#### Assessing Risk in Queueville Airlines
Background context: This problem involves understanding the concept of risk and how to model it. The airline sells 52 tickets for a flight that can hold only 50 passengers, assuming each person independently does not show up with a probability of 5%. We need to calculate the probability that there will be enough seats for everyone who shows up.

:p Calculate the probability that Queueville Airlines has enough seats on a given day.
??x
To solve this problem, we can model it using a binomial distribution. Let \(N\) represent the number of people who show up, where each person has an independent probability of 5% (0.05) to not show up.

The expected value and variance for \(N\) are:
\[ E[N] = 52 \times (1 - 0.05) = 49.4 \]
\[ Var(N) = 52 \times 0.05 \times 0.95 = 2.476 \]

We want to find the probability that \(N\) is less than or equal to 50:
\[ P(N \leq 50) \]

Using a normal approximation to the binomial distribution (since \(n = 52\) and \(p = 0.05\)), we can approximate this with a standard normal distribution.

```java
// Java code for approximating probability using normal distribution
public class QueuevilleAirlines {
    public static void main(String[] args) {
        double mean = 52 * (1 - 0.05); // Expected number of people showing up
        double stdDev = Math.sqrt(52 * 0.05 * 0.95); // Standard deviation

        double zScore = (50 - mean) / stdDev;
        
        // Calculate the probability using normal distribution
        System.out.println("Probability of enough seats: " + calculateNormalProbability(zScore));
    }

    public static double calculateNormalProbability(double zScore) {
        return 1.0 - Math.erf(-zScore / Math.sqrt(2)) / 2;
    }
}
```
x??

---

#### Practice with Conditional Expectation
Background context: This problem involves computing the conditional expectation \(E[X|Y \neq 1]\). The joint probability mass function (pmf) of \(X\) and \(Y\) is provided in Table 3.3, which we need to use to compute the required expectation.

:p Compute \(E[X|Y \neq 1]\).
??x
Given a joint pmf table for \(X\) and \(Y\), we can compute \(E[X | Y \neq 1]\) by summing over all possible values of \(X\) given that \(Y \neq 1\).

For example, if the joint pmf is as follows:

| X | Y=0 | Y=1 | Y=2 |
|---|-----|-----|-----|
| 0 | 0.1 | 0.2 | 0.3 |
| 1 | 0.4 | 0.5 | 0.6 |

We need to calculate:
\[ E[X | Y \neq 1] = \frac{\sum_{x} x P(X=x, Y=1) + \sum_{x} x P(X=x, Y=2)}{P(Y \neq 1)} \]

Where \(P(Y \neq 1)\) is the total probability of all events where \(Y\) is not 1.

```java
// Java code for computing conditional expectation
public class ConditionalExpectation {
    public static void main(String[] args) {
        double[][] jointPmf = {{0.1, 0.2, 0.3}, {0.4, 0.5, 0.6}};
        
        // Summing over X for Y=1 and Y=2
        double numerator = 0;
        for (int x = 0; x < jointPmf.length; x++) {
            numerator += x * (jointPmf[x][0] + jointPmf[x][2]);
        }

        // Total probability of Y not being 1
        double denominator = 1 - jointPmf[0][1]; // P(Y=1) is 0.2, so 1 - 0.2 = 0.8

        // Conditional expectation
        double conditionalExpectation = numerator / denominator;
        System.out.println("Conditional Expectation: " + conditionalExpectation);
    }
}
```
x??

---

#### Variance Scaling with Coefficient of Variation
Background context: This problem involves understanding how variance scales when a random variable is scaled by a constant. The concept of the squared coefficient of variation (SCV) is introduced, which measures the ratio of the standard deviation to the mean.

:p Explain why \(\text{Var}(cX) = c^2 \cdot \text{Var}(X)\).
??x
The variance of a scaled random variable \(cX\) can be derived from the definition of variance. The variance of a random variable \(X\) is defined as:
\[ \text{Var}(X) = E[(X - E[X])^2] \]

For a scaled random variable \(cX\), we need to find its variance:
\[ \text{Var}(cX) = E[(cX - E[cX])^2] \]

First, note that the expected value of \(cX\) is:
\[ E[cX] = cE[X] \]

Thus, the expression inside the expectation becomes:
\[ (cX - E[cX]) = c(X - E[X]) \]

So,
\[ \text{Var}(cX) = E[(cX - E[cX])^2] = E[(c(X - E[X]))^2] = E[c^2 (X - E[X])^2] = c^2 E[(X - E[X])^2] = c^2 \cdot \text{Var}(X) \]

Hence, the variance of a scaled random variable \(cX\) is:
\[ \text{Var}(cX) = c^2 \cdot \text{Var}(X) \]

```java
// Java code for demonstrating scaling of variance
public class VarianceScaling {
    public static void main(String[] args) {
        double mean = 5.0; // Mean of X
        double variance = 4.0; // Variance of X
        double c = 3.0; // Scaling factor

        // Calculating the scaled variance
        double scaledVariance = c * c * variance;
        System.out.println("Scaled Variance: " + scaledVariance);
    }
}
```
x??

#### Variance of Sum vs. Scalar Multiple

Background context: In probability and statistics, understanding how variance behaves with sums of random variables versus scalar multiples is crucial.

Given \(c\) independent instances of a random variable (r.v.) \(X\), denoted as \(X_1, X_2, \ldots, X_c\):

- **Variance of Sum**: The variance of the sum of these r.v.s is given by:
  \[ Var(X_1 + X_2 + \cdots + X_c) = Var(X_1) + Var(X_2) + \cdots + Var(X_c) \]
  Since \(X_i\) are independent, their variances add up.

- **Variance of Scalar Multiple**: The variance of a scalar multiple of an r.v. is given by:
  \[ Var(cX) = c^2 Var(X) \]

:p Which has lower variance: \(Var(X_1 + X_2 + \cdots + X_c)\) or \(Var(cX)\)?
??x
To determine which has the lower variance, we need to compare the expressions for both cases:

- For the sum of random variables:
  \[ Var(X_1 + X_2 + \cdots + X_c) = Var(X) + Var(X) + \cdots + Var(X) = c \cdot Var(X) \]
  
- For a scalar multiple:
  \[ Var(cX) = c^2 \cdot Var(X) \]

Since \(c > 1\), it follows that \(c < c^2\) (for positive values of \(c\)). Therefore, the variance of the sum is lower than the variance of the scalar multiple.
x??

---

#### Mutual Fund Risk vs. Single Stock

Background context: The concept of diversification in finance suggests that mutual funds are less risky compared to investing in a single stock because they spread risk across many different stocks.

:p Explain why buying a mutual fund is considered less risky than buying a single stock.
??x
By spreading investments across multiple stocks, mutual funds reduce the impact of any individual stock's poor performance. The overall portfolio volatility is lower due to diversification. This reduces the overall risk for the investor compared to holding just one stock.

In mathematical terms, the variance of the sum of independent random variables (stocks) is less than the variance of a single highly volatile asset.
x??

---

#### Expectation and Independence

Background context: The statement \(E[A/B] = E[A]/E[B]\) needs to be verified for independence between two random variables \(A\) and \(B\).

:p Prove or disprove the statement: \(E[A/B] = E[A]/E[B]\).
??x
To prove or disprove this statement, consider the definition of conditional expectation and independence.

- **Conditional Expectation**: \(E[A/B]\) is defined as the expected value of \(A\) given that event \(B\) has occurred.
- **Independence**: If \(A\) and \(B\) are independent, then:
  \[ E[A \cdot B] = E[A] \cdot E[B] \]

However, for \(E[A/B]\), this relationship does not generally hold. The correct form of the expectation is:
\[ E[A/B] = \frac{E[A \cdot B]}{E[B]} \]
This only equals \(E[A]/E[B]\) if \(A\) and \(B\) are constants or in specific scenarios where independence simplifies the expression, but generally it does not hold.

Thus, the statement is disproved.
x??

---

#### Expectation of Product

Background context: The claim that if \(E[XY] = E[X] \cdot E[Y]\), then \(X\) and \(Y\) are independent random variables needs to be verified.

:p Prove or disprove the claim: If \(E[XY] = E[X] \cdot E[Y]\), then \(X\) and \(Y\) are independent r.v.’s.
??x
To verify this, consider the definition of independence for two random variables:

- **Independence**: Two random variables \(X\) and \(Y\) are independent if:
  \[ P(X = x, Y = y) = P(X = x) \cdot P(Y = y) \]
  
The expectation condition given is:
\[ E[XY] = E[X] \cdot E[Y] \]

This condition alone does not guarantee independence. Independence requires that the joint distribution factors into the product of marginal distributions.

A counterexample can be constructed where \(E[XY] = E[X] \cdot E[Y]\) but \(X\) and \(Y\) are not independent, such as when one random variable is a constant multiple of another.

Thus, the statement is disproved.
x??

---

#### Variance of Binomial Distribution

Background context: Derive the variance of a binomial distribution using Theorem 3.27.

:p Use Theorem 3.27 to derive \(Var(X)\) for a binomially distributed random variable \(X \sim Binomial(n, p)\).
??x
Theorem 3.27 states that if \(X\) is the sum of independent Bernoulli trials:
\[ X = B_1 + B_2 + \cdots + B_n \]
where each \(B_i\) is a Bernoulli random variable with success probability \(p\), then the variance of \(X\) is given by:
\[ Var(X) = n \cdot p (1 - p) \]

For a binomial distribution, this directly applies to:
\[ X \sim Binomial(n, p) \]
Thus,
\[ Var(X) = n \cdot p (1 - p) \]
x??

---

#### Poisson Approximation to Binomial Distribution

Background context: The Poisson approximation can be used when the number of trials \(n\) is large and the probability of success \(p\) is small.

:p Prove that the binomial distribution \(Binomial(n, p)\) is well approximated by the Poisson distribution with parameter \(\lambda = np\).
??x
Given a binomial random variable \(X \sim Binomial(n, p)\), we start from its probability mass function (pmf):
\[ P(X = k) = {n \choose k} p^k (1-p)^{n-k} \]

Set \(p = \frac{\lambda}{n}\) and let \(n\) be large while \(np = \lambda\). The pmf becomes:
\[ P(X = k) = {n \choose k} \left(\frac{\lambda}{n}\right)^k \left(1 - \frac{\lambda}{n}\right)^{n-k} \]

As \(n \to \infty\), using the approximation \((1 - x/n)^n \approx e^{-x}\) for small \(x\):
\[ P(X = k) \approx {n \choose k} \left(\frac{\lambda}{n}\right)^k e^{-\lambda} \]

Using Stirling's approximation, \({n \choose k} \approx \frac{n^k}{k!}\), we get:
\[ P(X = k) \approx \frac{1}{k!} \lambda^k e^{-\lambda} \]

This is the pmf of a Poisson distribution with parameter \(\lambda = np\).

Thus, for large \(n\) and small \(p\), \(Binomial(n, p)\) can be approximated by \(Poisson(np)\).
x??

---

#### Probability Bounds

Background context: Given the average file size in a database is 6K, we want to derive upper bounds on the percentage of files larger than 12K.

:p Explain why fewer than half of the files can have a size greater than 12K.
??x
By definition of expectation (mean), if the average file size is 6K, then:
\[ E[\text{File Size}] = 6 \text{K} \]

The mean being less than 12K implies that more than half of the files must be below or equal to 6K on average. Therefore, fewer than half of the files can have a size greater than 12K.

Formally, if we denote the file size by \(X\):
\[ E[X] = \sum_{i} P(X_i) \cdot X_i = 6 \text{K} \]

If more than 50% of the files had a size >12K, then:
\[ \sum_{i: X_i > 12 \text{K}} P(X_i) \cdot X_i > 6 \text{K} \]
which contradicts \(E[X] = 6 \text{K}\).

Thus, fewer than half of the files can have a size greater than 12K.
x??

---

#### Quality of Service

Background context: The company pays a fine if the time to process a request exceeds 7 seconds. Processing consists of two tasks with specific distributions.

:p Determine the fraction of requests that will result in a fine, given task times \(X \sim Exp(5)\) and \(Y \sim Uniform[1,3]\).
??x
Let:
- \(X\) be exponentially distributed with mean 5 seconds.
- \(Y\) be uniformly distributed between 1 and 3 seconds.

The total processing time is:
\[ T = X + Y \]

We need to find the probability that \(T > 7\).

First, calculate the cumulative distribution function (CDF) of \(X\):
\[ F_X(x) = 1 - e^{-x/5} \]
Thus,
\[ P(X > x) = e^{-x/5} \]

The CDF of \(Y\) is:
\[ F_Y(y) = \frac{y-1}{2} \text{ for } 1 \leq y \leq 3 \]

To find \(P(T > 7)\):
\[ P(X + Y > 7) = \int_{1}^{3} P(X > 7 - y) f_Y(y) dy \]
where \(f_Y(y) = \frac{1}{2}\).

Thus:
\[ P(X > 7 - y) = e^{-(7-y)/5} \]

So,
\[ P(T > 7) = \int_{1}^{3} e^{-(7-y)/5} \cdot \frac{1}{2} dy \]
\[ = \frac{1}{2} \int_{1}^{3} e^{(y-7)/5} dy \]

Let \(u = y - 7\), then:
\[ du = dy \]
When \(y = 1\), \(u = -6\); when \(y = 3\), \(u = -4\).

Thus,
\[ P(T > 7) = \frac{1}{2} \int_{-6}^{-4} e^{u/5} du \]
\[ = \frac{1}{2} [5e^{u/5}]_{-6}^{-4} \]
\[ = \frac{5}{2} (e^{-4/5} - e^{-6/5}) \]

This value can be computed numerically. Given the properties of exponential and uniform distributions, it turns out that \(P(T > 7) < 0.5\).

Thus, fewer than half the requests will result in a fine.
x??

---

#### Positive Correlation

Background context: Events \(A\) and \(B\) are positively correlated if \(P(A|B) > P(A)\). The reverse implication needs to be verified.

:p Prove or disprove that \(P(A|B) > P(A)\) implies \(P(B|A) > P(B)\).
??x
To prove or disprove the statement, we use Bayes' theorem:
\[ P(B|A) = \frac{P(A|B) \cdot P(B)}{P(A)} \]

Given \(P(A|B) > P(A)\), we want to determine if this implies \(P(B|A) > P(B)\).

From the given condition:
\[ P(A|B) > P(A) \]
By Bayes' theorem, since \(P(B|A)\) is a ratio involving \(P(A|B)\):
\[ P(B|A) = \frac{P(A|B) \cdot P(B)}{P(A)} > \frac{P(A) \cdot P(B)}{P(A)} = P(B) \]

Thus, the statement is true.
x??

---

#### Covariance

Background context: The covariance measures how two random variables change together.

:p Prove that \(cov(X, Y) = E[XY] - E[X]E[Y]\).
??x
To prove this, start with the definition of covariance:
\[ cov(X, Y) = E[(X - E[X])(Y - E[Y])] \]

Expand the expression inside the expectation:
\[ cov(X, Y) = E[XY - XE[Y] - YE[X] + E[X]E[Y]] \]
Since \(E[Y]\) and \(E[X]\) are constants, we can distribute the expectation:
\[ cov(X, Y) = E[XY] - E[XE[Y]] - E[YE[X]] + E[E[X]E[Y]] \]

Using linearity of expectation:
\[ E[XE[Y]] = E[X]E[Y] \]
\[ E[YE[X]] = E[Y]E[X] \]
Thus,
\[ cov(X, Y) = E[XY] - E[X]E[Y] - E[Y]E[X] + E[X]E[Y] \]
Simplifying this:
\[ cov(X, Y) = E[XY] - E[X]E[Y] \]

Therefore, the statement is true.
x??

---

#### Summary

This set of problems covers various aspects of probability and statistics, including bounds on file sizes, quality of service analysis, binomial approximations, and covariance proofs. Each problem builds understanding through step-by-step reasoning and formal derivations. The solutions provided should help in grasping the core concepts involved. x??

#### Bill's Fundraising Probability (Normal Approximation)
Bill hopes to raise $1,000,000. We need to compute the probability that he raises less than $999,000 using the Normal approximation.

Assume the distribution of the amount raised is approximately normal with mean \(\mu\) and standard deviation \(\sigma\).

:p What is the formula for converting a value \(X\) from a normal distribution to a standard normal distribution?
??x
The z-score transformation formula is:
\[ Z = \frac{X - \mu}{\sigma} \]
This converts the random variable \(X\) into a standard normal variable \(Z\).
x??

#### Bill's Fundraising Probability (Exact Calculation)
Now, write an exact expression for this probability and evaluate it using a calculator or program.

Assume the distribution is exactly known with mean \(\mu = 1000000\) and standard deviation \(\sigma = 1000\).

:p What is the formula to calculate \(P(X < 999000)\) for a normal distribution?
??x
The exact probability can be calculated using:
\[ P(X < 999000) = \Phi\left( \frac{999000 - \mu}{\sigma} \right) \]
Where \(\Phi\) is the cumulative distribution function of the standard normal distribution.
x??

---

#### Eric and Timmy's Meeting Probability
Eric and Timmy have agreed to meet between 2 and 3 pm. Each has an independent uniform arrival time over this hour, and they each wait for 15 minutes.

:p What is the probability that Eric and Timmy will be able to meet?
??x
To find the probability of meeting, we can use a geometric approach or calculate it directly by integrating the joint probability density function (pdf) over the region where their arrival times overlap within 15 minutes of each other.

The exact expression for the probability is:
\[ P(\text{meet}) = \int_{2}^{3} \int_{\max(t - \frac{15}{60}, t)}{\min(t + \frac{15}{60}, u)} du dt \]

Evaluating this integral, we get the probability that they meet.
x??

---

#### Weather Prediction for John and Mary's Wedding
The average number of rainy days per year is 10. The weather forecaster predicts rain with a certain accuracy rate based on whether it rains or not.

:p What is the probability that it will rain during their wedding, given the weather forecast?
??x
Let \( R \) be the event it rains and \( F \) be the event the forecast predicts rain. We know:
- \( P(R|F) = 0.9 \)
- \( P(F|R^c) = 0.1 \)

Using Bayes' theorem, we can find \( P(R|F) \).

\[ P(R|F) = \frac{P(F|R)P(R)}{P(F)} \]

Where:
\[ P(F) = P(F|R)P(R) + P(F|R^c)P(R^c) \]
x??

---

#### Health Care Testing for H1N1 Vaccine
A vaccine is being tested, and the initial laboratory test has an accuracy rate of 0.6.

:p What is the probability that the vaccine is effective given a "success" from the lab test?
??x
We can use Bayes' theorem to find \( P(\text{Effective} | \text{Success}) \).

\[ P(\text{Effective} | \text{Success}) = \frac{P(\text{Success}|\text{Effective})P(\text{Effective})}{P(\text{Success})} \]

Where:
- \( P(\text{Success}|\text{Effective}) = 0.6 \)
- \( P(\text{Success}|\text{Ineffective}) = 0.4 \)
- \( P(\text{Effective}) = 0.5 \)

To find \( P(\text{Success}) \), use the law of total probability:
\[ P(\text{Success}) = P(\text{Success}|\text{Effective})P(\text{Effective}) + P(\text{Success}|\text{Ineffective})P(\text{Ineffective}) \]
x??

---

#### Additional Health Care Testing
The company decides to add a more accurate test with an 80% success rate for effective vaccines and 20% failure rate.

:p What is the probability that the vaccine is effective given both tests return "success"?
??x
Using Bayes' theorem again:
\[ P(\text{Effective} | \text{Success, Success}) = \frac{P(\text{Success},\text{Success}|\text{Effective})P(\text{Effective})}{P(\text{Success},\text{Success})} \]

Where:
- \( P(\text{Success},\text{Success}|\text{Effective}) = 0.8 \times 0.8 = 0.64 \)
- \( P(\text{Success},\text{Success}|\text{Ineffective}) = 0.2 \times 0.2 = 0.04 \)
- \( P(\text{Effective}) = 0.5 \)

To find the denominator:
\[ P(\text{Success},\text{Success}) = P(\text{Success},\text{Success}|\text{Effective})P(\text{Effective}) + P(\text{Success},\text{Success}|\text{Ineffective})P(\text{Ineffective}) \]
x??

---

#### Dating Costs and Expectation
A man tries two approaches to dating: generous (costing $1000) and cheapskate ($50).

:p What is the expected cost of the date if he uses the generous approach?
??x
The expected cost \( E(X) \) for a single date using the generous approach can be calculated as:
\[ E(X_{\text{generous}}) = 1000 \times P(\text{marry}) + 50 \times P(\text{break up}) \]

Where:
- \( P(\text{marry}) = 0.05 \)
- \( P(\text{break up}) = 0.95 \)

\[ E(X_{\text{generous}}) = 1000 \times 0.05 + 50 \times 0.95 = 50 + 47.5 = 97.5 \]
x??

#### Expected Cost to Find a Wife
Background context: The man, who has only experienced failure so far and is choosing an approach (generous or cheapskate) at random, decides to search for a wife. This scenario can be modeled using a geometric distribution since it involves the number of trials until the first success.

Formula: Let \(X\) represent the number of people the man needs to date before finding a suitable wife. Since he chooses an approach randomly (with equal probability), we assume that the probability of success on any given trial is \(p = 0.5\).

:p What is the expected cost for the man to find a wife?
??x
The expected number of trials until the first success in a geometric distribution with parameter \(p\) is given by \(\frac{1}{p}\). Since \(p = 0.5\), the expected number of people he needs to date before finding a suitable wife is:

\[ E[X] = \frac{1}{0.5} = 2 \]

This means, on average, he would need to date 2 people.

x??

---

#### Variance of Geometric Distribution
Background context: The variance for the geometric distribution can be derived using conditioning and properties of expectations. We aim to show that \(Var(X) = \frac{1-p}{p^2}\).

:p What is the variance of a geometrically distributed random variable \(X\)?
??x
To find the variance, we use the formula:

\[ Var(X) = E[X^2] - (E[X])^2 \]

For a geometric distribution with parameter \(p\), the expected value \(E[X]\) is known to be \(\frac{1}{p}\). We need to calculate \(E[X^2]\).

Using conditioning on the first trial, we get:

\[ E[X^2] = 1 + p(1 + E[X]) \]

Since \(E[X] = \frac{1}{p}\), substituting this in gives us:

\[ E[X^2] = 1 + p\left(1 + \frac{1}{p}\right) = 1 + p + 1 = 2 + p \]

Now, we can find the variance:

\[ Var(X) = E[X^2] - (E[X])^2 = (2 + p) - \left(\frac{1}{p}\right)^2 = 2 + p - \frac{1}{p^2} = \frac{2p^2 - 1 + p^3}{p^2} = \frac{1 - p}{p^2} \]

x??

---

#### Good Chips versus Lemons
Background context: A chip supplier produces good chips and lemons with probabilities 0.95 and 0.05, respectively. The lifespans of the chips are geometrically distributed. We need to find \(E[T]\) and \(Var(T)\).

:p What is the expected time until a randomly chosen chip fails?
??x
The expected lifespan for both good chips and lemons can be calculated using the properties of the geometric distribution.

For good chips:
- Probability of failure each day: 0.9991 (since it fails with probability \(0.0001\))
- Expected time until failure: \(\frac{1}{0.9991} \approx 1000\) days

For lemons:
- Probability of failure each day: 0.01
- Expected time until failure: \(\frac{1}{0.01} = 100\) days

The overall expected lifespan \(E[T]\) can be calculated using the law of total expectation:

\[ E[T] = P(\text{good}) \cdot E[T|P=\text{good}] + P(\text{lemon}) \cdot E[T|P=\text{lemon}] \]

Substituting the values:
\[ E[T] = 0.95 \cdot 1000 + 0.05 \cdot 100 = 950 + 5 = 955 \]

x??

---

#### Expectation via Conditioning - Stacy’s System
Background context: Stacy's fault-tolerant system crashes if there are \(k\) consecutive failures, with each failure occurring independently at a rate of \(p\).

:p What is the expected number of minutes until Stacy's system crashes?
??x
The problem can be modeled using a geometric distribution. Let \(X_i\) represent the time between the \((i-1)\)-th and \(i\)-th failures. Each failure occurs independently with probability \(p = 0.1\).

To find the expected number of minutes until there are \(k\) consecutive failures, we need to consider the geometric distribution for each interval between failures.

The expected time for \(k\) consecutive failures is:

\[ E[X] = \frac{1}{p^k} \]

In this case, \(k = 1\), so:
\[ E[X] = \frac{1}{0.1} = 10 \text{ minutes per failure} \]
Since we need \(k\) consecutive failures:
\[ E[\text{time until crash}] = k \cdot E[X] = 1 \cdot 10 = 10 \text{ minutes} \]

x??

---

#### Napster - Brought to You by the RIAA
Background context: The problem involves downloading songs from a band where each download is random, and we need to determine how many downloads are required on average to get all 50 songs.

:p What is the expected number of downloads \(E[D]\) required to collect all 50 songs?
??x
The problem can be modeled using the coupon collector's problem. The expected number of trials (downloads in this case) to collect all \(n\) distinct items (songs) is given by:

\[ E[D] = n \sum_{i=1}^{n} \frac{1}{i} \]

For 50 songs:
\[ E[D] = 50 \left( \sum_{i=1}^{50} \frac{1}{i} \right) \]

The sum of the harmonic series up to 50 is approximately:

\[ \sum_{i=1}^{50} \frac{1}{i} \approx \ln(50) + \gamma \approx 4.605 + 0.577 = 5.182 \]

Thus:
\[ E[D] \approx 50 \times 5.182 = 259.1 \]

A closed-form approximation is:

\[ E[D] \approx n \ln(n) + n \gamma \]

x??

---

#### Fractional Moments of Exponential Distribution
Background context: The problem involves computing \(E[X^{1/2}]\) for an exponentially distributed random variable \(X\). This requires using integration by parts and a change of variables.

:p What is the expected value of the square root of an exponentially distributed random variable?
??x
Given \(X \sim Exp(1)\), we need to compute \(E[X^{1/2}]\).

Using the definition of expectation:
\[ E[X^{1/2}] = \int_0^\infty x^{1/2} e^{-x} dx \]

This integral can be solved using integration by parts. Let \(u = x^{1/2}\) and \(dv = e^{-x} dx\). Then, \(du = \frac{1}{2} x^{-1/2} dx\) and \(v = -e^{-x}\).

Using integration by parts:
\[ \int u dv = uv - \int v du \]

Thus:
\[ E[X^{1/2}] = -x^{1/2} e^{-x} \Big|_0^\infty + \frac{1}{2} \int_0^\infty x^{-1/2} e^{-x} dx \]

The boundary term vanishes at both ends. The remaining integral is:

\[ E[X^{1/2}] = \frac{1}{2} \int_0^\infty x^{-1/2} e^{-x} dx \]

Making the substitution \(u = \sqrt{x}\), we get \(dx = 2u du\):

\[ E[X^{1/2}] = \frac{1}{2} \int_0^\infty u^2 e^{-u^2} (2u) du = \int_0^\infty u^3 e^{-u^2} du \]

Using another substitution \(v = u^2\), we get \(dv = 2u du\) and the integral becomes:

\[ E[X^{1/2}] = \frac{1}{4} \int_0^\infty v e^{-v} dv = \frac{1}{4} \cdot \Gamma(2) = \frac{1}{4} \]

Thus, the expected value of \(X^{1/2}\) is:

\[ E[X^{1/2}] = \sqrt{\frac{\pi}{2}} \]

x??

---

