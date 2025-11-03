# Flashcards: 7A003--Performance-modeling-and-design-of-computer-systems_processed (Part 4)

**Starting Chapter:** 3.10 Joint Probabilities and Independence

---

#### Variance of Bernoulli Distribution
Background context: The variance of a random variable \(X\) can be calculated using the formula \(\text{Var}(X) = E[(X - \mu)^2]\), where \(\mu\) is the expected value. For a Bernoulli distribution, which models a binary outcome with probability \(p\), we have two possible outcomes: 0 and 1.
The formula for the variance of a Bernoulli random variable \(X\) is:
\[ E[(X - p)^2] = p(1-p) \]
:p What is the formula to calculate the variance of a Bernoulli distribution?
??x
The variance of a Bernoulli random variable \(X\sim \text{Bernoulli}(p)\) is calculated as:
\[ \text{Var}(X) = p(1 - p) \]
This formula leverages the definition of variance and the properties of the Bernoulli distribution.
x??

---

#### Variance of Uniform Distribution
Background context: The uniform distribution \(X\sim \text{Uniform}(a, b)\) assigns equal probability to all values between \(a\) and \(b\). The expected value for a continuous uniform distribution is given by:
\[ E[X] = \frac{a + b}{2} \]
The variance of a uniform distribution can be derived using the formula for variance.
:p What is the formula to calculate the variance of a continuous uniform distribution?
??x
The variance of a continuous uniform random variable \(X\sim \text{Uniform}(a, b)\) is calculated as:
\[ \text{Var}(X) = \frac{(b - a)^2}{12} \]
This formula is derived from the integral of the squared deviation from the mean over the interval.
x??

---

#### Joint Probability Mass Function
Background context: In probability theory, joint probabilities refer to the combined probabilities of two or more events happening simultaneously. The joint probability mass function \(p_{X,Y}(x, y)\) for discrete random variables \(X\) and \(Y\) is defined as:
\[ p_{X,Y}(x, y) = P(X = x \text{ and } Y = y) \]
:p What is the definition of the joint probability mass function between two discrete random variables?
??x
The joint probability mass function \(p_{X,Y}(x, y)\) for two discrete random variables \(X\) and \(Y\) is defined as:
\[ p_{X,Y}(x, y) = P(X = x \text{ and } Y = y) \]
This represents the combined probability that both events \(X = x\) and \(Y = y\) occur.
x??

---

#### Joint Probability Density Function
Background context: For continuous random variables, joint probabilities are represented by a joint probability density function \(f_{X,Y}(x, y)\). The integral of this function over an area gives the probability that the two variables fall within that region. The definition for the joint probability density function is:
\[ P(a < X < b \text{ and } c < Y < d) = \int_c^d \int_a^b f_{X,Y}(x, y) \, dx \, dy \]
:p What is the relationship between \(f_X(x)\) and \(f_{X,Y}(x, y)\)?
??x
Applying the Law of Total Probability, the marginal probability density function \(f_X(x)\) can be obtained by integrating the joint probability density function \(f_{X,Y}(x, y)\) over all possible values of \(y\):
\[ f_X(x) = \int_{-\infty}^{\infty} f_{X,Y}(x, y) \, dy \]
Similarly, for the marginal density function of \(Y\):
\[ f_Y(y) = \int_{-\infty}^{\infty} f_{X,Y}(x, y) \, dx \]
:p What is the relationship between \(p_X(x)\) and \(f_{X,Y}(x, y)\)?
??x
The marginal probability mass function \(p_X(x)\) can be obtained by summing over all possible values of \(y\):
\[ p_X(x) = \sum_y p_{X,Y}(x, y) \]
For the continuous case with a joint density function:
\[ f_X(x) = \int_{-\infty}^{\infty} f_{X,Y}(x, y) \, dy \]
Similarly, for \(p_Y(y)\):
\[ p_Y(y) = \sum_x p_{X,Y}(x, y) \]
x??

---

#### Independence of Random Variables
Background context: Two random variables \(X\) and \(Y\) are said to be independent if the occurrence of one does not affect the probability of the other. The definition in terms of their joint distribution is:
\[ p_{X,Y}(x, y) = p_X(x) \cdot p_Y(y) \]
For continuous random variables:
\[ f_{X,Y}(x, y) = f_X(x) \cdot f_Y(y), \quad \forall x, y \]
:p How do you define the independence of two discrete random variables \(X\) and \(Y\)?
??x
Two discrete random variables \(X\) and \(Y\) are independent if their joint probability mass function satisfies:
\[ p_{X,Y}(x, y) = p_X(x) \cdot p_Y(y) \]
This means that the occurrence of one event does not affect the probability of the other.
x??

---

#### Expected Value of Product for Independent Random Variables
Background context: If two random variables \(X\) and \(Y\) are independent, their expected value of the product can be simplified using the property:
\[ E[XY] = E[X] \cdot E[Y] \]
:p What is the theorem that relates to the expected value of the product for independent random variables?
??x
Theorem 3.20 states that if two random variables \(X\) and \(Y\) are independent, then their expected value of the product can be expressed as:
\[ E[XY] = E[X] \cdot E[Y] \]
This theorem simplifies calculations involving the expected values of products in independent scenarios.
x??

#### Conditional Probabilities and Expectations: Discrete Case
Background context explaining the concept. We extend the idea of conditional probabilities from events to random variables, focusing on discrete cases first.

Example:
- Suppose we have a class with different hair colors: Blondes (1), Red-heads (2), Brunettes (3), and Black-haired people (4).
- Let \(X\) be a random variable whose value is the hair color.
- The probability mass function for \(X\) is given as follows:

  \[
  p_X(1) = P\{\text{Blonde}\} = \frac{5}{38}, \quad
  p_X(2) = P\{\text{Red}\} = \frac{2}{38}, \quad
  p_X(3) = P\{\text{Brown}\} = \frac{17}{38}, \quad
  p_X(4) = P\{\text{Black}\} = \frac{14}{38}
  \]

- Let \(A\) be the event that a person has light-colored hair (Blondes or Red-heads), so:

  \[
  P(A) = P(\{\text{Blonde, Red}\}) = \frac{7}{38}, \quad
  P(A^c) = 1 - P(A) = \frac{31}{38}
  \]

- The conditional probability mass function \(p_{X|A}(x)\) is defined as:

  \[
  p_{X|A}(x) = P(X=x | A) = \frac{P((X=x) \cap A)}{P(A)}
  \]

:p What is the question about this concept?
??x
- Compute \(p_{X|A}(\text{Blonde})\) and \(p_{X|A}(\text{Red})\).
x??

```java
// Java code to compute conditional probabilities for light-colored hair
public class ConditionalProbExample {
    public static void main(String[] args) {
        double p_A = 7.0 / 38; // P(A)
        
        // Conditional probabilities given A (light-colored hair)
        double p_X_given_A_Blond = 5.0 / 7;
        double p_X_given_A_Red = 2.0 / 7;
    }
}
```

---

#### Conditional Expectation: Discrete Case
Background context explaining the concept. The conditional expectation of a random variable \(X\) given an event or another random variable.

Example:
- Using the same example, let's say Blonde is represented by value 1 and Red-haired as 2.
- Compute \(E[X|A]\):

  \[
  E[X|A] = 1 \cdot p_{X|A}(1) + 2 \cdot p_{X|A}(2)
  \]

:p What is the question about this concept?
??x
- Calculate \(E[X|A]\).
x??

```java
// Java code to compute conditional expectation for light-colored hair
public class ConditionalExpExample {
    public static void main(String[] args) {
        // Given probabilities from previous example
        double p_X_given_A_Blond = 5.0 / 7;
        double p_X_given_A_Red = 2.0 / 7;
        
        // Calculate E[X|A]
        double E_X_given_A = 1 * p_X_given_A_Blond + 2 * p_X_given_A_Red; // Should be 9/7
    }
}
```

---

#### Conditioning on Random Variables: Discrete Case
Background context explaining the concept. We extend conditional probabilities and expectations to situations where one random variable depends on another.

Example:
- Consider two discrete random variables \(X\) and \(Y\), both taking values \{0, 1, 2\}.
- Their joint probability mass function is given by Table 3.3.
- Let's compute the conditional expectation \(E[X|Y=2]\):

  \[
  E[X|Y=2] = \sum_x x \cdot p_{X|Y}(x|2)
  \]

:p What is the question about this concept?
??x
- Calculate \(E[X|Y=2]\).
x??

```java
// Java code to compute conditional expectation given Y=2
public class ConditionalExpGivenY {
    public static void main(String[] args) {
        // Given joint probabilities from Table 3.3
        double p_X_given_Y_Blond_2 = 1.0 / 6; // P(X=0 and Y=2)/P(Y=2)
        double p_X_given_Y_Red_2 = 1.0 / 8;   // P(X=1 and Y=2)/P(Y=2)
        
        // Calculate E[X|Y=2]
        double E_X_given_Y_2 = 0 * p_X_given_Y_Blond_2 + 1 * p_X_given_Y_Red_2;
    }
}
```

---

#### Continuous Random Variables: Conditional Probability
Background context explaining the concept. The conditional probability and expectation for continuous random variables.

Example:
- For a continuous random variable \(X\) with an exponential distribution, say \(X \sim \text{Exp}(1/1000)\), we need to find the conditional density given that job is in bin 1.
- Given that jobs are sent to bin 1 if they require less than 500 CPU hours.

:p What is the question about this concept?
??x
- Compute \(f_{X|Y}(t)\) where \(Y\) is the event that the job is sent to bin 1.
x??

```java
// Java code to compute conditional density function for exponential distribution given Y=2 (bin 1)
public class ConditionalDensityExample {
    public static void main(String[] args) {
        // Define the parameters
        double lambda = 1000; // Rate parameter of Exponential
        
        // Given that job is in bin 1, t < 500
        double t = 200;
        
        // Conditional density function f_{X|Y}(t)
        if (t < 500) {
            double f_X_given_Y_t = lambda * Math.exp(-lambda * t);
            System.out.println("f_{X|Y}(" + t + ") = " + f_X_given_Y_t);
        } else {
            System.out.println("f_{X|Y}(t) = 0 for t >= 500");
        }
    }
}
```

---

#### Continuous Random Variables: Conditional Expectation
Background context explaining the concept. The conditional expectation of a continuous random variable given an event.

Example:
- For \(X \sim \text{Exp}(1/1000)\), and we need to find the expected duration if the job is in bin 1.
- Jobs are sent to bin 1 if they require less than 500 CPU hours, so we integrate from 0 to 500.

:p What is the question about this concept?
??x
- Compute \(E[X|Y=2]\) for an exponentially distributed job duration with mean 1000 processor-hours.
x??

```java
// Java code to compute conditional expectation given Y=2 (job in bin 1)
public class ConditionalExpContinuousExample {
    public static void main(String[] args) {
        // Define the parameters
        double lambda = 1000; // Rate parameter of Exponential
        double lower_bound = 0;
        double upper_bound = 500;
        
        // Compute E[X|Y=2]
        double expected_duration = (1 / lambda) * (Math.exp(-lambda * lower_bound) - Math.exp(-lambda * upper_bound));
    }
}
``` 

--- 
Note: The Java code snippets are simplified and assume the use of standard mathematical functions. They are meant to illustrate the logic involved in calculating conditional probabilities and expectations, not as fully functional implementations. \(\)

#### Law of Total Probability for Discrete Random Variables
The Law of Total Probability extends to random variables, allowing us to break down complex problems into simpler sub-problems. For discrete random variables \(X\) and partitioning events \(Y = y\), we have:
\[ P\{X=k\} = \sum_{y} P\{X=k|Y=y\}P\{Y=y\}. \]
This is a powerful tool for simplifying the calculation of probabilities.

:p What does the Law of Total Probability for Discrete Random Variables state?
??x
The law states that to find the probability \(P\{X=k\}\), we can sum over all possible values of \(Y\) (the conditioning event) the product of the conditional probability \(P\{X=k|Y=y\}\) and the marginal probability \(P\{Y=y\}\).

For example, if we want to find the probability that a geometric random variable \(N\) is less than 3:
```java
// P(N < 3) = P(N=1) + P(N=2)
// Using the formula: P(N=k | Y=y) * P(Y=y)
double p = 0.5; // Example parameter for a geometric distribution with success probability p
double prob_N_less_than_3 = (1 - Math.pow(1-p, 1)) + (1 - Math.pow(1-p, 2));
```
x??

---

#### Conditional Expectation and Linearity of Expectation for Discrete Random Variables
The theorem states that the expected value of a random variable \(X\) can be computed by summing the conditional expectations given each possible value of another random variable \(Y\), weighted by the probability of those values. For discrete random variables, we have:
\[ E[X] = \sum_y E[X|Y=y]P\{Y=y\}. \]

:p How is the expected value of a discrete random variable derived using conditioning?
??x
The expected value \(E[X]\) can be found by summing over all possible values of \(Y\) (the conditioning event), multiplying the conditional expectation \(E[X|Y=y]\) with the probability of each \(Y\):
\[ E[X] = \sum_y E[X|Y=y]P\{Y=y\}. \]

For instance, if we want to find the expected number of trials for a geometric distribution:
```java
// E[N | Y=1] * P(Y=1) + E[N | Y=0] * P(Y=0)
double p = 0.5; // Example parameter for a geometric distribution with success probability p
double exp_N = (1/p) * p + (2/p) * (1 - p);
```
x??

---

#### Linearity of Expectation
One of the most powerful theorems in probability, it states that for any random variables \(X\) and \(Y\), the expected value of their sum is equal to the sum of their individual expected values:
\[ E[X + Y] = E[X] + E[Y]. \]

:p What does the Linearity of Expectation theorem state?
??x
The linearity of expectation states that for any random variables \(X\) and \(Y\), the expected value of their sum is equal to the sum of their individual expected values:
\[ E[X + Y] = E[X] + E[Y]. \]
This holds true even if \(X\) and \(Y\) are not independent.

For example, when calculating the expected number of heads in two coin flips:
```java
// Let X1 and X2 be the outcomes of two coin flips
double exp_X1_plus_X2 = 0.5 + 0.5; // Each flip has an expected value of 0.5
```
x??

---

#### Application of Linearity of Expectation in Binomial Distribution
The binomial distribution can be expressed as a sum of indicator random variables, which simplifies the calculation of its mean using linearity of expectation.

:p How does linearity of expectation simplify the calculation of the expected value for a binomial distribution?
??x
By expressing the binomial random variable \(X\) as a sum of indicator random variables \(X_i\), we can use linearity of expectation to find the mean:
\[ E[X] = \sum_{i=1}^n E[X_i], \]
where each \(E[X_i]\) is equal to the probability of success in one trial.

For instance, if we have a binomial random variable with parameters \(n\) and \(p\):
```java
// Each X_i is an indicator variable that equals 1 with probability p
int n = 10; // Number of trials
double p = 0.3; // Probability of success in one trial
double exp_X = n * p; // Summing the expected values
```
x??

---

#### Application of Linearity of Expectation in Hat Problem
In a scenario where \(n\) people throw their hats into a circle and randomly pick one, we can use linearity of expectation to find the expected number of people who get back their own hat.

:p How does linearity of expectation apply to the hat problem?
??x
By defining indicator random variables \(I_i\), each equal to 1 if the \(i\)th person picks their own hat and 0 otherwise, we can use linearity of expectation:
\[ E[X] = \sum_{i=1}^n E[I_i], \]
where \(E[I_i]\) is the probability that the \(i\)th person picks their own hat.

For example, in a party with 10 people:
```java
// Each I_i has an expected value of 1/n if n trials are independent (not true here)
int n = 10; // Number of people
double exp_X = n * (1.0 / n); // Using symmetry and linearity of expectation
```
x??

---

