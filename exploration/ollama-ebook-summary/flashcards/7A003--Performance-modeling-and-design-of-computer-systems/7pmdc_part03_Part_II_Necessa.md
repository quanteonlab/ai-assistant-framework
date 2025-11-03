# Flashcards: 7A003--Performance-modeling-and-design-of-computer-systems_processed (Part 3)

**Starting Chapter:** Part II Necessary Probability Background

---

#### Quick Review of Undergraduate Probability
Background context: This chapter provides a rapid review of fundamental probability concepts necessary for understanding more advanced topics throughout the book. Key concepts include basic definitions, axioms, and common distributions.

:p What are the main topics covered in Chapter 3?
??x
Chapter 3 covers foundational probability theory such as sample spaces, events, probability measures, random variables, and some common distributions like Bernoulli, Binomial, Poisson, etc.
x??

---

#### Methods for Generating Random Variables
Background context: This chapter discusses techniques for generating random variables which are crucial in simulating queues. Common methods include inverse transform sampling, rejection sampling, and more specialized techniques.

:p What is the inverse transform sampling method?
??x
The inverse transform sampling method involves transforming a uniform random variable into a desired distribution by using the cumulative distribution function (CDF). Given a CDF \(F(x)\), we generate a uniform U(0,1) random number u and compute the value of x such that \(u = F(x)\).

```java
public class InverseTransform {
    public static double inverseTransform(double cdfValue, Function<Double, Double> inverseCDF) {
        return inverseCDF.apply(cdfValue);
    }
}
```

This method is particularly useful when the CDF can be easily inverted. x??

---

#### Sample Paths and Convergence of Random Variables
Background context: Chapter 5 delves into more advanced topics such as sample paths, convergence of sequences of random variables, and different types of averages (time and ensemble). These concepts are essential for understanding stochastic processes.

:p What is the concept of a sample path in probability theory?
??x
A sample path or realization of a stochastic process refers to the sequence of outcomes that occur over time. For example, if \(X(t)\) represents a random variable at time t, then a sample path is the actual sequence of values \{X(t_1), X(t_2), ...\} observed for a particular outcome.

```java
public class SamplePath {
    public List<Double> generateSamplePath(int steps, RandomGenerator generator) {
        List<Double> path = new ArrayList<>();
        double currentTime = 0;
        while (path.size() < steps) {
            // Simulate the next state using the random generator
            Double nextState = simulateNextState(currentTime++, generator);
            path.add(nextState);
        }
        return path;
    }

    private Double simulateNextState(double time, RandomGenerator generator) {
        // Logic to generate the next state based on the current time and a random number
        double randomNumber = generator.nextUniform();
        // Implement specific logic here
        return nextState;
    }
}
```

This method generates a sample path by simulating states at each time step. x??

---

#### Time Averages vs Ensemble Averages
Background context: Chapter 5 also discusses the distinction between time averages and ensemble averages, which are different ways to compute expected values over stochastic processes.

:p What is the difference between a time average and an ensemble average?
??x
- **Time Average**: This refers to the long-term average of a single sample path. It involves observing a particular realization over an extended period.
  
  ```java
  public double calculateTimeAverage(List<Double> samplePath, int windowSize) {
      double total = 0;
      for (int i = 0; i < samplePath.size(); i++) {
          if (i >= windowSize) {
              total += samplePath.get(i);
          }
      }
      return total / (samplePath.size() - windowSize + 1);
  }
  ```

- **Ensemble Average**: This involves averaging over multiple realizations of the same process. It provides a statistical expectation across different outcomes.

```java
public class EnsembleAverages {
    public double calculateEnsembleAverage(List<List<Double>> samplePaths, int timeStep) {
        double total = 0;
        for (List<Double> path : samplePaths) {
            total += path.get(timeStep);
        }
        return total / samplePaths.size();
    }
}
```

Time averages are specific to individual realizations, while ensemble averages consider multiple realizations. x??

---

#### Sample Space and Events
Background context: Probability is typically defined in terms of an experiment. The sample space, denoted by Ω, consists of all possible outcomes of the experiment. An event, E, is any subset of the sample space.

For example, if two dice are rolled, each outcome can be represented as a pair (i, j), where i and j represent the results of the first and second die respectively. The total number of outcomes is 36, which form the sample space Ω.

:p What is the definition of an event in probability?
??x
An event E is any subset of the sample space Ω.
x??

---

#### Unions and Intersections of Events
Background context: Events can be combined using set operations such as unions (E ∪ F) and intersections (E ∩ F). The complement of an event E, denoted by EC, includes all outcomes in Ω that are not in E.

For example, if we have events E1 = {(1, 3), (2, 2), (3, 1)} and E2 = {(4, 5), (5, 6)}, the union E1 ∪ E2 would include all outcomes from both sets. The intersection E1 ∩ E2 would be empty because there are no common outcomes between them.

:p How do you define the complement of an event?
??x
The complement of an event E, denoted by EC, is the set of points in Ω but not in E.
x??

---

#### Independent Events vs. Mutually Exclusive Events
Background context: Two events E1 and E2 are independent if the occurrence of one does not affect the probability of the other. They are mutually exclusive if they cannot occur simultaneously.

For example, consider rolling two dice. The event E1 = {(1, 3), (2, 2), (3, 1)} that the sum is 4 and the event E2 = {(6, 6)} that both dice show a 6 are mutually exclusive because they cannot happen at the same time.

:p Are events E1 and E2 independent or mutually exclusive?
??x
Events E1 and E2 are mutually exclusive.
x??

---

#### Probability of Events
Background context: The probability of an event E is defined as the sum of the probabilities of all sample points in E. If each sample point has equal probability, then P{E} = (number of outcomes in E) / total number of outcomes.

For example, if two dice are rolled and we want to find the probability that their sum is 4, we need to count how many pairs add up to 4 and divide by the total number of possible outcomes (36).

:p What is the formula for calculating the probability of an event?
??x
The probability of an event E is given by P{E} = (number of outcomes in E) / total number of outcomes.
x??

---

#### Probability on Discrete and Continuous Sample Spaces
Background context: The sample space can be either discrete, with a finite or countably infinite number of outcomes, or continuous, with uncountably many outcomes.

For example, rolling two dice is a discrete sample space because there are only 36 possible outcomes. However, if we were considering the time until a certain event happens in an interval, that would be a continuous sample space.

:p Can you provide an example of a continuous sample space?
??x
An example of a continuous sample space could be the time (in seconds) until a specific electronic component fails. The outcomes are uncountable because the time can take any value within a given interval.
x??

---

#### Partitions of Sets
Background context: If events E1, E2, ..., En partition set F, it means that every outcome in F belongs to exactly one of the events.

For example, if we define E1 = {(1, 3), (2, 2), (3, 1)} and E2 = {(4, 5), (5, 6)}, and their union covers all possible outcomes but no two events overlap, then they form a partition of the sample space.

:p What does it mean for events to partition a set?
??x
Events E1, E2, ..., En partition set F if every outcome in F belongs to exactly one of the events.
x??

---

#### Probability of Sample Space

Background context: The probability of the entire sample space, denoted as Ω, is always defined to be 1. This means that some event must occur.

:p What does P{Ω} equal?
??x
P{Ω} equals 1, indicating that the probability of the entire sample space occurring is certain.
x??

---

#### Union of Two Events

Background context: The probability of the union of two events E and F can be calculated using the formula \( P(E \cup F) = P(E) + P(F) - P(E \cap F) \). This formula accounts for the overlap between the events to avoid double-counting.

:p What is the formula for calculating the probability of the union of two events?
??x
The formula for calculating the probability of the union of two events E and F is \( P(E \cup F) = P(E) + P(F) - P(E \cap F) \). This formula ensures that the overlapping area between the two events (i.e., \( E \cap F \)) is not counted twice.
x??

---

#### Conditional Probabilities

Background context: The conditional probability of event E given event F, denoted as \( P(E|F) \), is defined by the formula \( P(E|F) = \frac{P(E \cap F)}{P(F)} \). This represents the probability that event E occurs under the condition that event F has already occurred.

:p How do you calculate the conditional probability of event E given event F?
??x
The conditional probability of event E given event F is calculated using the formula \( P(E|F) = \frac{P(E \cap F)}{P(F)} \). This formula gives the probability that event E occurs, given that event F has already occurred.

For example:
```java
public class ConditionalProbabilityExample {
    public static double calculateConditionalProbability(double PEandF, double PF) {
        return PEandF / PF;
    }
}
```
x??

---

#### Independence of Events

Background context: Two events E and F are said to be independent if the probability of their intersection is equal to the product of their individual probabilities, i.e., \( P(E \cap F) = P(E) \cdot P(F) \). If this condition holds, then knowing whether one event has occurred does not affect the probability of the other.

:p What is the definition of independent events?
??x
Two events E and F are defined as independent if \( P(E \cap F) = P(E) \cdot P(F) \). This means that the occurrence of one event does not influence the probability of the other event occurring.
x??

---

#### Example with Darts

Background context: The example uses the scenario of throwing a dart at an interval [0,1] to explain how probabilities are calculated for specific points and intervals. It also discusses conditional probabilities in this context.

:p In the dart-throwing experiment, what is the probability that the dart lands exactly at 0.3?
??x
The probability that the dart lands exactly at 0.3 is defined to be 0. This is because if it were greater than 0, say \(\epsilon_1 > 0\), then by similar reasoning, the probabilities of landing at any other point would also be \(\epsilon_1\). The sum of these mutually exclusive events' probabilities would exceed 1, which is not allowed since \( P(\Omega) = 1 \).

x??

---

#### Sandwich Choices Example

Background context: This example uses a table to illustrate conditional probability by calculating the fraction of days when certain sandwich choices are made.

:p What is \( P(\text{Cheese} | \text{Second half of week}) \)?
??x
The conditional probability that I eat a cheese sandwich given that it is in the second half of the week can be calculated as 2 out of 4, or \( \frac{2}{4} = \frac{1}{2} \). Alternatively, using the formula (3.1):
\[ P(\text{Cheese} | \text{Second half of week}) = \frac{P(\text{Cheese and Second half})}{P(\text{Second half})} = \frac{\frac{2}{7}}{\frac{4}{7}} = \frac{2}{4} = \frac{1}{2} \]

x??

---

#### Mutually Exclusive vs. Independent Events
Mutually exclusive events cannot occur at the same time, while independent events do not affect each other's outcomes. However, they can coexist if their probabilities are carefully considered.

:p Can mutually exclusive (non-null) events ever be independent?
??x
No, because the probability of one event occurring given that another has occurred is zero or undefined in this case.
```
// Example in Java to demonstrate non-overlapping events
public class DiceRoll {
    public static void main(String[] args) {
        boolean firstRollIs6 = false;
        boolean secondRollIs6 = true;
        
        if (firstRollIs6 && secondRollIs6) { // This line would always be false
            System.out.println("This event is impossible.");
        }
    }
}
```
x??

---

#### Independence of Events in Rolling Two Dice

:p Which pairs of events are independent when rolling two dice?
??x
Both pairs are considered independent. The outcome of the first roll does not affect the second, and vice versa.
```java
// Pseudo-code to simulate the independence of rolls
public class DieRolls {
    public static void main(String[] args) {
        boolean firstRollIs6 = Math.random() > 0.8; // Probability of 1/6
        boolean secondRollIs4 = Math.random() < 0.2; // Probability of 1/5 (arbitrary for demonstration)
        
        if (firstRollIs6 && secondRollIs4) {
            System.out.println("Both events occurred independently.");
        } else {
            System.out.println("Events did not occur simultaneously.");
        }
    }
}
```
x??

---

#### Conditional Independence

:p Can two events be independent but not conditionally independent given a third event?
??x
No, independence does not imply conditional independence. The definition of conditional independence is different and requires the probability of both events occurring together, given a third event, to equal the product of their individual probabilities given that event.
```java
// Pseudo-code for checking conditional independence
public class ConditionalIndependence {
    public static void main(String[] args) {
        double probEAndFGivenG = 0.5; // Hypothetical probability
        double probEGivenG = 0.3;     // Probability of E given G
        double probFGivenG = 0.2;     // Probability of F given G
        
        if (probEAndFGivenG == probEGivenG * probFGivenG) {
            System.out.println("Events are conditionally independent.");
        } else {
            System.out.println("Events are not conditionally independent.");
        }
    }
}
```
x??

---

#### Law of Total Probability

:p How does the law of total probability work?
??x
The law states that a probability can be computed by partitioning the sample space into disjoint events and summing the conditional probabilities of the event given each partition.
```java
// Pseudo-code for applying the Law of Total Probability
public class TotalProbability {
    public static void main(String[] args) {
        double probCacheFailure = 0.01; // 1/100 probability
        double probNetworkFailure = 0.01; // 1/100 probability
        double probTransactionFailsGivenCache = 5.0 / 6.0;
        double probTransactionFailsGivenNet = 0.25;
        
        double totalProb = (probTransactionFailsGivenCache * probCacheFailure) + 
                           (probTransactionFailsGivenNet * probNetworkFailure);
        System.out.println("Total probability of transaction failing: " + totalProb);
    }
}
```
x??

---

#### Bayes' Law

:p How can Bayes' law be used to find the reverse conditional probability?
??x
Bayes' law allows us to calculate \( P(F|E) \) given \( P(E|F) \), and other relevant probabilities. It is essential for updating beliefs based on new evidence.
```java
// Pseudo-code for applying Bayes' Law
public class BayesLaw {
    public static void main(String[] args) {
        double probCacheFailure = 0.01; // Probability of cache failure
        double probNetworkFailure = 0.01; // Probability of network failure
        double probTransactionFailsGivenCache = 5.0 / 6.0;
        double probTransactionFailsGivenNet = 0.25;
        
        double totalProbTransactionFails = (probTransactionFailsGivenCache * probCacheFailure) + 
                                           (probTransactionFailsGivenNet * probNetworkFailure);
        
        double probCacheGivenFail = (probTransactionFailsGivenCache * probCacheFailure) / totalProbTransactionFails;
        System.out.println("Probability of cache failure given transaction fails: " + probCacheGivenFail);
    }
}
```
x??

---

#### Extended Bayes Law

Background context explaining the theorem and its application. The formula given is an extension of Bayes' law for multiple events that partition the sample space.

Formula: 
\[ P\{F|E\} = \frac{P\{E|F\} \cdot P\{F\}}{\sum_{j=1}^{n} P\{E|F_j\} \cdot P\{F_j\}} \]

:p What does the Extended Bayes Law state and how is it used?
??x
The Extended Bayes Law extends the basic form of Bayes' theorem to multiple events that partition the sample space. It helps in calculating the probability of an event \( F \) given another event \( E \), when there are multiple possible conditions (events) that could influence the outcome.

Example problem: Given a test with 95% accuracy for a rare disease, and only 1 in 10,000 children having the disease. If the test comes back positive, we want to find the probability that the child has the disease.
```java
// P{Disease} = 1 / 10000
double P_Disease = 1 / 10000.0;

// P{Test Positive|Disease} = 0.95
double P_TestPositive_given_Disease = 0.95;

// P{Healthy} = 1 - P{Disease}
double P_Healthy = 1 - P_Disease;

// P{Test Positive|Healthy} = 0.05
double P_TestPositive_given_Healthy = 0.05;

// Calculate the probability using Extended Bayes Law
double P_PositiveGivenDisease = (P_TestPositive_given_Disease * P_Disease) / 
                                (P_TestPositive_given_Disease * P_Disease + 
                                 P_TestPositive_given_Healthy * P_Healthy);

System.out.println("Probability of having the disease given a positive test: " + P_PositiveGivenDisease);
```
x??

---

#### Discrete Random Variables

Background context explaining discrete random variables and their properties. A discrete random variable can take on at most a countably infinite number of values.

:p What is a discrete random variable, and how does it differ from a continuous one?
??x
A discrete random variable (r.v.) is a function that maps the outcome of an experiment to real numbers. It takes on only a finite or countably infinite number of distinct values. The key difference from a continuous r.v. is that in a continuous setting, there are uncountably many possible values.

Example: Rolling two dice and calculating the sum.
```java
// Define the possible outcomes for rolling two dice
int[] outcomes = {2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};

// Calculate probabilities
double P_XEquals3 = 2 / 36.0; // P{X=3}
System.out.println("Probability of getting a sum of 3: " + P_XEquals3);
```
x??

---

#### Continuous Random Variables

Background context explaining continuous random variables and their properties. A continuous r.v. can take on any value in an interval or on the real line.

:p What is a continuous random variable, and provide examples?
??x
A continuous random variable (r.v.) can take on any value within a given range or over the entire set of real numbers. Unlike discrete r.v.s, which have a finite or countably infinite number of values, continuous r.v.s have an uncountable number of possible outcomes.

Examples:
1. The time until the next arrival at a website.
2. The CPU requirement of an HTTP request.

These quantities are modeled as continuous because they can take on any value within a range (e.g., from 0 to infinity).

```java
// Example: Time between arrivals at a website is a continuous r.v.
double time = Math.random(); // Randomly generated time between 0 and 1

System.out.println("Generated time interval: " + time);
```
x??

---

#### Sum of Rolls on Two Dice

Background context explaining the concept of summing outcomes from rolling dice.

:p Which random variable is discrete, the sum of two dice rolls or a continuous one?
??x
The sum of the rolls of two dice is a discrete r.v. because it can only take on specific, finite values (2 through 12). Each value has a certain probability associated with it.

```java
// Example code to generate and calculate probabilities for sums of dice
int[] possibleSums = {2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
double P_XEquals7 = 6 / 36.0; // P{X=7}

System.out.println("Probability of getting a sum of 7: " + P_XEquals7);
```
x??

---

#### Number of Website Arrivals

Background context explaining the concept of counting events over time.

:p Which random variable is discrete, the number of arrivals at a website or a continuous one?
??x
The number of arrivals at a website by time \( t \) is a discrete r.v. because it can only take on integer values (0, 1, 2, ...). This is a countable set of possible outcomes.

```java
// Example code to simulate the number of arrivals
int[] arrivals = {0, 1, 2, 3, ...}; // Simulated arrivals

System.out.println("Number of simulated arrivals: " + arrivals.length);
```
x??

---

#### Time Until Next Arrival

Background context explaining time-based events and their modeling.

:p Which random variable is continuous, the time until the next arrival or a discrete one?
??x
The time until the next arrival at a website is a continuous r.v. because it can take on any non-negative real value (0 to infinity). This represents an uncountable set of possible outcomes.

```java
// Example code to generate and simulate waiting times
double waitingTime = Math.random(); // Randomly generated time from 0 to 1

System.out.println("Generated waiting time: " + waitingTime);
```
x??

---

#### CPU Requirement of HTTP Request

Background context explaining resource requirements in a continuous manner.

:p Which random variable is continuous, the CPU requirement of an HTTP request or a discrete one?
??x
The CPU requirement of an HTTP request is a continuous r.v. because it can take on any real value within a range (e.g., from 0 to infinity). This represents an uncountable set of possible outcomes.

```java
// Example code to generate and simulate CPU requirements
double cpuRequirement = Math.random(); // Randomly generated CPU requirement

System.out.println("Generated CPU requirement: " + cpuRequirement);
```
x??

#### Bernoulli Distribution
Background context: The Bernoulli distribution models a single binary outcome (success or failure) with probabilities \( p \) and \( 1-p \). It is often used to model events that have only two possible outcomes, such as a coin flip.

The probability mass function (p.m.f.) of the Bernoulli random variable \( X \) is given by:
\[ p_X(0) = 1 - p \]
\[ p_X(1) = p \]

:p What does the Bernoulli distribution represent?
??x
The Bernoulli distribution represents a single binary outcome, such as whether a coin flip results in heads (success with probability \( p \)) or tails (failure with probability \( 1-p \)).
x??

---

#### Binomial Distribution
Background context: The binomial distribution models the number of successes in a fixed number of independent Bernoulli trials. Each trial has two possible outcomes, success or failure, and is characterized by its own probability of success.

The p.m.f. of the binomial random variable \( X \) (number of successes in \( n \) trials) is given by:
\[ p_X(i) = P(X=i) = \binom{n}{i} p^i (1-p)^{n-i} \]
where \( i = 0, 1, 2, ..., n \).

:p How does the binomial distribution differ from the Bernoulli distribution?
??x
The binomial distribution differs from the Bernoulli distribution in that it models the number of successes in a fixed number of independent trials, whereas the Bernoulli distribution models a single trial.
x??

---

#### Geometric Distribution
Background context: The geometric distribution models the number of trials needed to get the first success in a sequence of independent Bernoulli trials. Each trial has two possible outcomes, success or failure, and is characterized by its own probability of success.

The p.m.f. of the geometric random variable \( X \) (number of trials until the first success) is given by:
\[ p_X(i) = P(X=i) = (1-p)^{i-1} p \]
where \( i = 1, 2, 3, ... \).

:p What does the geometric distribution represent?
??x
The geometric distribution represents the number of trials needed to get the first success in a sequence of independent Bernoulli trials.
x??

---

#### Poisson Distribution
Background context: The Poisson distribution is used to model the number of events occurring in a fixed interval of time or space, given that these events occur with a known constant mean rate \( \lambda \).

The p.m.f. of the Poisson random variable \( X \) (number of events) is given by:
\[ P(X=k) = \frac{\lambda^k e^{-\lambda}}{k!} \]
where \( k = 0, 1, 2, ... \).

:p How does the Poisson distribution differ from the binomial and geometric distributions?
??x
The Poisson distribution differs from the binomial and geometric distributions in that it models the number of events occurring in a fixed interval, whereas the binomial distribution models the number of successes in a fixed number of trials and the geometric distribution models the number of trials until the first success.
x??

---

#### Discrete Random Variables
Background context: Discrete random variables take on countable values. The probability mass function (p.m.f.) describes the probabilities associated with each value.

The p.m.f. \( p_X(a) \) is defined as:
\[ p_X(a) = P(X=a) \]

The cumulative distribution function (c.d.f.) \( F_X(a) \) is defined as:
\[ F_X(a) = P(X \leq a) = \sum_{x \leq a} p_X(x) \]
and
\[ F_X(a) = P(X > a) = \sum_{x > a} p_X(x) = 1 - F_X(a) \]

:p What is the probability mass function (p.m.f.) of a discrete random variable?
??x
The probability mass function (p.m.f.) of a discrete random variable \( X \) is given by:
\[ p_X(a) = P(X=a) \]
This function describes the probabilities associated with each value that the random variable can take.
x??

---

#### Cumulative Distribution Function (c.d.f.)
Background context: The cumulative distribution function (c.d.f.) of a random variable \( X \), denoted by \( F_X(a) \), is defined as:
\[ F_X(a) = P(X \leq a) = \sum_{x \leq a} p_X(x) \]
It provides the probability that the random variable takes on a value less than or equal to \( a \).

:p What does the cumulative distribution function (c.d.f.) represent?
??x
The cumulative distribution function (c.d.f.) of a random variable \( X \), denoted by \( F_X(a) \), represents the probability that the random variable takes on a value less than or equal to \( a \). It is defined as:
\[ F_X(a) = P(X \leq a) = \sum_{x \leq a} p_X(x) \]
x??

---

#### Poisson Distribution
The Poisson distribution models the number of events occurring within a fixed interval, given that these events occur independently and at a constant average rate. The probability mass function (pmf) for a Poisson random variable \( X \sim \text{Poisson}(\lambda) \) is:
\[ p_X(i) = e^{-\lambda} \frac{\lambda^i}{i!}, \quad i = 0, 1, 2, ... \]

The distribution often approximates the number of arrivals to a website or a router per unit time if \( n \) (the number of sources) is large and \( p \) (individual probability) is small.

:p What does the Poisson distribution model?
??x
The Poisson distribution models the number of events occurring in a fixed interval when these events occur independently at a constant average rate.
x??

---

#### Binomial Distribution Approximation to Poisson
When \( n \) is large and \( p \) is small, the binomial distribution can be approximated by the Poisson distribution with parameter \( \lambda = np \). This approximation becomes more accurate as both \( n \) increases and \( p \) decreases.

:p How does a Binomial distribution approximate a Poisson distribution?
??x
A Binomial distribution can approximate a Poisson distribution when the number of trials \( n \) is large and the probability of success \( p \) in each trial is small, with the parameter for the Poisson distribution being \( \lambda = np \).
x??

---

#### Continuous Random Variables and Probability Density Functions (p.d.f.)
Continuous random variables take on an uncountable number of values. The cumulative distribution function (c.d.f.) \( F_X(a) \) for a continuous r.v. is defined as:
\[ F_X(a) = P\{-\infty < X \leq a\} = \int_{-\infty}^{a} f_X(x) dx \]

The probability density function (p.d.f.) \( f_X(x) \) must satisfy the following conditions:
1. Non-negativity: \( f_X(x) \geq 0 \)
2. Total area under the curve is 1: \( \int_{-\infty}^{\infty} f_X(x) dx = 1 \)

:p What defines a valid probability density function (p.d.f.) for a continuous random variable?
??x
A valid p.d.f. for a continuous random variable must be non-negative and its total area under the curve must equal 1.
x??

---

#### Uniform Distribution
The uniform distribution \( U(a, b) \) models that any interval of length \( \delta \) between \( a \) and \( b \) is equally likely.

For \( X \sim U(a, b) \), the p.d.f. is:
\[ f_X(x) = \begin{cases} 
\frac{1}{b-a} & \text{if } a \leq x \leq b \\
0 & \text{otherwise}
\end{cases} \]

The c.d.f. is:
\[ F_X(x) = \int_{a}^{x} f(t) dt = \begin{cases} 
0 & \text{if } x < a \\
\frac{x-a}{b-a} & \text{if } a \leq x \leq b \\
1 & \text{if } x > b
\end{cases} \]

:p What is the probability density function (p.d.f.) of the uniform distribution?
??x
The p.d.f. of the uniform distribution \( U(a, b) \) is:
\[ f_X(x) = \begin{cases} 
\frac{1}{b-a} & \text{if } a \leq x \leq b \\
0 & \text{otherwise}
\end{cases} \]
x??

---

#### Exponential Distribution
The exponential distribution \( Exp(\lambda) \) models the time between events in a Poisson process. The p.d.f. is:
\[ f_X(x) = \begin{cases} 
\lambda e^{-\lambda x} & \text{if } x \geq 0 \\
0 & \text{otherwise}
\end{cases} \]

The c.d.f. is:
\[ F_X(x) = P\{X \leq x\} = \int_{-\infty}^{x} f(t) dt = \begin{cases} 
0 & \text{if } x < 0 \\
1 - e^{-\lambda x} & \text{if } x \geq 0
\end{cases} \]

:p What is the probability density function (p.d.f.) of the exponential distribution?
??x
The p.d.f. of the exponential distribution \( Exp(\lambda) \) is:
\[ f_X(x) = \begin{cases} 
\lambda e^{-\lambda x} & \text{if } x \geq 0 \\
0 & \text{otherwise}
\end{cases} \]
x??

---

#### Pareto Distribution
The Pareto distribution models the distribution of quantities where a relatively small number of items account for most of the value. The p.d.f. is:
\[ f_X(x) = \begin{cases} 
\alpha x^{-\alpha-1} & \text{if } x \geq 1 \\
0 & \text{otherwise}
\end{cases} \]

The c.d.f. is:
\[ F_X(x) = P\{X \leq x\} = \int_{-\infty}^{x} f(t) dt = \begin{cases} 
0 & \text{if } x < 1 \\
1 - x^{-\alpha} & \text{if } x \geq 1
\end{cases} \]

:p What is the probability density function (p.d.f.) of the Pareto distribution?
??x
The p.d.f. of the Pareto distribution is:
\[ f_X(x) = \begin{cases} 
\alpha x^{-\alpha-1} & \text{if } x \geq 1 \\
0 & \text{otherwise}
\end{cases} \]
x??

---

#### Expectation of a Random Variable (Discrete Case)
Background context: The expectation or mean of a discrete random variable \(X\) is calculated by summing each possible value weighted by its probability. This can be mathematically represented as:

\[ E[X] = \sum x \cdot P\{X=x\} \]

For example, if we consider the cost of lunch over a week:
- Monday: $7
- Tuesday: $7
- Wednesday: $5
- Thursday: $5
- Friday: $5
- Saturday: $0
- Sunday: $2

The average cost is calculated as follows:

\[ \text{Avg} = \frac{1}{7}(7 + 7 + 5 + 5 + 5 + 0 + 2) \]

:p What is the formula for calculating the expectation of a discrete random variable?
??x
The formula for calculating the expectation of a discrete random variable \(X\) is:
\[ E[X] = \sum x \cdot P\{X=x\} \]
This means you sum each possible value \(x\) of the random variable, multiplied by its corresponding probability \(P\{X=x\}\).
x??

---

#### Expectation of a Random Variable (Continuous Case)
Background context: The expectation or mean of a continuous random variable \(X\) is calculated by integrating over all possible values. This can be mathematically represented as:

\[ E[X] = \int_{-\infty}^{\infty} x \cdot f_X(x) \, dx \]

Where \(f_X(x)\) is the probability density function of \(X\).

:p What is the formula for calculating the expectation of a continuous random variable?
??x
The formula for calculating the expectation of a continuous random variable \(X\) is:
\[ E[X] = \int_{-\infty}^{\infty} x \cdot f_X(x) \, dx \]
This involves integrating each possible value \(x\) weighted by its probability density function \(f_X(x)\).
x??

---

#### Expectation of a Bernoulli Random Variable
Background context: A Bernoulli random variable is a discrete random variable that takes the value 1 with probability \(p\) and the value 0 with probability \(1-p\).

The expectation of a Bernoulli random variable \(X \sim \text{Bernoulli}(p)\) can be calculated as:

\[ E[X] = 0 \cdot (1 - p) + 1 \cdot p = p \]

:p What is the expected value of a Bernoulli random variable?
??x
The expected value of a Bernoulli random variable \(X \sim \text{Bernoulli}(p)\) is:
\[ E[X] = 0 \cdot (1 - p) + 1 \cdot p = p \]
This means the expectation is simply the probability parameter \(p\).
x??

---

#### Expectation of a Geometric Random Variable
Background context: A geometric random variable represents the number of trials needed to get the first success in repeated independent Bernoulli trials. The expected value (mean) can be calculated as:

\[ E[X] = \frac{1}{p} \]

Where \(X\) is a geometric random variable with parameter \(p\).

:p What is the expected number of tosses for getting heads if the probability of heads is 1/3?
??x
The expected number of tosses to get heads when the probability of heads is \( \frac{1}{3} \) can be calculated as:
\[ E[X] = \frac{1}{\frac{1}{3}} = 3 \]
This means, on average, it takes 3 tosses to get a head.
x??

---

#### Expectation and Variance of the Poisson Distribution
Background context: The Poisson distribution is used for modeling the number of events occurring in a fixed interval of time or space. For a random variable \(X \sim \text{Poisson}(\lambda)\), the expectation (mean) is equal to the parameter \(\lambda\).

The variance of a Poisson random variable \(X \sim \text{Poisson}(\lambda)\) can be calculated as:

\[ E[X] = \lambda \]

:p What is the expected value of a Poisson random variable?
??x
The expected value (mean) of a Poisson random variable \(X \sim \text{Poisson}(\lambda)\) is:
\[ E[X] = \lambda \]
This means that both the mean and variance of the Poisson distribution are equal to the parameter \(\lambda\).
x??

---

#### Expectation and Variance of the Exponential Distribution
Background context: The exponential distribution models the time between events in a Poisson process. For a random variable \(X \sim \text{Exponential}(\lambda)\), the expectation (mean) is given by:

\[ E[X] = \frac{1}{\lambda} \]

The variance of an Exponential random variable can be calculated as:

\[ Var(X) = \left( \frac{1}{\lambda} \right)^2 = \frac{1}{\lambda^2} \]

:p What is the expected value of an exponentially distributed random variable?
??x
The expected value (mean) of an Exponential random variable \(X \sim \text{Exponential}(\lambda)\) is:
\[ E[X] = \frac{1}{\lambda} \]
This means that if the rate parameter \(\lambda\) is 3 arrivals per second, the expected time until the next arrival is \(\frac{1}{3}\) seconds.
x??

---

#### Higher Moments and Expectation of a Function
Background context: The \(i\)-th moment of a random variable \(X\) can be defined as:
- Discrete case: \( E[X^i] = \sum x^i \cdot P\{X=x\} \)
- Continuous case: \( E[X^i] = \int_{-\infty}^{\infty} x^i \cdot f_X(x) \, dx \)

The expectation of a function \(g(X)\) can be calculated as:
\[ E[g(X)] = \sum g(x) \cdot P\{X=x\} \text{ (for discrete)} \]
\[ E[g(X)] = \int_{-\infty}^{\infty} g(x) \cdot f_X(x) \, dx \text{ (for continuous)} \]

:p What is the expectation of a function \(g(X)\)?
??x
The expectation of a function \(g(X)\) can be calculated as:
\[ E[g(X)] = \sum g(x) \cdot P\{X=x\} \text{ (for discrete)} \]
\[ E[g(X)] = \int_{-\infty}^{\infty} g(x) \cdot f_X(x) \, dx \text{ (for continuous)} \]
This involves summing or integrating the function \(g(x)\) weighted by the probability mass or density function of \(X\).
x??

---

#### Variance of a Random Variable
Background context: The variance of a random variable \(X\) measures how much \(X\) varies from its mean. It is defined as:
\[ Var(X) = E[(X - E[X])^2] \]
And can be equivalently expressed as:
\[ Var(X) = E[X^2] - (E[X])^2 \]

:p What is the definition of variance for a random variable?
??x
The variance of a random variable \(X\) is defined as:
\[ Var(X) = E[(X - E[X])^2] \]
This measures how much the values of \(X\) vary from its mean. It can also be expressed equivalently as:
\[ Var(X) = E[X^2] - (E[X])^2 \]
x??

---

