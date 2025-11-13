# Flashcards: 7A003--Performance-modeling-and-design-of-computer-systems_processed (Part 6)

**Starting Chapter:** Chapter 4 Generating Random Variables for Simulation. 4.1 Inverse-Transform Method

---

#### Inverse-Transform Method Overview
Background context: The inverse-transform method is a technique for generating random variables (RVs) from any given cumulative distribution function (CDF). It assumes that we can invert the CDF to map uniform RVs to the desired distribution. This method is useful in simulations where different types of distributions are needed.

:p Can you explain the basic idea behind the inverse-transform method?
??x
The goal is to use a uniform random variable $u $ from$U(0,1)$ and transform it into an instance $ x $ of the desired distribution with CDF $F_X(x)$. This transformation is based on the property that for any value $ x$, the probability of outputting a value in the interval $(0,x)$ should be equal to $P(X \leq x) = F_X(x)$.
x??

---

#### Continuous Case Mapping
Background context: For continuous distributions, we map the uniform random variable $u \in U(0,1)$ to a value $ x $ such that the probability of outputting a value in the interval $(0,x)$ is equal to the CDF evaluated at $ x $, i.e.,$ F_X(x) = P(X \leq x)$.

:p How does the mapping work for continuous distributions?
??x
The mapping $g^{-1}(·)$ takes each instance of a uniform random variable $u \in U(0,1)$ and maps it to a unique value $ x $. This relationship is given by $ u = F_X(x)$, which can be rearranged to find $ x = F_X^{-1}(u)$.
x??

---

#### Example: Exponential Distribution
Background context: The exponential distribution is commonly used for modeling the time between events in a Poisson process. Its CDF is given by $F_X(x) = 1 - e^{-\lambda x}$. We can use the inverse-transform method to generate samples from this distribution.

:p How do you generate an Exponential random variable using the inverse-transform method?
??x
To generate an Exponential random variable with parameter $\lambda $, we need to find the value of $ x $ such that $ F_X(x) = u $. Given$ F_X(x) = 1 - e^{-\lambda x}$, we solve for $ x$:

$$1 - e^{-\lambda x} = u$$
$$e^{-\lambda x} = 1 - u$$
$$-\lambda x = \ln(1 - u)$$
$$x = -\frac{1}{\lambda} \ln(1 - u)$$

Given $u \in U(0,1)$, setting $ x = -\frac{1}{\lambda} \ln(1 - u)$produces an instance of $ X \sim \text{Exp}(\lambda)$.
x??

---

#### Pseudocode for Inverse-Transform Method
Background context: The pseudocode illustrates the process step-by-step, making it easier to implement in programming languages.

:p Provide the pseudocode for generating a random variable using the inverse-transform method.
??x
```java
// Pseudocode for Generating an Exponential RV Using Inverse-Transform Method
function generateExponentialRV(double lambda) {
    // Generate a uniform random number u from U(0,1)
    double u = Math.random();  // or equivalent function in your programming language

    // Compute the inverse CDF to get x
    double x = -Math.log(1 - u) / lambda;

    return x;
}
```
x??

---

#### CDF of Exponential Distribution
Background context: The CDF of an exponential distribution is $F_X(x) = 1 - e^{-\lambda x}$.

:p What is the CDF for an exponential distribution with parameter $\lambda$?
??x
The CDF for an exponential distribution with parameter $\lambda$ is given by:

$$F_X(x) = 1 - e^{-\lambda x}$$x??

---

#### Inverse of Exponential CDF
Background context: The inverse of the CDF for an exponential distribution can be derived to find the value of $x $ that corresponds to a given uniform random variable$u$.

:p What is the inverse of the CDF for an exponential distribution?
??x
The inverse of the CDF for an exponential distribution with parameter $\lambda$ is:

$$F_X^{-1}(u) = -\frac{1}{\lambda} \ln(1 - u)$$x??

---

#### Uniform Random Variable Generation
Background context: Most operating systems provide a generator for uniform random variables in the interval $[0, 1]$. In some cases, they generate integers and then map them to this interval.

:p How do you typically generate a uniform random variable between 0 and 1?
??x
Most operating systems have built-in functions that generate uniformly distributed random numbers between 0 and 1. For example, in Java, `Math.random()` generates such a number. If the system provides only integer generation, it can be scaled down to fit within $[0, 1]$ by dividing by the maximum possible value (e.g.,$ N = 2^{32} - 1$).
x??

---

#### Discrete Random Variable Generation Using Cumulative Distribution Function (CDF)
Background context: In the discrete case, we generate a random variable $X $ that takes on values$x_0, x_1, \ldots, x_k $ with probabilities$ p_0, p_1, \ldots, p_k $. The cumulative distribution function (CDF) of $ X$is given by:
$$F_X(x) = \sum_{i=0}^{\lfloor x \rfloor} p_i$$where $\lfloor x \rfloor $ denotes the largest integer less than or equal to$x$.
:p How do we generate a discrete random variable using its CDF?
??x
To generate a discrete random variable using its CDF, follow these steps:
1. Arrange the values of $X $ in ascending order:$ x_0 < x_1 < \ldots < x_k$.
2. Generate a uniform random number $u $ from the interval$[0, 1]$.
3. If $0 < u \leq p_0 $, return $ x_0$.
4. If $p_0 < u \leq p_0 + p_1 $, return $ x_1$.
5. Continue this process until you find the appropriate interval for $u$.

This method works well if we have closed-form expressions for partial sums of probabilities, but it can be inefficient when dealing with a large number of values.

```java
public class DiscreteRandomVariableGenerator {
    private double[] cumulativeProbabilities;
    private int[] values;

    public DiscreteRandomVariableGenerator(double[] p) {
        this.values = new int[p.length];
        this.cumulativeProbabilities = new double[p.length + 1];

        // Calculate cumulative probabilities and assign values
        for (int i = 0; i < p.length; i++) {
            values[i] = i;
            if (i == 0) {
                cumulativeProbabilities[i] = p[0];
            } else {
                cumulativeProbabilities[i] = cumulativeProbabilities[i - 1] + p[i];
            }
        }
    }

    public int generateRandomVariable() {
        double u = Math.random(); // Generate a uniform random number in [0, 1]
        for (int i = 0; i < cumulativeProbabilities.length - 1; i++) {
            if (u >= cumulativeProbabilities[i] && u <= cumulativeProbabilities[i + 1]) {
                return values[i];
            }
        }
        return values[values.length - 1]; // Should never reach here
    }
}
```
x??

---

#### Discrete Case of Accept-Reject Method
Background context: The Accept-Reject method is a technique to generate random variables from a desired distribution when the cumulative distribution function (CDF) or probability density function (PDF) is not known. It involves generating instances of an auxiliary random variable $Q$ and accepting them with certain probabilities based on the ratio of their respective PDFs.
:p What are the steps in the Discrete Accept-Reject Method?
??x
The steps in the Discrete Accept-Reject Method are as follows:
1. Find a random variable $Q $ such that its probability mass function (PMF)$\{q_j\}$ satisfies $q_j > 0 \Leftrightarrow p_j > 0$.
2. Generate an instance of $Q $, denoted by $ j$.
3. Generate a uniform random number $U $ from the interval$[0, 1)$.
4. If $U < \frac{p_j}{c q_j}$ for some constant $c$ such that $\frac{p_j}{q_j} \leq c$ for all $j$, return $ P = j$. Otherwise, repeat from step 2.

The normalization constant $c$ ensures the acceptance probability is feasible.
x??

---

#### Continuous Case of Accept-Reject Method
Background context: The Accept-Reject method can also be applied to continuous random variables. It involves generating instances of a known auxiliary random variable $Y$ and accepting them based on a ratio involving their PDFs.

:p How do we apply the Accept-Reject Method for Continuous Random Variables?
??x
For the continuous case, the steps in the Accept-Reject Method are as follows:
1. Find a continuous random variable $Y $ such that its probability density function (PDF)$\{f_Y(t)\}$ satisfies $f_Y(t) > 0 \Leftrightarrow f_X(t) > 0$.
2. Determine a constant $c $ such that$\frac{f_X(t)}{f_Y(t)} \leq c $ for all$t $ where$f_X(t) > 0$.
3. Generate an instance of $Y $, denoted by $ t$.
4. With probability $\frac{f_X(t)}{c f_Y(t)}$, return $ X = t$. Otherwise, reject $ t$ and repeat from step 2.

This method is particularly useful when the CDF or PDF of the desired distribution is not easily invertible.
x??

---

#### Example: Generating a Normal Random Variable Using Accept-Reject Method
Background context: The example shows how to use the Accept-Reject Method to generate a normal random variable with mean 0 and variance 1. We use the absolute value of $N$(a standard exponential distribution) and then flip the sign with probability 0.5.

:p How do we apply the Accept-Reject Method to generate a Normal Random Variable?
??x
To apply the Accept-Reject Method to generate a normal random variable $X \sim N(0,1)$, follow these steps:
1. Generate an instance of $Y$ from the exponential distribution with rate 1.
2. Compute $t = |Y|$.
3. With probability $\frac{f_X(t)}{f_Y(t)}$, where $ f_X(t) = \sqrt{\frac{2}{\pi}} e^{-\frac{t^2}{2}}$and $ f_Y(t) = e^{-t}$:
   - Return $X = t $- Otherwise, return $ X = -t$ This method works because the PDF of the normal distribution can be expressed in terms of the exponential distribution.

```java
public class NormalRandomVariableGenerator {
    private double fY(double t) { // Exponential PDF
        return Math.exp(-t);
    }

    private double fX(double t) { // Normal (0,1) PDF
        return Math.sqrt(2.0 / Math.PI) * Math.exp(-0.5 * t * t);
    }

    public double generateNormalRandomVariable() {
        while (true) {
            double Y = -Math.log(Math.random()); // Generate an Exp(1)
            double t = Math.abs(Y);               // Get |Y|
            double u = Math.random();             // Uniform [0, 1]

            if (u < fX(t) / (fY(t) * 1.3)) {     // Acceptance ratio
                return t;                         // Return X
            } else {
                continue;                          // Rejection, try again
            }
        }
    }
}
```
x??

---

#### Poisson Random Variable Generation with Accept-Reject Method
Background context: The Poisson distribution has an infinite number of probabilities and no closed-form CDF. However, it can be generated using the Accept-Reject method by finding a suitable auxiliary distribution.

:p How do we use the Accept-Reject Method to generate a Poisson random variable?
??x
To use the Accept-Reject Method to generate a Poisson random variable with mean $\lambda$, follow these steps:
1. Choose an appropriate auxiliary distribution $Y$ whose PDF is easy to sample from.
2. Sample $Y$ and compute its value.
3. Compute the acceptance probability based on the ratio of the target Poisson PMF $p_i = e^{-\lambda} \frac{\lambda^i}{i!}$ and the chosen auxiliary distribution's PMF.

This method is effective but requires careful selection of the auxiliary distribution to minimize rejection rates.

```java
public class PoissonRandomVariableGenerator {
    private double lambda;

    public PoissonRandomVariableGenerator(double lambda) {
        this.lambda = lambda;
    }

    public int generatePoissonRandomVariable() {
        // Choose a suitable Y (e.g., exponential)
        while (true) {
            double y = -Math.log(Math.random()); // Generate Exp(1)
            double p = Math.exp(-lambda) * Math.pow(lambda, y) / factorial(y); // Poisson PMF
            if (y < lambda && Math.random() < p) { // Acceptance based on ratio
                return (int) y;
            }
        }
    }

    private long factorial(double x) {
        double result = 1.0;
        for (double i = 2; i <= x; i++) {
            result *= i;
        }
        return (long) result;
    }
}
```
x??

--- 

These flashcards cover the key concepts in generating discrete and continuous random variables using Accept-Reject methods, as well as their applications. Each card provides a detailed explanation of the process and relevant code examples where applicable.

#### Generating Random Variables for Poisson Distribution from Exponential Distributions
Background context: The Poisson distribution can be viewed as counting the number of instances of an Exponentially distributed random variable that occur by a fixed time. This provides another method to generate Poisson random variables, involving generating multiple instances of an Exponential random variable.

:p How can we use Exponential distributions to generate Poisson random variables?
??x
To generate Poisson random variables from Exponential distributions, you need to:
1. Generate Exponential random variables with a fixed rate parameter $\lambda$.
2. Count the number of events that occur within a given time interval.
3. This count follows a Poisson distribution.

The rate $\lambda$ is related to the mean arrival rate in a Poisson process, and each Exponential variable represents an interarrival time between events.

Example: If you want to generate a Poisson random variable for arrivals over 10 units of time with a rate $\lambda = 2$, you would:
- Generate multiple Exponential random variables (each representing the interarrival time).
- Count how many such intervals fit within 10 units of time.

This method is based on the memoryless property of the exponential distribution and the relationship between Poisson and Exponential distributions.
x??

---

#### Inverse-Transform Method for Continuous Distributions
Background context: The inverse-transform method is a technique to generate random variables from any continuous probability distribution. It involves transforming uniform random variables into random variables with the desired distribution.

:p How can we use the inverse-transform method to generate values from a given density function?
??x
To use the inverse-transform method for generating values from a continuous distribution with a given density function $f(t)$, follow these steps:
1. Find the cumulative distribution function (CDF) $F(x)$ of the target distribution.
2. Generate a uniform random variable $U \sim U(0, 1)$.
3. Set $X = F^{-1}(U)$. The value $ X$ will have the desired distribution.

Given density function:$f(t) = \frac{5}{4}t - 2 $, where $1 < t < 5$.

To apply this method:
- First, find the CDF of the given density function.
- Then, invert the CDF to get the inverse function $F^{-1}(u)$.
- Generate a uniform random variable $U \sim U(0, 1)$, and use it to compute $ X = F^{-1}(U)$.

Example: For the given density function, the CDF is:
$$F(t) = \int_{1}^{t} \left(\frac{5}{4}s - 2\right) ds$$

After computing the integral, find its inverse.

:p
??x
The CDF $F(t)$ can be computed as follows:
$$F(t) = \int_{1}^{t} \left(\frac{5}{4}s - 2\right) ds = \left[\frac{5}{8}s^2 - 2s\right]_1^t = \frac{5}{8}(t^2 - 1) - 2(t - 1)$$

Simplify the expression:
$$

F(t) = \frac{5}{8}t^2 - \frac{5}{8} - 2t + 2 = \frac{5}{8}t^2 - 2t + \frac{11}{8}$$

To find $F^{-1}(u)$, solve for $ t$in terms of $ u$:
$$u = \frac{5}{8}t^2 - 2t + \frac{11}{8}$$
$$8u = 5t^2 - 16t + 11$$
$$5t^2 - 16t + (11 - 8u) = 0$$

This is a quadratic equation in $t$. Solve it using the quadratic formula:
$$t = \frac{16 \pm \sqrt{(16)^2 - 4 \cdot 5 \cdot (11 - 8u)}}{2 \cdot 5}$$
$$t = \frac{16 \pm \sqrt{256 - 20(11 - 8u)}}{10}$$
$$t = \frac{16 \pm \sqrt{256 - 220 + 160u}}{10}$$
$$t = \frac{16 \pm \sqrt{36 + 160u}}{10}$$
$$t = \frac{16 \pm \sqrt{4(9 + 40u)}}{10}$$
$$t = \frac{16 \pm 2\sqrt{9 + 40u}}{10}$$
$$t = \frac{8 \pm \sqrt{9 + 40u}}{5}$$

Since $t > 1$, we take the positive root:
$$t = \frac{8 + \sqrt{9 + 40u}}{5}$$

This is the inverse function. Now, generate a uniform random variable $U \sim U(0, 1)$ and compute $X = F^{-1}(U)$.

:p
??x
The inverse of the CDF is:
$$t = \frac{8 + \sqrt{9 + 40u}}{5}$$

To generate a value from the distribution with the given density function, follow these steps in code:

```java
public double inverseTransform(double u) {
    return (8 + Math.sqrt(9 + 40 * u)) / 5;
}
```

Use this method to generate values from the specified continuous distribution by generating $U \sim U(0, 1)$ and applying the inverse function.

:p
??x
To generate a value from the given density function using the inverse-transform method:

1. Generate a uniform random variable $U \sim U(0, 1)$.
2. Use the inverse CDF:
$$t = \frac{8 + \sqrt{9 + 40u}}{5}$$

Here is an example in Java:

```java
public class InverseTransformExample {
    public double generateRandomValue() {
        // Generate a uniform random variable U from (0, 1)
        double u = Math.random();
        
        // Apply the inverse CDF to get the value of X
        return (8 + Math.sqrt(9 + 40 * u)) / 5;
    }
}
```

:p
??x
To generate a random value using the inverse-transform method, you would follow these steps in Java:

1. Generate a uniform random variable $U \sim U(0, 1)$.
2. Use the formula:
$$t = \frac{8 + \sqrt{9 + 40u}}{5}$$

Here is an example implementation:

```java
public class InverseTransformExample {
    public double generateRandomValue() {
        // Generate a uniform random variable U from (0, 1)
        double u = Math.random();
        
        // Apply the inverse CDF to get the value of X
        return (8 + Math.sqrt(9 + 40 * u)) / 5;
    }
}
```

This method transforms a uniform random variable into a value that follows the specified distribution.

x??

---

#### Simulating M/M/1 Queue
Background context: An M/M/1 queue is a single-server queueing system where job arrivals follow a Poisson process and service times are exponentially distributed. This problem asks you to simulate such a queue for different load levels (Arrival rates $\lambda$).

:p How do we simulate an M/M/1 queue with different arrival rates?
??x
To simulate an M/M/1 queue, follow these steps:
1. Set up the parameters: 
   - Service rate $\mu = 1 $(since $\frac{1}{\mu} = 1$).
2. Define the interarrival times as Exponential random variables with parameter $\lambda$.
3. Simulate job arrivals and service completions.
4. Track the state of the system: whether it is empty or occupied.

The goal is to measure the mean response time $E[T]$ for different arrival rates $\lambda = 0.5, 0.7,$ and $0.9$.

Example:
- For $\lambda = 0.5$, service times are exponentially distributed with rate 1.
- For each run, start from the empty state and simulate until there are 2000 arrivals.
- Record the response time of the 2001st arrival.

Here is a pseudocode for one "run":
```java
public void simulateMMSim() {
    double lambda = ...; // Set \lambda to one of the specified values
    int totalArrivals = 2000;
    
    List<Double> interarrivalTimes = new ArrayList<>();
    List<Double> serviceTimes = new ArrayList<>();
    
    for (int i = 0; i < totalArrivals; ++i) {
        // Generate Exponential random variable as interarrival time
        double arrivalTime = -Math.log(1 - Math.random()) / lambda;
        
        interarrivalTimes.add(arrivalTime);
    }
    
    int currentServiceStart = 0;
    double currentTime = 0.0;
    int arrivalsServed = 0;
    
    for (int i = 0; i < totalArrivals + 1; ++i) {
        // Increment time by interarrival time
        currentTime += interarrivalTimes.get(i);
        
        if (currentServiceStart == -1 || currentTime > currentServiceStart) {
            // Start new service
            currentServiceStart = currentTime + -Math.log(1 - Math.random()) / 1;
            ++arrivalsServed;
        }
    }
    
    // The response time of the 2001st arrival is recorded here.
}
```

:p
??x
To simulate an M/M/1 queue with different arrival rates, follow these steps:

1. Set up parameters:
   - Service rate $\mu = 1$.
   - Interarrival times are Exponential random variables with parameter $\lambda$.

2. Simulate the system for a given number of arrivals (e.g., 2000):
   - For each arrival, generate an interarrival time.
   - Track the state transitions: either start a new service or continue waiting.

3. Measure the response time of the 2001st arrival after running the simulation multiple times with different $\lambda $ values (e.g.,$\lambda = 0.5, 0.7,$ and $0.9$).

Here is an example pseudocode for one run:

```java
public void simulateMMSim(double lambda) {
    int totalArrivals = 2000;
    
    List<Double> interarrivalTimes = new ArrayList<>();
    for (int i = 0; i < totalArrivals; ++i) {
        // Generate Exponential random variable as interarrival time
        double arrivalTime = -Math.log(1 - Math.random()) / lambda;
        
        interarrivalTimes.add(arrivalTime);
    }
    
    int currentServiceStart = 0;
    double currentTime = 0.0;
    int arrivalsServed = 0;
    
    for (int i = 0; i < totalArrivals + 1; ++i) {
        // Increment time by interarrival time
        currentTime += interarrivalTimes.get(i);
        
        if (currentServiceStart == -1 || currentTime > currentServiceStart) {
            // Start new service
            currentServiceStart = currentTime + -Math.log(1 - Math.random()) / 1;
            ++arrivalsServed;
        }
    }
    
    // The response time of the 2001st arrival is recorded here.
}
```

This pseudocode simulates an M/M/1 queue for a given $\lambda$ value and records the response time of the 2001st arrival.

x??

---

#### Measuring Mean Response Time in M/M/1 Queue
Background context: The goal is to measure the mean response time $E[T]$ for different load levels (Arrival rates $\lambda$) by simulating an M/M/1 queue and averaging over multiple runs.

:p How do we measure the mean response time of the 2001st arrival in an M/M/1 simulation?
??x
To measure the mean response time $E[T]$ for different load levels (Arrival rates $\lambda$), follow these steps:

1. For each value of $\lambda $($0.5, 0.7,$ and $0.9$):
   - Run the simulation multiple times (e.g., $n = 200$ independent runs).
   - Record the response time of the 2001st arrival in each run.
   - Calculate the average response time.

Here is an example pseudocode for measuring the mean response time:

```java
public double measureMeanResponseTime(double lambda) {
    int nRuns = 200;
    List<Double> responseTimes = new ArrayList<>();
    
    for (int i = 0; i < nRuns; ++i) {
        // Simulate the M/M/1 queue and record the response time of the 2001st arrival
        double responseTime = simulateMMSim(lambda);
        
        responseTimes.add(responseTime);
    }
    
    // Calculate the mean response time
    return calculateMean(responseTimes);
}

public double simulateMMSim(double lambda) {
    int totalArrivals = 2000;
    
    List<Double> interarrivalTimes = new ArrayList<>();
    for (int i = 0; i < totalArrivals; ++i) {
        // Generate Exponential random variable as interarrival time
        double arrivalTime = -Math.log(1 - Math.random()) / lambda;
        
        interarrivalTimes.add(arrivalTime);
    }
    
    int currentServiceStart = 0;
    double currentTime = 0.0;
    int arrivalsServed = 0;
    
    for (int i = 0; i < totalArrivals + 1; ++i) {
        // Increment time by interarrival time
        currentTime += interarrivalTimes.get(i);
        
        if (currentServiceStart == -1 || currentTime > currentServiceStart) {
            // Start new service
            currentServiceStart = currentTime + -Math.log(1 - Math.random()) / 1;
            ++arrivalsServed;
        }
    }
    
    return currentTime; // The response time of the 2001st arrival is recorded here.
}

public double calculateMean(List<Double> times) {
    double sum = 0.0;
    for (double t : times) {
        sum += t;
    }
    return sum / times.size();
}
```

:p
??x
To measure the mean response time in an M/M/1 queue simulation, follow these steps:

1. Define a function `measureMeanResponseTime` that takes $\lambda$ as input.
2. For each value of $n = 200$ independent runs:
   - Call the `simulateMMSim` function to simulate the system and record the response time of the 2001st arrival.
3. After all runs, calculate the average response time.

Here is a detailed implementation:

```java
public double measureMeanResponseTime(double lambda) {
    int nRuns = 200;
    List<Double> responseTimes = new ArrayList<>();
    
    for (int i = 0; i < nRuns; ++i) {
        // Simulate the M/M/1 queue and record the response time of the 2001st arrival
        double responseTime = simulateMMSim(lambda);
        
        responseTimes.add(responseTime);
    }
    
    // Calculate the mean response time
    return calculateMean(responseTimes);
}

public double simulateMMSim(double lambda) {
    int totalArrivals = 2000;
    
    List<Double> interarrivalTimes = new ArrayList<>();
    for (int i = 0; i < totalArrivals; ++i) {
        // Generate Exponential random variable as interarrival time
        double arrivalTime = -Math.log(1 - Math.random()) / lambda;
        
        interarrivalTimes.add(arrivalTime);
    }
    
    int currentServiceStart = 0;
    double currentTime = 0.0;
    int arrivalsServed = 0;
    
    for (int i = 0; i < totalArrivals + 1; ++i) {
        // Increment time by interarrival time
        currentTime += interarrivalTimes.get(i);
        
        if (currentServiceStart == -1 || currentTime > currentServiceStart) {
            // Start new service
            currentServiceStart = currentTime + -Math.log(1 - Math.random()) / 1;
            ++arrivalsServed;
        }
    }
    
    return currentTime; // The response time of the 2001st arrival is recorded here.
}

public double calculateMean(List<Double> times) {
    double sum = 0.0;
    for (double t : times) {
        sum += t;
    }
    return sum / times.size();
}
```

This code measures the mean response time by running multiple simulations and averaging the results.

x??

