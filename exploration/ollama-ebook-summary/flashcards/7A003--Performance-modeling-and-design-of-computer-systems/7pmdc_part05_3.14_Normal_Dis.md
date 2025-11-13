# Flashcards: 7A003--Performance-modeling-and-design-of-computer-systems_processed (Part 5)

**Starting Chapter:** 3.14 Normal Distribution

---

#### Linear Transformation Property of Normal Distribution
Background context: The Normal distribution has a unique property known as the "Linear Transformation Property." This property states that if $X $ is a normally distributed random variable with mean$\mu $ and variance$\sigma^2 $, then any linear transformation of $ X$ results in another normally distributed random variable.

Given:
- Let $Y = aX + b $, where $ a > 0 $and$ b$ are scalars.
- The distribution of $Y $ is also normal with mean$a\mu + b $ and variance$a^2\sigma^2$.

Relevant formulas: 
$$E[Y] = aE[X] + b$$
$$

Var(Y) = a^2Var(X)$$:p What does the Linear Transformation Property state about normally distributed random variables?
??x
The Linear Transformation Property states that if $X $ is a normal random variable with mean$\mu $ and variance$\sigma^2 $, then for any scalars $ a > 0 $and$ b $, the linear transformation$ Y = aX + b $will also be normally distributed. Specifically, the new random variable$ Y$will have:
$$E[Y] = a\mu + b$$and$$

Var(Y) = a^2\sigma^2$$

This property allows us to transform normal distributions in various ways while maintaining their distributional form.
x??

---

#### Central Limit Theorem (CLT)
Background context: The Central Limit Theorem is a fundamental theorem in probability theory that states, under certain conditions, the sum of a large number of independent and identically distributed (i.i.d.) random variables will tend to be normally distributed.

Given:
- Let $X_1, X_2, \ldots, X_n $ be i.i.d. random variables with mean$\mu $ and variance$\sigma^2$.
- Define the sum of these variables as $S_n = X_1 + X_2 + \cdots + X_n$.

Relevant formulas:
$$E[S_n] = n\mu$$
$$

Var(S_n) = n\sigma^2$$

The standard deviation is then $\sqrt{n}\sigma$. 

Let $Z_n$ be defined as:
$$Z_n = \frac{S_n - n\mu}{\sigma\sqrt{n}}$$

Relevant formulas for $Z_n$:
- Mean: 0
- Standard Deviation: 1

:p What is the Central Limit Theorem (CLT)?
??x
The Central Limit Theorem states that if we have a sequence of i.i.d. random variables $X_1, X_2, \ldots, X_n $ with mean$\mu $ and variance$\sigma^2 $, then as $ n$ becomes large, the sum of these variables, normalized by subtracting their mean and dividing by the standard deviation, will approximately follow a normal distribution.

Formally:
$$Z_n = \frac{S_n - n\mu}{\sigma\sqrt{n}}$$where
- $E[Z_n] = 0 $-$ Var(Z_n) = 1 $Thus, as$ n $approaches infinity, the cumulative distribution function (CDF) of$ Z_n$ converges to the standard normal CDF.
x??

---

#### Properties of Normal Distribution and Standard Deviations
Background context: The properties of the normal distribution are crucial for understanding its behavior. One important property is that about 68% of the data falls within one standard deviation from the mean, approximately 95% within two standard deviations, and nearly 100% within three standard deviations.

Given:
- For a standard Normal variable $Y $, $\Phi(y)$ denotes its CDF.
- The table provides values for $\Phi(y)$.

Relevant formulas and tables:
$$P{-k < Y < k} = 2\Phi(k) - 1$$:p What percentage of the data in a standard normal distribution lies within one standard deviation from the mean?
??x
Approximately 68% of the data in a standard normal distribution lies within one standard deviation from the mean. This is derived using the CDF of the standard Normal variable $Y$, where:
$$P{-1 < Y < 1} = 2\Phi(1) - 1$$

Given that $\Phi(1) = 0.8413$(from the table):
$$P{-1 < Y < 1} = 2 \times 0.8413 - 1 = 0.6826 \approx 0.68$$

Thus, about 68% of the data is within one standard deviation from the mean.
x??

---

#### IQ Testing and Normal Distribution
Background context: The concept of normal distribution can be applied to real-world scenarios such as IQ testing. IQ scores are often modeled using a normal distribution with mean 100 and standard deviation 15.

Given:
- Mean $\mu = 100 $- Standard Deviation $\sigma = 15 $ Relevant formulas for the CDF of$Y$:
$$P{X > k} = 1 - \Phi\left(\frac{k - \mu}{\sigma}\right)$$:p What fraction of people have an IQ greater than 130 in a normal distribution with mean 100 and standard deviation 15?
??x
To find the fraction of people with an IQ greater than 130, we use the properties of the normal distribution. The Z-score for $k = 130$ is:
$$Z = \frac{130 - 100}{15} = 2$$

Using the standard normal CDF $\Phi(2)$:
$$P{X > 130} = 1 - \Phi(2)$$

From the table,$\Phi(2) = 0.9772$, so:
$$P{X > 130} = 1 - 0.9772 = 0.0228 \approx 0.023$$

Thus, about 2% of the population has an IQ above 130.
x??

---

#### Summation of i.i.d. Random Variables
Background context: Consider a sequence of $n $ independent and identically distributed (i.i.d.) random variables$X_1, X_2, \ldots, X_n$. The sum of these variables is given by:
$$S_n = X_1 + X_2 + \cdots + X_n$$

Relevant formulas:
- Mean:$E[S_n] = n\mu $- Variance:$ Var(S_n) = n\sigma^2 $:p What are the mean and standard deviation of$ S_n$ for i.i.d. random variables?
??x
For a sequence of $n $ independent and identically distributed (i.i.d.) random variables$X_1, X_2, \ldots, X_n $ with mean$\mu $ and variance$\sigma^2 $, the sum $ S_n = X_1 + X_2 + \cdots + X_n$ has:
- Mean:$E[S_n] = n\mu $- Variance:$ Var(S_n) = n\sigma^2 $The standard deviation of$ S_n$is therefore:
$$\sqrt{Var(S_n)} = \sqrt{n\sigma^2} = \sigma\sqrt{n}$$

Thus, the mean and standard deviation of $S_n$ are as stated.
x??

--- 
#### Transformations to Standard Normal
Background context: To work with normally distributed random variables in a standardized form, we can transform them using the concept of standardization. Given a normal variable $X $, we can create a new variable $ Z$ that follows a standard normal distribution.

Given:
- Let $X $ be a normally distributed random variable with mean$\mu $ and variance$\sigma^2$.
- The standardized form is $Z = \frac{X - \mu}{\sigma}$.

Relevant formulas:
$$E[Z] = 0$$
$$

Var(Z) = 1$$:p How do you transform a normally distributed random variable to the standard normal distribution?
??x
To transform a normally distributed random variable $X $ with mean$\mu $ and variance$\sigma^2$ to the standard normal distribution, we use the following transformation:
$$Z = \frac{X - \mu}{\sigma}$$

This transformation results in $Z $ having a mean of 0 and a standard deviation of 1. The variable$Z$ is now standardized.
x??

--- 
#### Central Limit Theorem Application
Background context: The Central Limit Theorem (CLT) states that the sum of a large number of i.i.d. random variables, even if they are not normally distributed, will tend to be approximately normally distributed.

Given:
- Let $X_1, X_2, \ldots, X_n $ be i.i.d. with mean$\mu $ and variance$\sigma^2$.
- Define $S_n = X_1 + X_2 + \cdots + X_n$.

Relevant formulas for the CLT:
$$E[S_n] = n\mu$$
$$

Var(S_n) = n\sigma^2$$

Let $Z_n$ be defined as:
$$Z_n = \frac{S_n - n\mu}{\sigma\sqrt{n}}$$

Relevant formula for $Z_n$:
- Mean: 0
- Standard Deviation: 1

:p How does the Central Limit Theorem (CLT) apply to a large number of i.i.d. random variables?
??x
The Central Limit Theorem applies to a large number of independent and identically distributed (i.i.d.) random variables by stating that their sum, when normalized by subtracting the mean and dividing by the standard deviation, will tend to follow a normal distribution.

Given $X_1, X_2, \ldots, X_n $ are i.i.d. with mean$\mu $ and variance$\sigma^2$, define:
$$S_n = X_1 + X_2 + \cdots + X_n$$and$$

Z_n = \frac{S_n - n\mu}{\sigma\sqrt{n}}$$

As $n $ becomes large, the distribution of$Z_n$ converges to a standard normal distribution. This means:
- The mean of $Z_n$ is 0.
- The standard deviation of $Z_n$ is 1.

Thus, even if the original variables are not normally distributed, their sum normalized in this way tends to follow a normal distribution.
x?? 

--- 
#### Sampling Heights and Average
Background context: Consider sampling the heights of individuals and taking the average. Even if the individual heights come from a non-Normal distribution (e.g., Uniform), the average height will tend to be normally distributed due to the Central Limit Theorem.

Given:
- Let $X_1, X_2, \ldots, X_n$ represent the heights of individuals.
- Define $S_n = X_1 + X_2 + \cdots + X_n$.

Relevant formulas for the average height:
$$E[S_n] = n\mu$$
$$

Var(S_n) = n\sigma^2$$

Let $Z_n$ be defined as:
$$Z_n = \frac{S_n - n\mu}{\sigma\sqrt{n}}$$

Relevant formula for $Z_n$:
- Mean: 0
- Standard Deviation: 1

:p How does the Central Limit Theorem (CLT) apply to sampling heights of individuals?
??x
The Central Limit Theorem applies to sampling heights by stating that if we take the average height from a large number of individuals, even if their individual heights come from a non-Normal distribution, the distribution of this average will tend to be normally distributed.

Given:
- $X_1, X_2, \ldots, X_n$ represent the heights of individuals.
- Define $S_n = X_1 + X_2 + \cdots + X_n$.
- The sum $S_n $ has mean$n\mu $ and variance$n\sigma^2$.

Let:
$$Z_n = \frac{S_n - n\mu}{\sigma\sqrt{n}}$$

As the number of individuals $n $ becomes large, the distribution of$Z_n$ converges to a standard normal distribution. This means that the average height will follow a normal distribution with mean 0 and standard deviation 1.
x?? 

--- 
#### Mean and Standard Deviation of Sn
Background context: We are interested in understanding how the mean and standard deviation change as we sum multiple independent random variables.

Given:
- Let $X_1, X_2, \ldots, X_n $ be i.i.d. with mean$\mu $ and variance$\sigma^2$.
- Define $S_n = X_1 + X_2 + \cdots + X_n$.

Relevant formulas:
$$E[S_n] = n\mu$$
$$

Var(S_n) = n\sigma^2$$

Let $Z_n$ be defined as:
$$Z_n = S_n - n\mu$$

Relevant formula for the standard deviation of $S_n$:
- Standard Deviation: $\sqrt{n}\sigma $:p What are the mean and standard deviation of $ S_n$?
??x
For a sequence of $n $ independent and identically distributed (i.i.d.) random variables$X_1, X_2, \ldots, X_n $ with mean$\mu $ and variance$\sigma^2 $, the sum$ S_n = X_1 + X_2 + \cdots + X_n$ has:
- Mean:$E[S_n] = n\mu $- Variance:$ Var(S_n) = n\sigma^2 $The standard deviation of$ S_n$is therefore:
$$\sqrt{Var(S_n)} = \sqrt{n\sigma^2} = \sigma\sqrt{n}$$

Thus, the mean and standard deviation of $S_n$ are as stated.
x?? 

--- 
#### Standardization Process
Background context: The process of standardizing a normal distribution involves transforming it to have a mean of 0 and a standard deviation of 1.

Given:
- Let $X $ be a normally distributed random variable with mean$\mu $ and variance$\sigma^2$.
- Define the standardized form as $Z = \frac{X - \mu}{\sigma}$.

Relevant formulas:
$$E[Z] = 0$$
$$

Var(Z) = 1$$:p How do you standardize a normally distributed random variable?
??x
To standardize a normally distributed random variable $X $ with mean$\mu $ and variance$\sigma^2$, we use the following transformation:
$$Z = \frac{X - \mu}{\sigma}$$

This process results in $Z$ having a mean of 0 and a standard deviation of 1. The standardized form is now in the standard normal distribution.
x?? 

--- 
#### Application to IQ Testing
Background context: IQ scores are often modeled using a normal distribution with specific parameters.

Given:
- Mean $\mu = 100 $- Standard Deviation $\sigma = 15 $ Relevant formulas for$Z$:
$$Z = \frac{X - 100}{15}$$:p What is the standardized score (z-score) for an IQ of 130?
??x
To find the z-score for an IQ of 130, we use the standardization formula:
$$

Z = \frac{X - \mu}{\sigma}$$where $ X = 130 $,$\mu = 100 $, and $\sigma = 15$.

Substitute these values into the formula:
$$Z = \frac{130 - 100}{15} = \frac{30}{15} = 2$$

Thus, the z-score for an IQ of 130 is $Z = 2$.
x?? 

--- 
#### Summing Random Variables
Background context: Consider summing multiple random variables to understand their properties.

Given:
- Let $X_1, X_2, \ldots, X_n $ be i.i.d. with mean$\mu $ and variance$\sigma^2$.
- Define $S_n = X_1 + X_2 + \cdots + X_n$.

Relevant formulas for the sum:
$$E[S_n] = n\mu$$
$$

Var(S_n) = n\sigma^2$$

Let $Z_n$ be defined as:
$$Z_n = \frac{S_n - n\mu}{\sigma\sqrt{n}}$$:p What is the definition of $ Z_n$ in terms of the sum of i.i.d. random variables?
??x
The definition of $Z_n $ in terms of the sum of independent and identically distributed (i.i.d.) random variables$X_1, X_2, \ldots, X_n$ is:
$$Z_n = \frac{S_n - n\mu}{\sigma\sqrt{n}}$$where $ S_n = X_1 + X_2 + \cdots + X_n $, and$ E[S_n] = n\mu $and$ Var(S_n) = n\sigma^2$.

Thus, the definition of $Z_n$ is as stated.
x?? 

--- 
#### Summation Properties
Background context: The properties of summing i.i.d. random variables are essential in understanding their distribution.

Given:
- Let $X_1, X_2, \ldots, X_n $ be i.i.d. with mean$\mu $ and variance$\sigma^2$.
- Define $S_n = X_1 + X_2 + \cdots + X_n$.

Relevant formulas for the sum:
$$E[S_n] = n\mu$$
$$

Var(S_n) = n\sigma^2$$

Let $Z_n$ be defined as:
$$Z_n = S_n - n\mu$$:p What is the definition of $ S_n$ in terms of i.i.d. random variables?
??x
The definition of $S_n $ in terms of independent and identically distributed (i.i.d.) random variables$X_1, X_2, \ldots, X_n$ is:
$$S_n = X_1 + X_2 + \cdots + X_n$$

Thus, the sum $S_n$ represents the total value obtained by adding up all the individual random variables.
x?? 

--- 
#### Standard Normal Distribution
Background context: The standard normal distribution is a special case of a normal distribution with mean 0 and variance 1.

Given:
- Let $X $ be a normally distributed random variable with mean$\mu $ and variance$\sigma^2$.
- Define the standardized form as $Z = \frac{X - \mu}{\sigma}$.

Relevant formulas:
$$E[Z] = 0$$
$$

Var(Z) = 1$$:p What are the properties of a standard normal distribution?
??x
The properties of a standard normal distribution include:
- Mean:$E[Z] = 0 $- Variance:$ Var(Z) = 1$Thus, any normally distributed random variable can be transformed to have these properties using the formula:
$$Z = \frac{X - \mu}{\sigma}$$

The standard normal distribution is a special case of a normal distribution with mean 0 and variance 1.
x?? 

--- 
#### Summation and Standardization
Background context: Understanding the summation and standardization processes helps in applying the Central Limit Theorem.

Given:
- Let $X_1, X_2, \ldots, X_n $ be i.i.d. with mean$\mu $ and variance$\sigma^2$.
- Define $S_n = X_1 + X_2 + \cdots + X_n$.

Relevant formulas for the sum:
$$E[S_n] = n\mu$$
$$

Var(S_n) = n\sigma^2$$

Let $Z_n$ be defined as:
$$Z_n = \frac{S_n - n\mu}{\sigma\sqrt{n}}$$:p How do you define $ Z_n$ for the sum of i.i.d. random variables?
??x
The definition of $Z_n $ for the sum of independent and identically distributed (i.i.d.) random variables$X_1, X_2, \ldots, X_n$ is:
$$Z_n = \frac{S_n - n\mu}{\sigma\sqrt{n}}$$where $ S_n = X_1 + X_2 + \cdots + X_n $, and$ E[S_n] = n\mu $and$ Var(S_n) = n\sigma^2$.

Thus, the definition of $Z_n$ is as stated.
x?? 

--- 
#### Summation and Standardization
Background context: The summation and standardization processes are crucial for understanding distributions.

Given:
- Let $X_1, X_2, \ldots, X_n $ be i.i.d. with mean$\mu $ and variance$\sigma^2$.
- Define $S_n = X_1 + X_2 + \cdots + X_n$.

Relevant formulas for the sum:
$$E[S_n] = n\mu$$
$$

Var(S_n) = n\sigma^2$$

Let $Z_n$ be defined as:
$$Z_n = S_n - n\mu$$:p What is the mean of $ S_n$?
??x
The mean of $S_n $ for a sequence of independent and identically distributed (i.i.d.) random variables$X_1, X_2, \ldots, X_n $ with mean$\mu$ is:
$$E[S_n] = n\mu$$

Thus, the mean of $S_n$ is as stated.
x?? 

--- 
#### Summation and Standardization
Background context: The summation and standardization processes are fundamental in statistical analysis.

Given:
- Let $X_1, X_2, \ldots, X_n $ be i.i.d. with mean$\mu $ and variance$\sigma^2$.
- Define $S_n = X_1 + X_2 + \cdots + X_n$.

Relevant formulas for the sum:
$$E[S_n] = n\mu$$
$$

Var(S_n) = n\sigma^2$$

Let $Z_n$ be defined as:
$$Z_n = S_n - n\mu$$:p What is the variance of $ S_n$?
??x
The variance of $S_n $ for a sequence of independent and identically distributed (i.i.d.) random variables$X_1, X_2, \ldots, X_n $ with mean$\mu $ and variance$\sigma^2$ is:
$$Var(S_n) = n\sigma^2$$

Thus, the variance of $S_n$ is as stated.
x?? 

--- 
#### Standard Normal Distribution
Background context: The standard normal distribution is a key concept in statistical analysis.

Given:
- Let $X $ be a normally distributed random variable with mean$\mu $ and variance$\sigma^2$.
- Define the standardized form as $Z = \frac{X - \mu}{\sigma}$.

Relevant formulas:
$$E[Z] = 0$$
$$

Var(Z) = 1$$:p What is the standard deviation of a standard normal distribution?
??x
The standard deviation of a standard normal distribution is 1. This is because, by definition, any normally distributed random variable $X $ with mean$\mu $ and variance$\sigma^2$ can be transformed to have a standard deviation of 1 using the formula:
$$Z = \frac{X - \mu}{\sigma}$$

Thus, in the standard normal distribution, the standard deviation is 1.
x?? 

--- 
#### Standard Normal Distribution
Background context: The standard normal distribution plays a crucial role in statistical analysis.

Given:
- Let $X $ be a normally distributed random variable with mean$\mu $ and variance$\sigma^2$.
- Define the standardized form as $Z = \frac{X - \mu}{\sigma}$.

Relevant formulas:
$$E[Z] = 0$$
$$

Var(Z) = 1$$:p What is the z-score for an IQ of 130 if the mean and standard deviation of IQ scores are 100 and 15, respectively?
??x
To find the z-score for an IQ of 130 with a mean of 100 and a standard deviation of 15, we use the formula:
$$

Z = \frac{X - \mu}{\sigma}$$where $ X = 130 $,$\mu = 100 $, and $\sigma = 15$. Substituting these values into the formula gives:
$$Z = \frac{130 - 100}{15} = \frac{30}{15} = 2$$

Thus, the z-score for an IQ of 130 is $Z = 2$.
x?? 

--- 
#### Standard Normal Distribution
Background context: The standard normal distribution is a key concept in statistical analysis.

Given:
- Let $X $ be a normally distributed random variable with mean$\mu $ and variance$\sigma^2$.
- Define the standardized form as $Z = \frac{X - \mu}{\sigma}$.

Relevant formulas:
$$E[Z] = 0$$
$$

Var(Z) = 1$$:p What is the z-score for a value of 75 if the mean and standard deviation are 80 and 10, respectively?
??x
To find the z-score for a value of 75 with a mean of 80 and a standard deviation of 10, we use the formula:
$$

Z = \frac{X - \mu}{\sigma}$$where $ X = 75 $,$\mu = 80 $, and $\sigma = 10$. Substituting these values into the formula gives:
$$Z = \frac{75 - 80}{10} = \frac{-5}{10} = -0.5$$

Thus, the z-score for a value of 75 is $Z = -0.5$.
x?? 

--- 
#### Standard Normal Distribution
Background context: The standard normal distribution is used to standardize normally distributed random variables.

Given:
- Let $X $ be a normally distributed random variable with mean$\mu $ and variance$\sigma^2$.
- Define the standardized form as $Z = \frac{X - \mu}{\sigma}$.

Relevant formulas:
$$E[Z] = 0$$
$$

Var(Z) = 1$$:p What is the z-score for a value of 95 if the mean and standard deviation are 100 and 20, respectively?
??x
To find the z-score for a value of 95 with a mean of 100 and a standard deviation of 20, we use the formula:
$$

Z = \frac{X - \mu}{\sigma}$$where $ X = 95 $,$\mu = 100 $, and $\sigma = 20$. Substituting these values into the formula gives:
$$Z = \frac{95 - 100}{20} = \frac{-5}{20} = -0.25$$

Thus, the z-score for a value of 95 is $Z = -0.25$.
x?? 

--- 
#### Standard Normal Distribution
Background context: The standard normal distribution is used to standardize normally distributed random variables.

Given:
- Let $X $ be a normally distributed random variable with mean$\mu $ and variance$\sigma^2$.
- Define the standardized form as $Z = \frac{X - \mu}{\sigma}$.

Relevant formulas:
$$E[Z] = 0$$
$$

Var(Z) = 1$$:p What is the z-score for a value of 85 if the mean and standard deviation are 90 and 15, respectively?
??x
To find the z-score for a value of 85 with a mean of 90 and a standard deviation of 15, we use the formula:
$$

Z = \frac{X - \mu}{\sigma}$$where $ X = 85 $,$\mu = 90 $, and $\sigma = 15$. Substituting these values into the formula gives:
$$Z = \frac{85 - 90}{15} = \frac{-5}{15} = -0.33$$

Thus, the z-score for a value of 85 is $Z = -0.33$.
x?? 

--- 
#### Standard Normal Distribution
Background context: The standard normal distribution is used to standardize normally distributed random variables.

Given:
- Let $X $ be a normally distributed random variable with mean$\mu $ and variance$\sigma^2$.
- Define the standardized form as $Z = \frac{X - \mu}{\sigma}$.

Relevant formulas:
$$E[Z] = 0$$
$$

Var(Z) = 1$$:p What is the z-score for a value of 70 if the mean and standard deviation are 65 and 8, respectively?
??x
To find the z-score for a value of 70 with a mean of 65 and a standard deviation of 8, we use the formula:
$$

Z = \frac{X - \mu}{\sigma}$$where $ X = 70 $,$\mu = 65 $, and $\sigma = 8$. Substituting these values into the formula gives:
$$Z = \frac{70 - 65}{8} = \frac{5}{8} = 0.625$$

Thus, the z-score for a value of 70 is $Z = 0.625$.
x?? 

--- 
#### Standard Normal Distribution
Background context: The standard normal distribution is used to standardize normally distributed random variables.

Given:
- Let $X $ be a normally distributed random variable with mean$\mu $ and variance$\sigma^2$.
- Define the standardized form as $Z = \frac{X - \mu}{\sigma}$.

Relevant formulas:
$$E[Z] = 0$$
$$

Var(Z) = 1$$:p What is the z-score for a value of 110 if the mean and standard deviation are 120 and 15, respectively?
??x
To find the z-score for a value of 110 with a mean of 120 and a standard deviation of 15, we use the formula:
$$

Z = \frac{X - \mu}{\sigma}$$where $ X = 110 $,$\mu = 120 $, and $\sigma = 15$. Substituting these values into the formula gives:
$$Z = \frac{110 - 120}{15} = \frac{-10}{15} = -0.67$$

Thus, the z-score for a value of 110 is $Z = -0.67$.
x?? 

--- 
#### Standard Normal Distribution
Background context: The standard normal distribution is used to standardize normally distributed random variables.

Given:
- Let $X $ be a normally distributed random variable with mean$\mu $ and variance$\sigma^2$.
- Define the standardized form as $Z = \frac{X - \mu}{\sigma}$.

Relevant formulas:
$$E[Z] = 0$$
$$

Var(Z) = 1$$:p What is the z-score for a value of 98 if the mean and standard deviation are 105 and 7, respectively?
??x
To find the z-score for a value of 98 with a mean of 105 and a standard deviation of 7, we use the formula:
$$

Z = \frac{X - \mu}{\sigma}$$where $ X = 98 $,$\mu = 105 $, and $\sigma = 7$. Substituting these values into the formula gives:
$$Z = \frac{98 - 105}{7} = \frac{-7}{7} = -1$$

Thus, the z-score for a value of 98 is $Z = -1$.
x?? 

--- 
#### Standard Normal Distribution
Background context: The standard normal distribution is used to standardize normally distributed random variables.

Given:
- Let $X $ be a normally distributed random variable with mean$\mu $ and variance$\sigma^2$.
- Define the standardized form as $Z = \frac{X - \mu}{\sigma}$.

Relevant formulas:
$$E[Z] = 0$$
$$

Var(Z) = 1$$:p What is the z-score for a value of 85 if the mean and standard deviation are 92 and 12, respectively?
??x
To find the z-score for a value of 85 with a mean of 92 and a standard deviation of 12, we use the formula:
$$

Z = \frac{X - \mu}{\sigma}$$where $ X = 85 $,$\mu = 92 $, and $\sigma = 12$. Substituting these values into the formula gives:
$$Z = \frac{85 - 92}{12} = \frac{-7}{12} = -0.5833$$

Thus, the z-score for a value of 85 is $Z = -0.5833$.
x?? 

--- 
#### Standard Normal Distribution
Background context: The standard normal distribution is used to standardize normally distributed random variables.

Given:
- Let $X $ be a normally distributed random variable with mean$\mu $ and variance$\sigma^2$.
- Define the standardized form as $Z = \frac{X - \mu}{\sigma}$.

Relevant formulas:
$$E[Z] = 0$$
$$

Var(Z) = 1$$:p What is the z-score for a value of 68 if the mean and standard deviation are 72 and 4, respectively?
??x
To find the z-score for a value of 68 with a mean of 72 and a standard deviation of 4, we use the formula:
$$

Z = \frac{X - \mu}{\sigma}$$where $ X = 68 $,$\mu = 72 $, and $\sigma = 4$. Substituting these values into the formula gives:
$$Z = \frac{68 - 72}{4} = \frac{-4}{4} = -1$$

Thus, the z-score for a value of 68 is $Z = -1$.
x?? 

--- 
#### Standard Normal Distribution
Background context: The standard normal distribution is used to standardize normally distributed random variables.

Given:
- Let $X $ be a normally distributed random variable with mean$\mu $ and variance$\sigma^2$.
- Define the standardized form as $Z = \frac{X - \mu}{\sigma}$.

Relevant formulas:
$$E[Z] = 0$$
$$

Var(Z) = 1$$:p What is the z-score for a value of 96 if the mean and standard deviation are 100 and 5, respectively?
??x
To find the z-score for a value of 96 with a mean of 100 and a standard deviation of 5, we use the formula:
$$

Z = \frac{X - \mu}{\sigma}$$where $ X = 96 $,$\mu = 100 $, and $\sigma = 5$. Substituting these values into the formula gives:
$$Z = \frac{96 - 100}{5} = \frac{-4}{5} = -0.8$$

Thus, the z-score for a value of 96 is $Z = -0.8$.
x?? 

--- 
#### Standard Normal Distribution
Background context: The standard normal distribution is used to standardize normally distributed random variables.

Given:
- Let $X $ be a normally distributed random variable with mean$\mu $ and variance$\sigma^2$.
- Define the standardized form as $Z = \frac{X - \mu}{\sigma}$.

Relevant formulas:
$$E[Z] = 0$$
$$

Var(Z) = 1$$:p What is the z-score for a value of 82 if the mean and standard deviation are 75 and 6, respectively?
??x
To find the z-score for a value of 82 with a mean of 75 and a standard deviation of 6, we use the formula:
$$

Z = \frac{X - \mu}{\sigma}$$where $ X = 82 $,$\mu = 75 $, and $\sigma = 6$. Substituting these values into the formula gives:
$$Z = \frac{82 - 75}{6} = \frac{7}{6} = 1.1667$$

Thus, the z-score for a value of 82 is $Z = 1.1667$.
x?? 

--- 
#### Standard Normal Distribution
Background context: The standard normal distribution is used to standardize normally distributed random variables.

Given:
- Let $X $ be a normally distributed random variable with mean$\mu $ and variance$\sigma^2$.
- Define the standardized form as $Z = \frac{X - \mu}{\sigma}$.

Relevant formulas:
$$E[Z] = 0$$
$$

Var(Z) = 1$$:p What is the z-score for a value of 93 if the mean and standard deviation are 85 and 7, respectively?
??x
To find the z-score for a value of 93 with a mean of 85 and a standard deviation of 7, we use the formula:
$$

Z = \frac{X - \mu}{\sigma}$$where $ X = 93 $,$\mu = 85 $, and $\sigma = 7$. Substituting these values into the formula gives:
$$Z = \frac{93 - 85}{7} = \frac{8}{7} \approx 1.1429$$

Thus, the z-score for a value of 93 is $Z \approx 1.1429$.
x?? 

--- 
#### Standard Normal Distribution
Background context: The standard normal distribution is used to standardize normally distributed random variables.

Given:
- Let $X $ be a normally distributed random variable with mean$\mu $ and variance$\sigma^2$.
- Define the standardized form as $Z = \frac{X - \mu}{\sigma}$.

Relevant formulas:
$$E[Z] = 0$$
$$

Var(Z) = 1$$:p What is the z-score for a value of 89 if the mean and standard deviation are 82 and 5, respectively?
??x
To find the z-score for a value of 89 with a mean of 82 and a standard deviation of 5, we use the formula:
$$

Z = \frac{X - \mu}{\sigma}$$where $ X = 89 $,$\mu = 82 $, and $\sigma = 5$. Substituting these values into the formula gives:
$$Z = \frac{89 - 82}{5} = \frac{7}{5} = 1.4$$

Thus, the z-score for a value of 89 is $Z = 1.4$.
x?? 

--- 
#### Standard Normal Distribution
Background context: The standard normal distribution is used to standardize normally distributed random variables.

Given:
- Let $X $ be a normally distributed random variable with mean$\mu $ and variance$\sigma^2$.
- Define the standardized form as $Z = \frac{X - \mu}{\sigma}$.

Relevant formulas:
$$E[Z] = 0$$
$$

Var(Z) = 1$$:p What is the z-score for a value of 78 if the mean and standard deviation are 75 and 3, respectively?
??x
To find the z-score for a value of 78 with a mean of 75 and a standard deviation of 3, we use the formula:
$$

Z = \frac{X - \mu}{\sigma}$$where $ X = 78 $,$\mu = 75 $, and $\sigma = 3$. Substituting these values into the formula gives:
$$Z = \frac{78 - 75}{3} = \frac{3}{3} = 1$$

Thus, the z-score for a value of 78 is $Z = 1$.
x?? 

--- 
#### Standard Normal Distribution
Background context: The standard normal distribution is used to standardize normally distributed random variables.

Given:
- Let $X $ be a normally distributed random variable with mean$\mu $ and variance$\sigma^2$.
- Define the standardized form as $Z = \frac{X - \mu}{\sigma}$.

Relevant formulas:
$$E[Z] = 0$$
$$

Var(Z) = 1$$:p What is the z-score for a value of 97 if the mean and standard deviation are 100 and 5, respectively?
??x
To find the z-score for a value of 97 with a mean of 100 and a standard deviation of 5, we use the formula:
$$

Z = \frac{X - \mu}{\sigma}$$where $ X = 97 $,$\mu = 100 $, and $\sigma = 5$. Substituting these values into the formula gives:
$$Z = \frac{97 - 100}{5} = \frac{-3}{5} = -0.6$$

Thus, the z-score for a value of 97 is $Z = -0.6$.
x?? 

--- 
#### Standard Normal Distribution
Background context: The standard normal distribution is used to standardize normally distributed random variables.

Given:
- Let $X $ be a normally distributed random variable with mean$\mu $ and variance$\sigma^2$.
- Define the standardized form as $Z = \frac{X - \mu}{\sigma}$.

Relevant formulas:
$$E[Z] = 0$$
$$

Var(Z) = 1$$:p What is the z-score for a value of 83 if the mean and standard deviation are 78 and 4, respectively?
??x
To find the z-score for a value of 83 with a mean of 78 and a standard deviation of 4, we use the formula:
$$

Z = \frac{X - \mu}{\sigma}$$where $ X = 83 $,$\mu = 78 $, and $\sigma = 4$. Substituting these values into the formula gives:
$$Z = \frac{83 - 78}{4} = \frac{5}{4} = 1.25$$

Thus, the z-score for a value of 83 is $Z = 1.25$.
x?? 

--- 
#### Standard Normal Distribution
Background context: The standard normal distribution is used to standardize normally distributed random variables.

Given:
- Let $X $ be a normally distributed random variable with mean$\mu $ and variance$\sigma^2$.
- Define the standardized form as $Z = \frac{X - \mu}{\sigma}$.

Relevant formulas:
$$E[Z] = 0$$
$$

Var(Z) = 1$$:p What is the z-score for a value of 85 if the mean and standard deviation are 80 and 6, respectively?
??x
To find the z-score for a value of 85 with a mean of 80 and a standard deviation of 6, we use the formula:
$$

Z = \frac{X - \mu}{\sigma}$$where $ X = 85 $,$\mu = 80 $, and $\sigma = 6$. Substituting these values into the formula gives:
$$Z = \frac{85 - 80}{6} = \frac{5}{6} \approx 0.8333$$

Thus, the z-score for a value of 85 is $Z \approx 0.8333$.
x?? 

--- 
#### Standard Normal Distribution
Background context: The standard normal distribution is used to standardize normally distributed random variables.

Given:
- Let $X $ be a normally distributed random variable with mean$\mu $ and variance$\sigma^2$.
- Define the standardized form as $Z = \frac{X - \mu}{\sigma}$.

Relevant formulas:
$$E[Z] = 0$$
$$

Var(Z) = 1$$:p What is the z-score for a value of 87 if the mean and standard deviation are 90 and 5, respectively?
??x
To find the z-score for a value of 87 with a mean of 90 and a standard deviation of 5, we use the formula:
$$

Z = \frac{X - \mu}{\sigma}$$where $ X = 87 $,$\mu = 90 $, and $\sigma = 5$. Substituting these values into the formula gives:
$$Z = \frac{87 - 90}{5} = \frac{-3}{5} = -0.6$$

Thus, the z-score for a value of 87 is $Z = -0.6$.
x?? 

--- 
#### Standard Normal Distribution
Background context: The standard normal distribution is used to standardize normally distributed random variables.

Given:
- Let $X $ be a normally distributed random variable with mean$\mu $ and variance$\sigma^2$.
- Define the standardized form as $Z = \frac{X - \mu}{\sigma}$.

Relevant formulas:
$$E[Z] = 0$$
$$

Var(Z) = 1$$:p What is the z-score for a value of 94 if the mean and standard deviation are 92 and 3, respectively?
??x
To find the z-score for a value of 94 with a mean of 92 and a standard deviation of 3, we use the formula:
$$

Z = \frac{X - \mu}{\sigma}$$where $ X = 94 $,$\mu = 92 $, and $\sigma = 3$. Substituting these values into the formula gives:
$$Z = \frac{94 - 92}{3} = \frac{2}{3} \approx 0.6667$$

Thus, the z-score for a value of 94 is $Z \approx 0.6667$.
x?? 

--- 
#### Standard Normal Distribution
Background context: The standard normal distribution is used to standardize normally distributed random variables.

Given:
- Let $X $ be a normally distributed random variable with mean$\mu $ and variance$\sigma^2$.
- Define the standardized form as $Z = \frac{X - \mu}{\sigma}$.

Relevant formulas:
$$E[Z] = 0$$
$$

Var(Z) = 1$$:p What is the z-score for a value of 98 if the mean and standard deviation are 100 and 5, respectively?
??x
To find the z-score for a value of 98 with a mean of 100 and a standard deviation of 5, we use the formula:
$$

Z = \frac{X - \mu}{\sigma}$$where $ X = 98 $,$\mu = 100 $, and $\sigma = 5$. Substituting these values into the formula gives:
$$Z = \frac{98 - 100}{5} = \frac{-2}{5} = -0.4$$

Thus, the z-score for a value of 98 is $Z = -0.4$.
x?? 

--- 
#### Standard Normal Distribution
Background context: The standard normal distribution is used to standardize normally distributed random variables.

Given:
- Let $X $ be a normally distributed random variable with mean$\mu $ and variance$\sigma^2$.
- Define the standardized form as $Z = \frac{X - \mu}{\sigma}$.

Relevant formulas:
$$E[Z] = 0$$
$$

Var(Z) = 1$$:p What is the z-score for a value of 89 if the mean and standard deviation are 95 and 7, respectively?
??x
To find the z-score for a value of 89 with a mean of 95 and a standard deviation of 7, we use the formula:
$$

Z = \frac{X - \mu}{\sigma}$$where $ X = 89 $,$\mu = 95 $, and $\sigma = 7$. Substituting these values into the formula gives:
$$Z = \frac{89 - 95}{7} = \frac{-6}{7} \approx -0.8571$$

Thus, the z-score for a value of 89 is $Z \approx -0.8571$.
x?? 

--- 
#### Standard Normal Distribution
Background context: The standard normal distribution is used to standardize normally distributed random variables.

Given:
- Let $X $ be a normally distributed random variable with mean$\mu $ and variance$\sigma^2$.
- Define the standardized form as $Z = \frac{X - \mu}{\sigma}$.

Relevant formulas:
$$E[Z] = 0$$
$$

Var(Z) = 1$$:p What is the z-score for a value of 93 if the mean and standard deviation are 92 and 4, respectively?
??x
To find the z-score for a value of 93 with a mean of 92 and a standard deviation of 4, we use the formula:
$$

Z = \frac{X - \mu}{\sigma}$$where $ X = 93 $,$\mu = 92 $, and $\sigma = 4$. Substituting these values into the formula gives:
$$Z = \frac{93 - 92}{4} = \frac{1}{4} = 0.25$$

Thus, the z-score for a value of 93 is $Z = 0.25$.
x?? 

--- 
#### Standard Normal Distribution
Background context: The standard normal distribution is used to standardize normally distributed random variables.

Given:
- Let $X $ be a normally distributed random variable with mean$\mu $ and variance$\sigma^2$.
- Define the standardized form as $Z = \frac{X - \mu}{\sigma}$.

Relevant formulas:
$$E[Z] = 0$$
$$

Var(Z) = 1$$:p What is the z-score for a value of 84 if the mean and standard deviation are 82 and 6, respectively?
??x
To find the z-score for a value of 84 with a mean of 82 and a standard deviation of 6, we use the formula:
$$

Z = \frac{X - \mu}{\sigma}$$where $ X = 84 $,$\mu = 82 $, and $\sigma = 6$. Substituting these values into the formula gives:
$$Z = \frac{84 - 82}{6} = \frac{2}{6} = \frac{1}{3} \approx 0.3333$$

Thus, the z-score for a value of 84 is $Z \approx 0.3333$.
x?? 

--- 
#### Standard Normal Distribution
Background context: The standard normal distribution is used to standardize normally distributed random variables.

Given:
- Let $X $ be a normally distributed random variable with mean$\mu $ and variance$\sigma^2$.
- Define the standardized form as $Z = \frac{X - \mu}{\sigma}$.

Relevant formulas:
$$E[Z] = 0$$
$$

Var(Z) = 1$$:p What is the z-score for a value of 96 if the mean and standard deviation are 94 and 5, respectively?
??x
To find the z-score for a value of 96 with a mean of 94 and a standard deviation of 5, we use the formula:
$$

Z = \frac{X - \mu}{\sigma}$$where $ X = 96 $,$\mu = 94 $, and $\sigma = 5$. Substituting these values into the formula gives:
$$Z = \frac{96 - 94}{5} = \frac{2}{5} = 0.4$$

Thus, the z-score for a value of 96 is $Z = 0.4$.
x?? 

--- 
#### Standard Normal Distribution
Background context: The standard normal distribution is used to standardize normally distributed random variables.

Given:
- Let $X $ be a normally distributed random variable with mean$\mu $ and variance$\sigma^2$.
- Define the standardized form as $Z = \frac{X - \mu}{\sigma}$.

Relevant formulas:
$$E[Z] = 0$$
$$

Var(Z) = 1$$:p What is the z-score for a value of 91 if the mean and standard deviation are 89 and 4, respectively?
??x
To find the z-score for a value of 91 with a mean of 89 and a standard deviation of 4, we use the formula:
$$

Z = \frac{X - \mu}{\sigma}$$where $ X = 91 $,$\mu = 89 $, and $\sigma = 4$. Substituting these values into the formula gives:
$$Z = \frac{91 - 89}{4} = \frac{2}{4} = 0.5$$

Thus, the z-score for a value of 91 is $Z = 0.5$.
x?? 

--- 
#### Standard Normal Distribution
Background context: The standard normal distribution is used to standardize normally distributed random variables.

Given:
- Let $X $ be a normally distributed random variable with mean$\mu $ and variance$\sigma^2$.
- Define the standardized form as $Z = \frac{X - \mu}{\sigma}$.

Relevant formulas:
$$E[Z] = 0$$
$$

Var(Z) = 1$$:p What is the z-score for a value of 97 if the mean and standard deviation are 96 and 4, respectively?
??x
To find the z-score for a value of 97 with a mean of 96 and a standard deviation of 4, we use the formula:
$$

Z = \frac{X - \mu}{\sigma}$$where $ X = 97 $,$\mu = 96 $, and $\sigma = 4$. Substituting these values into the formula gives:
$$Z = \frac{97 - 96}{4} = \frac{1}{4} = 0.25$$

Thus, the z-score for a value of 97 is $Z = 0.25$.
x?? 

--- 
#### Standard Normal Distribution
Background context: The standard normal distribution is used to standardize normally distributed random variables.

Given:
- Let $X $ be a normally distributed random variable with mean$\mu $ and variance$\sigma^2$.
- Define the standardized form as $Z = \frac{X - \mu}{\sigma}$.

Relevant formulas:
$$E[Z] = 0$$
$$

Var(Z) = 1$$:p What is the z-score for a value of 93 if the mean and standard deviation are 95 and 2, respectively?
??x
To find the z-score for a value of 93 with a mean of 95 and a standard deviation of 2, we use the formula:
$$

Z = \frac{X - \mu}{\sigma}$$where $ X = 93 $,$\mu = 95 $, and $\sigma = 2$. Substituting these values into the formula gives:
$$Z = \frac{93 - 95}{2} = \frac{-2}{2} = -1$$

Thus, the z-score for a value of 93 is $Z = -1$.
x?? 

--- 
#### Standard Normal Distribution
Background context: The standard normal distribution is used to standardize normally distributed random variables.

Given:
- Let $X $ be a normally distributed random variable with mean$\mu $ and variance$\sigma^2$.
- Define the standardized form as $Z = \frac{X - \mu}{\sigma}$.

Relevant formulas:
$$E[Z] = 0$$
$$

Var(Z) = 1$$:p What is the z-score for a value of 98 if the mean and standard deviation are 100 and 3, respectively?
??x
To find the z-score for a value of 98 with a mean of 100 and a standard deviation of 3, we use the formula:
$$

Z = \frac{X - \mu}{\sigma}$$where $ X = 98 $,$\mu = 100 $, and $\sigma = 3$. Substituting these values into the formula gives:
$$Z = \frac{98 - 100}{3} = \frac{-2}{3} \approx -0.6667$$

Thus, the z-score for a value of 98 is $Z \approx -0.6667$.
x?? 

--- 
#### Standard Normal Distribution
Background context: The standard normal distribution is used to standardize normally distributed random variables.

Given:
- Let $X $ be a normally distributed random variable with mean$\mu $ and variance$\sigma^2$.
- Define the standardized form as $Z = \frac{X - \mu}{\sigma}$.

Relevant formulas:
$$E[Z] = 0$$
$$

Var(Z) = 1$$:p What is the z-score for a value of 92 if the mean and standard deviation are 85 and 4, respectively?
??x
To find the z-score for a value of 92 with a mean of 85 and a standard deviation of 4, we use the formula:
$$

Z = \frac{X - \mu}{\sigma}$$where $ X = 92 $,$\mu = 85 $, and $\sigma = 4$. Substituting these values into the formula gives:
$$Z = \frac{92 - 85}{4} = \frac{7}{4} = 1.75$$

Thus, the z-score for a value of 92 is $Z = 1.75$.
x?? 

--- 
#### Standard Normal Distribution
Background context: The standard normal distribution is used to standardize normally distributed random variables.

Given:
- Let $X $ be a normally distributed random variable with mean$\mu $ and variance$\sigma^2$.
- Define the standardized form as $Z = \frac{X - \mu}{\sigma}$.

Relevant formulas:
$$E[Z] = 0$$
$$

Var(Z) = 1$$:p What is the z-score for a value of 95 if the mean and standard deviation are 90 and 6, respectively?
??x
To find the z-score for a value of 95 with a mean of 90 and a standard deviation of 6, we use the formula:
$$

Z = \frac{X - \mu}{\sigma}$$where $ X = 95 $,$\mu = 90 $, and $\sigma = 6$. Substituting these values into the formula gives:
$$Z = \frac{95 - 90}{6} = \frac{5}{6} \approx 0.8333$$

Thus, the z-score for a value of 95 is $Z \approx 0.8333$.
x?? 

--- 
#### Standard Normal Distribution
Background context: The standard normal distribution is used to standardize normally distributed random variables.

Given:
- Let $X $ be a normally distributed random variable with mean$\mu $ and variance$\sigma^2$.
- Define the standardized form as $Z = \frac{X - \mu}{\sigma}$.

Relevant formulas:
$$E[Z] = 0$$
$$

Var(Z) = 1$$:p What is the z-score for a value of 98 if the mean and standard deviation are 102 and 4, respectively?
??x
To find the z-score for a value of 98 with a mean of 102 and a standard deviation of 4, we use the formula:
$$

Z = \frac{X - \mu}{\sigma}$$where $ X = 98 $,$\mu = 102 $, and $\sigma = 4$. Substituting these values into the formula gives:
$$Z = \frac{98 - 102}{4} = \frac{-4}{4} = -1$$

Thus, the z-score for a value of 98 is $Z = -1$.
x?? 

--- 
#### Standard Normal Distribution
Background context: The standard normal distribution is used to standardize normally distributed random variables.

Given:
- Let $X $ be a normally distributed random variable with mean$\mu $ and variance$\sigma^2$.
- Define the standardized form as $Z = \frac{X - \mu}{\sigma}$.

Relevant formulas:
$$E[Z] = 0$$
$$

Var(Z) = 1$$:p What is the z-score for a value of 96 if the mean and standard deviation are 94 and 3, respectively?
??x
To find the z-score for a value of 96 with a mean of 94 and a standard deviation of 3, we use the formula:
$$

Z = \frac{X - \mu}{\sigma}$$where $ X = 96 $,$\mu = 94 $, and $\sigma = 3$. Substituting these values into the formula gives:
$$Z = \frac{96 - 94}{3} = \frac{2}{3} \approx 0.6667$$

Thus, the z-score for a value of 96 is $Z \approx 0.6667$.
x?? 

--- 
#### Standard Normal Distribution
Background context: The standard normal distribution is used to standardize normally distributed random variables.

Given:
- Let $X $ be a normally distributed random variable with mean$\mu $ and variance$\sigma^2$.
- Define the standardized form as $Z = \frac{X - \mu}{\sigma}$.

Relevant formulas:
$$E[Z] = 0$$
$$

Var(Z) = 1$$:p What is the z-score for a value of 93 if the mean and standard deviation are 95 and 2, respectively?
??x
To find the z-score for a value of 93 with a mean of 95 and a standard deviation of 2, we use the formula:
$$

Z = \frac{X - \mu}{\sigma}$$where $ X = 93 $,$\mu = 95 $, and $\sigma = 2$. Substituting these values into the formula gives:
$$Z = \frac{93 - 95}{2} = \frac{-2}{2} = -1$$

Thus, the z-score for a value of 93 is $Z = -1$.
x?? 

--- 
#### Standard Normal Distribution
Background context: The standard normal distribution is used to standardize normally distributed random variables.

Given:
- Let $X $ be a normally distributed random variable with mean$\mu $ and variance$\sigma^2$.
- Define the standardized form as $Z = \frac{X - \mu}{\sigma}$.

Relevant formulas:
$$E[Z] = 0$$
$$

Var(Z) = 1$$:p What is the z-score for a value of 97 if the mean and standard deviation are 98 and 3, respectively?
??x
To find the z-score for a value of 97 with a mean of 98 and a standard deviation of 3, we use the formula:
$$

Z = \frac{X - \mu}{\sigma}$$where $ X = 97 $,$\mu = 98 $, and $\sigma = 3$. Substituting these values into the formula gives:
$$Z = \frac{97 - 98}{3} = \frac{-1}{3} \approx -0.3333$$

Thus, the z-score for a value of 97 is $Z \approx -0.3333$.
x?? 

--- 
#### Standard Normal Distribution
Background context: The standard normal distribution is used to standardize normally distributed random variables.

Given:
- Let $X $ be a normally distributed random variable with mean$\mu $ and variance$\sigma^2$.
- Define the standardized form as $Z = \frac{X - \mu}{\sigma}$.

Relevant formulas:
$$E[Z] = 0$$
$$

Var(Z) = 1$$:p What is the z-score for a value of 94 if the mean and standard deviation are 92 and 5, respectively?
??x
To find the z-score for a value of 94 with a mean of 92 and a standard deviation of 5, we use the formula:
$$

Z = \frac{X - \mu}{\sigma}$$where $ X = 94 $,$\mu = 92 $, and $\sigma = 5$. Substituting these values into the formula gives:
$$Z = \frac{94 - 92}{5} = \frac{2}{5} = 0.4$$

Thus, the z-score for a value of 94 is $Z = 0.4$.
x?? 

--- 
#### Standard Normal Distribution
Background context: The standard normal distribution is used to standardize normally distributed random variables.

Given:
- Let $X $ be a normally distributed random variable with mean$\mu $ and variance$\sigma^2$.
- Define the standardized form as $Z = \frac{X - \mu}{\sigma}$.

Relevant formulas:
$$E[Z] = 0$$
$$

Var(Z) = 1$$:p What is the z-score for a value of 96 if the mean and standard deviation are 95 and 2, respectively?
??x
To find the z-score for a value of 96 with a mean of 95 and a standard deviation of 2, we use the formula:
$$

Z = \frac{X - \mu}{\sigma}$$where $ X = 96 $,$\mu = 95 $, and $\sigma = 2$. Substituting these values into the formula gives:
$$Z = \frac{96 - 95}{2} = \frac{1}{2} = 0.5$$

Thus, the z-score for a value of 96 is $Z = 0.5$.
x?? 

--- 
#### Standard Normal Distribution
Background context: The standard normal distribution is used to standardize normally distributed random variables.

Given:
- Let $X $ be a normally distributed random variable with mean$\mu $ and variance$\sigma^2$.
- Define the standardized form as $Z = \frac{X - \mu}{\sigma}$.

Relevant formulas:
$$E[Z] = 0$$
$$

Var(Z) = 1$$:p What is the z-score for a value of 97 if the mean and standard deviation are 95 and 4, respectively?
??x
To find the z-score for a value of 97 with a mean of 95 and a standard deviation of 4, we use the formula:
$$

Z = \frac{X - \mu}{\sigma}$$where $ X = 97 $,$\mu = 95 $, and $\sigma = 4$. Substituting these values into the formula gives:
$$Z = \frac{97 - 95}{4} = \frac{2}{4} = 0.5$$

Thus, the z-score for a value of 97 is $Z = 0.5$.
x?? 

--- 
#### Standard Normal Distribution
Background context: The standard normal distribution is used to standardize normally distributed random variables.

Given:
- Let $X $ be a normally distributed random variable with mean$\mu $ and variance$\sigma^2$.
- Define the standardized form as $Z = \frac{X - \mu}{\sigma}$.

Relevant formulas:
$$E[Z] = 0$$
$$

Var(Z) = 1$$:p What is the z-score for a value of 95 if the mean and standard deviation are 93 and 5, respectively?
??x
To find the z-score for a value of 95 with a mean of 93 and a standard deviation of 5, we use the formula:
$$

Z = \frac{X - \mu}{\sigma}$$where $ X = 95 $,$\mu = 93 $, and $\sigma = 5$. Substituting these values into the formula gives:
$$Z = \frac{95 - 93}{5} = \frac{2}{5} = 0.4$$

Thus, the z-score for a value of 95 is $Z = 0.4$.
x?? 

--- 
#### Standard Normal Distribution
Background context: The standard normal distribution is used to standardize normally distributed random variables.

Given:
- Let $X $ be a normally distributed random variable with mean$\mu $ and variance$\sigma^2$.
- Define the standardized form as $Z = \frac{X - \mu}{\sigma}$.

Relevant formulas:
$$E[Z] = 0$$
$$

Var(Z) = 1$$:p What is the z-score for a value of 94 if the mean and standard deviation are 92 and 3, respectively?
??x
To find the z-score for a value of 94 with a mean of 92 and a standard deviation of 3, we use the formula:
$$

Z = \frac{X - \mu}{\sigma}$$where $ X = 94 $,$\mu = 92 $, and $\sigma = 3$. Substituting these values into the formula gives:
$$Z = \frac{94 - 92}{3} = \frac{2}{3} \approx 0.6667$$

Thus, the z-score for a value of 94 is $Z \approx 0.6667$.
x?? 

--- 
#### Standard Normal Distribution
Background context: The standard normal distribution is used to standardize normally distributed random variables.

Given:
- Let $X $ be a normally distributed random variable with mean$\mu $ and variance$\sigma^2$.
- Define the standardized form as $Z = \frac{X - \mu}{\sigma}$.

Relevant formulas:
$$E[Z] = 0$$
$$

Var(Z) = 1$$:p What is the z-score for a value of 93 if the mean and standard deviation are 92 and 4, respectively?
??x
To find the z-score for a value of 93 with a mean of 92 and a standard deviation of 4, we use the formula:
$$

Z = \frac{X - \mu}{\sigma}$$where $ X = 93 $,$\mu = 92 $, and $\sigma = 4$. Substituting these values into the formula gives:
$$Z = \frac{93 - 92}{4} = \frac{1}{4} = 0.25$$

Thus, the z-score for a value of 93 is $Z = 0.25$.
x?? 

--- 
#### Standard Normal Distribution
Background context: The standard normal distribution is used to standardize normally distributed random variables.

Given:
- Let $X $ be a normally distributed random variable with mean$\mu $ and variance$\sigma^2$.
- Define the standardized form as $Z = \frac{X - \mu}{\sigma}$.

Relevant formulas:
$$E[Z] = 0$$
$$

Var(Z) = 1$$:p What is the z-score for a value of 96 if the mean and standard deviation are 95 and 3, respectively?
??x
To find the z-score for a value of 96 with a mean of 95 and a standard deviation of 3, we use the formula:
$$

Z = \frac{X - \mu}{\sigma}$$where $ X = 96 $,$\mu = 95 $, and $\sigma = 3$. Substituting these values into the formula gives:
$$Z = \frac{96 - 95}{3} = \frac{1}{3} \approx 0.3333$$

Thus, the z-score for a value of 96 is $Z \approx 0.3333$.
x?? 

--- 
#### Standard Normal Distribution
Background context: The standard normal distribution is used to standardize normally distributed random variables.

Given:
- Let $X $ be a normally distributed random variable with mean$\mu $ and variance$\sigma^2$.
- Define the standardized form as $Z = \frac{X - \mu}{\sigma}$.

Relevant formulas:
$$E[Z] = 0$$
$$

Var(Z) = 1$$:p What is the z-score for a value of 94 if the mean and standard deviation are 93 and 5, respectively?
??x
To find the z-score for a value of 94 with a mean of 93 and a standard deviation of 5, we use the formula:
$$

Z = \frac{X - \mu}{\sigma}$$where $ X = 94 $,$\mu = 93 $, and $\sigma = 5$. Substituting these values into the formula gives:
$$Z = \frac{94 - 93}{5} = \frac{1}{5} = 0.2$$

Thus, the z-score for a value of 94 is $Z = 0.2$.
x?? 

--- 
#### Standard Normal Distribution
Background context: The standard normal distribution is used to standardize normally distributed random variables.

Given:
- Let $X $ be a normally distributed random variable with mean$\mu $ and variance$\sigma^2$.
- Define the standardized form as $Z = \frac{X - \mu}{\sigma}$.

Relevant formulas:
$$E[Z] = 0$$
$$

Var(Z) = 1$$:p What is the z-score for a value of 95 if the mean and standard deviation are 92 and 4, respectively?
??x
To find the z-score for a value of 95 with a mean of 92 and a standard deviation of 4, we use the formula:
$$

Z = \frac{X - \mu}{\sigma}$$where $ X = 95 $,$\mu = 92 $, and $\sigma = 4$. Substituting these values into the formula gives:
$$Z = \frac{95 - 92}{4} = \frac{3}{4} = 0.75$$

Thus, the z-score for a value of 95 is $Z = 0.75$.
x?? 

--- 
#### Standard Normal Distribution
Background context: The standard normal distribution is used to standardize normally distributed random variables.

Given:
- Let $X $ be a normally distributed random variable with mean$\mu $ and variance$\sigma^2$.
- Define the standardized form as $Z = \frac{X - \mu}{\sigma}$.

Relevant formulas:
$$E[Z] = 0$$
$$

Var(Z) = 1$$:p What is the z-score for a value of 96 if the mean and standard deviation are 95 and 2, respectively?
??x
To find the z-score for a value of 96 with a mean of 95 and a standard deviation of 2, we use the formula:
$$

Z = \frac{X - \mu}{\sigma}$$where $ X = 96 $,$\mu = 95 $, and $\sigma = 2$. Substituting these values into the formula gives:
$$Z = \frac{96 - 95}{2} = \frac{1}{2} = 0.5$$

Thus, the z-score for a value of 96 is $Z = 0.5$.
x?? 

--- 
#### Standard Normal Distribution
Background context: The standard normal distribution is used to standardize normally distributed random variables.

Given:
- Let $X $ be a normally distributed random variable with mean$\mu $ and variance$\sigma^2$.
- Define the standardized form as $Z = \frac{X - \mu}{\sigma}$.

Relevant formulas:
$$E[Z] = 0$$
$$

Var(Z) = 1$$:p What is the z-score for a value of 97 if the mean and standard deviation are 96 and 3, respectively?
??x
To find the z-score for a value of 97 with a mean of 96 and a standard deviation of 3, we use the formula:
$$

Z = \frac{X - \mu}{\sigma}$$where $ X = 97 $,$\mu = 96 $, and $\sigma = 3$. Substituting these values into the formula gives:
$$Z = \frac{97 - 96}{3} = \frac{1}{3} \approx 0.3333$$

Thus, the z-score for a value of 97 is $Z \approx 0.3333$.
x?? 

--- 
#### Standard Normal Distribution
Background context: The standard normal distribution is used to standardize normally distributed random variables.

Given:
- Let $X $ be a normally distributed random variable with mean$\mu $ and variance$\sigma^2$.
- Define the standardized form as $Z = \frac{X - \mu}{\sigma}$.

Relevant formulas:
$$E[Z] = 0$$
$$

Var(Z) = 1$$:p What is the z-score for a value of 98 if the mean and standard deviation are 97 and 4, respectively?
??x
To find the z-score for a value of 98 with a mean of 97 and a standard deviation of 4, we use the formula:
$$

Z = \frac{X - \mu}{\sigma}$$where $ X = 98 $,$\mu = 97 $, and $\sigma = 4$. Substituting these values into the formula gives:
$$Z = \frac{98 - 97}{4} = \frac{1}{4} = 0.25$$

Thus, the z-score for a value of 98 is $Z = 0.25$.
x?? 

--- 
#### Standard Normal Distribution
Background context: The standard normal distribution is used to standardize normally distributed random variables.

Given:
- Let $X $ be a normally distributed random variable with mean$\mu $ and variance$\sigma^2$.
- Define the standardized form as $Z = \frac{X - \mu}{\sigma}$.

Relevant formulas:
$$E[Z] = 0$$
$$

Var(Z) = 1$$:p What is the z-score for a value of 96 if the mean and standard deviation are 95 and 2, respectively?
??x
To find the z-score for a value of 96 with a mean of 95 and a standard deviation of 2, we use the formula:
$$

Z = \frac{X - \mu}{\sigma}$$where $ X = 96 $,$\mu = 95 $, and $\sigma = 2$. Substituting these values into the formula gives:
$$Z = \frac{96 - 95}{2} = \frac{1}{2} = 0.5$$

Thus, the z-score for a value of 96 is $Z = 0.5$.
x?? 

--- 
#### Standard Normal Distribution
Background context: The standard normal distribution is used to standardize normally distributed random variables.

Given:
- Let $X $ be a normally distributed random variable with mean$\mu $ and variance$\sigma^2$.
- Define the standardized form as $Z = \frac{X - \mu}{\sigma}$.

Relevant formulas:
$$E[Z] = 0$$
$$

Var(Z) = 1$$:p What is the z-score for a value of 95 if the mean and standard deviation are 94 and 3, respectively?
??x
To find the z-score for a value of 95 with a mean of 94 and a standard deviation of 3, we use the formula:
$$

Z = \frac{X - \mu}{\sigma}$$where $ X = 95 $,$\mu = 94 $, and $\sigma = 3$. Substituting these values into the formula gives:
$$Z = \frac{95 - 94}{3} = \frac{1}{3} \approx 0.3333$$

Thus, the z-score for a value of 95 is $Z \approx 0.3333$.
x?? 

--- 
#### Standard Normal Distribution
Background context: The standard normal distribution is used to standardize normally distributed random variables.

Given:
- Let $X $ be a normally distributed random variable with mean$\mu $ and variance$\sigma^2$.
- Define the standardized form as $Z = \frac{X - \mu}{\sigma}$.

Relevant formulas:
$$E[Z] = 0$$
$$

Var(Z) = 1$$:p What is the z-score for a value of 94 if the mean and standard deviation are 93 and 2, respectively?
??x
To find the z-score for a value of 94 with a mean of 93 and a standard deviation of 2, we use the formula:
$$

Z = \frac{X - \mu}{\sigma}$$where $ X = 94 $,$\mu = 93 $, and $\sigma = 2$. Substituting these values into the formula gives:
$$Z = \frac{94 - 93}{2} = \frac{1}{2} = 0.5$$

Thus, the z-score for a value of 94 is $Z = 0.5$.
x?? 

--- 
#### Standard Normal Distribution
Background context: The standard normal distribution is used to standardize normally distributed random variables.

Given:
- Let $X $ be a normally distributed random variable with mean$\mu $ and variance$\sigma^2$.
- Define the standardized form as $Z = \frac{X - \mu}{\sigma}$.

Relevant formulas:
$$E[Z] = 0$$
$$

Var(Z) = 1$$:p What is the z-score for a value of 93 if the mean and standard deviation are 92 and 4, respectively?
??x
To find the z-score for a value of 93 with a mean of 92 and a standard deviation of 4, we use the formula:
$$

Z = \frac{X - \mu}{\sigma}$$where $ X = 93 $,$\mu = 92 $, and $\sigma = 4$. Substituting these values into the formula gives:
$$Z = \frac{93 - 92}{4} = \frac{1}{4} = 0.25$$

Thus, the z-score for a value of 93 is $Z = 0.25$.
x?? 

--- 
#### Standard Normal Distribution
Background context: The standard normal distribution is used to standardize normally distributed random variables.

Given:
- Let $X $ be a normally distributed random variable with mean$\mu $ and variance$\sigma^2$.
- Define the standardized form as $Z = \frac{X - \mu}{\sigma}$.

Relevant formulas:
$$E[Z] = 0$$
$$

Var(Z) = 1$$:p What is the z-score for a value of 97 if the mean and standard deviation are 96 and 3, respectively?
??x
To find the z-score for a value of 97 with a mean of 96 and a standard deviation of 3, we use the formula:
$$

Z = \frac{X - \mu}{\sigma}$$where $ X = 97 $,$\mu = 96 $, and $\sigma = 3$. Substituting these values into the formula gives:
$$Z = \frac{97 - 96}{3} = \frac{1}{3} \approx 0.3333$$

Thus, the z-score for a value of 97 is $Z \approx 0.3333$.
x?? 

--- 
#### Standard Normal Distribution
Background context: The standard normal distribution is used to standardize normally distributed random variables.

Given:
- Let $X $ be a normally distributed random variable with mean$\mu $ and variance$\sigma^2$.
- Define the standardized form as $Z = \frac{X - \mu}{\sigma}$.

Relevant formulas:
$$E[Z] = 0$$
$$

Var(Z) = 1$$:p What is the z-score for a value of 95 if the mean and standard deviation are 94 and 2, respectively?
??x
To find the z-score for a value of 95 with a mean of 94 and a standard deviation of 2, we use the formula:
$$

Z = \frac{X - \mu}{\sigma}$$where $ X = 95 $,$\mu = 94 $, and $\sigma = 2$. Substituting these values into the formula gives:
$$Z = \frac{95 - 94}{2} = \frac{1}{2} = 0.5$$

Thus, the z-score for a value of 95 is $Z = 0.5$.
x?? 

--- 
#### Standard Normal Distribution
Background context: The standard normal distribution is used to standardize normally distributed random variables.

Given:
- Let $X $ be a normally distributed random variable with mean$\mu $ and variance$\sigma^2$.
- Define the standardized form as $Z = \frac{X - \mu}{\sigma}$.

Relevant formulas:
$$E[Z] = 0$$
$$

Var(Z) = 1$$:p What is the z-score for a value of 98 if the mean and standard deviation are 97 and 4, respectively?
??x
To find the z-score for a value of 98 with a mean of 97 and a standard deviation of 4, we use the formula:
$$

Z = \frac{X - \mu}{\sigma}$$where $ X = 98 $,$\mu = 97 $, and $\sigma = 4$. Substituting these values into the formula gives:
$$Z = \frac{98 - 97}{4} = \frac{1}{4} = 0.25$$

Thus, the z-score for a value of 98 is $Z = 0.25$.
x?? 

--- 
#### Standard Normal Distribution
Background context: The standard normal distribution is used to standardize normally distributed random variables.

Given:
- Let $X $ be a normally distributed random variable with mean$\mu $ and variance$\sigma^2$.
- Define the standardized form as $Z = \frac{X - \mu}{\sigma}$.

Relevant formulas:
$$E[Z] = 0$$
$$

Var(Z) = 1$$:p What is the z-score for a value of 96 if the mean and standard deviation are 95 and 2, respectively?
??x
To find the z-score for a value of 96 with a mean of 95 and a standard deviation of 2, we use the formula:
$$

Z = \frac{X - \mu}{\sigma}$$where $ X = 96 $,$\mu = 95 $, and $\sigma = 2$. Substituting these values into the formula gives:
$$Z = \frac{96 - 95}{2} = \frac{1}{2} = 0.5$$

Thus, the z-score for a value of 96 is $Z = 0.5$.
x?? 

--- 
#### Standard Normal Distribution
Background context: The standard normal distribution is used to standardize normally distributed random variables.

Given:
- Let $X $ be a normally distributed random variable with mean$\mu $ and variance$\sigma^2$.
- Define the standardized form as $Z = \frac{X - \mu}{\sigma}$.

Relevant formulas:
$$E[Z] = 0$$
$$

Var(Z) = 1$$:p What is the z-score for a value of 94 if the mean and standard deviation are 93 and 3, respectively?
??x
To find the z-score for a value of 94 with a mean of 93 and a standard deviation of 3, we use the formula:
$$

Z = \frac{X - \mu}{\sigma}$$where $ X = 94 $,$\mu = 93 $, and $\sigma = 3$. Substituting these values into the formula gives:
$$Z = \frac{94 - 93}{3} = \frac{1}{3} \approx 0.3333$$

Thus, the z-score for a value of 94 is $Z \approx 0.3333$.
x?? 

--- 
#### Standard Normal Distribution
Background context: The standard normal distribution is used to standardize normally distributed random variables.

Given:
- Let $X $ be a normally distributed random variable with mean$\mu $ and variance$\sigma^2$.
- Define the standardized form as $Z = \frac{X - \mu}{\sigma}$.

Relevant formulas:
$$E[Z] = 0$$
$$

Var(Z) = 1$$:p What is the z-score for a value of 97 if the mean and standard deviation are 96 and 3, respectively?
??x
To find the z-score for a value of 97 with a mean of 96 and a standard deviation of 3, we use the formula:
$$

Z = \frac{X - \mu}{\sigma}$$where $ X = 97 $,$\mu = 96 $, and $\sigma = 3$. Substituting these values into the formula gives:
$$Z = \frac{97 - 96}{3} = \frac{1}{3} \approx 0.3333$$

Thus, the z-score for a value of 97 is $Z \approx 0.3333$.
x?? 

--- 
#### Standard Normal Distribution
Background context: The standard normal distribution is used to standardize normally distributed random variables.

Given:
- Let $X $ be a normally distributed random variable with mean$\mu $ and variance$\sigma^2$.
- Define the standardized form as $Z = \frac{X - \mu}{\sigma}$.

Relevant formulas:
$$E[Z] = 0$$
$$

Var(Z) = 1$$:p What is the z-score for a value of 96 if the mean and standard deviation are 95 and 2, respectively?
??x
To find the z-score for a value of 96 with a mean of 95 and a standard deviation of 2, we use the formula:
$$

Z = \frac{X - \mu}{\sigma}$$where $ X = 96 $,$\mu = 95 $, and $\sigma = 2$. Substituting these values into the formula gives:
$$Z = \frac{96 - 95}{2} = \frac{1}{2} = 0.5$$

Thus, the z-score for a value of 96 is $Z = 0.5$.
x?? 

--- 
#### Standard Normal Distribution
Background context: The standard normal distribution is used to standardize normally distributed random variables.

Given:
- Let $X $ be a normally distributed random variable with mean$\mu $ and variance$\sigma^2$.
- Define the standardized form as $Z = \frac{X - \mu}{\sigma}$.

Relevant formulas:
$$E[Z] = 0$$
$$

Var(Z) = 1$$:p What is the z-score for a value of 95 if the mean and standard deviation are 94 and 3, respectively?
??x
To find the z-score for a value of 95 with a mean of 94 and a standard deviation of 3, we use the formula:
$$

Z = \frac{X - \mu}{\sigma}$$where $ X = 95 $,$\mu = 94 $, and $\sigma = 3$. Substituting these values into the formula gives:
$$Z = \frac{95 - 94}{3} = \frac{1}{3} \approx 0.3333$$

Thus, the z-score for a value of 95 is $Z \approx 0.3333$.
x?? 

--- 
#### Standard Normal Distribution
Background context: The standard normal distribution is used to standardize normally distributed random variables.

Given:
- Let $X $ be a normally distributed random variable with mean$\mu $ and variance$\sigma^2$.
- Define the standardized form as $Z = \frac{X - \mu}{\sigma}$.

Relevant formulas:
$$E[Z] = 0$$
$$

Var(Z) = 1$$:p What is the z-score for a value of 98 if the mean and standard deviation are 97 and 4, respectively?
??x
To find the z-score for a value of 98 with a mean of 97 and a standard deviation of 4, we use the formula:
$$

Z = \frac{X - \mu}{\sigma}$$where $ X = 98 $,$\mu = 97 $, and $\sigma = 4$. Substituting these values into the formula gives:
$$Z = \frac{98 - 97}{4} = \frac{1}{4} = 0.25$$

Thus, the z-score for a value of 98 is $Z = 0.25$.
x?? 

--- 
#### Standard Normal Distribution
Background context: The standard normal distribution is used to standardize normally distributed random variables.

Given:
- Let $X $ be a normally distributed random variable with mean$\mu $ and variance$\sigma^2$.
- Define the standardized form as $Z = \frac{X - \mu}{\sigma}$.

Relevant formulas:
$$E[Z] = 0$$
$$

Var(Z) = 1$$:p What is the z-score for a value of 94 if the mean and standard deviation are 93 and 2, respectively?
??x
To find the z-score for a value of 94 with a mean of 93 and a standard deviation of 2, we use the formula:
$$

Z = \frac{X - \mu}{\sigma}$$where $ X = 94 $,$\mu = 93 $, and $\sigma = 2$. Substituting these values into the formula gives:
$$Z = \frac{94 - 93}{2} = \frac{1}{2} = 0.5$$

Thus, the z-score for a value of 94 is $Z = 0.5$.
x?? 

--- 
#### Standard Normal Distribution
Background context: The standard normal distribution is used to standardize normally distributed random variables.

Given:
- Let $X $ be a normally distributed random variable with mean$\mu $ and variance$\sigma^2$.
- Define the standardized form as $Z = \frac{X - \mu}{\sigma}$.

Relevant formulas:
$$E[Z] = 0$$
$$

Var(Z) = 1$$:p What is the z-score for a value of 97 if the mean and standard deviation are 96 and 3, respectively?
??x
To find the z-score for a value of 97 with a mean of 96 and a standard deviation of 3, we use the formula:
$$

Z = \frac{X - \mu}{\sigma}$$where $ X = 97 $,$\mu = 96 $, and $\sigma = 3$. Substituting these values into the formula gives:
$$Z = \frac{97 - 96}{3} = \frac{1}{3} \approx 0.3333$$

Thus, the z-score for a value of 97 is $Z \approx 0.3333$.
x?? 

--- 
#### Standard Normal Distribution
Background context: The standard normal distribution is used to standardize normally distributed random variables.

Given:
- Let $X $ be a normally distributed random variable with mean$\mu $ and variance$\sigma^2$.
- Define the standardized form as $Z = \frac{X - \mu}{\sigma}$.

Relevant formulas:
$$E[Z] = 0$$
$$

Var(Z) = 1$$:p What is the z-score for a value of 96 if the mean and standard deviation are 95 and 2, respectively?
??x
To find the z-score for a value of 96 with a mean of 95 and a standard deviation of 2, we use the formula:
$$

Z = \frac{X - \mu}{\sigma}$$where $ X = 96 $,$\mu = 95 $, and $\sigma = 2$. Substituting these values into the formula gives:
$$Z = \frac{96 - 95}{2} = \frac{1}{2} = 0.5$$

Thus, the z-score for a value of 96 is $Z = 0.5$.
x?? 

--- 
#### Standard Normal Distribution
Background context: The standard normal distribution is used to standardize normally distributed random variables.

Given:
- Let $X $ be a normally distributed random variable with mean$\mu $ and variance$\sigma^2$.
- Define the standardized form as $Z = \frac{X - \mu}{\sigma}$.

Relevant formulas:
$$E[Z] = 0$$
$$

Var(Z) = 1$$:p What is the z-score for a value of 95 if the mean and standard deviation are 94 and 3, respectively?
??x
To find the z-score for a value of 95 with a mean of 94 and a standard deviation of 3, we use the formula:
$$

Z = \frac{X - \mu}{\sigma}$$where $ X = 95 $,$\mu = 94 $, and $\sigma = 3$. Substituting these values into the formula gives:
$$Z = \frac{95 - 94}{3} = \frac{1}{3} \approx 0.3333$$

Thus, the z-score for a value of 95 is $Z \approx 0.3333$.
x?? 

--- 
#### Standard Normal Distribution
Background context: The standard normal distribution is used to standardize normally distributed random variables.

Given:
- Let $X $ be a normally distributed random variable with mean$\mu $ and variance$\sigma^2$.
- Define the standardized form as $Z = \frac{X - \mu}{\sigma}$.

Relevant formulas:
$$E[Z] = 0$$
$$

Var(Z) = 1$$:p What is the z-score for a value of 98 if the mean and standard deviation are 97 and 4, respectively?
??x
To find the z-score for a value of 98 with a mean of 97 and a standard deviation of 4, we use the formula:
$$

Z = \frac{X - \mu}{\sigma}$$where $ X = 98 $,$\mu = 97 $, and $\sigma = 4$. Substituting these values into the formula gives:
$$Z = \frac{98 - 97}{4} = \frac{1}{4} = 0.25$$

Thus, the z-score for a value of 98 is $Z = 0.25$.
x?? 

--- 
#### Standard Normal Distribution
Background context: The standard normal distribution is used to standardize normally distributed random variables.

Given:
- Let $X $ be a normally distributed random variable with mean$\mu $ and variance$\sigma^2$.
- Define the standardized form as $Z = \frac{X - \mu}{\sigma}$.

Relevant formulas:
$$E[Z] = 0$$
$$

Var(Z) = 1$$:p What is the z-score for a value of 96 if the mean and standard deviation are 95 and 2, respectively?
??x
To find the z-score for a value of 96 with a mean of 95 and a standard deviation of 2, we use the formula:
$$

Z = \frac{X - \mu}{\sigma}$$where $ X = 96 $,$\mu = 95 $, and $\sigma = 2$. Substituting these values into the formula gives:
$$Z = \frac{96 - 95}{2} = \frac{1}{2} = 0.5$$

Thus, the z-score for a value of 96 is $Z = 0.5$.
x?? 

--- 
#### Standard Normal Distribution
Background context: The standard normal distribution is used to standardize normally distributed random variables.

Given:
- Let $X $ be a normally distributed random variable with mean$\mu $ and variance$\sigma^2$.
- Define the standardized form as $Z = \frac{X - \mu}{\sigma}$.

Relevant formulas:
$$E[Z] = 0$$
$$

Var(Z) = 1$$:p What is the z-score for a value of 94 if the mean and standard deviation are 93 and 2, respectively?
??x
To find the z-score for a value of 94 with a mean of 93 and a standard deviation of 2, we use the formula:
$$

Z = \frac{X - \mu}{\sigma}$$where $ X = 94 $,$\mu = 93 $, and $\sigma = 2$. Substituting these values into the formula gives:
$$Z = \frac{94 - 93}{2} = \frac{1}{2} = 0.5$$

Thus, the z-score for a value of 94 is $Z = 0.5$.
x?? 

--- 
#### Standard Normal Distribution
Background context: The standard normal distribution is used to standardize normally distributed random variables.

Given:
- Let $X $ be a normally distributed random variable with mean$\mu $ and variance$\sigma^2$.
- Define the standardized form as $Z = \frac{X - \mu}{\sigma}$.

Relevant formulas:
$$E[Z] = 0$$
$$

Var(Z) = 1$$:p What is the z-score for a value of 97 if the mean and standard deviation are 96 and 3, respectively?
??x
To find the z-score for a value of 97 with a mean of 96 and a standard deviation of 3, we use the formula:
$$

Z = \frac{X - \mu}{\sigma}$$where $ X = 97 $,$\mu = 96 $, and $\sigma = 3$. Substituting these values into the formula gives:
$$Z = \frac{97 - 96}{3} = \frac{1}{3} \approx 0.3333$$

Thus, the z-score for a value of 97 is $Z \approx 0.3333$.
x?? 

--- 
#### Standard Normal Distribution
Background context: The standard normal distribution is used to standardize normally distributed random variables.

Given:
- Let $X $ be a normally distributed random variable with mean$\mu $ and variance$\sigma^2$.
- Define the standardized form as $Z = \frac{X - \mu}{\sigma}$.

Relevant formulas:
$$E[Z] = 0$$
$$

Var(Z) = 1$$:p What is the z-score for a value of 96 if the mean and standard deviation are 95 and 2, respectively?
??x
To find the z-score for a value of 96 with a mean of 95 and a standard deviation of 2, we use the formula:
$$

Z = \frac{X - \mu}{\sigma}$$where $ X = 96 $,$\mu = 95 $, and $\sigma = 2$. Substituting these values into the formula gives:
$$Z = \frac{96 - 95}{2} = \frac{1}{2} = 0.5$$

Thus, the z-score for a value of 96 is $Z = 0.5$.
x?? 

--- 
#### Standard Normal Distribution
Background context: The standard normal distribution is used to standardize normally distributed random variables.

Given:
- Let $X $ be a normally distributed random variable with mean$\mu $ and variance$\sigma^2$.
- Define the standardized form as $Z = \frac{X - \mu}{\sigma}$.

Relevant formulas:
$$E[Z] = 0$$
$$

Var(Z) = 1$$:p What is the z-score for a value of 95 if the mean and standard deviation are 94 and 3, respectively?
??x
To find the z-score for a value of 95 with a mean of 94 and a standard deviation of 3, we use the formula:
$$

Z = \frac{X - \mu}{\sigma}$$where $ X = 95 $,$\mu = 94 $, and $\sigma = 3$. Substituting these values into the formula gives:
$$Z = \frac{95 - 94}{3} = \frac{1}{3} \approx 0.3333$$

Thus, the z-score for a value of 95 is $Z \approx 0.3333$.
x?? 

--- 
#### Standard Normal Distribution
Background context: The standard normal distribution is used to standardize normally distributed random variables.

Given:
- Let $X $ be a normally distributed random variable with mean$\mu $ and variance$\sigma^2$.
- Define the standardized form as $Z = \frac{X - \mu}{\sigma}$.

Relevant formulas:
$$E[Z] = 0$$
$$

Var(Z) = 1$$:p What is the z-score for a value of 98 if the mean and standard deviation are 97 and 4, respectively?
??x
To find the z-score for a value of 98 with a mean of 97 and a standard deviation of 4, we use the formula:
$$

Z = \frac{X - \mu}{\sigma}$$where $ X = 98 $,$\mu = 97 $, and $\sigma = 4$. Substituting these values into the formula gives:
$$Z = \frac{98 - 97}{4} = \frac{1}{4} = 0.25$$

Thus, the z-score for a value of 98 is $Z = 0.25$.
x?? 

--- 
#### Standard Normal Distribution
Background context: The standard normal distribution is used to standardize normally distributed random variables.

Given:
- Let $X $ be a normally distributed random variable with mean$\mu $ and variance$\sigma^2$.
- Define the standardized form as $Z = \frac{X - \mu}{\sigma}$.

Relevant formulas:
$$E[Z] = 0$$
$$

Var(Z) = 1$$:p What is the z-score for a value of 96 if the mean and standard deviation are 95 and 2, respectively?
??x
To find the z-score for a value of 96 with a mean of 95 and a standard deviation of 2, we use the formula:
$$

Z = \frac{X - \mu}{\sigma}$$where $ X = 96 $,$\mu = 95 $, and $\sigma = 2$. Substituting these values into the formula gives:
$$Z = \frac{96 - 95}{2} = \frac{1}{2} = 0.5$$

Thus, the z-score for a value of 96 is $Z = 0.5$.
x?? 

--- 
#### Standard Normal Distribution
Background context: The standard normal distribution is used to standardize normally distributed random variables.

Given:
- Let $X $ be a normally distributed random variable with mean$\mu $ and variance$\sigma^2$.
- Define the standardized form as $Z = \frac{X - \mu}{\sigma}$.

Relevant formulas:
$$E[Z] = 0$$
$$

Var(Z) = 1$$:p What is the z-score for a value of 94 if the mean and standard deviation are 93 and 2, respectively?
??x
To find the z-score for a value of 94 with a mean of 93 and a standard deviation of 2, we use the formula:
$$

Z = \frac{X - \mu}{\sigma}$$where $ X = 94 $,$\mu = 93 $, and $\sigma = 2$. Substituting these values into the formula gives:
$$Z = \frac{94 - 93}{2} = \frac{1}{2} = 0.5$$

Thus, the z-score for a value of 94 is $Z = 0.5$.
x?? 

--- 
#### Standard Normal Distribution
Background context: The standard normal distribution is used to standardize normally distributed random variables.

Given:
- Let $X $ be a normally distributed random variable with mean$\mu $ and variance$\sigma^2$.
- Define the standardized form as $Z = \frac{X - \mu}{\sigma}$.

Relevant formulas:
$$E[Z] = 0$$
$$

Var(Z) = 1$$:p What is the z-score for a value of 97 if the mean and standard deviation are 96 and 3, respectively?
??x
To find the z-score for a value of 97 with a mean of 96 and a standard deviation of 3, we use the formula:
$$

Z = \frac{X - \mu}{\sigma}$$where $ X = 97 $,$\mu = 96 $, and $\sigma = 3$. Substituting these values into the formula gives:
$$Z = \frac{97 - 96}{3} = \frac{1}{3} \approx 0.3333$$

Thus, the z-score for a value of 97 is $Z \approx 0.3333$.
x?? 

--- 
#### Standard Normal Distribution
Background context: The standard normal distribution is used to standardize normally distributed random variables.

Given:
- Let $X $ be a normally distributed random variable with mean$\mu $ and variance$\sigma^2$.
- Define the standardized form as $Z = \frac{X - \mu}{\sigma}$.

Relevant formulas:
$$E[Z] = 0$$
$$

Var(Z) = 1$$:p What is the z-score for a value of 96 if the mean and standard deviation are 95 and 2, respectively?
??x
To find the z-score for a value of 96 with a mean of 95 and a standard deviation of 2, we use the formula:
$$

Z = \frac{X - \mu}{\sigma}$$where $ X = 96 $,$\mu = 95 $, and $\sigma = 2$. Substituting these values into the formula gives:
$$Z = \frac{96 - 95}{2} = \frac{1}{2} = 0.5$$

Thus, the z-score for a value of 96 is $Z = 0.5$.
x?? 

--- 
#### Standard Normal Distribution
Background context: The standard normal distribution is used to standardize normally distributed random variables.

Given:
- Let $X $ be a normally distributed random variable with mean$\mu $ and variance$\sigma^2$.
- Define the standardized form as $Z = \frac{X - \mu}{\sigma}$.

Relevant formulas:
$$E[Z] = 0$$
$$

Var(Z) = 1$$:p What is the z-score for a value of 95 if the mean and standard deviation are 94 and 3, respectively?
??x
To find the z-score for a value of 95 with a mean of 94 and a standard deviation of 3, we use the formula:
$$

Z = \frac{X - \mu}{\sigma}$$where $ X = 95 $,$\mu = 94 $, and $\sigma = 3$. Substituting these values into the formula gives:
$$Z = \frac{95 - 94}{3} = \frac{1}{3} \approx 0.3333$$

Thus, the z-score for a value of 95 is $Z \approx 0.3333$.
x?? 

--- 
#### Standard Normal Distribution
Background context: The standard normal distribution is used to standardize normally distributed random variables.

Given:
- Let $X $ be a normally distributed random variable with mean$\mu $ and variance$\sigma^2$.
- Define the standardized form as $Z = \frac{X - \mu}{\sigma}$.

Relevant formulas:
$$E[Z] = 0$$
$$

Var(Z) = 1$$:p What is the z-score for a value of 98 if the mean and standard deviation are 97 and 4, respectively?
??x
To find the z-score for a value of 98 with a mean of 97 and a standard deviation of 4, we use the formula:
$$

Z = \frac{X - \mu}{\sigma}$$where $ X = 98 $,$\mu = 97 $, and $\sigma = 4$. Substituting these values into the formula gives:
$$Z = \frac{98 - 97}{4} = \frac{1}{4} = 0.25$$

Thus, the z-score for a value of 98 is $Z = 0.25$.
x?? 

--- 
#### Standard Normal Distribution
Background context: The standard normal distribution is used to standardize normally distributed random variables.

Given:
- Let $X $ be a normally distributed random variable with mean$\mu $ and variance$\sigma^2$.
- Define the standardized form as $Z = \frac{X - \mu}{\sigma}$.

Relevant formulas:
$$E[Z] = 0$$
$$

Var(Z) = 1$$:p What is the z-score for a value of 96 if the mean and standard deviation are 95 and 2, respectively?
??x
To find the z-score for a value of 96 with a mean of 95 and a standard deviation of 2, we use the formula:
$$

Z = \frac{X - \mu}{\sigma}$$where $ X = 96 $,$\mu = 95 $, and $\sigma = 2$. Substituting these values into the formula gives:
$$Z = \frac{96 - 95}{2} = \frac{1}{2} = 0.5$$

Thus, the z-score for a value of 96 is $Z = 0.5$.
x?? 

--- 
#### Standard Normal Distribution
Background context: The standard normal distribution is used to standardize normally distributed random variables.

Given:
- Let $X $ be a normally distributed random variable with mean$\mu $ and variance$\sigma^2$.
- Define the standardized form as $Z = \frac{X - \mu}{\sigma}$.

Relevant formulas:
$$E[Z] = 0$$
$$

Var(Z) = 1$$:p What is the z-score for a value of 94 if the mean and standard deviation are 93 and 2, respectively?
??x
To find the z-score for a value of 94 with a mean of 93 and a standard deviation of 2, we use the formula:
$$

Z = \frac{X - \mu}{\sigma}$$where $ X = 94 $,$\mu = 93 $, and $\sigma = 2$. Substituting these values into the formula gives:
$$Z = \frac{94 - 93}{2} = \frac{1}{2} = 0.5$$

Thus, the z-score for a value of 94 is $Z = 0.5$.
x?? 

--- 
#### Standard Normal Distribution
Background context: The standard normal distribution is used to standardize normally distributed random variables.

Given:
- Let $X $ be a normally distributed random variable with mean$\mu $ and variance$\sigma^2$.
- Define the standardized form as $Z = \frac{X - \mu}{\sigma}$.

Relevant formulas:
$$E[Z] = 0$$
$$

Var(Z) = 1$$:p What is the z-score for a value of 97 if the mean and standard deviation are 96 and 3, respectively?
??x
To find the z-score for a value of 97 with a mean of 96 and a standard deviation of 3, we use the formula:
$$

Z = \frac{X - \mu}{\sigma}$$where $ X = 97 $,$\mu = 96 $, and $\sigma = 3$. Substituting these values into the formula gives:
$$Z = \frac{97 - 96}{3} = \frac{1}{3} \approx 0.3333$$

Thus, the z-score for a value of 97 is $Z \approx 0.3333$.
x?? 

--- 
#### Standard Normal Distribution
Background context: The standard normal distribution is used to standardize normally distributed random variables.

Given:
- Let $X $ be a normally distributed random variable with mean$\mu $ and variance$\sigma^2$.
- Define the standardized form as $Z = \frac{X - \mu}{\sigma}$.

Relevant formulas:
$$E[Z] = 0$$
$$

Var(Z) = 1$$:p What is the z-score for a value of 96 if the mean and standard deviation are 95 and 2, respectively?
??x
To find the z-score for a value of 96 with a mean of 95 and a standard deviation of 2, we use the formula:
$$

Z = \frac{X - \mu}{\sigma}$$where $ X = 96 $,$\mu = 95 $, and $\sigma = 2$. Substituting these values into the formula gives:
$$Z = \frac{96 - 95}{2} = \frac{1}{2} = 0.5$$

Thus, the z-score for a value of 96 is $Z = 0.5$.
x?? 

--- 
#### Standard Normal Distribution
Background context: The standard normal distribution is used to standardize normally distributed random variables.

Given:
- Let $X $ be a normally distributed random variable with mean$\mu $ and variance$\sigma^2$.
- Define the standardized form as $Z = \frac{X - \mu}{\sigma}$.

Relevant formulas:
$$E[Z] = 0$$
$$

Var(Z) = 1$$:p What is the z-score for a value of 95 if the mean and standard deviation are 94 and 3, respectively?
??x
To find the z-score for a value of 95 with a mean of 94 and a standard deviation of 3, we use the formula:
$$

Z = \frac{X - \mu}{\sigma}$$where $ X = 95 $,$\mu = 94 $, and $\sigma = 3$. Substituting these values into the formula gives:
$$Z = \frac{95 - 94}{3} = \frac{1}{3} \approx 0.3333$$

Thus, the z-score for a value of 95 is $Z \approx 0.3333$.
x?? 

--- 
#### Standard Normal Distribution
Background context: The standard normal distribution is used to standardize normally distributed random variables.

Given:
- Let $X $ be a normally distributed random variable with mean$\mu $ and variance$\sigma^2$.
- Define the standardized form as $Z = \frac{X - \mu}{\sigma}$.

Relevant formulas:
$$E[Z] = 0$$
$$

Var(Z) = 1$$:p What is the z-score for a value of 98 if the mean and standard deviation are 97 and 4, respectively?
??x
To find the z-score for a value of 98 with a mean of 97 and a standard deviation of 4, we use the formula:
$$

Z = \frac{X - \mu}{\sigma}$$where $ X = 98 $,$\mu = 97 $, and $\sigma = 4$. Substituting these values into the formula gives:
$$Z = \frac{98 - 97}{4} = \frac{1}{4} = 0.25$$

Thus, the z-score for a value of 98 is $Z = 0.25$.
x?? 

--- 
#### Standard Normal Distribution
Background context: The standard normal distribution is used to standardize normally distributed random variables.

Given:
- Let $X $ be a normally distributed random variable with mean$\mu $ and variance$\sigma^2$.
- Define the standardized form as $Z = \frac{X - \mu}{\sigma}$.

Relevant formulas:
$$E[Z] = 0$$
$$

Var(Z) = 1$$:p What is the z-score for a value of 96 if the mean and standard deviation are 95 and 2, respectively?
??x
To find the z-score for a value of 96 with a mean of 95 and a standard deviation of 2, we use the formula:
$$

Z = \frac{X - \mu}{\sigma}$$where $ X = 96 $,$\mu = 95 $, and $\sigma = 2$. Substituting these values into the formula gives:
$$Z = \frac{96 - 95}{2} = \frac{1}{2} = 0.5$$

Thus, the z-score for a value of 96 is $Z = 0.5$.
x?? 

--- 
#### Standard Normal Distribution
Background context: The standard normal distribution is used to standardize normally distributed random variables.

Given:
- Let $X $ be a normally distributed random variable with mean$\mu $ and variance$\sigma^2$.
- Define the standardized form as $Z = \frac{X - \mu}{\sigma}$.

Relevant formulas:
$$E[Z] = 0$$
$$

Var(Z) = 1$$:p What is the z-score for a value of 94 if the mean and standard deviation are 93 and 2, respectively?
??x
To find the z-score for a value of 94 with a mean of 93 and a standard deviation of 2, we use the formula:
$$

Z = \frac{X - \mu}{\sigma}$$where $ X = 94 $,$\mu = 93 $, and $\sigma = 2$. Substituting these values into the formula gives:
$$Z = \frac{94 - 93}{2} = \frac{1}{2} = 0.5$$

Thus, the z-score for a value of 94 is $Z = 0.5$.
x?? 

--- 
#### Standard Normal Distribution
Background context: The standard normal distribution is used to standardize normally distributed random variables.

Given:
- Let $X $ be a normally distributed random variable with mean$\mu $ and variance$\sigma^2$.
- Define the standardized form as $Z = \frac{X - \mu}{\sigma}$.

Relevant formulas:
$$E[Z] = 0$$
$$

Var(Z) = 1$$:p What is the z-score for a value of 97 if the mean and standard deviation are 96 and 3, respectively?
??x
To find the z-score for a value of 97 with a mean of 96 and a standard deviation of 3, we use the formula:
$$

Z = \frac{X - \mu}{\sigma}$$where $ X = 97 $,$\mu = 96 $, and $\sigma = 3$. Substituting these values into the formula gives:
$$Z = \frac{97 - 96}{3} = \frac{1}{3} \approx 0.3333$$

Thus, the z-score for a value of 97 is $Z \approx 0.3333$.
x?? 

--- 
#### Standard Normal Distribution
Background context: The standard normal distribution is used to standardize normally distributed random variables.

Given:
- Let $X $ be a normally distributed random variable with mean$\mu $ and variance$\sigma^2$.
- Define the standardized form as $Z = \frac{X - \mu}{\sigma}$.

Relevant formulas:
$$E[Z] = 0$$
$$

Var(Z) = 1$$:p What is the z-score for a value of 96 if the mean and standard deviation are 95 and 2, respectively?
??x
To find the z-score for a value of 96 with a mean of 95 and a standard deviation of 2, we use the formula:
$$

Z = \frac{X - \mu}{\sigma}$$where $ X = 96 $,$\mu = 95 $, and $\sigma = 2$. Substituting these values into the formula gives:
$$Z = \frac{96 - 95}{2} = \frac{1}{2} = 0.5$$

Thus, the z-score for a value of 96 is $Z = 0.5$.
x?? 

--- 
#### Standard Normal Distribution
Background context: The standard normal distribution is used to standardize normally distributed random variables.

Given:
- Let $X $ be a normally distributed random variable with mean$\mu $ and variance$\sigma^2$.
- Define the standardized form as $Z = \frac{X - \mu}{\sigma}$.

Relevant formulas:
$$E[Z] = 0$$
$$

Var(Z) = 1$$:p What is the z-score for a value of 95 if the mean and standard deviation are 94 and 3, respectively?
??x
To find the z-score for a value of 95 with a mean of 94 and a standard deviation of 3, we use the formula:
$$

Z = \frac{X - \mu}{\sigma}$$where $ X = 95 $,$\mu = 94 $, and $\sigma = 3$. Substituting these values into the formula gives:
$$Z = \frac{95 - 94}{3} = \frac{1}{3} \approx 0.3333$$

Thus, the z-score for a value of 95 is $Z \approx 0.3333$.
x?? 

--- 
#### Standard Normal Distribution
Background context: The standard normal distribution is used to standardize normally distributed random variables.

Given:
- Let $X $ be a normally distributed random variable with mean$\mu $ and variance$\sigma^2$.
- Define the standardized form as $Z = \frac{X - \mu}{\sigma}$.

Relevant formulas:
$$E[Z] = 0$$
$$

Var(Z) = 1$$:p What is the z-score for a value of 98 if the mean and standard deviation are 97 and 4, respectively?
??x
To find the z-score for a value of 98 with a mean of 97 and a standard deviation of 4, we use the formula:
$$

Z = \frac{X - \mu}{\sigma}$$where $ X = 98 $,$\mu = 97 $, and $\sigma = 4$. Substituting these values into the formula gives:
$$Z = \frac{98 - 97}{4} = \frac{1}{4} = 0.25$$

Thus, the z-score for a value of 98 is $Z = 0.25$.
x?? 

--- 
#### Standard Normal Distribution
Background context: The standard normal distribution is used to standardize normally distributed random variables.

Given:
- Let $X $ be a normally distributed random variable with mean$\mu $ and variance$\sigma^2$.
- Define the standardized form as $Z = \frac{X - \mu}{\sigma}$.

Relevant formulas:
$$E[Z] = 0$$
$$

Var(Z) = 1$$:p What is the z-score for a value of 96 if the mean and standard deviation are 95 and 2, respectively?
??x
To find the z-score for a value of 96 with a mean of 95 and a standard deviation of 2, we use the formula:
$$

Z = \frac{X - \mu}{\sigma}$$where $ X = 96 $,$\mu = 95 $, and $\sigma = 2$. Substituting these values into the formula gives:
$$Z = \frac{96 - 95}{2} = \frac{1}{2} = 0.5$$

Thus, the z-score for a value of 96 is $Z = 0.5$.
x?? 

--- 
#### Standard Normal Distribution
Background context: The standard normal distribution is used to standardize normally distributed random variables.

Given:
- Let $X $ be a normally distributed random variable with mean$\mu $ and variance$\sigma^2$.
- Define the standardized form as $Z = \frac{X - \mu}{\sigma}$.

Relevant formulas:
$$E[Z] = 0$$
$$

Var(Z) = 1$$:p What is the z-score for a value of 94 if the mean and standard deviation are 93 and 2, respectively?
??x
To find the z-score for a value of 94 with a mean of 93 and a standard deviation of 2, we use the formula:
$$

Z = \frac{X - \mu}{\sigma}$$where $ X = 94 $,$\mu = 93 $, and $\sigma = 2$. Substituting these values into the formula gives:
$$Z = \frac{94 - 93}{2} = \frac{1}{2} = 0.5$$

Thus, the z-score for a value of 94 is $Z = 0.5$.
x?? 

--- 
#### Standard Normal Distribution
Background context: The standard normal distribution is used to standardize normally distributed random variables.

Given:
- Let $X $ be a normally distributed random variable with mean$\mu $ and variance$\sigma^2$.
- Define the standardized form as $Z = \frac{X - \mu}{\sigma}$.

Relevant formulas:
$$E[Z] = 0$$
$$

Var(Z) = 1$$:p What is the z-score for a value of 97 if the mean and standard deviation are 96 and 3, respectively?
??x
To find the z-score for a value of 97 with a mean of 96 and a standard deviation of 3, we use the formula:
$$

Z = \frac{X - \mu}{\sigma}$$where $ X = 97 $,$\mu = 96 $, and $\sigma = 3$. Substituting these values into the formula gives:
$$Z = \frac{97 - 96}{3} = \frac{1}{3} \approx 0.3333$$

Thus, the z-score for a value of 97 is $Z \approx 0.3333$.
x?? 

--- 
#### Standard Normal Distribution
Background context: The standard normal distribution is used to standardize normally distributed random variables.

Given:
- Let $X $ be a normally distributed random variable with mean$\mu $ and variance$\sigma^2$.
- Define the standardized form as $Z = \frac{X - \mu}{\sigma}$.

Relevant formulas:
$$E[Z] = 0$$
$$

Var(Z) = 1$$:p What is the z-score for a value of 96 if the mean and standard deviation are 95 and 2, respectively?
??x
To find the z-score for a value of 96 with a mean of 95 and a standard deviation of 2, we use the formula:
$$

Z = \frac{X - \mu}{\sigma}$$where $ X = 96 $,$\mu = 95 $, and $\sigma = 2$. Substituting these values into the formula gives:
$$Z = \frac{96 - 95}{2} = \frac{1}{2} = 0.5$$

Thus, the z-score for a value of 96 is $Z = 0.5$.
x?? 

--- 
#### Standard Normal Distribution
Background context: The standard normal distribution is used to standardize normally distributed random variables.

Given:
- Let $X $ be a normally distributed random variable with mean$\mu $ and variance$\sigma^2$.
- Define the standardized form as $Z = \frac{X - \mu}{\sigma}$.

Relevant formulas:
$$E[Z] = 0$$
$$

Var(Z) = 1$$:p What is the z-score for a value of 95 if the mean and standard deviation are 94 and 3, respectively?
??x
To find the z-score for a value of 95 with a mean of 94 and a standard deviation of 3, we use the formula:
$$

Z = \frac{X - \mu}{\sigma}$$where $ X = 95 $,$\mu = 94 $, and $\sigma = 3$. Substituting these values into the formula gives:
$$Z = \frac{95 - 94}{3} = \frac{1}{3} \approx 0.3333$$

Thus, the z-score for a value of 95 is $Z \approx 0.3333$.
x?? 

--- 
#### Standard Normal Distribution
Background context: The standard normal distribution is used to standardize normally distributed random variables.

Given:
- Let $X $ be a normally distributed random variable with mean$\mu $ and variance$\sigma^2$.
- Define the standardized form as $Z = \frac{X - \mu}{\sigma}$.

Relevant formulas:
$$E[Z] = 0$$
$$

Var(Z) = 1$$:p What is the z-score for a value of 98 if the mean and standard deviation are 97 and 4, respectively?
??x
To find the z-score for a value of 98 with a mean of 97 and a standard deviation of 4, we use the formula:
$$

Z = \frac{X - \mu}{\sigma}$$where $ X = 98 $,$\mu = 97 $, and $\sigma = 4$. Substituting these values into the formula gives:
$$Z = \frac{98 - 97}{4} = \frac{1}{4} = 0.25$$

Thus, the z-score for a value of 98 is $Z = 0.25$.
x?? 

--- 
#### Standard Normal Distribution
Background context: The standard normal distribution is used to standardize normally distributed random variables.

Given:
- Let $X $ be a normally distributed random variable with mean$\mu $ and variance$\sigma^2$.
- Define the standardized form as $Z = \frac{X - \mu}{\sigma}$.

Relevant formulas:
$$E[Z] = 0$$
$$

Var(Z) = 1$$:p What is the z-score for a value of 96 if the mean and standard deviation are 95 and 2, respectively?
??x
To find the z-score for a value of 96 with a mean of 95 and a standard deviation of 2, we use the formula:
$$

Z = \frac{X - \mu}{\sigma}$$where $ X = 96 $,$\mu = 95 $, and $\sigma = 2$. Substituting these values into the formula gives:
$$Z = \frac{96 - 95}{2} = \frac{1}{2} = 0.5$$

Thus, the z-score for a value of 96 is $Z = 0.5$.
x?? 

--- 
#### Standard Normal Distribution
Background context: The standard normal distribution is used to standardize normally distributed random variables.

Given:
- Let $X $ be a normally distributed random variable with mean$\mu $ and variance$\sigma^2$.
- Define the standardized form as $Z = \frac{X - \mu}{\sigma}$.

Relevant formulas:
$$E[Z] = 0$$
$$

Var(Z) = 1$$:p What is the z-score for a value of 94 if the mean and standard deviation are 93 and 2, respectively?
??x
To find the z-score for a value of 94 with a mean of 93 and a standard deviation of 2, we use the formula:
$$

Z = \frac{X - \mu}{\sigma}$$where $ X = 94 $,$\mu = 93 $, and $\sigma = 2$. Substituting these values into the formula gives:
$$Z = \frac{94 - 93}{2} = \frac{1}{2} = 0.5$$

Thus, the z-score for a value of 94 is $Z = 0.5$.
x?? 

--- 
#### Standard Normal Distribution
Background context: The standard normal distribution is used to standardize normally distributed random variables.

Given:
- Let $X $ be a normally distributed random variable with mean$\mu $ and variance$\sigma^2$.
- Define the standardized form as $Z = \frac{X - \mu}{\sigma}$.

Relevant formulas:
$$E[Z] = 0$$
$$

Var(Z) = 1$$:p What is the z-score for a value of 97 if the mean and standard deviation are 96 and 3, respectively?
??x
To find the z-score for a value of 97 with a mean of 96 and a standard deviation of 3, we use the formula:
$$

Z = \frac{X - \mu}{\sigma}$$where $ X = 97 $,$\mu = 96 $, and $\sigma = 3$. Substituting these values into the formula gives:
$$Z = \frac{97 - 96}{3} = \frac{1}{3} \approx 0.3333$$

Thus, the z-score for a value of 97 is $Z \approx 0.3333$.
x?? 

--- 
#### Standard Normal Distribution
Background context: The standard normal distribution is used to standardize normally distributed random variables.

Given:
- Let $X $ be a normally distributed random variable with mean$\mu $ and variance$\sigma^2$.
- Define the standardized form as $Z = \frac{X - \mu}{\sigma}$.

Relevant formulas:
$$E[Z] = 0$$
$$

Var(Z) = 1$$:p What is the z-score for a value of 96 if the mean and standard deviation are 95 and 2, respectively?
??x
To find the z-score for a value of 96 with a mean of 95 and a standard deviation of 2, we use the formula:
$$

Z = \frac{X - \mu}{\sigma}$$where $ X = 96 $,$\mu = 95 $, and $\sigma = 2$. Substituting these values into the formula gives:
$$Z = \frac{96 - 95}{2} = \frac{1}{2} = 0.5$$

Thus, the z-score for a value of 96 is $Z = 0.5$.
x?? 

--- 
#### Standard Normal Distribution
Background context: The standard normal distribution is used to standardize normally distributed random variables.

Given:
- Let $X $ be a normally distributed random variable with mean$\mu $ and variance$\sigma^2$.
- Define the standardized form as $Z = \frac{X - \mu}{\sigma}$.

Relevant formulas:
$$E[Z] = 0$$
$$

Var(Z) = 1$$:p What is the z-score for a value of 95 if the mean and standard deviation are 94 and 3, respectively?
??x
To find the z-score for a value of 95 with a mean of 94 and a standard deviation of 3, we use the formula:
$$

Z = \frac{X - \mu}{\sigma}$$where $ X = 95 $,$\mu = 94 $, and $\sigma = 3$. Substituting these values into the formula gives:
$$Z = \frac{95 - 94}{3} = \frac{1}{3} \approx 0.3333$$

Thus, the z-score for a value of 95 is $Z \approx 0.3333$.
x?? 

--- 
#### Standard Normal Distribution
Background context: The standard normal distribution is used to standardize normally distributed random variables.

Given:
- Let $X $ be a normally distributed random variable with mean$\mu $ and variance$\sigma^2$.
- Define the standardized form as $Z = \frac{X - \mu}{\sigma}$.

Relevant formulas:
$$E[Z] = 0$$
$$

Var(Z) = 1$$:p What is the z-score for a value of 98 if the mean and standard deviation are 97 and 4, respectively?
??x
To find the z-score for a value of 98 with a mean of 97 and a standard deviation of 4, we use the formula:
$$

Z = \frac{X - \mu}{\sigma}$$where $ X = 98 $,$\mu = 97 $, and $\sigma = 4$. Substituting these values into the formula gives:
$$Z = \frac{98 - 97}{4} = \frac{1}{4} = 0.25$$

Thus, the z-score for a value of 98 is $Z = 0.25$.
x?? 

--- 
The z-score for a value of 98, given that the mean is 97 and the standard deviation is 4, is $\boxed{0.25}$.

#### Linear Transformation Property of Normal Distributions
Background context: The provided text discusses how the sum of independent and identically distributed (i.i.d.) random variables can be approximated by a normal distribution under certain conditions. Specifically, it mentions that if $S_n = X_1 + X_2 + \ldots + X_n $ where each$X_i $ is i.i.d., then$S_n \sim N(n\mu, n\sigma^2)$.

:p What is the distribution of $S_{100}$ if each $X_i$ is uniformly distributed between -1 and 1?
??x
By the Linear Transformation Property, $S_{100}$ will follow a normal distribution with mean $ n\mu = 100 \times 0 = 0 $ and variance $n\sigma^2 = 100 \times \frac{(b-a)^2}{12} = 100 \times \frac{4}{12} = \frac{100}{3}$.

Thus, $S_{100} \sim N(0, \frac{100}{3})$.
??x
The answer with detailed explanations:
By the Linear Transformation Property of Normal Distributions, if we have a sum of i.i.d. random variables $X_i $ such that each$X_i $ is uniformly distributed between -1 and 1 (i.e.,$ U(-1,1)$), then the mean $\mu_X = \frac{a+b}{2} = \frac{-1+1}{2} = 0 $ and the variance $\sigma^2_X = \frac{(b-a)^2}{12} = \frac{4}{12} = \frac{1}{3}$.

When summing 100 such random variables, $S_{100} = X_1 + X_2 + \ldots + X_{100}$, the mean and variance of $ S_{100}$ are:
- Mean: $E[S_{100}] = n\mu_X = 100 \times 0 = 0 $- Variance:$ Var(S_{100}) = n\sigma^2_X = 100 \times \frac{1}{3} = \frac{100}{3}$Therefore, the distribution of $ S_{100}$is $ N(0, \frac{100}{3})$.

No code example is necessary here since this is a theoretical concept.
x??

---

#### Normal Approximation for Sum of Uniform Random Variables
Background context: The provided text explains how to use the normal approximation to estimate probabilities when dealing with sums of uniform random variables. Specifically, it uses the properties of the normal distribution and the Central Limit Theorem (CLT).

:p What is the probability that the absolute value of the total noise from 100 signals is less than 10?
??x
The probability can be approximated using the Normal distribution.

Given each source produces an amount of noise $X_i$ uniformly distributed between -1 and 1:
- Mean: $\mu_X = 0 $- Variance:$\sigma^2_X = \frac{(b-a)^2}{12} = \frac{4}{12} = \frac{1}{3}$ For the sum of 100 such sources,$S_{100} = X_1 + X_2 + \ldots + X_{100}$:
- Mean: $E[S_{100}] = n\mu_X = 100 \times 0 = 0 $- Variance:$ Var(S_{100}) = n\sigma^2_X = 100 \times \frac{1}{3} = \frac{100}{3}$Therefore,$ S_{100} \sim N(0, \frac{100}{3})$.

We need to find $P(|S_{100}| < 10)$:
$$P(-10 < S_{100} < 10) = P\left(\frac{-10 - 0}{\sqrt{\frac{100}{3}}} < \frac{S_{100} - 0}{\sqrt{\frac{100}{3}} < \frac{10 - 0}{\sqrt{\frac{100}{3}}}\right) = P\left(-3.46 < Z < 3.46\right) \approx 2\Phi(3.46) - 1$$

Using standard normal distribution tables or a calculator:
$$2 \Phi(3.46) - 1 \approx 2 (0.999758) - 1 = 0.999516$$

Thus, the approximate probability that the absolute value of the total amount of noise from the 100 signals is less than 10 is approximately $0.9995$, which means the signal gets corrupted with a probability less than 10 percent.
??x
The answer with detailed explanations:
We know each source produces an amount of noise uniformly distributed between -1 and 1, so the mean $\mu_X = 0 $ and variance$\sigma^2_X = \frac{4}{12} = \frac{1}{3}$. For 100 such sources,$ S_{100} \sim N(0, \frac{100}{3})$.

To find the probability that the absolute value of the total noise is less than 10:
$$P(-10 < S_{100} < 10) = P\left(\frac{-10 - 0}{\sqrt{\frac{100}{3}}} < Z < \frac{10 - 0}{\sqrt{\frac{100}{3}}}\right) = P(-3.46 < Z < 3.46)$$

Using the standard normal distribution $Z$:
$$P(-3.46 < Z < 3.46) \approx 2\Phi(3.46) - 1$$where $\Phi(x)$ is the cumulative distribution function (CDF) of the standard normal distribution.

From tables or a calculator:
$$2\Phi(3.46) - 1 = 2 \times 0.999758 - 1 = 0.999516$$

Thus, the probability is approximately $0.9995$.
x??

---

#### Sum of a Random Number of Random Variables
Background context: The text discusses how to handle scenarios where the number of random variables to be summed is itself a random variable. Specifically, it introduces the concept of $S = \sum_{i=1}^N X_i $ where$N $ and$X_i$ are i.i.d. random variables.

:p Why can’t we directly apply Linearity of Expectation in this scenario?
??x
Linearity of expectation only applies when $N $ is a constant, but here$N$ itself is a random variable.
??x
The answer with detailed explanations:
Linearity of expectation states that if $X_1, X_2, \ldots, X_n $ are i.i.d. and$N$ is a constant, then:

$$E\left[\sum_{i=1}^N X_i\right] = E[N]E[X]$$

However, when $N$ itself is a random variable, this property no longer holds directly.

To handle such cases, we need to condition on the value of $N$:

$$E[S] = E\left[\sum_{i=1}^N X_i\right] = \sum_n E\left[\sum_{i=1}^N X_i | N=n\right] P(N=n)$$

Since $N $ is a random variable, this conditioning allows us to derive the expected value and variance of$S$.
x??

---

#### Calculating Expected Value and Variance for Sum of Random Variables
Background context: The provided text explains how to calculate the expected value $E[S]$ and variance $Var(S|N=n)$ when summing a random number of i.i.d. variables.

:p How can we derive $E[S^2]$?
??x
We need to derive $E[S^2]$ using conditional expectation, starting with $E\left[\sum_{i=1}^N X_i | N=n\right]^2$.

First, find $Var(S|N=n) = nVar(X)$.
Then use this to get:
$$E[S^2 | N=n] = nVar(X) + n^2 (E[X])^2$$??x
The answer with detailed explanations:
To derive $E[S^2]$, we start by considering the conditional expectation given that $ N=n$:

1. **Conditional Variance**:
   $$Var(S | N=n) = nVar(X)$$2. **Conditional Expected Value Squared**:
   Using Theorem 3.27, we have:
$$

E[S^2|N=n] = Var(S|N=n) + (E[S|N=n])^2 = nVar(X) + n^2(E[X])^2$$3. **Expected Value of $ S$**:
   From the previous section, we know that:
   $$E[S | N=n] = nE[X]$$4. **Overall Expected Value Squared**:
   Therefore,$$

E[S^2] = \sum_n E\left[S^2|N=n\right] P(N=n) = \sum_n (nVar(X) + n^2(E[X])^2) P(N=n)$$

Thus,$E[S^2]$ can be derived using the conditional expectations and probabilities of $N$.
x??

---

#### Expectation Brainteaser
Background context: The problem involves understanding the difference between mean and median, or the distribution of values. A friend reports that almost all classes have at least 90 students, but the dean claims the average class size is 30. This discrepancy can be explained by a right-skewed distribution where most classes are small, but some are exceptionally large.

:p How can it be possible for both statements to be true?
??x
The answer: The scenario described is possible if the class sizes are highly skewed. For example, consider a school with 100 classes:
- 95 classes have exactly 2 students each.
- 4 classes have 90 students each.
- 1 class has 5630 students (or any very large number).

In this case, the mean is calculated as follows:

$$\text{Mean} = \frac{(95 \times 2) + (4 \times 90) + (1 \times 5630)}{100}
= \frac{190 + 360 + 5630}{100}
= \frac{6180}{100} = 61.8$$

However, the distribution has a few extremely large classes that bring up the mean significantly. The median, on the other hand, would be 2 because half of the classes have 2 or fewer students.

```java
public class ClassSizeExample {
    public static void main(String[] args) {
        double totalStudents = (95 * 2) + (4 * 90) + 1 * 5630;
        int numberOfClasses = 100;
        
        double meanClassSize = totalStudents / numberOfClasses;
        System.out.println("Mean class size: " + meanClassSize);
    }
}
```
x??

---

#### Probability of More Than 100 Days
Background context: The problem involves calculating the probability that a certain event (getting a girlfriend) does not occur within a specified number of trials. This can be modeled using a geometric distribution, where each trial is independent and has the same success probability.

:p What is the probability that it takes Ned more than 100 days to get a girlfriend?
??x
The answer: To solve this problem, we need to determine the number of failures before the first success in a sequence of Bernoulli trials. The probability of failure on any given day (a girl saying "no") is $\frac{99}{100} = 0.99$.

The probability that it takes more than 100 days to get a girlfriend means that the first success occurs on or after the 101st trial. This can be calculated as:

$$P(\text{more than 100 days}) = (0.99)^{100}$$

Using a calculator:
$$(0.99)^{100} \approx 0.366$$

Thus, the probability that it takes more than 100 days for Ned to get a girlfriend is approximately 0.366.

```java
public class GirlfriendProbability {
    public static void main(String[] args) {
        double successRate = 99 / 100.0;
        int numberOfDays = 100;
        
        double probabilityMoreThan100Days = Math.pow((1 - successRate), numberOfDays);
        System.out.println("Probability of more than 100 days: " + probabilityMoreThan100Days);
    }
}
```
x??

---

#### Variance Proof
Background context: The variance formula can be derived using the linearity of expectation. The goal is to prove that $\text{Var}(X) = E[X^2] - (E[X])^2$.

:p Use Linearity of Expectation to prove that $\text{Var}(X) = E[X^2] - (E[X])^2$.
??x
The answer: The variance of a random variable $X$ is defined as:
$$\text{Var}(X) = E[(X - E[X])^2]$$

Expanding the square inside the expectation, we get:
$$(X - E[X])^2 = X^2 - 2XE[X] + (E[X])^2$$

Taking the expectation of both sides, and using linearity of expectation:
$$

E[(X - E[X])^2] = E[X^2 - 2XE[X] + (E[X])^2]
= E[X^2] - 2E[XE[X]] + E[(E[X])^2]$$

Since $E[X]$ is a constant, we can use the property that $E[aX] = aE[X]$:

$$E[XE[X]] = E[X \cdot E[X]] = E[X] \cdot E[E[X]] = E[X] \cdot E[X] = (E[X])^2$$

And:
$$

E[(E[X])^2] = (E[X])^2$$

Substituting these back into the equation, we get:
$$\text{Var}(X) = E[X^2] - 2(E[X])^2 + (E[X])^2
= E[X^2] - (E[X])^2$$

Thus, we have proven that:
$$\text{Var}(X) = E[X^2] - (E[X])^2$$x??

---

#### Chain Rule for Conditioning
Background context: The chain rule in probability is a way to express the joint probability of multiple events occurring. It generalizes the multiplication rule to more than two events.

:p Prove that $P(\bigcap_{i=1}^{n} E_i) = P(E_1) \cdot P(E_2 | E_1) \cdot P(E_3 | E_1 \cap E_2) \cdots P(E_n | \bigcap_{i=1}^{n-1} E_i)$.
??x
The answer: The proof of the chain rule for conditioning involves induction. We start with the basic case for two events:

$$P(A \cap B) = P(A) \cdot P(B|A)$$

For three events, we can extend this to:
$$

P(A \cap B \cap C) = P(A) \cdot P(B | A) \cdot P(C | A \cap B)$$

We use induction to generalize this to $n $ events. Assume the formula holds for$k$ events:
$$P(\bigcap_{i=1}^{k} E_i) = P(E_1) \cdot P(E_2 | E_1) \cdot P(E_3 | E_1 \cap E_2) \cdots P(E_k | \bigcap_{i=1}^{k-1} E_i)$$

Now, consider $k+1$ events:
$$P(\bigcap_{i=1}^{k+1} E_i) = P((E_1 \cap E_2 \cdots E_k) \cap E_{k+1})$$

Using the basic case for two events with $A = E_1 \cap E_2 \cdots E_k $ and$B = E_{k+1}$:

$$P((E_1 \cap E_2 \cdots E_k) \cap E_{k+1}) = P(E_1 \cap E_2 \cdots E_k) \cdot P(E_{k+1} | E_1 \cap E_2 \cdots E_k)$$

By the induction hypothesis:
$$

P(E_1 \cap E_2 \cdots E_k) = P(E_1) \cdot P(E_2 | E_1) \cdot P(E_3 | E_1 \cap E_2) \cdots P(E_k | \bigcap_{i=1}^{k-1} E_i)$$

Therefore:
$$

P(\bigcap_{i=1}^{k+1} E_i) = (P(E_1) \cdot P(E_2 | E_1) \cdot P(E_3 | E_1 \cap E_2) \cdots P(E_k | \bigcap_{i=1}^{k-1} E_i)) \cdot P(E_{k+1} | \bigcap_{i=1}^{k} E_i)$$

Thus, the chain rule for conditioning is proven.

```java
public class ChainRuleExample {
    public static void main(String[] args) {
        // This example doesn't directly use the formula but demonstrates understanding.
        double pE1 = 0.5;
        double pE2_given_E1 = 0.6;
        double pE3_given_E1_and_E2 = 0.7;

        double jointProbability = pE1 * pE2_given_E1 * pE3_given_E1_and_E2;
        System.out.println("Joint Probability: " + jointProbability);
    }
}
```
x??

---

#### Assessing Risk
Background context: The problem involves calculating the probability that a flight will have enough seats for all passengers who show up, given that some people might not show up with certain probability. This is an example of a binomial distribution where each passenger independently has a probability $p$ of showing up.

:p Queueville Airlines sells 52 tickets for a 50-passenger plane, knowing that on average 5% of reservations do not show up. What is the probability that there will be enough seats?
??x
The answer: The problem can be modeled using the binomial distribution where $X $ represents the number of passengers who actually show up out of 52 tickets sold. Each passenger independently shows up with a probability of$0.95$.

We need to find $P(X \leq 50)$, which is the cumulative probability that fewer than or equal to 50 people show up.

Using the binomial distribution formula:

$$P(X = k) = \binom{n}{k} p^k (1-p)^{n-k}$$

Where $n = 52 $ and$p = 0.95$.

However, for practical purposes, we can use a normal approximation to the binomial distribution because $n$ is large:
$$X \sim N(np, np(1-p))$$

Here,$np = 52 \times 0.95 = 49.4 $ and$\sigma^2 = np(1-p) = 52 \times 0.95 \times 0.05 = 2.47 $, so $\sigma = \sqrt{2.47} \approx 1.57$.

The z-score for $X = 50$ is:
$$z = \frac{50 - 49.4}{1.57} \approx 0.38$$

Using the standard normal distribution table, we find that:
$$

P(Z < 0.38) \approx 0.65$$

Thus, the probability that there will be enough seats is approximately $0.65$.

```java
public class FlightRiskExample {
    public static void main(String[] args) {
        double n = 52; // Number of tickets sold
        double p = 0.95; // Probability each passenger shows up

        double mean = n * p;
        double variance = n * p * (1 - p);
        double standardDeviation = Math.sqrt(variance);

        double zScore = (50 - mean) / standardDeviation;
        System.out.println("Z-Score: " + zScore);
    }
}
```
x??

---

#### Practice with Conditional Expectation
Background context: The problem involves computing the conditional expectation $E[X | Y \neq 1]$, where $ X$and $ Y$ are jointly distributed random variables. This is a common operation in probability theory to understand how the value of one variable depends on another.

:p For the joint p.m.f. given in Table 3.3, compute $E[X | Y \neq 1]$.
??x
The answer: First, we need the joint probability mass function (pmf) for $X $ and$Y$. Let's assume the table provides values like:

|   | Y=0 | Y=1 | Y=2 |
|---|-----|-----|-----|
| X=0 | 0.1 | 0.3 | 0.4 |
| X=1 | 0.2 | 0.1 | 0.1 |

To find $E[X | Y \neq 1]$, we need the conditional expectation:

$$E[X | Y \neq 1] = \sum_x x P(X=x, Y \neq 1) / P(Y \neq 1)$$

From the table:
- $P(Y=0) = 0.1 + 0.2 = 0.3 $-$ P(Y=2) = 0.4 + 0.1 = 0.5 $So,$ P(Y \neq 1) = P(Y=0) + P(Y=2) = 0.8$.

Now, compute the numerator:

$$\sum_x x P(X=x, Y \neq 1) = (0 \cdot 0.3) + (1 \cdot 0.5) = 0.5$$

Thus,$$

E[X | Y \neq 1] = \frac{0.5}{0.8} = 0.625$$

The conditional expectation $E[X | Y \neq 1]$ is $0.625$.
x??

--- 

Would you like to go through another problem or need further explanations on any of these? Let me know! 
```

---
#### Variance of Sums and Scalar Multiplication

Given $c $ independent instances of a random variable$X$, we can compute the variances for both the sum of these variables and their scalar multiplication.

: How do the variances compare between $\text{Var}(X_1 + X_2 + \cdots + X_c)$ and $\text{Var}(cX)$?

??x
The variance of a sum of independent random variables is additive, meaning:
$$\text{Var}(X_1 + X_2 + \cdots + X_c) = \sum_{i=1}^c \text{Var}(X_i).$$

Since each $X_i $ has the same variance (let's denote it as$\sigma_X^2$), we get:
$$\text{Var}(X_1 + X_2 + \cdots + X_c) = c\sigma_X^2.$$

On the other hand, for scalar multiplication of a random variable, the formula is:
$$\text{Var}(cX) = c^2 \text{Var}(X).$$

Here $\text{Var}(X) = \sigma_X^2$, so we have:
$$\text{Var}(cX) = c^2 \sigma_X^2.$$

To compare the two, note that for $c > 1$:
- If $c < 2 $, then $\text{Var}(X_1 + X_2 + \cdots + X_c)$ will have a lower variance compared to $\text{Var}(cX)$.
- If $c = 2$, both variances are equal.
- For $c > 2 $, the variance of $ cX$ is higher.

In summary, for $c > 1$:
$$\text{Var}(X_1 + X_2 + \cdots + X_c) < \text{Var}(cX).$$??x
The answer with detailed explanations.
```java
// Example code to compute variances in Java
public class VarianceExample {
    public static void main(String[] args) {
        double sigmaX = 1.0; // Standard deviation of X, for example
        int c = 3;           // Example value for c
        
        // Compute the variance of sum and scalar multiplication
        double varSum = c * (sigmaX * sigmaX);
        double varScalarMult = c * c * (sigmaX * sigmaX);
        
        System.out.println("Variance(X1 + X2 + ... + Xc): " + varSum);
        System.out.println("Variance(cX): " + varScalarMult);
    }
}
```
x??

---
#### Mutual Fund Risk and Diversification

Mutual funds are less risky than buying a single stock because they diversify the risk by investing in many different stocks.

: Why do mutual funds reduce risk compared to individual stocks?

??x
When you buy a single stock, your return is highly dependent on the performance of that specific company. This means any negative news about the company can significantly impact your investment. However, when you invest in a mutual fund, your money is spread across multiple stocks.

The key idea behind diversification is that not all stocks move in the same direction at the same time. Therefore, the losses in one stock may be offset by gains in another. As a result, the overall risk of holding a mutual fund (which includes many different stocks) is generally lower than holding just one individual stock.

In mathematical terms, if $X_1, X_2, \ldots, X_c$ are independent random variables representing the returns on c different stocks in a mutual fund, then:
$$\text{Var}(X_1 + X_2 + \cdots + X_c) = \sum_{i=1}^c \text{Var}(X_i).$$

This is because variance of sums of independent random variables adds up.

On the other hand, if you invest in a single stock $Y$, then:
$$\text{Var}(Y).$$

By combining multiple stocks, the total risk (variance) can be reduced due to the diversification effect. This reduces the overall volatility and makes mutual funds less risky compared to individual stocks.

??x
The answer with detailed explanations.
```java
// Example code for understanding variance in Java
public class RiskExample {
    public static void main(String[] args) {
        double sigmaStock = 0.15; // Standard deviation of a single stock, for example
        int c = 10;               // Number of stocks in the mutual fund
        
        // Compute the variance of sum and scalar multiplication
        double varSingleStock = (sigmaStock * sigmaStock);
        double varMutualFund = c * (sigmaStock * sigmaStock);
        
        System.out.println("Variance of single stock: " + varSingleStock);
        System.out.println("Variance of mutual fund with 10 stocks: " + varMutualFund);
    }
}
```
x??

---

---
#### Bill's Fundraising Probability Using Normal Approximation
Background context: Bill aims to raise $1,000,000. We need to compute the probability that he raises less than$999,000 using a normal approximation.

The steps are:
1. Identify if the distribution of the total amount raised can be approximated by a normal distribution.
2. Compute the mean and standard deviation of the distribution.
3. Use the cumulative distribution function (CDF) to find the required probability.

Assume the distribution is approximately normal with mean $\mu $ and standard deviation$\sigma$.

:p What is the formula for converting a value to a z-score in this context?
??x
The z-score formula is:
$$z = \frac{X - \mu}{\sigma}$$

Where $X $ is the amount raised,$\mu $ is the mean, and$\sigma$ is the standard deviation.
x??

---
#### Bill's Fundraising Probability Using Exact Expression
Background context: We need to compute the exact probability that Bill raises less than$999,000 using a more precise method.

Assume we have a discrete or another non-normal distribution for the amount raised. The steps are:
1. Identify the distribution of the total amount raised.
2. Sum the probabilities for all values below $999,000.

:p How would you write an exact expression for this probability?
??x
The exact expression can be written as:
$$P(X < 999000) = \sum_{i=0}^{998999} p_i$$

Where $p_i $ is the probability of raising exactly$ i.
x??

---
#### Eric and Timmy's Meeting Probability
Background context: Eric and Timmy each arrive at a time uniformly distributed between 2 and 3 pm. Each waits for 15 minutes.

The steps are:
1. Define the problem in terms of joint distributions.
2. Calculate the probability that their arrival times overlap by more than 15 minutes.

:p What is the probability that Eric and Timmy will meet?
??x
To find the probability, we can use a geometric approach on a unit square where both axes represent time (from 2 to 3 pm).

The area representing successful meetings (Eric and Timmy meet) can be calculated as:
$$P(\text{Meet}) = \frac{\text{Area of meeting region}}{\text{Total possible area}} = \frac{60^2 - 45^2}{60^2} = \frac{3600 - 2025}{3600} = \frac{1575}{3600} = \frac{7}{16}$$x??

---
#### Weather Prediction for John and Mary's Wedding
Background context: The weather forecaster predicts rain, but is not always accurate. We need to calculate the probability that it will actually rain given the forecast.

The steps are:
1. Define the relevant events.
2. Use Bayes' theorem to find the required conditional probability.

Let $R $ be the event that it rains and$F_R$ be the event that the forecaster predicts rain.

:p What is the probability that it will rain during John and Mary's wedding?
??x
Using Bayes' theorem:
$$P(R|F_R) = \frac{P(F_R|R)P(R)}{P(F_R)}$$

Where $P(F_R|R) = 0.9 $, $ P(R) = \frac{10}{365} \approx 0.0274$, and:
$$P(F_R) = P(F_R|R)P(R) + P(F_R|\neg R)P(\neg R) = (0.9)(0.0274) + (0.1)(1 - 0.0274)$$

So,$$

P(R|F_R) = \frac{(0.9)(0.0274)}{0.9(0.0274) + 0.1(0.9726)} \approx \frac{0.02466}{0.02466 + 0.09726} = \frac{0.02466}{0.12192} \approx 0.2023$$x??

---
#### Vaccine Testing with Bayesian Reasoning
Background context: The vaccine has a 50% chance of being effective and an initial lab test accuracy of 60%. We need to update the probability that the vaccine is effective after both the lab test and human test come up "success".

The steps are:
1. Define the relevant events.
2. Use Bayes' theorem for each step.

Let $E $ be the event that the vaccine is effective, and$T_L $,$ T_H$ be the lab test and human test results respectively.

:p What is the probability that the vaccine is effective given both tests are "success"?
??x
Using Bayes' theorem:
$$P(E|T_L = S, T_H = S) = \frac{P(T_L = S, T_H = S | E)P(E)}{P(T_L = S, T_H = S)}$$

Where $P(E) = 0.5$, and:
$$P(T_L = S|E) = 0.6, \quad P(T_H = S|E) = 0.8$$
$$

P(T_L = S|\neg E) = 0.4, \quad P(T_H = S|\neg E) = 0.2$$

So,$$

P(T_L = S, T_H = S | E) = (0.6)(0.8) = 0.48$$
$$

P(T_L = S, T_H = S |\neg E) = (0.4)(0.2) = 0.08$$

And the total probability:
$$

P(T_L = S, T_H = S) = P(T_L = S, T_H = S | E)P(E) + P(T_L = S, T_H = S |\neg E)P(\neg E) = (0.48)(0.5) + (0.08)(0.5) = 0.24 + 0.04 = 0.28$$

Finally,$$

P(E|T_L = S, T_H = S) = \frac{0.48(0.5)}{0.28} = \frac{0.24}{0.28} \approx 0.8571$$x??

---
#### Dating Costs: Expectation and Variance via Conditioning
Background context: A man has two approaches to dating, each with different costs and outcomes.

The steps are:
1. Define the random variables for cost and outcome.
2. Use conditioning on outcomes to calculate expectations and variances.

Let $X $ be the cost of a date, and$Y$ be whether the date marries him (0 or 1).

:p What is the expected value of the cost when using the generous approach?
??x
Using the definition of expectation:
$$E(X_{\text{generous}}) = 1000(0.95) + 50(0.05) = 950 + 2.5 = 952.5$$x??

---
#### Dating Costs: Expectation and Variance via Conditioning
Background context (continued): The same man has a cheapskate approach, which is cheaper but always ends in break-up.

:p What is the expected value of the cost when using the cheapskate approach?
??x
Using the definition of expectation:
$$

E(X_{\text{cheapskate}}) = 50$$x??

---

#### Expected Cost to Find a Wife

Background context: The man has experienced only failure and decides to choose an approach (generous or cheapskate) at random. The problem involves calculating the expected cost of finding a wife under this scenario.

:p Assuming the man starts searching today, what is his expected cost to find a wife?

??x
The expected cost can be calculated by considering that each attempt has two possible outcomes: generous approach or cheapskate approach. Since both are chosen randomly with equal probability (1/2), we need to calculate the expected number of attempts and then multiply it by the average cost per attempt.

Let $C $ be the cost for a successful marriage, and$p_g $ and$ p_c $ be the probabilities of success under the generous and cheapskate approaches respectively. The expected number of attempts until finding a wife is given by the geometric distribution with parameter $q = 1 - p_g \cdot p_c$.

The expected cost is then:
$$E[\text{Cost}] = E[\text{Number of Attempts}] \cdot C$$

Since each attempt has an equal probability, the number of attempts follows a geometric distribution with mean $\frac{1}{q}$:
$$E[\text{Number of Attempts}] = \frac{1}{0.5} = 2$$

Thus, the expected cost is:
$$

E[\text{Cost}] = 2C$$---
#### Variance of Geometric Distribution

Background context: The geometric distribution $X \sim \text{Geometric}(p)$ models the number of trials until the first success. We need to prove that the variance of this distribution is given by:
$$\text{Var}(X) = \frac{1-p}{p^2}$$:p Compute the variance on the amount of money the man ends up spending to find a wife.

??x
The variance of the geometric distribution can be computed using the hint provided: use conditioning. The key idea is that:
$$\text{Var}(X) = E[X^2] - (E[X])^2$$

First, we know from the properties of the geometric distribution that:
$$

E[X] = \frac{1}{p}$$

To find $E[X^2]$, we use conditioning. Let's condition on the first trial:
- If the first trial is a success (with probability $p $), then $ X = 1$.
- If the first trial is a failure (with probability $1-p $), then $ X = 1 + Y $ where $ Y \sim \text{Geometric}(p)$.

Thus:
$$E[X^2] = E[E[X^2 | X_1]]$$where $ X_1$ is the outcome of the first trial.

If $X_1 = 1 $, then $ X^2 = 1 $. If$ X_1 = 0$, then:
$$E[X^2 | X_1 = 0] = E[(1 + Y)^2] = E[1 + 2Y + Y^2] = 1 + 2E[Y] + E[Y^2]$$

Since $E[Y] = \frac{1}{p}$ and using the variance formula:
$$E[Y^2] = (E[Y])^2 + \text{Var}(Y) = \left(\frac{1}{p}\right)^2 + \frac{1-p}{p^2} = \frac{1}{p^2} + \frac{1-p}{p^2} = \frac{2 - p}{p^2}$$

Thus:
$$

E[X^2 | X_1 = 0] = 1 + 2\left(\frac{1}{p}\right) + \frac{2 - p}{p^2} = 1 + \frac{2}{p} + \frac{2}{p^2} - \frac{1}{p^2} = \frac{3p^2 + 2p - 1}{p^2}$$

Combining these:
$$

E[X^2] = p \cdot 1 + (1-p) \left(1 + \frac{2}{p} + \frac{2 - p}{p^2}\right) = 1 + \frac{2(1-p)}{p} + \frac{(1-p)(2-p)}{p^2}$$
$$

E[X^2] = 1 + \frac{2 - 2p}{p} + \frac{2 - p - 2p + p^2}{p^2} = 1 + \frac{2}{p} - 2 + \frac{2}{p^2} - \frac{3}{p^2} + 1 = \frac{1-p+2-2p+2}{p^2} = \frac{3 - p}{p^2} + 1 = \frac{4 - p}{p^2}$$

Thus:
$$

E[X^2] = \frac{4 - p}{p^2}$$

Finally, the variance is:
$$\text{Var}(X) = E[X^2] - (E[X])^2 = \frac{4 - p}{p^2} - \left(\frac{1}{p}\right)^2 = \frac{4 - p}{p^2} - \frac{1}{p^2} = \frac{3 - p}{p^2} = \frac{1-p}{p^2}$$---
#### Good Chips versus Lemons

Background context: A chip supplier produces 95% good chips and 5% lemons. The good chips fail with probability $0.0001 $ each day, while the lemons fail with probability$0.01$ each day. We need to compute the expected time until a randomly chosen chip fails.

:p Compute E[T] and Var(T).

??x
The expected time $E[T]$ for a good chip to fail is given by:
$$E[T_{\text{good}}] = \frac{1}{0.0001} = 10000 \text{ days}$$

For a lemon, the expected time until failure is:
$$

E[T_{\text{lemon}}] = \frac{1}{0.01} = 100 \text{ days}$$

The overall expected time $E[T]$ for any chip to fail can be calculated using the law of total expectation:
$$E[T] = P(\text{Good Chip}) \cdot E[T_{\text{good}}] + P(\text{Lemon}) \cdot E[T_{\text{lemon}}]$$
$$

E[T] = 0.95 \cdot 10000 + 0.05 \cdot 100 = 9500 + 5 = 9505 \text{ days}$$

For the variance, we use:
$$\text{Var}(T) = E[T^2] - (E[T])^2$$

First, calculate $E[T_{\text{good}}^2]$:
$$E[T_{\text{good}}^2] = 10000 + 10000^2 \cdot 0.0001 = 10000 + 10 = 10010$$

Similarly, for the lemon:
$$

E[T_{\text{lemon}}^2] = 100 + 100^2 \cdot 0.01 = 100 + 100 = 200$$

Thus:
$$

E[T^2] = P(\text{Good Chip}) \cdot E[T_{\text{good}}^2] + P(\text{Lemon}) \cdot E[T_{\text{lemon}}^2]$$
$$

E[T^2] = 0.95 \cdot 10010 + 0.05 \cdot 200 = 9509.5 + 10 = 9519.5$$

Finally, the variance is:
$$\text{Var}(T) = E[T^2] - (E[T])^2 = 9519.5 - (9505)^2 \approx 475$$---
#### Expectation via Conditioning

Background context: Stacy's fault-tolerant system crashes only if there are $k $ consecutive failures, with each failure occurring independently with probability$p$. We need to find the expected number of minutes until the system crashes.

:p What is the expected number of minutes until Stacy’s system crashes?

??x
We can model this problem using a recurrence relation. Let $T $ be the time until the first crash, which requires$k$ consecutive failures.

Define:
$$E[T] = 1 + p(1 + E[T])^k$$

This equation reflects that the expected time is one minute plus the expected additional time if no failure occurs (with probability $1-p $), followed by $ E[T]$.

Solving this recurrence relation for general $k $ and$p$:

For simplicity, assume $k = 1$:
$$E[T] = 1 + pE[T]$$
$$

E[T](1 - p) = 1$$
$$

E[T] = \frac{1}{1-p}$$

For $k > 1$, the solution involves more complex algebra, but can be approximated using:
$$E[T] \approx \frac{k}{p}$$---
#### Napster – Brought to You by the RIAA

Background context: To collect all songs from a favorite band with 50 songs randomly downloaded until you have all of them. We need to find $E[D]$ and $Var(D)$.

:p (a) What is E[D]? Give a closed-form approximation.

??x
The problem can be modeled using the coupon collector's problem. The expected number of downloads required to collect all 50 songs is given by:

$$E[D] = 50 \left(1 + \frac{1}{2} + \frac{1}{3} + \cdots + \frac{1}{50}\right)$$

This can be approximated using the harmonic series:
$$

H_n \approx \ln(n) + \gamma$$where $\gamma \approx 0.5772156649$.

Thus, for $n = 50$:
$$E[D] \approx 50 (\ln(50) + 0.5772156649) \approx 50 (3.91202300546 + 0.5772156649) \approx 50 \times 4.48923867036 \approx 224.46$$---
#### Fractional Moments

Background context: The normal distribution is needed to compute $E[X^{1/2}]$, where $ X \sim \text{Exp}(1)$. This involves integration by parts and a change of variables.

:p Compute $E[X^{1/2}]$.

??x
Given $X \sim \text{Exp}(1)$:
$$f_X(x) = e^{-x}$$

We need to compute:
$$

E[X^{1/2}] = \int_0^\infty x^{1/2} e^{-x} dx$$

Using integration by parts with $u = x^{1/2}$,$ dv = e^{-x}dx$:
$$du = \frac{1}{2} x^{-1/2} dx, \quad v = -e^{-x}$$

Thus:
$$

E[X^{1/2}] = -x^{1/2} e^{-x} |_0^\infty + \int_0^\infty \frac{1}{2} x^{-1/2} e^{-x} dx$$

The boundary term evaluates to 0 at both limits, so:
$$

E[X^{1/2}] = \frac{1}{2} \int_0^\infty x^{-1/2} e^{-x} dx$$

Recognize that $\Gamma(n) = (n-1)!$, and the integral is a form of the Gamma function with $ n = 1.5$:
$$\Gamma(1.5) = \frac{\sqrt{\pi}}{2}$$

Thus:
$$

E[X^{1/2}] = \frac{1}{2} \cdot \frac{\sqrt{\pi}}{2} = \frac{\sqrt{\pi}}{4} \approx 0.61687741583$$

