# Flashcards: 2B001--Financial-Data-Engineering_processed (Part 18)

**Starting Chapter:** Dimension 6 Data Availability and Completeness

---

#### Data Availability and Completeness
Data availability and completeness are critical dimensions of data quality, particularly important in finance. A dataset is considered incomplete or unavailable when essential data attributes or observations are missing due to various reasons such as voluntary reporting mechanisms, security concerns, market factors, technological reasons, publication delays, and the existence of a specific type of data with optional status.

:p What are the common causes of data availability and completeness issues in finance?
??x
Common causes include:
1. **Voluntary Data Reporting**: Respondents may decline to report certain information.
2. **Security and Confidentiality Concerns**: Firms might be reluctant to share sensitive data.
3. **Market Factors**: Illiquid instruments have fewer price observations, leading to nonsynchronous trading.
4. **Technological Reasons**: Lack of infrastructure for collecting OTC market transactions.
5. **Publication Delay**: Data creation and publication may not align in time.
6. **Data Time to Live (TTL)**: Data is considered expired after a certain period but may persist.

x??

---

#### Missing Data Mechanisms
Missing data can be categorized into three types: Missing Completely at Random (MCAR), Missing at Random (MAR), and Missing Not at Random (MNAR).

- **MCAR**: The missingness of the data on variable \(X\) is unrelated to any other variables in the dataset, whether observed or not.
- **MAR**: The missingness depends on other variables in the dataset but is independent of the value of the variable itself.
- **MNAR**: The missing values are related to the unobserved variable.

:p What does MCAR stand for and how is it defined?
??x
MCAR stands for "Missing Completely at Random." It means that the missingness of data on variable \(X\) is unrelated to any other variables in the dataset, whether observed or not. 

Example:
- A hedge fund reports its performance data randomly without considering its actual performance.

x??

---

#### Missing Data Mechanisms
Continuing from MCAR, MAR stands for "Missing at Random." The missingness depends on other variables in the dataset but is independent of the value of the variable itself.

- **MAR**: The mechanism that leads to observations being missing is related to one or more features in the dataset.
- Example: A hedge fund might not disclose performance data due to confidentiality concerns about its investment strategy, unrelated to its actual performance.

:p What does MAR stand for and how is it defined?
??x
MAR stands for "Missing at Random." It means that the missingness of data on variable \(X\) is related to other variables in the dataset but is independent of the value of the variable itself. 

Example:
- A hedge fund decides not to disclose its performance data due to confidentiality concerns about its investment strategy, unrelated to its actual performance.

x??

---

#### Missing Data Mechanisms
The final category of missing data mechanisms is MNAR (Missing Not at Random). In this case, the observations are missing for reasons related to the unobserved variable itself.

- **MNAR**: The mechanism that leads to observations being missing depends on the value of the variable \(X\).
- Example: A hedge fund decides not to report its performance data because it had a bad performance and wants to hide it from investors, or they are doing very well and do not want to attract more attention.

:p What does MNAR stand for and how is it defined?
??x
MNAR stands for "Missing Not at Random." It means that the missingness of data on variable \(X\) depends on the value of the variable itself. 

Example:
- A hedge fund decides not to report its performance data because it had a bad performance and wants to hide it from investors, or they are doing very well and do not want to attract more attention.

x??

---

#### Data Deletion Technique

Data deletion is a straightforward method where observations containing missing data are removed from the dataset. While this technique simplifies handling, it can lead to biased or sparse datasets if not used judiciously.

:p What does the data deletion technique involve?
??x
The data deletion technique involves removing any observation that contains missing data from the dataset.
x??

---

#### Omitted Variable Approach

Another approach is the omitted variable approach, where a variable with missing values is completely removed from the dataset. This can also result in loss of information and potentially biased results.

:p What does the omitted variable approach entail?
??x
The omitted variable approach involves removing an entire variable from the dataset if it contains missing values.
x??

---

#### Imputation Techniques

Imputation techniques aim to estimate missing values using specific methods. Common imputation techniques include mean substitution, entity resolution, and regression models.

:p What is a basic imputation technique?
??x
A basic imputation technique is mean substitution, which replaces missing values for variable X with the average value of X.
x??

---

#### Regression Model for Imputation

Regression-based imputation uses a model to predict missing values. This can be machine learning based or financial models like linear regression.

:p How does regression modeling work in data imputation?
??x
Regression modeling works by using historical data to build a predictive model that estimates the missing values. For instance, a linear regression model could use other variables to predict the value of X with missing data.
x??

---

#### Data Timeliness

Data timeliness is crucial for financial institutions as it ensures data availability and accuracy at the time it is needed. Key aspects include whether data is available when expected and if it reflects the most recent observations.

:p What does data timeliness ensure?
??x
Data timeliness ensures that data is available and up-to-date, which is critical for making timely decisions in financial contexts.
x??

---

#### Latency and Market Closure

Latency and market closure are common factors affecting data timeliness. Complex data pipelines can introduce delays, while market closures prevent real-time updates.

:p What are some factors influencing data timeliness?
??x
Factors such as latency (delay in data processing), market closure periods, time lags, and lengthy processes in data generation mechanisms can affect data timeliness.
x??

---

#### Data Refresh Rate

The refresh rate is the frequency at which data is updated to ensure it reflects the latest observations. This can range from real-time updates to regular schedules.

:p What is a data refresh rate?
??x
A data refresh rate refers to how frequently data is refetched and updated to reflect the most recent observations, ranging from real-time to scheduled updates.
x??

---

#### Data Caching

Data caching involves storing copies of data in temporary locations for fast access. However, cached data can become outdated over time, requiring periodic updates.

:p What is data caching?
??x
Data caching is a strategy where copies of data are stored temporarily to enable quick access. However, this cached data may need regular updates to stay current.
x??

---

#### Cache Invalidiation

Cache invalidation occurs when cached data becomes outdated and needs to be refreshed or replaced with the latest data.

:p What is cache invalidation?
??x
Cache invalidation refers to the process of ensuring that cached data remains up-to-date, requiring periodic updates to prevent using stale information.
x??

---

#### Data Constraints

Data constraints ensure that data adheres to predefined technical and business rules. Examples include extension, schema, non-null, range, value choice, uniqueness, referential integrity, and regular expression constraints.

:p What are some examples of data constraints?
??x
Examples of data constraints include:
- Extension constraint: Data is stored in allowed formats.
- Schema constraint: Data follows a predefined structure.
- Non-null constraint: A field does not contain null values.
- Range constraint: Values fall within given ranges.
- Value choice constraint: Fields assume values from fixed lists.
- Uniqueness constraint: Records must be unique across datasets.
- Referential integrity constraint: Values in one field are allowed only if they exist in another referenced field.
- Regular expression patterns: Fields contain values matching specific string patterns.
x??

---

#### Data Relevance

Data relevance ensures that available data aligns with the problem or purpose it aims to address, making it actionable and insightful.

:p What does data relevance ensure?
??x
Data relevance ensures that data is pertinent to the specific analytical problem, providing insights necessary for understanding and addressing the issue at hand.
x??

---

#### Financial Data Timeliness Challenges

Financial markets use stale price terms to describe outdated or inaccurate quoted values. Factors affecting timeliness include latency, market closure, time lags, and lengthy data processing mechanisms.

:p What challenges can affect financial data timeliness?
??x
Challenges affecting financial data timeliness include:
- Latency (processing delays)
- Market closures preventing real-time updates
- Time lags in data generation processes
- Lengthy data ingestion and transformation processes
x??

---

---
#### Trading Strategies and Data Requirements
Background context: The passage recommends a book for learning about trading strategies and their data requirements, emphasizing that traders need time series data for transactions (prices and volumes), orders, and news (economic releases and events).
:p What are the key types of data required for trading strategies?
??x
The key types of data required for trading strategies include:
- Time series data for transaction prices and volumes.
- Order book data.
- News and other economic event data.

These data points help traders make informed decisions based on market trends, liquidity, and external events. For example, a trader might use price and volume data to identify patterns and trends in the market, while news data could signal changes that affect asset prices.

```java
public class TradingStrategyData {
    private double[] transactionPrices;
    private long[] transactionVolumes;
    private List<Order> orders;
    private NewsEvent[] newsEvents;

    // Constructor and methods for initializing and processing these data types
}
```
x??
---

#### Data Integrity Principles in Financial Sector
Background context: The passage outlines nine key principles that ensure data integrity throughout its lifecycle, including standards, backups, archiving, lineage, catalogs, ownership, contracts, and reconciliation. These principles are crucial for maintaining consistent, traceable, usable, and reliable financial data.
:p What is the importance of ensuring data integrity in the financial sector?
??x
Ensuring data integrity is essential in the financial sector to maintain market efficiency, confidence, and stability while reducing costs. Data integrity helps prevent errors, ensures that data remains accurate and up-to-date, and supports compliance with regulatory requirements.

For instance, data standards can help align data across different systems and institutions, making it easier to integrate and use data effectively. Backups ensure that data can be recovered in case of loss or corruption, while contracts define responsibilities for maintaining data integrity among stakeholders.

```java
public class DataIntegrityCheck {
    public boolean checkDataIntegrity(Data[] data) {
        // Implement checks based on the nine principles outlined in the passage
        if (data.isStandardized() && data.hasBackups() && data.isArchived()) {
            return true;
        }
        return false;
    }
}
```
x??
---

#### Data Standards and Standardization
Background context: The concept of standards is crucial for financial markets, as it helps ensure uniformity in measures, agreements, conditions, or specifications between parties. According to Spivak and Brenner, standards are categorized into physical standards, terms and definitions, test methods, and more.
:p What do financial industry participants call for regarding standardization?
??x
Financial industry participants have called for standardization to increase market efficiency, confidence, and stability while reducing costs. For example, the adoption of the Legal Entity Identifier (LEI) standard can streamline regulatory compliance processes.

```java
public class StandardizationExample {
    public void implementStandard(String standardType) {
        switch (standardType) {
            case "Physical":
                // Implement physical standards like units of measure
                break;
            case "Terms and Definitions":
                // Implement terms, definitions, classes, grades, etc.
                break;
            default:
                throw new IllegalArgumentException("Unsupported standard type");
        }
    }
}
```
x??
---

