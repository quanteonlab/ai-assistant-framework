# Flashcards: 2B001--Financial-Data-Engineering_processed (Part 16)

**Starting Chapter:** Entity Resolution Software Libraries. Chapter 5. Financial Data Governance

---

#### Entity Resolution Software Libraries
Entity resolution is a well-known problem with extensive development and application. Many open-source tools have been developed, including fastLink, Dedupe, Splink, JedAI, RecordLinkage, Zingg, Ditto, and DeepMatcher. On the commercial side, vendors such as TigerGraph, Tamr, DataWalk, Senzing, Hightouch, and Quantexa offer ER solutions.
:p What are some examples of open-source tools for entity resolution?
??x
Examples include fastLink, Dedupe, Splink, JedAI, RecordLinkage, Zingg, Ditto, and DeepMatcher. These tools provide various functionalities to address the challenges in entity resolution across different datasets.
x??

---

#### Entity Resolution Software Libraries (Commercial)
In addition to open-source solutions, several commercial vendors offer ER tools and services. Some notable examples include TigerGraph, Tamr, DataWalk, Senzing, Hightouch, and Quantexa. These vendors typically provide more advanced features and comprehensive support compared to open-source alternatives.
:p What are some commercial vendors that offer entity resolution solutions?
??x
Commercial vendors such as TigerGraph, Tamr, DataWalk, Senzing, Hightouch, and Quantexa provide ER tools with advanced features and comprehensive support. These vendors often cater to enterprises requiring more robust and scalable solutions.
x??

---

#### Financial Entity Systems Challenges
This chapter discusses two primary challenges faced by financial institutions: named entity recognition (NER) and entity resolution (ER). NER involves extracting and identifying financial entities from both structured and unstructured data, while ER focuses on matching the same entity across multiple datasets. These tasks are complex and dynamic, requiring continuous updates and advancements in methodologies.
:p What are the two primary challenges discussed in this chapter?
??x
The two primary challenges discussed are named entity recognition (NER) and entity resolution (ER). NER involves extracting entities from financial data, while ER focuses on matching entities across multiple datasets.
x??

---

#### Financial Entity Systems Challenges: Named Entity Recognition (NER)
Named entity recognition is a crucial task in the context of financial data processing. It involves identifying and categorizing named entities like companies, individuals, addresses, dates, and amounts within structured and unstructured data sources. This process helps in extracting valuable information for further analysis.
:p What does named entity recognition involve?
??x
Named entity recognition involves identifying and categorizing named entities such as companies, individuals, addresses, dates, and amounts from both structured and unstructured financial data sources. This process is essential for extracting meaningful insights.
x??

---

#### Financial Entity Systems Challenges: Entity Resolution (ER)
Entity resolution focuses on matching records that refer to the same entity across multiple datasets. It involves determining whether two or more records represent the same individual, organization, or object by analyzing common attributes such as names, addresses, and IDs. This task is critical for data quality and consistency.
:p What does entity resolution focus on?
??x
Entity resolution focuses on matching records that refer to the same entity across multiple datasets by analyzing common attributes like names, addresses, and IDs. This ensures data quality and consistency.
x??

---

#### Financial Entity Systems Challenges: Dynamic Evolution of Solutions
The landscape of challenges and solutions in financial NER and ER is dynamic, evolving alongside advancements in data technologies and changing market requirements. To stay competitive, it's essential to be familiar with the latest updates, methodologies, and best practices.
:p How does the landscape of financial NER and ER evolve?
??x
The landscape of financial NER and ER evolves continuously due to advancements in data technologies and changes in market requirements. Staying current is crucial for maintaining a competitive edge.
x??

---

#### Financial Entity Systems Challenges: Importance of Domain Knowledge
To enhance the accuracy and efficiency of NER and ER systems, it's important to enrich your financial domain knowledge. Understanding the context and nuances of financial data can significantly improve system performance.
:p Why is financial domain knowledge important in NER and ER?
??x
Financial domain knowledge is crucial because understanding the context and nuances of financial data can significantly enhance the accuracy and efficiency of NER and ER systems by providing better context for entity recognition and resolution.
x??

---

#### Financial Entity Systems Challenges: Future Focus - Data Governance
The next chapter will explore the critical problem of financial data governance, focusing on concepts and best practices to ensure data quality, integrity, security, and privacy in the financial domain.
:p What is the focus of the next chapter?
??x
The next chapter focuses on financial data governance, exploring concepts and best practices for ensuring data quality, integrity, security, and privacy in the financial domain.
x??
---

#### Definition of Financial Data Governance
Background context explaining financial data governance. According to Google Cloud and Eryurek et al., data governance involves ensuring data security, privacy, accuracy, availability, and usability, encompassing actions, processes, and technology throughout the data life cycle.

:p What is the definition provided for financial data governance?
??x
Financial data governance is a technical and cultural framework that establishes rules, roles, practices, controls, and implementation guidelines to ensure the quality, integrity, security, and privacy of financial data in compliance with both general and financial domain-specific internal and external policies, standards, requirements, and regulations.
x??

---

#### Importance of Financial Data Governance
Background context discussing why financial institutions need and benefit from data governance. It involves addressing performance and risk management issues.

:p Why is defining and enforcing an effective financial data governance framework important for financial institutions?
??x
Defining and enforcing a financial data governance framework ensures high data quality, which impacts operational efficiency and decision-making in financial institutions. Additionally, it helps manage risks such as cyberattacks, data breaches, discriminatory biases, erratic model inputs, and noncompliance with legal and regulatory requirements.
x??

---

#### Components of Financial Data Governance
Background context on the three key components: data quality, data integrity, and data security and privacy.

:p What are the three main components of financial data governance?
??x
The three main components of financial data governance are:
1. **Data Quality**: Ensuring that data is accurate, complete, consistent, and relevant.
2. **Data Integrity**: Maintaining the accuracy and consistency of the data throughout its lifecycle.
3. **Data Security and Privacy**: Protecting the confidentiality, integrity, and availability of sensitive financial data.

For example, ensuring that financial records are free from errors (data quality) and that access controls are in place to prevent unauthorized access (data security).
x??

---

#### Value Proposition for Financial Data Governance
Background context on the value proposition of financial data governance. It includes performance improvement and risk management.

:p What are the key benefits of implementing financial data governance in a financial institution?
??x
The key benefits of implementing financial data governance include:
1. **Performance Improvement**: High data quality standards drive efficient operations, better insights into market activity, informed investment decisions, timely responses to new events, and accurate communication with stakeholders.
2. **Risk Management**: Mitigates risks such as cyberattacks, data breaches, model biases, data loss due to backups, decentralized processes, lack of oversight, privacy risks, and nonconformance with legal regulations.

For instance, solid governance ensures that financial institutions can operate more smoothly by reducing the need for constant quality checks.
x??

---

#### Regulatory Requirements
Background context on the regulatory requirements driving the need for financial data governance. Examples include Sarbanes-Oxley Act, Bank Secrecy Act, and GDPR.

:p What are some of the regulatory requirements driving the adoption of data governance in financial institutions?
??x
Some key regulatory requirements driving the adoption of data governance in financial institutions include:
- **Sarbanes–Oxley Act**: Ensures accurate financial reporting.
- **Bank Secrecy Act**: Mandates measures to prevent money laundering.
- **Basel Committee on Banking Supervision’s standard number 239 (BCBS 239)**: Focuses on data quality and integrity in banking.
- **European Union’s Solvency II Directive**: Addresses solvency and risk management of insurance companies.
- **California Consumer Privacy Act (CCPA)**: Grants consumers rights over their personal information.
- **EU’s General Data Protection Regulation (GDPR)**: Protects the privacy and data rights of EU citizens.

These regulations underscore the importance of robust data governance to ensure compliance.
x??

---

#### Data Quality Framework (DQF)
Background context explaining the concept. The DQF is essential for ensuring that financial data satisfies its intended use, particularly in decision-making and product development processes within financial institutions.
If applicable, add code examples with explanations.
:p What is a Data Quality Framework (DQF) used for?
??x
A Data Quality Framework ensures that financial data meets the requirements needed for various operational and analytical applications. It helps maintain high-quality data to support critical business operations like decision-making and product development.
??x

---

#### Importance of Data Quality in Financial Institutions
Background context explaining why data quality is crucial, especially with examples provided in the text.
:p Why is data quality so important in financial institutions?
??x
Data quality is vital because it directly impacts decision-making processes, operational efficiency, customer trust, and regulatory compliance. Poor data quality can lead to significant issues such as inaccurate reporting, missed opportunities for investment, and legal penalties.
??x

---

#### Data Errors in Financial Data
Background context explaining common types of errors in financial data with examples provided in the text.
:p What are some common types of data errors found in financial data?
??x
Common types of data errors include:
- Random measurement errors (e.g., $9.345 instead of $9.335)
- Wrong decimal places (e.g., a price of $111.34 instead of $11.134)
- Decimal precision issues (e.g., an exchange rate of 1.345 instead of 1.3458)
- Negative prices (e.g., one Apple stock is worth $-200)
- Dummy and test quotes
- Extra or removed digits
- Invalid date entries
- Inverted exchange rates
- Rounding issues
- Misspelled entity names
- Typos
- Invalid formatting
??x

---

#### Detecting Data Errors in Financial Data
Background context explaining how data errors can be detected, including the complexity of detection.
:p How are data errors detected in financial data?
??x
Data errors can be detected at both the single-record and dataset levels. Simple errors (e.g., negative prices) might use rule-based approaches (if the price is negative → error). More complex errors require statistical and data mining techniques, such as analyzing trends or patterns over time.
For example, a simple detection approach in pseudocode could be:
```pseudocode
function detectNegativePrices(priceList):
    for each price in priceList:
        if price < 0:
            return true
    return false
```
Complex errors might involve more sophisticated statistical analysis. For instance, identifying subtle intraday patterns that deviate from expected behavior.
??x

---

#### Financial Data Quality Dimensions (DQDs)
Background context explaining the nine DQDs presented in the text and their relevance to financial data.
:p What are the nine dimensions of data quality discussed for financial data?
??x
The nine dimensions of data quality discussed for financial data include:
1. Errors: Incorrect or invalid values.
2. Outliers: Values that significantly deviate from expected ranges.
3. Biases: Systematic deviations from true values.
4. Granularity: Appropriate detail level of the data.
5. Duplicates: Multiple records with identical information.
6. Availability and Completeness: Absence of missing or incomplete data.
7. Timeliness: Data is up-to-date and relevant for decision-making.
8. Constraints: Compliance with business rules or standards.
9. Relevance: Data aligns with the intended use in specific applications.
??x

---

#### Example of Data Error Impact
Background context explaining how a small error can significantly affect financial calculations.
:p How does a slight data error impact financial analysis?
??x
A small data error, such as a change in an exchange rate, can have significant impacts on financial calculations. For instance:
- If the correct exchange rate is 1 EUR = 1.07291 USD and it's mistakenly used as 1.07191 USD, this results in a $1 million loss.
- A decimal precision error using an incorrect exchange rate of 1.072 instead of 1.07291 leads to a $910,000 loss.
These examples illustrate how small errors can significantly affect financial outcomes and highlight the importance of rigorous data quality checks.
??x

---

#### Assessing Financial Data Errors Against References
Background context explaining the need for reference values when assessing data errors in finance.
:p Why is it important to assess financial data errors against appropriate reference values?
??x
Assessing financial data errors against appropriate reference values is crucial because many financial variables are subject to varying estimates, averages, or provider-specific values. For example, a EUR/USD quote can differ among Forex brokers. Therefore, errors must be evaluated in the context of these references to ensure accurate and reliable analysis.
??x

---

#### Steps for Handling Financial Data Errors
Background context explaining the process of identifying and handling financial data errors.
:p What are the steps involved in handling financial data errors?
??x
Handling financial data errors involves several steps:
1. Identification: Detecting erroneous records at both single-record or dataset levels.
2. Detection: Using rule-based approaches for simple errors and statistical/data mining techniques for complex ones.
3. Correction: Fixing identified errors to maintain data integrity.
4. Monitoring: Continuously monitoring the data quality to prevent future issues.

For example, a basic error detection approach could be:
```pseudocode
function detectErrors(dataSet):
    errors = []
    for each record in dataSet:
        if violatesRule(record):  # Define rules based on domain knowledge
            errors.append(record)
    return errors
```
??x

