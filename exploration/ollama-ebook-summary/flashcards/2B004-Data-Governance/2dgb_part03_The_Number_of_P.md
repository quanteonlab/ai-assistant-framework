# Flashcards: 2B004-Data-Governance_processed (Part 3)

**Starting Chapter:** The Number of People Working andor Viewing the Data Has Grown Exponentially

---

#### Data Governance Evolution

Background context explaining how data governance has evolved over time, including its historical scope and current importance.

Data governance has been around since there was data to govern. Traditionally, it was restricted to IT departments in regulated industries and focused on security concerns related to specific datasets like authentication credentials. Legacy data processing systems also required mechanisms for ensuring data quality and controlling access.

In the past, data governance was often seen as an IT function that was performed in silos based on data source type. For example, HR and financial data were controlled by one IT silo with strict controls, while sales data might have been managed more loosely.

However, recent regulatory changes like GDPR (General Data Protection Regulation) and CCPA (California Consumer Privacy Act), as well as the growing realization of the business value of data, are driving a shift towards holistic or centralized data governance. The vast increase in the size and variety of data collected, facilitated by technological advancements, is making traditional siloed approaches insufficient.

:p How has data governance evolved over time?
??x
Data governance has expanded from being primarily an IT function focused on security in regulated industries to becoming more integral across all departments due to the rise of global regulations like GDPR and CCPA, as well as the increased recognition of data's business value. This evolution requires a more comprehensive approach that goes beyond siloed management.
x??

---

#### Vast Growth in Data

Background context explaining the significant increase in the volume of data being collected globally.

In November 2018, International Data Corporation (IDC) predicted that the global datasphere would balloon to 175 ZB by 2025. This dramatic increase is due to the widespread adoption of technologies like big data and predictive analytics.

:p What prediction did IDC make about global data storage in 2025?
??x
IDC forecasted that the global datasphere would reach 175 Zettabytes (ZB) by 2025. This projection underscores the exponential growth in data collection and storage capabilities.
x??

---

#### Impact of Regulatory Changes

Background context on how regulations like GDPR and CCPA are affecting companies.

Regulations such as GDPR and CCPA have become significant factors influencing data governance practices across industries, not just traditionally regulated ones like healthcare and finance. These laws mandate stringent controls over personal data, requiring organizations to implement robust data management strategies.

:p How do regulatory changes like GDPR and CCPA impact data governance?
??x
Regulatory changes like GDPR and CCPA are reshaping data governance by imposing stricter requirements on how companies handle personal data. This necessitates more comprehensive and centralized approaches to managing and protecting data, affecting all industries beyond just traditionally regulated ones.
x??

---

#### Changes in Data Landscape

Background context describing the transformation of the data landscape due to technological advancements.

Technological advancements have led to a significant increase in the volume and variety of data being collected. Predictive analytics coupled with the rise of big data technologies has resulted in systems having a deeper understanding of user behavior than users themselves might realize. This shift necessitates more sophisticated data governance strategies.

:p How are technological advancements changing the data landscape?
??x
Technological advancements, particularly in areas like big data and predictive analytics, have led to an explosion in the volume and variety of data being collected. Systems now have a profound understanding of user behavior that goes beyond what users know themselves. This requires more advanced data governance strategies to ensure data quality, security, and compliance.
x??

---

#### Demand for Data Science Jobs
Background context explaining the significant increase in demand for data science jobs over time. The report by Indeed indicates a 78% jump in job demands between 2015 and 2018.

:p What was the percentage increase in data science job demand from 2015 to 2018 as reported by Indeed?
??x
The demand for data science jobs had increased by 78 percent between 2015 and 2018.
x??

---

#### Growth of Global Interaction with Data
Background context on the exponential growth in people interacting with data. IDC projects that over five billion people are now interacting with data, which is expected to increase to nearly seven and a half billion by 2025.

:p According to IDC's projection, what percentage of the world’s population will be interacting with data by 2025?
??x
IDC projects that by 2025, over six billion people (nearly 75% of the world’s population) will be interacting with data.
x??

---

#### Importance of Complex Systems for Data Management
Background context on the need for complex systems to manage data access, treatment, and usage due to the exponential growth in data users.

:p Why is there a greater need for complex systems to manage data access, treatment, and usage?
??x
The greater number of people working with and viewing data increases the risk of misuse. Thus, complex systems are needed to manage these processes effectively.
x??

---

#### Advancement in Data Collection Methods
Background context on how data collection methods have evolved from batch processing to real-time or near-real-time streaming.

:p What percentage of the global datasphere is predicted to be real-time by 2025 according to IDC?
??x
IDC predicts that nearly 30% of the global datasphere will be real-time by 2025.
x??

---

#### Example in Sports: NFL Next Gen Stats (NGS)
Background context on how advanced data collection methods have transformed sports analytics, with a specific focus on the NFL's implementation.

:p Describe what Next Gen Stats (NGS) is and how it works for the NFL.
??x
Next Gen Stats (NGS) is a league program where RFID chips are used to tag various components of football games. This allows real-time data collection on every player, including their location, speed, acceleration, and velocity during each play.

```java
// Pseudocode example
public class NGS {
    private Map<String, PlayerStats> players;
    
    public void updatePlayerPosition(Player player) {
        // Update the position of the player in real-time
        players.put(player.getName(), new PlayerStats(player.getLocation(), player.getSpeed()));
    }
}
```
x??

---

#### Why Data Governance is Becoming More Important
Background context on why data governance is becoming increasingly important due to the growing complexity and volume of data.

:p How does the advent of streaming data increase the need for complex setups and monitoring?
??x
The advent of streaming data increases the speed at which analytics can be performed, but it also introduces new risks such as infiltration. Therefore, there is a greater need for sophisticated systems to set up and monitor these processes to ensure security and compliance.
x??

---

#### Real-Time Data in Sports: Example with American Football
Background context on how sports have evolved from coarse-grained data collection to detailed real-time analytics, specifically mentioning the NFL's Next Gen Stats.

:p What specific questions did the NFL want to answer through NGS?
??x
The NFL wanted to understand what factors contribute to a successful run play, including whether it depends more on the ball carrier, their teammates (by way of blocking), or the coach’s play call. They also aimed to assess the role of the opposing defense.
x??

---

#### NFL Big Data Bowl
Background context: The NFL hosts an annual contest called the Big Data Bowl, challenging analytics experts to contribute innovative approaches to football data analysis. This event showcases advancements in data collection and usage in sports analytics, emphasizing better understanding of player skills and coaching strategies through advanced analytics.

:p What is the purpose of the NFL's Big Data Bowl?
??x
The Big Data Bowl aims to engage talented members of the analytics community by asking them to analyze trends, rethink player performance metrics, and innovate on football strategy. This contest highlights advancements in data collection methods and their application in sports analytics.
x??

---

#### Digitization of the World
Background context: The digitization of various sectors has led to an explosion in the amount of digital interactions per day for individuals using technology. By 2025, it is projected that each person will interact with data-creating technologies over 4900 times daily.

:p How many interactions are expected per person by 2025?
??x
By 2025, a person is expected to have more than 4,900 digital engagements per day. This translates to approximately one interaction every eighteen seconds.
x??

---

#### Data Governance Importance
Background context: The increasing volume and sensitivity of data collected from various sources are driving the need for robust data governance frameworks. Data-driven decision making in companies is becoming more prevalent, but it's crucial to ensure that sensitive data is handled appropriately.

:p Why is data governance important?
??x
Data governance is essential because it ensures proper handling and use of sensitive data while maintaining compliance with regulations and protecting customer privacy. It involves defining roles, responsibilities, and policies for managing data throughout its lifecycle.
x??

---

#### Use Cases for Data in Business Decisions
Background context: Companies are increasingly using data to drive better business outcomes through data-driven decision making. Examples include Amazon's targeted recommendations based on purchase history and Safari Books Online's real-time analytics.

:p What is an example of a company using data to make better business decisions?
??x
Amazon uses customer purchase history, browsing behavior, and post-purchase reviews to generate targeted product recommendations and drive future sales. This application of data results in improved personalization for customers.
x??

---

#### Real-Time Data Analytics Example: Safari Books Online
Background context: To enhance sales intelligence, Safari Books Online leveraged real-time analytics tools to process usage data from content delivery networks (CDNs) and web logs, providing near-instant insights.

:p How did Safari Books Online achieve better data-driven decisions?
??x
Safari Books Online transferred usage data from CDNs and web application logs into a cloud-native data warehouse for real-time analysis. This allowed the team to create dashboards, provide faster ad hoc queries, and deliver relevant user information in near-real time.
x??

---

#### Real-Time Data Analytics Example: California Design Den
Background context: By integrating smart analytics platforms, California Design Den improved its decision-making processes related to pricing and inventory management, leading to faster pricing decisions and better profitability.

:p How did California Design Den use data for pricing and inventory?
??x
California Design Den utilized smart analytics platforms to aggregate diverse data types (pricing and inventory) to make faster and more informed pricing decisions. This integration helped in optimizing sales and achieving higher profitability.
x??

---

