# Flashcards: 2B002---Fundamentals-of-Data-Engineering_processed (Part 7)

**Starting Chapter:** Principle 3 Architect for Scalability

---

#### Enterprise Architecture and Data Architecture Overview
Background context: The provided text discusses enterprise architecture, emphasizing its role in balancing flexibility and trade-offs. It introduces data architecture as a subset of enterprise architecture, focusing on defining it based on established frameworks such as TOGAF and DAMA DMBOK. The objective is to understand how these definitions align with practical implementations.
:p What are the key elements discussed in the provided text regarding enterprise and data architecture?
??x
The key elements discussed include:
- Enterprise architecture balances flexibility and trade-offs, constantly assessing and reevaluating due to dynamic changes.
- Data architecture is a subset of enterprise architecture, inheriting its properties such as processes, strategy, change management, and evaluating trade-offs.
- Data architecture focuses on designing systems to support evolving data needs through flexible and reversible decisions.

Data architecture is defined by:
1. TOGAF: Describing the structure and interaction of an enterprise's major types and sources of data, logical and physical assets, and data management resources.
2. DAMA DMBOK: Identifying the data needs of the enterprise and designing master blueprints to meet those needs.
??x
The answer with detailed explanations:
TOGAF defines data architecture as a comprehensive description of the structure and interaction of an organization's major types and sources of data, including logical and physical assets along with data management resources. DAMA DMBOK emphasizes identifying data needs and designing master blueprints to ensure alignment with business strategy. Both definitions highlight the importance of flexibility and continuous evaluation in data architecture.

---
#### Data Architecture vs. Enterprise Architecture
Background context: The text differentiates between enterprise architecture and data architecture, noting that data architecture is a subset of enterprise architecture but has its unique focus areas.
:p How does data architecture fit into the broader scope of enterprise architecture?
??x
Data architecture fits into the broader scope of enterprise architecture by inheriting its core properties such as processes, strategy, change management, and evaluating trade-offs. However, it specializes in addressing specific aspects related to data management.

??x
The answer with detailed explanations:
Data architecture is a specialized area within enterprise architecture that focuses on the design and governance of data systems. It inherits properties like strategic planning, process improvement, and resource allocation from enterprise architecture but narrows its focus to ensure that data-related processes are efficient and aligned with business objectives.
---
#### Data Engineering Architecture
Background context: The text introduces data engineering architecture as a subset of general data architecture, emphasizing its role in supporting the data lifecycle. It highlights the importance of understanding both operational and technical aspects within this domain.
:p How does data engineering architecture fit into the broader concept of data architecture?
??x
Data engineering architecture is a specialized component of data architecture that focuses on the systems and frameworks involved in managing and processing data throughout its lifecycle. It serves as a subset of general data architecture, tailored to address specific challenges in data ingestion, storage, transformation, and serving.

??x
The answer with detailed explanations:
Data engineering architecture specializes in defining the technical and operational requirements needed for efficient data management. It includes designing systems that handle data movement, storage, and transformation at various stages of the lifecycle, ensuring seamless integration and optimal performance.

---
#### Operational vs. Technical Architecture
Background context: The text differentiates between operational and technical aspects within data engineering architecture, providing clear definitions and examples to illustrate their roles.
:p What are the differences between operational and technical architectures in data engineering?
??x
Operational architecture encompasses the functional requirements related to people, processes, and technology, focusing on what needs to happen in terms of business operations. Technical architecture outlines how data is ingested, stored, transformed, and served during the lifecycle.

??x
The answer with detailed explanations:
Operational architecture deals with the functional aspects such as business processes that data supports, data quality management, and latency requirements. For example, determining which business processes depend on data and how quickly this data must be available for querying.
Technical architecture focuses on the systems and frameworks used to handle data ingestion, storage, transformation, and serving. An example is defining how 10 TB of data should be moved from a source database to a data lake every hour.

---
#### Example Scenario
Background context: The text provides an example scenario illustrating the practical application of technical architecture in data engineering.
:p Can you provide an example of a scenario involving both operational and technical architectures in data engineering?
??x
Consider a scenario where a financial services company needs to ingest, store, and analyze large volumes of transactional data for risk assessment. The operational architecture would define processes like who is responsible for maintaining data quality and the business rules that dictate when data is considered stale. The technical architecture would specify how this data should be moved from source systems to a data warehouse, stored efficiently, transformed into required formats, and made available for analytics.

??x
The answer with detailed explanations:
In this scenario, operational architecture defines:
- Who manages data quality (e.g., data stewards)
- Business rules for updating data
- Latency requirements for data availability

Technical architecture includes:
- Designing ETL processes to move data from source systems
- Choosing a data storage solution that meets scalability and performance needs
- Implementing transformations required for analysis
- Ensuring data is made available in a format suitable for querying and reporting.

This dual approach ensures both the business operations are aligned with strategic goals, and the technical infrastructure supports efficient data management.

---
#### Concept of Good Data Architecture
Background context: The provided text emphasizes that good data architecture should serve business requirements with reusable building blocks, maintain flexibility, and make appropriate trade-offs. It also highlights the importance of agility, reversibility, and continuous evolution in response to changing business needs and technological advancements.

:p What does "good" data architecture mean according to the text?
??x
Good data architecture serves business requirements using a common set of widely reusable building blocks while maintaining flexibility and making appropriate trade-offs. It acknowledges that the world is fluid and evolves with changes within the business and new technologies, ensuring it remains maintainable and adaptable.
x??

---
#### Agility as Foundation for Good Data Architecture
Background context: The text underscores that agility is fundamental to good data architecture. It recognizes the dynamic nature of the world and the accelerating pace of change in the data space.

:p Why is agility important for good data architecture?
??x
Agility is crucial because it acknowledges the fluidity of the business environment and technological landscape. Good data architectures are flexible and easily maintainable, evolving to meet changing business needs and leveraging new technologies to unlock more value.
x??

---
#### Principles of Good Data Architecture (AWS Well-Architected Framework)
Background context: The text highlights the AWS Well-Architected Framework with six pillars: operational excellence, security, reliability, performance efficiency, cost optimization, and sustainability.

:p What are the six pillars of the AWS Well-Architected Framework?
??x
The six pillars of the AWS Well-Architected Framework are:
1. Operational Excellence
2. Security
3. Reliability
4. Performance Efficiency
5. Cost Optimization
6. Sustainability
These principles guide the design and optimization of cloud architectures.
x??

---
#### Principles of Good Data Architecture (Google Cloud's Five Principles for Cloud-Native Architecture)
Background context: The text mentions Google Cloud’s five principles for cloud-native architecture, which include designing for automation, being smart with state, favoring managed services, practicing defense in depth, and always architecting.

:p What are the five principles of Google Cloud’s architecture?
??x
Google Cloud’s Five Principles for Cloud-Native Architecture are:
1. Design for Automation
2. Be Smart with State
3. Favor Managed Services
4. Practice Defense in Depth
5. Always Architecting
These principles aim to guide cloud architects towards designing flexible, secure, and efficient systems.
x??

---
#### Choosing Common Components Wisely
Background context: The text advises choosing common components wisely as one of the principles of data engineering architecture.

:p What does the principle "choose common components wisely" imply?
??x
This principle suggests selecting widely reusable building blocks that can be applied across different parts of a system, ensuring consistency and reducing redundancy. It emphasizes making smart choices to leverage existing solutions rather than reinventing the wheel.
x??

---

#### Plan for Failure
Background context: The text emphasizes the inevitability of failures and stresses the importance of planning for them to ensure robustness. Key terms include availability, reliability, recovery time objective (RTO), and recovery point objective (RPO).

:p What is the percentage of time an IT service or component in operational state called?
??x
The term used for the percentage of time an IT service or component is in an operable state is **availability**.
x??

---

#### Architect for Scalability
Background context: The text highlights that scalability should be a key consideration in architecture. This means designing systems to handle growth and change efficiently.

:p What does it mean to architect for scalability?
??x
Architecting for scalability involves designing systems to handle growth and change efficiently, ensuring they can scale up or down as needed without significant rework.
x??

---

#### Architecture is Leadership
Background context: The text states that architecture is a form of leadership. This implies that the choices made in architectural design have a profound impact on the overall system.

:p What does the author mean when he says "Architecture is leadership"?
??x
The statement suggests that architects play a pivotal role in leading the technical direction and making critical decisions that influence the entire organization’s technological landscape.
x??

---

#### Always Be Architecting
Background context: The text emphasizes continuous attention to architecture, advocating for ongoing efforts to improve and refine systems.

:p Why should one always be architecting?
??x
One should always be architecting because continuous improvement of architectural designs ensures better preparedness for future changes and challenges, leading to more robust and adaptable systems.
x??

---

#### Build Loosely Coupled Systems
Background context: The text advises building loosely coupled systems, which means designing components that can operate independently while still working together.

:p What is a benefit of building loosely coupled systems?
??x
A key benefit of building loosely coupled systems is improved resilience and flexibility. Components can be updated or scaled independently without affecting the entire system.
x??

---

#### Make Reversible Decisions
Background context: The text recommends making reversible decisions to allow for easy changes in the future, which helps maintain agility.

:p What does it mean to make reversible decisions?
??x
Making reversible decisions means designing systems and solutions that can be easily altered or undone if needed. This approach allows for flexibility and adaptability without locking teams into permanent choices.
x??

---

#### Prioritize Security
Background context: The text stresses the importance of prioritizing security in system design to protect against potential threats.

:p Why is security a critical factor in data architecture?
??x
Security is crucial because it protects sensitive information from unauthorized access, breaches, and other cyber threats. It ensures compliance with regulations and maintains trust with users.
x??

---

#### Embrace FinOps
Background context: The text advocates for integrating financial management practices into the development of data architectures to optimize costs and resource allocation.

:p What does embracing FinOps mean in the context of data architecture?
??x
Embracing FinOps means integrating financial management practices directly into the design and operation of data systems. This approach helps in optimizing costs, budgeting effectively, and ensuring efficient resource utilization.
x??

---

#### Choose Common Components Wisely
Background context: The text discusses the importance of selecting common components that can be widely used across an organization to facilitate collaboration and increase agility.

:p What is a primary job of a data engineer according to the text?
??x
A primary job of a data engineer, as stated in the text, is to choose common components and practices that can be used widely across an organization.
x??

---

#### Evaluating Failure Scenarios
Background context: The text introduces key terms for evaluating failure scenarios such as availability, reliability, RTO, and RPO.

:p What does RTO stand for and what does it measure?
??x
RTO stands for Recovery Time Objective. It measures the maximum acceptable time for a service or system outage.
x??

---

#### Scalability: Scaling Up and Down
Background context explaining the concept. In data systems, scalability encompasses two main capabilities—scaling up to handle significant quantities of data and scaling down when load decreases or ceases. These abilities are crucial for managing extreme loads temporarily and cutting costs during periods of low activity.
:p What does scalability in data systems include?
??x
Scalability includes the ability to scale up (to handle significant data volumes) and scale down (automatically reduce capacity after a transient load spike). This ensures efficient use of resources by adapting to varying loads dynamically.
x??

---

#### Elasticity and Zero Scaling
Background context explaining the concept. Elastic systems can scale dynamically in response to load, ideally in an automated fashion. Some scalable systems can also scale to zero: they shut down completely when not in use. Many serverless systems (e.g., serverless functions and OLAP databases) automatically scale to zero.
:p What is meant by elastic scaling?
??x
Elastic scaling refers to the ability of a system to dynamically adjust its resources based on the current load, either increasing or decreasing them as needed. This ensures that resources are efficiently utilized without over-provisioning.
x??

---

#### Serverless Systems and Cost Efficiency
Background context explaining the concept. Some systems can scale to zero by shutting down when not in use, making serverless architectures an efficient choice for cost management during periods of low activity.
:p Can you explain how serverless systems work?
??x
Serverless systems automatically manage scaling based on demand. When there is no load, they scale down or shut down entirely, leading to minimal resource utilization and cost efficiency. For example:
```java
public class ServerlessFunction {
    @FunctionName("example")
    public void run(@TimerTrigger(name = "myTimer", expression = "0 * * * * ?") Timer timer) throws IOException {
        // Function logic here
    }
}
```
This code demonstrates a serverless function that runs based on a trigger, scaling down when not needed.
x??

---

#### Relational Databases vs. Complex Clusters
Background context explaining the concept. Deploying inappropriate scaling strategies can result in overcomplicated systems and high costs. A straightforward relational database with one failover node may be appropriate for an application instead of a complex cluster arrangement.
:p When might a simple relational database be more suitable than a complex cluster?
??x
A simple relational database with a failover node is often more suitable when the application requires robust data management but does not need the additional complexity and cost of a large-scale distributed system. This setup ensures reliable data storage while keeping costs low.
x??

---

#### Data Architecture Leadership
Background context explaining the concept. Data architects are responsible for technology decisions, architecture descriptions, and disseminating these choices through effective leadership and training. They should have high technical competence but delegate most individual contributor work to others.
:p What roles do data architects play in an organization?
??x
Data architects lead technology decisions and architecture descriptions, ensuring that the chosen technologies and architectures are well-documented and communicated across teams. They also train others on best practices and ensure consistency in architectural design.
x??

---

#### Balancing Common Components with Flexibility
Background context explaining the concept. Cloud environments allow data architects to balance common component choices with flexibility that enables innovation within projects, avoiding the traditional command-and-control approach of forcing all teams to use one proprietary database technology.
:p How do cloud environments support data architecture decisions?
??x
Cloud environments provide a flexible platform where data architects can choose common components and services while ensuring that each project has the freedom to innovate using different technologies as needed. This balance helps avoid the rigidity of forced uniformity, allowing teams to adopt the best tools for their specific needs.
x??

---

#### Technical Leadership and Data Architecture
Background context: Martin Fowler describes an ideal software architect who focuses on mentoring the development team to improve their skills. This approach provides greater leverage than being a sole decision-maker, avoiding becoming an architectural bottleneck.

:p Who is responsible for improving the development team’s ability in this scenario?
??x
An ideal data architect is responsible for improving the development team's ability. They mentor current data engineers, make careful technology choices in consultation with their organization, and disseminate expertise through training and leadership.
x??

---

#### Principle 5: Always Be Architecting
Background context: This principle encourages data architects to constantly design new things based on business and technological changes rather than just maintaining the existing state. Architects need to develop a baseline architecture, target architecture, and sequencing plan.

:p What does an architect’s job entail according to this principle?
??x
An architect’s job involves developing deep knowledge of the baseline architecture (current state), developing a target architecture, and mapping out a sequencing plan to determine priorities and the order of architecture changes. Modern architecture should not be command-and-control or waterfall but collaborative and agile.
x??

---

#### Principle 6: Build Loosely Coupled Systems
Background context: This principle promotes creating systems where components can operate independently with minimal dependencies on each other, enabling teams to test, deploy, and change systems without communication bottlenecks.

:p What are the key properties of a loosely coupled system?
??x
The key properties of a loosely coupled system include:
1. Systems broken into many small components.
2. Interfaces with other services through abstraction layers such as a messaging bus or an API.
3. Internal changes to a system component do not require changes in other parts because details are hidden behind stable APIs.
4. Each component can evolve and improve separately, leading to no global release cycle but separate updates for each component.
x??

---

#### Bezos API Mandate
Background context: The Bezos API mandate is a set of guidelines issued by Amazon's CEO in 2002 that required all teams to expose data and functionality through service interfaces, enabling loose coupling and eventually leading to the development of AWS.

:p What are the key requirements of the Bezos API mandate?
??x
The key requirements of the Bezos API mandate include:
1. All teams must expose their data and functionality through service interfaces.
2. Teams must communicate with each other exclusively through these interfaces (no direct linking, no shared memory model).
3. Service interfaces must be designed from the ground up to be externalizable, meaning they should plan for potential exposure to developers outside the company.
x??

---

#### Loosely Coupled Systems in Organizations
Background context: The principles of building loosely coupled systems can be translated into organizational characteristics, promoting independence and agility among teams.

:p How does a loosely coupled system benefit organizations?
??x
A loosely coupled system benefits organizations by breaking down components into small parts that interface with each other through abstraction layers such as messaging buses or APIs. This design hides internal details, reducing the need for changes in other parts due to internal updates. Consequently, there is no global release cycle; instead, components can be updated independently.
x??

---

