# Flashcards: 2B004-Data-Governance_processed (Part 20)

**Starting Chapter:** What Should You Monitor. Data Lineage Monitoring

---

#### Data Quality Monitoring

Background context: In Chapters 1, 2, and 5, we introduced data quality as a critical aspect of ensuring that organizations can trust their data for further calculations. High-quality data is essential to support business decisions. Data quality monitoring involves tracking attributes such as completeness, accuracy, duplication, and conformity.

:p What are the key attributes that should be tracked and measured in data quality monitoring?
??x
The key attributes include:
- Completeness: Identifying what data is missing or not usable.
- Accuracy: Ensuring correctness and consistency of data values.
- Duplication: Detecting repeated records to avoid redundancy issues.
- Conformity: Ensuring data is stored in a standard format.

These attributes map to specific business requirements set forth by the governance initiative. Monitoring helps create validation controls and trigger alerts when thresholds are exceeded, ensuring timely mitigation of critical issues.
x??

---

#### Process and Tools for Monitoring Data Quality

Background context: A comprehensive data quality monitoring system ensures that data meets predefined standards throughout its lifecycle. This involves establishing a baseline, setting up quality signals, and implementing automatic tagging.

:p What are the key elements involved in setting up a data quality monitoring process?
??x
The key elements include:
- Establishing a baseline to understand the current state of data quality.
- Comparing results over time for proactive management of ongoing improvements.
- Defining quality signal rules by the data governance committee to ensure compliance with policies and standards.
- Creating controls for validation, enabling quality monitoring and reporting.

To get started, determine baseline metrics for data quality levels and build a business case. Over time, this process helps improve the governance program.
x??

---

#### Data Lineage Monitoring

Background context: Data lineage tracks the origin, flow, transformation, and usage of data as it moves through various stages in its lifecycle. This is crucial for ensuring data integrity, usability, and security.

:p What are some key attributes to monitor during data lineage?
??x
Key attributes include:
- Data transformations: Tracking changes and hops as data moves along the lifecycle.
- Technical metadata: Understanding more about data elements via automatic tagging based on their source.
- Data quality test results: Measuring data quality at specific points in the lifecycle.
- Reference data values: Using these to understand backward and forward lineage for root cause analysis.

Monitoring lineage helps track why a dashboard has different results than expected and aids in understanding the movement of sensitive data across the organization. It also provides an audit trail for compliance and risk management purposes.
x??

---

#### Process and Tools for Monitoring Data Lineage

Background context: Monitoring lineage involves capturing data at multiple levels to understand complex transformations and dependencies. This is essential for alerting, auditing, and complying with defined rules.

:p How can monitoring lineage help in tracking the behavior of actors?
??x
Monitoring lineage helps by:
- Alerting when actor outputs are incorrect.
- Investigating inputs to determine issues.
- Augmenting or removing actors from the process flow as needed.
- Providing an audit trail for successful or attempted data breaches, helping to identify affected business areas.

Code Example:
```java
public class LineageMonitor {
    public void monitorActorOutputs(Actor actor) {
        // Logic to check if output is correct
        if (!isOutputCorrect(actor)) {
            alertUser(actor);
            updateActorBehavior(actor);
        }
    }

    private boolean isOutputCorrect(Actor actor) {
        // Check logic for correctness of outputs
        return false;
    }

    private void alertUser(Actor actor) {
        // Send alerts to relevant parties
    }

    private void updateActorBehavior(Actor actor) {
        // Update actor's behavior or remove from process flow
    }
}
```
x??

---

---
#### Compliance Monitoring Importance
Compliance monitoring is crucial because state and federal regulations, industry standards, and governance policies must be understood to ensure effective compliance. Noncompliance can lead to significant fines and damage to a company's reputation and customer trust. The Ponemon Institute reports that noncompliance costs an average of $14.82 million, including fines, forced compliance costs, loss of trust, and lost business.
:p Why is compliance monitoring essential?
??x
Compliance monitoring ensures that a company stays in line with legal requirements and standards set by state, federal, and industry bodies. It helps avoid substantial financial penalties and maintains customer trust and loyalty. Regular audits and tracking of data access are necessary to demonstrate compliance during government inspections.
```java
// Example: Checking Compliance Status
public boolean checkComplianceStatus() {
    // Logic to verify current compliance status against regulatory requirements
    return true; // Placeholder for actual logic
}
```
x??

---
#### Internal Legal Representative in Compliance Monitoring
In the absence of a dedicated legal team, an internal legal representative (attorney) is often responsible for keeping up with laws and regulations. This individual must continuously monitor changes and ensure that any necessary updates are implemented to maintain compliance.
:p Who typically handles compliance monitoring in organizations?
??x
An internal legal representative or privacy tsar, who may also be a security professional, is generally tasked with compliance monitoring. They need to stay informed about regulatory changes and implement updates as required. This ensures the company remains compliant during audits by government agencies.
```java
// Example: Updating Compliance Procedures Based on New Regulations
public void updateComplianceProcedures(String newRegulation) {
    // Logic to incorporate new regulations into existing compliance procedures
    System.out.println("Updating procedures based on " + newRegulation);
}
```
x??

---
#### Creating a Monitoring Plan for Compliance
A monitoring plan involves several steps: auditing the current governance structure, identifying risks, and prioritizing them. It also includes defining roles and responsibilities, and using automated tools to enhance compliance.
:p What are the key steps in creating a compliance monitoring plan?
??x
Key steps include:
1. Auditing the current governance structure with relevant regulations.
2. Identifying and prioritizing risks.
3. Defining roles and responsibilities.
4. Implementing an automated monitoring program that informs regulatory agencies of failed audits and mitigation strategies.

Example pseudocode for creating a monitoring plan:
```python
def createMonitoringPlan():
    auditCurrentGovernance()
    identifyAndPrioritizeRisks()
    defineRolesAndResponsibilities()
    implementAutomatedComplianceTools()
```
x??

---
#### Monitoring Program Performance

Monitoring program performance involves tracking progress against the program’s aims and objectives. Key metrics include number of lines of business, issue handling status, engagement levels, value-added interactions, and ROI from data governance investments.
:p What are some key measures for monitoring program performance?
??x
Key measures include:
1. Number of involved parties (lines of business, functional areas, project teams).
2. Status and impact of issues handled by the governance function.
3. Engagement and participation levels across the organization.
4. Value-added interactions like training and project support.
5. Business value ROI from data governance investments.

Example code to track performance:
```java
public class PerformanceTracker {
    private int linesOfBusiness;
    private List<Issue> issues;

    public void trackPerformance() {
        // Logic to track and report on various performance metrics
    }
}
```
x??

---

#### Cybersecurity Statistics and Trends
Background context: The provided text discusses significant cybersecurity statistics and trends, highlighting the growing threat of cyberattacks. According to Cybersecurity Ventures, the damage related to cybercrime is projected to hit $6 trillion annually by 2021. Additionally, consumer data privacy and protection are becoming more critical as new legislation holds businesses accountable for breaches.
:p What are some key cybersecurity statistics mentioned in the text?
??x
The text mentions that the damage from cybercrime is expected to reach $6 trillion annually by 2021 according to Cybersecurity Ventures. It also states that after the Equifax data breach, their stock price fell more than 30 percent.
x??

---

#### Importance of Security Monitoring
Background context: The text emphasizes the importance of security monitoring in preventing and mitigating cyber threats. It explains that attacks can significantly impact a business's brand and shareholder reputation beyond just financial losses. Security monitoring involves collecting and analyzing information to detect suspicious behavior or unauthorized system changes.
:p Why is security monitoring crucial for businesses?
??x
Security monitoring is crucial because it helps detect and respond to potential breaches before they cause significant damage, maintaining the organization's reputation and financial stability. For instance, even if a company has implemented robust security measures, unanticipated breaches can still occur, impacting stock prices and confidence in security protocols.
x??

---

#### Types of Security Monitoring Items
Background context: The text outlines various areas where security monitoring can be applied, such as security alerts and incidents, network events, server logs, application events, server patch compliance, endpoint events, identity access management, and data loss. Each area involves specific activities to ensure continuous security.
:p What are some examples of security monitoring items mentioned in the text?
??x
Some examples include:
- Security alerts and incidents: Data exfiltration or unusual port activity.
- Network events: Monitoring network activity, including device statuses and IP addresses.
- Server logs: Continuously monitoring server activities to detect errors and performance issues.
- Application events: Ensuring software and applications are accessible and performing smoothly.
x??

---

#### Continuous Security Monitoring
Background context: The text describes continuous security monitoring as a process that provides real-time visibility into an organization’s security posture. This approach helps in staying ahead of cyber threats, reducing the response time to attacks, and complying with industry and regulatory requirements.
:p What is continuous security monitoring?
??x
Continuous security monitoring involves regularly collecting and analyzing information from various sources to detect potential threats or unauthorized changes. It ensures that organizations can respond quickly to emerging security issues while adhering to compliance standards.

```java
public class SecurityMonitoring {
    public void monitorNetworkEvents() {
        // Code to continuously check network activity for unusual events
    }
    
    public void logServerActivities() {
        // Code to record and analyze server logs for errors and performance issues
    }
}
```
x??

---

#### Role of Network and Endpoint Security Technologies
Background context: The text explains how network security monitoring tools aggregate data from different sources, while endpoint security technologies provide visibility at the host level. Both approaches help in detecting threats earlier.
:p What are the differences between network and endpoint security technologies?
??x
Network security monitoring tools aggregate logs from various sources to detect failures and anomalies in real-time. Endpoint security technologies focus on providing detailed visibility at the host level, allowing for early detection of threats in the process flow.

```java
public class NetworkSecurityMonitoring {
    public void collectLogs() {
        // Code to collect and analyze network-related logs
    }
}

public class EndpointSecurity {
    public void monitorEvents() {
        // Code to detect and respond to endpoint events
    }
}
```
x??

---

#### Factors in Choosing a Security Monitoring Solution
Background context: The text mentions that the choice of security monitoring solution depends on factors such as business needs, team size, budget, available technologies, and desired level of sophistication. Both internal implementation and outsourcing options have their pros and cons.
:p What factors should be considered when choosing a security monitoring solution?
??x
Factors to consider include:
- Business needs: The specific security challenges the organization faces.
- Team size: Internal capabilities versus the need for external expertise.
- Budget: Cost of implementing and maintaining in-house vs. outsourcing.
- Available technologies: Existing infrastructure and tools.
- Level of sophistication: The desired complexity and depth of monitoring.

```java
public class SecurityMonitoringSelection {
    public void evaluateOptions() {
        // Code to assess internal implementation versus outsourcing based on business needs, budget, and technology stack
    }
}
```
x??
---

