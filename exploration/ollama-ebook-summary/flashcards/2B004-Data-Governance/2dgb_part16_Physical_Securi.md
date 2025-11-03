# Flashcards: 2B004-Data-Governance_processed (Part 16)

**Starting Chapter:** Physical Security

---

#### Multi-Tenancy Considerations for Enterprises Moving to Cloud

Background context: When large enterprises migrate their systems from on-premises to the cloud, they face significant changes in how data security and access control are managed. On-premises environments typically have physical and network perimeter controls that can be lost when using multi-tenant cloud architectures.

:p What are some key differences between on-premises and cloud-based security models?
??x
In on-premises setups, organizations manage their own infrastructure with full control over the physical data center and network. They use Kerberos-based or directory-based authentication methods to secure access. Moving to a public cloud involves sharing resources with other tenants, which can introduce new risks related to unauthorized access and data leakage.

Cloud providers offer various security measures but require a shift in how security policies are implemented. Enterprises must adapt their existing security practices to fit the cloud environment while leveraging the benefits of cloud-native services.

??x
The answer focuses on highlighting the differences between physical control, authentication methods, and the need for adapting security strategies when moving from on-premises to the cloud.
```java
public class CloudSecurityAdaptation {
    public void adaptOnPremiseSecurity(String currentMethod) {
        if (currentMethod.equals("Kerberos")) {
            System.out.println("Migrate to cloud identity and access management (IAM).");
        } else if (currentMethod.equals("Directory-based")) {
            System.out.println("Similarly, switch to IAM for authentication.");
        }
    }
}
```
x??

---

#### Bare Metal Infrastructure in Cloud

Background context: Some cloud providers offer "bare metal" infrastructure or "government clouds," which provide data center management tailored to specific security requirements. However, these solutions can increase costs and introduce additional technical debt.

:p What is the main advantage of using bare metal infrastructure in the cloud?
??x
The primary advantage of using bare metal infrastructure in the cloud is that it provides a dedicated physical server for exclusive use by the customer, offering enhanced control over the hardware and potentially better security. This setup mimics on-premises data center management but within a cloud environment.

However, this approach can lead to increased costs compared to standard shared virtual machines and may create more silos, complicating resource sharing and management across different services or applications.

??x
The response emphasizes the enhanced control over hardware provided by bare metal infrastructure while also mentioning its potential drawbacks in terms of cost and complexity.
```java
public class BareMetalInfrastructure {
    public boolean isBareMetalCostEffective(int cost, int savings) {
        return (cost - savings) < 0;
    }
}
```
x??

---

#### Cloud-Native Security Policies

Background context: Moving to the cloud involves adopting a different approach to security. While many on-premises security practices can be adapted, it is often more effective to implement cloud-native security policies that leverage the unique features of the cloud environment.

:p How does applying cloud-native security policies differ from a lift-and-shift approach?
??x
Applying cloud-native security policies involves integrating modern and best-practice security solutions designed for cloud environments. This approach allows organizations to take full advantage of the elasticity, democratization, and lower operating costs offered by public clouds. In contrast, a lift-and-shift approach involves simply moving existing on-premises security measures to the cloud without making any changes or optimizations.

The benefits of using cloud-native security policies include better data protection through enhanced monitoring, advanced threat detection mechanisms, and leveraging the expertise of dedicated security teams within cloud providers.

??x
The response contrasts the advantages of integrating cloud-native security with the limitations of a lift-and-shift approach, emphasizing the value in adapting to the cloud environment.
```java
public class CloudNativeSecurity {
    public void applyCloudNativePolicies() {
        System.out.println("Implement IAM systems and leverage cloud-native security services.");
    }
}
```
x??

---

#### Security Teams and Threat Detection

Background context: Public clouds benefit from dedicated, world-class security teams that actively monitor for threats. These teams undergo specialized training and use a variety of tools to identify vulnerabilities and prevent attacks.

:p What is an example of how public cloud providers maintain high levels of security?
??x
Public cloud providers such as Google have dedicated security teams like Project Zero. This team focuses on preventing targeted attacks by reporting bugs to software vendors, filing them in external databases, and using commercial and custom tools for security scanning, penetration testing, quality assurance (QA), and software security reviews.

This proactive approach helps ensure that the public cloud environment remains secure against a wide range of threats.

??x
The response highlights specific examples like Google's Project Zero and how they contribute to maintaining high levels of security within cloud environments.
```java
public class SecurityTeam {
    public void reportBug(String bugDescription) {
        System.out.println("Reporting " + bugDescription + " to vendors.");
    }
}
```
x??

---

#### Security by Obscurity
Background context: The passage argues against relying solely on security by obscurity, explaining that the scale and sophistication of tools used in cloud environments change the security landscape. It highlights how advanced AI tools can quickly scan for sensitive data or unsafe content, and how processing large volumes of data in real time is made possible.
:p What are some drawbacks of relying on security by obscurity?
??x
Relying on security by obscurity means that you believe your system's security depends solely on its obscurity—hiding vulnerabilities rather than fixing them. This approach has several significant limitations:
1. **Lack of Defense Mechanisms**: If the security mechanism is discovered, the attacker can exploit it.
2. **Vulnerabilities Are Ultimately Exposed**: Security through obscurity does not address underlying weaknesses in your system.
3. **Increased Attack Surface**: As technology evolves, what was previously hidden might eventually be uncovered.

??x
The answer with detailed explanations:
Relying on security by obscurity means that you believe your system's security depends solely on its obscurity—hiding vulnerabilities rather than fixing them. This approach has several significant limitations:
1. **Lack of Defense Mechanisms**: If the security mechanism is discovered, the attacker can exploit it.
2. **Vulnerabilities Are Ultimately Exposed**: Security through obscurity does not address underlying weaknesses in your system.
3. **Increased Attack Surface**: As technology evolves, what was previously hidden might eventually be uncovered.

---


#### Cloud Tooling and Governance
Background context: The text discusses how cloud tools can enhance governance practices by enabling advanced security measures like AI-driven data scanning and real-time processing capabilities. These tools help in applying governance practices that may not be feasible on-premises.
:p How do cloud tools aid in governance?
??x
Cloud tools aid in governance by providing advanced functionalities such as:
1. **AI-Enabled Data Scanning**: Tools like AI can quickly scan datasets for sensitive information or unsafe content, which helps in maintaining data integrity and compliance.
2. **Real-Time Processing Capabilities**: This allows organizations to process large volumes of data in real time, enabling timely responses to data breaches or security incidents.

??x
The answer with detailed explanations:
Cloud tools aid in governance by providing advanced functionalities such as:
1. **AI-Enabled Data Scanning**: Tools like AI can quickly scan datasets for sensitive information or unsafe content, which helps in maintaining data integrity and compliance.
2. **Real-Time Processing Capabilities**: This allows organizations to process large volumes of data in real time, enabling timely responses to data breaches or security incidents.

---


#### Virtual Machine Security
Background context: The passage highlights the importance of virtual machine (VM) security in cloud environments. It mentions specific tools like Google Cloud's Shielded VM and Microsoft Azure's Confidential Compute that offer enhanced security features for VMs.
:p What is an example of a virtual machine security feature mentioned?
??x
An example of a virtual machine security feature mentioned is **Google Cloud’s Shielded VM**. This tool provides verifiable integrity to Compute Engine virtual machine (VM) instances by offering:
- Cryptographically protected baseline measurements of the VM's image.
- Tamperproof virtual machines with alerts on runtime state changes.

??x
The answer with detailed explanations:
An example of a virtual machine security feature mentioned is **Google Cloud’s Shielded VM**. This tool provides verifiable integrity to Compute Engine virtual machine (VM) instances by offering:
- Cryptographically protected baseline measurements of the VM's image.
- Tamperproof virtual machines with alerts on runtime state changes.

```java
// Example Pseudocode for Shielded VM Configuration
public class ShieldedVMConfig {
    public void configureShieldedVM(String vmImageId, String securityBaseline) {
        // Set up cryptographic protection for the VM
        setCryptographicProtection(vmImageId);
        
        // Monitor runtime state changes and alert on tampering attempts
        enableRuntimeStateMonitoring(securityBaseline);
    }
    
    private void setCryptographicProtection(String imageId) {
        // Logic to protect the VM with cryptographic methods
    }
    
    private void enableRuntimeStateMonitoring(String baseline) {
        // Logic to monitor and alert on runtime state changes
    }
}
```

---


#### Data Governance and Detection Infrastructure
Background context: The text explains how data governance includes establishing a strong detection and response infrastructure for data exfiltration events. This helps in rapidly detecting risky or improper activity, limiting the "blast radius" of such activities, and minimizing the window of opportunity for malicious actors.
:p How does a strong detection and response infrastructure help in data security?
??x
A strong detection and response infrastructure helps in data security by:
1. **Rapid Detection**: Quickly identifying suspicious or risky activities.
2. **Limiting Blast Radius**: Reducing the extent of potential damage caused by an incident.
3. **Minimizing Window of Opportunity**: Decreasing the time available for a malicious actor to exploit vulnerabilities.

??x
The answer with detailed explanations:
A strong detection and response infrastructure helps in data security by:
1. **Rapid Detection**: Quickly identifying suspicious or risky activities.
2. **Limiting Blast Radius**: Reducing the extent of potential damage caused by an incident.
3. **Minimizing Window of Opportunity**: Decreasing the time available for a malicious actor to exploit vulnerabilities.

---


#### Cloud Traffic and Security Surface
Background context: The text discusses how different cloud providers have varying security surfaces based on their traffic patterns and encryption methods. For example, public clouds with private fiber connections and default encryption have different security considerations compared to those with public internet traffic.
:p How does the type of cloud traffic affect its security surface?
??x
The type of cloud traffic affects its security surface in several ways:
1. **Private Fiber vs Public Internet**: Public clouds with private fiber connections are less susceptible to external threats but may face other internal risks.
2. **Default Encryption**: Clouds that default to encryption reduce the risk of data exposure, even if a breach occurs.

??x
The answer with detailed explanations:
The type of cloud traffic affects its security surface in several ways:
1. **Private Fiber vs Public Internet**: Public clouds with private fiber connections are less susceptible to external threats but may face other internal risks.
2. **Default Encryption**: Clouds that default to encryption reduce the risk of data exposure, even if a breach occurs.

---

#### Physical Security Model
Background context: The physical security of data centers is crucial to protect sensitive information from unauthorized access, theft, and damage. A layered security model involves multiple safeguards like electronic access cards, alarms, and biometrics.

:p What are some examples of physical security measures in a data center?
??x
Examples include electronic access cards, alarms, vehicle access barriers, perimeter fencing, metal detectors, biometric systems, and laser-beam intrusion detection. The data center should also be monitored 24/7 by high-resolution cameras.
x??

---
#### Human Security Measures
Background context: Alongside automated security measures like surveillance cameras, human security guards are essential to ensure physical safety. Guards need rigorous background checks and training.

:p How can companies ensure that the data center access is limited?
??x
The data center should be accessed only by a smaller subset of approved employees with specific roles. Human security includes routine patrols by experienced security guards who have undergone rigorous background checks and training.
x??

---
#### Layered Security in Data Centers
Background context: A layered approach to physical security involves multiple safeguards, both automated (like cameras) and manual (like human guards), to ensure comprehensive protection.

:p What does a 24/7 monitoring system include?
??x
A 24/7 monitoring system includes high-resolution interior and exterior cameras that can detect and track intruders. Additionally, human security is provided through routine patrols by experienced guards.
x??

---
#### Access Control by Sections
Background context: In cases where regulatory compliance requires specific access controls, data center floors are divided into sections to limit who can access which areas.

:p Why might you need to control access by sections?
??x
You might need to control access by sections when regulatory compliance requires that maintenance be performed by citizens of certain countries or personnel with security clearances.
x??

---
#### Uninterrupted Power Supply and Environmental Controls
Background context: Ensuring an uninterrupted power supply and environmental controls are crucial for maintaining the data center's operations. This includes having redundant power systems, cooling systems, and fire suppression equipment.

:p What is necessary to ensure smooth running of servers?
??x
Redundant power systems with primary and alternate sources are necessary. Cooling systems should maintain a constant operating temperature, reducing the risk of service outages.
x??

---
#### Fire Detection and Suppression Systems
Background context: Proper fire detection and suppression equipment help prevent hardware damage. These systems need to be integrated with security operations consoles for effective response.

:p How can you integrate fire detection systems into the overall security system?
??x
Fire detection and suppression equipment should be tied in with the security operations console so that heat, fire, and smoke detectors trigger audible and visible alarms. This integration ensures a faster response.
x??

---
#### Incident Management Process
Background context: A rigorous incident management process is required to inform customers about potential data breaches affecting system or data confidentiality, integrity, and availability.

:p What does NIST provide for devising a security incident management program?
??x
The US National Institute of Standards and Technology (NIST) provides guidance on devising a security incident management program through NIST SP 800-61.
x??

---
#### Equipment Tracking and Retirement Procedures
Background context: Proper tracking and retirement procedures are necessary to ensure data security throughout the lifecycle of equipment. This includes rendering hard drives useless by encryption and physically destroying malfunctioning ones.

:p How should a company handle retired hard drives?
??x
When retiring hard drives, all data must be encrypted when stored. The disk should be verifiably erased by writing zeros to the drive. Malfunctioning drives that cannot be erased should be physically destroyed using methods like crushing, deformation, shredding, breakage, and recycling.
x??

---
#### Security System Exercises
Background context: Regular exercises of all aspects of the data center security system help identify vulnerabilities and ensure preparedness for various scenarios.

:p What should regular tests include?
??x
Regular tests should consider a variety of scenarios, including insider threats and software vulnerabilities. These tests should encompass disaster recovery and incident management processes.
x??

---

#### Perimeter Network Security Model
Background context: The simplest form of network security is a perimeter model, where applications and personnel within the network are trusted, while all others outside the network are not. However, this approach has significant limitations.

:p What are the main issues with using a simple perimeter network security model?
??x
The main issues include:
1. No perimeter can be 100% safe; eventually, a breach will occur.
2. Not all applications within the trusted network may remain secure; a security breach or insider threat could expose sensitive data.

Example of logic in code (not directly related to this concept but for illustration):
```java
public class SecurityBreaches {
    public void checkPerimeterSecurity() {
        boolean perimeterSafe = false;
        // Simulate checking system for vulnerabilities and breaches
        if (!perimeterSafe) {
            System.out.println("The network has been breached.");
        }
    }
}
```
x??

---

#### Data Protection in Cloud Networks
Background context: Network security faces challenges when data needs to travel across devices, known as "hops," which can introduce vulnerabilities. A cloud provider's ability to limit public internet hops and use private fiber can enhance security.

:p Why is securing data in transit important?
??x
Securing data in transit is crucial because data is vulnerable to unauthorized access as it travels between devices. This vulnerability necessitates using strong encryption and protecting endpoints from illegal request structures.

Example of gRPC usage:
```java
public class GRPCService {
    public void secureDataTransmission() {
        // Define a service with methods for remote calls
        ServiceDefinition service = new ServiceDefinition();
        
        // Use Protocol Buffers for IDL and message format
        MessageFormat message = new MessageFormat();
        System.out.println("Transmitting data securely using gRPC.");
    }
}
```
x??

---

#### Physical Security Measures in Cloud Networks
Background context: To ensure applications serve only secure traffic, firewalls, access control lists, and traffic analysis are used. It is ideal for servers to communicate with a controlled list of servers rather than allowing broad access.

:p What measures can be taken to enforce network security in cloud environments?
??x
Measures include using firewalls, access control lists (ACLs), and traffic analysis to enforce network segregation and prevent DDoS attacks. Additionally, default deny policies for server-to-server communication are recommended to restrict unauthorized access.

Example of firewall configuration:
```java
public class FirewallConfiguration {
    public void configureFirewall() {
        // Define rules allowing only necessary protocols and services
        boolean allowTraffic = true;
        if (!allowTraffic) {
            System.out.println("Blocking unauthorized traffic.");
        }
    }
}
```
x??

---

#### Zoom Bombing Incidents
Background context: The rise in video conferencing usage during the pandemic has led to security issues, including "Zoom bombing," where malicious actors infiltrate meetings. This highlights the need for robust privacy and data security measures.

:p What are some of the challenges with video conferencing platforms like Zoom?
??x
Challenges include potential privacy breaches due to poorly secured meetings and concerns over how customer data is collected and used. These issues can lead to legal scrutiny and public criticism, emphasizing the importance of robust security practices.

Example of securing a Zoom meeting:
```java
public class SecureZoomMeeting {
    public void secureMeeting() {
        // Set up secure meeting options
        boolean secure = true;
        
        if (!secure) {
            System.out.println("Meeting not properly secured.");
        }
    }
}
```
x??

---

#### Security Improvements by Zoom
Background context: In response to security concerns, Zoom implemented several improvements. One notable change was the alerting feature for organizers when meeting links are posted online.

:p How did Zoom address security issues during video conferencing?
??x
Zoom improved security through various measures, including alerting organizers if their meeting link is posted publicly. This helps prevent unauthorized access and misuse of meetings.

Example of implementing secure meeting alerts:
```java
public class SecureMeetingAlerts {
    public void setupAlerts() {
        boolean sendAlert = true;
        
        if (sendAlert) {
            System.out.println("Sending alert to organizer that the meeting link is online.");
        }
    }
}
```
x??

---

---
#### Data Exfiltration Overview
Data exfiltration is a scenario where data is extracted by an authorized person or application and shared with unauthorized third parties or moved to insecure systems. This can happen either maliciously or accidentally, often due to compromised accounts.

:p What does data exfiltration refer to?
??x
Data exfiltration refers to the unauthorized transfer of sensitive information from a secure system to an untrusted third party or an insecure environment. This can occur through various vectors and may be intentional or accidental.
x??

---
#### Mitigation Strategies for Data Exfiltration
To address data exfiltration risks in cloud environments, new security approaches are required due to the shared nature of public clouds.

:p What are some common methods used to mitigate data exfiltration risks in cloud environments?
??x
Common mitigation strategies include:
- Deploying specialized agents that monitor user and host activity.
- Using explicit chokepoints such as network proxy servers, network egress servers, and cross-project networks.
- Implementing policies that limit the volume and frequency of data transmission.
- Auditing email metadata and scanning content for threats.
- Scanning outbound traffic to identify sensitive information.

For example, implementing a policy that limits downloads and keeping logs of accessed data can help mitigate risks:
```java
public class DataExfiltrationPolicy {
    public void restrictDownloads() {
        // Logic to limit file downloads and keep access logs
    }
}
```
x??

---
#### Email as an Exfiltration Vector
Email is a common vector for data exfiltration, especially when sensitive information is shared via business email or mobile devices.

:p How can emails be used as vectors for data exfiltration?
??x
Emails can be used to exfiltrate data by transmitting sensitive information from secure systems to untrusted third parties. This can include contents of organization emails, calendars, databases, images, planning documents, business forecasts, and source code.

Mitigation strategies include:
- Limiting the volume and frequency of data transmission.
- Auditing email metadata such as sender and recipient addresses.
- Scanning email content using automated tools for common threats.
- Alerting on insecure channels and attempts to transmit sensitive information.

Example pseudocode:
```java
public class EmailSecurityPolicy {
    public void auditEmailMetadata(String from, String to) {
        // Logic to monitor and log email metadata
    }
}
```
x??

---
#### Insecure Device Downloads
Sensitive data can be exfiltrated when it is downloaded onto unmonitored or insecure devices.

:p How does downloading sensitive data on unmonitored devices contribute to data exfiltration?
??x
Downloading sensitive data onto unmonitored or insecure devices poses a risk of data exfiltration. Sensitive information stored in files ("data lake") can be accessed and potentially leaked if not properly managed.

Mitigation strategies include:
- Avoiding the storage of sensitive data in files.
- Keeping such data in managed storage like an enterprise data warehouse.
- Establishing policies that prohibit downloads.
- Logging access to requested and served data.
```java
public class DataDownloadPolicy {
    public void restrictDownloads() {
        // Logic to enforce download restrictions and keep logs
    }
}
```
x??

---
#### Permission Management
Permission management is crucial in preventing unauthorized data exfiltration, especially when employees have sufficient permissions to transmit sensitive data.

:p How does permission management impact the risk of data exfiltration?
??x
Permission management plays a critical role in reducing the risk of data exfiltration. Maintaining precise and narrowly scoped permissions helps limit access to sensitive information. Comprehensive and immutable audit logs can provide visibility into who is accessing what data, enabling better detection and response to potential breaches.

Mitigation strategies include:
- Implementing strict permission controls.
- Logging all outbound connections and monitoring for anomalies.
- Avoiding public IP addresses for virtual machines.
- Disabling remote management software like RDP.
```java
public class PermissionManagement {
    public void managePermissions() {
        // Logic to enforce fine-grained permissions and logging
    }
}
```
x??

---
#### Insider Threats Near Termination
Employee termination poses a risk of data exfiltration due to disgruntled or terminated employees attempting to steal sensitive information.

:p How can organizations mitigate the risk of insider threats near employee terminations?
??x
Mitigating risks during employee terminations involves:
- Connecting logging and monitoring systems to HR software.
- Setting more conservative thresholds for alerting security teams when abnormal behavior is detected by these users.

For example, implementing a policy that monitors logins and activities close to termination dates:
```java
public class InsiderThreatControl {
    public void monitorTerminationPeriod() {
        // Logic to monitor user activity near the termination date
    }
}
```
x??

---

