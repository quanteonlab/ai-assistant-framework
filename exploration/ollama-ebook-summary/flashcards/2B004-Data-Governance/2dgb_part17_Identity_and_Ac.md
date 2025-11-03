# Flashcards: 2B004-Data-Governance_processed (Part 17)

**Starting Chapter:** Identity and Access Management

---

---
#### Compartmentalizing Data and Permissions
Compartmentalizing data involves organizing it into logical units that are easier to manage, secure, and control. This is often done by dividing data based on line of business or common workloads accessing that data. Fine-grained access controls are set up for each compartment, limiting who can access sensitive information.
:p What is the primary benefit of compartmentalizing data?
??x
The primary benefit of compartmentalizing data is to minimize the blast radius of data exfiltration events by limiting access and visibility to sensitive data. This approach ensures that only authorized users with specific permissions have access to certain types of data, reducing the risk of unauthorized data exposure.
x??

---
#### Fine-Grained Access Control Lists
Access control lists (ACLs) are used to manage who can access resources within an organization. By granting access in a time-bound manner and sparingly, organizations can enhance security by ensuring that sensitive data is not accessible to all users at all times.
:p How does using fine-grained ACLs help in managing access to sensitive data?
??x
Using fine-grained ACLs helps manage access to sensitive data more effectively by allowing specific permissions for different users or groups. This approach ensures that only necessary personnel have access, and even then, only for the duration required. For example, a developer might be granted temporary read-only access during development but no further.
```java
// Pseudocode for applying fine-grained ACLs
public class DataAccess {
    private Map<String, Boolean> userPermissions = new HashMap<>();

    public void grantAccess(String user, String data, boolean readOnly) {
        // Granting read or write permissions based on the role
        if (user.equals("admin")) {
            userPermissions.put(data, true); // Full access granted
        } else {
            userPermissions.put(data, readOnly); // Read-only access granted
        }
    }

    public void revokeAccess(String user, String data) {
        // Revoking permissions for a specific user on certain data
        userPermissions.remove(data);
    }
}
```
x??

---
#### Providing Simulated or Tokenized Data to Development Teams
Development teams often require sensitive data for testing and development. However, this can introduce risks if the environment where such data is stored is not secure. Therefore, it's recommended to provide simulated or tokenized versions of real data to minimize these risks.
:p Why should development teams be provided with simulated or tokenized data?
??x
Development teams should be provided with simulated or tokenized data because doing so minimizes the risk associated with creating cloud infrastructure and handling sensitive information. Simulated data can mimic the structure and behavior of real data without exposing actual sensitive information, reducing potential security risks.
```java
// Pseudocode for generating tokenized data
public class DataSimulator {
    private String[] realData = {"confidential1", "secret2", "topsecret3"};

    public String generateTokenizedData(String original) {
        // Tokenizing the input data to replace sensitive information
        return "token-" + original.hashCode();
    }

    public void simulateAccess(String original, String user) {
        System.out.println(user + " accessed: " + generateTokenizedData(original));
    }
}
```
x??

---
#### Immutable Logging Trails
Logging trails are crucial for maintaining transparency and traceability of data access and movement within an organization. By implementing immutable logging, organizations can ensure that records of all activities related to data cannot be altered or deleted.
:p What is the purpose of using immutable logging?
??x
The purpose of using immutable logging is to increase transparency into the access and movement of data in an organization. Immutable logs ensure that any changes or deletions are detected, providing a complete audit trail for compliance and security purposes.
```java
// Pseudocode for implementing immutable logging
public class Logger {
    private List<String> logs = new ArrayList<>();

    public void logAction(String action) {
        // Adding the action to the immutable list of logs
        synchronized (logs) {  // Ensuring thread safety
            logs.add(action);
        }
    }

    public List<String> getLogs() {
        return Collections.unmodifiableList(logs); // Returning an unmodifiable view
    }
}
```
x??

---
#### VPC-SC for Data Protection
VPC Service Controls (VPC-SC) create perimeters around a set of services and data, ensuring that the data is inaccessible from outside these perimeters. This approach enhances security by restricting access to authorized VPC networks.
:p How does VPC-SC contribute to mitigating data exfiltration risks?
??x
VPC-SC contributes to mitigating data exfiltration risks by creating perimeters around specific services and their associated data. Only clients within authorized VPC networks can access these resources, while unauthorized attempts are blocked, even if the credentials used are valid.
```java
// Pseudocode for implementing VPC-SC
public class VPCServiceControls {
    private Map<String, List<String>> servicePerimeters = new HashMap<>();

    public void createPerimeter(String serviceName) {
        // Creating a new perimeter for a specific service
        servicePerimeters.put(serviceName, Collections.emptyList());
    }

    public boolean isAuthorizedAccess(String serviceName, String clientIP) {
        // Checking if the client IP is authorized to access the service
        return servicePerimeters.containsKey(serviceName);
    }
}
```
x??

---
#### Secure Code and Data Lineage
Secure code development ensures that the application code used for producing or transforming data is trusted. Binary authorization mechanisms like Kritis can be used to enforce security policies during the deployment of container-based applications.
:p How does binary authorization contribute to secure code?
??x
Binary authorization contributes to secure code by enforcing security policies at deploy time, such as requiring verified digital signatures before deploying a container image. This ensures that only trusted images are deployed and can access sensitive data.
```java
// Pseudocode for implementing binary authorization
public class BinaryAuthorization {
    private Map<String, Boolean> signedImages = new HashMap<>();

    public void verifyImage(String imageName) {
        // Verifying the digital signature of a container image
        if (signedImages.containsKey(imageName)) {
            System.out.println("Image " + imageName + " is verified.");
        } else {
            System.out.println("Image " + imageName + " has not been verified.");
        }
    }

    public void addSignedImage(String imageName) {
        // Adding a verified image to the list
        signedImages.put(imageName, true);
    }
}
```
x??

---
#### Zero-Trust Model
The zero-trust model shifts access controls from network perimeters to individual users and devices. This approach assumes that all network traffic is untrusted and requires authentication, authorization, and encryption for accessing enterprise resources.
:p What distinguishes the zero-trust model from traditional security models?
??x
The zero-trust model differs from traditional perimeter-based security by assuming that an internal network is untrustworthy. It enforces strict access controls at every layer of the network, ensuring that users must be authenticated and authorized before accessing any resource, regardless of whether they are inside or outside the corporate network.
```java
// Pseudocode for implementing zero-trust access control
public class ZeroTrustAccessControl {
    private Map<String, Boolean> userCredentials = new HashMap<>();

    public void authenticateUser(String username, String password) {
        // Authenticating a user based on credentials
        if (userCredentials.containsKey(username) && userCredentials.get(username).equals(password)) {
            System.out.println("User " + username + " authenticated.");
        } else {
            System.out.println("Authentication failed for user " + username);
        }
    }

    public void authorizeAccess(String resource, String user) {
        // Authorizing access to a specific resource
        if (isResourceAuthorized(resource)) {
            System.out.println("Access granted to " + resource + " for user " + user);
        } else {
            System.out.println("Access denied for user " + user + " to " + resource);
        }
    }

    private boolean isResourceAuthorized(String resource) {
        // Check if the resource has been authorized
        return true; // Simplified logic, in practice, this would involve complex checks
    }
}
```
x??

---

#### User Tracking and Management
All users are tracked and managed in a user database and a group database that tightly integrate with HR processes. This integration manages job categorization, usernames, and group memberships for all users.

:p What is the purpose of integrating user tracking and management with HR processes?
??x
The integration ensures consistent and accurate tracking of user information such as job roles, departmental affiliations, and access levels, which are critical for managing security and compliance in an enterprise environment. This integration helps in maintaining a centralized view of user permissions that align with their job responsibilities.
x??

---

#### Centralized User-Authentication Portal
A centralized user-authentication portal validates two-factor credentials for users requesting access to enterprise resources.

:p What does the centralized authentication process typically validate?
??x
The centralized authentication process typically validates both the identity (username or service account) and the credentials provided by the user. This ensures that only authorized individuals can access enterprise resources, enhancing security.
x??

---

#### Unprivileged Network
An unprivileged network that closely resembles an external network but operates within a private address space is defined and deployed. This network connects to the internet and limited infrastructure and configuration management systems.

:p What is the purpose of the unprivileged network?
??x
The purpose of the unprivileged network is to provide a secure, controlled environment for devices that do not require full access to internal enterprise resources. It acts as a buffer between external threats and sensitive internal systems, reducing the risk of unauthorized access or data breaches.
x??

---

#### Managed Device Network Access Control
All managed devices are assigned to this network while physically located in the office, and there needs to be a strictly managed access control list (ACL) between this network and other parts of the network.

:p What is an ACL and why is it necessary?
??x
An Access Control List (ACL) is a security mechanism that defines which devices or users are allowed to communicate with each other. It is necessary in the context of managing device networks to ensure that only authorized traffic can pass between different segments of the network, maintaining network integrity and security.
x??

---

#### Internet-Facing Access Proxy
Enterprise applications are exposed via an internet-facing access proxy that enforces encryption between the client and the application.

:p What is the role of an internet-facing access proxy?
??x
An internet-facing access proxy serves as a gateway for external clients to securely connect to enterprise resources. It ensures encrypted communication, providing a layer of security and protecting sensitive data during transmission.
x??

---

#### Access Control Manager (Endpoint Verification)
An access control manager interrogates multiple data sources to determine the level of access given to a single user or device at any point in time.

:p What is endpoint verification?
??x
Endpoint verification is a process where an access control manager checks and confirms the current level of access for a specific user or device. This ensures that users and devices have appropriate permissions based on their roles, locations, and other factors.
x??

---

#### Authentication Types
Authentication determines who you are. User accounts represent data scientists, business analysts, or administrators, while service accounts are managed by Cloud IAM.

:p What is the difference between a user account and a service account?
??x
A user account represents human users such as data scientists, business analysts, or administrators, used for interactive access to applications. A service account, on the other hand, is a non-human account managed by Cloud IAM that provides automated access to resources based on predefined roles and permissions.
x??

---

#### API Credentials
Cloud APIs reject requests that do not contain valid application credentials (no anonymous requests are processed). Valid credential types include API keys and OAuth 2.0 client credentials.

:p What are the two main types of valid credentials for Cloud APIs?
??x
The two main types of valid credentials for Cloud APIs are API keys and OAuth 2.0 client credentials. API keys are used to identify registered applications, while OAuth 2.0 client credentials provide a more secure two-factor authentication mechanism suitable for interactive user access.
x??

---

#### Access Control in Detail
Access control encompasses authentication, authorization, and auditing. Authentication determines who you are, authorization determines what you can do, and auditing logs record what you did.

:p What does the term "access control" refer to?
??x
Access control refers to the process of managing who (authentication) is allowed to access resources (authorization), and ensuring that all actions taken by users or devices are logged for audit purposes. It involves several components including authentication, authorization, and auditing.
x??

---

---
#### Service Accounts and Application Credentials
Service accounts provide both the application credential and an identity. They enable applications to access resources on behalf of a user or another service without requiring individual credentials for each interaction.

:p What is a service account, and what does it do?
??x
A service account is an identity in Google Cloud that provides both the application credential (service account key) and an identity. Service accounts allow applications to interact with Google Cloud resources securely on behalf of users or other services.
x??

---
#### Roles and Permissions
Roles determine the level of access granted to identities within a project. A role consists of a set of permissions, allowing granular control over what actions can be performed.

:p What is a role in IAM?
??x
A role in Identity and Access Management (IAM) defines the level of access an identity has within a project. Roles consist of a set of permissions that determine which actions are allowed.
x??

---
#### Custom Roles
Custom roles can be created to provide specific, granular access to resources. These roles allow you to combine or restrict predefined roles based on your organization's needs.

:p How do custom roles work?
??x
Custom roles enable the creation of roles with a tailored set of permissions that meet specific organizational requirements. You can build upon predefined roles or remove permissions from them to create more granular control over access.
x??

---
#### Role Assignment and Group Management
Roles are typically assigned to groups, which allows for easier management when users change job functions or responsibilities.

:p How do you manage role assignments for multiple users?
??x
Role assignments are often managed through groups. By assigning roles to a group, you can easily update access for all members of that group when they change jobs or responsibilities. This avoids the need to update permissions on a per-user basis.
x??

---
#### Predefined Roles and Permissions
Predefined roles like `dataViewer` provide specific sets of permissions. However, custom roles may be needed to restrict certain actions while allowing others.

:p What are predefined roles?
??x
Predefined roles in IAM come with built-in sets of permissions that define the level of access an identity has within a project. For example, the `dataViewer` role allows users to access metadata but not table data.
x??

---
#### Permission Granularity and Least Privilege Principle
IAM policies should be set up to provide the least amount of privileges necessary for each identity.

:p What is the principle of least privilege?
??x
The principle of least privilege (PoLP) states that users and applications should only have access to the resources and permissions they need to perform their tasks. This minimizes potential damage if an account or application is compromised.
x??

---
#### Resource-Level Permissions
Permissions are typically granted at a resource level, meaning identities receive specific permissions on individual datasets or tables rather than all resources in a project.

:p How are permissions managed in IAM?
??x
In IAM, permissions are usually assigned at the resource level. This means that an identity is granted access to specific datasets or tables, not just all resources within a project. This approach helps prevent permission creep and ensures that identities have only the necessary permissions.
x??

---
#### Identity-Aware Proxy (IAP)
Identity-Aware Proxy can be used as a central authorization layer for applications accessed via HTTPS, providing an application-level access control model.

:p What is IAP?
??x
Identity-Aware Proxy (IAP) acts as a central authorization layer for applications that are accessed via HTTPS. It allows you to define and enforce application-level access policies using groups, making it easier to manage access across multiple applications.
x??

---
#### Application-Level Access Control with IAP
IAP can be used to implement group-based application access control, allowing certain resources to be accessible by specific groups of users.

:p How does IAP help in managing application access?
??x
Identity-Aware Proxy (IAP) helps manage application access at the application level rather than relying on network-level firewalls. You can define policies centrally and apply them across all applications, ensuring that only authorized groups have access to resources.
x??

---

---
#### Policies
Policies are rules or guardrails that enable your developers to move quickly, but within the boundaries of security and compliance. These policies can be applied to users and resources.

:p What is a policy?
??x
A policy is a set of rules or guidelines designed to ensure secure and compliant operations while allowing flexibility for developers to work efficiently.
x??

---
#### Hierarchical Policies
Hierarchical policies allow you to create and enforce consistent policies across your organization. These can be assigned at the organizational level, business unit, project, or team levels.

:p How do hierarchical policies work?
??x
Hierarchical policies enable consistent enforcement of security and compliance rules by organizing them in a structured manner. Lower-level policies cannot override higher-level ones, ensuring that critical rules are managed centrally.
x??

---
#### Context-Aware Access
Context-Aware Access is an approach that works with the zero-trust network security model to enforce granular access control based on user identity and request context.

:p What is Context-Aware Access?
??x
Context-Aware Access is a method of enforcing fine-grained access control by considering various attributes such as the device being used, IP address, and other contextual factors. It improves security by limiting access based on real-time conditions.
x??

---
#### Identity-Aware Proxy (IAP)
Identity-Aware Proxy (IAP) enables employees to access corporate apps and resources from untrusted networks without using a Virtual Private Network (VPN).

:p What is IAP?
??x
Identity-Aware Proxy (IAP) is a service that allows secure access to internal applications and resources from external or untrusted networks. It works by verifying the user's identity before granting access.
x??

---
#### Cloud Identity and Access Management (Cloud IAM)
Cloud Identity and Access Management (IAM) manages permissions for cloud resources, enabling fine-grained control over who can access what.

:p What is Cloud IAM?
??x
Cloud IAM is a service that manages permissions and access controls for cloud resources. It provides granular control over who can perform specific actions on resources.
x??

---
#### Access Context Manager
Access Context Manager (ACM) is a rules engine that enables fine-grained access control based on context.

:p What is Access Context Manager?
??x
Access Context Manager is a rules engine that allows for dynamic and context-aware access policies. It evaluates user requests against predefined rules to determine whether access should be granted.
x??

---
#### Endpoint Verification
Endpoint verification involves collecting device details to ensure compliance and security of the devices accessing corporate resources.

:p What is endpoint verification?
??x
Endpoint verification is a process that collects information about user devices to ensure they meet certain security standards before granting network or resource access.
x??

---
#### Data Loss Prevention (DLP)
Data Loss Prevention (DLP) helps in identifying and protecting sensitive data by scanning data stores for known patterns.

:p What is DLP?
??x
Data Loss Prevention (DLP) is a system that scans data stores to identify and protect sensitive information. It uses pattern matching to detect sensitive data like credit card numbers, personal addresses, etc.
x??

---
#### Cloud Data Loss Prevention
Cloud DLP can be used with AI methods to scan tables and files for protecting sensitive data.

:p How does Cloud DLP work?
??x
Cloud Data Loss Prevention (DLP) leverages AI methods to automatically detect and protect sensitive data in tables and files. It scans the data using predefined patterns and rules to identify potential security risks.
x??

---
#### Example: Context-Aware Access Rules
Context-aware access can provide different levels of access based on device type, patch status, etc.

:p How does context-aware access determine user permissions?
??x
Context-aware access determines user permissions by evaluating multiple factors such as the device being used and its security posture. For example, an employee might get full edit access from a managed corporate device but only view access if accessing from a non-patched device.
```java
public class ContextAwareAccess {
    public boolean hasPermission(User user, Device device) {
        // Check device type and patch status
        if (device.isCorporateManaged() && !user.hasUnpatchedDevice()) {
            return true; // Allow full access
        } else {
            return false; // Grant limited or no access
        }
    }
}
```
x??

---

#### Cloud DLP Scan for BigQuery Tables
Background context: The document describes how to use Cloud DLP (Data Loss Prevention) to scan and de-identify sensitive data in BigQuery tables. Cloud DLP provides tools that can automatically detect patterns, formats, and checksums of various information types.
:p What is the primary purpose of using Cloud DLP for scanning a BigQuery table?
??x
The primary purpose of using Cloud DLP for scanning a BigQuery table is to identify sensitive data within the table so that appropriate de-identification measures can be applied. This helps in ensuring compliance with data privacy regulations and protects sensitive information from unauthorized access.
x??

---

#### De-Identifying Sensitive Data Using Cloud DLP
Background context: After identifying sensitive data, Cloud DLP offers various methods to de-identify this data, such as masking, tokenization, pseudonymization, and date shifting. These techniques allow for the protection of data without replicating customer data.
:p How can you use Cloud DLP to protect identified sensitive data?
??x
You can use Cloud DLP to redact or otherwise de-identify sensitive data by applying various techniques such as masking (replacing parts of text with asterisks), tokenization (replacing sensitive information with tokens), pseudonymization (replacing personally identifiable information with fictional identifiers), and date shifting (modifying the dates in a way that does not reveal the original date). These methods ensure that while data is protected, its utility for analysis or storage remains intact.
x??

---

#### Encryption Overview
Background context: The document explains how encryption can be used to protect sensitive data even if it falls into an attacker's hands. Encryption creates a "chokepoint" by centrally managing keys and enforcing access controls through audit trails. It also contributes to privacy by allowing manipulation of data for backup purposes without exposing content.
:p What is the role of encryption in protecting data?
??x
Encryption plays a crucial role in protecting data by ensuring that even if an attacker gains access to storage devices containing your data, they cannot understand or decrypt it without the correct encryption keys. Encryption acts as a "chokepoint" where access to data can be enforced and audited centrally. It also enhances privacy by allowing systems to manipulate data for backup purposes while preventing engineers from accessing content.
x??

---

#### Customer-Managed Encryption Keys (CMEK)
Background context: The document discusses the use of CMEKs, which are customer-managed encryption keys stored in a key management system like Cloud KMS. This allows organizations to have control over their own encryption keys for compliance or security reasons. Multiple layers of key wrapping protect master keys from exposure outside of KMS.
:p How do CMEKs work to protect data?
??x
CMEKs, managed by the customer in a key management system like Cloud KMS, ensure that sensitive data is encrypted using keys that only the organization has control over. This approach reduces the risk of unauthorized access and meets regulatory compliance requirements. The use of multiple layers of key wrapping prevents exposure of master keys outside of KMS, providing an additional layer of security.
x??

---

#### Envelope Encryption with CMEK
Background context: Envelope encryption involves using a data encryption key (DEK) to encrypt sensitive data and a key encryption key (KEK) managed in Cloud KMS to protect the DEK. This method ensures that even if the DEK is leaked, it cannot be used without access to the KEK.
:p How does envelope encryption with CMEK work?
??x
Envelope encryption works by using a Data Encryption Key (DEK) to encrypt sensitive data and a centrally managed Key Encryption Key (KEK) in Cloud KMS. The KEKs are rotated through key rings, further securing the master keys. When accessing encrypted data, native cloud tools request unwrapping of the DEK from Cloud KMS using the KEK. This method reduces the risk of unauthorized access since an unprotected DEK cannot be used without the corresponding KEK.
x??

---

#### Crypto-Shredding for User Data
Background context: The document explains that deleting records associated with a specific user can be achieved by assigning a unique encryption key to each user ID and encrypting all sensitive data related to that user. This approach allows records to be deleted simply by removing the corresponding encryption key, making the user's data unusable.
:p How does crypto-shredding work?
??x
Crypto-shredding works by associating a unique encryption key with each user ID and encrypting all sensitive data related to that user using this key. To delete a user’s records, you can simply remove the associated encryption key. This approach ensures that once the key is deleted, the user's records become unusable in all tables of your data warehouse, including backups and temporary tables.
x??

---

---
#### k-anonymity
Background context explaining the concept of k-anonymity. This ensures that each group in a dataset has at least k individuals, preventing identification.
:p What is k-anonymity?
??x
K-anonymity is a technique used to ensure privacy by ensuring that each group in a dataset contains at least k individuals. If any individual can be identified as part of the data, they could potentially be linked back to their personal information through various queries or cross-matching methods.
For example, if you have a database with 10 people and set k = 3, then every combination of attributes (like age group, location) must contain at least 3 records. This way, even if someone knows the details of an individual's record, they cannot be certain that it belongs to any specific person without more information.
```java
public class KAnonymityExample {
    public boolean isKAnonymous(int k, List<Map<String, String>> dataset) {
        for (Map<String, String> record : dataset) {
            int count = 0;
            for (Map<String, String> otherRecord : dataset) {
                if (record.equals(otherRecord)) continue;
                if (isSimilar(record, otherRecord)) count++;
                if (count >= k - 1) return true; // At least k-1 similar records
            }
        }
        return false; // Not k-anonymous
    }

    private boolean isSimilar(Map<String, String> record1, Map<String, String> record2) {
        // Implement logic to check if two records are similar based on attributes
        return true; // Simplified for example purposes
    }
}
```
x??
---

#### Adding Noise to Data
Background context explaining how adding "statistically insignificant noise" can be used to ensure differential privacy by preserving individual privacy while maintaining data utility.
:p How does adding noise to a dataset help maintain privacy?
??x
Adding "statistically insignificant noise" to datasets, particularly for discrete values like age or gender, helps preserve the privacy of individuals. The idea is that small random fluctuations in the data make it difficult to infer exact values about an individual while still allowing the overall trends and patterns to be observed.

For example, if you have a dataset where the average salary is $50,000, by adding some noise (e.g., randomly adjusting this value by a few hundred dollars), the resulting number might be something like $49,783. This small change can make it harder to pinpoint the exact salary of an individual while still providing useful aggregated information.

The formula for adding noise could look like this:
```java
double noisyValue = originalValue + randomNoise;
```
where `randomNoise` is a value drawn from a probability distribution (e.g., Laplace or Gaussian) with appropriate parameters to ensure that the noise does not significantly distort the overall data trends.
x??
---

#### l-Diversity and t-Distance
Background context explaining additional techniques like l-diversity and t-distance, which generalize the data and reduce granularity further to enhance privacy.
:p What are l-Diversity and t-Distance?
??x
l-Diversity and t-Distance are advanced techniques used in differential privacy to enhance privacy while maintaining utility. These methods aim to prevent re-identification by adding more complexity to how sensitive attributes are handled.

- **l-Diversity**: This technique involves ensuring that for each combination of quasi-identifiers (e.g., age, gender), there are at least l different values of the sensitive attribute (e.g., salary) in the dataset. For example, if l = 3, then every combination of age and gender should have at least three distinct salaries associated with it.

- **t-Distance**: This concept involves ensuring that for each combination of quasi-identifiers, there is a minimum distance t between any two values of the sensitive attribute. For instance, if t = 5000, then within every group defined by age and gender, no salary difference should be less than $5000.

These techniques help in making it much harder to re-identify individuals based on patterns or specific combinations of attributes.
x??
---

