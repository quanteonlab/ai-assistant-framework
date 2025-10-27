# Flashcards: 2B004-Data-Governance_processed (Part 25)

**Starting Chapter:** How Does Google Handle Data. Privacy SafeADH as a Case Study

---

#### Static Checks Mechanism
Background context: ADH (Ads Data Hub) uses static checks to prevent breaches of privacy and security. These checks include looking for obvious breaches, such as listing out user IDs, and blocking certain analytics functions that can potentially expose user IDs or distill a single user's information.
:p What are the key features of the static checks mechanism in ADH?
??x
The static checks mechanism in ADH includes mechanisms to prevent exposure of user IDs and block functions that could potentially reveal individual user data. This is achieved by identifying and blocking actions like listing out specific user IDs or performing analytics operations that can distill unique information about a single user.
```java
// Pseudocode for checking if a query might expose user IDs
public boolean checkStaticViolations(String query) {
    // Implementation to detect obvious violations, e.g., direct ID exposure
    return !query.contains("SELECT userId") && 
           !query.contains("JOIN users ON");
}
```
x??

---

#### Aggregations Mechanism
Background context: ADH ensures that responses to queries are aggregated so that individual user information cannot be identified. Typically, the system only returns data for 50 or more users per query.
:p How does aggregation in ADH work to protect individual user privacy?
??x
In ADH, aggregation mechanisms ensure that each row of response data corresponds to a group of multiple users, typically at least 50 users, thus preventing the identification of any single individual. This is done by responding only with aggregate data.
```java
// Pseudocode for aggregating query results
public List<Map<String, Object>> aggregateResults(List<Map<String, Object>> rawResults) {
    // Aggregate by grouping and counting distinct user IDs, ensuring no fewer than 50 users per group
    Map<String, Integer> aggregatedCounts = new HashMap<>();
    rawResults.forEach(result -> {
        String userId = (String) result.get("userId");
        if (!aggregatedCounts.containsKey(userId)) {
            aggregatedCounts.put(userId, 1);
        } else {
            int count = aggregatedCounts.get(userId);
            if (++count >= 50) { // Only aggregate when the user has at least 50 records
                aggregatedCounts.put(userId, count);
            }
        }
    });
    return new ArrayList<>(aggregatedCounts.entrySet());
}
```
x??

---

#### Differential Privacy Requirements
Background context: ADH employs differential privacy requirements to prevent users from gathering information about individual users by comparing sufficiently aggregated results. This is done through comparative analysis of current and previous query results.
:p What are differential privacy requirements in the context of ADH?
??x
Differential privacy requirements in ADH involve comparing current query results with previous ones, as well as within the same result set, to prevent users from identifying individual users by making subtle comparisons. This is done to ensure that even when running multiple analyses, no single user can be identified based on aggregated data.
```java
// Pseudocode for differential privacy checks
public boolean checkDifferentialPrivacyViolations(Map<String, Integer> currentResults, Map<String, Integer> previousResults) {
    Set<String> uniqueUsers = new HashSet<>(currentResults.keySet()).retainAll(previousResults.keySet());
    if (uniqueUsers.size() < 50) { // Threshold for minimum user overlap
        return true; // Violation detected
    }
    return false;
}
```
x??

---

#### Gmail's Data Extraction Mechanism
Background context: Google has built tools to extract structured data from emails, enabling assistive experiences while maintaining privacy. This is achieved by analyzing common templates in business-to-consumer emails.
:p How does Google maintain user privacy when extracting structured data from emails?
??x
Google maintains user privacy by scanning emails for common template structures, which are typically used for B2C communications. By understanding these templates and filtering out transient information, Google can extract useful data without directly viewing individual emails. This is done through a process of backtracking groups of emails to the business, generating template-free extraction patterns.
```java
// Pseudocode for extracting structured data from emails
public Map<String, String> extractStructuredData(String[] emailContent) {
    // Backtrack to identify common templates and extract relevant information
    StringBuilder templateBuilder = new StringBuilder();
    Set<String> uniqueKeywords = new HashSet<>();
    
    for (String content : emailContent) {
        if (!content.contains("transient")) { // Filter out transient sections
            uniqueKeywords.addAll(Arrays.asList(content.split(" ")));
        }
    }
    
    // Identify common patterns in keywords
    String templatePattern = getTemplateFromKeywords(uniqueKeywords);
    return extractDataUsingTemplate(emailContent, templatePattern);
}

private String getTemplateFromKeywords(Set<String> keywords) {
    // Logic to identify and build a template pattern from the keywords
    return "template_pattern";
}

private Map<String, String> extractDataUsingTemplate(String[] content, String template) {
    // Logic to extract structured data using the identified template
    return new HashMap<>();
}
```
x??

---

