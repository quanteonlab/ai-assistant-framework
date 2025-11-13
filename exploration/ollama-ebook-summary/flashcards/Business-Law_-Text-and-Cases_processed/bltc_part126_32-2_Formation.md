# Flashcards: Business-Law_-Text-and-Cases_processed (Part 126)

**Starting Chapter:** 32-2 Formation of the Agency Relationship

---

#### Agency by Estoppel
Agency by estoppel occurs when a principal causes a third person to believe that another person is their agent, and the third person acts on this belief to their detriment. The principal is then "estopped" or prevented from denying the agency relationship. This doctrine applies even if no formal authority has been granted.
:p What does it mean for a principal to be "estopped" in an agency by estoppel situation?
??x
In an agency by estoppel scenario, the principal cannot deny that another person is their agent because they have induced a third party to rely on this belief, and the third party acted detrimentally based on that reliance. This legal principle prevents the principal from claiming otherwise.
x??

---

#### Reasonable Belief of Agency Relationship
For agency by estoppel to apply, it must be shown that the third person reasonably believed an agency relationship existed. An ordinary, prudent person familiar with business practices and customs would have been justified in concluding that the agent had authority.
:p How can a court determine if a third party's belief in an agency relationship was reasonable?
??x
A court would consider whether a reasonable, cautious person in the position of the third party, knowing all relevant facts and acting under similar circumstances, would believe that the agent had authority. This involves evaluating factors such as the principal's conduct, statements, or actions.
x??

---

#### Created by Principal’s Conduct
Agency by estoppel is not created solely by an agent's acts or declarations; rather, it is the deeds or statements of the principal that create this legal doctrine. The third party must show that the principal gave them reason to believe in the agent's authority.
:p How does a principal's conduct lead to agency by estoppel?
??x
A principal creates agency by estoppel through their own actions, declarations, or behavior. For example, if a principal allows an unauthorized person to act on their behalf repeatedly and openly without disputing it, this can create an appearance of authority that justifies the third party's reasonable belief in the agent’s authority.
x??

---

#### Case Example: Francis Azur vs Chase Bank
In the case where Francis Azur, CEO of ATM Corporation, allowed Michelle Vanek (his personal assistant) to take unauthorized cash advances from his credit card, and he later tried to deny her authority, the court concluded that Azur was estopped. This decision hinged on Azur's repeated actions over seven years, which created an appearance of authorization.
:p What led to the application of agency by estoppel in the Azur case?
??x
Azur was estopped from denying Vanek’s authority because his repeated actions and lack of action over seven years gave Chase Bank reason to believe that Vanek had the CEO's approval for cash advances. This created an appearance of authorization, even though it wasn't formal.
x??

---

#### Application to Hospital Negligence
The case involving Akron General Health System and Aaron Riedel highlights whether a hospital can be held liable when its emergency room physician, who is an independent contractor, causes negligence leading to patient harm. The hospital argued that the physician was not their agent or employee.
:p Can a hospital be held liable for the actions of an independent contractor in this context?
??x
The court would consider if the hospital's conduct created an appearance of agency through which the independent contractor acted as an agent, thus applying the doctrine of agency by estoppel. If the hospital allowed the physician to represent them or create an appearance of authority, it could be held liable for any resulting negligence.
x??

---

---
#### Principal-Agent Relationship Duties

Background context: The principal-agent relationship is fiduciary, based on trust. Both parties owe each other a duty to act with utmost good faith.

:p What are the five main duties an agent owes to their principal?
??x
An agent owes the principal the following five duties:

1. **Performance**: The agent agrees to use reasonable diligence and skill in performing the work.
2. **Notification**: The agent must notify the principal of all matters that come to her or his attention concerning the subject matter of the agency.
3. **Loyalty**: The agent must not interfere with the principal's business or engage in activities that compete with the principal’s interests.
4. **Obedience**: The agent must comply with lawful instructions given by the principal.
5. **Accounting**: The agent must provide a full accounting of all actions taken on behalf of the principal.

??x
---

#### Standard of Care for Agents

Background context: The standard of care required from an agent is typically that expected of a reasonable person under similar circumstances, interpreted to mean ordinary care. However, if an agent represents themselves as having special skills, they are expected to exercise those specific skills.

:p What happens when an agent fails to perform their duties?
??x
If an agent fails to perform their duties, liability for breach of contract may result. For example, if the agent has represented themselves as possessing special skills but fails to use them appropriately, this constitutes a breach of duty and can lead to legal consequences.

Example: 
```java
public class Agent {
    private boolean hasSpecialSkills;
    
    public void performTask() throws Exception {
        if (!hasSpecialSkills) {
            // Perform task with ordinary care
            System.out.println("Performing task with ordinary care.");
        } else {
            // Attempt to perform the task using special skills, but fail to do so appropriately
            throw new Exception("Failed to use special skills properly.");
        }
    }
}
```
??x
---

#### Gratuitous Agents

Background context: Not all agency relationships are based on a formal contract. In some cases, an agent acts gratuitously—without payment. A gratuitous agent is subject only to tort liability.

:p How does the law view gratuitous agents when they fail to perform their duties?
??x
A gratuitous agent cannot be liable for breach of contract because there is no contract; hence, they are not bound by the same standards as contractual agents. However, a gratuitous agent must still perform in an acceptable manner and is subject to the same standards of care and duty to perform.

Example:
```java
public class GratuitousAgent {
    public void act() throws Exception {
        // Attempting to act acceptably but failing due to negligence
        if (shouldPerformTask()) {
            System.out.println("Attempting to perform task.");
        } else {
            throw new Exception("Failed to perform task negligently.");
        }
    }

    private boolean shouldPerformTask() {
        return false; // Simulating a failure
    }
}
```
??x
---

#### Duty of Notification

Background context: An agent has the duty to notify the principal of all matters that come to their attention concerning the subject matter of the agency.

:p What is an example where the duty of notification applies?
??x
Example 32.7 involves Perez, who is about to negotiate a contract to sell paintings to Barber’s Art Gallery for $25,000. Perez's agent learns that Barber is insolvent and will be unable to pay for the paintings. The agent has a duty to inform Perez of Barber’s insolvency because it is relevant to the subject matter of the agency.

```java
public class NotificationExample {
    public void notifyClient() throws Exception {
        // Simulating learning about Barber's insolvency
        boolean isInsolvent = true;
        
        if (isInsolvent) {
            throw new Exception("Barber is insolvent and cannot pay for the paintings.");
        }
    }
}
```
??x
---

---
#### Nature of Business and Loss Determination
Background context: When a principal is liable for an agent's lost profits, courts often look at the nature of the business and the type of loss to determine the amount. This rule applies even if the acts are performed by gratuitous agents (agents who act without formal compensation but with implied authority).
:p What does the court consider when determining a principal's liability for an agent's lost profits?
??x
Courts typically examine the nature of the business and the type of loss to decide the amount of liability. For example, if a principal grants an exclusive territory to an agent and later competes within that territory, the principal could be liable for the agent’s lost sales or profits.
x??
---

---
#### Cooperation Duty of Principal
Background context: A principal has a duty to cooperate with their agent and assist in performing their duties. This means principals should not do anything that would prevent agents from fulfilling their obligations. For example, if a principal grants an exclusive territory, they cannot compete within it or allow another agent to do so.
:p What is the cooperation duty of a principal?
??x
A principal must cooperate with their agent and assist them in performing their duties by doing nothing to prevent that performance. For instance, if a principal grants an exclusive territory, they should not sell products themselves or allow other agents to compete within that territory.
x??
---

---
#### Exclusive Agency Agreement Example
Background context: An example of an exclusive agency agreement is provided where Penny (the principal) grants Andrew (the agent) the right to sell her organic skin care products exclusively in a certain territory. If Penny sells these products herself or allows another agent to do so, she violates this exclusive agency and could be held liable for the agent’s lost sales.
:p In Example 32.11, what does Penny (the principal) do that violates the exclusive agency agreement?
??x
Penny (the principal) violates the exclusive agency agreement by starting to sell organic skin care products herself within Andrew’s territory or permitting another agent to do so.
x??
---

---
#### Real Estate Case Example
Background context: The provided case involves Christopher Jones and Andrea Woolston, where Jones engaged Woolston as his exclusive buyer's agent. Over several months, Woolston spent a significant amount of time searching for properties suitable for Jones, conducting multiple viewings, and researching town halls.
:p In the real estate case example, what did Andrea Woolston do to assist Christopher Jones?
??x
Andrea Woolston assisted Christopher Jones by spending a substantial amount of time searching for properties, researching available properties at six town halls, showcasing several properties personally, and having multiple appointments where they viewed multiple properties together. Additionally, she visited many properties alone to determine their suitability.
x??
---

#### Agency Formation and Duties
Background context: This section discusses the formation of agency relationships, duties owed by principals to agents, and scenarios where these duties might be breached. It also touches on the distinction between employees and independent contractors.

:p Who is the principal and who is the agent in this scenario? By which method was an agency relationship formed between Scott and Blatt?
??x
Scott is the agent, while James Blatt is the principal. The agency relationship was formed by a contract that stated "Nothing in this contract shall be construed as creating the relationship of employer and employee." This indicates that they intended to create an independent contractor relationship rather than an employment relationship.

The contract specifies that it is terminable at will by either party, which further supports the independent contractor classification. :?
---

#### Instructions and Authority
Background context: The text explains how principals can instruct agents about their authority, and in some cases, these instructions limit what the agent can do on behalf of the principal.

:p Can you provide an example where Logan’s actions might be within his inherent authority despite Gutierrez's instruction not to purchase more inventory?
??x
In this situation, Logan could argue that purchasing more inventory is part of managing the business and maintaining sufficient stock levels. Therefore, even though Gutierrez instructed him not to purchase any more inventory for the month, Logan may have the inherent authority to make such a decision in response to an urgent order.

If the local business canceled their order, Logan would likely be safe from indemnification demands by Gutierrez because he acted within what is commonly understood as part of managing the company's operations. :?
---

#### Instructions vs. Advice
Background context: The text discusses how it can be difficult to distinguish between instructions that limit an agent’s authority and advice that does not.

:p Explain the difference between instructions and advice in the context of Logan's situation.
??x
Instructions are clear directives from the principal (Gutierrez) that limit or restrict the actions the agent (Logan) can take. For example, Gutierrez specifically told Logan "Don’t purchase any more inventory this month," which is a direct instruction.

Advice, on the other hand, is guidance given by the principal but does not necessarily restrict the agent’s authority to act within certain limits. If Gutierrez had said something like “Be careful with purchasing decisions” or “Think twice before buying new stock,” that would be considered advice rather than an instruction limiting Logan's authority.

In this case, Gutierrez's statement is clear and direct, making it an instruction rather than advice. :?
---

#### Employee vs. Independent Contractor
Background context: The text provides a scenario where Scott (the agent) may have been misclassified as an independent contractor instead of an employee by James Blatt (the principal).

:p What facts would the court consider most important in determining whether Scott was an employee or an independent contractor?
??x
The court would likely consider several key factors, including:

- Whether Scott had control over her own working hours and manner of work.
- The degree to which Scott's job was integrated into Blatt’s business.
- The level of supervision provided by Blatt, if any.
- Whether Scott was paid a fixed salary or on a commission basis.
- The extent to which the work performed by Scott was essential to Blatt’s core operations.
- The nature of Scott’s services (whether they were personal in nature or could be performed by someone else).

In this case, facts like Scott financing her own office and staff, receiving payment based on performance rather than a fixed salary, and being able to sell products from competitors’ companies would suggest that she was likely an independent contractor. :?
---

#### Breach of Agency Duties
Background context: The text discusses the duties owed by principals to agents in an agency relationship, including indemnification.

:p Which of the four duties that Blatt owed Scott in their agency relationship has probably been breached?
??x
Blatt probably breached the duty of indemnification. By withholding client contact information from Scott, Blatt may have hindered her ability to perform her duties effectively and sell more insurance for Massachusetts Mutual. This action could potentially result in lost sales and commissions, which would make Blatt liable to indemnify Scott.

The specific duty of indemnification is typically to protect the agent from losses incurred due to the principal’s actions or omissions. In this scenario, withholding contact information can be seen as an omission by the principal that has negatively impacted the agent's performance. :?
---

#### Employee versus Independent Contractor: Hemmerling Case
Stephen Hemmerling was a driver for Happy Cab Company. He paid certain fixed expenses, followed various rules regarding cab usage and working hours, and solicited fares according to company guidelines. However, rates were set by the state, and Happy Cab did not withhold taxes from Hemmerling's pay.

Hemmerling was injured in an accident while driving the cab and filed a claim for workers' compensation benefits, which are not available to independent contractors.
:p On what basis might the court hold that Hemmerling was an employee?
??x
The court might hold that Hemmerling was an employee based on several factors. First, despite paying fixed expenses, Hemmerling followed specific rules set by Happy Cab Company regarding cab usage and working hours. Second, although rates were state-set, this does not necessarily negate the employer-employee relationship if other factors point to control and dependency. Third, since taxes were not withheld from his pay, it suggests that he was not treated as an independent contractor by the company.

The key factor is whether Hemmerling was subject to sufficient control or direction by Happy Cab Company in performing his work.
??x
The answer with detailed explanations includes considering the following:

- **Control and Direction**: Hemmerling had to follow specific rules, which indicates that he was under the supervision of the employer. This is a significant factor suggesting an employee relationship.

- **Fees and Expenses**: Paying fixed expenses does not automatically make him independent; the nature of those expenses matters. If they are for equipment or materials used in performing services, it might be more indicative of an independent contractor status.

- **Tax Treatment**: Taxes were not withheld from Hemmerling's pay, which is a common practice with employees. This suggests that Happy Cab Company treated him as a direct employee rather than an independent contractor.

In summary:
```java
public class EmployeeVsContractor {
    public boolean isEmployee(String[] factors) {
        // Factors include control and direction, fees and expenses, tax treatment
        for (String factor : factors) {
            if (factor.equals("Control and Direction")) return true;
            if (factor.equals("Tax Treatment") && !factor.equals("Independent Contractor Status")) return true;
        }
        return false;
    }
}
```
This code example helps to illustrate the logic in determining employee status based on key factors.
x??

---

#### Employment Relationships: Moore Case
William Moore owned Moore Enterprises, a wholesale tire business. His son Jonathan worked as an employee while in high school. Later, Jonathan started his own business called Morecedes Tire and began selling regrooved tires to businesses, including his father's company.

A decade later, William offered Jonathan work with Moore Enterprises on the first day telling him to load certain tires without detailed instructions.
:p Was Jonathan an independent contractor?
??x
The court might determine that Jonathan was an independent contractor based on several factors. First, despite working for his father earlier in high school, there is no clear indication of a continued employer-employee relationship post-high school and after establishing his own business. Second, the fact that William did not provide detailed instructions to Jonathan suggests he had considerable autonomy in performing his tasks.

The key factor here is whether Jonathan was given significant control over how he performed his work.
??x
The answer with detailed explanations includes considering the following:

- **Control Over Work**: If Jonathan was told to load tires without specific instructions, it indicates that he had substantial freedom and discretion in determining how to complete the task. This is a hallmark of an independent contractor relationship.

- **Business Independence**: Jonathan's establishment of his own business as Morecedes Tire suggests a high degree of independence from Moore Enterprises. This further supports the notion that he was not working under the same conditions as when he was employed by his father in high school.

In summary:
```java
public class EmploymentRelationship {
    public boolean isIndependentContractor(String[] factors) {
        // Factors include control over work and business independence
        for (String factor : factors) {
            if (factor.equals("Control Over Work") && !factor.equals("Detailed Instructions")) return true;
            if (factor.equals("Business Independence") && factor.equals("Own Business")) return true;
        }
        return false;
    }
}
```
This code example helps to illustrate the logic in determining independent contractor status based on key factors.
x??

---

#### Agent’s Duties to Principal: Miller v. Harris
William and Maxine Miller were shareholders of Claimsco International, Inc. They filed a suit against other shareholders, Michael Harris and the accountant John Verchota.

They claimed that at Harris's instruction, Verchota had taken actions that placed them at a disadvantage relative to the other shareholders. Specifically, Verchota allegedly adjusted the company’s books to maximize their financial liabilities and falsely reported distributions of income without transferring actual income.
:p Which duty are the Millers referring to?
??x
The Millers are likely referring to the duty of loyalty or fiduciary duty that an agent owes to his principal. This duty requires the agent to act in the best interest of the principal, avoiding conflicts of interest and protecting the principal's assets.

Specifically, Verchota, as the accountant, had a duty to ensure accurate financial reporting and proper management of Claimsco’s funds. Adjusting the books to maximize their liabilities or falsely reporting distributions breaches this fiduciary obligation.
??x
The answer with detailed explanations includes considering:

- **Duty of Loyalty**: This is a fundamental principle in agency law that requires agents to act in the best interest of their principals, avoiding conflicts and ensuring accurate financial reporting.

- **Specific Actions**: The allegations include actions such as adjusting books to maximize liabilities (which could be seen as detrimental) and falsely reporting distributions without transferring actual income. These specific actions clearly indicate a breach of fiduciary duty.

In summary:
```java
public class FiduciaryDuty {
    public boolean isBreachOfLoyalty(String[] actions) {
        // Actions include adjusting books to maximize liabilities, false reports
        for (String action : actions) {
            if (action.equals("Adjusting Books to Maximize Liabilities")) return true;
            if (action.equals("False Reports Without Income Transfer")) return true;
        }
        return false;
    }
}
```
This code example helps to illustrate the logic in determining a breach of fiduciary duty based on specific actions.
x??

---

