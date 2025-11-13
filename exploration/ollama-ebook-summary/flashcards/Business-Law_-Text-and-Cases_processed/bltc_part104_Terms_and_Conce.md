# Flashcards: Business-Law_-Text-and-Cases_processed (Part 104)

**Starting Chapter:** Terms and Concepts. Issue Spotters. Business Scenarios and Case Problems

---

---
#### Assignment and Delegation
Background context: Assignments and delegations involve transferring rights or duties under a contract to a third party. The terms "assignee," "assignor," "delegator," and "delegatee" are used to describe the parties involved.

:p What happens if Brian assigns his debt to Ed, but Ed doesn't pay Jeff?
??x
Jeff can sue Ed for breach of the assignment because the assignee (Ed) is now responsible for performing the assigned obligation. If Ed fails to pay, Jeff has a direct claim against Ed.
```java
// Pseudocode for understanding the scenario
public class Assignment {
    public void assignDebt(int amount, String assignor, String assignee, String obligor) {
        if (assignee.doesNotPay(obligor)) { // Assume a method that checks payment
            obligor.sue(assignee); // Jeff sues Ed for the debt
        }
    }
}
```
x??
---

#### Privity of Contract
Background context: The doctrine of privity of contract means that only parties to a contract can enforce its terms. Third parties are generally not directly bound by contracts unless they are specifically named as beneficiaries.

:p Can Good Credit Company enforce the contract against Frank after Eagle assigns its right to payment?
??x
No, Good Credit Company cannot enforce the contract against Frank because it is not a party to the original agreement between Eagle and Frank. The assignment does not create privity of contract with Frank.
```java
// Pseudocode for understanding the scenario
public class ContractEnforcement {
    public void checkEnforceability(String assignor, String assignee, String obligor) {
        if (!obligor.isPartOfOriginalContract(assignor)) { // Check if obligor is part of original contract
            System.out.println("Cannot enforce contract against " + obligor);
        }
    }
}
```
x??
---

#### Third Party Beneficiary
Background context: A third party beneficiary is someone who benefits from a contract between two other parties and can enforce the contract terms. Intended beneficiaries are those who explicitly or implicitly agreed to be bound by the contract, while incidental beneficiaries are those who benefit as a result of the performance of the contract.

:p Can Ian Faught sue Jackson for failing to enforce the insurance provision in the lease?
??x
Yes, Ian Faught can sue Jackson because he is an intended third party beneficiary of the lease provision requiring the restaurant to carry insurance. Since Jackson had a duty to ensure the restaurant has insurance and failed to do so, Faught can hold her liable.
```java
// Pseudocode for understanding the scenario
public class BeneficiaryRights {
    public void checkThirdPartyEnforcement(String beneficiary, String obligor) {
        if (beneficiary.isIntendedThirdParty()) { // Check if Faught is an intended third party
            System.out.println("Faught can sue Jackson.");
        } else {
            System.out.println("Faught cannot sue Jackson.");
        }
    }
}
```
x??
---

#### Delegation of Duties
Background context: The delegator transfers a duty to the delegatee, who then performs that task. However, the delegator remains responsible for ensuring that the duties are performed.

:p Can Jackson delegate her duty to maintain the buildings to Dunn?
??x
Yes, Jackson can delegate her duty to maintain the buildings to Dunn. The delegation does not relieve Jackson of her responsibility to ensure that the maintenance is done properly.
```java
// Pseudocode for understanding the scenario
public class Delegation {
    public void checkDelegation(String delegator, String delegatee) {
        if (delegator.isAuthorizedToDelegate()) { // Check if Jackson can delegate duties
            System.out.println("Jackson can delegate to Dunn.");
        } else {
            System.out.println("Jackson cannot delegate to Dunn.");
        }
    }
}
```
x??
---

#### Liability for Delegate’s Actions
Background context: The delegator remains responsible for the actions of the delegatee, unless there is a specific agreement to limit liability.

:p Who can be held liable for Dunn’s failure to fix the ceiling?
??x
Jackson can be held liable for Dunn’s failure to fix the ceiling because she delegated her duty to maintain the buildings. As the delegator, Jackson remains responsible for ensuring that the maintenance tasks are performed.
```java
// Pseudocode for understanding the scenario
public class Liability {
    public void checkLiability(String delegator, String delegatee) {
        if (delegatee.failsToPerformDuty()) { // Check if Dunn fails to perform duty
            System.out.println("Jackson is liable.");
        } else {
            System.out.println("No liability for Jackson.");
        }
    }
}
```
x??
---

#### Conditions in Contracts

Background context: In most contracts, promises are unconditional and must be performed. However, certain conditions can qualify these obligations based on future events.

:p What are conditions in contracts?
??x
Conditions in contracts are qualifications that base a party's performance or non-performance on the occurrence or non-occurrence of a possible future event.
x??

---

#### Conditions Precedent

Background context: A condition precedent is an event that must be fulfilled before a party’s performance can be required. It precedes the absolute duty to perform.

:p What is a condition precedent?
??x
A condition precedent is a qualification in a contract where an event, not certain to occur, must happen for the obligation to perform to arise.
x??

---

#### Example: University Housing Lease

Background context: The lease between James Maciel and Regent University (RU) was conditional on his status as a RU student.

:p What did the court's decision in Maciel v. Commonwealth of Virginia affirm?
??x
The court affirmed that being enrolled as a student at RU was a condition precedent to living in its student housing.
x??

---

#### Discharge of Contractual Duties

Background context: The primary way to discharge contractual duties is through performance, which means fulfilling the obligations as agreed.

:p How can a contract be discharged?
??x
A contract can be discharged by the performance of the duties agreed upon. This means that when one party pays and the other transfers possession (in the case of a sale), the contract is considered fulfilled.
x??

---

#### Types of Conditions in Contracts

Background context: There are three types of conditions in contracts—conditions precedent, conditions subsequent, and concurrent conditions.

:p List the three types of conditions in contracts?
??x
The three types of conditions in contracts are:
1. **Conditions Precedent**: An event that must occur before a party’s performance can be required.
2. **Conditions Subsequent**: Events that may terminate or modify an existing obligation after it has arisen.
3. **Concurrent Conditions**: Events that occur simultaneously with the performance, often seen in real estate transactions where both parties have obligations tied to each other's actions.
x??

---

#### Conditions Subsequent

Background context: A condition subsequent is an event that can terminate a party’s obligation after its duty has initially arisen.

:p What is a condition subsequent?
??x
A condition subsequent is an event that, if it occurs, can terminate or modify the existing legal obligations of the parties in a contract.
x??

---

#### Concurrent Conditions

Background context: Concurrent conditions involve simultaneous obligations where one party's action triggers another’s duty to perform.

:p Explain concurrent conditions.
??x
Concurrent conditions are events that occur simultaneously with each other, often seen in real estate transactions. For example, both parties may have reciprocal duties tied to each other's actions, such as the seller transferring property and the buyer paying for it at the same time.
x??

---

#### Implied Conditions

Background context: Conditions can also be classified as express (stated explicitly) or implied (understood from the nature of the contract).

:p What are implied conditions?
??x
Implied conditions in contracts are those that are not explicitly stated but are understood to exist based on the nature and purpose of the agreement.
x??

---

#### Example: Contractual Sale

Background context: A simple example is a contract for the sale of a vehicle where performance includes both payment and transfer of possession.

:p In a sales contract, what discharges the parties' obligations?
??x
In a sales contract, the discharge of duties occurs when both parties fulfill their obligations, such as the buyer paying $48,000 to the seller and the seller transferring possession of the Lexus to the buyer.
x??

---

---
#### Complete Performance
Complete performance occurs when a party fulfills all terms of an agreement exactly as stated. This means that conditions expressly outlined in the contract must be fully met for complete performance to occur. Any deviation from these conditions is considered a breach and discharges the other party's obligation to perform.
:p What happens if a builder fails to meet specific construction specifications?
??x
If the builder fails to meet the specific construction specifications, their performance is not considered complete. This failure would be considered a material breach of contract. The other party’s obligation to pay is discharged unless the specifications were not conditions for payment as explicitly stated in the contract.
x??

---
#### Substantial Performance
Substantial performance allows a party who has performed most but not all terms of a contract to enforce it, even if there are minor deficiencies or variances. For an action to qualify as substantial performance, certain criteria must be met:
1. The party must have performed in good faith.
2. The deviation from the original agreement should be minor and easily remedied by compensation (monetary damages).
3. The performance should still create substantially the same benefits as those promised in the contract.

:p What is required for a party to claim substantial performance?
??x
For a party to claim substantial performance, they must have:
1. Performed in good faith.
2. Not significantly deviated from the original agreement such that any variations are minor and can be easily remedied by compensation (monetary damages).
3. Created substantially the same benefits as those promised in the contract.

Courts will evaluate these requirements on a case-by-case basis, considering all relevant facts of the situation.
x??

---
#### Case in Point 18.6: Angele Jackson Guient and Borjius Guient
In this case, Angele Jackson Guient and Borjius Guient hired Sterling Doucette and his company to build their home for $177,000. After paying a total of $159,300, the Guients withheld the final $17,700 because of alleged deficiencies in work, delays, and incomplete construction.

:p What was the outcome of Doucette's breach-of-contract action?
??x
The state appellate court ruled that Doucette was not entitled to recover the balance. The Guients had taken possession of the home, which failed inspections, and required additional work by other subcontractors before they could move in. Therefore, the court determined that Doucette did not substantially perform the contract as it did not create substantially the same benefits promised.
x??

---
#### Effect on Duty to Perform
Even if one party has performed substantially, the other party’s duty to perform remains absolute. This means that the non-performing party cannot be relieved of their obligations based solely on substantial performance.

:p How does the effect on duty to perform work?
??x
If one party performs substantially but not completely, the other party still must fulfill their contractual duties unless the contract explicitly states otherwise. The doctrine of substantial performance does not absolve the non-performing party from their obligations; it only allows the performing party to seek partial enforcement or compensation for the breach.
x??

---

---
#### Mason's Contract with Jen
Mason has signed a contract to mount a new heat pump on a concrete platform, but it must be performed according to what standard? 
:p What is the typical performance standard for contracts like this?
??x
The performance standard is generally to the satisfaction of a reasonable person. However, when third parties have superior knowledge or training in the subject matter (such as a supervising engineer), courts may require personal satisfaction.
x??

---
#### Material Breach of Contract
What constitutes a material breach? 
:p Can you define what a material breach means?
??x
A material breach is when performance is not at least substantial, meaning it significantly fails to meet the contract's requirements. The nonbreaching party can be excused from their obligations and can sue for damages.
x??

---
#### Example 18.8: Garth Brooks' Breach of Contract
In this case, what did the hospital fail to do that led to a material breach? 
:p What was the specific issue in the contract between Garth Brooks and the hospital?
??x
The hospital failed to build a women’s health center as promised, which was explicitly named after Brooks's mother. This failure was deemed a material breach since it significantly deviated from the terms of the contract.
x??

---
#### Material vs Minor Breach
What distinguishes a minor breach from a material breach? 
:p Can you explain the difference between a minor and a material breach?
??x
A minor breach is not so significant that it excuses the nonbreaching party's duty to perform. However, once cured, they must resume performance. A material breach, on the other hand, significantly fails to meet contract requirements and fully excuses the nonbreaching party.
x??

---
#### Case Involving Marc and Bree Kohel
What were the key elements of the sales contract between Marc and Bree Kohel, and Bergen Auto Enterprises? 
:p What are the important details of the contract in this case?
??x
Marc and Bree Kohel agreed to purchase a used 2009 Mazda for $26,430.22 with a trade-in credit of $7,000 on their 2005 Nissan Altima. They still owed $8,118.28 on the Nissan, which was applied as a net pay-off to the new car. Upon taking possession, they found that the VIN tag of the Nissan was missing, preventing its sale.
x??

---
#### C/Java Code Example for Contractual Breach
How would you represent the situation where a vehicle's VIN tag is missing in Java code?
:p Can you write a simple Java method to check if a VIN tag is present and handle the situation accordingly?
??x
```java
public class CarSaleContract {
    private String vinTag;

    public CarSaleContract(String vinTag) {
        this.vinTag = vinTag;
    }

    /**
     * Checks if the VIN tag is present.
     * If not, it throws an exception indicating that the car cannot be sold.
     */
    public void checkVinTag() throws Exception {
        if (vinTag == null || vinTag.isEmpty()) {
            throw new Exception("VIN tag missing. Car cannot be sold.");
        }
    }

    public static void main(String[] args) {
        CarSaleContract contract = new CarSaleContract(null);
        try {
            contract.checkVinTag();
        } catch (Exception e) {
            System.out.println(e.getMessage());
        }
    }
}
```
x??

---

---
#### Material Alteration of the Contract
To discourage parties from altering written contracts, the law allows an innocent party to be discharged when the other party has materially altered a written contract without consent. For instance, suppose that a party alters a material term of a contract, such as the stated quantity or price, without the knowledge or consent of the other party.
:p What happens if one party materially alters a written contract without the consent of the other party?
??x
The innocent party can treat the contract as discharged and is no longer obligated to perform under the terms that were altered without their knowledge. This prevents one party from unilaterally changing the terms of an agreement, thereby protecting the interests of the non-altering party.
x??

---
#### Statutes of Limitations
Statutes of limitations restrict the period during which a party can sue on a particular cause of action. After the applicable limitations period has passed, a suit can no longer be brought. The limitations period for bringing suits for breach of oral contracts usually is two to three years, and for written contracts, four to five years.
:p What are statutes of limitations?
??x
Statutes of limitations are legal rules that limit the time during which a party may bring a lawsuit based on a particular cause of action. These laws ensure that legal actions are timely and that evidence remains fresh and relevant over time.
x??

---
#### Bankruptcy
A proceeding in bankruptcy attempts to allocate the debtor’s assets to the creditors in a fair and equitable fashion. Once the assets have been allocated, the debtor receives a discharge in bankruptcy. A discharge in bankruptcy ordinarily prevents the creditors from enforcing most of the debtor's contracts. Partial payment of a debt after discharge in bankruptcy will not revive the debt.
:p What is the effect of a discharge in bankruptcy on the debtor?
??x
A discharge in bankruptcy releases the debtor from personal liability for certain debts, including most contractual obligations, which means that the creditors cannot enforce these contracts against the debtor. The debtor can no longer be held personally responsible to perform under those agreements.
x??

---
#### Impossibility of Performance
After a contract has been made, supervening events (such as a fire) may make performance impossible in an objective sense. This is known as impossibility of performance and can discharge a contract. The doctrine of impossibility of performance applies only when the parties could not have reasonably foreseen, at the time the contract was formed, the event that rendered performance impossible.
:p What does the doctrine of impossibility of performance cover?
??x
The doctrine of impossibility of performance covers situations where an unforeseeable event occurs after a contract is made, making it objectively impossible to perform the agreed terms. For example, if a fire destroys the goods being sold under a contract for sale, the seller can be discharged from their obligation to deliver because this was not reasonably foreseeable at the time the contract was formed.
x??

---

#### Temporary Impossibility of Performance
Background context explaining the concept. In this case, Hurricane Katrina caused temporary damage to a house, which made it impossible for the seller to perform his contractual obligations under the original terms of the agreement. The court ruled that while the event was unexpected, the seller still had an obligation to repair the property and fulfill the contract as originally planned.
:p What is the situation described in the Payne v. Hurwitz case?
??x
In the Payne v. Hurwitz case, Hurricane Katrina damaged Keefe Hurwitz's home, causing estimated repairs of $60,000. Despite this, Hurwitz refused to repair and sell the property at the originally agreed price of$241,500. The court ruled that while the damage was significant, it did not make performance impossible; thus, Hurwitz still had to fulfill his contractual obligations.
x??

---

#### Substantial Burdens Due to Changes in Circumstances
Background context explaining the concept. In some cases, external factors can change significantly over time, making contract performance substantially more burdensome for one or both parties. For example, during World War II, Gene Autry's contract with a Hollywood movie company was suspended due to his military service. After the war ended and the purchasing power of the dollar declined, performing the original contract would have been too costly.
:p How did World War II affect Gene Autry’s contract?
??x
During World War II, Gene Autry's contract with a Hollywood movie company was temporarily impossible to perform due to his military service. After the war ended in 1945 and the purchasing power of the dollar had declined significantly, performing the original contract would have been substantially more burdensome for Autry.
x??

---

#### Impossibility of Performance as a Valid Defense
Background context explaining the concept. The doctrine of impossibility of performance applies only when an event or condition makes it impossible to fulfill contractual obligations. However, courts today are more likely to allow parties to raise this defense compared to historical standards where such events were rarely excused.
:p Why might courts increasingly accept impossibility as a valid defense?
??x
Courts may increasingly accept impossibility of performance as a valid defense because they balance the freedom to contract against the potential injustice that could arise from enforcing contracts when performance has become impractical or excessively burdensome. This approach ensures that parties are not held to obligations that have become unreasonably difficult due to unforeseen circumstances.
x??

---

#### Balancing Freedom of Contract and Justice
Background context explaining the concept. Courts must balance the freedom to enter into contracts against potential injustice if a contract is enforced when performance has become impossible or excessively burdensome. This balancing act is crucial in deciding whether to apply the impossibility doctrine.
:p How do courts balance freedom of contract with justice?
??x
Courts balance freedom of contract with justice by considering whether enforcing a contract under certain circumstances would be unfair or unreasonable. They weigh the original intent and terms of the contract against the changed reality, ensuring that parties are not held to obligations that have become unreasonably difficult due to unforeseen events.
x??

---

#### Long-term Impacts of Accepting Impossibility as a Defense
Background context explaining the concept. If courts increasingly accept impossibility of performance as a valid defense, it could lead to uncertain contract terms and potentially weaken the enforcement of agreements. This might make parties more hesitant to enter into contracts, knowing that unexpected events could lead to non-performance.
:p Why might those entering into contracts be worse off if courts accept impossibility as a defense?
??x
If courts increasingly accept impossibility of performance as a valid defense, it could result in less certainty and predictability in contractual agreements. This uncertainty might deter parties from entering into long-term or significant contracts, as they would face the risk that unforeseen events could relieve them of their obligations. As a result, the overall freedom to contract might be compromised, leading to fewer business transactions.
x??

---

