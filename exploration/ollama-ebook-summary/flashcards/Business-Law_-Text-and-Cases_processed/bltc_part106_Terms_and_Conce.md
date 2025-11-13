# Flashcards: Business-Law_-Text-and-Cases_processed (Part 106)

**Starting Chapter:** Terms and Concepts

---

---
#### Quasi Contract Overview
Quasi contract is a legal theory under which an obligation is imposed by law, even when there is no actual agreement between parties. The main purpose of quasi contracts is to prevent unjust enrichment and ensure that one party does not benefit at the expense of another without providing fair compensation.
:p What is a quasi contract?
??x
A quasi contract is a legal theory where a court imposes an obligation on a party in the absence of an actual agreement, ensuring fairness by preventing unjust enrichment. It allows recovery for benefits conferred under circumstances that make a contract implied by law.
x??

---
#### When Quasi Contract is Used
Quasi contracts are used when there is no actual contract or agreement between parties, and justice requires that one party be compensated for the benefit they provided. This theory can also apply to situations where an existing contract is unenforceable but partial performance has been made by one of the parties.
:p In what scenarios might a quasi contract be applied?
??x
A quasi contract may be used when there is no actual written agreement between parties, or when an existing contract is unenforceable due to reasons such as fraud or illegality. It can also apply if one party has partially performed under a contract that is unenforceable.
x??

---
#### Requirements of Quasi Contract
To recover under the theory of quasi contract, specific conditions must be met: 
1. The plaintiff (party seeking recovery) must have conferred a benefit on the defendant.
2. The plaintiff provided the benefit with the reasonable expectation of being paid for it.
3. The plaintiff did not act as a volunteer.
4. The defendant would be unjustly enriched if allowed to retain the benefit without paying.
:p What are the four requirements for recovering under quasi contract?
??x
The four requirements for recovering under quasi contract are:
1. The plaintiff must have conferred a benefit on the defendant.
2. The plaintiff provided the benefit with the reasonable expectation of being paid.
3. The plaintiff did not act as a volunteer.
4. The defendant would be unjustly enriched if allowed to retain the benefit without paying.
x??

---
#### Example 19.15 - Application of Quasi Contract
In this example, Ericson contracts to build two oil derricks for Petro Industries but does not have a written contract. After completing one derrick and receiving no payment, Ericson can sue under the theory of quasi contract because all conditions for recovery are met.
:p Can Ericson recover in quasi contract based on Example 19.15?
??x
Yes, Ericson can recover in quasi contract based on Example 19.15 because:
- He conferred a benefit by building one oil derrick.
- He provided the benefit with the reasonable expectation of being paid.
- He did not act as a volunteer.
- Petro Industries would be unjustly enriched if allowed to retain the benefit without paying for it.
x??

---
#### Unenforceable Contracts and Quasi Contract
If a contract is unenforceable, quasi contractual recovery may still be available. This allows the party who has performed under the unenforceable contract to recover the reasonable value of their partial performance.
:p How can a party recover in a situation where the main contract is unenforceable?
??x
A party can recover in a situation where the main contract is unenforceable by claiming quasi contractual recovery. This allows them to receive payment for the reasonable value of the work or benefits they have provided, even though the original agreement cannot be enforced.
x??

---

---
#### Consequential Damages
Consequential damages are those that result from the breach of contract and are not within the usual contemplation of both parties at the time the contract is made. These damages are often harder to prove than direct or incidental damages.

:p What would be the measure of recovery for Haney if he sues Greg?
??x
If Haney sues Greg, the measure of recovery would likely include the cost of Ipswich's work to finish the shed and any additional expenses Haney incurred as a result of not having a fully completed storage shed. These could be considered consequential damages.

Relevant: If Haney can prove that he suffered additional losses beyond the direct costs (like loss of use, inconvenience, or other indirect consequences), these would be recoverable.

```java
public class DamagesExample {
    public void measureOfDamages() {
        // Assume Haney paid Ipswich $500 to complete the shed and incurred additional expenses.
        int ipswichCost = 500;
        int additionalExpenses = 300; // Example of additional expenses
        int totalRecovery = ipswitchCost + additionalExpenses;
        System.out.println("Total recovery: " + totalRecovery);
    }
}
```
x??

---
#### Incidental Damages
Incidental damages are those that arise naturally and directly from the breach. They usually include costs incurred to mitigate the loss, such as hiring a third party to complete the work.

:p Is Lyle liable for Marley’s expenses in providing for the cattle?
??x
Lyle is not necessarily liable for Marley’s expenses in providing for the cattle because there was no indication that Lyle had any reason to know of the cattle when they made the contract. Marley would need to prove that these were direct and natural consequences of the breach, which in this case seems unlikely.

Relevant: Incidental damages are those costs directly resulting from the breach and can be mitigated by reasonable steps taken by the non-breaching party. In this scenario, there is no clear evidence that Lyle was aware of or had reason to know about the cattle before the breach.

x??

---
#### Liquidated Damages
Liquidated damages are a predetermined sum agreed upon in advance as compensation for breach of contract. They serve as a form of pre-emptive compensation and can be enforced if they are a genuine pre-estimate of loss.

:p Discuss who is correct when Cohen claims to retain the deposit.
??x
Cohen is correct in retaining the deposit because the contract specifically states that if the buyer breaches, Cohen will keep the deposit as liquidated damages. Additionally, Cohen sold the property to another party for $105,000, which is higher than the original purchase price of$100,000, further justifying his right to retain the deposit.

Relevant: Liquidated damages are typically enforceable if they represent a genuine pre-estimate of loss. In this case, it appears that both parties agreed on the liquidated damages clause, and Cohen's subsequent sale for more money does not negate his initial agreement.

```java
public class LiquidatedDamagesExample {
    public void liquidatedDamages() {
        int deposit = 10000; // $10,000 down payment
        int salePriceToBallard = 105000;
        boolean breach = true; // Windsor breached the contract

        if (breach && salePriceToBallard > 90000) {
            System.out.println("Cohen can retain the deposit.");
        } else {
            System.out.println("Windsor should get her $10,000 back.");
        }
    }
}
```
x??

---
#### Specific Performance
Specific performance is an equitable remedy that requires a party to perform their obligations as stated in the contract. It is appropriate when monetary damages are inadequate.

:p In which situation would specific performance be an appropriate remedy?
??x
In (a), specific performance would likely be appropriate because Thompson has already breached the contract by refusing to sell the property, and the contract for sale of real estate often cannot be easily replaced with money alone. In (b), specific performance is less likely because Amy's refusal to perform a service does not involve unique goods that cannot be replicated.

Relevant: Specific performance is appropriate when the subject matter of the contract is unique or irreplaceable, such as property, and monetary damages are inadequate.

```java
public class SpecificPerformanceExample {
    public void specificPerformance() {
        boolean houseIsUnique = true; // House for sale
        boolean serviceCanBePerformedElsewhere = false;

        if (houseIsUnique && !serviceCanBePerformedElsewhere) {
            System.out.println("Specific performance is an appropriate remedy.");
        } else {
            System.out.println("Specific performance may not be appropriate.");
        }
    }
}
```
x??

---
#### Limitation of Liability Clause
A limitation of liability clause caps the amount of damages a party can claim in case of breach. It often includes clauses that limit the damages for specific types of losses, such as nominal or liquidated damages.

:p Can X Entertainment seek specific performance if Bruno breaches the contract?
??x
X Entertainment cannot seek specific performance based on the given information because specific performance is not an equitable remedy for services rendered under a contract for personal labor. The limitation-of-liability clause does not affect this determination; thus, X Entertainment would need to rely on other remedies such as liquidated damages.

Relevant: Specific performance is generally not available in contracts involving personal services or goods that can be replicated easily. In this case, the contract focuses on a specific individual's work and stunts, which makes specific performance less likely.

x??

---
#### Mitigation of Damages
Mitigation of damages refers to the non-breaching party’s obligation to take reasonable steps to minimize their losses after a breach has occurred. This can include finding alternative solutions or reducing costs where possible.

:p Suppose Bruno is injured by an X Entertainment employee, what could be X Entertainment's liability?
??x
X Entertainment would likely only be liable for nominal damages in such a case because the limitation-of-liability clause states that their liability is limited to nominal damages if Bruno is injured during filming. Nominal damages are minimal and often awarded when no actual loss can be proven.

Relevant: Nominal damages are typically a small amount of money (e.g., $1) given as a symbolic acknowledgment of legal rights without substantial compensation.

```java
public class LimitationOfLiabilityExample {
    public void liabilityForInjury() {
        boolean isNominalDamage = true; // Limited to nominal damage

        if (isNominalDamage) {
            System.out.println("X Entertainment's liability for injury is limited to nominal damages.");
        } else {
            System.out.println("X Entertainment may be liable for more than nominal damages.");
        }
    }
}
```
x??

---

#### Minors and Contract Validity
Corelli is a minor when he purchases the painting. This situation involves understanding minors' rights and responsibilities in contract law.

:p Is the contract void if Corelli, a minor, purchases the painting?
??x
The contract is not necessarily void but may be voidable by Corelli after reaching adulthood. A contract entered into by a minor is generally considered voidable rather than void, meaning that once the minor reaches the age of majority (18 in this case), they can disaffirm the contract and avoid its obligations.

If Corelli decides to disaffirm the contract, he can request the return of the $2,500. However, if Shelley refuses to return it, Corelli may bring a court action to recover the money.
```java
// Example scenario
if (Corelli.isMinor() && Corelli.hasReachedMajorityAge()) {
    Corelli.requestReturnOfMoney();
} else {
    // Handle other cases where disaffirmance is not applicable
}
```
x??

---

#### Statute of Frauds and Oral Contracts
The contract between Corelli and Shelley involves an oral agreement for the sale of a painting, which is still in progress. The statute of frauds requires certain contracts to be in writing.

:p Is the contract enforceable if it was made orally?
??x
No, the contract would not be enforceable because it does not meet the requirements of the statute of frauds, which generally require written agreements for the sale of goods over a certain value. Since the painting is still being created and there's no writing documenting the agreement, Shelley’s sale to another buyer would likely be upheld in court.

In this case, Corelli cannot sue Shelley because he lacks a valid enforceable contract due to the oral nature of their agreement.
```java
// Example scenario
if (isOralContract()) {
    // Check if written form is required by statute of frauds
    if (!requiresWrittenForm()) {
        return false; // Contract not enforceable
    }
}
```
x??

---

#### Capacity to Contract
Corelli and Shelley have entered into a written contract, but Corelli’s son argues that his father was acting irrationally when he signed the agreement due to mental incompetence.

:p Is the contract enforceable if Corelli was mentally incompetent at the time of signing?
??x
The enforceability of the contract depends on whether Corelli lacked capacity at the time of signing. If Corelli was indeed incapacitated, his son may have a valid argument that the contract is unenforceable due to lack of mental competence.

Corelli’s son would need to provide evidence showing that Corelli did not understand the nature and consequences of the agreement when he signed it. If this can be proven, the court might rule in favor of disaffirming the contract.
```java
// Example scenario
if (Corelli.isIncapacitatedAtSigning()) {
    // Evaluate capacity based on evidence
    if (!Corelli.hasCapacity()) {
        return false; // Contract not enforceable due to lack of capacity
    }
}
```
x??

---

#### Impossibility of Performance
The contract requires Shelley to deliver the painting to Corelli’s gallery in two weeks, but Shelley fails to do so because her mother is ill. Corelli sues for $1,500 in damages.

:p Who will win this lawsuit if performance was impossible due to personal illness?
??x
Shelley would likely win the lawsuit because the contract specifies that the painting must be available for a third-party sale within two weeks. Shelley’s mother falling ill and requiring her care is an act of God or a supervening event, making performance objectively impossible.

Since Shelley has a valid defense based on impossibility of performance, she will not be liable to Corelli for damages.
```java
// Example scenario
if (Shelley.isIllDueToPersonalReasons() && !isPerformancePossible()) {
    return false; // Shelley is not liable due to impossibility of performance
}
```
x??

---

#### Agreement in E-Contracts and Forum Selection Clauses
Corelli accepts an offer from Shelley’s website by clicking an “I accept” box, agreeing to a forum-selection clause that specifies California courts for any disputes.

:p Can Corelli sue Shelley in a Texas state court?
??x
No, Corelli cannot sue Shelley in a Texas state court. The forum-selection clause in the contract stipulates that any disputes should be resolved by a court in California, where Shelley resides. Courts generally honor such clauses unless they are proven to be unreasonable or unconscionable.

Shelley’s claim that any suit must be filed in a California court is likely valid, and Corelli would need to file his lawsuit there.
```java
// Example scenario
if (Corelli.acceptedOffer() && forumSelectionClauseExists()) {
    if (!isForumClauseEnforceable()) {
        return false; // Suit can proceed in Texas state court
    } else {
        return true; // Must follow the agreed forum
    }
}
```
x??

