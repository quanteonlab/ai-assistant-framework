# Flashcards: Business-Law_-Text-and-Cases_processed (Part 103)

**Starting Chapter:** Issue Spotters. Business Scenarios and Case Problems

---

---
#### Statute of Frauds and E-Contracts
In many countries, laws like the Statute of Frauds require certain types of contracts to be in writing. The United States is one such country where this applies, except for sales of real estate which often have their own specific requirements.

The Statute of Frauds generally requires written evidence (such as a signed contract) for agreements that cannot be performed within one year, involve the sale of goods over $500, involve interests in land, and more. E-mails can sometimes suffice under certain conditions but are not always considered formal enough without additional context.

:p Suppose that a week after Vollmer gave Lang the funds, she sent him an e-mail containing the terms of their loan agreement with her named typed at the bottom. Lang did not respond to the e-mail. Is this sufficient as a writing under the Statute of Frauds?
??x
The e-mail alone may be insufficient because it lacks active confirmation or acknowledgment from the other party (Lang). For a written contract to meet the requirements of the Statute of Frauds, there usually needs to be some form of acceptance by both parties. Absent Lang's response indicating agreement, this would not typically satisfy the writing requirement.

To enforce the oral contract, Vollmer might rely on exceptions to the Statute of Frauds or other legal theories.
x??

---
#### Exceptions to the Statute of Frauds
The Statute of Frauds allows certain exceptions for enforcement of contracts that do not have a written agreement. These include: agreements that cannot be performed within one year; suretyships; sales of goods over $500; and more.

If the court finds that the contract falls under the Statute of Frauds, it may still be enforced if an exception applies. The court recognizes every possible exception to the Statute of Frauds discussed in the chapter.

:p Assume that at trial the court finds that the contract between Vollmer and Lang falls within the Statute of Frauds. Further assume that the state in which the court sits recognizes every exception to the Statute of Frauds discussed in the chapter. What exception provides Vollmer with the best chance of enforcing the oral contract in this situation?
??x
The best chance for Vollmer would be to argue an exception such as "suretyship" or "promissory estoppel." 

Suretyship involves a promise by one party (Vollmer) to be responsible for another's debt, which can fall under exceptions to the Statute of Frauds. Promissory Estoppel is also an option if Vollmer relied on Lang’s agreement and suffered loss due to that reliance.

```java
public class SuretyshipExample {
    public boolean isSuretyshipApplicable(String partyA, String partyB) {
        // Logic to check if suretyship applies based on the contract terms
        return true; // Simplified example
    }
}
```
x??

---
#### Parol Evidence Rule and Oral Contracts
The parol evidence rule generally prevents parties from introducing oral agreements or prior written statements that contradict a valid, integrated written agreement. However, this rule does not apply if there is a material misrepresentation.

If Vollmer appeals the decision in favor of Lang and introduces new arguments regarding the Statute of Frauds for the first time, the court may consider other exceptions to the Statute of Frauds that were previously available but not raised by Lang during the trial. 

For example, if both parties relied on oral agreements and there was substantial reliance or promissory estoppel, this could provide a basis for enforcement.

:p Suppose that at trial, Lang never raises the argument that the parties’ agreement violates the Statute of Frauds, and the court rules in favor of Vollmer. Then Lang appeals and raises the Statute of Frauds for the first time. What exception can Vollmer now argue?
??x
Vollmer could argue an exception such as promissory estoppel or reliance on a prior oral agreement.

Promissory Estoppel is particularly relevant if there was significant reliance by one party (Lang) on the oral agreement, leading to detrimental actions. This would allow Vollmer to enforce the contract despite the lack of a written document.

```java
public class PromissoryEstoppelExample {
    public boolean canEnforceBasedOnReliance(String relianceDetails) {
        // Logic to check if promissory estoppel applies based on the facts provided
        return true; // Simplified example
    }
}
```
x??

---
#### GamesCo and Mid-Statelaws on Contract Enforcement
Contracts that require a writing under the Statute of Frauds generally cannot be enforced unless there is a signed, written agreement. However, if parties have performed part of the contract (like paying for goods received), this can sometimes create an implied-in-fact contract.

:p GamesCo orders $800 worth of game pieces from Mid-State Plastic, Inc., and pays for $450 worth. GamesCo then says it wants no more pieces. Can Midstate enforce a deal for the full $800?
??x
Mid-Statel may not be able to enforce the full contract because there is no written agreement for the remaining amount. However, if Mid-State has already delivered and GamesCo accepted the goods (performance), this could create an implied-in-fact contract.

The key here would be whether the delivery of $450 worth of game pieces constitutes a substantial performance that obligates both parties to continue under the original terms or not.

```java
public class ImpliedInFactExample {
    public boolean canEnforceImpliedContract(String deliveredAmount, String totalOrder) {
        if (deliveredAmount >= 500 && totalOrder == "800") {
            return true;
        } else {
            return false;
        }
    }
}
```
x??

---
#### Employment Contracts and the One-Year Rule
The Statute of Frauds often requires employment contracts to be in writing if they exceed one year. The reason for this is that such long-term agreements are significant, and written documentation ensures clarity.

In scenarios where an oral agreement is made without a writing, courts may apply exceptions based on specific circumstances (e.g., reliance or promissory estoppel).

:p On May 1, by telephone, Yu offers to hire Benson to perform personal services. On May 5, Benson returns Yu’s call and accepts the offer. Discuss fully whether this contract falls under the Statute of Frauds in the following circumstances.
??x
For part (a) where the contract calls for a one-year employment period with immediate performance:

- **Yes**, this likely requires a written agreement due to the Statute of Frauds.

For part (b) where the contract calls for an employment term of nine months starting on September 1:

- **No**, this does not require a written agreement because it is within one year and there's no indication that Benson will start immediately, which might reduce the immediate impact or reliance issues.

For part (c) where Benson must submit a research report with a deadline of two years:

- **Yes**, similar to part (a), this likely requires a written agreement due to the extended duration.

```java
public class EmploymentContractExample {
    public boolean isWrittenAgreementNeeded(String employmentTerm, String startDate, String reportDeadline) {
        if ("1 year".equals(employmentTerm) || "2 years".equals(reportDeadline)) {
            return true;
        } else if (startDate.equals("immediate") && "9 months".equals(employmentTerm)) {
            return false;
        }
        return false; // Simplified example
    }
}
```
x??

---
#### Collateral Promises and the Statute of Frauds
A collateral promise is a separate agreement that does not form an integrated part of the main contract. The Statute of Frauds generally applies to such agreements unless they fall under specific exceptions (e.g., suretyship).

In this scenario, Mallory's promise to pay for her brother’s lawn mower on credit if he fails to do so is a collateral agreement.

:p Mallory promises a local hardware store that she will pay for a lawn mower that her brother is purchasing on credit if the brother fails to pay the debt. Must this promise be in writing to be enforceable? Why or why not?
??x
This promise does not need to be in writing because it involves a collateral agreement, which can be enforced without a written document under certain circumstances.

However, if Mallory relies heavily on her promise and the hardware store acts upon it by selling the lawn mower to her brother based on this oral agreement, promissory estoppel could be used as an exception to enforce the oral agreement.

```java
public class CollateralPromiseExample {
    public boolean canEnforceOralPromises(String relianceDetails) {
        if (relianceDetails.contains("promissory estoppel")) {
            return true;
        } else {
            return false;
        }
    }
}
```
x??

---
#### Parol Evidence Rule and Contract Enforcement
The parol evidence rule generally precludes the introduction of oral agreements or prior written statements that contradict a valid, integrated written agreement. However, this rule does not apply if there is a material misrepresentation.

:p The issue spotter mentions Paula orally agrees to work with Next Corporation for two years and then is fired. What will the court say?
??x
The court would likely consider whether there was any written agreement or reliance on oral terms by either party.

If there were no written agreement but both parties acted based on a verbal understanding, the parol evidence rule might be invoked to enforce that understanding unless one party can prove material misrepresentation.

```java
public class ParolEvidenceExample {
    public boolean canEnforceOralAgreement(String relianceDetails) {
        if (relianceDetails.contains("material misrepresentation")) {
            return false;
        } else {
            return true; // Simplified example
        }
    }
}
```
x??

---
#### Assignment of Contract Rights
Background context: In an assignment, one party (the assignor) transfers their rights under a contract to another party (the assignee). This is common in business and finance, where lending institutions frequently transfer their right to receive payments on loans. The assignee gains the right to demand performance from the other original party to the contract.

:p What does an assignment involve?
??x
An assignment involves one party transferring their rights under a contract to another party. For example, if Tia has a loan with a bank and the bank assigns its right to receive payments to PNC Mortgage, Tia will make her payments to PNC Mortgage when it's time to repay the loan.
x??

---
#### Assignor and Assignee
Background context: In an assignment, the party transferring their rights is known as the assignor. The party receiving these rights is called the assignee. For instance, in Example 17.1, Tia receives a notice that the bank has transferred its rights to receive payments on her loan to PNC Mortgage.

:p Who are the parties involved in an assignment?
??x
In an assignment, the party transferring their rights (the original contractual rights) is known as the assignor, and the party receiving these rights is called the assignee.
x??

---
#### Rights of the Assignor After Assignment
Background context: When a contract right is assigned unconditionally, the assignor's rights are extinguished. This means that once an assignment occurs, the assignor no longer has any claim or interest in the transferred rights.

:p What happens to the assignor's rights after they have been assigned?
??x
After an unconditional assignment of contractual rights, the assignor's rights are extinguished, meaning the assignor no longer has any claim or interest in the transferred rights. The assignee now has the right to demand performance from the other party to the contract.
x??

---
#### Obligee and Obligor
Background context: In a bilateral contract, parties have corresponding rights and duties. The obligee is the person to whom a duty is owed, while the obligor is the person who is obligated to perform that duty. For example, in Example 17.2, Brower is the obligor because he is contracted to pay Horton $1,000.

:p Who are the obligee and obligor in a contract?
??x
In a contract, the obligee is the person to whom a duty (obligation) is owed, while the obligor is the person who is obligated to perform that duty. In Example 17.2, Brower is the obligor because he is contracted to pay Horton $1,000.
x??

---
#### Financial Institutions and Assignments
Background context: Financial institutions often assign their rights to receive payments under loan contracts to other firms, allowing for efficient business operations. For instance, a bank might assign its right to receive mortgage payments to PNC Mortgage.

:p How do financial institutions use assignments?
??x
Financial institutions use assignments to efficiently manage and transfer the rights to receive payments from borrowers. This allows them to recover their investments more effectively and operate smoothly. For example, if a bank assigns its mortgage payment collection rights to another firm like PNC Mortgage, it can focus on lending while PNC handles collections.
x??

---
#### Privity of Contract
Background context: The principle of privity of contract states that only the parties who entered into a contract have rights and liabilities under it. Third parties typically do not have direct rights or duties unless they are specifically involved in the agreement.

:p What is the principle of privity of contract?
??x
The principle of privity of contract means that only the parties who entered into a contract have rights and liabilities under it. This implies that a third party, who was not directly involved in the contract, generally does not have any rights or duties under it.

For instance, if Brower is contracted to pay Horton $1,000, Horton cannot sue Brower's landlord for payment since they are not parties to the original contract.
x??

---
#### Exceptions to Privity of Contract
Background context: While privity of contract generally limits third-party rights, there are exceptions. One such exception is in product liability laws, where a person injured by a defective product can recover damages even if they did not purchase the product directly.

:p What is an example of when privity of contract does not apply?
??x
An example of when privity of contract does not apply is under product liability laws. Even though someone was not the direct purchaser of a defective product, they can still recover damages if they are injured by it. This exception allows for broader accountability in cases where products cause harm.
x??

---

#### When Duties Are Personal in Nature

Background context: This concept deals with situations where a contract requires personal performance, and delegation is not allowed without the obligee's consent. The nature of duties can be either personal or nonpersonal.

:p Under what circumstances cannot contractual duties be delegated?
??x
Contractual duties cannot be delegated if they are inherently personal in nature, meaning special trust has been placed in the obligor, or performance depends on their unique skills and talents.
x??

#### Example 17.10 - Personal Performance

Background context: In this example, O’Brien contracted with Brodie for a specific task that requires Brodie's personal skills, i.e., performing veterinary surgery.

:p What happens if Brodie delegates her duties to Lopez without O'Brien's consent?
??x
The delegation is not effective because the performance required is of a personal nature and depends on the unique skills of the obligor. O'Brien must give consent for this delegation.
x??

#### When Performance by a Third Party Will Vary Materially

Background context: Here, the focus is on whether delegating duties to a third party can alter the expectations set in the original contract.

:p What if performance by a third party significantly deviates from what was expected?
??x
Delegation cannot be effective if it materially varies from the obligations agreed upon between the parties. The obligee’s expectations under the contract should remain consistent.
x??

#### Example 17.11 - Material Alteration of Expectations

Background context: This example illustrates how delegating duties can change the outcome and expectations set by one party.

:p What issue arises when Merilyn delegates her duty to Donald?
??x
Jared’s expectations are altered because he did not trust Donald's ability to select worthy recipients. Thus, this delegation is ineffective as it changes the nature of the original agreement.
x??

#### When the Contract Prohibits Delegation

Background context: An antidelegation clause in a contract explicitly states that duties cannot be delegated.

:p What happens if there’s an explicit prohibition against delegation?
??x
Delegation is not allowed under such contracts, even to other parties within the same firm. In Example 17.12, Belisario cannot delegate the auditing duty despite being part of the same company.
x??

#### Effect of a Delegable Duty

Background context: This concept explains that once duties are validly delegated, the obligee must accept performance from the delegatee.

:p What obligation does the obligee have when a valid delegation occurs?
??x
The obligee is legally required to accept performance from the delegatee. Refusal to do so can only happen if the duty cannot be delegated in the first place.
x??

#### Example 17.13 - Obligee’s Acceptance

Background context: This example demonstrates that once duties are validly delegated, the obligee must accept them.

:p What must Alicia do regarding Liam's performance?
??x
Alicia (the obligee) must accept performance from Liam (the delegatee). Refusal to perform by Liam should not affect Alicia’s obligations under the original contract.
x??

#### Liability of Delegator and Delegatee

Background context: This concept addresses that even if duties are delegated, both parties remain liable for nonperformance.

:p What is the liability status of Bryan and Liam in Example 17.13?
??x
Both Bryan (the delegator) and Liam (the delegatee) remain liable to Alicia. Alicia can sue either or both of them for nonperformance.
x??

---

---
#### Assignment of "All Rights"
Background context: When a contract includes an "assignment of all rights," it typically implies both an assignment of rights and a delegation of duties. Courts generally interpret such general language to mean that the assignor is delegating their obligations as well, making them liable if the assignee fails to perform contractual obligations.
:p What does the phrase "all rights" in a contract imply?
??x
The phrase "all rights" typically implies both an assignment of all the rights under the contract and a delegation of duties. The assignor remains responsible for performing the contractual obligations if the assignee fails to do so.
```java
// Example code to illustrate concept:
public class ContractExample {
    public void assignContract() {
        // Code to handle assignment and delegation of rights and duties
        String originalRights = "all my rights under the contract";
        boolean assignAndDelegate = true; // Assume this is a condition that triggers both actions
        if (assignAndDelegate) {
            System.out.println("Assigning all rights and delegating duties.");
        } else {
            System.out.println("Not assigning or delegating.");
        }
    }
}
```
x??
---

#### Third Party Beneficiaries
Background context: A contract can be structured to benefit a third party, in which case the original parties agree that the contract performance should be rendered directly to this third person. As an intended beneficiary, the third party gains legal rights and can sue the promisor directly if there is a breach of contract.
:p What happens when a contract is intended to benefit a third party?
??x
When a contract is intended to benefit a third party, that party becomes an "intended third-party beneficiary" who has direct legal rights. They can sue the promisor (the party making the promise) directly for any breach of the contract.
```java
// Example code to illustrate concept:
public class BeneficiaryExample {
    public void notifyBeneficiary() {
        String beneficiaryName = "John Doe";
        boolean isIntendedBeneficiary = true; // Assume this condition is met
        if (isIntendedBeneficiary) {
            System.out.println(beneficiaryName + " has the right to sue the promisor directly.");
        } else {
            System.out.println("No direct rights for this party.");
        }
    }
}
```
x??
---

#### Determining Who Is the Promisor
Background context: In a bilateral contract, both parties make promises that can be enforced. However, it's crucial to identify which of these parties is making the promise that benefits the third party, as they are considered the promisor. Allowing direct action against the promisor by the third party streamlines legal proceedings.
:p Who is the promisor in a contract involving a third-party beneficiary?
??x
In a contract where both parties make promises, the person who makes a promise that directly benefits the third party is known as the "promisor." This identification is important because it allows the third party to sue the promisor directly if there's a breach of the contract.
```java
// Example code to illustrate concept:
public class PromisorExample {
    public void identifyPromisor() {
        String partyA = "Party A";
        String partyB = "Party B";
        boolean partyAPromisesBenefit = true; // Assume Party A makes a promise benefiting a third party
        if (partyAPromisesBenefit) {
            System.out.println(partyA + " is the promisor.");
        } else {
            System.out.println("No specific promisor identified.");
        }
    }
}
```
x??
---

#### Rights and Duties Under Contract Law
Background context: The laws governing assignments and delegations specify which rights can be assigned, which duties can be delegated, and under what conditions. These rules help in determining the enforceability of such actions.
:p What are the basic principles for assigning rights and delegating duties?
??x
The basic principles for assigning rights and delegating duties include:
- Rights to receive funds, ownership in real estate, and rights to negotiable instruments can typically be assigned unless restricted by a statute or contract terms.
- Duties cannot generally be delegated without the obligor's consent. Exceptions exist if performance depends on personal skills or trust has been placed in the original performer.

```java
// Example code to illustrate concept:
public class RightsAndDutiesExample {
    public void checkAssignmentDelegation() {
        boolean canAssign = true; // Assume assignment is possible for given rights
        boolean canDelegate = false; // Assume delegation requires consent

        if (canAssign) {
            System.out.println("Rights can be assigned.");
        } else {
            System.out.println("Rights cannot be assigned.");
        }

        if (canDelegate) {
            System.out.println("Duties can be delegated.");
        } else {
            System.out.println("Duties cannot be delegated without consent.");
        }
    }
}
```
x??
---

#### Legal Status of Third-Party Beneficiaries in Contract Disputes
In this context, the legal framework surrounding third-party beneficiaries and their rights to sue when a corporate entity is suspended is discussed. The court's decision hinges on whether Bozzio can sue as a third-party beneficiary despite Missing Persons, Inc.'s suspended status.
:p What is the issue with suing as a third-party beneficiary when the promisor (corporate party) is suspended?
??x
The issue lies in whether a third-party beneficiary like Bozzio can bring an action while the contracting corporate party (Missing Persons, Inc.) is suspended. The court initially ruled that this suspension prevents her from suing under such circumstances.
x??

---

#### Contractual Waiver and Third-Party Beneficiaries
The contract contains a clause stating "not to assert any claims against Capitol," which Bozzio argues was intended for internal band disputes over royalties, not third-party beneficiary rights. This distinction is crucial in understanding the scope of the waiver.
:p How does the "look solely to" clause affect Bozzio's right as a third-party beneficiary?
??x
The "look solely to" clause is intended to prevent individual band members from suing Capitol for disputes over internal allocations and distributions of royalties that have already been properly accounted for. However, it does not explicitly address or waive the rights of third-party beneficiaries like Bozzio.
x??

---

#### California Supreme Court Interpretation
The text mentions that there are no California supreme court decisions regarding whether a suspended corporation's incapacity precludes suit by a third-party beneficiary. The court looked to intermediate appellate courts for guidance, finding that such incapacities may not necessarily bar a lawsuit.
:p What does the court conclude about suing as a third-party beneficiary when the promisee is suspended?
??x
The California Court of Appeal suggested that a third-party beneficiary's right to sue might proceed despite the promisee’s incapacity. Therefore, the district court erred in concluding that Bozzio cannot bring a claim due to Missing Persons, Inc.'s suspended status.
x??

---

#### Motion to Dismiss and Legal Analysis
The text discusses the district court's motion to dismiss based on the suspended status of the contracting corporate party. The appellate court disagreed with this dismissal, arguing that it constitutes reversible error as there is no legal precedent supporting such a conclusion.
:p Why did the district court grant the motion to dismiss?
??x
The district court granted the motion to dismiss because it believed that allowing Bozzio to sue while Missing Persons, Inc. was suspended would exploit the corporate form by permitting her to gain benefits without complying with corporate obligations. However, the appellate court overturned this decision.
x??

---

#### Case-specific Analysis and Remand
The case involves a specific recording contract where Bozzio is a third-party beneficiary. The appellate court remanded the case for further factual development to determine if she was indeed an intended third-party beneficiary and whether she waived her rights under the "look solely to" clause.
:p What does the court's decision mean for Bozzio’s claim?
??x
The court's decision means that Bozzio can pursue her claim as a third-party beneficiary despite Missing Persons, Inc.'s suspended status. The case is remanded to develop a record that will allow consideration of whether she was an intended third-party beneficiary and if the waiver applies in this context.
x??

---

