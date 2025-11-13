# Flashcards: Business-Law_-Text-and-Cases_processed (Part 100)

**Starting Chapter:** 14-2 Legality

---

---
#### Legality of Contracts
Background context: For a contract to be valid and enforceable, it must serve a legal purpose. If a contract violates federal or state statutory laws, it is deemed illegal and unenforceable. Additionally, contracts involving tortious acts (e.g., defamation, fraud) are contrary to public policy and thus invalid.

:p What constitutes an illegal contract due to illegality?
??x
A contract that involves doing something prohibited by federal or state statutory law, such as selling illegal drugs, committing fraud, or hiding violations of securities laws or environmental regulations. 
x??

---
#### Contracts Contrary to Statute
Background context: Statutes often outline what can and cannot be included in contracts. Violations include making loans at usurious interest rates (rates above the lawful maximum) or engaging in gambling activities that are otherwise illegal.

:p How do statutes affect contract legality?
??x
Statutes can render a contract illegal by setting rules for permissible transactions, such as interest rate caps on loans. If a lender exceeds these limits, the contract is unenforceable.
x??

---
#### Contracts to Commit a Crime
Background context: Any contract that involves committing a crime is illegal and void. Examples include selling illegal drugs or aiding in hiding corporate violations of laws.

:p What happens if a contract involves committing a crime?
??x
If a contract involves committing a crime, it is considered illegal and unenforceable. For instance, contracts to sell illegal drugs or hide violations of securities laws are not enforceable.
x??

---
#### Object of Contract Made Illegal by Statute
Background context: Sometimes, the object or performance of a contract becomes illegal after it has been entered into due to subsequent statutes. In such cases, the contract is considered discharged (terminated) by law.

:p What happens if the object of a contract becomes illegal post-agreement?
??x
If the object or performance of a contract becomes illegal after the agreement was made, the contract is typically terminated. For example, selling illegal drugs after both parties have agreed to such a transaction would make the original contract void.
x??

---
#### Usury and Its Effects on Contracts
Background context: Most states have laws capping interest rates for loans. If a loan exceeds these caps, it can lead to usurious contracts that are illegal but may still be partially enforceable under state law.

:p What is usury and how does it affect contracts?
??x
Usury refers to charging an interest rate on a loan that exceeds the statutory maximum set by state laws. While such contracts are illegal, states often limit lenders' ability to collect interest beyond this legal cap.
x??

---
#### Gambling Contracts
Background context: Traditionally, gambling was considered illegal and void under contract law. However, many states now permit certain forms of regulated gambling.

:p What is the current status of gambling contracts in most states?
??x
While traditional gambling contracts were often deemed illegal, many states now allow and regulate specific types of gambling, such as horse racing, video poker machines, state-operated lotteries, and Native American reservation-based gambling.
x??

---
#### Contracts by Mentally Incompetent Persons
Background context: If a person is declared mentally incompetent or lacks the capacity to understand the agreement, their contract may be voidable. The enforceability depends on whether a guardian was appointed and if the party had the ability to comprehend the contract at the time of formation.

:p What happens when a mentally incompetent person enters into a contract?
??x
If a court has declared someone mentally incompetent or lacked capacity to understand a contract, any agreement made by that individual is void from the outset. The contract can be enforced if a guardian was appointed and the party had the capacity at the time of formation.
x??

---

#### Procedural Unconscionability
Background context: Procedural unconscionability occurs when there is a lack of meaningful choice for one party due to unfair contract terms or conditions. This can include factors like inconspicuous print, unintelligible language, and disparity in bargaining power between the parties.
:p What does procedural unconscionability involve?
??x
Procedural unconscionability involves situations where one party lacks a meaningful choice due to unfair contract terms or conditions. Key factors include:
- Inconspicuous print: Small font size used to hide important details.
- Intelligibility of the language: Complex, confusing wording that makes the agreement hard to understand.
- Opportunity for questions: Limited ability for the weaker party to ask clarifying questions about the contract.
- Disparity in bargaining power: Significant imbalance between the parties' negotiating positions.

For example:
```java
// Code Example: Inconspicuous Print and Complex Language
public class Contract {
    private void showTerms(String terms) {
        // Displaying terms with small font size to make it inconspicuous
        System.out.println("\033[1m" + "\033[24m" + "Important Terms Here... \n" + terms);
    }
}
```
x??

---

#### Substantive Unconscionability
Background context: Substantive unconscionability occurs when a contract or its provisions are oppressive, overly harsh, and deprive one party of the benefits of the agreement or leave them without a remedy for non-performance by the other. Courts generally focus on whether certain provisions heavily favor one party over another.
:p What does substantive unconscionability involve?
??x
Substantive unconscionability involves terms that are oppressive and overly harsh, typically leaving one party with no meaningful benefits from the agreement or without a remedy for non-performance by the other. Courts often look at whether:
- A provision deprives one party of the benefits of the agreement.
- A provision leaves one party without a remedy for non-performance by the other.

For example, an exculpatory clause that gives a business entity access to courts but requires another party to arbitrate any disputes can be considered unconscionable due to its harsh impact on the weaker party.
```java
// Code Example: Exculpatory Clause
public class ExculpatoryClause {
    private void grantAccess(String entity, String clause) {
        if (entity.equals("Business")) {
            // Business gets access to courts; other party must arbitrate disputes
            System.out.println(clause + "\nBusiness gets full legal protection.");
        } else {
            System.out.println(clause + "\nOther party is restricted to arbitration only.");
        }
    }
}
```
x??

---

#### Exculpatory Clauses and Public Policy
Background context: Exculpatory clauses are provisions in contracts that release one party from liability, regardless of fault. Courts generally disfavor these clauses and may refuse to enforce them if they violate public policy, especially in contexts like commercial rentals, employment, or work-related injuries.
:p What is the role of exculpatory clauses?
??x
Exculpatory clauses are provisions that release a party from liability for monetary or physical injury, no matter who is at fault. They are often disfavored by courts due to their potential to violate public policy:
- Unfavorable in commercial property rental agreements.
- Almost always unenforceable in residential property leases and employment contracts.

For example:
```java
// Code Example: Exculpatory Clause Violation
public class ExculpatoryViolation {
    private void checkPolicy(String context, String clause) {
        if (context.equals("Commercial Property") || context.equals("Employment")) {
            System.out.println(clause + "\nIs against public policy and unenforceable.");
        } else {
            System.out.println(clause + "\nMay be enforceable depending on jurisdiction.");
        }
    }
}
```
x??

---

#### Unconscionability in Contracts
Background context: An unconscionable contract or clause is one that is void for reasons of public policy. It can occur if a party enters into the contract or term due to lack of knowledge, understanding, or because it was entered under duress.
:p What defines an unconscionable contract?
??x
An unconscionable contract or clause is defined as one that is void for reasons of public policy. This occurs when:
- The contract was entered into due to a party's lack of knowledge or understanding.
- The term became part of the contract under such circumstances.

Key factors courts consider include:
- Inconspicuous print: Font size used to hide important details.
- Intelligibility of language: Clarity and simplicity in terms of the agreement.
- Opportunity for questions: Ability for one party to ask clarifying questions about the contract.
- Disparity in bargaining power: Imbalance between parties' negotiating positions.

For example:
```java
// Code Example: Assessing Conspicuousness and Intelligibility
public class ContractReview {
    private boolean checkConspicuousPrint(String terms) {
        // Check for small font size or complex language to determine conspicuousness
        return terms.contains("\033[24m");
    }
}
```
x??

---

#### Summary Judgment and Ambiguity of Releases

Background context explaining that this concept deals with legal judgments, specifically regarding summary judgment on the grounds of releases being ambiguous. The text discusses how courts interpret releases and whether they are clear enough to bar claims.

:p What is a release in a legal context, and why might it be considered ambiguous?
??x
A release in a legal context refers to an agreement by which one party (the plaintiff) agrees to release another party (often the defendant or event sponsor) from any liability arising out of certain events. It can be ambiguous if its language does not clearly define who is being released, leading to disputes over whether it applies to all relevant parties.

In this case, Mrs. Holmes and her husband filed a lawsuit against KSDK after she tripped over an audio-visual box during the event. The defendants argued that there was a release in place that covered them, but the plaintiffs claimed it was ambiguous because it did not specifically name everyone involved.
x??

---

#### Interpretation of Releases

Background context explaining that interpretations of releases are governed by contract principles and must be unambiguous to be enforceable.

:p How does the court determine if a release is unambiguous?
??x
The court determines if a release is unambiguous based on whether its language can be reasonably interpreted in only one way. The text states, "Contract terms are ambiguous only if the language may be given more than one reasonable interpretation."

In this scenario, the release described several entities to be released: the St. Louis Affiliate of Susan G. Komen for the Cure, their affiliates and affiliated individuals, any Event sponsors and their agents and employees, and all other persons or entities associated with the event.
x??

---

#### Use of "Any" in Releases

Background context explaining that the use of the word "any" can be interpreted broadly unless there is evidence to suggest otherwise.

:p How does the court interpret the use of "any" in a release?
??x
The court interprets the use of "any" in a release as all-inclusive, meaning it excludes nothing and is not ambiguous. The text states that "the word 'any' when used with a class in a release is all-inclusive, it excludes nothing, and it is not ambiguous."

In this case, the release stated that it released claims against "any Event sponsors." This unambiguously releases liability for all event sponsors without exclusion.
x??

---

#### Prospective Releases

Background context explaining that there may be different standards for interpreting prospective releases compared to retroactive ones.

:p How do courts treat specificity in prospective releases?
??x
Courts require more specificity in a prospective release, which covers future acts of negligence, because it involves predicting unknown events. The text states, "plaintiffs argue that this reasoning does not apply to the use of 'any' with classes of persons in a prospective release for future acts of negligence because courts require more specificity in a prospective release."

This means that while the release was unambiguous when considering all parties involved in the event, it might need more detailed specification if it were intended to cover future actions.
x??

---

#### Summary Judgment

Background context explaining that summary judgment is an order by a judge dismissing a case without trial because there are no material facts in dispute.

:p What does summary judgment mean in this legal context?
??x
Summary judgment means that the court has determined, based on the available evidence, that there are no material facts in dispute and thus no need for a trial. In this case, the circuit court entered summary judgment in favor of KSDK because it found that plaintiffs' claims were barred by the language of the release.

The text provides details about how the release covered various entities involved with the event, and despite the plaintiffs' arguments, the court held that the release was unambiguous and applied to all relevant parties.
x??

---

#### Summary Judgment in Favor

Background context explaining that a summary judgment entered in favor means the defendant wins without a trial.

:p What does it mean when the circuit court enters summary judgment in defendants’ favor?
??x
When the circuit court enters summary judgment in defendants' favor, it means that after reviewing all the evidence, the court finds no genuine issues of material fact and grants a win to the defendants. In this case, the court ruled that plaintiffs' claims were barred by the release.

The text explains how KSDK argued their case based on the language of the release, which was interpreted as unambiguous.
x??

---

#### Plaintiffs’ Claims

Background context explaining that the plaintiffs made specific claims about an incident during the event and the injuries sustained.

:p What specific claims did the plaintiffs make in their lawsuit?
??x
The plaintiffs alleged that while Mrs. Holmes was a participant in the Event, she tripped and fell over an audio-visual box owned and operated by KSDK. They claimed that this box was placed on the ground without barricades or warnings in a high pedestrian traffic area, leading to her injuries.

The text provides context about how KSDK was involved as an event sponsor and had employees (Lynn Beall and Michael Shively) who were responsible for arranging live coverage of the event.
x??

---

#### Circuit Court’s Ruling

Background context explaining that the circuit court ruled on the plaintiffs’ claims based on the language of the release.

:p What did the circuit court rule regarding the summary judgment?
??x
The circuit court entered summary judgment in favor of defendants, ruling that plaintiffs' claims were barred by the language of the release. The court found that the release was not ambiguous and applied to KSDK and all other relevant parties involved with the event.

The text provides details about how the release described various entities that could be released from liability, and why the use of "any" in this context was unambiguous.
x??

---

---
#### Joan's Lease Contract
Background context: Joan, at 16 years old, signs a one-year lease for an apartment. Her parents inform her she can return to live with them at any time. Later, unable to pay rent, she moves back home after two months.
:p Can Kenwood enforce the lease against Joan?
??x
Kenwood may not be able to enforce the lease due to Joan's age and lack of contractual capacity. In most jurisdictions, individuals under 18 are considered minors and their legal contracts are subject to additional scrutiny. While a minor can enter into certain "necessaries" (such as food, clothing, shelter), leases are typically not considered necessities. Moreover, the parents' statement that Joan could return home if needed might indicate an intent to override her lease obligation.
x??

---
#### Sun Airlines Liability Clause
Background context: Sun Airlines includes a clause in its tickets stating it is not liable for any injury caused by the airline's negligence. If the cause of an accident is found to be the airline’s negligence, can Sun Airlines use this clause as a defense?
:p Can Sun Airlines use the liability clause as a defense?
??x
No, Sun Airlines cannot use the clause as a defense if the cause of the accident was its own negligence. This is because the clause would likely be considered void for illegality under most jurisdictions' laws. The airline's business activities must comply with relevant legal requirements, and attempting to absolve itself of all liability could be seen as engaging in unconscionable conduct or fraud.
x??

---
#### Covenants Not to Compete
Background context: Hotel Lux contracts with Chef Perlee for a one-year term at $30,000 per month. The contract includes a covenant not to compete clause that restricts Perlee from working as a chef in New York, New Jersey, or Pennsylvania for a year after leaving the hotel.
:p Discuss how successful Hotel Lux will be in seeking to enjoin Perlee.
??x
Hotel Lux might face challenges in enjoining Perlee. The enforceability of non-compete clauses is heavily dependent on local laws and the specific wording of the clause. Generally, such clauses must be reasonable in scope and duration; they cannot unduly restrict a former employee's ability to work in their chosen field.

In this case, the geographic restriction (New York, New Jersey, or Pennsylvania) may be seen as too broad, making it difficult for Hotel Lux to prove that Perlee’s new employment poses an unfair competitive threat. Additionally, the timing of Perlee’s move and the fact that he was hired by a restaurant just across the state line might weaken Hotel Lux's case.
x??

---
#### Intoxication and Contract Formation
Background context: Kira sold a diamond necklace to Charlotte for $100 after having several drinks. The next day, Kira offered$100 back to Charlotte, claiming she could void the contract due to intoxication at the time of formation.
:p Was Kira correct in her claim that the contract is voidable?
??x
Yes, Kira's contract with Charlotte is likely voidable due to her intoxication. Intoxication can render a person incapable of entering into a valid contract. In many jurisdictions, courts recognize that individuals under the influence of alcohol or drugs lack the capacity to form a binding agreement. Therefore, Kira has the option to rescind the contract if she was intoxicated at the time of its formation.
x??

---
#### Mental Competence and Arbitration
Background context: Dorothy Drury, suffering from dementia, signed an arbitration clause as part of her residency agreement when admitted to an assisted living facility managed by Assisted Living Concepts. After sustaining injuries in a fall, the facility sought to compel arbitration under the agreement.
:p Was Dorothy bound to the residency agreement?
??x
Dorothy may not have been fully capable of understanding or entering into the residency agreement due to her mental incompetence from dementia and chronic confusion. Courts generally require that parties to a contract be competent to understand the nature and consequences of their actions. Given her condition, it is possible that Dorothy's assent was not valid, thus potentially rendering the arbitration clause unenforceable.
x??

---

#### Release Validity and Liability
Background context: Sue Ann Apolinar hired a guide through Arkansas Valley Adventures, LLC for a rafting excursion. She signed a release that detailed potential hazards and risks, including overturning, unpredictable currents, obstacles, and drowning. The release clearly stated her signature discharged the outfitter from liability for all claims arising in connection with the trip.

:p What are the arguments for and against enforcing the release that Apolinar signed?
??x
The arguments for enforcing the release include:
- Apolinar voluntarily signed a detailed release before the excursion, indicating an awareness of potential risks.
- The language of the release is clear and unambiguous, discharging liability in exchange for her participation.

On the other hand, the arguments against enforcing the release are:
- Apolinar may not have fully understood or appreciated the risks involved due to her lack of rafting experience.
- Arkansas Valley Adventures, LLC had a duty to ensure the safety of its participants, which was breached when the current swept her into a logjam.

Enforcing such releases is often debated as it can protect businesses from liability but may also be seen as an attempt to shift responsibility unfairly onto the participant. 
??x
The arguments for enforcing the release are based on Apolinar's voluntary and informed consent, as evidenced by her signature. The release details specific risks, and signing it indicates acceptance of these hazards.

On the other hand, some might argue that Apolinar may not have fully understood or appreciated the risks involved due to her lack of experience in rafting. Furthermore, Arkansas Valley Adventures LLC had a duty to ensure the safety of its participants, which was breached when the current swept her into a logjam.
??x
```java
// Pseudocode for logic behind release enforcement
if (participant_signed_release && understands_risks) {
    release_enforced = true;
} else if (participant_lacks_experience && provider_breached_duty_of_care) {
    release_enforced = false;
}
```
x??

---

#### Disaffirmance of Minor’s Deed
Background context: Bonney McWilliam’s sixteen-year-old daughter, Mechelle, signed a deed transferring her one-half interest in a house to Bonney. Mechelle was described as an "emotionally troubled teenager" with a history of substance abuse and a fractured relationship with her mother.

:p Could the transfer of Mechelle's interest be disaffirmed?
??x
The transfer of Mechelle’s interest could potentially be disaffirmed due to her minority status and lack of full capacity at the time of signing. Minors often lack the legal capacity to enter into binding contracts, including deeds, due to their inability to fully understand the implications.

Given that Mechelle was described as "an emotionally troubled teenager" with a history of substance abuse, it is reasonable to argue that she may not have had the mental capacity or understanding required to make such a significant decision.
??x
The transfer could be disaffirmed because Mechelle was a minor and lacked full legal capacity at the time of signing. Minors often do not fully understand the implications of such decisions due to their age and emotional state, making it possible for them to disavow the agreement.

Since Mechelle’s behavior indicates she may have been under significant emotional stress or influence, her actions could be considered voidable.
??x
```java
// Pseudocode for disaffirmance logic
if (minor_status && lack_of_understanding) {
    transfer_disaffirmed = true;
} else {
    transfer_disaffirmed = false;
}
```
x??

---

#### Enforceability of Surrogacy Contract
Background context: P.M. and C.M., a married couple, contracted with T.B. for surrogacy services in Iowa. The contract was to have T.B. carry the pregnancy to term and hand over the baby at birth to the Ms. However, during the pregnancy, relations between the parties deteriorated, and T.B. refused to honor the agreement.

:p Is the surrogacy contract between the Ms and the Bs enforceable?
??x
The enforceability of the surrogacy contract between P.M., C.M., and T.B. is a complex issue. Iowa’s criminal statute prohibits selling babies but exempts surrogacy, creating a gray area in legal interpretation.

On one hand, the contract was signed with consideration (money and medical expenses) from both sides, suggesting mutual agreement.
- The Ms provided funds and sperm for fertilization.
- T.B. agreed to carry the pregnancy and hand over the baby at birth.

However, Iowa’s criminal statute prohibiting selling babies may create a legal barrier to enforcement of such contracts, despite the exemption for surrogacy. This could imply that while the contract might be valid in terms of mutual agreement and consideration, it might not be enforceable due to broader legal restrictions.
??x
The surrogacy contract between P.M., C.M., and T.B. is enforceable based on mutual agreement and consideration, but its enforceability may be constrained by Iowa’s criminal statute that prohibits selling babies, even if the statute exempts surrogacy.

Despite the exemption for surrogacy, the legal framework could still view such agreements as potentially unethical or in conflict with broader public policy.
??x
```java
// Pseudocode for contract enforceability logic
if (mutual_agreement && consideration) {
    contract_enforceable = true;
} else if (statute_prohibits_selling_babies) {
    contract_enforceable = false;
}
```
x??

---

#### Injury to Minor at Trampoline Park
Background context: Jacob Blackwell, a minor, suffered a torn tendon and a broken tibia during a dodgeball tournament at Sky High Sports Nashville Operations LLC’s trampoline park in Nashville, Tennessee.

:p What ethical considerations arise when dealing with injuries to minors?
??x
Ethical considerations in this scenario include the duty of care owed by the operators of the trampoline park to ensure the safety and well-being of participants, especially minors. Additionally, there are questions about informed consent, risk management, and the responsibility for providing medical care.

Key ethical issues:
- **Duty of Care**: Sky High Sports Nashville Operations LLC has a duty to provide a safe environment for all participants.
- **Informed Consent**: The operators should ensure that all participants, including minors, understand the risks involved.
- **Risk Management**: Proper risk management measures should be in place to prevent injuries, especially those that can have significant long-term impacts on a minor’s health.

Ethically, it is crucial to balance the business interests of the park with the safety and well-being of its patrons, particularly minors who are more vulnerable to injury.
??x
Ethical considerations include:
- **Duty of Care**: Sky High Sports Nashville Operations LLC must ensure that all participants are in a safe environment.
- **Informed Consent**: Minors should be informed about potential risks before participating.
- **Risk Management**: The park must have adequate risk management measures to prevent serious injuries.

These ethical concerns highlight the need for operators to prioritize safety and well-being, especially when dealing with minors.
??x
```java
// Pseudocode for ethics in injury prevention
if (safe_environment && informed_consents_given) {
    duty_of_care_met = true;
} else if (risk_management_in_place) {
    risk_management_effective = true;
}
```
x??

---

#### Unilateral Mistakes of Fact
Background context: A unilateral mistake is made by only one of the contracting parties. Generally, a unilateral mistake does not give the mistaken party any right to relief from the contract. Normally, the contract is enforceable unless it involves material facts that were mistakenly omitted.
:p What is a unilateral mistake in the context of contracts?
??x
A unilateral mistake occurs when only one party makes an error regarding a fact, and this error significantly impacts their decision to enter into the contract. The other party must not be aware of or should not have known about this mistake for it to affect enforceability.
x??

---

#### Unilateral Mistakes in Contract Example
Background context: The example provided involves Elena selling her jet ski for $2,500 but mistakenly typing$1,500 into the email. This is a unilateral mistake that can impact the enforceability of the contract if certain conditions are met.
:p What is an example scenario where a unilateral mistake might occur?
??x
Elena intends to sell her jet ski for $2,500. She sends Chin an email offering to sell it for$1,500 due to a typing error. If Elena intended to sell at the higher price and this was a material fact that influenced the decision to enter into the contract.
x??

---

#### Contract Enforcement Despite Unilateral Mistakes
Background context: Even with a unilateral mistake, the general rule is that the contract is still enforceable if the other party (Chin) did not know or should not have known about the error. However, exceptions exist under certain conditions.
:p How might a unilateral mistake affect the enforceability of a contract?
??x
A unilateral mistake can affect the enforceability of a contract if the other party knew or should have known about the mistake, or if there was a significant mathematical error made inadvertently and without gross negligence. In such cases, the contract may be rescinded.
x??

---

#### Bilateral (Mutual) Mistakes of Fact
Background context: A bilateral mistake is when both parties make an error regarding a basic assumption underlying the contract. This mutual misunderstanding can lead to avoidance of the contract under certain conditions.
:p What defines a bilateral (mutual) mistake in contracts?
??x
A bilateral mistake occurs when both parties have made an error concerning a fundamental assumption on which the contract was based. This mutual misunderstanding can potentially void the contract if it involves material facts and is significant enough to influence the decision-making process of either party.
x??

---

#### Exceptions to General Rule for Mistakes in Contract Law
Background context: There are at least two exceptions to the general rule that a unilateral mistake does not affect the enforceability of a contract. These include when the other party knows or should have known about the mistake, and when there is a substantial mathematical error made inadvertently.
:p What are the main exceptions to the general rule regarding mistakes in contracts?
??x
The main exceptions to the general rule for mistakes in contracts are:
1. The other party knew or should have known that a mistake was made.
2. There was a substantial mathematical error due to inadvertence and without gross negligence.

These exceptions can lead to the contract being unenforceable if they involve material facts.
x??

---

#### Material Fact in Mistake of Contract
Background context: For a mistake to be relevant, it must involve some material fact that significantly influences the decision-making process. This means the error must be something crucial and not trivial.
:p What does "material fact" mean in the context of mistakes in contracts?
??x
A material fact is one that would influence a reasonable person's decision to enter into a contract. The mistake must be significant enough to affect the party's decision-making process, making it relevant to the validity or enforceability of the agreement.
x??

---

#### Lack of Voluntary Consent and Contractual Mistakes
Background context: A lack of voluntary consent (assent) can make a contract unenforceable if it is due to mistakes, misrepresentations, undue influence, or duress. This means that the parties did not genuinely agree to the terms.
:p How does a lack of voluntary consent impact the enforceability of a contract?
??x
A lack of voluntary consent (assent) makes a contract unenforceable if it is due to factors such as mistakes, misrepresentations, undue influence, or duress. This means that one party did not truly agree to the terms of the contract, indicating there was no true "meeting of the minds."
x??

---

