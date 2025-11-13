# Flashcards: Business-Law_-Text-and-Cases_processed (Part 119)

**Starting Chapter:** Terms and Concepts. Issue Spotters. Chapter 28 Banking

---

---
#### Signature Liability and UCC Provisions
Background context: The Uniform Commercial Code (UCC) has specific provisions regarding signature liability on negotiable instruments. In this scenario, the checks used by an employee named Mahar to embezzle funds from Golden Years indicate potential violations of these provisions.

:p Which provision of the UCC applies to the situation where signature stamps are used to commit fraud and result in embezzlement?
??x
The relevant provision is Section 3-402 of the UCC, which deals with the transferability of signatures on instruments. This section outlines who may sign an instrument and under what circumstances.

Explanation: Section 3-402 sets forth requirements for signatures or other authenticating marks. It states that a signature must be genuine unless it is authorized by law or contract to use a facsimile, stamp, or other means of authentication. In this scenario, the unauthorized use of signature stamps to fraudulently transfer funds violates UCC rules.

```java
public class SignatureProvision {
    public boolean isAuthorizedSignature(String signature) {
        // Check if the provided signature meets legal and contractual requirements
        return isValidSignature(signature);
    }
}
```
x??

---
#### Transfer or Presentment Warranties Violation
Background context: Transfer warranties ensure that a transferor delivers an instrument to a transferee free from defenses. Presentment warranties require the transferor to present the instrument for payment in good faith and with due diligence.

:p Describe any transfer or presentment warranties that Mahar may have violated.
??x
Mahar, as the drawer of the checks, breached his transfer warranties by:

1. **Transfer Warranties**: Mahar did not ensure that the checks he transferred were free from defenses. He knowingly used unauthorized signatures to embezzle funds, thereby transferring instruments with potential fraudulent claims.

2. **Presentment Warranties**: Mahar failed to present the checks for payment in good faith and due diligence, as his actions were fraudulent and intended to defraud Golden Years.

Explanation: Under UCC Section 3-417 (transfer warranties) and Section 3-408 (presentment warranties), a transferor must ensure that an instrument is free from defenses and must present it for payment with good faith. Mahar's actions violated both of these provisions by using unauthorized signatures and attempting to defraud the company.

```java
public class WarrantiesViolation {
    public boolean isViolated(String signature, String intent) {
        // Check if the signature is authorized and the intent is fraudulent
        return !isValidSignature(signature) && intent.equals("fraudulent");
    }
}
```
x??

---
#### Loss Bearer in UCC Context
Background context: Under the Uniform Commercial Code (UCC), the party that must bear a loss when an instrument is fraudulently endorsed or used can vary based on the circumstances and the nature of the transaction.

:p Which party, Golden Years or Star Bank, must bear the loss in this situation? Why?
??x
Golden Years must bear the loss. According to UCC Section 3-419, a holder in due course (HDC) obtains rights free from defenses that exist against the prior holders of an instrument. Star Bank, as an HDC when it accepted and deposited the checks, has no recourse against Golden Years for fraud perpetrated by employees who signed the checks.

Explanation: Since Star Bank processed the checks as legitimate transactions without knowledge of the fraudulent activity, it is considered a holder in due course (HDC). An HDC can only be subjected to defenses that existed at the time of transfer. The unauthorized signature and subsequent fraud are not defenses against an HDC. Thus, Golden Years remain responsible for the loss.

```java
public class LossBearingParty {
    public String determineLossBearer(String bankStatus) {
        if (bankStatus.equals("holder in due course")) {
            return "Golden Years";
        } else {
            return "Star Bank";
        }
    }
}
```
x??

---
#### Rye's Forgery and Suchin Corporation
Background context: The scenario involves an employee, Rye, who forged a check on behalf of Suchin Corporation to benefit himself or another party without authorization.

:p Does Suchin have any recourse against the bank for the payment? Why?
??x
Suchin does not have any recourse against Viceroy Bank for the payment. According to UCC Section 3-417, a transferor (Suchin) is only liable if it knows or has reason to know that an instrument transferred is fraudulent.

Explanation: In this case, Rye forged Suchin's signature and cashed a check at Viceroy Bank without Suchin’s knowledge or consent. Since Suchin was unaware of the forgery and had no reasonable basis to suspect fraud, it cannot hold the bank responsible for honoring the fraudulent check. The bank acted in good faith when processing the check.

```java
public class ForgeryRecourse {
    public boolean hasRecourse(String employeeSignature) {
        // Check if Suchin was aware or reasonably suspected of forgery
        return !SuchinWasAware(employeeSignature);
    }
}
```
x??

---
#### Skye's Blatant Forged Check and Discharge
Background context: The case involves a scenario where Jim, acting on behalf of Skye, forged her check to withdraw more money than intended.

:p Was the bookstore a holder in due course on Skye’s check? Why or why not?
??x
The bookstore was not a holder in due course (HDC) because it did not act in good faith and did not follow due diligence when accepting the check. According to UCC Section 3-419, an HDC must have taken the instrument for value in good faith without notice of any defense.

Explanation: The clerk at the bookstore accepted a $200 check with only$100 intended by Skye. This suggests that the clerk did not verify the amount and acted negligently or fraudulently, thereby losing the protection of an HDC status. Had the clerk verified the check more thoroughly or had any suspicion about its authenticity, the bookstore would have had notice of a defense.

```java
public class DischargeCheck {
    public boolean isHolderInDueCourse(String checkAmount, String intendedAmount) {
        // Check if the clerk acted in good faith and with due diligence
        return !checkAmount.equals(intendedAmount);
    }
}
```
x??

---
#### Material Alteration Defense
Background context: Williams used a pencil to write a $1,000 check to Stein for a used car. Later, Stein altered the amount to$10,000 and negotiated it to Boz.

:p Can Williams successfully raise the universal (real) defense of material alteration to avoid payment on the check? Explain.
??x
Williams can indeed raise the universal (real) defense of material alteration to avoid payment on the check. According to UCC Section 3-409, a holder in due course (HDC) is protected from defenses that existed at the time of transfer unless it acquired its instrument with knowledge of such a defense.

Explanation: Boz took the altered check as an HDC but failed to notice the alteration since he acted in good faith. However, Williams had clear evidence of the material alteration and could prove it before the transaction. Thus, Williams can claim that Boz's failure to detect the forgery means that Williams is not liable for payment.

```java
public class MaterialAlteration {
    public boolean raiseDefense(String checkAmount, String alteredAmount) {
        // Check if there was a material alteration and if it was known at the time of transfer
        return !checkAmount.equals(alteredAmount);
    }
}
```
x??

---
#### Signature Liability with Indorsement
Background context: The scenario involves Grace indorsing a negotiable promissory note to Adam, who then negotiated it to Keith. Waldo filed for bankruptcy and cannot be held liable.

:p Discuss whether Keith can hold Waldo, Grace, or Adam liable on the note.
??x
Keith can only hold Waldo liable on the note since he has filed for bankruptcy. According to UCC Section 3-405, if a debtor is discharged in bankruptcy, all debts of that debtor are discharged, including any negotiable instruments issued by them.

Explanation: Waldo's discharge in bankruptcy means that any debt, such as the promissory note, is nullified. Therefore, Keith cannot hold Waldo liable for payment. As Grace indorsed and Adam negotiated the note, they would still be responsible if Waldo were not bankrupt or if there was no bankruptcy protection.

```java
public class NoteLiability {
    public String determineLiableParty(String debtorStatus) {
        // Check debtor status to determine liability
        return debtorStatus.equals("bankrupt") ? "Waldo" : "Not liable";
    }
}
```
x??

---

---
#### Creditor-Debtor Relationship
A creditor-debtor relationship is established when a customer makes cash deposits into a checking account. In this relationship, the customer becomes a creditor for the amount deposited, and the bank acts as a debtor.

:p How does a creditor-debtor relationship work in a banking context?
??x
In a banking context, when a customer deposits money into their checking account, they become a creditor to the bank because they are providing funds that the bank can use. In turn, the bank becomes a debtor for this amount, as it has an obligation to return these funds upon request or to make payments on behalf of the customer.
```java
public class BankCustomerRelationship {
    public void deposit(double amount) {
        // Code to update customer's balance and bank's debt
    }
}
```
x??

---
#### Agency Relationship in Banking
An agency relationship arises when a customer writes a check, instructing the bank to make a payment. The bank acts as the agent for the customer, obligated to honor the request.

:p What is an example of an agency relationship between a customer and a bank?
??x
When a customer writes a check, they are ordering the bank to pay a specified amount to the holder when presented with the check. In this scenario, the bank acts as the agent for the customer, obligated to follow through on this request.

```java
public class BankAgency {
    public void processCheck(double amount) {
        // Code to honor the check and pay the specified amount
    }
}
```
x??

---
#### Contractual Relationship in Banking
Contractual relationships arise when a bank-customer relationship is established. These rights and duties depend on the nature of the transaction.

:p What are some examples of contractual obligations between a customer and a bank?
??x
Examples include depositing funds (customer becomes a creditor, bank becomes a debtor), writing checks (bank acts as an agent to pay specified amounts), and making withdrawals or payments. The specific terms and conditions outlined in their agreement define these rights and duties.

```java
public class BankContract {
    public void handleTransaction(String transactionType) {
        // Code to manage different types of transactions based on contract terms
    }
}
```
x??

---
#### Case in Point 28.3: Royal Arcanum Hospital Association of Kings County, Inc.
This case involves a dispute over unauthorized withdrawals from a corporate account where the bank was not held liable because the account terms did not include a two-signature requirement.

:p What did the court rule regarding Capital One Bank's liability in this case?
??x
The court ruled that Capital One Bank was not liable for the payment of unauthorized withdrawals on Royal Arcanum Hospital Association of Kings County, Inc.'s corporate accounts. The reason was that the contract terms never included a two-signature requirement for transactions.

```java
public class Case28_3 {
    public void determineLiability() {
        if (accountTermsIncludeTwoSignatures()) {
            // Bank would be liable
        } else {
            // Bank not liable as per the court decision
        }
    }

    private boolean accountTermsIncludeTwoSignatures() {
        return false; // Based on the case details provided
    }
}
```
x??

---

#### Good Faith Obligations in Contractual Discretion

Background context: The case discusses a legal dispute involving West Bank and its customers, the Leggs. The court ruled that the plaintiffs could pursue their claim based on a potential breach of the express duty of good faith in the sequencing of bank card transactions.

:p How did other courts' decisions influence the lower court's ruling regarding the Leggs' claim?
??x
Other courts had held that when one party is given discretion to act under a contract, said discretion must be exercised in good faith. The lower court cited these cases, which found that banks have a duty of good faith when exercising their discretion to sequence transactions.

```java
// Pseudocode for Good Faith Discretion
public class BankTransactionSequencing {
    private void applyGoodFaithDiscretion(Customer customer) {
        if (customer.getAgreement().includesGoodFaithClause()) {
            // Logic to ensure sequencing is done in good faith
        } else {
            // Default logic without good faith clause
        }
    }
}
```
x??

---

#### Notice Adequacy of Sequencing Change

Background context: The lower court considered whether West Bank provided adequate notice to its customers regarding a change in the sequencing of bank card transactions. The document titled "Miscellaneous Fees" mentioned that checks would be paid in order daily.

:p Was the footnote in the "Miscellaneous Fees" document adequate notice to the Leggs about the change in transaction sequencing?
??x
The footnote in the "Miscellaneous Fees" document may not have been adequate notice because it did not specifically inform customers of a change in the bank's discretionary power regarding transaction sequencing. The Leggs reasonably expected consistent behavior from the bank, and the lack of notification could lead to a breach of good faith.

```java
// Pseudocode for Notice Adequacy Check
public class NoticeCheck {
    public boolean isNoticeAdequate(String noticeText) {
        return noticeText.contains("Sequencing order changed") || 
               noticeText.contains("Discretionary changes affecting customer transactions");
    }
}
```
x??

---

#### Impact of Absence of Good Faith Clause

Background context: The case highlighted the importance of an explicit good faith clause in contract agreements. Without such a clause, the bank's discretion to sequence transactions might be interpreted differently.

:p How would the outcome have been different if West Bank’s "Deposit Account Agreement" did not include an obligation for the bank to exercise good faith?
??x
Without an explicit good faith clause, the bank's duty to act in good faith when exercising its discretion regarding transaction sequencing may not have been as clearly defined. This could potentially weaken the Leggs' argument that the change in sequencing violated their reasonable expectations and breached a fiduciary obligation.

```java
// Pseudocode for Good Faith Clause Impact
public class AccountAgreement {
    public boolean includesGoodFaithClause() {
        return agreementText.contains("exercise good faith");
    }
}
```
x??

---

---
#### Kenneth Wulf Case Background
Auto-Owners Insurance Company employee, Kenneth Wulf, stole $546,000 worth of checks and deposited them into his account at Bank One using a forged "Auto-Owners Insurance Deposit Only" stamp. When discovered, Auto-Owners sued Bank One for negligence.
:p What was the outcome of the lawsuit between Auto-Owners and Bank One?
??x
The court ruled in favor of the bank, finding that Bank One's conduct was not a significant factor in bringing about the loss. Instead, Auto-Owners' own negligence contributed substantially to its losses.
x??

---
#### Timely Examination of Bank Statements Required
Banks typically send monthly statements detailing account activity and provide customers with information (check number, amount, date) to identify each check paid by the bank [UCC 4–406(a), (b)]. Banks are required to retain canceled checks or legible images for seven years [UCC 4–406(b)].
:p What is the duty of a customer in relation to their bank statements and checks?
??x
Customers have a duty to promptly examine bank statements with reasonable care and report any alterations or forged signatures [UCC 4–406(c)]. They also need to notify the bank within thirty days if they discover unauthorized transactions or discrepancies.
x??

---
#### Consequences of Failure to Detect Forgeries
If a customer fails to detect forgeries, the bank’s liability is limited. The customer must report any first forged check within thirty calendar days [UCC 4–406(d)(2)]. If not reported in time, the bank is discharged from liability for all previously paid forged checks.
:p What happens if a customer does not notify the bank about unauthorized transactions within thirty days?
??x
If the customer fails to report the first forged check within thirty days, the bank’s liability for all previously paid forged checks prior to notification is discharged.
x??

---
#### Denise Kaplan Case Background
Denise Kaplan had two accounts with JPMorgan Chase Bank (JPMC). Her agreement required her to review statements and report discrepancies within thirty days. Joel Kaplan later added his name to the accounts, making unauthorized withdrawals when Denise could not access her monthly bank statements.
:p What issue did Denise discover in this case?
??x
Denise discovered that her husband, Joel Kaplan, had been making unauthorized withdrawals from their joint accounts after she was unable to access her monthly bank statements due to them being sent to his e-mail address.
x??

---

#### Forgery on Checks
Background context: The UCC (Uniform Commercial Code) governs the handling of checks, especially when there are issues like forgery or alterations. In this case, MBP issued a check with two forged indorsements, which is problematic under the UCC.

:p What was the key issue in the Michigan Basic Property Insurance Association v. Washington case regarding check forgery?
??x
The key issue was that Joyce Washington forged the indorsements of both Countrywide Home Loans and T&C Federal Credit Union on a check issued by MBP, making it a case involving two forged indorsements.

:x??

---

#### Indemnity Clause in Bank Accounts
Background context: The UCC includes provisions for when checks are improperly paid due to forgery. In some cases, an indemnity clause may shift the loss from the bank to the customer if certain conditions are met.

:p How did the state appellate court determine that Fifth Third Bank was not liable to MBP?
??x
The state appellate court found that Fifth Third Bank was not liable because there was a contract between the parties stipulating that if the bank was not promptly notified of forgeries, the loss would fall on the customer. Since MBP did not promptly notify the bank, they were responsible for any forged indorsements.

:x??

---

#### Altered Checks
Background context: A check is altered when it has been changed without authorization, such as by increasing the amount. The bank has a duty to examine checks before final payment and may be liable if alterations are undetected.

:p What happens if a customer writes a check that is later altered in favor of them?
??x
If a customer writes a check for $11 but it is later altered to $111, the customer’s account will be charged $11 (the amount ordered by the customer). The bank would normally be responsible for the remaining$100.

:p How does the UCC address negligence in altering checks?
??x
If a person carelessly writes a check and leaves large gaps where additional numbers or words can be inserted, they are barred from protesting when the bank unknowingly and in good faith pays whatever amount is shown. Additionally, if the bank can trace its loss on successive altered checks to the customer’s failure to discover an initial alteration, it may reduce liability for reimbursing the customer.

:x??

---

#### Commercial Standards of Care
Background context: Banks are required to observe reasonable commercial standards of care in paying on a customer's checks. This includes examining and validating the checks before payment.

:p What is the bank's responsibility when handling altered checks?
??x
The bank has an implicit duty to examine checks before making final payments. If it fails to detect an alteration, it can be liable to its customer for the loss because it did not pay as the customer ordered. The bank’s loss is the difference between the original amount of the check and the amount actually paid.

:x??

---

---
#### Depositary Bank Definition
Background context: The depositary bank is the first bank to receive a check for payment. This can be from a customer's personal account or when a customer deposits a check into their account at another bank.
:p What is a depositary bank?
??x
The depositary bank is the first bank that receives a check for payment, such as when an individual deposits a tax-refund check into a local bank. 
x??

---
#### Payor Bank Definition
Background context: The payor bank is the bank on which a check is drawn and is responsible for paying the face amount of the check unless it is dishonored.
:p What is a payor bank?
??x
The payor bank is the bank from which the money will be withdrawn when a check is presented. It is typically the bank where the drawer has an account, and it is responsible for honoring or dishonoring the check based on its records.
x??

---
#### Intermediary Bank Role
Background context: An intermediary bank handles checks during some phase of the collection process other than acting as the depositary or payor bank. The banking industry is evolving with Check 21, which affects how these roles are managed in the collection process.
:p What role does an intermediary bank play?
??x
An intermediary bank plays a role in handling checks during the collection process but neither acts as the depositary bank nor the payor bank. Its main task is to transfer the check from one bank to another, ensuring it reaches the appropriate payor bank.
x??

---
#### On-Us Item Definition and Provisional Credit
Background context: An on-us item is a check payable by the depositary bank that also serves as the payor bank. The banking system typically issues a provisional credit for such items within the same day, which becomes final after the second banking day if not dishonored.
:p What is an on-us item?
??x
An on-us item is a check that is payable by the depositary bank and also serves as the payor bank. The depositary bank usually provides a provisional credit for such items within the same day, which becomes final after the second banking day if not dishonored.
x??

---
#### Check Collection Process Between Different Banks
Background context: When checks are between different banks, the depositary bank must present the check to the payor bank through intermediary banks. The payor bank is liable for the face amount of the check unless it is dishonored.
:p How does a check collection process work between different banks?
??x
In a scenario where checks are between different banks, the depositary bank initiates the process by presenting the check to the appropriate payor bank either directly or through intermediary banks. The payor bank is then liable for the face amount of the check unless it is dishonored.
x??

---
#### Banking Day and Time Limits
Background context: Each bank in the collection chain must pass a check on before midnight of the next banking day following its receipt to ensure timely processing. A "banking day" refers to any part of a day when banks are open for business.
:p What is a banking day, and what time limits apply?
??x
A banking day is any part of a day when banks are open for business. Each bank in the collection chain must pass checks on by midnight of the next banking day following their receipt to ensure timely processing.
x??

---

