# Flashcards: Business-Law_-Text-and-Cases_processed (Part 125)

**Starting Chapter:** 31-4 Bankruptcy Relief under Chapter 12 and Chapter 13

---

---
#### Filing of Plan within 120 Days
Background context: The Bankruptcy Code requires that only the debtor may file a plan within the first 120 days after the order for relief. This period can be extended, but not beyond eighteen months from the date of the order for relief. If the debtor fails to meet this deadline or obtain an extension, any party may propose a plan.

:p What is the time frame for filing a bankruptcy plan by the debtor?
??x
The debtor must file a plan within 120 days after the order for relief. This period can be extended but not beyond eighteen months from the date of the order for relief.
x??

---
#### Small-Business Debtor's Filing Period
Background context: For small-business debtors, if they choose to avoid a creditors' committee, the filing time is 180 days instead of 120. This extended period is specific to small businesses.

:p How long can a small-business debtor file a bankruptcy plan?
??x
Small-business debtors have an extended filing period of 180 days to submit their bankruptcy plan.
x??

---
#### Acceptance of the Plan by Creditors
Background context: The plan must be accepted by each class of creditors. For acceptance, a majority of the creditors in the class, representing two-thirds of the total claim, must vote to approve it. If debtor's consent is not obtained within 180 days, any party may propose a plan.

:p What conditions must be met for the plan to be accepted?
??x
For the plan to be adopted, each class of creditors must accept it through a majority vote representing two-thirds of the total claim in that class. If the debtor fails to get creditor consent within 180 days, any party can propose a new plan.
x??

---
#### Confirmation of the Plan
Background context: The court must confirm the plan if all classes of creditors have accepted it and the debtor has paid all postpetition domestic-support obligations. However, the court may refuse confirmation if the plan is not in the best interests of the creditors. For small-business debtors, the court must confirm the plan within 45 days unless extended.

:p What does the court consider when confirming a bankruptcy plan?
??x
The court confirms a bankruptcy plan only if all classes of creditors have accepted it and the debtor has paid all postpetition domestic-support obligations. Additionally, even if all classes accept the plan, the court can refuse confirmation if it is not in the best interests of the creditors. For small-business debtors, the court must confirm the plan within 45 days (extendable).
x??

---
#### Modification of the Plan
Background context: The bankruptcy plan can be modified by the debtor, the DIP (Debtor-in-Possession), the trustee, the U.S. trustee, or a holder of an unsecured claim. If an unsecured creditor objects to the plan, specific rules apply regarding property value distribution.

:p Who can modify a bankruptcy plan?
??x
The bankruptcy plan can be modified by the debtor, the DIP (Debtor-in-Possession), the trustee, the U.S. trustee, or a holder of an unsecured claim.
x??

---
#### Cram-Down Provision for Confirmation
Background context: The court may confirm a plan over objections from creditors if it is fair and equitable and does not discriminate unfairly against any creditors. This provision allows the confirmation even if only one class of creditors has accepted the plan.

:p Under what circumstances can the court use cram-down to confirm a bankruptcy plan?
??x
The court can confirm a bankruptcy plan over objections if it demonstrates that the plan does not discriminate unfairly against any creditors and is fair and equitable.
x??

---
#### Discharge in Bankruptcy
Background context: For individual debtors, discharge only occurs after the completion of the plan. However, for other types of debtors, the court may order discharge at any time after the plan is confirmed. On discharge, the debtor receives a reorganization discharge from all claims not protected under the plan.

:p When can a bankruptcy debtor be discharged?
??x
For individual debtors, discharge only occurs after the completion of the bankruptcy plan. For other types of debtors, the court may order discharge at any time after the plan is confirmed.
x??

---

---
#### Repayment Plan Components
Background context: A Chapter 13 bankruptcy repayment plan must include specific elements to be valid. These components ensure that the debtor's obligations are met through a structured financial plan.
:p What are the main components required for a valid Chapter 13 repayment plan?
??x
The main components required for a valid Chapter 13 repayment plan include:
- The turning over of future earnings or income to the trustee as necessary.
- Full payment, via deferred cash payments, of all claims entitled to priority, such as taxes.
- Identical treatment of all claims within a particular class. Note that co-debtors can be listed as a separate class.

The plan may either provide for full repayment in full or a lesser amount. The debtor must begin making payments under the proposed plan within thirty days after filing and continue to make timely payments. Failure to do so will result in the court converting the case to Chapter 7 bankruptcy or dismissing the petition.
x??

---
#### Allowable Expenses
Background context: A debtor preparing a repayment plan must apply the means test to determine disposable income available for creditors' claims, while deducting only appropriate expenses.
:p How does a debtor calculate allowable expenses for their repayment plan?
??x
A debtor calculates allowable expenses by applying the means test. This process involves identifying the amount of disposable income that will be available to repay creditors after necessary living expenses are deducted from monthly income.

Example: 
- Monthly Income = $5,000
- Essential Living Expenses (ELE) = $2,000

Disposable Income = Monthly Income - ELE

The debtor can deduct appropriate expenses such as:
- Housing and utilities
- Transportation
- Health care
- Insurance
- Taxes
- Childcare
- Educational expenses

However, deductions for car ownership and operation must be relevant to the life of the Chapter 13 plan. For instance, if a debtor does not make loan or lease payments on a vehicle, they may not claim a car-ownership deduction.

Code Example:
```java
public class ExpensesCalculator {
    public double calculateDisposableIncome(double monthlyIncome, double essentialLivingExpenses) {
        return monthlyIncome - essentialLivingExpenses;
    }
}
```
x??

---
#### Length of the Plan
Background context: The duration of the repayment plan in a Chapter 13 bankruptcy can be either three or five years based on the debtor's family income. If the family income is higher than the median family income for the relevant geographic area, the term must be three years.
:p What determines the length of a repayment plan in a Chapter 13 bankruptcy?
??x
The length of a repayment plan in a Chapter 13 bankruptcy can be either three or five years based on the debtor's family income. If the family income is greater than the median family income for the relevant geographic area under the means test, the term of the proposed plan must be three years.

Example:
- Family Income = $80,000 (above median)
- Median Family Income for Geographic Area = $75,000

In this case, the repayment plan would need to last three years. If the family income is less than or equal to the median family income, the term can be five years.

Code Example:
```java
public class PlanDuration {
    public int getPlanTerm(double familyIncome, double medianFamilyIncome) {
        if (familyIncome > medianFamilyIncome) {
            return 3; // Three-year plan
        } else {
            return 5; // Five-year plan
        }
    }
}
```
x??

---

---
#### Tina's Bankruptcy Petition
Background context: Tina, after working as a salesperson and filing for bankruptcy, has debts that include student loans, taxes accruing within the last year, and a claim based on her misuse of customers' funds. These debts are subject to dischargeability in bankruptcy.
:p Are all these debts dischargeable in bankruptcy?
??x
In bankruptcy, student loans are typically not dischargeable unless there is a unique circumstance such as undue hardship, which Tina does not mention. Taxes accruing within the last year are generally dischargeable. However, the claim against Tina based on her misuse of customers’ funds during employment likely involves fraudulent activity, which can disqualify this debt from dischargeability.
??x
The answer focuses on the specific nature and context of each type of debt:

- **Student Loans**: Typically not dischargeable unless undue hardship is proven. In Tina’s case, there's no indication that undue hardship applies.
- **Taxes Accruing Within the Last Year**: Generally dischargeable in Chapter 7 bankruptcy.
- **Misuse of Customers’ Funds**: Likely involves fraud or misuse of funds, which can disqualify this debt from dischargeability.

```java
public class BankruptcyCase {
    public boolean isDebtDischargeable(String typeOfDebt) {
        switch (typeOfDebt) {
            case "student loans":
                return false; // unless undue hardship is proven
            case "taxes within the last year":
                return true;
            case "misuse of customers’ funds":
                return false; // likely involves fraud
            default:
                throw new IllegalArgumentException("Unknown debt type");
        }
    }
}
```
x??
---

#### Ogden's Loan Recovery by Quentin
Background context: Ogden, a vice president at Plumbing Service Inc. (PSI), loans the company $10,000 on May 1st, which is repaid on June 1st. The firm files for bankruptcy on July 1st, and Quentin is appointed as trustee.
:p Can Quentin recover the $10,000 paid to Ogden on June 1st?
??x
The claim that Ogden received repayment of his loan before PSI filed for bankruptcy does not generally allow for recovery by the trustee. The payment was made after the petition had been filed and thus is considered a preferential transfer or an invalid post-petition transaction, which typically cannot be recovered.
??x
Explanation: Under U.S. bankruptcy law, transactions that are post-petition (i.e., occurring after the filing of the bankruptcy) are generally not subject to recovery by the trustee, unless they fit specific criteria such as being made within a preferential period or being an improper post-petition transfer.

```java
public class BankruptcyTransaction {
    public boolean canRecoverPostPetitionPayment(Date paymentDate, Date bankruptcyFilingDate) {
        return paymentDate.after(bankruptcyFilingDate);
    }
}
```
x??
---

#### Burke's Voluntary versus Involuntary Bankruptcy
Background context: Burke is a rancher with an $500,000 property and$70,000 in debt. She cannot pay off her creditors due to a drought that has ruined her crops and cattle.
:p Can Burke voluntarily petition herself into bankruptcy? Explain.
??x
Burke can voluntarily petition for bankruptcy under the Bankruptcy Code. The fact that she has an exemption of almost all property value under state law (with only $70,000 in debt) makes a voluntary petition feasible as it would allow her to protect her exempt property while discharging her debts.
??x
Explanation: Under U.S. bankruptcy laws, individuals can voluntarily file for Chapter 7 or Chapter 13 bankruptcy. Since Burke’s total indebtedness ($70,000) is significantly less than the value of her property ($500,000), and most of it is exempt under state law, a voluntary petition would allow her to discharge her debts without liquidating her entire property.

```java
public class BankruptcyPetition {
    public boolean canVoluntarilyFile(String debtor, int totalDebt, int propertyValue, String[] exemptProperties) {
        // Assume all properties are not subject to automatic exemption rules and need manual check
        return true; // Since the debt is less than the property value and most of it is exempt
    }
}
```
x??
---

#### Distribution of Property in Montoro's Bankruptcy Case
Background context: Montoro, who petitioned himself into voluntary bankruptcy, has three major claims against his estate. These include a friend holding a $2,500 note, an employee claiming back wages, and a loan from the United Bank of the Rockies.
:p What amount will each party receive in Montoro's distribution?
??x
The proceeds from liquidating Montoro’s nonexempt property ($5,000) need to be distributed among the claims. The priority is as follows:
1. **Carlton**: $2,500 (full payment on his note)
2. **Elmer**: $2,000 (back wages claim, up to 90% of the proceeds available after creditor payments)
3. **United Bank**: $500 (remaining amount distributed pro rata or as much as possible)

Since Elmer’s claim is for back wages and typically has a higher priority than unsecured loans in bankruptcy distributions.
??x
Explanation: 
1. **Carlton's $2,500 note** gets paid first entirely because it is a secured claim with a specific amount due.
2. **Elmer**: As an employee, his claim for back wages usually ranks above other unsecured claims and will get priority payment up to 90% of the available funds ($4,500).
3. **United Bank’s $5,000 loan** is last in line after all other secured and higher-priority unsecured claims have been paid.

```java
public class DistributionOfProceeds {
    public void distributeProceeds(int totalProceeds, Map<String, Integer> claims) {
        int remainingProceeds = totalProceeds;
        for (Map.Entry<String, Integer> entry : claims.entrySet()) {
            String claimant = entry.getKey();
            int claimAmount = entry.getValue();
            
            if (claimAmount <= remainingProceeds) { // Full payment
                System.out.println("Paid " + claimAmount + " to " + claimant);
                remainingProceeds -= claimAmount;
            } else { // Pro rata or limited distribution
                System.out.println("Paid " + remainingProceeds + " to " + claimant);
                break; // Stop if there's no more funds left
            }
        }
    }
}
```
x??
---

#### Discharge in Bankruptcy for Harman's Guaranty
Background context: Caroline McAfee loaned $400,000 to Carter Oaks Crossing. Joseph Harman signed a personal guaranty for this note and defaulted on it when the company failed to repay.
:p Would the obligation under the guaranty have been discharged in bankruptcy?
??x
The obligation under the guaranty was not likely discharged in bankruptcy because the guaranty itself was not listed among Harman’s debts in his bankruptcy filing. Bankruptcy discharge applies only to claims explicitly listed or properly disclosed by a debtor.
??x
Explanation: 
- **Dischargeability**: For an obligation to be discharged, it must be included in the bankruptcy schedule. If Harman did not list this guaranty as one of his debts, the claim remains valid and enforceable against him after the bankruptcy.
- **Personal Guaranty**: Personal guarantees are separate legal agreements that remain binding even if the principal debtor files for bankruptcy.

```java
public class BankruptcyDischarge {
    public boolean isGuarantyDischarged(String guarantor, String guarantyDebt) {
        // Check if the guaranty was listed in the bankruptcy schedule
        return false; // If not, it remains binding
    }
}
```
x??

---
#### Discharge under Chapter 13
Background context: In re Shankle, a case involving James Thomas and Jennifer Clark, who were married with two children. They purchased a home with a mortgage and later took out a second mortgage. After their divorce, the court gave custody of the children to Clark and required her to pay the first mortgage. Additionally, they agreed that both would make equal payments on the second mortgage, with Clark receiving all proceeds from the sale of the home.

Clark sold the house but learned Auto Now had a lien due to Thomas's failure to make car payments. She used the proceeds to settle the liens and mortgages. Thomas later filed for Chapter 13 bankruptcy.

:p Can Jennifer Clark prevent James Thomas from discharging his debt obligations related to the mortgage and second mortgage?
??x
Clark can argue that these debts are non-dischargeable because they were part of domestic support obligations. However, courts typically distinguish between direct support (child support) and indirect support (such as joint financial responsibilities). In this case, while there is a shared responsibility for the mortgages, it's unclear if these debts fall under domestic support.

In the bankruptcy context, 11 U.S.C. § 523(a)(15) does not automatically make these obligations non-dischargeable unless they are explicitly categorized as such by the court or statute.

Clark might need to show that Thomas's obligation was integral to the family structure and directly related to their support duties, which could be a complex legal issue.
x??

---
#### Liquidation Proceedings: Dismissal of Bankruptcy Petition
Background context: Jeffrey Krueger and Michael Torres were shareholders in Cru Energy, Inc. They were involved in litigation over control of the company. Krueger formed Kru with similar business plans to delay proceedings by filing a Chapter 7 bankruptcy petition without disclosing his interest in Kru.

Krueger called an unauthorized shareholder meeting and voted shares he did not have the authority to vote, removing Torres from the board and electing himself chairman, president, CEO, and treasurer. He also dismissed all of Cru's claims against him.

:p Can the bankruptcy court dismiss Krueger’s petition for bankruptcy?
??x
The bankruptcy court likely has grounds to dismiss Krueger's petition based on his misconduct and lack of disclosure. Under 11 U.S.C. § 707(b), a bankruptcy case may be dismissed if the debtor has engaged in fraud, abuse, or concealment of assets.

Krueger's actions, such as forming Kru with an apparent intent to circumvent Cru Energy, can be considered fraudulent or abusive behavior towards the court and other shareholders. The bankruptcy court would likely find that these actions constitute a violation of 11 U.S.C. § 707(b) and justify dismissing his petition.

Code example: 
```java
if (bankruptcyPetitioner.hasFraudulentIntent() ||
    bankruptcyPetitioner.hasAbusedTheProcess()) {
    // Dismiss the case
}
```
x??

---
#### Reorganization Plan: Confirmation Requirement
Background context: In re Krueger, T ranswest Resort Properties, Inc., and four related companies filed for Chapter 11 bankruptcy. They proposed a joint reorganization plan that was approved by several creditor classes but Grasslawn Lodging objected based on the "per debtor" vs. "per plan" interpretation of the Bankruptcy Code.

Grasslawn argued that the confirmation requirement applied per debtor, meaning at least one class member for each debtor must accept the plan. Since Grasslawn was the only objecting party and the sole creditor for two of the debtors, they concluded the plan did not meet the test for confirmation.

:p Does the joint reorganization plan in Transwest Resort Properties, Inc., meet the legal requirement for confirmation?
??x
The legality of the joint reorganization plan depends on the interpretation of the "per debtor" vs. "per plan" confirmation requirement under 11 U.S.C. § 1129(a)(10).

If interpreted as "per debtor," at least one class member from each debtor must accept the plan, which in this case means Grasslawn must accept it for all five debtors. Since they did not, the plan does not meet the confirmation requirement.

However, if interpreted as "per plan" (i.e., at least one class member from a single debtor's combined group of related companies), then only two classes need to agree, which is the case here.

Given the ambiguity in legal interpretation, it would be up to the bankruptcy court to decide. If they follow a strict "per debtor" approach, the plan cannot proceed as Grasslawn did not accept it for all debtors.

```java
if (confirmationRequirement == "per debtor") {
    // Check if at least one class member accepted from each debtor
} else if (confirmationRequirement == "per plan") {
    // Check if at least one class member accepted from the combined group
}
```
x??

#### Agency Relationships
Agency relationships are a crucial legal framework where one party (the principal) authorizes another (the agent) to act on their behalf and subject to their control. This concept is fundamental in both employment law and business transactions.

:p What defines an agency relationship according to the Restatement (Third) of Agency?
??x
An agency relationship results from the manifestation of consent by one person (principal) to another (agent) that the latter shall act on behalf and instead of the principal in negotiating and transacting business with third parties. The agent's actions are subject to the principal's control.
x??

---

#### Fiduciary Duty in Agency
The term "fiduciary" is central to understanding agency relationships. A fiduciary relationship involves a high degree of trust and confidence, where the fiduciary (agent) has a duty to act primarily for the benefit of the other party (principal).

:p How does the term "fiduciary" apply in an agency context?
??x
In an agency context, a fiduciary refers to the agent who undertakes to act primarily for the principal's benefit. The relationship is characterized by trust and confidence, where the agent must prioritize the principal’s interests.
x??

---

#### Employer-Employee Agency Relationship
Employees often act as agents of their employers when dealing with third parties in business transactions. This means that any actions or representations made by employees within the scope of their employment are binding on the employer.

:p In what situations can an employee be considered an agent for their employer?
??x
An employee is typically deemed to be an agent for their employer if they engage in activities related to their job duties, especially when dealing with third parties. Their actions or representations made within the scope of their employment are legally binding on the employer.
x??

---

#### Agency Law vs Employment Law
Agency law has a broader reach than employment law because it can exist outside traditional employer-employee relationships. Furthermore, agency law is based on common law principles, while much employment law is statutory.

:p How does agency law differ from employment law?
??x
Agency law encompasses a wider range of relationships beyond just employer-employee interactions. It relies on common law principles, whereas employment law is often statutory and specific to the relationship between employers and employees.
x??

---

#### Corporate Officers as Agents
Corporate officers serve in representative capacities for corporations and are considered agents who have authority to bind the corporation through their actions.

:p Can you give an example of how corporate officers function as agents?
??x
Certainly, a corporate officer serves as an agent by conducting business on behalf of the corporation. For instance, when signing a contract, the officer has the authority to bind the corporation legally. This is because the officer acts within the scope of their duties and responsibilities.
x??

---

#### Multi-Location Business Operations through Agents
Using agents allows principals (e.g., corporations) to conduct multiple business operations simultaneously in different locations.

:p How do agents facilitate multi-location business operations?
??x
Agents enable businesses to operate across various locations by representing the principal. For example, a corporate officer working from an office in one city can negotiate and enter into contracts on behalf of the company's headquarters located elsewhere. This allows for efficient and simultaneous operation in multiple regions.
x??

---

#### Legal Ruling on Coker v. Pershad, Five Star, and AAA
The case of Coker v. Pershad, Five Star, and AAA involved a suit where Coker alleged that AAA was responsible for the tortious conduct by Pershad. The court ruled that Pershad was an employee of Five Star, which was classified as an independent contractor under AAA. An appellate court upheld this ruling. The significant point is that because AAA did not control Five Star’s work, it was not held liable for any tort committed by Five Star's employees.
:p What does the legal case Coker v. Pershad, Five Star, and AAA illustrate about employer liability in a multi-tiered structure?
??x
The court case illustrates how multi-tiered employment structures can affect liability. In this scenario, despite Coker alleging that AAA was responsible for Pershad’s tortious conduct, the court ruled that since Pershad was an employee of Five Star, and Five Star was an independent contractor under AAA, AAA was not held liable because it did not control Five Star's work.
??x
---

#### Classification of Uber and Lyft Drivers as Independent Contractors
The transportation-for-hire industry has significantly evolved with the advent of companies like Uber and Lyft. These companies operate in numerous countries and cities worldwide. However, their business model often classifies drivers as independent contractors rather than employees. This classification allows the platforms to avoid providing certain benefits and protections typically given to employees.
:p What are the implications for workers when they are classified as independent contractors by digital platforms like Uber and Lyft?
??x
When workers are classified as independent contractors, they do not receive many of the legal protections and benefits that employees enjoy. For example, they are responsible for their own insurance, taxes, and often have no minimum wage protection or anti-discrimination laws to rely on.
??x
---

#### Misclassification Lawsuits in California
In California, several lawsuits have been filed by Uber and Lyft drivers seeking reclassification as employees rather than independent contractors. Two federal court judges allowed separate lawsuits to proceed, questioning the classification of these workers. The main argument is that these workers should receive more protections typically afforded to traditional employees.
:p How did some courts handle worker misclassification claims in California for Uber and Lyft drivers?
??x
In California, some courts have allowed cases to go before juries to determine if on-demand drivers should be considered employees rather than independent contractors. For instance, judges permitted separate lawsuits involving Uber and Lyft drivers to proceed, challenging their classification status.
??x
---

#### Settlement with Lyft in Worker Misclassification Lawsuit
Lyft faced a significant worker misclassification lawsuit but did not seek reclassification as employees through settlement. Instead, the company agreed to change its terms of service to better align with California’s independent contractor regulations. This included prohibiting the deactivation of drivers' accounts without reason and providing them with due process.
:p What was Lyft's approach in settling a worker misclassification lawsuit?
??x
Lyft took a different path from full reclassification by accepting a settlement that altered its terms of service to better fit California’s independent contractor status regulations. Specifically, the company agreed not to deactivate drivers' accounts without reason and provided them with fair hearings.
??x
---

