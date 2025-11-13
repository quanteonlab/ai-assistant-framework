# Flashcards: Business-Law_-Text-and-Cases_processed (Part 158)

**Starting Chapter:** 51-3 Trusts

---

---
#### Per Capita Distribution
Per capita distribution is a method of distributing an estate where each person in a specified group receives an equal share. This term is derived from Latin, meaning "per head" or "for each person."

:p What does per capita distribution mean?
??x
In the context of estate distribution, per capita means that each individual within a designated class receives an equal share of the estate.
For example, if Michael's estate is to be distributed per capita and Becky, Holly, and Paul are part of this group, they would each receive one-third of the estate.

```java
// Example of pseudocode for distributing assets per capita
public void distributeEstate(List<String> beneficiaries) {
    double totalAssets = 1000000; // Total value of Michael's estate
    int numBeneficiaries = beneficiaries.size();
    
    if (numBeneficiaries > 0) {
        double sharePerPerson = totalAssets / numBeneficiaries;
        
        for (String beneficiary : beneficiaries) {
            System.out.println(beneficiary + " receives: " + sharePerPerson);
        }
    } else {
        System.out.println("No beneficiaries to distribute the estate.");
    }
}
```
x?
---

#### Trusts
A trust is an arrangement where property is transferred from one person (the settlor or grantor) to a trustee for the benefit of another party, known as the beneficiary. Trusts can be created during a person's lifetime or after their death.

:p What is a trust in estate law?
??x
In estate law, a trust is an arrangement by which property is transferred from one person (the settlor or grantor) to a trustee for the benefit of another party, known as the beneficiary. The settlor can create a trust during their lifetime or after death.

```java
// Example of pseudocode defining a basic trust structure
public class Trust {
    private String settlor; // Name of the person creating the trust
    private String trustee; // Name of the person managing the trust
    private List<String> beneficiaries; // Names of the people benefiting from the trust
    
    public Trust(String settlor, String trustee, List<String> beneficiaries) {
        this.settlor = settlor;
        this.trustee = trustee;
        this.beneficiaries = beneficiaries;
    }
    
    public void distributeAssets(double totalAssets) {
        double sharePerPerson = totalAssets / beneficiaries.size();
        
        for (String beneficiary : beneficiaries) {
            System.out.println(beneficiary + " receives: " + sharePerPerson);
        }
    }
}
```
x?
---

#### Express Trusts
An express trust is a trust created or declared in explicit terms, typically in writing. It can be established during the grantor's lifetime or after their death.

:p What characterizes an express trust?
??x
An express trust is characterized by its creation or declaration in explicit terms and usually in writing. Each type of express trust has unique characteristics. For example, a living trust (inter vivos) created during one’s lifetime to pass assets to heirs without probate is a form of express trust.

```java
// Example of pseudocode for an express trust
public class ExpressTrust {
    private String grantor; // Name of the person creating the trust
    private List<String> beneficiaries; // Names of the people benefiting from the trust
    
    public ExpressTrust(String grantor, List<String> beneficiaries) {
        this.grantor = grantor;
        this.beneficiaries = beneficiaries;
    }
    
    public void createTrust(double totalAssets) {
        for (String beneficiary : beneficiaries) {
            System.out.println(beneficiary + " will receive assets from the trust.");
        }
    }
}
```
x?
---

#### Living Trusts
A living trust, or inter vivos trust, is a trust created by a grantor during their lifetime. These trusts are popular for estate planning because they can pass assets to heirs without probate.

:p What is a living trust and its benefits?
??x
A living trust, also known as an inter vivos trust (meaning "between or among the living"), is a trust created by a grantor during their lifetime. Its primary benefit is that it allows assets held in the trust to pass directly to heirs without going through probate after death.

```java
// Example of pseudocode for creating a living trust
public class LivingTrust {
    private String grantor; // Name of the person creating the trust
    private List<String> beneficiaries; // Names of the people benefiting from the trust
    
    public LivingTrust(String grantor, List<String> beneficiaries) {
        this.grantor = grantor;
        this.beneficiaries = beneficiaries;
    }
    
    public void transferAssets(double totalAssets) {
        System.out.println(grantor + " transfers assets to " + beneficiaries.get(0));
        // Additional logic for transferring assets
    }
}
```
x?
---

#### Revocable Living Trusts
A revocable living trust is the most common type of living trust, where the grantor retains control over the trust property during their lifetime. The grantor can amend, alter, or revoke the trust.

:p What are the characteristics of a revocable living trust?
??x
A revocable living trust allows the grantor to retain control over the trust property and its assets during their lifetime. Key features include the ability to:
- Amend, alter, or revoke the trust.
- Serve as trustee or co-trustee if desired.
- Arrange to receive income earned by the trust assets.

```java
// Example of pseudocode for a revocable living trust
public class RevocableLivingTrust extends LivingTrust {
    private boolean isRevocable; // Indicates whether the trust can be revoked
    
    public RevocableLivingTrust(String grantor, List<String> beneficiaries) {
        super(grantor, beneficiaries);
        this.isRevocable = true;
    }
    
    public void revokeTrust() {
        if (isRevocable) {
            System.out.println("The trust has been successfully revoked.");
        } else {
            System.out.println("This trust cannot be revoked.");
        }
    }
}
```
x?
---

#### Wills and Trusts Overview
Background context: This section discusses statutory limitations on trustees' investment powers, their fiduciary duties, and how trust receipts and expenses are allocated between income and principal. It also covers scenarios where trusts may be terminated and other estate-planning issues.
:p What is typically restricted for trustees in terms of investments according to statutes?
??x
Statutes generally confine trustees to conservative debt securities such as government, utility, and railroad bonds, as well as certain real estate loans.
x??

---

#### Discretionary Investment Powers
Background context: While statutory restrictions often limit investment options, a grantor may grant discretionary powers to the trustee. In such cases, statutes are advisory, and the trustee's decisions must comply with the prudent person rule.
:p Can you explain what happens if a trustee fails to comply with the terms of the trust or the controlling statute?
??x
If a trustee fails to comply with the terms of the trust or the controlling statute, he or she is personally liable for any resulting loss. The trustee has a fiduciary duty to carry out the purposes of the trust.
x??

---

#### Allocation Between Principal and Income
Background context: A grantor may provide one beneficiary with a life estate and another with the remainder interest in the trust. This can lead to questions about how receipts and expenses should be allocated between income and principal.
:p How are ordinary and extraordinary receipts and expenses typically allocated?
??x
Ordinary receipts and expenses are generally charged to the income beneficiary, while extraordinary receipts and expenses are allocated to the principal beneficiaries. For instance, rent from trust realty is considered ordinary, but long-term improvements or sale proceeds would be extraordinary.
x??

---

#### Termination of a Trust
Background context: The terms of a trust should state explicitly when it will terminate. This can be based on events like the beneficiary's or trustee's death, a specific date, or fulfillment of the trust's purpose. If no date is specified, the trust terminates when its purpose has been fulfilled.
:p What happens if a trust instrument does not specify termination upon the beneficiary’s death?
??x
If a trust instrument does not provide for termination on the beneficiary’s death, the beneficiary’s death will not end the trust. Similarly, without an express provision, a trust will not terminate on the trustee’s death either.
x??

---

#### Estate-Planning Issues
Background context: Estate planning involves making decisions about asset inheritance and care of minor children. It also includes preparing for contingencies like illness or incapacity.
:p What are some key considerations in estate planning?
??x
Key considerations include deciding who will inherit the family home and other assets, determining guardianship for minor children, and preparing for potential health issues through advance directives.
x??

---

---
#### Issue Spotter 1: Capacity to Make a Will
Sheila's will was made when she was competent but was subsequently declared incompetent. The issue is whether Toby and Umeko can have Sheila’s will revoked due to her lack of capacity at the time of execution.

:p Can Toby and Umeko revoke Sheila’s will on the grounds that she did not have the capacity to make a will?
??x
No, Toby and Umeko cannot revoke Sheila's will based on the grounds that she was incompetent when making it. Under most legal systems, the validity of a will is determined by whether the testator had the requisite mental capacity at the time they executed the will, not at the time of their death or subsequent incompetence.

The will's validity would have been established when Sheila made it, assuming all other formal requirements were met (such as having proper witnesses). Since Sheila was deemed competent during the execution of her will, the will remains valid and enforceable even after she became incompetent later.
x??
---

---
#### Issue Spotter 2: Intestacy Laws
Rafael died intestate, meaning he did not have a will. His relatives include his spouse, biological children, adopted children, sisters, brothers, uncles, aunts, cousins, nephews, and nieces.

:p Who inherits Rafael’s estate based on the rules of intestacy?
??x
In most jurisdictions, intestacy laws determine the distribution of an estate based on family relationships. The primary beneficiaries would typically be:

1. **Spouse**: Usually gets a portion of the estate (varies by jurisdiction).
2. **Children**: Biological and adopted children are usually considered first.
3. **Parents**: If no surviving spouse or children, parents may inherit.
4. **Siblings and other relatives**: Further down the line if applicable.

The specific distribution rules vary significantly depending on local laws, but generally, Rafael's estate would be divided according to these primary relationships unless there is a clear order of precedence defined by statute.
x??
---

---
#### Business Scenario 51-1: Validity of Wills
Benjamin’s will leaves his property equally to his two children (Edward and Patricia) and their grandchildren per stirpes. The will was witnessed and signed in the presence of Patricia and Benjamin's lawyer.

:p Is the will valid according to the given information?
??x
Yes, the will appears to be valid based on the provided details:
- **Formalities**: A will needs proper witnesses (in this case, Patricia and Benjamin’s lawyer) and a signature by the testator.
- **Distribution**: The distribution as per stirpes is clear.

If there are no other issues such as undue influence or fraud, the will should be valid. However, Edward claims it's invalid, so we would need to consider his specific arguments (e.g., fraud, mistake, lack of capacity).

The will’s validity could also depend on local laws regarding witness requirements and whether all necessary formalities were met.
x??
---

---
#### Business Scenario 51-2: Specific Bequests
Gary Mendel's will states that he left a 1966 red Ferrari to his daughter, Roberta. A year before his death, Mendel sold the 1966 Ferrari and purchased a 1969 Ferrari.

:p Will Roberta inherit the 1969 Ferrari under her father’s will?
??x
No, Roberta would not inherit the 1969 Ferrari. Under most legal systems, specific bequests are linked to the item named in the will at the time of the testator's death. The 1966 red Ferrari Mendel promised was sold and replaced by a 1969 Ferrari, which is no longer the same property.

In such cases, courts typically use substitution: they look for a similar item or substitute to be given to the beneficiary.
x??
---

---
#### Business Scenario 51-3: Revocation of Wills
James made an initial will naming his mother as sole beneficiary. Later, he married Lisa and then divorced her but did not change his will. He eventually remarried Mandis.

:p If James died while married to Lisa without changing his will, would the estate go to his mother or Lisa?
??x
If James died while married to Lisa without changing his will, the estate would likely still go to his mother, Carol. Typically, a will's terms override intestacy laws unless changed explicitly. Since the original will named his mother as the sole beneficiary and he did not update it despite getting married, the will remains valid.

However:
- If James made a new will after marrying Lisa, it might leave everything to Lisa.
- If he simply updated the will to include Mandis without removing Carol, then both could potentially inherit according to the terms of the last valid will.

The exact distribution would depend on whether and when any changes were made to his will.
x??
---

---
#### Business Scenario 51-4: Wills and Guardian's Rights
Elnora Maxey was the guardian of Sean Hall after his parents died. She left two houses in her estate to Hall, but Jordan (Hall’s new guardian) paid for mortgage and tax payments on the houses.

:p What happens when Hall dies intestate with those houses remaining in Maxey's estate?
??x
When Hall dies intestate, the legal system will determine how his property is distributed. Since he inherited two houses from Elnora Maxey's estate but Jordan (his guardian) paid for mortgage and tax payments on them, it’s important to consider these details:

1. **Legal Ownership**: The houses are part of Hall's estate because they were bequeathed to him.
2. **Payment Responsibility**: Since Jordan paid the mortgage and taxes, he could potentially claim a lien or have some rights over the property.

Upon Hall's death, Jordan would become the administrator of Hall’s estate. Typically, the legal process involves:
- Filing probate to identify and distribute Hall’s assets.
- Paying off debts and liens (Jordan might seek reimbursement for his payments).
- Distributing the remaining assets according to intestacy laws if no will is found.

The exact distribution would be determined by local probate laws and may require court intervention.
x??
---

