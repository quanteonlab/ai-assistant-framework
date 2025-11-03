# Flashcards: 2B001--Financial-Data-Engineering_processed (Part 12)

**Starting Chapter:** Derivative Identifiers

---

#### SEDOL Technical Documentation and Usage
Background context: The Sedol (Stock Exchange Daily Official List) is a seven-character alphanumeric code used to identify securities traded on the London Stock Exchange (LSE) and other UK exchanges. It was initially used exclusively for UK securities but has since expanded globally.
:p What are the key features of SEDOL?
??x
SEDOL codes provide unique identifiers for securities listed in multiple jurisdictions, making them ideal for international trading. They consist of seven characters: an alphabetic character followed by five alphanumeric characters and a trailing numeric check digit. Prior to March 2004, they were exclusively numeric.
x??

---
#### SEDOL Code Structure
Background context: The structure of the SEDOL code is essential for understanding its validation process. The code consists of seven characters with specific positions assigned weights for validation purposes.
:p What is the format of a SEDOL code?
??x
A typical SEDOL code follows this format: Alpha-Num-Num-Num-Num-Num-CheckDigit where:
1. Alpha - An alphabetic character (first position)
2. Num - A numeric digit (second to sixth positions)
3. CheckDigit - The trailing numeric check digit

For example, the SEDOL for HSBC is 0540528.
x??

---
#### SEDOL Validation Process
Background context: To ensure the accuracy of a SEDOL code, a validation process is used where the check digit ensures that the weighted sum of all characters is a multiple of 10. This helps prevent errors in data entry or transmission.
:p How do you validate a SEDOL code?
??x
To validate a SEDOL code, follow these steps:
1. Convert non-numeric characters to digits based on their position (A = 10, B = 11, etc.).
2. Multiply each of the first six numbers by their corresponding weights: 1 for the first position, 3 for the second, 1 for the third, 7 for the fourth, 3 for the fifth, and 9 for the sixth.
3. Get the sum of all values.
4. Calculate the check digit as (10 - (sum modulo 10)) modulo 10.

Here is a Python pseudocode example:
```python
def validate_sedol(sedol):
    # Convert non-numeric characters to digits
    sedol = [ord(char) if char.isalpha() else int(char) for char in sedol]
    
    # Apply weights and calculate sum
    weighted_sum = 0
    weights = [1, 3, 1, 7, 3, 9]
    for i, digit in enumerate(sedol[:6]):
        weighted_sum += digit * weights[i]

    # Calculate check digit
    check_digit = (10 - (weighted_sum % 10)) % 10

    return sedol[-1] == check_digit
```
x??

---
#### SEDOL for UK Securities
Background context: For UK securities, the SEDOL is embedded within the UK ISIN code by adding the country code at the beginning, followed by two padding zeros, then the SEDOL, and finally, the ISIN check code. This ensures a unique identifier that can be used globally.
:p What is the format of an SEDOL for UK securities?
??x
For UK securities, the SEDOL is embedded within the UK ISIN (International Securities Identification Number) by adding the country code "GB" at the beginning, followed by two padding zeros, then the SEDOL, and finally, the ISIN check code. For example, HSBC has an SEDOL of 0540528 and an ISIN code of GB0005405286.
x??

---
#### Ticker Symbols
Background context: A ticker symbol is a short and unique series of letters assigned to financial securities for listing and trading purposes. There are no standard rules, but organizations like exchanges or financial data providers often generate these symbols based on company names.
:p What is the purpose of a ticker symbol?
??x
The primary purpose of a ticker symbol is to provide a unique identifier for financial securities (mostly stocks) on an exchange. Ticker symbols facilitate easy identification and trading of securities, making them indispensable in financial markets.
x??

---
#### Example of Ticker Symbol Validation
Background context: Tickers can vary widely in length and format depending on the organization issuing them. For example, US exchanges typically use one to four characters for tickers, while NASDAQ may extend this with additional symbols.
:p How do you validate a ticker symbol?
??x
Ticker validation primarily involves checking if the ticker matches known formats or patterns used by specific exchanges. Since there are no standardized rules, validation is often done through predefined lists or databases.

For example, in Python:
```python
def validate_ticker(ticker):
    # Predefined list of valid tickers for a given exchange
    valid_tickers = ["AAPL", "MSFT", "GOOG"]
    
    return ticker in valid_tickers
```
x??

---

#### Bloomberg Tickers
Background context: Financial data providers assign proprietary ticker symbols to financial instruments. The Bloomberg ticker is a unique identifier within the Bloomberg ecosystem, which can include exchange-specific tickers, market sectors, and instrument-specific information such as bond maturity or option expiry.

:p What is a Bloomberg Ticker and what components does it typically contain?
??x
A Bloomberg Ticker is a unique identifier for financial entities within the Bloomberg ecosystem. It includes several components:
- Exchange-specific ticker: Identifies where the security is listed.
- Market sector: Specifies the type of instrument (e.g., stock, bond).
- Exchange code: Codes to indicate the specific exchange.
- Instrument-specific information: Details like bond maturity or option expiry.
Example structure: BBG000123456

```java
public class BloombergTicker {
    private String exchangeCode;
    private String ticker;
    private String marketSector;
    
    public BloombergTicker(String exchangeCode, String ticker, String marketSector) {
        this.exchangeCode = exchangeCode;
        this.ticker = ticker;
        this.marketSector = marketSector;
    }
}
```
x??

---

#### Refinitiv Instrument Code (RIC)
Background context: Another well-known proprietary ticker symbol is the Refinitiv Instrument Code (RIC), used mainly on LSEG platforms. RIC tickers consist of a security’s ticker symbol with an optional character that identifies the exchange.

:p What is the structure of a Refinitiv Instrument Code (RIC) and how does it identify securities?
??x
The structure of a Refinitiv Instrument Code (RIC) includes:
- The security's ticker symbol: Identifies the specific financial instrument.
- An optional character to denote the exchange, e.g., ".N" for the New York Stock Exchange.

Example RIC: IBM.N

```java
public class RefinitivInstrumentCode {
    private String tickerSymbol;
    private char exchangeIdentifier;
    
    public RefinitivInstrumentCode(String tickerSymbol, char exchangeIdentifier) {
        this.tickerSymbol = tickerSymbol;
        this.exchangeIdentifier = exchangeIdentifier;
    }
}
```
x??

---

#### Stock Tickers and Uniqueness
Background context: Stock tickers can vary by exchange and country. To reliably identify a stock, both the ticker and the exchange or country of listing are often required. Tickers are not immutable and may change due to corporate actions such as mergers.

:p Why is it important to consider both the ticker and the exchange when identifying stocks?
??x
It is crucial to consider both the ticker and the exchange because:
- Stock tickers can vary by exchange and country, making them non-unique.
- Tickers are not immutable; they may change due to corporate actions like mergers.

Example: The ticker "XON" was used by Exxon before the merger with Mobil Oil in 1999. After the merger, it changed to "XOM".

```java
public class StockIdentifier {
    private String ticker;
    private Exchange exchange;
    
    public StockIdentifier(String ticker, Exchange exchange) {
        this.ticker = ticker;
        this.exchange = exchange;
    }
}
```
x??

---

#### Tickers Reassignment and Changes
Background context: Tickers are not guaranteed to remain unique and can be reassigned over time. An example is the ticker SNOW, which was used by Intrawest Resorts Holdings, Inc., until it was reassigned to Snowflake after a delisting.

:p Provide an example of how tickers can change over time.
??x
An example is the ticker SNOW:
- Prior to 2017: Assigned to Intrawest Resorts Holdings, Inc. on NYSE.
- In May 2017: Intrawest was acquired by Henry Crown and KSL Capital Partners and transformed into a privately owned company.
- After delisting Intrawest, the ticker SNOW was reassigned to Snowflake.

```java
public class TickerChange {
    private String oldTicker;
    private String newCompany;
    
    public TickerChange(String oldTicker, String newCompany) {
        this.oldTicker = oldTicker;
        this.newCompany = newCompany;
    }
}
```
x??

---

#### Derivative Instruments Identification
Background context: Derivatives are complex financial instruments that derive their value from underlying assets. Identifying derivative instruments is challenging due to several constituent elements and the flexible, customizable nature of derivatives.

:p Why is identifying derivative instruments more challenging compared to other types of financial instruments?
??x
Identifying derivative instruments is more challenging because:
- Several constituent elements must be considered.
- Derivatives can easily become very complex products.
- A substantial portion is traded OTC, complicating tracking and identification.

```java
public class DerivativeIdentifier {
    private String underlyingAsset;
    private String strikePrice;
    
    public DerivativeIdentifier(String underlyingAsset, double strikePrice) {
        this.underlyingAsset = underlyingAsset;
        this.strikePrice = strikePrice;
    }
}
```
x??

---
#### Option Symbol Structure
Background context explaining how option symbols are structured and used. The OSI format is a 21-character alphanumeric code that includes the underlying stock or ETF ticker, expiration date (YYMMDD), option type (C for call, P for put), and strike price represented as price * 1000.

:p What is the structure of an OSI option symbol?
??x
The structure of an OSI option symbol consists of four parts:
1. A root (ticker) symbol representing the underlying stock or ETF.
2. An expiration date in the format YYMMDD.
3. An option type: C for a call, P for a put.
4. The strike price, which is represented by the price multiplied by 1000 and padded with leading zeros to make it eight digits long. A decimal point falls three places from the right.

For example:
- Ticker symbol: IBM
- Expiration date: 230625 (June 25, 2023)
- Option type: C for call or P for put
- Strike price: $145 -> 145000.000

The complete option symbol could look like this:
IBM230625C145000.000

x??
---
#### OTC Derivatives Identification System
Background context explaining the identification system for over-the-counter (OTC) derivatives using a combination of identifiers: OTC ISIN, Unique Product Identifier (UPI), and Classification of Financial Instruments (CFI).

:p What is the combined identifier scheme used to identify OTC derivatives?
??x
The combined identifier scheme for OTC derivatives consists of three identifiers:
1. **OTC ISIN**: Allocated by the Derivatives Service Bureau (DSB) with a custom "EZ" code prefix.
2. **Unique Product Identifier (UPI)**: Defined in ISO 4914 to identify specific OTC derivative products.
3. **Classification of Financial Instruments (CFI)**: Generated by the DSB during the OTC ISIN generation process.

These three identifiers work together to provide detailed information about the derivative product, starting from a broad classification down to granular details.

For example:
- CFI can tell that it is "single currency, fix-float, interest rate swap with a constant notional schedule and cash delivery."
- UPI provides more specific details such as the reference rate term (e.g., three-month USD-LIBOR-BBAR).
- OTC ISIN offers even more granular information like the full name of the instrument.

x??
---

#### Alternative Instrument Identifier (AII)
Background context: AII is used within the European Union for reporting purposes to identify derivatives traded on regulated markets that do not have an ISIN. It consists of concatenated descriptive fields such as exchange code, product code, derivative type, put/call identifier, expiry/delivery date, and strike price.

:p What is AII?
??x
AII stands for Alternative Instrument Identifier and is used to uniquely identify derivatives traded on regulated markets without an ISIN. It concatenates several descriptive fields like the exchange code, product code, derivative type, put/call indicator, expiry or delivery date, and strike price.
x??

---

#### Financial Instrument Global Identifier (FIGI)
Background context: FIGI is a 12-character alphanumeric identifier for financial instruments, introduced by Bloomberg in 2009. It covers millions of active and inactive securities across various asset classes and was adopted as an open industry standard by the Object Management Group.

:p What is FIGI?
??x
FIGI is a unique 12-character alphanumeric identifier for financial instruments developed by Bloomberg. It aims to provide global identification covering stocks, bonds, derivatives, loans, indexes, funds, and digital assets.
x??

---

#### Hierarchical Structure of FIGI
Background context: The FIGI system has three levels—Global FIGI, Composite Global FIGI, and Share Class Global FIGI—which form a hierarchical structure for identifying financial instruments across different venues and countries.

:p What are the three levels in the FIGI hierarchy?
??x
The three levels in the FIGI hierarchy are:
1. **Global FIGI**: Identifies a specific instrument at a particular trading venue.
2. **Composite Global FIGI**: Aggregates multiple venue-level FIGIs within the same country, providing broader identification for all venues of a specific instrument within that country.
3. **Share Class Global FIGI**: Further aggregates FIGIs to cover financial instruments across multiple countries, providing a global view of a single instrument regardless of the trading venue and country.

For example, Amazon common stock on different US exchanges would have distinct FIGIs but could be referenced generically with composite or share class FIGIs.
x??

---

#### Structure of FIGI Codes
Background context: Each level of FIGI codes has a specific structure. The identifier consists of letters in [B, C, D, F, G, H, J, K, L, M, N, P , Q, R, S, T, V , W , X, Y , Z] and zero to nine digits.

:p What is the structure of a FIGI code?
??x
A FIGI code consists of:
- **First two characters**: Identify the certified issuer (e.g., "BB" for Bloomberg).
- **Third character**: Always 'G' indicating it's a global identifier.
- **Characters 4–11**: Alphanumeric reference ID.
- **Trailing check digit**.

The structure can be visualized as:
```
BBG000BLNQ16
```

For validation, the Luhn algorithm is used to ensure correctness. Here’s an example in Python for checking the validity of a FIGI code:

```python
def validate_figi(figi):
    # Remove the check digit
    figi = figi[:-1]
    
    # Convert non-numeric characters to digits (A=10, B=11, etc.)
    figi = [ord(c) - 55 if c.isalpha() else int(c) for c in figi]
    
    # Double every second digit
    doubled = [2 * n if i % 2 == 1 else n for i, n in enumerate(figi)]
    
    # Sum the digits and compute the check digit
    sum_digits = sum(doubled)
    check_digit = (10 - (sum_digits % 10)) % 10
    
    return check_digit == figi[-1]

# Example usage:
figi_code = "BBG000BLNQ1"
print(validate_figi(figi_code))
```

x??

---

#### FactSet Permanent Identifier (FPID)
Background context: The FactSet permanent identifier (FPID) is a proprietary identification system developed by FactSet to offer a stable and unified identifier for securities. It includes three levels of identifiers:
1. **Security Level**: Identifies the security globally.
2. **Regional Level**: Identifies the security at the regional level per currency.
3. **Listing Level**: Identifies the security at the market level.

FactSet provides two APIs to work with FPID:
- **FactSet Symbology API**: A symbol/identifier resolution service that maps a wide variety of identifiers (such as CUSIPs, SEDOLS, ISINs, Bloomberg FIGI) to FactSet’s native symbology.
- **FactSet Concordance API**: Enables users to programmatically match the FactSet identifier for a specific entity based on attributes like name, URL, and location.

:p What is the FactSet permanent identifier (FPID)?
??x
The FactSet Permanent Identifier is a proprietary system developed by FactSet that provides a stable and unified identification method for securities across different levels: global security level, regional per currency level, and market listing level. It allows users to map various identifiers to their native symbology through APIs.
x??

---
#### LSEG Permanent Identifier (PermID)
Background context: The London Stock Exchange Group (LSEG) Permanent Identifier (PermID) is a unique identifier used within the information model of LSEG to identify and reference various objects, such as organizations, instruments, funds, issuers, and people. It ensures accurate and unambiguous referencing even when relationships between entities change over time.

PermID has valuable attributes, including:
- A unique web address or Uniform Resource Identifier (URI) that offers a permanent, direct link to the identified entity.
- Open-source nature: Accessible via web pages or API-based entity search and matching services.

Example PermIDs for Apple Inc.:
- Organizational details: https://permid.org/1-4295905573
- Instrument details: https://permid.org/1-8590932301
- Quote information: https://permid.org/1-25727408109

:p What is the LSEG Permanent Identifier (PermID)?
??x
The LSEG Permanent Identifier is a unique identifier used within the LSEG information model to identify and reference various objects such as organizations, instruments, funds, issuers, and people. It provides permanent, direct links through URIs and is open-source, allowing access via web pages or API-based services.
x??

---
#### Digital Asset Identifiers
Background context: With the emergence of blockchain and distributed ledger technologies (DLT), digital assets have become a new type of financial market entity. According to ISO 22739 vocabulary, a digital asset is an "asset that exists only in digital form or which is the digital representation of another asset."

Digital assets can be further categorized into:
- **Fungible Digital Assets**: Identical and interchangeable with similar assets (e.g., one dollar).
- **Nonfungible Digital Assets (NFTs)**: Unique and nondivisible (e.g., a painting).

The most common place for exchanging digital assets is on a blockchain, which, according to ISO 22739 vocabulary, refers to a "distributed ledger with confirmed blocks organized in an append-only, sequential chain using cryptographic links."

:p What are digital assets?
??x
Digital assets refer to assets that exist solely in digital form or serve as the digital representation of another asset. They can be fungible (identical and interchangeable) or nonfungible (unique and nondivisible). These assets often reside on a blockchain, which is a distributed ledger with confirmed blocks linked together using cryptographic techniques.
x??

---

#### Tokenization Process
Tokenization is the process of converting something into a digital asset and adding it to a blockchain system. This method has allowed for the creation of various types of fungible digital assets, which are used extensively today.

:p What is tokenization?
??x
Tokenization involves transforming physical or tangible assets (like real estate) or intangible assets (like music rights) into digital tokens that can be stored and traded on a blockchain. These digital assets can represent ownership, utility, or financial instruments.
x??

---

#### Crypto Assets
Crypto assets are developed using cryptographic techniques and include notable examples such as cryptocurrencies like Bitcoin and Ethereum. These assets serve multiple purposes, including being used as a means of payment or investment.

:p What are crypto assets?
??x
Crypto assets are digital representations of value that use cryptography for secure transactions and to control the creation of additional units. They operate on blockchain technology, enabling decentralization and security.
x??

---

#### Security Tokens
Security tokens represent ownership rights in a company or asset and can be exchanged on distributed ledgers like blockchain. Similar to traditional securities certificates, they provide a digital alternative.

:p What are security tokens?
??x
Security tokens are digital assets that represent a financial interest or ownership right in an underlying asset or company. They enable tokenization of traditional financial instruments and can be traded transparently on blockchain platforms.
x??

---

#### Nonfungible Tokens (NFTs)
Nonfungible tokens represent unique digital assets such as images, videos, and games. Unlike fungible tokens, each NFT is distinct and cannot be exchanged for another identical NFT.

:p What are nonfungible tokens?
??x
Nonfungible tokens (NFTs) are unique digital assets that can be bought, sold, and traded on blockchain networks. Each NFT has a unique identifier or metadata that distinguishes it from other tokens.
x??

---

#### Utility Tokens
Utility tokens are used to access specific services or features within a blockchain-based ecosystem.

:p What are utility tokens?
??x
Utility tokens are digital assets designed to grant access to specific services, products, or applications within a particular blockchain project. They are often used as an incentive for early users.
x??

---

#### Digital Token Identifier (DTI)
The DTI is defined in ISO 24165-1 and is used to identify fungible tokens and digital ledgers. It helps reduce confusion and increase trust by providing a universal method of identification.

:p What is the Digital Token Identifier (DTI)?
??x
The Digital Token Identifier (DTI) is a standard for identifying fungible tokens and their associated digital ledgers on blockchain networks. It provides a unique identifier to help distinguish between different tokens and enhance transparency.
x??

---

#### Industry and Sector Identifiers: Standard Industrial Classification (SIC)
Industry identifiers like the SIC code are used by financial firms to categorize businesses into various industries, aiding in market analysis and risk assessment.

:p What is the SIC classification system?
??x
The SIC (Standard Industrial Classification) system uses a four-character numerical code to classify industries based on their primary activities. It was created by the U.S. government to standardize data collection across different sectors.
x??

---

#### Industry and Sector Identifiers: North American Industry Classification System (NAICS)
NAICS is another industry classification framework used in North America, providing more detailed classifications than SIC.

:p What is the NAICS system?
??x
The NAICS (North American Industry Classification System) is a comprehensive framework for classifying industries in North America. It offers finer detail and updated categories compared to the older SIC system.
x??

---

#### Example of an SIC Code
SIC code 6021 (National Commercial Banks) belongs to industry group 602 (Commercial Banks), which is part of major group 60 (Depositary Institutions).

:p What does SIC code 6021 represent?
??x
SIC code 6021 represents National Commercial Banks, which fall under the broader category of Commercial Banks within the Depositary Institutions sector.
x??

---

---
#### SIC Code Shortcomings
Background context explaining why and how the SIC system had limitations. The SIC (Standard Industrial Classification) system faced several issues, including ambiguity, mismatched classifications, overlapping categories, and restrictions on accommodating new industries due to its four-digit structure.

:p What were some of the main shortcomings of the SIC code?
??x
The SIC code produced ambiguous, mis-matched, and overlapping classifications. It also restricted the addition of new business sectors and industries due to its rigid four-digit system.
x??

---
#### NAICS Code Structure
Explanation on how the North American Industrial Classification System (NAICS) addresses these shortcomings by introducing a more flexible six-character numeric code.

:p How does NAICS address the limitations of SIC?
??x
NAICS introduced a more flexible six-character numeric code, allowing for better classification and expansion into new business sectors. The first two digits indicate the major sector, the third digit indicates the subsector, the fourth digit designates the industry group, the fifth digit identifies the specific industry, and the sixth code specifies the national industry.
x??

---
#### Example of NAICS Code
Providing a detailed breakdown of how an NAICS code is structured to illustrate its use.

:p What does the NAICS code 522110 signify?
??x
NAICS code 522110 identifies commercial banks. The first two digits (52) define the sector as Finance and Insurance, the next three digits (522) identify the subsector as Credit Intermediation and Related Activities, the fourth digit (5221) defines the industry group as Depository Credit Intermediation, and the last two digits (10) specify Commercial Banking.
x??

---
#### NAICS Code Structure Breakdown
Further explanation on how to interpret the NAICS code structure.

:p How is an NAICS code structured?
??x
An NAICS code consists of six characters. The first two digits indicate the major sector, the third digit indicates the subsector, the fourth digit designates the industry group, the fifth digit identifies the specific industry, and the sixth code specifies the national industry.
x??

---
#### BIC Code Structure
Explanation on what a Business Identifier Code (BIC) is, its purpose, and how it is formatted.

:p What is a BIC and why is it important for banks?
??x
A Business Identifier Code (BIC), also known as SWIFT or Bank Identifier Code, is an alphanumeric code used to identify banks, financial institutions, and nonfinancial institutions worldwide. It is crucial for identifying entities during international money transfers and message routing.

The BIC consists of either 8 or 11 characters:
- Bank code: Four-character alphabetic code that usually represents a shortened version of the bank’s name.
- Country code: Two-character ISO 3166-1 alpha-2 code indicating the country where the bank is located.
- Location code: Two-character alphanumeric code designating the main office location.
- Branch code (optional): Three-character alphanumeric code representing a specific branch.

Example format:
```
BANKCOXX
```

x??

---
#### BIC Code Example
Detailed explanation of how to construct and use a BIC code, including an example.

:p How is a BIC code constructed?
??x
A BIC code is constructed as follows:

- Bank code: Four-character alphabetic characters identifying the bank. It usually looks like a shortened version of that bank’s name.
- Country code: A two-character ISO 3166-1 alpha-2 code indicating the country where the bank is located.
- Location code: Two-character alphanumeric code that designates where the bank’s main office is.
- Branch code (optional): An optional three-character alphanumeric code representing a specific branch.

For example, if you have Bank of America in New York with a main office and no specific branch:

```
BANKCOUSYY
```

Where:
- BANK: Shortened version of the bank's name.
- CO: Country code for the U.S. (United States).
- US: Location code for New York City.
x??

---

#### BIC Code Structure and Usage
Background context: The BIC (Bank Identifier Code) is used to identify financial institutions, especially for international transactions. It consists of 8-11 characters.
:p What does a typical BIC code look like?
??x
A typical BIC code looks like this: UNCRITMMLMXXX. Here, "UNCR" identifies the bank (UniCredit Banca), "IT" is the country code for Italy, "MM" is the office location code for Milan, and "XXX" indicates the head office.
x??

---
#### BSB Code in Australia
Background context: The BSB (Bank State Branch) code is a six-digit code used to identify branches of Australian financial institutions.
:p What is the structure of a BSB code?
??x
The BSB code has three parts:
- First 2 digits: Bank code
- Next 2 digits: State code
- Last 2 digits: Branch identifier
For example, "048167" could be broken down as follows: 
- 04: Bank code (Westpac Banking Corporation)
- 81: State code (Victoria)
- 67: Branch identifier
x??

---
#### ABA Routing Number in the U.S.
Background context: The ABA Routing Number is a nine-digit code used to identify financial institutions involved in various payment operations in the United States.
:p What is an example of an ABA Routing Number?
??x
An example of an ABA Routing Number could be "021000021". This number identifies a specific branch or division of the Bank of America.
x??

---
#### IBAN Structure and Validation
Background context: The International Bank Account Number (IBAN) is used to identify bank accounts when conducting money transfers. It consists of three main parts: country code, check digits, and Basic Bank Account Number (BBAN).
:p What are the main components of an IBAN?
??x
The main components of an IBAN include:
1. Country code following ISO 3166-1 alpha-2 convention.
2. Two check digits.
3. Basic Bank Account Number (BBAN) — up to 30 alphanumeric characters that include the bank code, branch identifier, and account number.
For example, a Luxembourgish IBAN might be LU28 0019 4006 4475 0000.
x??

---
#### Mod-97 Algorithm for IBAN Validation
Background context: The mod-97 algorithm is used to validate IBAN codes. It involves rearranging the code and converting non-numeric characters before applying a modulo operation.
:p How does the mod-97 algorithm work?
??x
The mod-97 algorithm works as follows:
1. Ensure the IBAN has a valid length.
2. Move the first four characters to the end of the string.
3. Convert all non-numeric characters to digits based on their ordinal position in the alphabet plus 9 (A = 10).
4. Treat the result as an integer and check if the modulo 97 is equal to 1.
Here's a Python snippet for validation:
```python
def validate_iban(iban):
    # Move the first four characters to the end of the string
    iban = iban[4:] + iban[:4]
    # Convert non-numeric characters to digits
    iban = ''.join(str(ord(c) - 55) if c.isalpha() else c for c in iban)
    # Check if modulo 97 of the number is equal to 1
    return int(iban) % 97 == 1
```
x??

---
#### Payment Card Number or PAN
Background context: The Primary Account Number (PAN) is defined by ISO/IEC 7812 and is used to define payment cards, including credit, debit, and gift cards. It typically includes the bank code, branch identifier, and account number.
:p What is a PAN?
??x
A PAN (Primary Account Number) is a unique identifier for payment cards, which includes:
- Bank code: Identifies the card issuer.
- Branch identifier: Specific to the branch or division of the issuing bank.
- Account number: The specific account linked to the cardholder.
The PAN is typically laser-printed on the front of most payment cards and contains up to 19 digits.
x??

---

