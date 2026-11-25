# Banking & Financial Services — Domain Knowledge Base

## Lending and Credit Regulations

The Truth in Lending Act (TILA) requires lenders to provide borrowers with clear, standardised disclosures of loan terms and costs before they commit. Key disclosures include the Annual Percentage Rate (APR), total finance charges, the payment schedule, and the total amount financed. Any loan origination system must generate TILA-compliant disclosure documents and present them to applicants before final acceptance.

The Equal Credit Opportunity Act (ECOA) prohibits credit discrimination based on race, colour, religion, national origin, sex, marital status, age, or the receipt of public-assistance income. Automated credit-decisioning engines must be auditable to prove that no prohibited factor influenced the approval or rejection decision. Adverse-action notices must be sent to rejected applicants within 30 days, citing the specific reasons for denial.

The Fair Credit Reporting Act (FCRA) governs how consumer credit information is collected, shared, and used. Systems that pull credit reports must obtain the applicant's written consent, use the data only for permissible purposes, and provide a process for consumers to dispute inaccuracies. Credit data must be stored securely and access-logged.

The Dodd-Frank Wall Street Reform and Consumer Protection Act established the Consumer Financial Protection Bureau (CFPB), which supervises consumer-facing financial products. Loan origination systems must support CFPB reporting requirements and implement fair-lending analytics to detect disparate impact across protected classes.

## Know Your Customer (KYC) and Anti-Money Laundering (AML)

The Bank Secrecy Act (BSA) requires financial institutions to assist government agencies in detecting and preventing money laundering. Systems must maintain records of cash purchases of negotiable instruments, file Currency Transaction Reports (CTRs) for cash transactions exceeding $10,000, and report suspicious activities.

Customer Due Diligence (CDD) rules require banks to identify and verify the identity of customers opening accounts. The four core elements are: identifying the customer, identifying beneficial owners of legal-entity customers, understanding the nature and purpose of the relationship, and conducting ongoing monitoring. Systems must collect government-issued ID, verify against authoritative sources, and screen against sanctions lists (OFAC, UN, EU).

Enhanced Due Diligence (EDD) applies to higher-risk customers such as Politically Exposed Persons (PEPs), customers from high-risk jurisdictions, and accounts with unusual transaction patterns. EDD requires additional verification steps, more frequent monitoring, and senior-management approval for onboarding.

Suspicious Activity Reports (SARs) must be filed within 30 days when a transaction of $5,000 or more appears suspicious. The SAR filing system must be confidential — customers must not be informed that a SAR has been filed. Systems should implement automated transaction-monitoring rules that flag anomalies for compliance-officer review.

## Payment Processing Standards

The Payment Card Industry Data Security Standard (PCI-DSS) applies to any system that stores, processes, or transmits cardholder data. The 12 requirements include: maintaining a firewall, not using vendor-supplied default passwords, protecting stored cardholder data (encryption at rest), encrypting data in transit over public networks, using anti-virus software, developing secure systems, restricting access on a need-to-know basis, assigning unique IDs to each person with computer access, restricting physical access, tracking and monitoring all access to network resources and cardholder data, regularly testing security systems, and maintaining an information-security policy.

PCI-DSS requires that the Primary Account Number (PAN) be rendered unreadable anywhere it is stored, using strong cryptography (AES-256 or equivalent). Card verification values (CVV/CVC) must never be stored after authorisation. Tokenisation should replace PANs in non-essential systems.

Real-Time Gross Settlement (RTGS) and instant-payment systems (e.g., UPI in India, Faster Payments in the UK, FedNow in the US) require sub-second processing with guaranteed delivery. Systems must handle idempotency, duplicate detection, and reconciliation within strict SLA windows.

## Data Security and Privacy for Financial Institutions

Financial institutions handle Personally Identifiable Information (PII) and sensitive financial data that must be protected under multiple regulations. The Gramm-Leach-Bliley Act (GLBA) requires institutions to explain their information-sharing practices and to safeguard sensitive data with an information-security programme.

Data classification typically follows a tiered model: Public, Internal, Confidential, and Restricted. Customer financial data (account numbers, balances, transaction history) is Restricted. Access must follow the principle of least privilege, with role-based access controls (RBAC) enforced at the application, database, and infrastructure layers.

Encryption requirements: data at rest must be encrypted using AES-256 or equivalent; data in transit must use TLS 1.2 or higher; key management must follow industry standards (NIST SP 800-57). Hardware Security Modules (HSMs) should be used for cryptographic key storage in production environments.

Audit logging must capture who accessed what data, when, from where, and what action was performed. Logs must be tamper-evident (append-only or write-once storage), retained for a minimum of 7 years for regulatory purposes, and regularly reviewed. Log integrity must be verifiable through cryptographic checksums or blockchain-anchored timestamps.

## Core Banking System Integration

Core banking systems (e.g., Temenos T24, Finacle, FIS) are the central ledger for a bank's operations. Any new module (lending, payments, onboarding) must integrate with the core system via well-defined APIs or middleware (e.g., IBM MQ, Kafka). Integration patterns include synchronous request-response for real-time queries and asynchronous event-driven messaging for batch operations and notifications.

Data consistency between satellite systems and the core requires careful transaction management. The SAGA pattern or two-phase commit protocols may be necessary for distributed transactions. Eventual consistency is acceptable for non-critical reads but not for balance updates or regulatory reporting.

ISO 20022 is the international standard for financial messaging, replacing older formats like SWIFT MT. New systems should support ISO 20022 message formats for interbank communication, payment initiation, and reporting.

## Risk Management in Financial Software

Operational risk management requires systems to have business-continuity and disaster-recovery plans with defined Recovery Time Objectives (RTO) and Recovery Point Objectives (RPO). For critical banking systems, RTO is typically under 4 hours and RPO under 1 hour. Active-active or active-passive failover architectures should be evaluated based on criticality.

Credit risk models used in automated lending must be validated independently before deployment, monitored for drift, and re-validated at least annually. Model Risk Management (MRM) guidelines (SR 11-7 in the US) require documentation of model development, assumptions, limitations, and ongoing performance metrics.

Fraud detection systems must operate in real-time for transaction monitoring and use a combination of rule-based engines and machine-learning models. False-positive rates must be managed to avoid customer friction while maintaining detection effectiveness. Fraud models must be explainable for regulatory examination.

## Digital Banking and Mobile Requirements

Digital banking channels must comply with accessibility standards (WCAG 2.1 AA minimum). Multi-factor authentication (MFA) is required for all customer-facing applications — typically combining something the user knows (password/PIN), something the user has (phone/token), and optionally something the user is (biometrics).

Session management must enforce automatic timeout after inactivity (typically 5-15 minutes for banking apps), secure token rotation, and device fingerprinting for anomaly detection. OAuth 2.0 with PKCE is the recommended authorisation framework for mobile banking APIs.

Mobile banking apps must support offline capabilities where appropriate (e.g., viewing cached balances), with clear indicators of data freshness. Push notifications for transaction alerts must be delivered in near-real-time and must not contain sensitive information (e.g., show partial account numbers only).
