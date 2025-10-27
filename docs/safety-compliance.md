# Safety & Compliance Guidelines for Minors in India

This platform supports sensitive mental health interactions. When handling conversations that involve users who may be minors located in India, the following regulatory and ethical safeguards must be enforced.

## Regulatory Landscape

- **Protection of Children from Sexual Offences (POCSO) Act, 2012** – Mandates prompt reporting of abuse, storage of evidence, and cooperation with child protection authorities.
- **Juvenile Justice (Care and Protection of Children) Act, 2015** – Defines procedures for child welfare committees and rehabilitation services; relevant when escalation is required.
- **Mental Healthcare Act, 2017** – Grants minors (with guardian involvement) the right to access mental healthcare and outlines confidentiality exceptions for harm prevention.
- **Information Technology (Intermediary Guidelines and Digital Media Ethics Code) Rules, 2021** – Requires platforms to implement due diligence, content moderation, and data retention policies.
- **National Commission for Protection of Child Rights (NCPCR) Guidelines** – Provide best practices for helplines, consent, and psychosocial support.

## Operational Safeguards

1. **Age & Location Determination**
   - Capture self-declared age and location via the frontend prior to initiating a session.
   - Flag sessions as "minor" when age < 18 or location metadata indicates India.

2. **Guardian Consent Workflow**
   - Require guardian/parental consent for non-crisis conversations with minors.
   - Log consent artifacts (timestamp, guardian name, consent method) in a secure audit store.

3. **Mandatory Human Oversight**
   - Route all high-risk topics (self-harm, abuse, exploitation) to an on-call clinician within defined SLAs (<5 minutes for imminent risk).
   - Display on-screen messaging clarifying that the AI assistant is not a substitute for emergency services.

4. **Escalation & Reporting**
   - Implement automated alerts for keywords or graph-derived risk scores that fall under POCSO mandatory reporting clauses.
   - Maintain contact details for local Child Welfare Committees and law enforcement and document hand-off procedures.

5. **Data Protection**
   - Store transcripts encrypted at rest; restrict access via RBAC.
   - Retain chat histories for the minimum legally required period (currently 180 days under IT Rules) unless advised otherwise by counsel.
   - Anonymise or delete personal data upon request, subject to legal holds.

6. **Content Guardrails**
   - Enforce policy-aware prompts in `TherapeuticChatHandler` that forbid the model from giving diagnostic labels, prescribing medication, or discouraging professional help.
   - Use Graph RAG policies to prioritise child protection guidance and helpline information (e.g. CHILDLINE 1098).

7. **Audit & Accountability**
   - Log every intervention (retrieval context, response, escalation status) with trace IDs.
   - Perform quarterly compliance reviews against NCPCR guidelines and document remediation.

## Staff Training & Access Control

- Provide regular POCSO and safeguarding training for all staff with access to transcripts or escalation workflows.
- Use least-privilege access; only vetted clinicians should view identifiable minor data.
- Implement background checks and confidentiality agreements for support personnel.

## Incident Response

- Maintain an incident playbook covering abuse disclosures, imminent harm, and technical failures that could jeopardise minors.
- Ensure 24/7 availability of escalation contacts; integrate monitoring alerts with on-call platforms (PagerDuty, Opsgenie, etc.).
- Report notable incidents to internal compliance officers within 24 hours and document follow-up actions.

Failure to comply can result in legal penalties and revocation of service permissions. Always consult legal counsel when updating policies or expanding to new jurisdictions.
