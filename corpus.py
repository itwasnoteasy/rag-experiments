"""
Synthetic telecom/insurance knowledge base — 20 documents.
Each document has: id, title, category, content.
"""

CORPUS = [
    # ─── Device Insurance (10 docs) ───────────────────────────────────────────

    {
        "id": "DI-001",
        "title": "How to File a Device Insurance Claim",
        "category": "device_insurance",
        "content": (
            "Filing a device insurance claim is straightforward. Log in to your account portal "
            "at my.carrier.com, navigate to 'Insurance & Protection', and select 'File a Claim'. "
            "You will need your device's IMEI number, the date the damage or loss occurred, and a "
            "brief description of the incident. Claims must be filed within 60 days of the incident. "
            "Once submitted, a claims adjuster will review your request within 1–2 business days. "
            "Approved claims for repairs are fulfilled through our network of certified repair centers, "
            "while replacement devices are shipped within 3–5 business days. Keep the confirmation "
            "email you receive after filing; it contains your claim reference number (format: CLM-XXXXXXXX) "
            "needed for any follow-up inquiries. Claims can also be initiated by calling 1-800-555-0147."
        ),
    },
    {
        "id": "DI-002",
        "title": "Device Insurance Deductible Amounts by Plan Tier",
        "category": "device_insurance",
        "content": (
            "Deductible amounts for device insurance vary by plan tier and device category. Under the "
            "Basic Protection Plan, deductibles are $29 for standard phones, $49 for premium smartphones, "
            "and $99 for tablets. The Enhanced Protection Plan reduces these to $0 for standard phones, "
            "$29 for premium smartphones, and $49 for tablets. The Total Coverage Plan offers a flat $0 "
            "deductible across all device categories. Deductibles are charged per approved claim and must "
            "be paid at the time of claim approval via credit card or account credit. Note that deductible "
            "amounts are subject to annual review and may change; customers on legacy plans (pre-2022) may "
            "have different deductible schedules. Contact customer support or check policy document "
            "INS-POL-2024 for legacy plan details."
        ),
    },
    {
        "id": "DI-003",
        "title": "Coverage Exclusions Under Device Insurance",
        "category": "device_insurance",
        "content": (
            "Not all damage or loss scenarios are covered under our device insurance plans. Common "
            "exclusions include: intentional damage or abuse, cosmetic damage that does not affect device "
            "functionality (scratches, dents), damage resulting from unauthorized modifications or "
            "jailbreaking, loss due to mysterious disappearance without a police report, damage from "
            "extreme temperature exposure beyond manufacturer specifications, and normal wear and tear "
            "including battery degradation below 80% capacity. Pre-existing conditions identified at "
            "enrollment are also excluded. Coverage does not apply if the device is used for commercial "
            "resale purposes. Business accounts under our SMB plans are subject to separate commercial "
            "device coverage terms outlined in addendum COM-2024. When in doubt, review the full exclusions "
            "list in your policy documentation before filing a claim to avoid denial fees."
        ),
    },
    {
        "id": "DI-004",
        "title": "Device Swap Process After Claim Approval",
        "category": "device_insurance",
        "content": (
            "Once your insurance claim is approved and a device swap is selected as the resolution, "
            "the process works as follows. A certified refurbished or new device of the same make and "
            "model — or a comparable substitute if unavailable — will be shipped to your registered "
            "address. Shipping typically takes 3–5 business days via standard delivery or 1–2 days with "
            "expedited shipping (additional fee applies). You must return your damaged device within "
            "10 business days using the prepaid return label included in the swap package. Failure to "
            "return the damaged device results in a non-return fee of up to $200 charged to your account. "
            "Upon receiving the replacement, transfer your SIM card and restore your data from a backup. "
            "The replacement device carries a 90-day warranty covering manufacturing defects. If you "
            "experience issues with the replacement within 90 days, contact claims support referencing "
            "your original claim number."
        ),
    },
    {
        "id": "DI-005",
        "title": "Water Damage Policy and What's Covered",
        "category": "device_insurance",
        "content": (
            "Water and liquid damage is one of the most common insurance claims we receive. Our "
            "device insurance plans cover accidental liquid immersion, including drops in sinks, "
            "pools, and puddles. However, coverage is subject to conditions: the damage must be "
            "accidental, not resulting from intentional submersion beyond the device's rated IP "
            "water-resistance level (e.g., taking a non-waterproof phone scuba diving). Saltwater "
            "corrosion damage may require additional documentation. When filing a water damage claim, "
            "do not attempt to charge the device as this can cause additional board damage. Place the "
            "device in a dry environment and file your claim within 48 hours for best results. Our "
            "certified repair centers can assess water damage and attempt data recovery as part of the "
            "claim process. Note that data recovery is not guaranteed and is not a covered service "
            "under Basic Protection Plan."
        ),
    },
    {
        "id": "DI-006",
        "title": "Enrolling in Device Insurance After Purchase",
        "category": "device_insurance",
        "content": (
            "Customers can enroll in device insurance within 30 days of purchasing a new device. "
            "After this window, enrollment is only possible during the annual open enrollment period "
            "in October, subject to a device inspection. To enroll, visit the Insurance section in "
            "your account portal or ask a representative at any retail location. You'll need to "
            "confirm your device's IMEI and agree to the plan terms. Monthly premiums range from "
            "$5.99 for the Basic Protection Plan to $17.99 for Total Coverage. SMB account holders "
            "managing multiple lines can enroll devices in bulk through the admin portal and may "
            "qualify for volume discounts on premiums. Insurance charges appear as a separate line "
            "item on your monthly bill labeled 'Device Protection Plan'. You can cancel insurance "
            "at any time, but mid-cycle cancellations are not prorated."
        ),
    },
    {
        "id": "DI-007",
        "title": "Claim Limits and Claim Frequency Policy",
        "category": "device_insurance",
        "content": (
            "Device insurance plans have claim frequency limits to prevent abuse. Under the Basic "
            "Protection Plan, you may file a maximum of 2 claims per 12-month rolling period. The "
            "Enhanced Protection Plan allows up to 3 claims per 12-month period. The Total Coverage "
            "Plan permits unlimited claims, though this is subject to review for patterns suggesting "
            "fraud. Each approved claim counts toward your limit regardless of claim outcome (repair "
            "vs. replacement). Denied claims do not count against your limit. If you reach your claim "
            "limit, you must wait until the rolling 12-month window resets before filing again. "
            "Business accounts under SMB plans have per-line claim limits that mirror the individual "
            "plans selected for each line. Claim history is visible in your account portal under "
            "'Insurance History' and resets on the anniversary of your first claim within that period."
        ),
    },
    {
        "id": "DI-008",
        "title": "Third-Party Repair and Insurance Implications",
        "category": "device_insurance",
        "content": (
            "Using a third-party repair service for your device can affect your insurance coverage. "
            "If a device is repaired by an unauthorized technician and subsequently submitted for an "
            "insurance claim, our adjusters will assess whether the prior repair contributed to the "
            "current damage. In cases where third-party repairs caused or worsened the issue, the "
            "claim may be partially or fully denied. To maintain full coverage, always use our "
            "network of certified repair centers for out-of-warranty repairs. A list of certified "
            "centers is available in the app under 'Find a Repair Center'. If you're unsure whether "
            "a repair will affect coverage, call our insurance advisory line at 1-800-555-0199 "
            "before proceeding. SMB customers managing a fleet of devices should note that bulk "
            "repair agreements with third parties may void device insurance on those units unless "
            "a formal vendor exception is filed with the account team."
        ),
    },
    {
        "id": "DI-009",
        "title": "Insurance Appeals Process for Denied Claims",
        "category": "device_insurance",
        "content": (
            "If your device insurance claim is denied, you have the right to appeal the decision. "
            "Appeals must be submitted within 30 days of the denial notice. To appeal, log in to "
            "your account portal, navigate to 'Insurance History', find the denied claim, and select "
            "'Appeal This Decision'. You will be prompted to provide additional documentation or "
            "a written explanation. Common grounds for successful appeals include: new evidence "
            "not submitted with the original claim, incorrect categorization of damage type by the "
            "adjuster, or documentation errors. An independent review team will assess appeals within "
            "5–7 business days. If the appeal is upheld, your claim will be processed at no additional "
            "cost. If denied again, you may escalate to our Ombudsman review process by emailing "
            "insurance-appeals@carrier.com. Note that the Ombudsman process can take up to 30 days "
            "and is the final internal review step."
        ),
    },
    {
        "id": "DI-010",
        "title": "Temporary Loaner Device During Claim Processing",
        "category": "device_insurance",
        "content": (
            "Customers with Enhanced Protection or Total Coverage plans are eligible for a temporary "
            "loaner device while their claim is being processed. Loaners are available at participating "
            "retail locations and must be requested within 24 hours of filing your claim. A valid "
            "government-issued ID and your claim reference number are required to pick up a loaner. "
            "Loaner devices are standard smartphones capable of basic calling, messaging, and data. "
            "They are preconfigured with a temporary SIM tied to your existing number for seamless "
            "continuity. The loaner must be returned within 3 business days after your replacement "
            "device is delivered or your original device is repaired. Late returns incur a $15/day "
            "fee. Loaner devices are not available for tablet claims. Basic Protection Plan subscribers "
            "are not eligible for loaners but may purchase a temporary device at a discounted rate "
            "through the insurance portal during claim processing."
        ),
    },

    # ─── SMB Onboarding (10 docs) ─────────────────────────────────────────────

    {
        "id": "SMB-001",
        "title": "SMB Account Setup: Getting Started",
        "category": "smb_onboarding",
        "content": (
            "Setting up a Small and Medium Business (SMB) account gives your organization access to "
            "dedicated business plans, priority support, and the admin portal for centralized account "
            "management. To get started, visit business.carrier.com or visit a business-focused retail "
            "location. You will need: a valid business license or EIN, a primary contact's government "
            "ID, a business bank account or credit card for billing, and the name and number of lines "
            "you wish to activate. The setup process typically takes 30–60 minutes for accounts with "
            "fewer than 10 lines. Larger accounts (10+ lines) are assigned a dedicated business "
            "account manager who will guide setup over a scheduled call. Once your account is created, "
            "login credentials for the admin portal are emailed to the primary account holder within "
            "2 business days. SMB accounts receive a consolidated monthly bill covering all lines, "
            "devices, and add-on services."
        ),
    },
    {
        "id": "SMB-002",
        "title": "Number Porting for Business Accounts",
        "category": "smb_onboarding",
        "content": (
            "Porting your existing business phone numbers to our network preserves continuity for "
            "your clients and staff. To port numbers, submit a Port Authorization Code (PAC) or "
            "account transfer PIN from your current carrier for each number you wish to transfer. "
            "Bulk number porting for SMB accounts allows porting up to 100 numbers simultaneously "
            "via a CSV upload in the admin portal under 'Number Management > Port Numbers'. Standard "
            "porting takes 5–7 business days; expedited porting (2–3 business days) is available for "
            "an additional fee. During the porting window, calls to ported numbers are automatically "
            "forwarded to a temporary number we provide. Toll-free number porting requires a separate "
            "authorization form (TF-PORT-2024) and may take up to 10 business days. Confirm that your "
            "contract with the previous carrier allows porting without early termination fees before "
            "initiating the process."
        ),
    },
    {
        "id": "SMB-003",
        "title": "Choosing the Right Business Plan",
        "category": "smb_onboarding",
        "content": (
            "Selecting the right plan for your business depends on your team's data needs, travel "
            "patterns, and budget. Our SMB plan tiers are: Business Starter (5GB data/line, domestic "
            "only, $25/line/month), Business Pro (unlimited domestic data + 10GB international roaming, "
            "$40/line/month), and Business Elite (unlimited domestic + international, priority network "
            "access, dedicated support, $60/line/month). All plans include unlimited talk and text "
            "within the domestic network. Volume discounts apply for accounts with 5+ lines: 10% off "
            "for 5–9 lines, 15% off for 10–24 lines, and 20% off for 25+ lines. Plans can be mixed "
            "within an account — for example, field staff on Business Pro and office-only staff on "
            "Business Starter. Plan changes take effect at the start of the next billing cycle. "
            "Contact your account manager to model cost scenarios before committing."
        ),
    },
    {
        "id": "SMB-004",
        "title": "Understanding SMB Billing Cycles",
        "category": "smb_onboarding",
        "content": (
            "SMB accounts operate on a monthly billing cycle aligned to your account activation date. "
            "For example, if your account was activated on the 15th, your billing cycle runs from the "
            "15th to the 14th of the following month. Invoices are generated on the first day of each "
            "new billing cycle and are available in the admin portal under 'Billing > Invoices' within "
            "24 hours. Payment is due within 30 days of the invoice date. Auto-pay can be enabled using "
            "a credit card or ACH bank transfer, with a 1% discount applied to auto-pay accounts. "
            "Mid-cycle additions (new lines, devices, or add-ons) are prorated to the remaining days "
            "in the current cycle. Usage overages — such as exceeding data caps on Business Starter — "
            "are billed at $10 per additional GB and appear on the following invoice. Download CSV "
            "exports of usage data per line for expense reporting through the admin portal."
        ),
    },
    {
        "id": "SMB-005",
        "title": "Admin Portal: Features and Access Management",
        "category": "smb_onboarding",
        "content": (
            "The SMB admin portal at business.carrier.com/portal is the central hub for managing "
            "your organization's account. Key features include: user and line management (add, "
            "suspend, or remove lines), device inventory tracking, usage dashboards per line and "
            "department, billing and invoice access, number porting workflows, and insurance "
            "enrollment for all devices. Role-based access control (RBAC) allows the primary account "
            "holder to create sub-administrator accounts with limited permissions — for example, a "
            "finance admin can view invoices without modifying lines. Two-factor authentication (2FA) "
            "is mandatory for all admin portal logins. If a sub-admin loses access, the primary "
            "account holder can reset credentials under 'User Management > Admin Accounts'. For "
            "enterprise-tier customers (50+ lines), an API is available for integrating portal data "
            "with internal HR or expense management systems. Documentation is at "
            "developer.carrier.com/smb-api."
        ),
    },
    {
        "id": "SMB-006",
        "title": "Activating New Lines for Existing SMB Accounts",
        "category": "smb_onboarding",
        "content": (
            "Adding new lines to an existing SMB account is done through the admin portal or by "
            "contacting your dedicated account manager. In the portal, navigate to 'Lines > Add New "
            "Line', select the plan tier, choose whether to port a number or get a new number, and "
            "assign the line to an employee profile. New lines are typically activated within 4 hours "
            "during business hours (Monday–Friday, 8AM–8PM ET). Same-day activation is not guaranteed "
            "for lines added after 6PM ET. If you need a physical SIM card, one will be shipped within "
            "2 business days. eSIM activation is immediate and recommended for faster onboarding. "
            "Newly activated lines are prorated on the first invoice. SMB accounts with 25+ lines "
            "can request bulk activations (up to 50 lines at once) by submitting a bulk activation "
            "form to their account manager at least 3 business days in advance."
        ),
    },
    {
        "id": "SMB-007",
        "title": "Business Account Security: SIM Swap and Fraud Prevention",
        "category": "smb_onboarding",
        "content": (
            "Protecting your business account from fraud is a shared responsibility. We implement "
            "several safeguards: SIM swap requests for business lines require approval from the "
            "primary account holder via a verification call to the registered business number, not "
            "employee personal numbers. Admin portal sessions time out after 15 minutes of inactivity. "
            "Suspicious activity alerts are sent to the primary email on file when changes such as "
            "new admin creation or bulk line modifications occur outside business hours. We recommend "
            "setting a Business Account PIN (different from your personal account PIN) to authorize "
            "sensitive changes. In the event of suspected account takeover, call our Business Security "
            "hotline at 1-800-555-0200 immediately — available 24/7. Do not attempt to make account "
            "changes via regular customer service when a takeover is suspected, as this may delay "
            "the incident response process."
        ),
    },
    {
        "id": "SMB-008",
        "title": "Onboarding Checklist for New SMB Customers",
        "category": "smb_onboarding",
        "content": (
            "To ensure a smooth onboarding experience, follow this checklist after your SMB account "
            "is created. Step 1: Log in to the admin portal and change the temporary password. "
            "Step 2: Enable two-factor authentication for all admin accounts. Step 3: Add employee "
            "profiles under 'User Management' and assign lines. Step 4: Upload your device inventory "
            "if bringing existing devices to the network (BYOD). Step 5: Configure billing preferences "
            "and enable auto-pay if desired. Step 6: Enroll devices in insurance plans through "
            "'Insurance & Protection'. Step 7: Set up usage alerts to notify admins when a line "
            "approaches its data cap. Step 8: Review and configure call forwarding and voicemail "
            "settings for each line. Step 9: Download the Business Mobile app for on-the-go account "
            "management. Step 10: Schedule a 30-minute onboarding call with your account manager to "
            "review the setup and answer questions."
        ),
    },
    {
        "id": "SMB-009",
        "title": "Porting Numbers Away: Leaving Our Network",
        "category": "smb_onboarding",
        "content": (
            "If your business decides to move to another carrier, you have the right to port your "
            "numbers away. To initiate an outbound port, your new carrier will request a PAC code "
            "or account number and transfer PIN from you. Your SMB account transfer PIN is available "
            "in the admin portal under 'Account Settings > Security > Transfer PIN'. Generating a "
            "new PIN does not automatically initiate a port — the new carrier must use it within "
            "30 days. Outbound porting does not terminate your account automatically; you must "
            "separately cancel your service to avoid continued billing. Note that porting numbers "
            "away while under contract may trigger early termination fees (ETFs) as specified in "
            "your service agreement. Device installment plans are not affected by number porting "
            "and must be paid in full or continued per the installment schedule regardless of "
            "whether you remain a customer."
        ),
    },
    {
        "id": "SMB-010",
        "title": "International Roaming for SMB Accounts",
        "category": "smb_onboarding",
        "content": (
            "Business Pro and Business Elite plan subscribers can use their devices internationally "
            "with automatic roaming in 180+ countries. Roaming must be enabled per line in the admin "
            "portal under 'Line Settings > International Roaming' before travel. Business Starter "
            "plan lines do not include international roaming; temporary international add-ons ($10/day "
            "unlimited or $5/day + $0.02/MB) can be purchased through the portal. For extended "
            "international assignments (30+ days), consider requesting a temporary local SIM through "
            "our international partner program to reduce costs. Roaming usage is tracked in real time "
            "in the admin dashboard. Set per-line roaming caps to prevent unexpected overages — once "
            "the cap is reached, data is suspended until the admin raises or removes the cap. Voice "
            "calls while roaming are billed per minute unless on Business Elite, which includes "
            "unlimited international calling to 40 countries."
        ),
    },
]

if __name__ == "__main__":
    from tabulate import tabulate

    rows = [(doc["id"], doc["category"], doc["title"][:60]) for doc in CORPUS]
    print(tabulate(rows, headers=["ID", "Category", "Title"], tablefmt="rounded_outline"))
    print(f"\nCorpus loaded: {len(CORPUS)} documents")
