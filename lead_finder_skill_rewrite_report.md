# Lead Finder skill rewrite: Delhi NCR AI/software shortlist

## Recommended operating rule

The skill should return only companies supported by public evidence for an AI/software product, an NCR location or presence, and a public employee-size signal. It should preserve uncertainty instead of converting a company-size band into an exact headcount, distinguish qualified rows from supported-but-not-explicitly-SaaS rows, disclose conflicts, and stop at the supported count rather than padding the list.[1][2][3][4][5][6][7][8][9][10]

## Draft report

Ten publicly supported NCR-area rows are available. Six have the strongest combined product and location evidence: Rezo.ai, Makunai Global, Fleetx, GobbleCube, Almonds AI, and FutureCraft AI. InteligenAI and DHI AI are supported AI software/services companies, but SaaS-product status is not explicit. LOQO AI has an AI product, but its SaaS distribution model is not explicit. SAAS AI LABS has strong AI-SaaS and employee evidence, while its NCR signal is based on a public Noida hiring-location reference rather than a headquarters field.[1][2][3][4][5][6][7][8][9][10]

| # | Company | AI/software product evidence | NCR evidence | Employee evidence | Qualification |
|---:|---|---|---|---|---|
| 1 | Rezo.ai | Unified Agentic AI platform for enterprise customer experience; conversational automation, voicebots, speech-to-text and QA. [1] | Headquarters: Noida, Uttar Pradesh. [1] | LinkedIn company size: 51–200 employees. [1] | Qualified |
| 2 | Makunai Global | Cloud CRM and AI platform covering CRM, WhatsApp automation, IVR, workflow automation and B2C commerce. [2] | Headquarters: Noida, Uttar Pradesh. [2] | LinkedIn company size: 51–200 employees; a third-party profile reports 78 employees. [2] | Qualified |
| 3 | InteligenAI | Production-ready enterprise AI systems, LLM applications, agentic systems, RAG and process automation. [3] | Headquarters: Gurgaon, Haryana. [3] | LinkedIn company size: 11–50 employees. [3] | Supported AI software company; SaaS-product status is not explicit |
| 4 | Fleetx | AI-powered fleet and logistics optimization platform covering fleet management, machine learning, real-time visibility and supply-chain control towers. [4] | Headquarters: Gurgaon, Haryana. [4] | LinkedIn company size: 201–500 employees. [4] | Qualified |
| 5 | GobbleCube | Agentic growth operating system for brands, with AI, SaaS, quick-commerce, performance-marketing automation and marketplace optimization capabilities. [5] | Headquarters: Gurugram, Haryana. [5] | LinkedIn company size: 51–200 employees. [5] | Qualified |
| 6 | Almonds AI | AI-driven omnichannel engagement platform providing automation, smart conversations and customer-experience solutions. [6] | Headquarters: Gurgaon, Haryana. [6] | Reported workforce: 215; LinkedIn company-size band: 51–200 employees. [6] | Qualified; headcount-band inconsistency disclosed |
| 7 | SAAS AI LABS | AI-powered SaaS products including GPT-based institutional chat windows, AI chatbots, LLM applications and intelligent automation. [7] | Public workforce data indicates India-based operations; a public company profile reports a Noida on-site hiring post. [7] | Reported workforce: 11; company-size band: 11–50 employees. [7] | Qualified; NCR evidence is based on public hiring-location evidence |
| 8 | DHI AI | AI/digital technology company providing application development, managed services and advisory solutions across multiple industries. [8] | Headquarters: New Delhi, Delhi; additional presence in Mumbai, Kolkata, Bengaluru and Pune. [8] | Reported team size: 16 employees. [8] | Supported AI software/services company; SaaS-product status is not explicit |
| 9 | LOQO AI | AI content company using proprietary Cinematography AI to produce long-form shows from ideation through final video output. [9] | Headquarters: Delhi, Delhi. [9] | Reported team size: approximately 8 employees. [9] | Supported AI product company; SaaS distribution model is not explicit |
| 10 | FutureCraft AI | Brand-first AI content ideation and creation product; publicly described as a bootstrapped SaaS startup. [10] | Headquarters: Delhi, Delhi. [10] | Reported team size: approximately 3 employees. [10] | Qualified |

## Screening notes

- Do not claim that all rows meet a 100–1,000 employee requirement: the supplied employee evidence includes bands and estimates below 100, and several figures are global rather than NCR-only. [1][2][3][4][5][6][7][8][9][10]
- Keep “qualified” separate from “supported”: InteligenAI, DHI AI and LOQO AI should remain visible only when the user accepts non-explicit SaaS evidence. [3][8][9]
- Preserve the Almonds AI discrepancy (reported workforce 215 versus LinkedIn band 51–200) rather than choosing one figure silently. [6]
- Treat SAAS AI LABS as NCR-presence evidence, not as a verified NCR headquarters claim. [7]
- A future research loop should seek a second source for each critical field before outreach, but it must not replace a missing field with inference.

## Implementation behavior for the rewritten skill

1. Parse geography, product type, employee constraints, target count, output format and delivery intent before research.
2. Search in batches, register each retrieved URL immediately, and attach the returned citation ID to the evidence record before drafting.
3. Require separate product, NCR and employee evidence fields; allow a row only when the requested qualification policy is met.
4. Route ambiguous rows to a disclosed “supported, qualification caveat” state instead of silently qualifying them.
5. Return the supported rows, an explicit shortfall, inline citations, and a mechanically rendered Sources section.
6. Verify that every factual row has a registered source and that no citation ID or URL was invented.

Employee figures are public estimates and may represent global rather than NCR-only headcount. Verify current company status and contact details before outreach.

## Sources

[1] https://www.linkedin.com/company/rezo-ai
[2] https://www.linkedin.com/company/makunai-global
[3] https://www.linkedin.com/company/inteligenai?originalSubdomain=in
[4] https://www.linkedin.com/company/fleetx-ai
[5] https://www.linkedin.com/company/gobblecube
[6] https://linkedin.com/company/almondsai
[7] https://in.linkedin.com/company/saas-ai-labs
[8] https://getlatka.com/companies/dhi-digital.com
[9] https://getlatka.com/companies/loqo.ai
[10] https://getlatka.com/companies/futurecraft.ing
