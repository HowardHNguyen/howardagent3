# Marketing Operations AI Assistant (Agentic Prototype)

## Overview

This application is a **Marketing Operations AI Assistant (Agentic Prototype)** designed to support Marketing teams by providing fast, consistent, and reliable answers grounded in **internal documentation, standards, and operating models**.

The app demonstrates how **agentic AI** can be applied practically within a MarTech context—helping teams clarify definitions, summarize guidance, and draft internal-ready artifacts—while remaining aligned with enterprise governance and compliance expectations.

> This is an exploratory, internal prototype intended to demonstrate feasibility and inform a broader Agentic AI strategy for Marketing.

---

## What this app is (and is not)

### What it *is*
- A **Retrieval-Augmented Generation (RAG)** application
- Grounded entirely in **user-provided internal documents**
- Designed to reduce knowledge friction and enforce consistency
- Built to demonstrate **rapid prototyping → production evolution**

### What it *is not*
- Not a replacement for Microsoft Copilot
- Not trained on proprietary company data
- Not a general-purpose chatbot
- Not a production system (yet)

---

## Key capabilities

- Upload internal documents (PDF, DOCX, TXT)
- Parse and index content into a searchable knowledge base
- Ask natural-language questions grounded in uploaded content
- Generate clear, actionable responses aligned with internal standards
- Demonstrate an **agentic loop**:
  
  **Retrieve → Reason → Respond**

---

## How this complements Microsoft Copilot

Microsoft Copilot excels at:
- General enterprise productivity
- Cross-tool assistance (Outlook, Teams, PowerPoint, Excel)
- Broad language understanding

This assistant focuses on:
- Marketing-specific standards and definitions
- Internal operating models and workflows
- Consistency, governance, and domain-aware reasoning

Together, they form a layered approach:
- **Microsoft Copilot** → general enterprise productivity  
- **Marketing Operations AI Assistant** → domain-specific enablement

---

## High-level architecture

The application is structured into four logical layers:

1. **User Interface**
   - Streamlit-based chat and document management UI

2. **Knowledge Ingestion**
   - Document parsing (PDF, DOCX, TXT)
   - Text chunking and normalization
   - Embedding generation

3. **Agentic RAG Pipeline**
   - Retrieval of relevant document chunks
   - LLM reasoning using retrieved context
   - Response generation aligned to internal content

4. **Model Layer**
   - LLM for natural language reasoning
   - Embeddings model for semantic search

See the **Architecture & Technical Details** tab in the app for interactive data flow diagrams.

---

## Data handling & storage (important)

### Current prototype behavior
- Uploaded files are held in **session memory**
- Files are written briefly to a **temporary filesystem** for parsing
- Temporary files are deleted immediately after processing
- Parsed text and embeddings are stored **in memory only**
- No data is persisted across sessions or app restarts

> When the app restarts, all uploaded content is discarded.

This design intentionally keeps the prototype **low-risk** and avoids unintended data retention.

---

## Production evolution (future considerations)

A production-ready version would explicitly introduce:
- Persistent vector storage (e.g., Azure AI Search, FAISS, Pinecone)
- Enterprise authentication (SSO / RBAC)
- Source citations and traceability
- Audit logs and observability
- Scheduled ingestion from systems such as SharePoint or Confluence
- Governance controls aligned with HIPAA / PHI / PII policies

These are **intentional design choices**, not omissions.

---

## Running the app locally

- pip install -r requirements.txt
- streamlit run streamlit_app.py

## Required environment variables

Set via environment variables or Streamlit secrets:
- OPENAI_API_KEY
- GROQ_API_KEY

## Intended use

This app is intended for:
- Demonstrating agentic AI concepts
- Rapid prototyping and experimentation
- Stakeholder review and feedback
- Informing AI/ML roadmap decisions

It is not intended for production use without additional governance, security, and integration work.

---

## License & usage note

© 2025 Howard Nguyen, PhD. Prototype provided for demonstration and internal evaluation purposes only.
