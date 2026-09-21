"""Streamlit app
Run:
  streamlit run streamlit_app.py
"""

import streamlit as st

try:
    from langchain_core.messages import HumanMessage
except ImportError:
    from langchain.schema import HumanMessage

from document_loader import DocumentLoader
from rag import graph, config, retriever
from retriever import EmbeddingServiceError


# =========================
# Page config
# =========================
st.set_page_config(page_title="Agentic AI Knowledge Platform (RAG + Workflows)", layout="wide")
st.title("📚 Agentic AI Knowledge Platform (RAG + Workflows)")

# =========================
# Session state
# =========================
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "uploaded_files" not in st.session_state:
    st.session_state.uploaded_files = []
if "rag_ready" not in st.session_state:
    st.session_state.rag_ready = False

# =========================
# Tabs
# =========================
tab_chat, tab_about, tab_howto, tab_arch = st.tabs(
    ["💬 Chat", "ℹ️ About this app", "🧭 How to use / guideline", "🏗️ Architecture & Technical Details"]
)

# =========================
# TAB 1: CHAT
# =========================
with tab_chat:
    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("Chat Interface")

        for msg in st.session_state.chat_history:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

        user_input = st.chat_input("Ask a question about your documents...")

        if user_input:
            st.session_state.chat_history.append({"role": "user", "content": user_input})
            with st.chat_message("user"):
                st.markdown(user_input)

            if not st.session_state.rag_ready:
                reply = "Please upload documents first, then click **Build Knowledge Base**."
                st.session_state.chat_history.append({"role": "assistant", "content": reply})
                with st.chat_message("assistant"):
                    st.markdown(reply)
            else:
                with st.chat_message("assistant"):
                    try:
                        with st.spinner("Thinking..."):
                            result = graph.invoke(
                                {"messages": [HumanMessage(content=user_input)]},
                                config=config,
                            )
                            answer = result["messages"][-1].content
                            st.markdown(answer)
                        st.session_state.chat_history.append({"role": "assistant", "content": answer})
                    except EmbeddingServiceError as e:
                        answer = f"⚠️ Could not answer your question: {e}"
                        st.error(answer)
                        st.session_state.chat_history.append({"role": "assistant", "content": answer})

    with col2:
        st.subheader("Document Management")

        uploads = st.file_uploader(
            "Upload Documents",
            type=list(DocumentLoader.supported_extensions),
            accept_multiple_files=True,
        )

        if uploads:
            existing = {f.name for f in st.session_state.uploaded_files}
            for f in uploads:
                if f.name not in existing:
                    st.session_state.uploaded_files.append(f)

        st.write("### Uploaded Files")
        if st.session_state.uploaded_files:
            for f in st.session_state.uploaded_files:
                st.write(f"- {f.name}")
        else:
            st.info("No files uploaded yet.")

        if st.button("Build Knowledge Base"):
            if not st.session_state.uploaded_files:
                st.warning("Please upload documents first.")
            else:
                try:
                    with st.spinner("Indexing documents..."):
                        retriever.add_documents_from_uploads(st.session_state.uploaded_files)
                        st.session_state.rag_ready = True
                    st.success("Knowledge base ready.")
                except EmbeddingServiceError as e:
                    st.error(f"⚠️ Could not build the knowledge base: {e}")

        if st.button("Clear Session"):
            st.session_state.chat_history = []
            st.session_state.uploaded_files = []
            st.session_state.rag_ready = False
            st.success("Session cleared.")

# =========================
# TAB 2: ABOUT
# =========================
with tab_about:
    st.header("About this app")

    st.markdown(
        """
This application is an **Agentic AI Knowledge Platform** designed to demonstrate how large language models (LLMs) can be safely and effectively embedded into enterprise environments using **retrieval-augmented generation (RAG)** and **human-in-the-loop workflows**.

Rather than functioning as a generic chatbot, the platform grounds AI reasoning in **trusted internal knowledge sources**, enabling consistent, explainable, and auditable outputs aligned with organizational standards.


### What this platform demonstrates

This prototype showcases a reusable **agentic AI architecture** that can be applied across business domains.  
In the current implementation, **Marketing Operations** is used as a reference use case, but the underlying design is **domain-agnostic**.

Core capabilities demonstrated include:
- Retrieval of relevant enterprise knowledge prior to reasoning (RAG)
- Context-aware responses constrained to approved sources
- Clear separation between ephemeral processing and persistent knowledge layers
- A foundation for evolving AI from **answers** to **assisted workflows**


### Why this matters for enterprises

Organizations increasingly face:
- Knowledge fragmentation across documents, wikis, and platforms
- Inconsistent definitions of metrics, standards, and processes
- Low trust in AI due to hallucinations and lack of traceability
- Difficulty moving from AI experiments to operational value

This platform addresses these challenges by positioning AI as a **knowledge interpreter and workflow assistant**, not an autonomous decision maker.


### How this differs from general AI assistants

This platform is **not a replacement for tools like Microsoft Copilot**.

Instead:
- General AI assistants focus on broad productivity and cross-application tasks
- This platform focuses on **domain-specific knowledge, governance, and operational consistency**

Together, they form a layered AI strategy:
- **General AI assistants** → productivity and communication  
- **Agentic AI Knowledge Platform** → trusted reasoning over enterprise knowledge


### What the platform does today

- Ingests internal documents (PDF, DOCX, TXT)
- Indexes content for semantic retrieval
- Answers questions grounded strictly in provided knowledge
- Demonstrates the **Retrieve → Reason → Respond** agentic loop
- Operates with session-scoped data for safety and experimentation


### How this evolves beyond the prototype

The architecture is intentionally designed to evolve into a production-grade platform with:
- Persistent vector storage and metadata management
- Enterprise authentication and role-based access control
- Audit logging and usage observability
- Integration with systems such as SharePoint, Confluence, and MarTech tools
- Agentic workflows that assist with drafting, validation, and execution — always with human approval


### Important note

This is an **exploratory prototype** intended to demonstrate architectural patterns, validate agentic AI concepts, and support stakeholder review.

It is evaluated as an **internal enablement platform**, not a consumer chatbot.  
Production deployment would include additional governance, security, and compliance controls aligned with enterprise requirements.
        """
    )

# =========================
# TAB 3: HOW TO USE
# =========================
with tab_howto:
    st.header("How to use / guideline")

    st.markdown(
        """
This platform is designed to support both **knowledge discovery** and **workflow acceleration**.


### Step 1 — Provide trusted knowledge

Upload internal documents such as:
- Guidelines and standards
- Playbooks and SOPs
- Frameworks and reference materials

These documents become the **source of truth** the AI reasons over.


### Step 2 — Build the knowledge index

Click **Build Knowledge Base** to:
- Parse and normalize content
- Create semantic embeddings
- Prepare the platform for retrieval-based reasoning

This step simulates how enterprise ingestion pipelines would work in production.


### Step 3 — Ask questions or initiate workflows

Use natural language to:
- Clarify definitions and standards  
  *“What is our definition of MQL?”*
- Summarize or interpret guidance  
  *“Summarize our attribution framework and limitations.”*
- Generate draft artifacts  
  *“Draft a campaign brief aligned with our standards.”*
- Validate decisions  
  *“Create a QA checklist for this launch.”*


### Best practices for deeper results

- Ask **context-rich questions** (role, channel, market, timeframe)
- Specify the desired output format (bullets, checklist, summary)
- Treat outputs as **drafts** for review and approval


### Current limitations (intentional)

- Knowledge is session-scoped (no persistence yet)
- Manual upload is used instead of automated ingestion
- Workflow outputs are draft-only

These constraints keep the prototype safe while validating value.


### Production vision

In production, these interactions evolve into:
- Guided workflows with approvals
- Write-back to enterprise systems
- Role-aware assistance
- Measurable productivity gains
        """
    )

# =========================
# TAB 4: ARCHITECTURE (ASCII DIAGRAMS)
# =========================
with tab_arch:
    st.header("Architecture & Technical Details")

    st.markdown(
        """
The Agentic AI Knowledge Platform is built on a **modular, enterprise-oriented architecture** designed to support a clear evolution from rapid experimentation to governed, production-scale deployment.

The two diagrams below illustrate this progression:

1. **Current Prototype (Ephemeral Only)** — optimized for speed, safety, and iteration  
2. **Production Evolution (Ephemeral + Persistent Layers)** — optimized for trust, scale, and governance

Together, they show how the platform moves from a proof-of-capability into a durable enterprise system without architectural rework.

### Data Flow: Current Prototype (Ephemeral Only)

This diagram represents the **current state of the platform**, intentionally designed to be lightweight, low-risk, and fast to iterate.

Key characteristics:
- All document processing (parsing, chunking, embedding) happens **ephemerally at runtime**
- Uploaded content exists only in **session memory** and temporary storage
- No files, embeddings, or chat history persist across restarts
- The AI reasons **only over content explicitly provided by the user**

This design allows stakeholders to safely evaluate:
- The quality of retrieval-augmented reasoning
- The usefulness of responses for real business questions
- The agentic interaction pattern (Retrieve → Reason → Respond)

without introducing data retention, access, or governance risk.

        """
    )

    diagram_ephemeral = r"""
+------------------------+        +-------------------------+
|     Ops Team Member    |        |       Streamlit UI      |
|     (upload + ask)     |------->|  upload + chat interface|
+------------------------+        +-----------+-------------+
                                            |
                                            | (session state)
                                            v
                                  +---------+----------+
                                  |  Session Memory    |
                                  | - uploaded_files   |
                                  | - chat_history     |
                                  | - rag_ready flag   |
                                  +---------+----------+
                                            |
                                            | (temp write for parsing)
                                            v
                                  +---------+----------+
                                  | Temp File (/tmp)   |
                                  | short-lived         |
                                  +---------+----------+
                                            |
                                            v
+---------------------+     +----------------+     +-------------------+
| Parse docs to text  |---->| Chunk text     |---->| Create embeddings |
+---------------------+     +----------------+     +---------+---------+
                                                              |
                                                              v
                                                   +----------+----------+
                                                   | In-Memory Vector    |
                                                   | Store (ephemeral)   |
                                                   +----------+----------+
                                                              ^
                                                              |
                                        +---------------------+------------------+
                                        |     RAG Pipeline: Retrieve -> Answer   |
                                        +---------------------+------------------+
                                                              |
                                                              v
                                                   +----------+----------+
                                                   | Answer shown in UI  |
                                                   +---------------------+

Notes:
- Temp files are deleted after parsing.
- Nothing is persisted across app restarts.
"""
    st.code(diagram_ephemeral, language="text")

    st.markdown(
        """
**Why this design matters**

The ephemeral prototype establishes confidence in three critical areas:
- **Accuracy** — responses are grounded in known sources
- **Consistency** — definitions and standards are interpreted uniformly
- **Trust** — the system does not retain or reuse data unintentionally

This ensures the platform is evaluated as an **internal enablement tool**, not a black-box chatbot.
        """
    )

    st.divider()

    st.markdown(
"""
### Data Flow: Production Evolution (Ephemeral + Persistent Layers)

This diagram illustrates the **target production architecture**, where the same agentic reasoning pattern is preserved while introducing enterprise-grade capabilities.

The key architectural shift is **intentional persistence**:
- Knowledge is stored deliberately
- Access is governed explicitly
- Usage is observable and auditable

Importantly, ephemeral processing remains — it is complemented, not replaced, by persistent layers.
"""
    )

    diagram_production = r"""
+------------------------+      +---------------------+      +----------------------+
|     Ops Team Member    |----->| Internal Web UI     |----->| SSO / Identity (IdP) |
+------------------------+      +----------+----------+      +----------------------+
                                          |
                                          v
                                 +--------+--------+
                                 | Gateway / Proxy |
                                 | TLS + Rate Limit|
                                 +--------+--------+
                                          |
                                          v
                              +-----------+------------+
                              |  RAG Service            |
                              | Retrieve -> Reason ->   |
                              | Respond + Citations     |
                              +-----------+------------+
                                          |
                                          v
                             +------------+-------------+
                             | Access Control (RBAC/ACL)|
                             +------------+-------------+
                                          |
                                          v
                             +------------+-------------+
                             | Persistent Vector Store   |
                             | (Azure AI Search, etc.)   |
                             +------------+-------------+
                                          ^
                                          |
         +---------------------+----------+----------+----------------------+
         |                     |                     |                      |
         v                     v                     v                      v
+------------------+   +------------------+   +------------------+   +------------------+
| SharePoint /     |   | Confluence /     |   | MarTech Tools     |   | Data Platforms    |
| Drive Docs       |   | Wiki/Pages       |   | (AJO/CJA/SFMC)    |   | (Snowflake/DBX)   |
+--------+---------+   +--------+---------+   +--------+---------+   +--------+---------+
         \___________________________  Ingestion Service  ____________________________/
                                      (scheduled or on-demand)
                                              |
                                              v
                                 +------------+-------------+
                                 | Ephemeral Processing      |
                                 | parse -> chunk -> embed   |
                                 +------------+-------------+
                                              |
                                              v
                                 +------------+-------------+
                                 | Metadata Store            |
                                 | source, ACL, timestamps   |
                                 +------------+-------------+

Also:
- Audit logs capture who asked what, when, and what sources were used.
- Optional raw docs can be stored in controlled object storage (Blob/S3) if required.
"""
    st.code(diagram_production, language="text")

    st.markdown(
        """
**What changes in the production architecture**

In a production deployment, the platform introduces:

- **Persistent vector storage**  
  Enables reliable retrieval across sessions and users

- **Metadata and ownership tracking**  
  Associates knowledge with source systems, timestamps, and stewardship

- **Enterprise identity and access control**  
  Ensures users only retrieve content they are authorized to see

- **Audit and observability layers**  
  Capture who asked what, when, and based on which sources

- **Managed ingestion pipelines**  
  Shift from manual uploads to systems such as SharePoint or Confluence

### From knowledge assistance to agentic workflows

With this architecture in place, the platform can evolve beyond Q&A into **agentic, human-in-the-loop workflows**, such as:
- Drafting campaign briefs and SOPs
- Generating QA and compliance checklists
- Validating decisions against standards
- Assisting execution while preserving human approval

This progression enables AI to move from **informational support** to **operational acceleration** — without sacrificing trust or governance.

### Executive takeaway

The architecture demonstrates a deliberate progression:
- Start ephemeral to prove value safely
- Add persistence to scale knowledge reliably
- Introduce governance to enable enterprise adoption
- Extend into workflows to unlock productivity gains

This approach allows the organization to adopt agentic AI **incrementally, responsibly, and with measurable impact**.
        """
    )


# =========================
# Footer
# =========================
st.markdown(
    """
<hr style="margin-top:32px;margin-bottom:8px;">
<div style="text-align:center;font-size:12px;color:gray;">
  © 2025 Howard Nguyen, PhD. For prototype and demonstration only.
</div>
""",
    unsafe_allow_html=True,
)
