<div align="center">

# 🏆 Engineering Challenge
## The Pragmatic Knowledge Assistant

**An Advanced Hands-on AI Engineering Case Study**

[![Difficulty: Advanced](https://img.shields.io/badge/Difficulty-Advanced-red.svg?style=for-the-badge)](#)
[![Domain: GenAI & RAG](https://img.shields.io/badge/Domain-GenAI%20%26%20RAG-blueviolet.svg?style=for-the-badge)](#)
[![Format: Capstone Challenge](https://img.shields.io/badge/Format-Capstone%20Challenge-success.svg?style=for-the-badge)](#)

---

</div>

## 🏢 Scenario & Business Context

You have been hired as an **AI Engineer** by **Sathish Technologies**. 

The company recently deployed an internal AI assistant connected to their company documentation using standard **Retrieval-Augmented Generation (RAG)**. However, the prototype has been an expensive disaster, and employee satisfaction is at an all-time low. 

The engineering team identified **four critical architectural failures** in their current system:

> [!CAUTION]
> ### 1. The Blind Retrieval Waste
> The system searches the internal vector database for *every single prompt*. When an engineer asks:  
> `How does quicksort work?` or `Write a Python regex for validating email addresses`, the system wastes several seconds retrieving unrelated internal HR memos and company policies. The LLM gets confused by irrelevant context, token bills are skyrocketing, and response latency is unacceptable.

> [!WARNING]
> ### 2. The Exact-Match Blindspot
> When engineers search for specific acronyms, function names, error codes, or technical identifiers (e.g., `ERR_AUTH_902`, `BM25Okapi`, or config flags), neural vector embeddings frequently fail to surface the right document. Semantic vector models match general "meaning" rather than exact character tokens.

> [!WARNING]
> ### 3. Information Staleness
> When staff ask questions about current news, latest library releases, market benchmarks, or external updates, the internal database is silent or provides outdated answers. The assistant has zero bridge to the outside world.

> [!NOTE]
> ### 4. The "Black Box" Problem
> Users have no visibility into how the assistant derived its answer. Did it use internal company knowledge? Did it use general AI reasoning? Did it search the web? Is it hallucinating? Nobody knows, and nobody trusts it.

---

## 🎯 The Challenge

Your challenge is to design, architect, and build a **Pragmatic Knowledge Assistant** that eliminates all four failures.

You and your team must start from first principles.

> [!IMPORTANT]
> **Do not copy a generic RAG tutorial.** You are expected to design, justify, and defend every single component in your architecture.

---

## 📋 System Requirements

Your solution must satisfy the following five core engineering requirements:

### 1. Intelligent Intent & Routing Strategy (The Decision Engine)

* **Rule Number One:** **DO NOT ALWAYS USE RAG.** Retrieval introduces latency, computational cost, and potential context noise.
* Your application must feature an **Intelligent Decision Engine** at the front door that inspects query intent and autonomously routes to one of **four explicit operational paths**:

| Route Identifier | Name | When to Select | Operational Behavior | Example Query |
|---|---|---|---|---|
| **Route A** | 🧠 **Model Knowledge Only** | General world knowledge, programming syntax, standard algorithms, definitions, math, logic | **Zero retrieval overhead.** Query is answered directly by model reasoning without touching databases or external APIs. | *"What is a Python dictionary comprehension?"* |
| **Route B** | 📂 **Local RAG** | Proprietary documents, organizational policies, internal system architectures, uploaded files | Executes search **strictly** across the local internal knowledge repository. | *"According to our internal architecture docs, what is our system design?"* |
| **Route C** | 🌐 **Web Search** | Real-time events, breaking news, latest library releases, live docs, or explicit search requests | Executes a live web query to fetch and synthesize up-to-date internet sources. | *"What is the latest AI model released this week?"* |
| **Route D** | 🔄 **Both Fused** | Inquiries requiring a direct comparison of internal designs against external industry standards | Queries **both** internal local documents and live web search, synthesizing both into a coherent comparison. | *"Compare our internal system architecture with latest industry approaches."* |

#### Requirements for the Decision Engine:
- Must return a structured classification containing the chosen route and the technical reasoning.
- Must reject arbitrary, ambiguous, or unhandled route names.
- Must include resilient fallback handling if the router encounters an API error, rate limit, or network timeout.

---

### 2. Robust Multi-Method Information Retrieval
* Relying solely on one search mechanism is insufficient.
* Design a retrieval system that addresses both:
  - **Keyword / Lexical Precision:** Finding exact technical terms, acronyms, and codes.
  - **Semantic / Conceptual Understanding:** Finding relevant ideas even when different words or synonyms are used.
* *Key architectural questions your team must resolve:*
  - How will you structure, chunk, and index your internal text documents?
  - How will you combine results from two entirely different scoring scales (e.g. lexical scores vs vector cosine similarities) without biasing the outcome?

---

### 3. Live External Information Bridge
* Provide a mechanism for the assistant to look up live, authoritative information from the web when queries demand real-time or external facts.
* The system must handle network delays, missing credentials, or sparse search results gracefully.

---

### 4. Factual Grounding & Verifiable Citations
* The final generated response must answer the query directly using only verified information.
* If internal documents or web results do not contain enough information, the assistant must explicitly state that rather than guess.
* Users must be able to verify where the information came from (source files for internal knowledge; URLs and page titles for web knowledge).

---

### 5. Interactive User Interface & Dynamic Knowledge Ingestion
* Build an intuitive interface (web dashboard or interactive console) where users can:
  - Ask questions and observe the system's decision-making process in real time (selected route, consulted sources, and step-by-step pipeline state).
  - **Feed new knowledge to the assistant dynamically:** Allow users to either upload a text file or type/paste a custom paragraph, and make that new knowledge immediately searchable without having to restart the application or rebuild databases from scratch.

---

## 🚫 Constraints & Rules

1. **Think for Yourself:** You are responsible for deciding which libraries, data structures, algorithms, and models to use.
2. **Local First:** The system must run entirely on a single developer machine without requiring any external paid services, cloud infrastructure, or containerization platforms.
3. **Security:** API keys and credentials must never be committed to source code or displayed on screen.
4. **Resilience:** The application must never crash on malformed queries, empty searches, or temporary network timeouts.

---

## 🧪 Challenge Benchmark Scenarios

Your application will be evaluated live by feeding it the following query types. Your system's Decision Engine must autonomously select the appropriate route for each:

| # | Scenario Type | Expected Autonomous Route | Evaluation Criteria |
|---|---|---|---|
| **1** | **Foundational Concept** | `Model Knowledge Only` | Zero retrieval hops made. Answer generated rapidly from foundation model knowledge alone. |
| **2** | **Proprietary Knowledge** | `Local RAG` | Retrieves internal documents. Sources cited from local files. |
| **3** | **Real-Time / External** | `Web Search` | Executes live external search. Sources include web titles and URLs. |
| **4** | **Cross-Context Comparative** | `Both Fused` | Queries both internal knowledge and external web sources, presenting a synthesized comparison. |
| **5** | **Dynamic Live Ingestion** | `Local RAG (Updated)` | A new paragraph or document is added through the UI and immediately queried, proving dynamic indexing works. |

---

## 🎤 Project Defense & Review

Be prepared to defend your design choices during the technical review:

* *Why did you choose your specific routing method over alternative approaches?*
* *How did you resolve the difference in scoring scales between your search channels?*
* *What happens if a user submits a query that touches on both internal and external facts?*
* *What is the latency and cost impact of your architecture compared to a standard naive RAG pipeline?*
* *How does your system prevent hallucination when retrieved search results are sparse or contradictory?*
