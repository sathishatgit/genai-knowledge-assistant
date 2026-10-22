# Engineering Challenge: The Pragmatic Knowledge Assistant

---

## 🏢 Scenario & Business Problem

You have been hired as an AI Engineer by **Sathish Technologies**. 

The company recently deployed an internal AI assistant connected to their documentation using standard "Retrieval-Augmented Generation" (RAG). However, the prototype has been an expensive disaster, and employee satisfaction is at an all-time low. 

The engineering team identified four critical failures in their current system:

1. **The Blind Retrieval Waste:** 
   The system searches the internal database for *every single question*. When an engineer asks `"How does quicksort work?"` or `"Write a Python script to reverse a string"`, the system wastes several seconds retrieving unrelated internal HR memos and company policies. The LLM gets confused by the irrelevant context, token usage is skyrocketing, and response time is unacceptable.

2. **The Exact-Match Blindspot:**
   When engineers search for specific acronyms, function names, error codes, or technical identifiers (e.g., `ERR_AUTH_902` or `BM25Okapi`), the neural vector search frequently fails to surface the right document because semantic vector models look for "meaning" rather than exact characters.

3. **Information Staleness:**
   When staff ask questions about current news, latest library releases, market benchmarks, or external updates, the internal database is silent or provides outdated answers. The assistant has no bridge to the outside world.

4. **The "Black Box" Problem:**
   Users have no visibility into how the assistant derived its answer. Did it use internal company knowledge? Did it use general AI reasoning? Did it search the web? Is it hallucinating? Nobody knows, and nobody trusts it.

---

## 🎯 The Challenge

Your challenge is to design, architect, and build a **Pragmatic Knowledge Assistant** that solves these real-world problems.

You and your team must start from first principles. **Do not copy a generic RAG tutorial.** You must think through and justify every layer of your design.

---

## 📋 System Requirements

Your solution must satisfy the following core requirements:

### 1. Intelligent Intent & Routing Strategy (The Decision Engine)
* **Rule Number One: DO NOT ALWAYS USE RAG.** Retrieval introduces latency, computational cost, and potential context noise.
* Your application must feature an **Intelligent Decision Engine** at the front door that analyzes the user's question and decides between four explicit operational routes:

  1. 🧠 **Route A: Model Knowledge Only (Direct LLM)**
     * **When to select:** General world knowledge, programming syntax, standard algorithms, definitions, mathematical problems, and logic.
     * **Behavior:** Zero retrieval overhead. The query is routed directly to the LLM without touching internal databases or external search APIs.
     * *Example:* *"What is a Python dictionary comprehension?"*

  2. 📂 **Route B: Local RAG (Internal Knowledge Base)**
     * **When to select:** Questions asking about proprietary internal documents, organization policies, internal architectures, or uploaded company files.
     * **Behavior:** Executes retrieval strictly across the internal knowledge repository.
     * *Example:* *"According to our internal architecture docs, what is our system design?"*

  3. 🌐 **Route C: Web Search (Live External Knowledge)**
     * **When to select:** Real-time information, breaking news, recent model releases, current market rates, live documentation updates, or explicit requests to browse the web.
     * **Behavior:** Executes a live web search to retrieve up-to-date internet sources.
     * *Example:* *"What is the latest AI model release announced this week?"*

  4. 🔄 **Route D: Both Fused (Internal + Web Synthesis)**
     * **When to select:** Questions that require comparing internal company systems or documents against current external industry trends or standards.
     * **Behavior:** Queries both internal local documents and live web search, synthesizing both into a coherent comparison.
     * *Example:* *"Compare our internal system architecture with the latest industry approaches."*

* **Requirements for the Decision Engine:**
  * Must output a clear, structured decision containing the selected route and the underlying reason.
  * Must strictly reject arbitrary or unrecognized route names.
  * Must include resilient fallback handling if the routing classifier encounters an error or rate limit.

### 2. Robust Multi-Method Information Retrieval
* Relying solely on one search mechanism is insufficient.
* Design a retrieval system that addresses both:
  * **Keyword/Lexical Precision:** Finding exact technical terms, acronyms, and codes.
  * **Semantic/Conceptual Understanding:** Finding relevant ideas even when different words are used.
* How will you structure, chunk, and index your internal text documents?
* How will you combine results from two entirely different scoring scales (e.g. lexical scores vs vector similarities) without biasing the outcome?

### 3. Live External Information Bridge
* Provide a way for the assistant to look up live, authoritative information from the web when the question demands real-time or external facts.
* The system must handle network delays, missing credentials, or unhelpful search results gracefully.

### 4. Factual Grounding & Verifiable Citations
* The final generated response must answer the query directly using only verified information.
* If internal documents or web results do not contain enough information, the assistant must explicitly say so rather than guess.
* Users must be able to verify where the information came from (source files for internal knowledge; URLs/page titles for web knowledge).

### 5. Interactive User Interface & Dynamic Knowledge Ingestion
* Build an intuitive interface (web dashboard or interactive console) where anyone can:
  * Ask questions and observe the system's decision-making process in real time (what route was chosen, what sources were consulted, and why).
  * **Feed new knowledge to the assistant dynamically:** Allow users to either upload a text file or type/paste a custom paragraph, and make that new knowledge immediately searchable without having to restart the application or rebuild databases from scratch.

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
| **5** | **Dynamic Live Ingestion** | `Local RAG (Updated)` | A new paragraph or document is added through the UI and immediately queried, proving the dynamic indexing works. |

---

## 🎤 Project Defense & Review

Be prepared to defend your design choices during the review:

* *Why did you choose your specific routing method over alternative approaches?*
* *How did you resolve the difference in scoring scales between your search channels?*
* *What happens if a user submits a query that touches on both internal and external facts?*
* *What is the latency and cost impact of your architecture compared to a standard naive RAG pipeline?*
* *How does your system prevent hallucination when retrieved search results are sparse or contradictory?*
