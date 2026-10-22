# Intelligent GenAI Knowledge Assistant

An educational, realistic, and complete Generative AI application built for live hands-on workshops and real-world multi-route RAG architectures.

> **👨‍💻 Developed by:** **Sathish** (AI Engineer)  
> **🔗 LinkedIn:** [Sathishbabu Shanmugam](https://www.linkedin.com/in/sathishbabu-shanmugam/)  
> **🐙 GitHub:** [@sathishatgit](https://github.com/sathishatgit)

---

## 1. Project Overview

The **Intelligent GenAI Knowledge Assistant** demonstrates how modern AI systems solve real-world queries by dynamically selecting the optimal retrieval and reasoning path rather than forcing every question through a fixed RAG pipeline.

The application combines:
- **Google Gemini** for reasoning, routing, and final response synthesis
- **Jina AI Embeddings** (`jina-embeddings-v3`) for dense semantic representations
- **ChromaDB** for local persistent vector storage without Docker
- **Okapi BM25** for lexical keyword and exact terminology retrieval
- **Reciprocal Rank Fusion (RRF)** for fusing lexical and vector search rankings
- **Exa Search API** for real-time web retrieval

---

## 2. Target Architecture

```text
                         USER QUERY
                              |
                              v
                     INTELLIGENT ROUTER
                              |
          +-------------------+-------------------+
          |                   |                   |
          v                   v                   v
      DIRECT LLM          LOCAL RAG          WEB SEARCH
                              |                   |
                    +---------+---------+         |
                    |                   |         |
                    v                   v         v
                  BM25             ChromaDB      Exa
                    |              + Jina         |
                    |             Embeddings      |
                    +---------+---------+----------+
                              |
                              v
                       RRF / Ranking
                              |
                              v
                     SELECTED CONTEXT
                              |
                              v
                           GEMINI
                              |
                              v
                        FINAL ANSWER
```

### Combined Route (`both`):
```text
User Query
    |
    v
Router
    |
    +--------------------+
    |                    |
    v                    v
Local RAG             Exa Search
 (BM25 + ChromaDB)       |
    |                    |
    +---------+----------+
              |
              v
       Combined Context
              |
              v
            Gemini
              |
              v
        Final Answer
```

---

## 3. Why Intelligent Routing?

Traditional AI tutorials often direct every single user prompt through RAG. In production systems, this is wasteful and counterproductive.

An intelligent router:
1. **Reduces Latency**: Bypasses retrieval indexing, embedding API calls, and vector lookups for general questions.
2. **Saves Cost**: Avoids unnecessary external embedding and search API billing.
3. **Improves Answer Quality**: Prevents irrelevant or low-scoring document chunks from polluting the LLM context window.
4. **Enables Real-Time Knowledge**: Transparently switches to web search when queries ask about recent updates, breaking news, or live events.

---

## 4. Why Don't We Always Use RAG?

Retrieval-Augmented Generation (RAG) is powerful, but **it has significant overhead**:
- **Computational Cost**: Every retrieval requires tokenizing text, generating embedding vectors via network calls, performing vector distance calculations, and injecting hundreds of tokens into the prompt context.
- **Context Pollution**: If a user asks "What is a Python list?", injecting chunks about company policies or internal system design is irrelevant noise that distracts the model.
- **Staleness**: Local documents cannot answer "What is the latest Gemini model released this week?". Forcing local RAG here leads to hallucinations or answers stating that the document does not contain the answer.

Therefore, our architecture evaluates each question and selects one of four distinct routes:
- `direct_llm`: For general programming, concepts, and foundational reasoning.
- `local_rag`: For questions regarding internal company documents and architectures.
- `web_search`: For real-time updates and current web facts.
- `both`: For synthesizing internal architecture comparisons against current external industry standards.

---

## 5. Project Structure

```text
genai-rag-workshop/
│
├── data/
│   ├── documents/
│   │   ├── python.txt
│   │   ├── machine_learning.txt
│   │   ├── generative_ai.txt
│   │   └── rag.txt
│   │
│   ├── chroma/                 # Local persistent ChromaDB storage
│   └── bm25_index.json         # Serialized BM25 lexical index
│
├── src/
│   ├── __init__.py
│   ├── config.py               # Central environment variables & paths
│   ├── document_loader.py      # Plain text document loading
│   ├── chunker.py              # Overlapping character-based chunking
│   ├── embeddings.py           # Jina AI Embeddings REST API client
│   ├── vector_store.py         # Local ChromaDB persistent vector store
│   ├── bm25_search.py          # Okapi BM25 lexical keyword engine
│   ├── rrf.py                  # Reciprocal Rank Fusion algorithm
│   ├── hybrid_search.py        # BM25 + Vector + RRF coordinator
│   ├── exa_search.py           # Exa AI Web Search client
│   ├── gemini.py               # Google Gemini API generation
│   ├── router.py               # Intelligent LLM classification router
│   └── rag.py                  # End-to-end multi-route orchestrator
│
├── ingest.py                   # Document ingestion pipeline
├── main.py                     # Interactive CLI application
├── check_setup.py              # Workshop environment verification
├── requirements.txt            # Python dependencies
├── .env.example                # Configuration template
├── .env                        # Secret environment variables (ignored by git)
├── .gitignore                  # Git ignore rules
└── README.md                   # Complete documentation
```

---

## 6. Installation & Virtual Environment

### Step 1: Clone or Navigate to the Workspace
```bash
cd "d:/Gen AI EGSP"
```

### Step 2: Create and Activate Virtual Environment
**Windows (PowerShell):**
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

**Linux / macOS:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Required Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

---

## 7. Requirements

The project uses a minimal, carefully selected dependency set:
- `google-generativeai`: Google Gemini API client
- `chromadb`: Local embedded vector database (no Docker required)
- `rank-bm25`: Fast, standard implementation of Okapi BM25
- `requests`: HTTP client for Jina and Exa REST APIs
- `python-dotenv`: Environment configuration management
- `langchain` & `langchain-community`: AI orchestration utilities

> **Note**: `sentence-transformers` is intentionally omitted. Dense embeddings are generated via the Jina AI REST API to keep the local installation lightweight and fast.

---

## 8. API Keys & `.env` Configuration

This project requires **3 free API keys**. All services listed below offer a free tier — no credit card required to get started.

---

### 🔑 How to Get Your API Keys

#### 1. Google Gemini API Key (for LLM routing + response generation)
| | |
|---|---|
| **Sign up at** | https://aistudio.google.com/ |
| **Steps** | Sign in with Google → Click **"Get API Key"** → **"Create API key"** → Copy the key |
| **Free Tier** | Gemini 1.5 Flash: **15 requests/min, 1 million tokens/day** (no credit card needed) |
| **Env variable** | `GEMINI_API_KEY` |

#### 2. Jina AI API Key (for text embeddings)
| | |
|---|---|
| **Sign up at** | https://jina.ai/ |
| **Steps** | Click **"Get API Key"** on the homepage → Sign up with email or Google → Copy the key |
| **Free Tier** | **1 million tokens free** on signup (no credit card needed) |
| **Env variable** | `JINA_API_KEY` |

#### 3. Exa AI API Key (for live web search)
| | |
|---|---|
| **Sign up at** | https://exa.ai/ |
| **Steps** | Click **"Get Started"** → Create account → Go to **Dashboard → API Keys** → Copy the key |
| **Free Tier** | **1,000 free searches/month** (no credit card needed) |
| **Env variable** | `EXA_API_KEY` |

---

### ⚙️ Configure Your `.env` File

Copy `.env.example` to `.env`:
```bash
# Windows
copy .env.example .env

# Linux / macOS
cp .env.example .env
```

Open `.env` and paste your keys:
```ini
# Google Gemini API Key — https://aistudio.google.com/
GEMINI_API_KEY=your_gemini_api_key_here

# Jina AI Embeddings API Key — https://jina.ai/
JINA_API_KEY=your_jina_api_key_here

# Exa AI Search API Key — https://exa.ai/
EXA_API_KEY=your_exa_api_key_here

# Model Configuration (defaults shown below — change if needed)
GEMINI_MODEL=gemini-1.5-flash
JINA_EMBEDDING_MODEL=jina-embeddings-v3
```

> ⚠️ **Never commit your `.env` file to Git.** It is already listed in `.gitignore` to protect your keys.

---

## 9. Setup Verification

Run `check_setup.py` to verify your environment, dependencies, and configuration:
```bash
python check_setup.py
```

Expected output:
```text
========================================
 GenAI Workshop Setup Check
========================================
[OK] Python version: 3.12.7
[OK] Gemini package
[OK] ChromaDB
[OK] BM25
[OK] Requests
[OK] python-dotenv
[OK] LangChain
[OK] LangChain Community
[OK] .env file found
[OK] Gemini API key configured
[OK] Jina API key configured
[OK] Exa API key configured
[OK] ChromaDB local initialization verified
========================================
Setup completed successfully! All components ready.
========================================
```

---

## 10. Document Ingestion

Before querying internal documents, run the ingestion script to process educational documents from `data/documents/`:
```bash
python ingest.py
```

What `ingest.py` performs:
1. Loads text files (`python.txt`, `machine_learning.txt`, `generative_ai.txt`, `rag.txt`).
2. Splits documents into overlapping chunks (400 characters, 80 character overlap).
3. Generates 1024-dimensional embeddings via Jina AI API (`jina-embeddings-v3`).
4. Stores chunks and vectors in local ChromaDB (`data/chroma/`).
5. Builds and saves the Okapi BM25 lexical index to `data/bm25_index.json`.

---

## 11. Running the Application

### Option A: Interactive Terminal CLI
```bash
python main.py
```

You can also run non-interactive queries using the `--query` flag:
```bash
python main.py --query "What is a Python list?"
```

To force a specific route for testing:
```bash
python main.py --query "What is RAG?" --route local_rag
```

### Option B: AGUI Live Interactive Web Application (Phase 15)
Start the modern live web UI server:
```bash
python app.py
```
Then open your browser at:
```text
http://localhost:8000
```
Features in the AGUI Live Web Dashboard:
- Real-time multi-step execution visualizer (Router ➔ Retrieval ➔ Fusion ➔ Gemini).
- Service health badges for Gemini, Jina, Exa, and ChromaDB.
- Preset one-click workshop query buttons for all 4 routing paths.
- Route selector allowing both Auto (Gemini classification) and Forced overrides.
- Ingestion control: Re-index knowledge base directly from the UI.
- Interactive expandable citations with direct web links and document file badges.

---

## 12. Example Questions Across All 4 Routes

### Route 1: `direct_llm`
```text
You: What is a Python list?

Route: DIRECT LLM
Reason: General programming concept that does not require private documents or real-time lookup.

Answer:
A Python list is a built-in, mutable, ordered sequence of elements...
```

### Route 2: `local_rag`
```text
You: According to our documents, what is our RAG architecture?

Route: LOCAL RAG
Reason: The question asks about internal documentation and system architecture.

Retrieved Sources:
- rag.txt

Answer:
According to internal documentation, our RAG system consists of:
1. Ingestion: Jina AI Embeddings (jina-embeddings-v3) and ChromaDB in data/chroma/.
2. Retrieval: Hybrid Search combining BM25Okapi and ChromaDB fused via RRF (k=60).
3. Generation: Google Gemini (gemini-1.5-flash) for factually grounded synthesis.
```

### Route 3: `web_search`
```text
You: What is the latest Gemini model?

Route: WEB SEARCH
Reason: Requires up-to-date, real-time external information.

Web Sources:
- Gemini Models Overview
  https://deepmind.google/technologies/gemini/

Answer:
Google's current flagship model family includes Gemini 1.5 Pro and Gemini 1.5 Flash...
```

### Route 4: `both`
```text
You: Compare our RAG architecture with the latest RAG approaches.

Route: BOTH
Reason: Requires internal architecture knowledge and current external industry trends.

Internal Sources:
- rag.txt

Web Sources:
- Advanced RAG Architectures in 2025/2026
  https://...

Answer:
Comparison Analysis:
- Internal Architecture: Uses a 3-stage pipeline (Chunking -> Jina Embeddings -> ChromaDB + BM25 -> RRF -> Gemini).
- Industry Advances: Modern approaches frequently introduce Agentic RAG, graph-based knowledge retrieval (GraphRAG), and speculative decoding...
```

---

## 13. Core Concepts Explained for Students

### BM25 (Lexical Search)
Okapi BM25 is a probabilistic retrieval model based on term frequency (TF) and inverse document frequency (IDF). It scores documents by how frequently query keywords appear relative to document length and general rarity.
- **Best for**: Exact keyword matching, product IDs, acronyms, code identifiers.
- **Weakness**: Cannot match synonyms or handle paraphrasing ("automobile" vs "car").

### Dense Vector Embeddings (Semantic Search)
Neural embedding models (such as Jina AI's `jina-embeddings-v3`) map text chunks into high-dimensional geometric vector space (e.g., 1024 dimensions). Text segments with similar meanings land close together.
- **Best for**: Conceptual understanding, synonyms, answering questions in natural phrasing.
- **Weakness**: Can occasionally miss exact technical acronyms or rare numbers.

### ChromaDB (Local Vector Database)
ChromaDB provides local vector persistence using an embedded SQLite metadata store and HNSW index. It operates directly inside your Python process without requiring Docker containers or external servers.

### Hybrid Search & Reciprocal Rank Fusion (RRF)
BM25 scores and vector cosine similarities cannot be simply added together because they operate on completely different numerical scales.
**Reciprocal Rank Fusion** solves this by evaluating ranking positions rather than raw scores:
$$\text{RRF Score}(d) = \sum_{m \in M} \frac{1}{k + \text{rank}_m(d)}$$
Where $k$ is a smoothing constant (standard: 60) and $\text{rank}_m(d)$ is the 1-indexed position of document $d$ in retrieval method $m$. If a document appears near the top in both lexical and semantic searches, its fused score is amplified.

### Exa AI (Neural Web Search)
Exa uses neural embeddings to search the live internet, understanding semantic queries and returning clean webpage text snippets and URLs.

### Intelligent Routing with Gemini
The router acts as the traffic controller. It feeds the user question into Gemini with a strict system prompt and enforces structured JSON output. If the JSON is invalid or specifies an unapproved tool, the system falls back safely without crashing.

---

## 14. Troubleshooting

| Issue | Cause | Solution |
|---|---|---|
| `Missing required API key: GEMINI_API_KEY` | `.env` file has empty placeholder | Add your actual Gemini API key to `.env` |
| `Jina AI API HTTP error 401` | Invalid or expired Jina key | Verify your key at [jina.ai](https://jina.ai/) |
| `Exa Search API HTTP error 401` | Invalid or expired Exa key | Verify your key at [exa.ai](https://exa.ai/) |
| `No .txt documents found` | `data/documents/` is empty | Run `python ingest.py` after adding documents |
| `ChromaDB count is 0` | Ingestion was run without JINA key | Set `JINA_API_KEY` in `.env` and rerun `python ingest.py` |

---

## 15. Author & Credits

- **Creator & Lead Developer:** **Sathish** (AI Engineer)
- **LinkedIn:** [https://www.linkedin.com/in/sathishbabu-shanmugam/](https://www.linkedin.com/in/sathishbabu-shanmugam/)
- **GitHub Profile:** [https://github.com/sathishatgit](https://github.com/sathishatgit)
- **Repository:** [https://github.com/sathishatgit/genai-knowledge-assistant](https://github.com/sathishatgit/genai-knowledge-assistant)

---

## 16. Contributing & Pull Requests

This is an open repository — anyone can clone, study, and experiment with the project! Direct pushes to the repository are restricted; all contributions are welcome through **Pull Requests (PRs)**:

1. **Fork** the repository to your own GitHub account.
2. **Clone** your fork locally:
   ```bash
   git clone https://github.com/<your-username>/genai-knowledge-assistant.git
   ```
3. **Create a new branch** for your feature or bugfix:
   ```bash
   git checkout -b feature/your-feature-name
   ```
4. **Make your changes** and test thoroughly.
5. **Commit and push** to your fork:
   ```bash
   git commit -m "feat: describe your change"
   git push origin feature/your-feature-name
   ```
6. **Open a Pull Request (PR)** against the `master` branch of [sathishatgit/genai-knowledge-assistant](https://github.com/sathishatgit/genai-knowledge-assistant).


