# Brain

A multi-agent system inspired by human cognitive architecture. Instead of one monolithic LLM, it splits reasoning across specialized agents coordinated by a central Memory Agent that owns all context and conversation history.

## Architecture

```
User input
    │
    ▼
Memory Agent (classify + orient)
    │
    ├── coverage = full  ──────────────────────────┐
    │                                              │
    └── coverage = partial / none                  │
            │                                      │
            ▼                                      │
       Search Agent ──► Search Validator           │
            │               │                      │
            │         valid─┤                      │
            │               └─retry (×3)           │
            │               └─clarify ──► END      │
            ▼                                      │
    Memory Store Knowledge                         │
            │                                      │
            ▼                                      │
    Memory Update Coverage                         │
            │                                      ▼
            └──────────────────────────► QA Draft Agent
                                                   │
                                                   ▼
                                        Orchestrator (score 1-10)
                                                   │
                                         approved──┤
                                                   └─retry (×2)
                                                   │
                                                   ▼
                                             QA Final
                                                   │
                                                   ▼
                                        Memory Store Episode ──► END
```

### Agents

| Agent | Role |
|---|---|
| **Memory Agent** | Central intelligence — classifies turns, builds oriented context, persists episodes |
| **Search Agent** | Web (DuckDuckGo) + academic (Semantic Scholar) search, coverage-aware |
| **Search Validator** | Quality gate on search results before storing |
| **QA Agent** | Generates answers from oriented context only (never raw history) |
| **Orchestrator** | Scores QA drafts 1–10, triggers retry if quality is low |
| **Goal Evaluator** | Last resort — generates clarifying questions when search fails |

### Memory layers

| Layer | Technology | Purpose |
|---|---|---|
| Knowledge store | ChromaDB (cosine similarity) | Long-term factual memory |
| Episode diary | SQLite | Full conversation history with reasoning traces |
| Oriented context | In-memory dict | Per-turn view prepared by Memory Agent |
| Graph state | LangGraph + SQLite | Checkpointed agent state |

## Stack

- **LangGraph** — agent graph with retry loops and conditional routing
- **LangChain Google GenAI** — Gemini 2.0 Flash Lite (LLM + embeddings)
- **ChromaDB** — vector store for semantic knowledge retrieval
- **DuckDuckGo + Semantic Scholar** — free search tools (no API keys beyond Gemini)

## Setup

### Local

```bash
git clone https://github.com/Seydifa/Brain.git
cd Brain
pip install -e .
pip install langchain-google-genai ddgs langgraph-checkpoint-sqlite
echo "GOOGLE_API_KEY=your_key_here" > .env
python main.py
```

Get a free Gemini API key at https://aistudio.google.com/apikey

### Google Colab (recommended)

1. Open `colab.ipynb` in [Google Colab](https://colab.research.google.com)
2. Select **L4 GPU** runtime (Colab Pro) or T4 (free tier)
3. Run cells in order — Google Drive is mounted for data persistence
4. Optionally switch to local Ollama models (cell 6)

## Usage

```
Brain is ready. Type 'exit' to quit.

You: What caused World War 2?

Brain: World War 2 was caused by a combination of factors...

You: What was the Treaty of Versailles?   ← follow-up; Memory Agent detects continuity

Brain: (references previous turn, extends context)
```

Enable debug output to see the full reasoning trace:

```bash
BRAIN_DEBUG=1 python main.py
```

## Project structure

```
Brain/
├── agents/
│   ├── memory_agent.py       # 4 LangGraph nodes (classify, store, update, episode)
│   ├── search_agent.py       # DuckDuckGo + Semantic Scholar, coverage-aware
│   ├── search_validator.py   # Quality gate on search results
│   ├── qa_agent.py           # Draft + final response nodes
│   ├── orchestrator.py       # QA quality scorer (1–10)
│   └── goal_evaluator.py     # Clarification question generator
├── memory/
│   ├── agent.py              # Central memory intelligence
│   ├── store.py              # ChromaDB knowledge store
│   ├── awareness.py          # Knowledge coverage assessment
│   └── episodes.py           # SQLite episode diary
├── core/
│   ├── graph.py              # LangGraph graph definition (10 nodes)
│   └── state.py              # Shared BrainState TypedDict
├── main.py                   # CLI REPL with rate-limit retry
└── colab.ipynb               # Google Colab notebook
```
