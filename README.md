# Real Estate ML — Intelligent Property Price Prediction & Agentic Advisory

> A capstone project combining ensemble machine learning with an agentic AI advisory system for residential real estate valuation and investment analysis.

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Milestone Status](#2-milestone-status)
3. [Tech Stack](#3-tech-stack)
4. [System Architecture](#4-system-architecture)
5. [Repository Structure](#5-repository-structure)
6. [Module Breakdown](#6-module-breakdown)
7. [Environment Setup & API Keys](#7-environment-setup--api-keys)
8. [Installation](#8-installation)
9. [Running the Application](#9-running-the-application)
10. [Training the Model](#10-training-the-model)
11. [Model Details](#11-model-details)
12. [AI Advisory Agent (Milestone 2)](#12-ai-advisory-agent-milestone-2)
13. [RAG System](#13-rag-system)
14. [Web Application](#14-web-application)
15. [Model Performance](#15-model-performance)
16. [Deployment](#16-deployment)
17. [Troubleshooting](#17-troubleshooting)
18. [Known Limitations](#18-known-limitations)
19. [Versioning](#19-versioning)
20. [References & Resources](#20-references--resources)
21. [Disclaimer](#21-disclaimer)

---

## 1. Project Overview

This project is a two-milestone capstone that builds from a classical machine learning price predictor into a fully agentic AI advisory system for residential real estate.

**Milestone 1** trains an ensemble ML model (Random Forest + Ridge Regression) on the Ames Housing Dataset and serves it through an interactive Streamlit web interface. Users input property details and receive an instant price estimate.

**Milestone 2** extends the system with a LangGraph-based autonomous agent. The agent retrieves neighborhood market data from a Chroma vector database (RAG), runs mathematical price validation against comparable properties, and generates a structured investment recommendation: **BUY**, **HOLD**, or **INVESTIGATE FURTHER**.

Both milestones are fully implemented and accessible from the same Streamlit application.

---

## 2. Milestone Status

| Milestone | Description | Status |
|-----------|-------------|--------|
| Milestone 1 | ML-Based Property Price Prediction | Complete |
| Milestone 2 | Agentic AI Advisory with LangGraph + RAG | Complete |

> **Note:** The AI Advisory agent uses rule-based analysis by default. Connecting an OpenAI API key (see [Section 7](#7-environment-setup--api-keys)) enables full natural language narrative generation inside the agent nodes.

---

## 3. Tech Stack

| Layer | Technology |
|-------|-----------|
| Web Interface | Streamlit |
| ML Framework | Scikit-Learn (Pipeline, ColumnTransformer) |
| ML Models | Random Forest (500 estimators) + Ridge Regression |
| Agent Framework | LangGraph |
| LLM Integration | LangChain + OpenAI (optional) |
| Vector Database | ChromaDB |
| Data Processing | Pandas, NumPy |
| Visualizations | Matplotlib |
| Model Serialization | Joblib |
| Data Validation | Pydantic |
| HTTP Client | HTTPX |
| Environment Config | python-dotenv |
| Language | Python 3.8+ |

---

## 4. System Architecture

```
+-------------------------------------------------------------------+
|                      Streamlit UI (app.py)                        |
|                                                                   |
|  Tab 1: Price Prediction   Tab 2: Model Performance               |
|  Tab 3: AI Advisory        Tab 4: How It Works   Tab 5: About     |
+---------------------+-----------------------+---------------------+
                      |                       |
          +-----------v----------+   +--------v------------------------+
          |    ML Pipeline       |   |   Real Estate Advisory Agent    |
          |   (Milestone 1)      |   |        (Milestone 2)            |
          |                      |   |                                 |
          |  KNN Imputer         |   |  Node 1: retrieve_market_data  |
          |  RobustScaler        |   |  Node 2: analyze_property      |
          |  OneHotEncoder       |   |  Node 3: validate_price        |
          |  RandomForest (500)  |   |  Node 4: analyze_comparables   |
          |  Ridge Regression    |   |  Node 5: analyze_market_pos    |
          +----------------------+   |  Node 6: generate_recommend    |
                                     |  Node 7: generate_report       |
                                     +--------+------------------------+
                                              |
                              +--------------v--------------+
                              |        ChromaDB (RAG)        |
                              |                              |
                              |  market_insights             |
                              |  comparable_properties       |
                              |  regulations_trends          |
                              +------------------------------+
```

**Data flow:**

```
User Input
    |
    v
Feature Engineering (21 features)
    |
    v
ML Model (Random Forest + Ridge) --> Predicted Price
                                          |
                                          v
                          LangGraph Agent receives (features + price)
                                          |
                         +----------------+----------------+
                         |                                 |
                    RAG Query                       ML Predicted Price
                         |                                 |
                  Market Data Retrieved            Price Validation
                  Comparable Properties            (mathematical)
                  Regulations                             |
                         |                                 |
                         +-----------+--------------------+
                                     |
                             BUY / HOLD / INVESTIGATE
                                     |
                             Structured Advisory Report
```

---

## 5. Repository Structure

```
real-estate-ml/
|
+-- .devcontainer/                   # Dev container config (VSCode / GitHub Codespaces)
|   +-- devcontainer.json
|
+-- .streamlit/                      # Streamlit theme and server configuration
|   +-- config.toml
|
+-- chroma_db/                       # ChromaDB vector store (auto-generated on first run)
|
+-- data/
|   +-- ames.csv                     # Ames Housing Dataset (2,930 residential properties)
|
+-- docs/                            # Additional documentation assets
|
+-- models/                          # Generated after running src/train.py
|   +-- model.pkl                    # Trained ensemble model (serialized with joblib)
|   +-- metrics.json                 # All performance metrics (accuracy, R2, MAE, RMSE)
|   +-- model_info.txt               # Human-readable model summary
|
+-- src/                             # Core application source code
|   +-- train.py                     # Model training pipeline
|   +-- model_utils.py               # ML inference utilities, feature preparation
|   +-- agent.py                     # LangGraph agent orchestrator (Milestone 2)
|   +-- rag_system.py                # ChromaDB RAG implementation
|   +-- prompts.py                   # LLM system prompts + hallucination reduction
|   +-- advisory_ui.py               # Streamlit UI components for the advisory tab
|
+-- app.py                           # Main Streamlit application entry point
+-- requirements.txt                 # All Python dependencies
|
+-- README.md                        # This file
+-- AGENT_DOCUMENTATION.md           # Deep-dive technical docs for the agent
+-- CAPSTONE_COMPLETION_SUMMARY.md   # Summary of both milestones
+-- DEPLOYMENT_GUIDE.md              # Step-by-step deployment instructions
+-- MILESTONE2_SUMMARY.md            # Milestone 2 specific summary
+-- PROJECT_DOCUMENT.md              # Full academic project document
+-- STREAMLIT_DEPLOYMENT.md          # Streamlit Cloud deployment walkthrough
+-- init_git.sh                      # Git initialization helper
+-- .gitignore                       # Excludes: models/, .env, chroma_db/, __pycache__/
```

---

## 6. Module Breakdown

The codebase is organized with clean separation between ML logic, agent logic, RAG logic, and UI. Nothing is mixed into `app.py` that belongs in `src/`.

### `app.py` — Application Entry Point

The main Streamlit file. Loads the trained model, initializes the 5-tab layout, and routes user interactions to the correct backend. Contains no ML training code, no agent logic, and no RAG queries — all of that is delegated to `src/`.

### `src/train.py` — Training Pipeline

Handles the full training workflow end-to-end:
- Loads `data/ames.csv`
- Engineers 4 features from raw inputs (`House_Age`, `Quality_Area`, `Quality_Condition_Score`, `Total_Floor_Area`)
- Builds the Scikit-Learn `ColumnTransformer` and `Pipeline`
- Trains Random Forest (500 estimators) and Ridge Regression
- Runs 5-fold cross-validation
- Saves `models/model.pkl`, `models/metrics.json`, `models/model_info.txt`

### `src/model_utils.py` — ML Inference Utilities

Contains:
- `ModelPredictor` class — wraps `model.pkl` and exposes a clean `.predict()` interface
- `prepare_property_features()` — converts raw UI input dict to the 21-feature DataFrame the model expects
- `get_property_options_from_dataset()` — reads `ames.csv` to supply dropdown options (neighborhoods, building types, house styles) to the UI

### `src/agent.py` — LangGraph Agent Orchestrator

The core of Milestone 2. Defines:
- `PropertyAnalysisState` dataclass — the shared state object that flows through all 7 nodes
- `RealEstateAdvisoryAgent` class — builds and compiles the LangGraph `StateGraph`
- All 7 node functions (`retrieve_market_data`, `analyze_property`, `validate_price`, `analyze_comparables`, `analyze_market_position`, `generate_recommendation`, `generate_advisory_report`)

Agent logic is fully decoupled from the UI. It accepts a plain dictionary and returns a structured report dictionary. The UI in `advisory_ui.py` just calls the agent and renders results.

### `src/rag_system.py` — ChromaDB RAG Implementation

Manages the local vector database:
- Initializes 3 Chroma collections on first use
- `initialize_sample_market_data()` — seeds the database with sample market data
- `query_market_insights()` — retrieves neighborhood trend documents
- `query_comparable_properties()` — retrieves historical sales records for comparables
- `query_regulations()` — retrieves local tax and zoning data

### `src/prompts.py` — System Prompts + Hallucination Reduction

Stores all LLM system prompts used in agent nodes. Each prompt includes explicit grounding constraints so GPT cannot fabricate data. Separating prompts here means they can be updated, reviewed, or tested independently of agent orchestration logic.

### `src/advisory_ui.py` — Advisory Tab UI Components

Contains:
- `render_advisory_tab()` — the full Milestone 2 UI (input form, report display, JSON/text download buttons)
- `render_how_it_works()` — content for the "How It Works" explanation tab

Keeping UI components here rather than in `app.py` keeps the main application file clean and short.

---

## 7. Environment Setup & API Keys

### No API key required to run the core application

The ML price predictor and rule-based advisory agent work with zero API keys. You can install, train, and run the app without any external services.

### Optional: OpenAI API Key (LLM-enhanced advisory narratives)

Without a key: the agent produces mathematical analysis only (price validation, deviation signal, recommendation).

With a key: analysis nodes generate natural language narratives via GPT inside `src/agent.py`, giving richer property analysis and comparable property commentary.

**Step 1: Create a `.env` file in the project root**

```bash
touch .env
```

**Step 2: Add your credentials to `.env`**

```env
# .env

# Required for LLM-enhanced advisory narratives
OPENAI_API_KEY=sk-your-openai-api-key-here

# Optional: override default model (defaults to gpt-3.5-turbo)
OPENAI_MODEL=gpt-4-turbo-preview
```

**Step 3: The app loads `.env` automatically on startup**

`python-dotenv` is called inside `src/agent.py` — no manual export or shell configuration needed:

```python
# Inside src/agent.py
from dotenv import load_dotenv
load_dotenv()  # Reads .env from project root automatically
```

> **Security rule:** `.env` is in `.gitignore` and will never be committed to the repository. Never hardcode API keys in source files. If you accidentally commit a key, rotate it immediately via your OpenAI dashboard.

### Capability matrix: with vs without API key

| Feature | No API Key | With OpenAI Key |
|---------|------------|----------------|
| Price Prediction (Tab 1) | Full | Full |
| Model Performance (Tab 2) | Full | Full |
| Advisory: Price Validation | Full (math) | Full (math) |
| Advisory: Recommendation Signal | Full (rule-based) | Full (rule-based) |
| Advisory: Property Analysis narrative | Not available | GPT-generated |
| Advisory: Comparable Analysis narrative | Not available | GPT-generated |

---

## 8. Installation

### Prerequisites

- Python 3.8 or higher
- pip
- 2GB RAM minimum (4GB recommended for smooth LLM usage)
- Git

### Step 1: Clone the repository

```bash
git clone https://github.com/CosmicGalactus/real-estate-ml.git
cd real-estate-ml
```

### Step 2: Create a virtual environment

```bash
# Create virtual environment
python -m venv venv

# Activate on macOS/Linux
source venv/bin/activate

# Activate on Windows
venv\Scripts\activate
```

### Step 3: Install all dependencies

```bash
pip install -r requirements.txt
```

Full package list:

| Package | Version | Purpose |
|---------|---------|---------|
| `pandas` | >=1.3.0 | Data loading and manipulation |
| `numpy` | >=1.21.0 | Numerical operations |
| `scikit-learn` | >=0.24.0 | ML pipeline, models, preprocessing |
| `streamlit` | >=1.28.0 | Web application interface |
| `matplotlib` | >=3.4.0 | Charts and visualizations |
| `joblib` | >=1.0.0 | Model serialization/deserialization |
| `langgraph` | >=0.0.1 | Agent workflow orchestration |
| `langchain` | >=0.1.0 | LLM chaining utilities |
| `langchain-openai` | >=0.0.1 | OpenAI LLM integration |
| `langchain-community` | >=0.0.1 | Community integrations |
| `chromadb` | >=0.4.0 | Vector database for RAG |
| `httpx` | >=0.24.0 | Async HTTP client |
| `pydantic` | >=2.0.0 | Data validation and schemas |
| `python-dotenv` | >=1.0.0 | `.env` file loading |

### Step 4: Configure environment variables (optional)

```bash
echo "OPENAI_API_KEY=your_key_here" > .env
```

See [Section 7](#7-environment-setup--api-keys) for full details.

### Step 5: Verify the installation

```bash
python -c "
import pandas, numpy, sklearn, streamlit, langgraph, chromadb, pydantic
print('All packages installed successfully')
"
```

Expected output: `All packages installed successfully`

---

## 9. Running the Application

### Train the model first (required on first run)

```bash
python src/train.py
```

This must complete before the app can start. It generates `models/model.pkl` and `models/metrics.json`.

### Start the Streamlit application

```bash
streamlit run app.py
```

Opens at `http://localhost:8501` automatically.

### Additional run options

```bash
# Custom port
streamlit run app.py --server.port 8080

# Headless mode (for remote servers, no browser auto-open)
streamlit run app.py --server.headless true

# Debug logging
streamlit run app.py --logger.level=debug
```

---

## 10. Training the Model

```bash
python src/train.py
```

**What happens during training:**

1. Loads `data/ames.csv` (2,930 Ames, Iowa residential properties)
2. Drops rows with missing target value (`SalePrice`)
3. Engineers 4 new features:
   - `House_Age` = 2026 − Year Built
   - `Quality_Area` = Overall Qual × Gr Liv Area
   - `Quality_Condition_Score` = Overall Qual × Overall Cond
   - `Total_Floor_Area` = 1st Flr SF + Total Bsmt SF
4. Splits data 80/20 (train/test), stratified
5. Builds `ColumnTransformer`:
   - Numeric path: KNN Imputer (k=5) → RobustScaler
   - Categorical path: OneHotEncoder (handle\_unknown='ignore')
6. Trains Random Forest (500 estimators, max\_depth=20, random\_state=42, n\_jobs=-1)
7. Trains Ridge Regression as baseline
8. Runs 5-fold cross-validation on the full Random Forest pipeline
9. Saves all outputs

**Expected training time:** 3–5 minutes on standard laptop hardware.

**Generated files:**

```
models/
+-- model.pkl        # Serialized Scikit-Learn Pipeline (load with joblib.load)
+-- metrics.json     # Accuracy, precision, R2, MAE, RMSE, MAPE, CV scores, dataset sizes
+-- model_info.txt   # Human-readable summary of hyperparameters and results
```

**Loading the model manually (for custom inference):**

```python
import joblib
import pandas as pd

model = joblib.load("models/model.pkl")

# Input must be a DataFrame with exactly these 21 columns
sample = pd.DataFrame({
    'Gr Liv Area': [1500],
    'Total Bsmt SF': [800],
    'Overall Qual': [7],
    'Overall Cond': [6],
    'Year Built': [2000],
    'House_Age': [26],
    '1st Flr SF': [1200],
    'Garage Area': [480],
    'Lot Area': [9000],
    'Bedroom AbvGr': [3],
    'Full Bath': [2],
    'Half Bath': [0],
    'Kitchen AbvGr': [1],
    'TotRms AbvGrd': [7],
    'Garage Cars': [2],
    'Quality_Area': [10500],
    'Quality_Condition_Score': [42],
    'Total_Floor_Area': [2000],
    'Neighborhood': ['CollgCr'],
    'Bldg Type': ['1Fam'],
    'House Style': ['2Story']
})

price = model.predict(sample)[0]
print(f"Predicted price: ${price:,.0f}")
```

---

## 11. Model Details

### Architecture

The model is a Scikit-Learn `Pipeline` that chains preprocessing and a Random Forest Regressor. Ridge Regression runs separately as a baseline comparison.

```python
Pipeline([
    ('preprocessor', ColumnTransformer([
        ('numeric', Pipeline([
            ('knn_imputer', KNNImputer(n_neighbors=5)),
            ('scaler', RobustScaler())
        ]), numeric_features),
        ('categorical',
            OneHotEncoder(handle_unknown='ignore'),
            categorical_features)
    ])),
    ('regressor', RandomForestRegressor(
        n_estimators=500,
        max_depth=20,
        min_samples_split=2,
        max_features='sqrt',
        random_state=42,
        n_jobs=-1
    ))
])
```

### Feature Set (21 total)

**Numeric features (18):**

| Feature | Description |
|---------|-------------|
| `Gr Liv Area` | Above-grade living area (sq ft) |
| `Total Bsmt SF` | Total basement area (sq ft) |
| `1st Flr SF` | First floor area (sq ft) |
| `Garage Area` | Garage area (sq ft) |
| `Lot Area` | Total lot size (sq ft) |
| `Overall Qual` | Overall material and finish quality (1–10) |
| `Overall Cond` | Overall condition rating (1–10) |
| `Year Built` | Original year of construction |
| `House_Age` | 2026 minus Year Built (engineered) |
| `Bedroom AbvGr` | Bedrooms above ground |
| `Full Bath` | Full bathrooms |
| `Half Bath` | Half bathrooms |
| `Kitchen AbvGr` | Kitchens above ground |
| `TotRms AbvGrd` | Total rooms above ground |
| `Garage Cars` | Garage capacity in cars |
| `Quality_Area` | Overall Qual × Gr Liv Area (engineered) |
| `Quality_Condition_Score` | Overall Qual × Overall Cond (engineered) |
| `Total_Floor_Area` | 1st Flr SF + Total Bsmt SF (engineered) |

**Categorical features (3):**

| Feature | Options |
|---------|---------|
| `Neighborhood` | CollgCr, Veenker, Crawfor, NoRidge, Mitchel, Somerst, NWAmes, OldTown, BrkSide, Sawyer, NridgHt, NAmes, Blmngtn, BrDale, IDOTRR, MeadowV |
| `Bldg Type` | 1Fam, 2FmCon, Duplex, TwnhsE, TwnhsI |
| `House Style` | 2Story, 1Story, 1.5Fin, 1.5Unf, SFoyer, SLvl |

### Design decisions

- **KNN Imputation** preserves feature relationships better than simple median fill
- **RobustScaler** handles outliers in price data better than StandardScaler
- **Random Forest** captures the non-linear interactions between quality, area, and price
- **500 estimators** reduces variance over the default 100 at minimal runtime cost
- **max\_features='sqrt'** reduces tree correlation, improving ensemble diversity
- **n\_jobs=-1** uses all available CPU cores during training and inference

---

## 12. AI Advisory Agent (Milestone 2)

The advisory agent is a stateful LangGraph `StateGraph`. It accepts property features and the ML-predicted price, runs through 7 sequential nodes, and produces a structured investment report.

### Agent State

Every node reads from and writes to a shared `PropertyAnalysisState` dataclass:

```python
@dataclass
class PropertyAnalysisState:
    # Inputs
    property_features: Dict[str, Any]
    user_preferences: Dict[str, Any]
    predicted_price: Optional[float]

    # Retrieved from RAG
    market_insights: List[Dict]
    comparable_properties: List[Dict]
    regulations: List[Dict]

    # Analysis outputs (filled by nodes 2-5)
    property_analysis: Optional[str]
    comparable_analysis: Optional[str]
    price_validation: Optional[Dict]
    market_position: Optional[Dict]

    # Final outputs (filled by nodes 6-7)
    recommendation: Optional[str]
    advisory_report: Optional[Dict[str, Any]]

    # Metadata
    analysis_timestamp: str
    errors: List[str]
```

### 7-Node Workflow

| # | Node | What it does |
|---|------|-------------|
| 1 | `retrieve_market_data` | Queries ChromaDB using neighborhood and property size. Populates `market_insights`, `comparable_properties`, and `regulations` in state. |
| 2 | `analyze_property` | Formats property features into context. Identifies key value drivers. If `OPENAI_API_KEY` is set, generates a GPT narrative. |
| 3 | `validate_price` | Pure mathematical validation. Calculates average price/sqft from comparables, computes expected price, measures % deviation from the ML prediction. No LLM involved. |
| 4 | `analyze_comparables` | Selects top 3 comparable properties. Compares size, location, condition, and price. Provides adjustment reasoning. |
| 5 | `analyze_market_position` | Assesses comparable data quality, counts available insights, and determines signal reliability. |
| 6 | `generate_recommendation` | Rule-based investment signal derived from the deviation % and market position quality. |
| 7 | `generate_advisory_report` | Compiles all node outputs into a single structured JSON report with a disclaimer. |

### Recommendation logic

```
Price deviation from ML prediction:
  <= 15%   -->  BUY              (price is aligned with market)
  15-25%   -->  HOLD             (requires deeper investigation)
  > 25%    -->  INVESTIGATE      (significant anomaly detected)
  No data  -->  INSUFFICIENT DATA
```

### Agent input format

```python
{
    "features": {
        "address": str,
        "neighborhood": str,
        "sqft": float,
        "bedrooms": int,
        "bathrooms": int,
        "year_built": int,
        "garage_cars": int,
        "quality": int,         # 1-10
        "condition": int        # 1-10
    },
    "preferences": {
        "investment_type": str, # "primary_residence" | "rental" | "flip"
        "risk_tolerance": str   # "low" | "medium" | "high"
    },
    "predicted_price": float    # output from src/model_utils.py
}
```

### Agent output format

```python
{
    "summary": {
        "property_address": str,
        "predicted_price": str,
        "key_finding": str
    },
    "comparable_analysis": str,
    "price_validation": {
        "comparable_avg_price": float,
        "price_per_sqft": float,
        "predicted_price": float,
        "deviation_percent": float,
        "signal": "REASONABLE | NEEDS_REVIEW | ANOMALY"
    },
    "recommendation": str,
    "disclaimer": str,
    "analysis_timestamp": str
}
```

### Hallucination reduction

Every LLM system prompt in `src/prompts.py` includes these hard constraints:

1. Only use facts from the provided property data and market data
2. Never fabricate neighborhood names, prices, or market statistics
3. If information is not available, state "Data not available" explicitly
4. Base all claims on quantifiable metrics
5. Do not speculate about future market trends without supporting data

Price validation (Node 3) is entirely mathematical — it is immune to hallucination by design, regardless of LLM availability.

---

## 13. RAG System

The RAG system (`src/rag_system.py`) manages a local ChromaDB instance that gives the agent grounded, fact-based context about neighborhoods and comparable properties.

### Three collections

**`market_insights`** — Neighborhood-level market conditions

```
Content:  Neighborhood name, avg price/sqft, inventory levels, days on market, typical price range
Metadata: { "type": "neighborhood", "neighborhood": str, "date": str }
Query by: Neighborhood name + property size
```

**`comparable_properties`** — Historical residential sales

```
Content:  Address, sqft, beds/baths, neighborhood, sale price, year built, garage details
Metadata: { "type": "comparable", "price": float, "sqft": float, "neighborhood": str }
Query by: Neighborhood + property type + size
```

**`regulations_trends`** — Local tax and zoning information

```
Content:  Property tax rates, homestead exemptions, zoning classifications
Metadata: { "type": "regulation", "category": str, "date": str }
Query by: Area + regulation category
```

### Initialize or reset the database

The database auto-initializes with sample data on the first run of the advisory tab. To manually reset it:

```bash
rm -rf ./chroma_db

python -c "
from src.rag_system import RealEstateRAG, initialize_sample_market_data
rag = RealEstateRAG()
initialize_sample_market_data(rag)
print('RAG system initialized with sample data')
"
```

### Scaling to a production dataset

The current setup uses a local Chroma instance with sample data. To connect to a real database with MLS data:

```python
import chromadb

# Replace the local client in src/rag_system.py with:
chroma_client = chromadb.HttpClient(
    host="your-chroma-server.example.com",
    port=8000
)
```

---

## 14. Web Application

The app runs at `http://localhost:8501` after `streamlit run app.py`.

### Tab 1: Price Prediction

Input controls:
- Living area, basement area, 1st floor area, garage area, lot area (sq ft)
- Overall quality and condition (1–10 sliders)
- Year built
- Bedrooms, bathrooms, kitchens
- Neighborhood, building type, house style (dropdowns)

Outputs:
- Predicted price (large display)
- Property summary card
- Model accuracy metrics in sidebar

### Tab 2: Model Performance

Displays:
- Test accuracy, precision, R² score
- Test MAE and RMSE
- Cross-validation mean R²
- Training sample count and number of features used

### Tab 3: AI Advisory

Inputs:
- Property address (optional, for report labeling)
- Investment type: primary residence / rental / flip
- Risk tolerance: low / medium / high
- Uses the same property data already entered in Tab 1

Outputs:
- Full structured advisory report
- Price validation result with deviation %
- Comparable properties analysis
- Investment recommendation with reasoning
- Downloadable as JSON or plain text

### Tab 4: How It Works

- Plain-language walkthrough of each agent node
- How to interpret BUY / HOLD / INVESTIGATE signals
- Guidance on when data quality affects confidence

### Tab 5: About

- Full project overview
- Technology stack
- Use cases (homebuyers, investors, agents, analysts)
- Data sources and accuracy context
- Legal disclaimers

---

## 15. Model Performance

| Metric | Training | Testing |
|--------|----------|---------|
| Accuracy | 96.40% | 90.45% |
| Precision | 98.28% | 88.87% |
| R² Score | 0.9828 | 0.8887 |
| MAE | $5,442 | $14,250 |
| RMSE | $7,655 | $20,343 |
| MAPE | 3.60% | 9.55% |

**Prediction tolerance:**
- 42.40% of predictions fall within 5% of actual price
- 68.87% fall within 10% of actual price

**5-Fold Cross-Validation:**

| Fold | R² Score |
|------|---------|
| 1 | 0.8746 |
| 2 | 0.9089 |
| 3 | 0.8909 |
| 4 | 0.8236 |
| 5 | 0.8964 |
| **Mean** | **0.8789 +/- 0.0298** |

**Error distribution:**
- Mean Absolute Error: $13,614
- Standard Deviation of Error: $13,344
- Maximum Error observed: $87,409

---

## 16. Deployment

### Streamlit Community Cloud (Recommended)

1. Fork this repository to your GitHub account
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Click "New app"
4. Select your forked repo, set main file to `app.py`
5. Under "Advanced settings" → "Secrets", add:
   ```
   OPENAI_API_KEY = "sk-your-key-here"
   ```
6. Click Deploy

> The model file (`models/model.pkl`) must exist in the repository before deploying, since Streamlit Cloud cannot run `src/train.py` at deploy time. Commit your trained model or add a startup script.

### HuggingFace Spaces

1. Create a new Space at [huggingface.co/spaces](https://huggingface.co/spaces)
2. Choose Streamlit as the SDK
3. Connect this GitHub repository
4. Go to Settings → Repository secrets → add `OPENAI_API_KEY`
5. Auto-deploys on every push to `main`

### Render (Free Tier)

1. Create a new Web Service at [render.com](https://render.com)
2. Connect your GitHub repository
3. Set build command:
   ```
   pip install -r requirements.txt && python src/train.py
   ```
4. Set start command:
   ```
   streamlit run app.py --server.port=10000 --server.headless=true
   ```
5. Add environment variable: `OPENAI_API_KEY` = your key
6. Deploy

---

## 17. Troubleshooting

**`ModuleNotFoundError: No module named 'sklearn'`**
```bash
pip install --upgrade scikit-learn
```

**`ModuleNotFoundError: No module named 'langgraph'`**
```bash
pip install -r requirements.txt
```

**App crashes immediately with FileNotFoundError**

The model files are missing. Train the model first:
```bash
python src/train.py
```

**Advisory tab returns "INSUFFICIENT DATA" for all properties**

The RAG database has no entries for that neighborhood. Reset and reinitialize:
```bash
rm -rf ./chroma_db
python -c "
from src.rag_system import RealEstateRAG, initialize_sample_market_data
rag = RealEstateRAG()
initialize_sample_market_data(rag)
"
```

**Streamlit app won't start**
```bash
streamlit cache clear
streamlit run app.py --logger.level=debug
```

**Predictions are slow**

Confirm `n_jobs=-1` is set in the RandomForest inside `src/train.py`. Retrain if it was missing. Also verify `@st.cache_resource` wraps `load_model()` in `app.py` so the model loads once per session, not on every user interaction.

**OpenAI API errors in the advisory tab**

- Verify `.env` exists and contains a valid `OPENAI_API_KEY`
- Check your OpenAI usage limits at [platform.openai.com/usage](https://platform.openai.com/usage)
- The agent falls back to rule-based analysis if the API call fails

---

## 18. Known Limitations

- **Geographic scope:** Trained on Ames, Iowa data only. Price predictions for properties in other cities or countries will be unreliable and should not be used for real decisions.
- **LLM narratives are optional:** Full natural language analysis requires an OpenAI API key. Without it, the advisory tab provides mathematical signals only.
- **Sample market data:** ChromaDB ships with synthetic neighborhood and comparable data. For actual investment research, replace this with real MLS data or public records.
- **No live market data:** The model does not account for current mortgage rates, recent market movements, or macroeconomic conditions.
- **No feedback or retraining loop:** Recommendation accuracy is not tracked. The agent does not learn from outcomes or user corrections.
- **Price ceiling:** The Ames dataset has a price ceiling around $550,000. Predictions for luxury properties above this range will underestimate significantly.

---

## 19. Versioning

This repository uses semantic commit messages. The `main` branch reflects the latest stable state covering both milestones.

**Commit type conventions:**

| Type | Meaning |
|------|---------|
| `feat:` | New feature added |
| `fix:` | Bug fixed |
| `docs:` | Documentation change |
| `refactor:` | Code restructured, no behavior change |
| `chore:` | Dependencies, config, or tooling |

**Key milestones in commit history:**

- Initial: Milestone 1 baseline with Linear Regression
- Added Random Forest pipeline with KNN imputation
- Achieved 90.45% test accuracy
- Milestone 2: LangGraph 7-node agent implementation
- Milestone 2: ChromaDB RAG with 3 collections
- Milestone 2: Advisory UI + downloadable report
- Final: Documentation pass and README update

---

## 20. References & Resources

| Resource | URL |
|----------|-----|
| Ames Housing Dataset | [kaggle.com/c/house-prices-advanced-regression-techniques](https://www.kaggle.com/c/house-prices-advanced-regression-techniques) |
| Scikit-Learn Docs | [scikit-learn.org](https://scikit-learn.org) |
| Streamlit Docs | [docs.streamlit.io](https://docs.streamlit.io) |
| LangGraph Docs | [langchain-ai.github.io/langgraph](https://langchain-ai.github.io/langgraph/) |
| LangChain Docs | [python.langchain.com](https://python.langchain.com) |
| ChromaDB Docs | [docs.trychroma.com](https://docs.trychroma.com) |
| OpenAI API Reference | [platform.openai.com/docs](https://platform.openai.com/docs) |
| GitHub Repository | [CosmicGalactus/real-estate-ml](https://github.com/CosmicGalactus/real-estate-ml) |

---

## 21. Disclaimer

This tool is for **educational and informational purposes only**. It is not financial, legal, or real estate advice.

Predictions are based on historical data from Ames, Iowa and do not account for current market conditions, recent transactions, individual property condition, or any buyer/seller-specific circumstances.

Always consult a licensed real estate professional, financial advisor, or legal counsel before making property investment decisions.

---

**License:** Educational use — Capstone Project

**Status:** Active

**Last Updated:** April 2026

**GitHub:** [CosmicGalactus/real-estate-ml](https://github.com/CosmicGalactus/real-estate-ml)
