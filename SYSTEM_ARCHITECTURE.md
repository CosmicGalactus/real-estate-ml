# System Architecture: Data Flow Pipeline

## Overview
The Real Estate ML system implements a two-stage pipeline: **ML-based price prediction** (Milestone 1) + **Agentic AI advisory** (Milestone 2). Data flows from user input → prediction → analysis → recommendations.

---

## Data Flow Architecture

### **Stage 1: Input Layer**
```
User Input (Web UI)
    ↓
Property Details:
    - Address
    - Square Footage
    - Bedrooms/Bathrooms
    - Year Built
    - Quality/Condition (1-10)
    - Garage Cars
    - Neighborhood
    - Investment Type
    - Risk Tolerance
```

### **Stage 2: Feature Engineering & Preparation**
```
Raw Property Data
    ↓
Feature Engineering (src/train.py)
    - House Age: CURRENT_YEAR - Year Built
    - Quality_Area: Gr Liv Area × Overall Quality
    - Basement Ratio: Total Bsmt SF / (Gr Liv Area + 1)
    - Garage Efficiency: Garage Area / (Garage Cars + 1)
    - Quality_Condition_Score: Quality × Condition
    - Total_Floor_Area: 1st Flr SF + 2nd Flr SF
    - Polynomial Features: Squared/Cubed versions
    - Log Transforms: Log(Gr Liv Area), Log(Total Bsmt SF)
    ↓
Feature Matrix (25 numeric + 3 categorical features)
    ↓
KNN Imputation (k=5 neighbors)
    ↓
StandardScaler (normalize to 0-1)
```

### **Stage 3: ML Prediction Pipeline**
```
Preprocessed Features
    ↓
Scikit-Learn Pipeline:
    ├── ColumnTransformer
    │   ├── Numeric Path: KNNImputer → StandardScaler
    │   └── Categorical Path: OneHotEncoder
    ↓
Ensemble Model (VotingRegressor):
    ├── 90% Random Forest Regressor
    │   - n_estimators: 300
    │   - max_depth: 22
    │   - Features: 25 numeric + encoded categorical
    │
    └── 10% Ridge Regression (alpha=0.1)
    ↓
Predicted Price (float)
    ↓ (saved temporarily for next stage)
```

### **Stage 4: Agent Analysis Pipeline**
```
Predicted Price + Property Features
    ↓
PropertyAdvisor Agent (src/agent.py)
    ├── STEP 1: Price Validation
    │   - Validate prediction ±15%
    │   - Check against market baseline
    │   └── OUTPUT: validation_result
    │
    ├── STEP 2: Property Analysis
    │   - Calculate Quality/Condition scores
    │   - Assess basement, garage, floor features
    │   - Compute efficiency ratios
    │   └── OUTPUT: property_analysis
    │
    ├── STEP 3: Market Assessment (Optional RAG)
    │   - Query: "Market trends in [Neighborhood]"
    │   - Retrieve from Chroma vector DB
    │   - Extract comparable properties
    │   └── OUTPUT: market_positioning
    │
    └── STEP 4: Recommendation Generation
        - Evaluate: Price Position, Market Viability
        - Generate: BUY / HOLD / PASS / INVESTIGATE
        └── OUTPUT: recommendation + confidence_score
    ↓
AgentState (with reasoning_history)
```

### **Stage 5: Knowledge Retrieval (RAG)**
```
User Query / Property Context
    ↓
RealEstateKnowledgeBase (src/rag_system.py)
    ↓
Semantic Search:
    - Chroma Vector DB (cosine similarity)
    - Persistent storage: ./chroma_db/
    ↓
Stored Knowledge:
    ├── Market trends (appreciation %, neighborhood viability)
    ├── Comparable properties (recent sales, prices, features)
    └── Regulations (local real estate rules)
    ↓
Retrieved Results (top_k=3 most similar)
    ↓ (returned to agent for context)
```

### **Stage 6: Report Generation**
```
Agent Analysis + Market Insights
    ↓
Structured Advisory Report:
    1. SUMMARY
       - Property overview
       - Market position
       - Key finding (investment thesis)
    
    2. COMPARABLE ANALYSIS
       - Similar Property 1-3
       - Price comparisons
       - Feature analysis
    
    3. INVESTMENT RECOMMENDATION
       - Recommendation (BUY/HOLD/PASS)
       - Supporting rationale (3+ data points)
       - Risk assessment
       - Timeline
    
    4. DISCLAIMER
       - Data currency
       - Past performance caveat
       - Professional advice note
    ↓
JSON Report Object
```

### **Stage 7: Output Layer (UI)**
```
Advisory Report
    ↓
Streamlit Web Interface (app.py + advisory_ui.py)
    ├── Tab 1: Price Prediction
    │   - Input form → Prediction → Display price
    │
    └── Tab 2: AI Advisory
        - Input form → Analysis → Display report
        - Show: Recommendation, confidence, reasoning
    ↓
User-Facing Dashboard
```

---

## Component Interaction Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                     Streamlit Web App                       │
│  (app.py - Entry point, UI rendering)                      │
└──────────┬──────────────────────────────────────────────────┘
           │
           ├─→ advisory_ui.render_advisory_tab()
           │       │
           │       ├─→ ModelPredictor.predict()
           │       │   └─→ models/model.pkl (Random Forest Ensemble)
           │       │       └─→ Predicted Price
           │       │
           │       └─→ PropertyAdvisor.analyze()
           │           ├─→ Validate Price
           │           ├─→ Analyze Properties
           │           ├─→ RAG Query (optional)
           │           │   └─→ RealEstateKnowledgeBase.search()
           │           │       └─→ chromadb (Chroma Vector DB)
           │           └─→ Generate Report
           │               └─→ Advisory Report (JSON)
           │
           └─→ render_how_it_works()
                   └─→ Educational content
```

---

## Data Transformations Summary

| Stage | Input | Process | Output |
|-------|-------|---------|--------|
| **1. Ingestion** | Property details (10 fields) | UI form collection | Raw property object |
| **2. Feature Eng.** | Raw property data | 14 engineered features | Feature matrix (28 features) |
| **3. Preprocessing** | Feature matrix | KNN imputation + scaling | Normalized features |
| **4. ML Prediction** | Normalized features | Random Forest + Ridge | Single price prediction |
| **5. Analysis** | Price + features | Agent reasoning (4 steps) | Analysis report |
| **6. RAG Retrieval** | Analysis query | Semantic search | Market insights (3 docs) |
| **7. Report Gen.** | Analysis + insights | Structure formatting | JSON advisory report |
| **8. UI Render** | JSON report | Streamlit formatting | Web dashboard |

---

## Error Handling & Fallbacks

```
If Trained Model Unavailable:
    ModelPredictor.predict() 
    → Falls back to heuristic estimation
    → Price = 300000 + (sqft × 100) + (quality × 5000)
    → Still passes to Agent for analysis

If RAG Query Fails:
    PropertyAdvisor.analyze()
    → Continues without market insights
    → Provides recommendation based on ML prediction alone
    → Logs warning to user

If Agent Confidence < 70%:
    Recommendation
    → Returns "INVESTIGATE FURTHER"
    → Recommends consulting real estate professional
    → Provides reasoning for low confidence
```

---

## Configuration Management

All configuration centralized in `src/config.py`:
- **Paths**: DATA_FILE, MODEL_FILE, CHROMA_DB_PATH
- **Model Hyperparameters**: RF_N_ESTIMATORS=300, RF_MAX_DEPTH=22, KNN_NEIGHBORS=5
- **UI Constants**: NEIGHBORHOODS, min/max ranges for all inputs
- **Data Constants**: CURRENT_YEAR, feature multipliers

Environment variables in `.env.example`:
- API keys for LLM services
- Optional overrides for paths and database connections

---

## Key Architectural Decisions

1. **Stateless UI**: Each prediction is independent, no session state needed
2. **Fallback Strategy**: System continues even if trained model unavailable
3. **Ensemble Model**: Combines Random Forest (90%) + Ridge (10%) for robustness
4. **RAG Optional**: Agent works with or without market knowledge base
5. **Configuration Externalized**: No hardcoding of constants/paths
6. **Multi-Stage Reasoning**: Agent mimics LangGraph workflows (ReAct pattern)

---

## Scalability Considerations

### Current Design
- Single user, synchronous processing
- Model loaded once at startup (cached)
- Chroma DB stored locally

### For Production Scale
- Add batch prediction endpoint (FastAPI/Flask)
- Load balance Streamlit app
- Move Chroma to hosted vector DB (Pinecone, Weaviate)
- Cache frequently searched properties
- Add Redis for prediction caching
- Implement async predictions (Celery + message queue)
- Add monitoring/logging (Application Insights)
