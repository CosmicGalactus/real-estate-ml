# Real Estate Advisory Agent - Technical Documentation

## Overview

The Real Estate Advisory Agent (Milestone 2) extends the ML-based price prediction system into an autonomous AI application that analyzes properties, retrieves market insights, and generates investment recommendations.

---

## Architecture

### System Components

```
┌─────────────────────────────────────────────────────────────┐
│                       Streamlit UI                          │
│  ├─ Price Prediction Tab (Milestone 1)                    │
│  ├─ Model Performance Tab                                  │
│  ├─ AI Advisory Tab (Calls Agent)                         │
│  ├─ How It Works Explanation                              │
│  └─ About & Documentation                                 │
└────────────────┬────────────────────────────────────────────┘
                 │
         ┌───────▼────────────┐
         │   Property Input   │
         │  + Preferences     │
         └────────┬───────────┘
                  │
         ┌────────▼──────────────────┐
         │ Real Estate Advisory Agent│ (LangGraph-based)
         │  ├─ state_graph.py        │
         │  └─ 7-Node Workflow       │
         └────────┬──────────────────┘
                  │
         ┌────────┴────────┐
         │                 │
    ┌────▼─────┐    ┌─────▼──────┐
    │ RAG System│    │   ML Model │
    │  (Chroma) │    │  (Inference)
    └───────────┘    └────────────┘
         │
    ┌────▼──────────────┐
    │ Market Insights   │
    │ Comparable Props  │
    │ Regulations       │
    └───────────────────┘
```

### Technology Stack

- **Framework**: LangGraph (agent orchestration)
- **RAG**: Chroma (vector database)
- **LLM Integration**: Hooks for OpenAI/local LLMs
- **UI**: Streamlit
- **ML Backend**: Scikit-Learn (existing model)
- **Language**: Python 3.8+

---

## LangGraph Workflow

### State Definition

```python
@dataclass
class PropertyAnalysisState:
    # Inputs
    property_features: Dict[str, Any]
    user_preferences: Dict[str, Any]
    predicted_price: Optional[float]
    
    # Retrieved Data
    market_insights: List[Dict]
    comparable_properties: List[Dict]
    regulations: List[Dict]
    
    # Analysis Results
    property_analysis: Optional[str]
    comparable_analysis: Optional[str]
    price_validation: Optional[str]
    market_position: Optional[str]
    
    # Outputs
    recommendation: Optional[str]
    advisory_report: Optional[Dict[str, Any]]
    
    # Metadata
    analysis_timestamp: str
    errors: List[str]
```

### 7-Node Workflow

#### Node 1: `retrieve_market_data`
**Purpose**: Fetch relevant market data from RAG system

**Process**:
- Query market insights based on neighborhood & property size
- Retrieve comparable properties from database
- Get applicable regulations for the area

**Output**: Populates `market_insights`, `comparable_properties`, `regulations`

**Hallucination Prevention**: Only retrieves data from pre-indexed knowledge base

---

#### Node 2: `analyze_property`
**Purpose**: Analyze property characteristics

**Process**:
- Format property features into readable context
- Compare with market data
- Identify key value drivers

**Output**: `property_analysis` string

**Hallucination Prevention**: System prompt enforced with explicit constraints

---

#### Node 3: `validate_price`
**Purpose**: Validate predicted price against comparables

**Process**:
1. Extract comparable prices and square footage
2. Calculate average price per sq ft
3. Calculate expected price for subject property
4. Compare with ML prediction
5. Flag if deviation > 15% (REASONABLE) or > 25% (ANOMALY)

**Output**: `price_validation` dict with signal flag

**Hallucination Prevention**: Mathematical validation, no speculation

---

#### Node 4: `analyze_comparables`
**Purpose**: Generate structured comparable property analysis

**Process**:
- Select top 3 comparable properties
- Compare specific features (location, size, condition)
- Provide price adjustment reasoning
- Determine market position

**Output**: `comparable_analysis` string

**Hallucination Prevention**: Only references actual property data from RAG

---

#### Node 5: `analyze_market_position`
**Purpose**: Assess property's position in market

**Process**:
- Count available market insights
- Assess comparable properties data quality
- Determine price signal reliability
- Identify market trends

**Output**: `market_position` dict with metrics

---

#### Node 6: `generate_recommendation`
**Purpose**: Generate investment recommendation

**Process**:
- Rule-based logic from price validation signal
- Consider market position
- Generate BUY/HOLD/INVESTIGATE recommendation

**Output**: `recommendation` string

**Logic**:
- REASONABLE signal → "BUY - Price aligned with market"
- NEEDS REVIEW → "HOLD - Requires deeper analysis"
- ANOMALY → "INVESTIGATE FURTHER"
- No data → "INSUFFICIENT DATA"

---

#### Node 7: `generate_advisory_report`
**Purpose**: Compile final structured advisory report

**Process**:
- Aggregate all analysis stages
- Format into structured report with:
  - **Summary**: Property overview + key finding
  - **Comps**: Comparable analysis
  - **Validation**: Price validation results
  - **Recommendation**: Investment decision
  - **Disclaimer**: Legal/financial notices

**Output**: `advisory_report` dict

---

## RAG System (Vector Database)

### Chroma Collections

#### Collection 1: `market_insights`
**Purpose**: Neighborhood trends and market conditions

**Example Documents**:
```
"Northridge neighborhood shows stable market with average 
price per sqft of $165. Market inventory is moderate with 
45-60 days on market. Average home price: $330,000-$360,000 
for 1500-2200 sqft properties."

Metadata: {
  "type": "neighborhood",
  "neighborhood": "Northridge",
  "date": "2026-04-19"
}
```

**Queries**: Neighborhood + bedrooms + sqft combinations

---

#### Collection 2: `comparable_properties`
**Purpose**: Historical sales data for price validation

**Example Documents**:
```
"123 Oak Street, 2100 sqft, 3BR/2BA, Northridge, 
Sold: $345,000, Built: 2005, Garage: 2-car, Pool: Yes"

Metadata: {
  "type": "comparable",
  "price": 345000,
  "sqft": 2100,
  "neighborhood": "Northridge"
}
```

**Queries**: Neighborhood + property type + size filters

---

#### Collection 3: `regulations_trends`
**Purpose**: Local regulations, zoning, and market trends

**Example Documents**:
```
"Property tax assessed at 0.75% of appraised value. 
Homestead exemption available for primary residences up to $50,000."

Metadata: {
  "type": "regulation",
  "category": "taxes",
  "date": "2026-04-19"
}
```

---

## Hallucination Reduction Strategies

### 1. Prompt Engineering with Constraints

**System Prompts Include**:
```python
IMPORTANT RULES:
1. ONLY use facts provided in the property data and market data
2. NEVER make up neighborhood names, prices, or market data
3. If information is not available, explicitly state "Data not available"
4. Base all recommendations on quantifiable metrics
5. Always include context and reasoning for each claim
6. Do not speculate about future market trends without data support
```

### 2. Data Grounding

- All text comes from RAG-indexed knowledge base
- Metadata tags ensure fact traceability
- No free-form generation without explicit grounding

### 3. Structured Output Format

- Fixed report sections (Summary, Comps, Action, Disclaimer)
- No narrative flexibility that could introduce speculation
- Explicit signal flags (REASONABLE/NEEDS REVIEW/ANOMALY)

### 4. Validation Layers

- Mathematical price validation (not LLM-based)
- Quantitative metrics for all claims
- Conservative recommendation logic

### 5. Explicit Data Availability Statements

- "Data not available" when knowledge gaps exist
- No inference to fill missing information
- Conservative default to "NEED MORE DATA"

---

## Integration Points

### 1. User Input → Agent

**Streamlit UI collects**:
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
        "quality": 1-10,
        "condition": 1-10
    },
    "preferences": {
        "investment_type": str,
        "risk_tolerance": str
    },
    "predicted_price": float  # from ML model
}
```

### 2. Agent → Advisory Report

**Output Structure**:
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
        "signal": "REASONABLE|NEEDS_REVIEW|ANOMALY"
    },
    "recommendation": str,
    "disclaimer": str,
    "analysis_timestamp": str
}
```

### 3. ML Model Integration

**Location**: `src/model_utils.py`

**Functions**:
- `ModelPredictor.predict()` - Get price prediction
- `prepare_property_features()` - Convert UI input to model features
- `get_property_options_from_dataset()` - Supply UI options

---

## Deployment Considerations

### 1. LLM Integration (Optional but Recommended)

For enhanced analysis beyond current mock implementation:

```python
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    model="gpt-4-turbo-preview"
)

# Use in prompts for enhanced analysis nodes
```

### 2. Database Scaling

**Current**: In-memory Chroma with sample data

**Production**:
```python
chroma_client = chromadb.HttpClient(
    host="chroma-server.example.com",
    port=8000
)
```

### 3. Caching Strategy

```python
@st.cache_resource
def init_agent_system():
    # Initialize once per session
    rag = RealEstateRAG()
    agent = RealEstateAdvisoryAgent(rag_system=rag)
    return agent
```

---

## Testing

### Unit Tests

```bash
pytest tests/test_agent.py
pytest tests/test_rag.py
pytest tests/test_model_utils.py
```

### Integration Tests

```python
# Test full workflow
agent = RealEstateAdvisoryAgent()
property_input = {...}
result = agent.analyze_property(property_input)
assert result["success"] == True
assert "advisory_report" in result
```

### Manual Testing

1. Start Streamlit: `streamlit run app.py`
2. Navigate to "AI Advisory" tab
3. Fill property details
4. Click "Generate Advisory Report"
5. Verify report generation
6. Download report in JSON/text format

---

## Performance Metrics

### Processing Time

- Retrieve market data: ~200ms
- Analyze property: ~100ms
- Validate price: ~50ms
- Analyze comparables: ~100ms
- Market position: ~50ms
- Generate recommendation: ~50ms
- Generate report: ~100ms
- **Total**: ~650ms typical

### Accuracy Measures

- Price validation deviance tracking
- Comparable match quality scoring
- Market insight relevance ranking

---

## Known Limitations

1. **LLM Not Integrated**: Currently uses mock analysis function
   - Fix: Integrate with OpenAI/Anthropic API
   
2. **Sample Data Only**: RAG uses small test dataset
   - Fix: Load real MLS data sources
   
3. **No Historical Trends**: Market predictions unavailable
   - Fix: Add time-series data and trend analysis
   
4. **Rule-Based Recommendations**: Not learning from outcomes
   - Fix: Add model retraining pipeline

---

## Future Enhancements

### Phase 1: LLM Integration
- [ ] Add OpenAI/Anthropic integration
- [ ] Enable true natural language analysis
- [ ] Implement advanced prompt strategies

### Phase 2: Data Expansion
- [ ] Integrate with MLS databases
- [ ] Add real-time market data feeds
- [ ] Expand comparable property database

### Phase 3: Learning Loop
- [ ] Track recommendation accuracy
- [ ] Implement feedback mechanism
- [ ] Fine-tune agent behavior

### Phase 4: Multi-Agent System
- [ ] Asset manager agent
- [ ] Financing advisor agent
- [ ] Risk analyst agent
- [ ] Multi-agent consensus

---

## Files Reference

| File | Purpose |
|------|---------|
| `src/agent.py` | Main LangGraph agent orchestrator |
| `src/rag_system.py` | Chroma RAG implementation |
| `src/prompts.py` | System prompts + hallucination reduction |
| `src/model_utils.py` | ML model integration utilities |
| `src/advisory_ui.py` | Streamlit UI components |
| `app.py` | Main Streamlit application |

---

## Support & Troubleshooting

### Import Issues

```bash
# Reinstall dependencies
pip install -r requirements.txt

# Check imports
python -c "from src.agent import RealEstateAdvisoryAgent; print('OK')"
```

### Chroma Issues

```bash
# Clear and reinit RAG
rm -rf ./chroma_db
python -c "from src.rag_system import RealEstateRAG, initialize_sample_market_data; rag = RealEstateRAG(); initialize_sample_market_data(rag)"
```

### Streamlit Issues

```bash
# Clear cache
streamlit cache clear

# Run with debugging
streamlit run app.py --logger.level=debug
```

---

## Contact & References

- **Project**: Intelligent Property Price Prediction & Agentic Real Estate Advisory
- **GitHub**: [Real-Estate-ML](https://github.com/CosmicGalactus/real-estate-ml)
- **Status**: Milestone 2 - In Progress
