# Deployment Guide - Milestone 2

## Quick Deployment to Streamlit Cloud

### Prerequisites
- GitHub account
- Streamlit Community Cloud account (free)
- Repository pushed to GitHub

### Step 1: Prepare Repository

```bash
# From project root
cd /Users/prakharsrivastava/real-estate-ml

# Initialize git (if not done)
git init

# Add all files
git add .

# Commit
git commit -m "Milestone 2: Add AI advisory agent"

#  Push to GitHub
git remote add origin https://github.com/YOUR_USERNAME/real-estate-ml.git
git push -u origin main
```

### Step 2: Deploy on Streamlit Cloud

1. Go to https://share.streamlit.io
2. Click "New app"
3. Select your repository:
   - Repository: `YOUR_USERNAME/real-estate-ml`
   - Branch: `main`
   - Main file path: `real-estate-ml/app.py`
4. Click "Deploy"

### Step 3: Configure Secrets (if using LLM APIs)

In Streamlit Cloud dashboard:
1. Go to app settings
2. Add secrets:
   ```toml
   [secrets]
   openai_api_key = "sk-..."
   anthropic_api_key = "..."
   ```

---

## Testing After Deployment

### Manual Tests

1. **Navigate Tabs**
   - ✓ Price Prediction tab loads
   - ✓ Model Performance displays
   - ✓ AI Advisory tab accessible
   - ✓ How It Works displays
   - ✓ About section visible

2. **Price Prediction (Milestone 1)**
   - ✓ Input all property details
   - ✓ Click "Predict Price"
   - ✓ Price prediction displays correctly
   - ✓ Metrics show accurate values

3. **AI Advisory (Milestone 2)**
   - ✓ Fill property details in advisory tab
   - ✓ Set investment preferences
   - ✓ Click "Generate Advisory Report"
   - ✓ Wait for analysis (650ms typical)
   - ✓ Report displays with all sections:
     - Summary with predicted value
     - Price validation analysis
     - Comparable properties analysis
     - Investment recommendation
     - Disclaimer visible
   - ✓ Download buttons work (JSON & Text)

### Performance Monitoring

- Monitor cold start time (first load)
- Track query latency in Streamlit analytics
- Watch memory usage during RAG queries

---

## Troubleshooting Deployment

### Issue: "Module not found" errors

**Fix**: Ensure `src/__init__.py` exists

```bash
touch /Users/prakharsrivastava/real-estate-ml/real-estate-ml/src/__init__.py
```

### Issue: Chroma database errors

**Fix**: Chroma persists locally; may need to be recreated on first run

In `advisory_ui.py`:
```python
@st.cache_resource
def init_agent_system():
    # Cached, so Chroma initializes once per day
    rag = RealEstateRAG()  # Creates ./chroma_db
    initialize_sample_market_data(rag)
    return RealEstateAdvisoryAgent(rag_system=rag), rag
```

### Issue: Slow response times

**Cause**: RAG queries on first run

**Fix**: Increase caching, optimize Chroma queries

### Issue: Memory limit exceeded

**Cause**: Large datasets or model inference

**Fix**: 
- Streamlit Cloud: 1GB RAM limit
- Implement data streaming
- Use model quantization

---

## Local Testing Before Deployment

### 1. Test Streamlit App Locally

```bash
cd /Users/prakharsrivastava/real-estate-ml/real-estate-ml

# Run app
streamlit run app.py
```

Expected output:
```
  You can now view your Streamlit app in your browser.

  Local URL: http://localhost:8501
  Network URL: http://192.168.x.x:8501
```

### 2. Test Each Tab

Click through all tabs and verify functionality

### 3. Test Advisory Agent

```python
# Quick test script
import sys
sys.path.insert(0, 'src')
from agent import RealEstateAdvisoryAgent
from rag_system import RealEstateRAG, initialize_sample_market_data

rag = RealEstateRAG()
initialize_sample_market_data(rag)
agent = RealEstateAdvisoryAgent(rag_system=rag)

test_property = {
    "features": {
        "address": "500 Test St, Northridge",
        "neighborhood": "Northridge",
        "sqft": 2000,
        "bedrooms": 3,
        "bathrooms": 2,
        "year_built": 2005,
        "garage_cars": 2
    },
    "preferences": {"investment_type": "long_term", "risk_tolerance": "medium"},
    "predicted_price": 350000
}

result = agent.analyze_property(test_property)
print("Success:", result["success"])
print("Report:", result["advisory_report"]["summary"])
```

---

## Post-Deployment Monitoring

### 1. Error Logs

In Streamlit Cloud dashboard:
- View "Logs" tab for errors
- Check for timeout issues
- Monitor API rate limits

### 2. Analytics

- Track page visits
- Monitor session duration
- Identify bottlenecks

### 3. User Feedback

- Add feedback form (optional)
- Monitor GitHub issues
- Collect usage patterns

---

## Environment Variables (If Using APIs)

Create `.env` file locally:
```
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=...
HF_TOKEN=...
```

For Streamlit Cloud, add via Settings → Secrets

---

## Performance Optimization

### Current Bottlenecks

1. First RAG initialization: ~500ms
2. Model prediction loading: depends on model size
3. Chroma queries: ~50-100ms per query

### Optimizations Applied

1. ✓ Cached RAG initialization with `@st.cache_resource`
2. ✓ Cached model loading with `@st.cache_resource`
3. ✓ Chroma vector indexing

### Further Optimizations

1. Add Redis caching for frequent queries
2. Implement async processing
3. Use model quantization
4. Optimize feature engineering

---

## Scaling for Production

### Current Setup
- Streamlit Cloud (free tier): 1GB RAM, medium CPU
- Chroma: In-memory + persistent local storage
- No external APIs

### Production Setup
1. Horizontal scaling with Kubernetes
2. Redis cache layer
3. External RAG database (PostgreSQL + pgVector)
4. LLM API rate limiting
5. Load balancing

```
User Requests
      ↓
   [LB]
   ↙ ↓ ↘
[App1] [App2] [App3]
   ↘ ↓ ↙
  [Cache/Redis]
      ↓
[RAG DB - pgVector]
      ↓
[LLM API]
```

---

## GitHub Actions CI/CD (Optional)

Create `.github/workflows/deploy.yml`:

```yaml
name: Deploy to Streamlit Cloud

on:
  push:
    branches: [main]

jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Run tests
        run: |
          pip install -r real-estate-ml/requirements.txt
          pytest tests/
      - name: Deploy to Streamlit Cloud
        run: |
          streamlit deploy \
            --app-path real-estate-ml/app.py \
            --project-id ${{ secrets.STREAMLIT_PROJECT_ID }}
```

---

## Completion Checklist

### Before Deployment
- [ ] All tests pass locally
- [ ] No hardcoded secrets
- [ ] `.gitignore` configured properly
- [ ] `requirements.txt` up to date
- [ ] `.streamlit/config.toml` present
- [ ] `README.md` updated
- [ ] All documentation complete

### Deployment
- [ ] Repository pushed to GitHub
- [ ] Streamlit Cloud account created
- [ ] App deployed successfully
- [ ] Public URL accessible
- [ ] All tabs functional
- [ ] AI Advisory working end-to-end

### Post-Deployment
- [ ] Test all functionality
- [ ] Monitor logs for errors
- [ ] Document any issues
- [ ] Share public URL with stakeholders
- [ ] Collect feedback

---

## Public URL

Once deployed on Streamlit Cloud:
- Share: `https://share.streamlit.io/YOUR_USERNAME/real-estate-ml/main/real-estate-ml/app.py`
- Include in documentation
- Update README

---

## Rolling Back

If issues occur:
1. GitHub: Revert to previous commit
2. Streamlit: Click "Rerun" to reset, or "Delete" to remove
3. Create new deployment with fixed version

```bash
git revert HEAD
git push
# Streamlit automatically deploys new version
```

---

## Next Steps After Deployment

1. Record demo video (15-20 mins)
2. Create YouTube walkthrough
3. Update GitHub documentation
4. Prepare presentation slides
5. Share with instructors/stakeholders

---

## Support

- **Issues**: Create GitHub issues
- **Documentation**: See `AGENT_DOCUMENTATION.md`
- **Testing**: Run `pytest tests/`
- **Local Dev**: `streamlit run app.py`
