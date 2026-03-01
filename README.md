# Intelligent Property Price Prediction System

**Capstone Project: ML-Based Real Estate Analytics with Agentic AI Evolution**

A sophisticated machine learning system for property price prediction with plans to extend into an intelligent agentic real estate advisory assistant.

## 🎯 Project Overview

### Milestone 1: ML-Based Property Price Prediction (Current)

An advanced ensemble machine learning system that predicts real estate property prices with **90.81% accuracy** using state-of-the-art techniques.

**Key Features:**
- ✓ 4-Model Ensemble (Random Forest + Gradient Boosting + Ridge + AdaBoost)
- ✓ 25 Numeric + 3 Categorical Features
- ✓ Advanced Feature Engineering (14 engineered features)
- ✓ 90.81% Test Accuracy
- ✓ Interactive Streamlit Web Application
- ✓ Comprehensive Market Analytics

### Milestone 2: Agentic AI Advisory (Coming Soon)

Extension into autonomous AI agent for property analysis and investment recommendations using LangGraph.

---

## 📊 Model Performance

| Metric | Training | Testing |
|--------|----------|---------|
| **Accuracy** | 94.36% | 90.81% |
| **Precision** | 96.26% | 90.24% |
| **R² Score** | 0.9626 | 0.9024 |
| **MAE** | $8,624 | $13,614 |
| **RMSE** | $11,307 | $19,055 |
| **MAPE** | 5.64% | 9.19% |

**Prediction Quality:**
- 43.11% within 5% of actual price
- 72.99% within 10% of actual price

---

## 🏗️ System Architecture

```
Property Data → Preprocessing → Feature Engineering → Ensemble Model → Price Prediction
                      ↓              ↓                    ↓
              • Missing values  • 14 new features   • Random Forest
              • Scaling         • Interactions      • Gradient Boost
              • Encoding        • Log transforms    • Ridge Regression
                                                    • AdaBoost
```

---

## 📦 Installation

### Prerequisites

- Python 3.8 or higher
- pip or conda package manager
- 2GB RAM minimum

### Step-by-Step Setup

#### 1. Clone Repository

```bash
git clone <repository-url>
cd real-estate-ml
```

#### 2. Create Virtual Environment

```bash
# Using venv (Python 3.8+)
python -m venv venv

# Activate virtual environment
# On macOS/Linux:
source venv/bin/activate

# On Windows:
venv\Scripts\activate
```

#### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

**Installed Packages:**
- pandas>=1.3.0
- numpy>=1.21.0
- scikit-learn>=0.24.0
- streamlit>=1.28.0
- matplotlib>=3.4.0
- joblib>=1.0.0

#### 4. Verify Installation

```bash
python -c "import pandas, sklearn, streamlit; print('✓ All packages installed')"
```

---

## 🚀 Quick Start

### 1. Train the Model

Train on the Ames Housing dataset:

```bash
python src/train.py
```

**Output:**
- `models/model.pkl` - Trained ensemble model
- `models/metrics.json` - Detailed performance metrics
- `models/model_info.txt` - Model summary

**Training takes ~3-4 minutes on standard hardware**

### 2. Run the Web Application

```bash
streamlit run app.py
```

Access at: `http://localhost:8501`

### 3. Make Predictions

Use the interactive form to:
1. Enter property details (area, quality, age, etc.)
2. Select property characteristics (neighborhood, type, style)
3. Get instant price predictions
4. View prediction confidence metrics

---

## 📁 Project Structure

```
real-estate-ml/
│
├── data/
│   └── ames.csv                      # Ames Housing Dataset (2930 properties)
│
├── models/
│   ├── model.pkl                     # Trained ensemble model
│   ├── metrics.json                  # Performance metrics
│   └── model_info.txt                # Model summary
│
├── src/
│   ├── train.py                      # Training pipeline
│   ├── predict.py                    # Prediction utilities
│   └── utils.py                      # Helper functions
│
├── app.py                            # Streamlit web application
├── requirements.txt                  # Project dependencies
├── README.md                         # This file
├── PROJECT_DOCUMENT.md               # Comprehensive project documentation
├── .gitignore                        # Git ignore rules
└── config.yaml                       # Configuration file
```

---

## 🔧 Training Configuration

### Model Hyperparameters

**Random Forest:**
- Estimators: 300
- Max Depth: 22
- Min Samples Split: 2

**Gradient Boosting:**
- Estimators: 300
- Learning Rate: 0.04
- Max Depth: 5

**Ridge Regression:**
- Alpha: 0.1

**AdaBoost:**
- Estimators: 100
- Learning Rate: 0.05

### Feature Set (28 Total Features)

**Numeric Features (25):**
- Area: Gr Liv Area, Total Bsmt SF, 1st Flr SF, Garage Area, Lot Area
- Quality: Overall Qual, Overall Cond
- Age: Year Built, House Age, Years Since Remodel
- Rooms: Bedroom AbvGr, Full Bath, Half Bath, Kitchen AbvGr, TotRms AbvGrd
- Garage: Garage Cars, Garage Efficiency
- Engineered: Quality_Area, Quality_Area_Squared, Basement_Ratio, Quality_Condition_Score, Total_Floor_Area, Log transforms

**Categorical Features (3):**
- Neighborhood
- Building Type
- House Style

---

## 📊 Web Application Features

### Property Input Form
- Living area, basement area, garage details
- Quality and condition ratings
- Number of rooms and bathrooms
- Age and remodel information

### Prediction Output
- Estimated property price
- Confidence metrics (accuracy %)
- Error range estimate
- Market comparable analysis

### Analytics Dashboard
- Market trends visualization
- Feature importance charts
- Price distribution analysis
- Neighborhood comparisons

---

## 🔍 Model Details

### Ensemble Strategy

The system uses a **Voting Regressor** combining 4 diverse models:

1. **Random Forest** - Handles non-linear relationships, robust to outliers
2. **Gradient Boosting** - Sequentially improves predictions, captures complex patterns
3. **Ridge Regression** - Stable baseline, prevents overfitting
4. **AdaBoost** - Adaptive weighting, focuses on difficult predictions

Each model contributes equally to the final prediction.

### Data Preprocessing Pipeline

```python
Pipeline([
    ("knn_imputer", KNNImputer(n_neighbors=5)),     # Handle missing values
    ("scaler", RobustScaler()),                       # Scale features
    ("onehot_encoder", OneHotEncoder()),              # Encode categories
])
```

---

## 📈 Performance Validation

### Cross-Validation Results (5-Fold)

- Fold 1: R² = 0.8746
- Fold 2: R² = 0.9089
- Fold 3: R² = 0.8909
- Fold 4: R² = 0.8236
- Fold 5: R² = 0.8964
- **Mean: 0.8789 ± 0.0298**

### Prediction Error Analysis

- Mean Absolute Error: $13,614
- Standard Deviation: $13,344
- Maximum Error: $87,409

---

## 💾 Model Persistence

### Saving Model

```python
import joblib
joblib.dump(model, 'models/model.pkl')
```

### Loading Model

```python
model = joblib.load('models/model.pkl')
predictions = model.predict(X_new)
```

---

## 🌐 Deployment Options

### Option 1: HuggingFace Spaces (Recommended)

```bash
# 1. Create Space on HuggingFace
# 2. Select Streamlit template
# 3. Connect GitHub repository
# 4. Auto-deploy on push
```

### Option 2: Streamlit Community Cloud

```bash
# 1. Connect GitHub repository
# 2. Select this app.py file
# 3. Deploy with one click
# 4. Share public URL
```

### Option 3: Render (Free Tier)

```bash
# Setup: Create Web Service
# Build: pip install -r requirements.txt
# Start: streamlit run app.py --server.port=10000
```

---

## 📝 Git Workflow

### Commit Message Format

```
[TYPE] Short description (50 chars max)

Detailed explanation (72 chars per line)
- Change point 1
- Change point 2
```

### Types
- `feat:` New feature
- `fix:` Bug fix
- `docs:` Documentation
- `refactor:` Code refactoring
- `test:` Test additions
- `chore:` Maintenance

### Example Commits
```
feat: Implement ensemble model with 90.81% accuracy

- Added Random Forest with 300 estimators
- Added Gradient Boosting with learning rate 0.04
- Added Ridge and AdaBoost for diversity
- Achieved 9.19% MAPE on test set
- Cross-validation mean R²: 0.8789

fix: Handle missing values with KNN imputation

- Replaced median imputation with KNN (k=5)
- Improved feature relationships preservation
- Reduced error variance

docs: Add comprehensive project documentation

- Added PROJECT_DOCUMENT.md
- Updated README with setup guide
- Added architecture diagrams
- Documented all hyperparameters
```

---

## 🧪 Testing & Validation

### Unit Tests

```bash
python -m pytest tests/
```

### Model Validation

```bash
python src/validate_model.py
```

Checks:
- Model format and integrity
- Prediction consistency
- Performance metrics

---

## 📚 Documentation

- **PROJECT_DOCUMENT.md** - Comprehensive project documentation
- **README.md** - This file
- **src/train.py** - Docstrings for training pipeline
- **app.py** - Streamlit app documentation

---

## 🔐 Security & Best Practices

- ✓ No hardcoded credentials
- ✓ Input validation on predictions
- ✓ Error handling and logging
- ✓ Model versioning with timestamps
- ✓ Reproducible results (random_state=42)

---

## 🐛 Troubleshooting

### Issue: "ModuleNotFoundError: No module named 'sklearn'"

**Solution:**
```bash
pip install --upgrade scikit-learn
```

### Issue: "Streamlit app won't start"

**Solution:**
```bash
# Clear cache
rm -rf ~/.streamlit/
streamlit run app.py --logger.level=debug
```

### Issue: "Model predictions are slow"

**Solution:**
- Ensure n_jobs=-1 in RandomForest (uses all cores)
- Check system resources
- Consider model quantization for deployment

---

## 📞 Support & Contact

**Questions?** Check:
1. PROJECT_DOCUMENT.md for detailed information
2. Inline code documentation
3. Error messages and logs

---

## 📜 License

This project is for educational purposes as part of a capstone project.

---

## 🎓 Capstone Project Milestones

### ✓ Milestone 1: ML-Based Price Prediction (Current)
- Advanced ensemble model
- 90.81% accuracy
- Streamlit web application
- Comprehensive documentation

### ⏳ Milestone 2: Agentic AI Advisory (In Progress)
- LangGraph agent framework
- Autonomous market research
- Risk assessment module
- Investment recommendations

---

## 🚀 Future Enhancements

- [ ] Real-time market data integration
- [ ] Advanced visualization dashboard
- [ ] API endpoint for batch predictions
- [ ] Model explainability (SHAP values)
- [ ] A/B testing framework
- [ ] Agentic AI with LangGraph
- [ ] RAG-based market insights
- [ ] Multi-property portfolio analysis

---

**Project Status:** ✓ Active Development

**Last Updated:** March 1, 2026

**Team:** Capstone Project Team

**GitHub:** [Repository URL]

**Deployed:** [Deployment URL]

---

## 📖 Additional Resources

- [Scikit-Learn Documentation](https://scikit-learn.org)
- [Streamlit Documentation](https://docs.streamlit.io)
- [Ames Housing Dataset](https://www.kaggle.com/c/house-prices-advanced-regression-techniques)
- [LangGraph Documentation](https://langchain-ai.github.io/langgraph/)
- [Real Estate Market Analysis](https://www.zillow.com)
