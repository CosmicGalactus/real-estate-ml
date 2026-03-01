import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import KNNImputer
from sklearn.ensemble import RandomForestRegressor, VotingRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_absolute_percentage_error
import joblib
import json
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')


def load_data(path):
    df = pd.read_csv(path)
    print(f"Loaded: {df.shape[0]} rows, {df.shape[1]} cols")
    return df


def engineer_features(df):
    df = df.copy()
    
    for col in df.select_dtypes(include=[np.number]).columns:
        if df[col].isna().sum() > 0:
            df[col].fillna(df[col].median(), inplace=True)
    
    # Basic interactions and features
    if 'Gr Liv Area' in df.columns and 'Overall Qual' in df.columns:
        df['Quality_Area'] = df['Gr Liv Area'] * df['Overall Qual']
        df['Quality_Area_Squared'] = df['Quality_Area'] ** 2
    
    if 'Year Built' in df.columns:
        df['House_Age'] = 2026 - df['Year Built']
        df['House_Age_Squared'] = df['House_Age'] ** 2
    
    if 'Year Remod/Add' in df.columns:
        df['Years_Since_Remodel'] = 2026 - df['Year Remod/Add']
    
    if 'Total Bsmt SF' in df.columns and 'Gr Liv Area' in df.columns:
        df['Has_Basement'] = (df['Total Bsmt SF'] > 0).astype(int)
        df['Basement_Ratio'] = df['Total Bsmt SF'] / (df['Gr Liv Area'] + 1)
    
    if 'Garage Cars' in df.columns and 'Garage Area' in df.columns:
        df['Has_Garage'] = (df['Garage Cars'] > 0).astype(int)
        df['Garage_Efficiency'] = df['Garage Area'] / (df['Garage Cars'] + 1)
    
    if 'Overall Qual' in df.columns and 'Overall Cond' in df.columns:
        df['Quality_Condition_Score'] = df['Overall Qual'] * df['Overall Cond']
    
    if '1st Flr SF' in df.columns and '2nd Flr SF' in df.columns:
        df['Total_Floor_Area'] = df['1st Flr SF'] + df['2nd Flr SF']
    
    # Additional polynomial features
    if 'Gr Liv Area' in df.columns:
        df['Gr_Liv_Area_Squared'] = df['Gr Liv Area'] ** 2
        df['Gr_Liv_Area_Cubed'] = df['Gr Liv Area'] ** 3
    
    if 'Overall Qual' in df.columns:
        df['Overall_Qual_Squared'] = df['Overall Qual'] ** 2
    
    # Log transforms for skewed features
    for col in ['Gr Liv Area', 'Total Bsmt SF', 'Lot Area']:
        if col in df.columns:
            df[f'{col}_Log'] = np.log1p(df[col])
    
    return df


def prepare_features(df):
    df = df.dropna(subset=["SalePrice"])
    
    numeric_features = [
        "Gr Liv Area", "Total Bsmt SF", "1st Flr SF", "Garage Area", "Lot Area",
        "Overall Qual", "Overall Cond", "Year Built", "House_Age",
        "Bedroom AbvGr", "Full Bath", "Half Bath", "Kitchen AbvGr",
        "TotRms AbvGrd", "Garage Cars", "Quality_Area",
        "Quality_Condition_Score", "Total_Floor_Area"
    ]
    
    categorical_features = ["Neighborhood", "Bldg Type", "House Style"]
    
    numeric_features = [f for f in numeric_features if f in df.columns and df[f].dtype in ['int64', 'float64']]
    categorical_features = [f for f in categorical_features if f in df.columns]
    
    X = df[numeric_features + categorical_features]
    y = df["SalePrice"]
    
    # Stricter outlier removal - top 1% and bottom 1%
    Q1, Q3 = y.quantile(0.01), y.quantile(0.99)
    mask = (y >= Q1) & (y <= Q3)
    X, y = X[mask], y[mask]
    
    print(f"Features: {len(numeric_features)} numeric, {len(categorical_features)} categorical")
    print(f"Samples: {len(X)} | Price range: ${y.min():,.0f} - ${y.max():,.0f}")
    
    return X, y, numeric_features, categorical_features


def build_pipeline(numeric_features, categorical_features):
    numeric_pipeline = Pipeline([
        ("imputer", KNNImputer(n_neighbors=5)),
        ("scaler", StandardScaler())
    ])
    
    categorical_pipeline = Pipeline([
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False, min_frequency=2))
    ])
    
    preprocessor = ColumnTransformer([
        ("num", numeric_pipeline, numeric_features),
        ("cat", categorical_pipeline, categorical_features)
    ])
    
    # Ultra-optimized Random Forest
    rf_model = RandomForestRegressor(
        n_estimators=500,
        max_depth=30,
        min_samples_split=2,
        min_samples_leaf=1,
        max_features=0.8,
        random_state=42,
        n_jobs=-1,
        bootstrap=True,
        warm_start=False
    )
    
    # Ridge with optimized alpha
    ridge_model = Ridge(alpha=0.1)
    
    # Weighted ensemble  
    voting_model = VotingRegressor([
        ('rf', rf_model),
        ('ridge', ridge_model)
    ], weights=[0.9, 0.1])
    
    model = Pipeline([
        ("preprocessor", preprocessor),
        ("regressor", voting_model)
    ])
    
    return model


def train_and_evaluate(model, X_train, X_test, y_train, y_test):
    print("\nTraining ensemble model...")
    model.fit(X_train, y_train)
    
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    train_mae = mean_absolute_error(y_train, y_train_pred)
    test_mae = mean_absolute_error(y_test, y_test_pred)
    train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
    test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)
    test_mape = mean_absolute_percentage_error(y_test, y_test_pred)
    
    cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='r2')
    
    print("\n" + "="*60)
    print("RESULTS")
    print("="*60)
    print(f"{'Metric':<15} {'Train':<20} {'Test':<20}")
    print("-"*60)
    print(f"{'MAE':<15} ${train_mae:>18,.0f} ${test_mae:>18,.0f}")
    print(f"{'RMSE':<15} ${train_rmse:>18,.0f} ${test_rmse:>18,.0f}")
    print(f"{'R² Score':<15} {train_r2:>19.4f} {test_r2:>19.4f}")
    print("-"*60)
    print(f"\nCross-Validation (5-fold): {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
    for i, s in enumerate(cv_scores, 1):
        print(f"  Fold {i}: {s:.4f}")
    
    accuracy = 100 * (1 - test_mape)
    precision = test_r2 * 100
    
    print(f"\nAccuracy: {accuracy:.2f}%")
    print(f"Precision: {precision:.2f}%")
    print("="*60)
    
    return model, {
        'train_mae': train_mae,
        'test_mae': test_mae,
        'train_rmse': train_rmse,
        'test_rmse': test_rmse,
        'train_r2': train_r2,
        'test_r2': test_r2,
        'test_mape': test_mape,
        'accuracy': accuracy,
        'precision': precision,
        'cv_mean': cv_scores.mean(),
        'cv_std': cv_scores.std(),
    }


def save_model(model, metrics, X_train, y_train):
    joblib.dump(model, 'models/model.pkl')
    
    metrics['train_size'] = len(X_train)
    metrics['num_features'] = X_train.shape[1]
    metrics['timestamp'] = datetime.now().isoformat()
    metrics['model_name'] = 'Random Forest + Linear Regression Ensemble'
    metrics['description'] = 'Weighted voting ensemble (75% RF, 25% LR)'
    
    with open('models/metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)
    
    with open('models/model_info.txt', 'w') as f:
        f.write(f"Model: Random Forest + Linear Regression Ensemble\n")
        f.write(f"RF Estimators: 150 | Max Depth: 18\n")
        f.write(f"Weights: 75% RF, 25% LR\n")
        f.write(f"Test R²: {metrics['test_r2']:.4f}\n")
        f.write(f"Test Accuracy: {metrics['accuracy']:.2f}%\n")
        f.write(f"Test Precision: {metrics['precision']:.2f}%\n")
    
    print("\nModel saved!")


def main():
    print("="*60)
    print("REAL ESTATE PRICE PREDICTION")
    print("="*60)
    
    df = load_data('data/ames.csv')
    df = engineer_features(df)
    X, y, num_feat, cat_feat = prepare_features(df)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    print(f"Train/Test Split: {len(X_train)} / {len(X_test)}")
    
    model = build_pipeline(num_feat, cat_feat)
    trained_model, metrics = train_and_evaluate(model, X_train, X_test, y_train, y_test)
    
    save_model(trained_model, metrics, X_train, y_train)


if __name__ == "__main__":
    main()
