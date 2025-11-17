#!/usr/bin/env python3
"""
Validation script for Online Shoppers Analysis
This script runs all the analysis steps to ensure there are no errors
"""

print("="*80)
print("VALIDATION SCRIPT - ONLINE SHOPPERS PURCHASING INTENTION ANALYSIS")
print("="*80)

# Step 1: Fetch dataset
print("\n[1/8] Fetching dataset...")
try:
    from ucimlrepo import fetch_ucirepo

    online_shoppers_purchasing_intention_dataset = fetch_ucirepo(id=468)
    X = online_shoppers_purchasing_intention_dataset.data.features
    y = online_shoppers_purchasing_intention_dataset.data.targets

    print(f"✓ Dataset loaded: {X.shape} features, {y.shape} targets")
except Exception as e:
    print(f"✗ Error loading dataset: {e}")
    exit(1)

# Step 2: Import libraries
print("\n[2/8] Importing libraries...")
try:
    import pandas as pd
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend for validation
    import matplotlib.pyplot as plt
    import seaborn as sns

    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import (
        classification_report, confusion_matrix,
        roc_auc_score, roc_curve,
        accuracy_score, precision_score,
        recall_score, f1_score
    )

    from sklearn.linear_model import LogisticRegression
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.svm import SVC
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.naive_bayes import GaussianNB
    from xgboost import XGBClassifier

    import warnings
    warnings.filterwarnings('ignore')

    print("✓ All libraries imported successfully")
except Exception as e:
    print(f"✗ Error importing libraries: {e}")
    exit(1)

# Step 3: Data Exploration
print("\n[3/8] Exploring data...")
try:
    df = X.copy()
    df['Revenue'] = y

    print(f"  - Dataset shape: {df.shape}")
    print(f"  - Missing values: {df.isnull().sum().sum()}")
    print(f"  - Duplicates: {df.duplicated().sum()}")
    print(f"  - Revenue distribution:\n{df['Revenue'].value_counts()}")

    print("✓ Data exploration complete")
except Exception as e:
    print(f"✗ Error in data exploration: {e}")
    exit(1)

# Step 4: Data Preprocessing
print("\n[4/8] Preprocessing data...")
try:
    df_processed = df.copy()
    X_prep = df_processed.drop('Revenue', axis=1)
    y_prep = df_processed['Revenue']

    # One-hot encoding
    month_encoded = pd.get_dummies(X_prep['Month'], prefix='Month', drop_first=True)
    visitor_encoded = pd.get_dummies(X_prep['VisitorType'], prefix='VisitorType', drop_first=True)
    X_prep['Weekend'] = X_prep['Weekend'].astype(int)
    X_prep = X_prep.drop(['Month', 'VisitorType'], axis=1)
    X_prep = pd.concat([X_prep, month_encoded, visitor_encoded], axis=1)
    y_prep = y_prep.astype(int)

    print(f"  - Processed features shape: {X_prep.shape}")
    print(f"  - Total features: {X_prep.shape[1]}")

    print("✓ Data preprocessing complete")
except Exception as e:
    print(f"✗ Error in preprocessing: {e}")
    exit(1)

# Step 5: Train-Test Split & Scaling
print("\n[5/8] Splitting and scaling data...")
try:
    X_train, X_test, y_train, y_test = train_test_split(
        X_prep, y_prep, test_size=0.2, random_state=42, stratify=y_prep
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns, index=X_train.index)
    X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns, index=X_test.index)

    print(f"  - Training set: {X_train_scaled.shape}")
    print(f"  - Testing set: {X_test_scaled.shape}")

    print("✓ Splitting and scaling complete")
except Exception as e:
    print(f"✗ Error in splitting/scaling: {e}")
    exit(1)

# Step 6: Model Training
print("\n[6/8] Training models...")
try:
    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        'Decision Tree': DecisionTreeClassifier(random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
        'XGBoost': XGBClassifier(n_estimators=100, random_state=42, eval_metric='logloss'),
        'K-Nearest Neighbors': KNeighborsClassifier(n_neighbors=5),
        'Naive Bayes': GaussianNB(),
        'Support Vector Machine': SVC(kernel='rbf', random_state=42, probability=True)
    }

    results = []
    trained_models = {}

    for name, model in models.items():
        print(f"  - Training {name}...", end=" ")
        model.fit(X_train_scaled, y_train)

        y_pred = model.predict(X_test_scaled)
        y_pred_proba = model.predict_proba(X_test_scaled)[:, 1] if hasattr(model, 'predict_proba') else None

        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_pred_proba) if y_pred_proba is not None else None

        results.append({
            'Model': name,
            'Accuracy': accuracy,
            'Precision': precision,
            'Recall': recall,
            'F1-Score': f1,
            'ROC-AUC': roc_auc
        })

        trained_models[name] = model
        print(f"✓ F1={f1:.4f}")

    print("✓ All models trained successfully")
except Exception as e:
    print(f"✗ Error in model training: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

# Step 7: Model Evaluation
print("\n[7/8] Evaluating models...")
try:
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('F1-Score', ascending=False).reset_index(drop=True)

    print("\n" + "="*90)
    print("MODEL PERFORMANCE COMPARISON")
    print("="*90)
    print(results_df.to_string(index=False))
    print("="*90)

    best_model_name = results_df.iloc[0]['Model']
    print(f"\n🏆 BEST MODEL: {best_model_name}")
    print(f"   F1-Score: {results_df.iloc[0]['F1-Score']:.4f}")
    print(f"   Accuracy: {results_df.iloc[0]['Accuracy']:.4f}")

    print("\n✓ Model evaluation complete")
except Exception as e:
    print(f"✗ Error in model evaluation: {e}")
    exit(1)

# Step 8: Feature Importance
print("\n[8/8] Analyzing feature importance...")
try:
    tree_based_models = ['Random Forest', 'Gradient Boosting', 'XGBoost', 'Decision Tree']

    for model_name in tree_based_models:
        if model_name in trained_models:
            model = trained_models[model_name]
            if hasattr(model, 'feature_importances_'):
                importances = model.feature_importances_
                feature_imp_df = pd.DataFrame({
                    'Feature': X_train.columns,
                    'Importance': importances
                }).sort_values('Importance', ascending=False).head(5)

                print(f"\n  Top 5 features for {model_name}:")
                for idx, row in feature_imp_df.iterrows():
                    print(f"    {row['Feature']}: {row['Importance']:.4f}")

    print("\n✓ Feature importance analysis complete")
except Exception as e:
    print(f"✗ Error in feature importance: {e}")
    exit(1)

# Final Summary
print("\n" + "="*80)
print("✓✓✓ VALIDATION COMPLETE - ALL STEPS PASSED! ✓✓✓")
print("="*80)
print("\nSummary:")
print(f"  - Dataset: {df.shape}")
print(f"  - Features after encoding: {X_prep.shape[1]}")
print(f"  - Models trained: {len(models)}")
print(f"  - Best model: {best_model_name} (F1-Score: {results_df.iloc[0]['F1-Score']:.4f})")
print("\nThe notebook is ready to use!")
