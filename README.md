# Online Shoppers Purchasing Intention - Data Science Analysis

## Dataset
**Source:** [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/468/online+shoppers+purchasing+intention+dataset)

- **Total Sessions:** 12,330
- **Features:** 17 (10 numerical, 8 categorical)
- **Target:** Revenue (Purchase vs No Purchase)
- **Class Distribution:** 84.5% No Purchase, 15.5% Purchase

## Project Structure

```
HK251_ADS_ShoppersIntention/
├── ADS_Assignment.ipynb    # Main analysis notebook
├── validate_analysis.py     # Validation script
├── sakar2018.pdf           # Research paper
└── README.md               # This file
```

## Notebook Contents

The `ADS_Assignment.ipynb` notebook contains a complete data science pipeline:

1. **Data Loading & Exploration**
   - Load dataset from UCI repository
   - Inspect data structure, missing values, duplicates
   - Descriptive statistics and initial analysis

2. **Exploratory Data Analysis (EDA)**
   - Distribution plots for numerical features
   - Correlation heatmap
   - Comparison between purchasers vs non-purchasers
   - Categorical feature analysis

3. **Data Preprocessing**
   - One-hot encoding for categorical variables
   - Train-test split (80/20) with stratification
   - Feature scaling using StandardScaler

4. **Model Building & Training**
   - 8 Classification models:
     - Logistic Regression
     - Decision Tree
     - Random Forest
     - Gradient Boosting
     - XGBoost
     - K-Nearest Neighbors
     - Naive Bayes
     - Support Vector Machine

5. **Model Evaluation & Comparison**
   - Performance metrics (Accuracy, Precision, Recall, F1-Score, ROC-AUC)
   - Confusion matrix
   - ROC curve
   - Classification report

6. **Feature Importance Analysis**
   - Top 15 features for each tree-based model
   - Visualization of feature importance

7. **Key Insights & Business Recommendations**
   - Summary of findings
   - Actionable business recommendations

## How to Run

### Option 1: Local Jupyter Notebook

```bash
# Install required packages
pip install ucimlrepo xgboost pandas numpy matplotlib seaborn scikit-learn

# Launch Jupyter
jupyter notebook ADS_Assignment.ipynb
```

### Option 2: Google Colab

1. Upload `ADS_Assignment.ipynb` to Google Colab
2. Run all cells sequentially

### Option 3: Validation Script

```bash
# Run the validation script to test the pipeline
python validate_analysis.py
```

## Requirements

```
pandas>=1.0.0
numpy>=1.20.0
matplotlib>=3.3.0
seaborn>=0.11.0
scikit-learn>=0.24.0
xgboost>=1.4.0
ucimlrepo>=0.0.7
```

## Key Findings

### Most Important Features
1. **PageValues** - Strongest predictor of purchase behavior
2. **ProductRelated_Duration** - Time spent on product pages
3. **ExitRates** - Exit rate metrics
4. **BounceRates** - Bounce rate metrics
5. **Month** - Seasonal shopping patterns

### Business Recommendations
1. Optimize page value metrics
2. Enhance product page engagement
3. Reduce exit and bounce rates
4. Target returning visitors
5. Focus on peak seasons (November, May)

## Model Performance

The notebook trains 8 different models and compares their performance. Typically:
- **Best performers:** Random Forest, XGBoost, Gradient Boosting
- **Metrics evaluated:** Accuracy, Precision, Recall, F1-Score, ROC-AUC
- **Best F1-Score:** ~0.65-0.70 (depending on random state)

## Notes

- The dataset has class imbalance (84.5% no purchase, 15.5% purchase)
- Stratified splitting is used to maintain class distribution
- All visualizations are included in the notebook
- The code is well-documented with comments

## References

- Sakar, C.O., Polat, S.O., Katircioglu, M. et al. Real-time prediction of online shoppers' purchasing intention using multilayer perceptron and LSTM recurrent neural networks. Neural Comput & Applic 31, 6893–6908 (2019). https://doi.org/10.1007/s00521-018-3523-0

## Author

Applied Data Science Course Project
HK251 - Online Shoppers Purchasing Intention Analysis
