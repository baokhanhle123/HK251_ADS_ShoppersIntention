# ShopSmart E-commerce: Online Shoppers Purchasing Intention Analysis

## HK251 - Applied Data Science Course Project

A comprehensive data science project analyzing online shoppers' purchasing intention using RapidMiner Studio. This project follows the CRISP-DM methodology and includes a detailed Vietnamese report.

## Business Context

**ShopSmart E-commerce** is a fictional Vietnamese e-commerce platform facing a common challenge: only ~15% of website visitors complete a purchase. This project builds machine learning models to predict purchasing intention based on visitor behavior, enabling targeted marketing interventions.

## Dataset

**Source:** [UCI Machine Learning Repository - Online Shoppers Purchasing Intention Dataset](https://archive.ics.uci.edu/dataset/468/online+shoppers+purchasing+intention+dataset)

| Property | Value |
|----------|-------|
| Total Sessions | 12,330 |
| Features | 17 (10 numerical, 7 categorical) |
| Target Variable | Revenue (TRUE/FALSE) |
| Class Distribution | 84.5% No Purchase, 15.5% Purchase |

## Project Structure

```
HK251_ADS_ShoppersIntention/
├── Report/                         # LaTeX Report (Vietnamese)
│   ├── main.tex                    # Main document with title page
│   ├── sections/
│   │   ├── 1_Introduction.tex      # Business context & problem statement
│   │   ├── 2_Methods.tex           # Data understanding & preparation
│   │   ├── 3_Experiments.tex       # Modeling & evaluation
│   │   ├── 4_Improvement.tex       # Discussion & recommendations
│   │   └── 5_Conclusion.tex        # Summary & future work
│   └── references.bib              # Bibliography
├── Project_Requirement/            # Course requirements (Vietnamese)
├── ADS_Assignment.ipynb            # Reference Python implementation
├── sakar2018.pdf                   # Original research paper
└── README.md                       # This file
```

## Methodology

### Framework: CRISP-DM
The project follows Cross-Industry Standard Process for Data Mining with phases:
1. Business Understanding
2. Data Understanding
3. Data Preparation
4. Modeling
5. Evaluation
6. Deployment (recommendations)

### Tool: RapidMiner Studio
All modeling and evaluation performed using RapidMiner operators.

### Algorithms (4 Models)
| Algorithm | RapidMiner Operator |
|-----------|---------------------|
| Decision Tree | `Decision Tree` |
| Random Forest | `Random Forest` |
| K-Nearest Neighbors | `k-NN` |
| Logistic Regression | `Logistic Regression` |

### Class Imbalance Handling
Two techniques compared:
- **SMOTE** (Synthetic Minority Over-sampling Technique)
- **Random Oversampling**

### Validation
- 5-fold Stratified Cross-Validation
- Metrics: Accuracy, Precision, Recall, F1-Score, ROC-AUC

## Report Structure (Vietnamese)

| Section | Title | Content |
|---------|-------|---------|
| 1 | Giới thiệu (Introduction) | ShopSmart business case, problem statement, objectives |
| 2 | Phương pháp (Methods) | Data understanding, EDA, preprocessing, SMOTE/Oversampling |
| 3 | Thí nghiệm (Experiments) | Model training, evaluation, comparison |
| 4 | Thảo luận (Discussion) | Results interpretation, business implications |
| 5 | Kết luận (Conclusion) | Summary, contributions, future work |

## Key Findings

### Most Important Features
1. **PageValues** - Strongest predictor (value of pages visited)
2. **ExitRates** - Negative indicator (page exit probability)
3. **BounceRates** - Negative indicator (single-page sessions)
4. **ProductRelated_Duration** - Positive correlation with purchase

### Model Performance
- **Best Model:** Random Forest (balanced precision/recall)
- **SMOTE:** Better balance between Precision and Recall
- **Oversampling:** Higher Recall but lower Precision

### Business Recommendations
1. Optimize PageValue through strategic content placement
2. Reduce exit rates on product pages
3. Target visitors with high ProductRelated_Duration
4. Focus marketing during peak months (November, May)

## Compile LaTeX Report

```bash
cd Report
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

Or use your preferred LaTeX editor (TeXstudio, Overleaf, etc.)

## Reference Python Implementation

The `ADS_Assignment.ipynb` notebook provides a Python reference implementation using scikit-learn. This was used for initial exploration but the final analysis uses RapidMiner.

```bash
# To run the Python notebook (optional)
pip install ucimlrepo pandas numpy matplotlib seaborn scikit-learn
jupyter notebook ADS_Assignment.ipynb
```

## References

1. Sakar, C.O., Polat, S.O., Katircioglu, M. et al. (2019). Real-time prediction of online shoppers' purchasing intention using multilayer perceptron and LSTM recurrent neural networks. *Neural Computing and Applications*, 31, 6893-6908. https://doi.org/10.1007/s00521-018-3523-0

2. UCI Machine Learning Repository. Online Shoppers Purchasing Intention Dataset. https://archive.ics.uci.edu/dataset/468

3. RapidMiner Documentation. https://docs.rapidminer.com/

4. CRISP-DM Methodology. https://www.datascience-pm.com/crisp-dm-2/

## Course Information

- **Course:** HK251 - Applied Data Science
- **Project:** Online Shoppers Purchasing Intention Analysis
- **Tool:** RapidMiner Studio
