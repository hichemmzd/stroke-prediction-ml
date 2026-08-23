# Stroke Prediction — Healthcare Data Analysis & ML

End-to-end data science project on the [Stroke Prediction Dataset](https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset) (Kaggle), covering the full pipeline: data cleaning → KPIs → visualization → predictive modeling.

## Dataset
5,110 patient records with 12 clinical/demographic attributes (age, gender, hypertension, heart disease, glucose level, BMI, smoking status, etc.), target variable: `stroke` (binary).

## 1. Data Cleaning
- `bmi` had 201 missing values (~3.9%). Investigation showed these missing values were **not random**: patients with missing BMI had a stroke rate of ~20% vs. ~5% overall — a strong MNAR (Missing Not At Random) signal.
- Imputed using the column median to preserve the full dataset without discarding this signal.

## 2. Key Performance Indicators (KPIs)
- Average age of stroke patients: **67 years**
- Patients with hypertension + heart disease + stroke: **13**
- Stroke cases by gender: **141 female / 108 male**
- Stroke rate by smoking status (highest: formerly smoked, ~7.9%; overlapping confidence intervals with "smokes" — difference not conclusive)
- Average glucose level: **132.5** (stroke) vs **104.8** (no stroke) — a clear gap; BMI showed only a small difference (30.1 vs 28.8)

## 3. Visualization
Five charts (age distribution, gender breakdown, correlation heatmap, glucose boxplot, smoking status bar chart) built with `seaborn`/`matplotlib`. The correlation heatmap showed no single variable strongly correlated with `stroke` in isolation (age: 0.25, the highest) — consistent with stroke risk being multifactorial.

## 4. Modeling
- **Preprocessing:** Label Encoding for binary categorical variables, One-Hot Encoding for multi-category variables (smoking status, work type) to avoid imposing false ordinal relationships.
- **Train/test split:** 80/20, stratified by nothing extra (baseline split), `random_state=42`.
- **Class imbalance:** `stroke=1` represents only ~5% of the data — handled via `class_weight='balanced'`.

| Model | Accuracy | Recall (stroke) | Precision (stroke) | F1 (stroke) |
|---|---|---|---|---|
| Logistic Regression | 0.80 | **0.71** | 0.19 | 0.30 |
| Random Forest | 0.93 | 0.06 | 0.25 | 0.10 |

**Key finding:** Random Forest scores much higher on accuracy but is effectively useless for the actual goal — it misses 58 of 62 real stroke cases. Logistic Regression, despite lower accuracy, catches 71% of real stroke cases. This is a textbook illustration of why **accuracy is a misleading metric on imbalanced datasets**, and why the choice of model/metric must match the real-world objective (in a medical screening context, missing a true positive is far costlier than a false alarm).

## Tools
Python, pandas, scikit-learn, seaborn, matplotlib.

## Author
Hichem Mezaad — Statistics Engineering student, ENSSEA.
