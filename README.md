# Bank Marketing Classification Project

## 1. Project Overview & Business Problem
Banks and financial institutions in general run direct marketing campaigns via phone calls to promote term deposit subscriptions. Calling every client is expensive, time-consuming, and can cause customer fatigue, if poorly targeted.

The goal of this project is to build a machine learning classification pipeline in R that predicts **prior to placing a phone call** whether a client will subscribe to a term deposit (`SUBSCRIBED = yes` or `no`). By identifying high-propensity clients in advance, the bank can focus its outreach on the most promising leads and optimize campaign ROI.

---

## 2. Dataset & Preprocessing Pipeline
The dataset contains **39,883 observations** across **22 variables** detailing client demographics, previous campaign interactions, and macroeconomic indicators.

### Key Data Cleaning & Feature Engineering Steps
* **Preventing Data Leakage (`duration` removal):** The `duration` variable (call length) is only known *after* a phone call ends. Including it creates severe look-ahead bias and an unrealistic model for pre-call targeting. Therefore, `duration` was explicitly removed before modeling.
* **Feature Engineering (`newpdays`):** The raw `pdays` variable uses the special numeric code `999` for clients not previously contacted, which distorts numeric distributions. It was transformed into a 4-level categorical factor (`newpdays`): `no_previous_contact`, `up_to_10_days`, `more_than10`, and `NA`.
* **Multicollinearity Reduction:** Correlation analysis across numeric predictors using a threshold of **|r| ≥ 0.85** revealed strong multicollinearity among macroeconomic indicators:
  * `emp.var.rate` & `euribor3m` (r = 0.98)
  * `emp.var.rate` & `nr.employed` (r = 0.95)
  * `euribor3m` & `nr.employed` (r = 0.96)
  
  To eliminate redundancy, `nr.employed` was retained while `emp.var.rate` and `euribor3m` were dropped.
* **Data Cleaning:** Standardized inconsistent string casing (`otherproduct`), converted categorical variables into properly ordered R factors, and removed 7 exact duplicate rows, resulting in a clean modeling dataset of **39,876 observations** and **18 candidate predictors**.

---

## 3. Experimental Design & Evaluation Strategy

### Handling Class Imbalance
The target variable `SUBSCRIBED` is heavily imbalanced: **90.0% `no`** vs. **10.0% `yes`**. A naive classifier that predicts `no` for every client achieves **90.11% accuracy** (the No Information Rate) while failing to capture a single subscriber. 

Consequently, raw accuracy is an inadequate metric. Models in this project are tuned and evaluated using **F1-Score**, **Precision**, **Recall (Sensitivity)**, **Specificity**, **Balanced Accuracy**, and **AUC-ROC**.

### Data Splitting Workflow
Using a fixed random seed (`set.seed(100)`) for reproducibility, the cleaned dataset was partitioned into three stratified subsets:
1. **Training Set (60% — 23,926 rows):** Used to fit models, tune hyperparameters (such as Random Forest `ntree` and `mtry`), and generate candidate feature subsets via Recursive Feature Elimination (RFE).
2. **Test Set (20% — 7,975 rows):** Used to calibrate decision probability thresholds, evaluate candidate models, and select the best-performing model architecture.
3. **Validation Set (20% — 7,975 rows):** Held out as an unseen final benchmark to verify how the winning model generalizes.

---

## 4. Models Implemented & Test Set Results

### 1. Baseline Model: Logistic Regression (Threshold = 0.50)
A standard binomial Logistic Regression model was fitted using all 18 candidate predictors with a default probability cutoff of `0.50`.
* **Accuracy:** 90.42% (Balanced Accuracy: 57.56%)
* **Precision:** 55.27%
* **Sensitivity (Recall):** 16.60%
* **Specificity:** 98.53%
* **F1-Score:** 0.2554

*Takeaway:* While precision and specificity are high, the standard 0.50 threshold is far too conservative for imbalanced data, missing over 83% of actual subscribers.

### 2. Random Forest & Feature Permutation
* **Hyperparameter Tuning:** Grid search over `ntree` in (50, 100, 200, 300) and `mtry` in (2, 3, 4) using Out-Of-Bag (OOB) error rate, training the full model with `ntree = 300` and `mtry = 3`.
* **Threshold Optimization:** Sweeping probability cutoffs from `0.01` to `0.99` identified an optimal threshold of **0.18**, boosting the F1-Score to **0.4509** (Sensitivity: **49.43%**, Precision: **41.45%**, Specificity: **92.33%**, AUC: **0.7460**).
* **Interpretability & Feature Ablation:** Variable importance plots (`varImpPlot`), permutation shuffling, and Partial Dependence Plots (PDPs) highlighted `nr.employed`, `age`, `month`, `cons.conf.idx`, `job`, and `education` as primary drivers. Removing the lowest-impact features (`loan` and `default`) in Model 2 yielded an F1-Score of **0.4480** and AUC of **0.7460**, confirming Model 1 as the stronger Random Forest specification.

### 3. Naive Bayes with Custom Recursive Feature Elimination (RFE)
A custom backward Recursive Feature Elimination (RFE) algorithm was built:
* Starting with all 18 predictors, the algorithm iteratively removed one variable at a time on the training set, evaluating probability thresholds (`0.05` to `0.50`) to maximize F1-Score (breaking ties with AUC).
* All 18 resulting variable subsets were evaluated on the test set.
* **Winning Subset (Subset #10 — 9 Variables):** Retained `age`, `cons.price.idx`, `nr.employed`, `education`, `default`, `contact`, `month`, `day_of_week`, and `newpdays` at an optimal threshold of **0.43**.
* **Performance:** Achieved the highest discriminatory power (**AUC = 0.7625**) and higher **Precision (41.76%)** than Random Forest (**41.45%**), alongside an **F1-Score of 0.4473** (Sensitivity: **48.16%**, Specificity: **92.62%**, Accuracy: **88.23%**) while cutting the feature space in half.

### Model Comparison Summary (Test Set)

| Model | Features Used | Threshold | Precision | Recall (Sensitivity) | Specificity | F1-Score | AUC |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression (Baseline)** | 18 | 0.50 | 55.27% | 16.60% | 98.53% | 0.2554 | — |
| **Random Forest (Model 1)** | 18 | 0.18 | 41.45% | 49.43% | 92.33% | 0.4509 | 0.7460 |
| **Random Forest (Model 2)** | 16 | 0.17 | 41.15% | 49.18% | 92.28% | 0.4480 | 0.7460 |
| **Naive Bayes (RFE Champion)** | **9** | **0.43** | **41.76%** | **48.16%** | **92.62%** | **0.4473** | **0.7625** |

---

## 5. Final Validation Step
The 9-variable **Naive Bayes (RFE)** model was selected as the final champion due to its parsimony (requiring only half the predictors), fast computation, and superior AUC (0.7625) and Precision (41.76%). As the final step of the pipeline, the model was evaluated on the untouched **Validation Set (`valid.set`)** to verify out-of-sample generalization.

---

## 6. Repository Structure

```text
bank-marketing-classification/
├── data/
│   └── bank.csv                        # Raw dataset
├── src/
│   └── classification code github.R    # Complete R script (EDA, Preprocessing, RF, NB RFE, Validation)
└── README.md                           # Project documentation
```

---

## 7. How to Run the Project

### Prerequisites
Ensure you have **R** installed along with the required packages. Run the following command in your R console to install them:

```r
install.packages(c("e1071", "randomForest", "corrplot", "caret", "ROCR", "pdp"))
```

### Execution
1. Clone this repository to your local machine:
   ```bash
   git clone https://github.com/lambrossklivas/bank-marketing-classification.git
   ```
2. Open `src/classification code github.R` in RStudio.
3. Run the script. When `bank <- read.csv(file.choose())` executes, a file-picker dialog will appear—select `data/bank.csv` from the project folder and the entire pipeline will execute end-to-end.
