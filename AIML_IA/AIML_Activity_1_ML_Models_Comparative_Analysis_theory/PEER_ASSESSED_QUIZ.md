# MIT ACADEMY OF ENGINEERING, ALANDI (D), PUNE
### DEPARTMENT OF COMPUTER ENGINEERING | ACADEMIC YEAR: 2026–2027
### THIRD YEAR B.TECH (SEMESTER - V) | COURSE CODE: 2304322T
### COURSE: ARTIFICIAL INTELLIGENCE AND MACHINE LEARNING
## ACTIVITY NO. 2: PEER-ASSESSED QUIZ (5 MARKS)

---
- **Activity Title:** Implementation and Performance Evaluation with Comparative Analysis of Machine Learning Models for Selected Real-Time Applications
- **Student Name:** Tejas / Group Candidate
- **Seat Number:** TY-BTECH-V-01
- **PRN Number:** 202401040128
- **Total Marks:** 5 Marks (1 Mark Per Question)
- **Time Allocated:** 10 Minutes
---

### Instructions to Peer Reviewers:
1. Each question carries **1 Mark**.
2. Select the single best answer for each question.
3. Compare candidate responses against the verified Answer Key and Rationales at the end of the document.

---

### Question 1: Clinical Metric Prioritization [CO2, Level 4]
In deploying a Machine Learning model for real-time Cardiovascular Disease (Coronary Artery Disease) screening in an emergency department triage, which evaluation metric must be maximized as the primary clinical priority, and why?

- **[A]** **Accuracy**, because it measures overall correct predictions across all healthy and diseased patients.
- **[B]** **Precision**, because preventing false alarms avoids hospital financial expense for unneeded tests.
- **[C]** **Recall (Sensitivity)**, because failing to diagnose a cardiac patient (False Negative) can result in fatal myocardial infarction before intervention.
- **[D]** **Specificity**, because healthy individuals should never undergo stress electrocardiography testing.

---

### Question 2: Data Leakage Prevention during Preprocessing [CO4, Level 4]
In the machine learning development workflow, feature scaling (`StandardScaler`) was applied to continuous clinical predictors (`age`, `trestbps`, `chol`, `thalach`, `oldpeak`). Which procedure correctly avoids data leakage between partitions?

- **[A]** Applying `StandardScaler.fit_transform()` on the entire dataset before calling `train_test_split()`.
- **[B]** Fitting the `StandardScaler` strictly on the training set (`X_train`), followed by `.transform()` on both `X_train` and the test set (`X_test`).
- **[C]** Fitting one `StandardScaler` on `X_train` and fitting a separate independent `StandardScaler` on `X_test`.
- **[D]** Performing Winsorization and z-score standardization on testing folds during each split of cross-validation.

---

### Question 3: Ensemble Regularization & Variance Reduction [CO2, Level 4]
During hyperparameter optimization of the Random Forest Classifier via `GridSearchCV`, restricting tree depth (`max_depth = 3`) and specifying `min_samples_leaf = 2` primarily accomplishes which objective?

- **[A]** Minimizes model bias by allowing individual decision trees to memorize complex non-linear clinical noise.
- **[B]** Mitigates model variance and curbs overfitting by preventing leaf nodes from isolating single anomalous patients.
- **[C]** Forces the Gini impurity calculation to execute in $O(1)$ constant time complexity.
- **[D]** Converts the bagging ensemble directly into a sequential boosting ensemble.

---

### Question 4: Multi-Model Comparative Behavior [CO4, Level 4]
When comparing the performance of **K-Nearest Neighbors (KNN)**, **Support Vector Machines (SVM RBF)**, and a **Single Decision Tree (CART)** on the standardized heart disease dataset, why did the Single Decision Tree achieve significantly lower Recall ($67.86\%$) compared to KNN ($96.43\%$) and SVM ($96.43\%$)?

- **[A]** Decision trees cannot process continuous physiological variables like cholesterol and resting blood pressure.
- **[B]** Single unpruned decision trees suffer from high variance, axis-aligned orthogonal split rigidity, and greedy split decisions that overfit training noise.
- **[C]** Decision trees require min-max normalization rather than standard z-score normalization.
- **[D]** K-Nearest Neighbors completely eliminates the curse of dimensionality on tabular datasets.

---

### Question 5: Algorithmic Fairness & Healthcare Ethics (PO6) [CO4, Level 4]
Cardiovascular disease in women often presents with atypical symptoms (e.g., shortness of breath, fatigue, back pain) rather than classical crushing substernal chest pain. When deploying an AI Clinical Decision Support System (CDSS), what ethical governance measure is mandatory to prevent demographic bias?

- **[A]** Remove the biological `sex` attribute from the training data so the model becomes demographic-blind.
- **[B]** Audit the model's false-negative rates and sensitivity separately across male and female subgroups to guarantee equalized diagnostic opportunity.
- **[C]** Increase the classification decision threshold to $0.80$ for female patients to minimize false positives.
- **[D]** Replace the machine learning model entirely with an unweighted rule-based expert lookup table.

---

## 📋 VERIFIED ANSWER KEY & CLINICAL RATIONALES

| Question No. | Correct Option | Marks | Concept Tested |
|:---|:---:|:---:|:---|
| **Q1** | **[C]** | 1 M | Clinical Cost Asymmetry & Sensitivity (Recall) |
| **Q2** | **[B]** | 1 M | Data Leakage Prevention via Strict Train-Fitted Scalers |
| **Q3** | **[B]** | 1 M | Bias-Variance Tradeoff & Ensemble Tree Regularization |
| **Q4** | **[B]** | 1 M | Single-Tree High Variance vs. Non-linear Kernel / Distance Boundaries |
| **Q5** | **[B]** | 1 M | Subgroup Demographic Fairness & Ethical AI Deployment (PO6) |

### Detailed Pedagogical Rationales:

1. **Question 1 (Option C):** In medical diagnostic systems, the loss function is heavily asymmetric. A **False Negative** (a patient with critical coronary obstruction diagnosed as healthy) delays catheterization/revascularization, potentially resulting in patient mortality. A **False Positive** merely leads to a secondary non-invasive confirmatory diagnostic test (e.g., cardiac echo or CT angiography). Hence, **Recall (Sensitivity)** is paramount.
2. **Question 2 (Option B):** Calculating mean ($\mu$) and standard deviation ($\sigma$) across the entire cohort leaks statistical parameters of the unseen test distribution into the training pipeline. The transformer must strictly compute $\mu_{\text{train}}$ and $\sigma_{\text{train}}$ on `X_train`, using those exact frozen parameters to transform `X_test`.
3. **Question 3 (Option B):** Unconstrained decision trees grow until leaf nodes achieve 100% Gini purity, capturing idiosyncratic sensor noise and clinical outliers. Constraining `max_depth` and enforcing minimum leaf occupancy regularizes base learners and reduces ensemble generalization variance.
4. **Question 4 (Option B):** Single CART trees partition feature space via greedy axis-orthogonal hyperplanes without lookahead, making them highly vulnerable to sample variance. SVM with an RBF kernel and KNN map smooth non-linear decision manifolds, dramatically reducing missed boundary cases.
5. **Question 5 (Option B):** Simply dropping sensitive attributes ("fairness through unawareness") fails because proxy attributes (e.g., lower peak heart rate, differing cholesterol distributions) still encode gender differences. Subgroup stratification and equalized odds auditing ensure reliable performance across all demographics.

---
### Peer Assessment Score Sheet
- **Peer Evaluator Name:** ___________________________________
- **Peer Evaluator PRN:** ___________________________________
- **Total Score Awarded:** _____ / 5 Marks
- **Signature of Evaluator:** ____________________ | **Date:** _______________
