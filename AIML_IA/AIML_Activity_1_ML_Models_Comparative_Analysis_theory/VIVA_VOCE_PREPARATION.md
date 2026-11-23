# MIT ACADEMY OF ENGINEERING, ALANDI (D), PUNE
### DEPARTMENT OF COMPUTER ENGINEERING | ACADEMIC YEAR 2026–2027
### COURSE: ARTIFICIAL INTELLIGENCE AND MACHINE LEARNING (COURSE CODE: 2304322T)
## ACTIVITY NO. 2: VIVA-VOCE & ORAL EXAMINATION MASTER GUIDE

---
- **Activity Title:** Implementation and Performance Evaluation with Comparative Analysis of Machine Learning Models for Selected Real-Time Applications
- **Candidate Name:** Tejas / Candidate
- **Seat Number:** TY-BTECH-V-01 | **PRN:** 202401040128
- **Course Outcomes Tested:** CO2 (Analyze algorithms, apply theoretical concepts) & CO4 (Evaluate modern tools, design ML workflows, consider ethics)
---

### SECTION 1: PROBLEM STATEMENT & DOMAIN UNDERSTANDING

#### Q1: What is the clinical problem your project addresses, and why is machine learning appropriate here?
**Model Answer:**
Cardiovascular diseases (CVDs), primarily Coronary Artery Disease (CAD), are responsible for ~17.9 million deaths annually. Gold-standard diagnosis relies on invasive coronary angiography, which carries procedural risks, high cost, and delays. Machine learning provides non-invasive, objective risk stratification by synthesizing routine clinical indicators (demographics, blood pressure, cholesterol, resting ECG, stress-test heart rate, and ST-depression). It serves as a rapid emergency triage tool to determine whether immediate cardiac catheterization is required.

#### Q2: In clinical diagnostics, why is Recall (Sensitivity) prioritized over overall Accuracy?
**Model Answer:**
Medical diagnostics suffer from extreme **loss function asymmetry**. A **False Negative (Type II error)** occurs when an ischemic cardiac patient is erroneously diagnosed as healthy and discharged without care, potentially resulting in fatal acute myocardial infarction. In contrast, a **False Positive (Type I error)** leads to confirmatory non-invasive testing (e.g., stress echocardiogram or CT angiography), which causes minor inconvenience and monetary cost, but zero mortality risk. Therefore, maximizing Recall ($\frac{TP}{TP + FN}$) is the primary clinical objective.

---

### SECTION 2: DATA ENGINEERING & PREPROCESSING RIGOR

#### Q3: How did you handle missing values and outliers in the UCI Cleveland dataset?
**Model Answer:**
- **Missing Values:** 4 missing entries in `ca` (fluoroscopy vessels) were imputed using the training set median ($0.0$), and 2 missing entries in `thal` (scintigraphy) were imputed using the mode ($3.0$, Normal). We avoided mean imputation because both are discrete physiological categories.
- **Outliers:** Rather than dropping clinical records, we applied **99th-percentile Winsorization** to serum cholesterol (`chol` capped at $406.7$ mg/dL) and resting blood pressure (`trestbps` capped at $180.0$ mmHg). This preserved valid high-risk patients while preventing extreme leverage points from skewing distance- and gradient-based algorithms.

#### Q4: What is Data Leakage, and how did your pipeline strictly prevent it?
**Model Answer:**
Data leakage occurs when information from outside the training dataset (such as test set statistics) is inadvertently used to train the model, resulting in over-optimistic performance estimates that fail in production.
We prevented data leakage by:
1. Executing an 80:20 **Stratified Train-Test Split** prior to scaling.
2. Fitting the `StandardScaler` strictly on `X_train` (`scaler.fit_transform(X_train)`).
3. Applying the pre-computed $\mu$ and $\sigma$ parameters to the test partition via `scaler.transform(X_test)`.
4. Computing all imputation values (median, mode) and Winsorization percentiles solely from `X_train`.

#### Q5: Why did you use One-Hot Encoding with `drop_first=True` for nominal features?
**Model Answer:**
Features such as Chest Pain Type (`cp`) and Resting ECG (`restecg`) are nominal, meaning there is no inherent numerical ordering (e.g., asymptomatic chest pain is not mathematically "$4\times$" typical angina). Label encoding would impose a false linear distance. One-hot encoding creates binary indicator variables, and `drop_first=True` removes the $k$-th dummy to prevent multicollinearity (the "dummy variable trap") in linear models.

---

### SECTION 3: ALGORITHM MECHANISMS & HYPERPARAMETER OPTIMIZATION

#### Q6: Explain the difference between Bagging (Random Forest) and Boosting (Gradient Boosting).
**Model Answer:**
- **Bagging (Bootstrap Aggregating):** Builds independent base trees in parallel on bootstrap samples of the training data. Each tree split considers a random feature subset. The final prediction averages individual tree probabilities, primarily **reducing ensemble variance** without increasing bias.
- **Boosting:** Builds trees sequentially. Each subsequent tree is trained to correct the pseudo-residuals (gradients of the loss function) of the prior ensemble. Boosting primarily **reduces bias**, but can be more prone to overfitting noisy clinical labels if the learning rate or tree depth is unconstrained.

#### Q7: How does restricting `max_depth = 3` in Random Forest prevent overfitting?
**Model Answer:**
An unconstrained decision tree splits until all leaves are 100% pure ($I_G = 0$), memorizing idiosyncratic noise, measurement errors, and outlier patients. By capping `max_depth = 3` and enforcing `min_samples_leaf = 2` via `GridSearchCV`, we regularize the tree structures. Each tree is forced to learn generalizable, clinically plausible decision boundaries, stabilizing the ensemble's out-of-fold generalization error.

#### Q8: Why did Kernel SVM (RBF) perform significantly better than the Multi-Layer Perceptron (MLP)?
**Model Answer:**
Our dataset has $N=303$ instances. Deep learning models and multi-layer perceptrons have large parameter spaces (weights and biases across layers) that require thousands of samples to regularize effectively without memorizing noise. Support Vector Machines with a Radial Basis Function (RBF) kernel map input features into an infinite-dimensional Reproducing Kernel Hilbert Space (RKHS) and solve a convex quadratic optimization problem to find the global maximum-margin hyperplane, making SVM vastly superior for compact tabular datasets.

#### Q9: Why was K-Nearest Neighbors (KNN) competitive in accuracy, but problematic for production deployment?
**Model Answer:**
KNN achieved $93.44\%$ accuracy and $96.43\%$ recall because standardized clinical features formed tight local neighborhoods. However, KNN is an instance-based lazy learner: it does not learn a parameterized model during training. At test time, computing Euclidean distances against all training instances takes $O(N \cdot d)$ time. In our benchmark, KNN's inference latency was **$1.140$ ms per sample**, which is **$23\times$ slower** than Random Forest ($0.048$ ms) and **$35\times$ slower** than SVM ($0.033$ ms). For hospital systems processing hundreds of thousands of patient records, KNN is computationally inefficient.

---

### SECTION 4: PRODUCTION DEPLOYMENT & ETHICS (PO6, PO12)

#### Q10: How does your system address demographic bias between male and female patients?
**Model Answer:**
Clinical literature demonstrates that cardiovascular disease symptoms in women are frequently atypical (e.g., dyspnea, fatigue, back pain) rather than classical crushing chest pain. Standard models trained on male-heavy cohorts risk high false-negative rates for women. We audited false-negative rates separately across biological sex cohorts and implemented calibrated sensitivity thresholds to ensure equalized odds and equal opportunity across demographics.

#### Q11: How would you monitor this model for Data Drift in a hospital setting?
**Model Answer:**
Over time, changes in hospital patient demographics, clinical testing equipment (e.g., new ECG machines), or seasonal health trends cause input distributions to diverge from training data (covariate shift). In production, we deploy **Evidently AI** with automated **Population Stability Index (PSI)** and **Kolmogorov-Smirnov (KS)** tests to continuously compare incoming feature distributions against baseline training distributions. If PSI exceeds $0.2$, automated alerts trigger model retraining and physician notification.

#### Q12: Why must AI in healthcare always operate under a "Human-in-the-Loop" architecture?
**Model Answer:**
Machine learning models are probabilistic estimators, not infallible diagnostic authorities. Deployment as an autonomous prescribing or diagnostic entity introduces catastrophic clinical risk and legal liability. CardioGuard AI operates strictly as a **Clinical Decision Support System (CDSS)**: it provides probabilistic risk scores, calibrated confidence intervals, and explainable feature attributions (Gini importance, decision trees) to augment the physician's clinical judgment, keeping the licensed medical professional in full control of patient care.
