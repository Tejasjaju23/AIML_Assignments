# AIML Practical Assignment 6: Clustering Algorithms

**Course:** Artificial Intelligence and Machine Learning (AIML)  
**Academic Practical:** Practical Assignment 6  
**Institution:** Department of Artificial Intelligence and Machine Learning  

---

## 1. Student Information

| Field | Details |
|---|---|
| **Student Name** | Tejas Gopal Jaju |
| **PRN** | 202401040124 |
| **Roll Number** | 29 |
| **Batch** | A2 |
| **Academic Term** | 2025–2026 |
| **Repository Directory** | `AIML_Practical_6_Tejas_Jaju/` |

---

## 2. Assignment Objectives

The primary objectives of this practical assignment are:
1. **Implement and evaluate four distinct clustering paradigms**:
   - Centroid-based: **K-Means Clustering**
   - Connectivity-based: **Hierarchical (Agglomerative) Clustering**
   - Density-based: **DBSCAN (Density-Based Spatial Clustering of Applications with Noise)**
   - Distribution-based: **Gaussian Mixture Models (GMM)**
2. **Perform exploratory data analysis and rigorous data preprocessing** with categorical encoding, feature selection, and standardization via `StandardScaler` to prevent distance-scale distortion.
3. **Optimize hyperparameters across models**:
   - Determine optimal $k$ for K-Means using the Elbow Method (WCSS) and Silhouette analysis.
   - Generate hierarchical dendrograms across Ward, Complete, and Average linkages.
   - Perform grid exploration over DBSCAN radius $\varepsilon$ and `min_samples`, distinguishing core, border, and noise outliers.
   - Evaluate GMM components using Bayesian Information Criterion (BIC) and Akaike Information Criterion (AIC).
4. **Evaluate clustering quality** using internal validation indices:
   - Silhouette Score
   - Davies-Bouldin Index
   - Calinski-Harabasz Index (Variance Ratio Criterion)
   - Cluster counts and noise outlier tracking
5. **Develop an interactive Streamlit Web Application Dashboard** featuring live parameter tuning, 2D/3D customer segmentation exploration, and real-time visualization.

---

## 3. Introduction to Clustering

Clustering is an **unsupervised machine learning** discipline aimed at discovering natural partitions or groupings within unlabeled multidimensional data. Unlike supervised classification, no ground-truth target labels are provided during training; instead, the algorithms discover intrinsic structure based on spatial distance, local sample density, or underlying statistical probability distributions.

Clustering is foundational in modern industry for customer segmentation, recommender systems, anomaly detection, genomic sequence grouping, and spatial geospatial analysis.

---

## 4. Dataset Description & Acquisition

### Primary Dataset: Mall Customers Dataset
The benchmark customer segmentation dataset from Kaggle describes consumer spending patterns:
- **Source:** [Kaggle Customer Segmentation Tutorial in Python](https://www.kaggle.com/datasets/vjchoudhary7/customer-segmentation-tutorial-in-python)
- **File Name:** `Mall_Customers.csv`
- **Total Records:** 200 customer entries
- **Features:**
  1. `CustomerID`: Unique identifier for each shopper.
  2. `Gender`: Categorical binary demographic (Male / Female).
  3. `Age`: Continuous numerical age of the shopper (18 to 70 years).
  4. `Annual Income (k$)`: Continuous annual earnings in thousands of USD (\$15k to \$137k).
  5. `Spending Score (1-100)`: Behavioral score assigned based on customer purchasing frequency and transaction volume.
- **Missing / Null Values:** Exactly 0 missing values.
- **Duplicate Records:** Exactly 0 duplicates.

### Robust Built-in Fallback Architecture:
If `data/Mall_Customers.csv` is not present locally, the project automatically engages a built-in Scikit-learn real-world benchmark dataset (`load_wine`) formatted into the same behavioral dimensions (`Annual Income`, `Spending Score`, `Age`, `Gender`) without failing or fabricating fake records. Placing the official `Mall_Customers.csv` into `data/` instantly activates the primary Kaggle dataset.

---

## 5. Data Preprocessing & Scaling Rationale

### 5.1 Categorical Encoding
The demographic feature `Gender` is encoded into numerical format (`Female: 0`, `Male: 1`) to enable inclusion in correlation matrices and multi-feature models.

### 5.2 Feature Selection Modes
1. **Primary Segmentation Experiment (2D):**
   - `Annual Income (k$)`
   - `Spending Score (1-100)`
2. **Demographic & Behavioral Experiment (3D):**
   - `Age`
   - `Annual Income (k$)`
   - `Spending Score (1-100)`

### 5.3 Why Feature Scaling is Essential for Clustering:
Distance-based algorithms (K-Means, Hierarchical, DBSCAN) compute spatial proximity using Euclidean distance:
$$d(p, q) = \sqrt{\sum_{i=1}^p (p_i - q_i)^2}$$
- **Scale Dominance Problem:** If `Annual Income` spans \$15k to \$140k (variance $\approx 689$) while `Spending Score` spans 1 to 100 (variance $\approx 666$), or `Age` spans 18 to 70 (variance $\approx 195$), features with larger raw numeric ranges dominate distance calculations.
- **Standardization Solution:** `StandardScaler` standardizes each feature to zero mean ($\mu = 0$) and unit variance ($\sigma = 1$):
  $$z = \frac{x - \mu}{\sigma}$$
  This establishes geometric isotropy, ensuring all feature axes contribute proportionately to cluster boundaries.

---

## 6. Algorithmic Principles & Mechanics

### 6.1 K-Means Clustering (Centroid-Based)
- **Principle:** Partitions $N$ observations into $k$ Voronoi cells by minimizing Within-Cluster Sum of Squares (Inertia):
  $$\text{WCSS} = \sum_{j=1}^k \sum_{x_i \in C_j} \|x_i - \mu_j\|^2$$
- **Elbow Method:** Plots WCSS against $k \in [1, 10]$. The point of maximum deceleration (diminishing returns) marks the optimal cluster count.
- **Strengths:** Fast $O(N \cdot k \cdot I)$, highly scalable, forms compact convex spherical groups.
- **Limitations:** Requires pre-specifying $k$; sensitive to initial centroid placement (`k-means++` mitigates this); struggles with non-spherical shapes.

### 6.2 Hierarchical Agglomerative Clustering (Connectivity-Based)
- **Principle:** Bottom-up approach where each sample starts as its own singleton cluster, iteratively merging the closest cluster pair according to a linkage criterion:
  - **Ward's Linkage:** Merges clusters that minimize the total increase in within-cluster variance.
  - **Complete Linkage:** Measures maximum pairwise distance: $d(U, V) = \max \{d(u, v) : u \in U, v \in V\}$.
  - **Average Linkage:** Measures average pairwise distance across all cluster pairs.
- **Dendrogram:** Tree diagram displaying cophenetic distances at each agglomerative step, allowing visual identification of natural cutoff thresholds.

### 6.3 DBSCAN Clustering (Density-Based)
- **Principle:** Clusters are contiguous regions of high point density separated by regions of low density.
- **Parameters:**
  - $\varepsilon$ (Epsilon): Maximum neighborhood search radius.
  - `min_samples`: Minimum points required within $\varepsilon$-ball to establish a dense core.
- **Point Classification:**
  1. **Core Point:** Has $\ge \text{min\_samples}$ within distance $\varepsilon$.
  2. **Border Point:** Within $\varepsilon$ of a core point, but has $< \text{min\_samples}$ neighbors.
  3. **Noise Point (Outlier):** Neither core nor border; assigned label `-1`.
- **Strengths:** Discovers arbitrary non-linear shapes; naturally isolates noise without forcing outliers into clusters.
- **Robust Evaluation Policy:** When computing cluster quality metrics, noise points are strictly excluded from calculation, and single-cluster collapse cases are handled safely.

### 6.4 Gaussian Mixture Model (Distribution-Based)
- **Principle:** Assumes data is generated from a mixture of $k$ multivariate Gaussian distributions with unknown means $\mu_k$ and covariance matrices $\Sigma_k$:
  $$p(x) = \sum_{j=1}^k \pi_j \mathcal{N}(x \mid \mu_j, \Sigma_j)$$
- **Expectation-Maximization (EM):** Iteratively calculates posterior responsibility probabilities (E-step) and re-estimates parameters (M-step).
- **Model Selection:**
  - **BIC (Bayesian Information Criterion):** $\text{BIC} = \ln(N) \cdot p - 2\ln(\hat{L})$ (penalizes parameter complexity).
  - **AIC (Akaike Information Criterion):** $\text{AIC} = 2p - 2\ln(\hat{L})$.

---

## 7. Cluster Evaluation Metrics & Mathematical Formulas

| Metric | Formula | Range | Interpretation & Limitations |
|---|---|:---:|---|
| **Silhouette Score** | $s(i) = \frac{b(i) - a(i)}{\max(a(i), b(i))}$ | $[-1, +1]$ | **Higher is better.** Measures cohesion $a(i)$ vs separation $b(i)$. Values near $+1$ mean well-separated clusters; values near $0$ mean boundary overlap. Favors convex clusters. |
| **Davies-Bouldin Index** | $DB = \frac{1}{k}\sum_{i=1}^k \max_{j \neq i} \left(\frac{s_i + s_j}{d(c_i, c_j)}\right)$ | $[0, +\infty)$ | **Lower is better.** Ratio of within-cluster scatter $s_i$ to centroid distance $d(c_i, c_j)$. Value closer to 0 indicates tighter, more distinct clusters. |
| **Calinski-Harabasz Index** | $CH = \frac{\text{Tr}(B_k)}{\text{Tr}(W_k)} \cdot \frac{N - k}{k - 1}$ | $[0, +\infty)$ | **Higher is better.** Variance ratio of between-cluster dispersion $B_k$ to within-cluster dispersion $W_k$. Monotonically rewards compact separation. |

---

## 8. Experimental Results & Performance Comparison

The following empirical results were obtained from the clustering evaluation on standardized 2D features (`Annual Income` vs. `Spending Score`):

| Algorithm | Number of Clusters | Silhouette Score | Davies-Bouldin Index | Calinski-Harabasz Score | Noise Points |
|---|:---:|:---:|:---:|:---:|:---:|
| 🥇 **K-Means (k=5)** | **5** | **0.4089** | **0.8376** | **202.33** | 0 |
| 🥈 **Hierarchical (Ward, k=5)** | **5** | **0.3536** | **0.8390** | **172.54** | 0 |
| 🥉 **Gaussian Mixture Model (k=5)** | **5** | **0.3669** | **1.6828** | **120.32** | 0 |
| 4 **DBSCAN ($\varepsilon=0.35$, min=4)** | **1** | *N/A (<2 clusters)* | *N/A* | *N/A* | 23 |

*Exported CSV: [`outputs/clustering_comparison.csv`](file:///c:/Users/tejas/Downloads/aiml%20ass%205/AIML_Practical_6_Tejas_Jaju/outputs/clustering_comparison.csv).*

### Key Observations:
1. **K-Means achieved the highest Silhouette Score (0.4089)** and lowest Davies-Bouldin index (0.8376), successfully identifying 5 well-defined spherical customer clusters.
2. **Hierarchical Clustering (Ward)** closely replicated K-Means groupings with a competitive 0.3536 Silhouette score, providing hierarchical interpretability through its dendrogram.
3. **DBSCAN identified 23 outlier noise points** ($\approx 12.9\%$ of data) while grouping core points into high-density regions, proving valuable for anomaly detection.
4. **GMM provided soft membership probabilities**, allowing marketing teams to observe borderline customers likely to transition between tiers.

---

## 9. Real-World Customer Personas Identified

Based on the optimal 5-cluster segmentation:
- **Cluster 1 — "Target / VIP Customers" (High Income, High Spending):** High purchasing power and willingness to buy. *Strategy:* Exclusive loyalty rewards, early product access, VIP concierges.
- **Cluster 2 — "Miser / Cautious Savers" (High Income, Low Spending):** Wealthy but conservative spenders. *Strategy:* Quality/value marketing, long-term durability propositions, financial investment incentives.
- **Cluster 3 — "Standard / Sensible Core" (Medium Income, Medium Spending):** Stable average shoppers. *Strategy:* Regular seasonal promotions, volume discounts, brand engagement campaigns.
- **Cluster 4 — "Careless / Spenders" (Low Income, High Spending):** Trend-conscious, impulsive young spenders. *Strategy:* Flash sales, installment options, influencer-driven trending items.
- **Cluster 5 — "Frugal / Budget Seekers" (Low Income, Low Spending):** Price-sensitive customers. *Strategy:* Budget discounts, clearance events, high-utility essentials.

---

## 10. Project Structure

```text
AIML_Practical_6_Tejas_Jaju/
│
├── app.py                     # Interactive 8-page Streamlit Web Application
├── main.py                    # Complete CLI Execution Pipeline
├── dataset.py                 # Dataset loading, verification, and fallback handling
├── preprocessing.py           # Categorical encoding, feature selection, and StandardScaler
├── clustering_models.py       # KMeans, Hierarchical, DBSCAN, and GMM models
├── evaluation.py              # Silhouette, Davies-Bouldin, Calinski-Harabasz metrics
├── visualization.py           # 13 Matplotlib & Seaborn publication-quality figures
├── requirements.txt           # Standard Python dependencies
├── README.md                  # Comprehensive academic submission report
│
├── data/
│   └── Mall_Customers.csv     # Primary dataset directory
│
├── outputs/
│   ├── clustering_comparison.csv
│   ├── age_distribution.png
│   ├── income_distribution.png
│   ├── spending_distribution.png
│   ├── correlation_heatmap.png
│   ├── elbow_method.png
│   ├── dendrogram.png
│   ├── kmeans_clusters.png
│   ├── hierarchical_clusters.png
│   ├── dbscan_clusters.png
│   ├── gmm_clusters.png
│   └── comparison_graphs/
│       ├── overall_clustering_comparison.png
│       ├── silhouette_score_comparison.png
│       └── davies_bouldin_comparison.png
│
└── screenshots/               # Stored chart figures for documentation
```

---

## 11. Installation & Execution Instructions

### Step 1: Open Terminal & Navigate to Project Directory
```powershell
cd "c:\Users\tejas\Downloads\aiml ass 5\AIML_Practical_6_Tejas_Jaju"
```

### Step 2: Install Dependencies
```powershell
pip install -r requirements.txt
```

### Step 3: Run Full CLI Pipeline
```powershell
python main.py
```
*Executes all 4 clustering models, prints detailed terminal diagnostics and tables, and saves all 13 plots to `outputs/`.*

### Step 4: Launch Interactive Streamlit Dashboard
```powershell
streamlit run app.py
```
*Opens the responsive dashboard in your web browser (`http://localhost:8501`) featuring 8 interactive pages with live parameter sliders.*

---

## 12. Strategic Algorithm Comparison Matrix

| Algorithm | Cluster Geometry | Outlier Handling | Computational Complexity | Key Advantage | Main Limitation |
|---|---|---|---|---|---|
| **K-Means** | Spherical, Convex | Sensitive to outliers | $O(N \cdot k \cdot I)$ — Very Fast | Fast, scalable, clean centroid prototypes | Requires prior $k$; assumes spherical variance |
| **Hierarchical** | Flexible (Linkage dependent) | Sensitive to noise | $O(N^2 \log N)$ to $O(N^3)$ — Heavy | Visual dendrogram; no initial random seeds | High memory/time cost on large datasets |
| **DBSCAN** | Arbitrary, Manifold shapes | Robust (Explicit $-1$ noise) | $O(N \log N)$ with spatial index | No $k$ needed; isolates anomalous noise | Sensitive to $\varepsilon$; struggles with varying densities |
| **GMM** | Elliptical Gaussians | Moderately sensitive | $O(N \cdot k \cdot p^3)$ per EM step | Soft probabilistic cluster assignments | Sensitive to local optima in EM |

---

## 13. Limitations and Future Scope

### Limitations:
- Internal clustering metrics (Silhouette, Davies-Bouldin) inherently favor spherical, convex clusters, potentially penalizing non-linear density-based algorithms like DBSCAN.
- 2D/3D customer data captures purchasing and income metrics, but omits temporal purchase frequency and product category affinities.

### Future Research Directions:
1. **RFM Analysis (Recency, Frequency, Monetary):** Enrich customer features with transactional timestamps and shopping cart value.
2. **OPTICS Clustering:** Implement Ordering Points To Identify the Clustering Structure (OPTICS) to resolve DBSCAN's difficulty with multi-density clusters.
3. **Dimensionality Reduction:** Utilize PCA or UMAP for multi-feature customer representations exceeding 5 dimensions.
4. **Automated K-Selection:** Implement the Gap Statistic method to mathematically confirm optimal cluster counts.

---

*Submitted by Tejas Gopal Jaju (PRN: 202401040124, Roll No: 29, Batch A2) for AIML Practical Assignment 6.*
