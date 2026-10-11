# Smartcart Customer Clustering

## Overview
This project performs customer segmentation using machine learning clustering techniques. By analyzing customer demographics and purchasing behaviors from the `smartcart.csv` dataset, the notebook identifies distinct customer profiles to help drive targeted marketing strategies. 

## Dataset
The project uses the `smartcart.csv` dataset, which includes customer attributes such as:
* **Demographics:** Year of birth, education level, marital status, income, and number of children at home.
* **Purchasing Behavior:** Spending amounts across various categories (Wines, Fruits, Meat, Fish, Sweets, Gold Products).
* **Engagement Metrics:** Number of purchases made via deals, web, catalog, and in-store, as well as website visits.

## Methodology

### 1. Data Preprocessing & Feature Engineering
* **Imputation:** Handled missing `Income` values by filling them with the median.
* **Feature Creation:** Derived new variables for deeper analysis:
  * `Age` (calculated based on the year 2026).
  * `Customer_Tenure_Days` (calculated from `Dt_Customer`).
  * `Total_Spending` (sum of all product category expenditures).
  * `Total_Children` (combined `Kidhome` and `Teenhome`).
* **Categorical Grouping:** Simplified `Education` (Undergraduate, Graduate, Postgraduate) and grouped marital status into `Living_With` (Alone, Partner).
* **Outlier Removal:** Filtered out extreme values (`Age` > 90 and `Income` > 600,000).

### 2. Encoding and Scaling
* Applied **One-Hot Encoding** to categorical features (`Education`, `Living_With`).
* Standardized the numerical features using **StandardScaler** to ensure uniform contribution across all dimensions.

### 3. Dimensionality Reduction
* Applied **Principal Component Analysis (PCA)** to reduce the feature space down to 3 principal components, retaining the most significant variance while enabling 3D visualization.

### 4. Clustering Models
Evaluated cluster tendencies to determine the optimal number of segments (K=4) using the **Elbow Method** (via the `kneed` library) and **Silhouette Scores**.
* **K-Means Clustering:** Applied with 4 clusters to partition the customer base.
* **Agglomerative Hierarchical Clustering:** Applied using "ward" linkage to validate the segmentation structure.

### 5. Visualization & Insights
* Generated correlation heatmaps to understand feature relationships.
* Created 3D scatter plots of the PCA components to visualize cluster separability.
* Plotted total spending against income, color-coded by cluster, to profile customer purchasing power and behavior.

## Technologies Used
* **Data Manipulation:** `pandas`
* **Visualization:** `matplotlib`, `seaborn`
* **Machine Learning & Preprocessing:** `scikit-learn` (KMeans, AgglomerativeClustering, PCA, StandardScaler, OneHotEncoder)
* **Optimization Tools:** `kneed` (KneeLocator)

## How to Run
1. Clone the repository.
2. Ensure you have the required libraries installed: `pip install pandas matplotlib seaborn scikit-learn kneed`
3. Open `smartcart.ipynb` in Jupyter Notebook or your preferred IDE.
4. Ensure `smartcart.csv` is located in the same directory.
5. Run all cells to reproduce the data cleaning, clustering models, and visualizations.
