# Machine Learning References

![Langchains](https://img.shields.io/badge/XG%20Boost-XGBoost-blue)
![Python](https://img.shields.io/badge/Python-3.10+-green)
![Machine Learning](https://img.shields.io/badge/Machine%20Learning-Models-orange)

A structured collection of machine learning notebooks, datasets, and small scratch implementations for learning, practice, and interview revision.

## Why this repository exists

This repo is designed as a personal learning notebook bank where each file focuses on one ML concept, algorithm, or preprocessing technique. You can use it in two modes:

1. Education: learn topics end-to-end with examples.
2. Revision: quickly revisit a specific concept before interviews, projects, or exams.

## Repository snapshot

- 58 Jupyter notebooks (`.ipynb`)
- 25 datasets (`.csv`)
- 2 HTML exports
- 2 scratch Python files in `Kmeans Scratch/`
- 1 Streamlit visualization helper

## Tech stack

- Python 3.9+
- Jupyter Notebook / JupyterLab
- Core libraries used across notebooks:
  - `numpy`
  - `pandas`
  - `matplotlib`
  - `seaborn`
  - `scikit-learn`
  - `xgboost` (in boosting notebooks if used)
  - `optuna` (hyperparameter tuning)
  - `streamlit` (for `streamlit-viz-tool.py`)

## Quick start

```bash
# 1) Clone repository
git clone <your-repo-url>
cd Machine-Learning-References

# 2) Create virtual environment
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux/Mac
source .venv/bin/activate

# 3) Install common packages
pip install numpy pandas matplotlib seaborn scikit-learn jupyter optuna streamlit

# 4) Launch notebooks
jupyter notebook
```

## Project structure

```text
Machine-Learning-References/
|-- README.md
|-- *.ipynb                         # concept notebooks
|-- *.csv                           # datasets used by notebooks
|-- output.html
|-- output_backpack-prediction.html
|-- streamlit-viz-tool.py           # interactive logistic-regression boundary demo
`-- Kmeans Scratch/
    |-- kmeans.py                   # custom K-Means implementation
    `-- app.py                      # usage/demo script for custom K-Means
```

## Notebook guide (detailed)

### 1) EDA and data understanding

- `Univariate_EDA.ipynb`: Univariate analysis for single-column distributions.
- `Bivariate&Multivariate.ipynb`: Bivariate and multivariate exploratory analysis.
- `Pandas_Profilling.ipynb`: Automated profiling style workflow.
- `Handling Date and Time.ipynb`: Datetime feature parsing and extraction.
- `Fetch_from_api.ipynb`: Pulling/processing data from APIs.
- `Web_Scraping.ipynb`: Basic web scraping workflow for data collection.

### 2) Missing values and imputation

- `Complete Case Analysis.ipynb`: Dropping rows with missing values.
- `Simple Imputer.ipynb`: Mean/median/mode style simple imputation.
- `Simple Imputer in Categorical.ipynb`: Categorical imputation patterns.
- `KNN imputer.ipynb`: KNN-based missing-value estimation.
- `Random_sample_imputation.ipynb`: Random sample based imputation.
- `Missing Indicator.ipynb`: Missingness as a predictive feature.
- `automatic selector imputer.ipynb`: Selection-driven imputation strategy.

### 3) Feature engineering and preprocessing

- `One_hot_Encoding.ipynb`: One-hot encoding for nominal categories.
- `Ordinal_Encoding.ipynb`: Ordinal encoding for ordered categories.
- `Column Transformer.ipynb`: Mixed preprocessing with sklearn `ColumnTransformer`.
- `Function_Transformer.ipynb`: Custom transform functions in pipelines.
- `Handling_Mixed_column.ipynb`: Numeric/categorical mixed-column workflows.
- `Binning.ipynb`: Continuous-to-binned feature conversion.
- `Normalization.ipynb`: MinMax normalization.
- `Standarization.ipynb`: Standardization (z-score style scaling).
- `Power Tranformer.ipynb`: Power transforms for skewed data.

### 4) Outlier detection and treatment

- `Zscore Outliers.ipynb`: Z-score based outlier detection.
- `IQR_Outliers.ipynb`: IQR method for robust outlier handling.
- `Percentile_Outlier_detection.ipynb`: Percentile threshold approach.

### 5) Linear models and gradient descent

- `Simple_LR.ipynb`: Simple linear regression.
- `Simple_LR_Scratch.ipynb`: Linear regression from scratch.
- `Multiple_LR.ipynb`: Multiple linear regression.
- `Multiple_LR_Scrach.ipynb`: Multiple regression scratch attempt.
- `Polynomial_Regression.ipynb`: Polynomial feature regression.
- `Ridge_from_scratch.ipynb`: Ridge regression implementation intuition.
- `Lasso_Regression.ipynb`: L1 regularized regression.
- `Elastic_Net_Reg.ipynb`: Elastic Net regression.
- `Gradient_Descent.ipynb`: Gradient descent fundamentals.
- `gradient_descent_basic.ipynb`: Basic GD flow and updates.
- `Gradient_descent_Main.ipynb`: Main GD workflow.
- `stochastic_Gradient_descent.ipynb`: SGD optimization.
- `Mini_batch_GD.ipynb`: Mini-batch GD optimization.

### 6) Classification

- `Perceptron_Logistic.ipynb`: Perceptron and logistic foundations.
- `Gradien_Descent_Logistic.ipynb`: Logistic regression with GD perspective.
- `Softmax_Regression.ipynb`: Multiclass logistic regression.
- `Polynomial_Logistic_Regression.ipynb`: Nonlinear decision boundary via polynomial features.
- `Naive Bayes.ipynb`: Probabilistic classification baseline.
- `Kernel  Trick SVM.ipynb`: SVM kernel trick intuition and usage.
- `DTreeViz.ipynb`: Decision tree visualization and interpretation.
- `Random_Forest.ipynb`: Ensemble tree classification/regression concepts.

### 7) Clustering and unsupervised learning

- `kmeans_Scratch.ipynb`: K-Means concept from scratch.
- `kmeans_student.ipynb`: K-Means practice notebook.
- `Hierarchical_Cluster.ipynb`: Agglomerative clustering flow.
- `DBScanAlgo.ipynb`: Density-based clustering (DBSCAN).
- `MNIST_DATASET_PCA.ipynb`: PCA use on image data.
- `pca_step_by_step.ipynb`: PCA explained step by step.

### 8) Ensemble methods and boosting

- `GradientBoostingStepByStep.ipynb`: Gradient boosting explained in sequence.
- `Copy of gradient-boosting-classification.ipynb`: Gradient boosting classification practice.
- `Stacking.ipynb`: Stacking ensemble strategy.
- `Imbalanced_Data.ipynb`: Handling class imbalance with model/data strategies.

### 9) Pipelines, model selection, and metrics

- `SklearnPipeline_titanic_DS.ipynb`: End-to-end sklearn pipeline pattern.
- `Optuna.ipynb`: Hyperparameter optimization with Optuna.
- `RocAucCurves.ipynb`: ROC-AUC based model evaluation.

### 10) Project/practice notebooks

- `backpack-prediction.ipynb`: Applied prediction workflow on backpack data.
- `Practice.ipynb`: Mixed concept practice notebook.

## Python scripts (internal implementations)

### `Kmeans Scratch/kmeans.py`

Custom `KMeans` class implementation with:

- Random centroid initialization
- Point-to-centroid distance assignment (Euclidean)
- Centroid movement by cluster mean
- Early stop when centroids no longer change

Use this for conceptual understanding of the K-Means algorithm internals.

### `Kmeans Scratch/app.py`

Demo runner for custom K-Means:

- Generates synthetic blobs via `make_blobs`
- Fits custom `KMeans`
- Visualizes clustered points with Matplotlib

### `streamlit-viz-tool.py`

Interactive Streamlit app for logistic regression decision boundaries:

- Selects binary/multiclass synthetic datasets
- Lets you configure regularization and solver parameters
- Plots decision regions
- Reports test accuracy

## Dataset map

- `1.ushape.csv`: Synthetic/nonlinear clustering style data.
- `cars.csv`: Car features dataset.
- `concrete.csv`: Concrete strength style regression dataset.
- `covid_toy.csv`: Toy dataset for missing/imputation practice.
- `customer.csv`: Customer-level sample dataset.
- `data_science_job.csv`: Job-related dataset for analysis.
- `heart.csv`: Heart disease style classification dataset.
- `hierarchical-clustering-with-python-and-scikit-learn-shopping-data.csv`: Shopping data for hierarchical clustering.
- `messages.csv`: Text/message style classification dataset.
- `movies.csv`: Movie metadata-like dataset for EDA/features.
- `orders.csv`: Order/transaction style dataset.
- `placement.csv`: Student placement classification/regression dataset.
- `placement_LR.csv`: Placement dataset variant for linear regression.
- `play_tennis.csv`: Small classic classification dataset.
- `Social_Network_Ads.csv`: Ad conversion classification dataset.
- `student_clustering.csv`: Student feature data for clustering.
- `titanic.csv`: Titanic dataset for preprocessing/classification.
- `titanic_2.csv`: Titanic variant for practice.
- `titanic_toy.csv`: Small Titanic sample.
- `train._housecsv.csv`: Housing-style training dataset.
- `train_backpack.csv`: Backpack project training data.
- `weight-height.csv`: Weight-height regression/outlier dataset.
- `wine_data.csv`: Wine dataset variant.
- `WineQT.csv`: Wine quality dataset.

## HTML exports

- `output.html`: Exported analysis/report output.
- `output_backpack-prediction.html`: Export of backpack prediction notebook/report.

## How to use this repo for revision

### Fast revision path (high impact)

1. Preprocessing: Imputation + Encoding + Scaling + Outlier notebooks.
2. Core supervised ML: LR, Logistic, SVM, Tree, Random Forest.
3. Optimization: GD/SGD/Mini-batch notebooks.
4. Unsupervised: K-Means, Hierarchical, DBSCAN, PCA.
5. Evaluation: ROC-AUC and pipeline notebook.
6. Tuning: `Optuna.ipynb`.

### Weekly study plan suggestion

1. Week 1: EDA + preprocessing notebooks.
2. Week 2: Linear models + optimization notebooks.
3. Week 3: Classification + ensembles.
4. Week 4: Unsupervised learning + pipeline + tuning + recap.

## Notes and improvements you can add later

- Add a `requirements.txt` for exact reproducible environment.
- Add notebook-level headers/objectives where missing.
- Standardize file naming (`Standarization`/`Power Tranformer`/`Scrach` typos) for easier discovery.
- Add one capstone project folder with train/validate/test workflow.

## License

Add your preferred license (`MIT`, `Apache-2.0`, etc.) in a `LICENSE` file.
