# 🧠 Titanic ML Project

A complete end-to-end machine learning pipeline built on the Titanic dataset — from data preparation to model inference and scoring.  
This project demonstrates best practices in structuring, documenting, and versioning ML experiments.

---

## 📁 Folder Structure

```
titanic_ml_project/
│
├── data/
│   ├── train.csv
│   └── test.csv
|   └──clean/
|      ├── schema_*.json # JSON files defining the data schema (columns, types, constraints)
|      └── test_clean_*.csv # Cleaned and preprocessed test datasets ready for model inference
│
├── notebooks/
│   ├── 01_data_preparation.ipynb
│   ├── 02_feature_engineering.ipynb
│   ├── 03_feature_selection_and_model_building.ipynb
│   ├── 04_model_evaluation.ipynb
│   ├── 05_inference_and_scoring.ipynb
│   └── src/
│       ├── engineering.py
│       └── __pycache__/
│
├── outputs/
│   ├── *.csv      # Selected features and results
|   ├── *.joblib   # Serialized trained model pipelines
|   ├── *.txt      # Metrics summary
|   ├── *.png      # Feature importance plot
│
└── README.md
```

---

## 🧩 Project Overview

The Titanic dataset is one of the most popular datasets in machine learning.  
The goal is to predict whether a passenger survived the Titanic shipwreck using features like class, age, sex, and fare.
The model selection is automated via GridSearchCV, comparing Logistic Regression and RandomForestClassifier with hyperparameter tuning.

This project aims to illustrate:
- A **modular, reproducible ML workflow**
- The **transition from Jupyter notebooks to production code**
- Best practices for **feature engineering, model selection, and evaluation**

---

## ⚙️ Workflow

| Step | Notebook | Description |
|------|-----------|-------------|
| 1️⃣ | `01_data_preparation.ipynb` | Load and clean raw Titanic data. Handle missing values and outliers. |
| 2️⃣ | `02_feature_engineering.ipynb` | Create new derived feature `FamilySize`. Encode categorical variables. |
| 3️⃣ | `03_feature_selection_and_model_building.ipynb` | Compara Logistic Regression e RandomForest com GridSearchCV e escolhe o melhor modelo automaticamente. |
| 4️⃣ | `04_model_evaluation.ipynb` | Avalia o modelo final (agora salvo como `.joblib`) com múltiplas métricas. |
| 5️⃣ | `05_inference_and_scoring.ipynb` | Carrega o modelo `.joblib` e gera predições para novos dados. |

---

## 🧠 Model Details

- **Model Selection**: Automated via GridSearchCV comparing Logistic Regression and RandomForestClassifier.
- **Final Model**: Best pipeline selected by ROC-AUC (may vary depending on data or random state).
- **Frameworks**: `scikit-learn`, `pandas`, `numpy`, `matplotlib`
- **Feature Selection**: Automática via `SelectFromModel` (para LR) e importância de features (para RF)
- **Cross-validation**: Stratified K-Fold (k=5)
- **Scoring Metrics**: Accuracy, ROC-AUC, F1-Score

---

## 📦 Outputs

| File | Description |
|------|--------------|
| `*_selected_features.csv` | List of selected features used in the final model |
| `*_reduced_pipeline.joblib` | Serialized scikit-learn pipeline for inference |
| `*_metrics.txt` | Summary of evaluation metrics |
| `*_importance.png` | Feature importance plot |

---

## 🧰 Code Organization

The reusable Python code for feature transformations and preprocessing is modularized under:
```
notebooks/src/engineering.py
```
This allows notebooks to import standardized transformations using:
```python
from src.engineering import build_features, preprocess_data
```

---

## 🚀 How to Run

1. **Clone the repository:**
   ```bash
   git clone https://github.com/doglasp/titanic_ml_project.git
   cd titanic_ml_project
   ```

2. **Create and activate a virtual environment:**
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # (or .venv\Scripts\activate on Windows)
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the notebooks:**
   Open JupyterLab or VSCode and execute the notebooks in order.

---

## 🧪 Example Usage (CLI or Script)

To use the trained pipeline for prediction:

```python
import joblib
import pandas as pd

# Load test data and model
model = joblib.load("outputs/<nome_do_modelo>.joblib")
test_df = pd.read_csv("data/test.csv")

# Predict
preds = model.predict(test_df)
print(preds[:10])
```

---

## 🧾 Requirements

| Package | Version |
|----------|----------|
| Python | 3.11+ |
| pandas | ≥ 2.0 |
| scikit-learn | ≥ 1.4 |
| numpy | ≥ 1.24 |
| matplotlib | ≥ 3.8 |
| joblib | ≥ 1.3 |

---

## 📈 Example Outputs

Feature importance visualization:  
*(example)*  
![Feature Importance](outputs/titanic_feature_sel_20251015-143210_importance.png)

---

## 🧭 Next Steps

- Automate pipeline using **Prefect** or **Airflow**
- Containerize the environment with **Docker**
- Deploy model as an API endpoint using **FastAPI** or **Flask**
- Add unit tests and CI/CD workflows

---

## 👤 Author

**Doglas Parise, PhD**    
🔗 [LinkedIn](https://www.linkedin.com/in/doglas-parise)  
💻 [GitHub](https://github.com/doglasp)

---

## 📝 License

This project is released under the **MIT License**.
