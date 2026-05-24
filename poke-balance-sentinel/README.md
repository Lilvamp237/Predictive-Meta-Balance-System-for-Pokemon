# 🛡️ Pokémon Predictive Meta-Balance System

> **A Machine Learning Decision-Support Tool for Game Developers and Competitive Players.**

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3+-orange.svg)
![Pandas](https://img.shields.io/badge/Pandas-2.1+-150458.svg)

This repository contains the code, data, and models for a university Machine Learning group project. The goal of this system is to analyze the mathematical base stats and typings of Pokémon (Generations 1-9) to predict their impact on game balance and their long-term viability in the competitive meta.

---

## 👥 Project Team & Roles
* **Member 1:** Data Pre-processing, Scaling, & Feature Engineering
* **Member 2:** Exploratory Data Analysis (EDA)
* **Member 3:** Designer Model (Unsupervised Balance Risk Detection)
* **Member 4:** Gamer Model (Supervised Longevity Prediction)
* **Member 5:** Model Evaluation, Tuning, & Selection
* **Member 6:** Dashboard UI/UX Deployment (Streamlit)

---

## 🧠 Model Architectures & Methodology

Our system deploys two distinct machine learning models tailored to two different stakeholders. 

### 1. The Designer Model (Balance Risk Classification)
* **Target Audience:** Game Developers and Balance Teams
* **Algorithm:** `sklearn.ensemble.IsolationForest` (Unsupervised Anomaly Detection)
* **The Methodology Shift:** Initially, we attempted a Supervised Learning approach (`balance_risk.ipynb`), using a hard-coded threshold (Base Stat Total > 580) as the target variable. We quickly identified that this caused severe **Data Leakage**—the model simply learned to sum the features rather than discovering true balance risks.
* **The Solution:** We transitioned to an Unsupervised `IsolationForest` (`balance_risk_unsupervised.ipynb`). By feeding the model the entire Pokédex without labels, it mapped the "mathematical norms" of Pokémon stat distributions. 
* **The Result:** The model successfully identifies historically "broken" min-maxed Pokémon (e.g., *Blissey*, *Shuckle*) and modern power-creep designs (e.g., *Glastrier*) as statistical anomalies (Score < 0), regardless of their overall stat total.

### 2. The Gamer Model (Competitive Longevity Prediction)
* **Target Audience:** Competitive Players and Theorycrafters
* **Algorithm:** `sklearn.ensemble.RandomForestRegressor` (Supervised Regression)
* **The Objective:** To predict how many future generations (1 to 5+) a Pokémon's stat spread will remain viable against historical power creep.
* **The Methodology:** Member 4 engineered highly specific competitive features, including:
  * **Stat Efficiency:** `Base Stat Total / Mean Dataset BST (427.68)`
  * **Type Coverage:** Number of types (Single vs. Dual typing)
* **The Result:** The Random Forest Regressor significantly outperformed baseline Linear Regression models, successfully capturing the non-linear relationship between highly specialized stat distributions and long-term meta survival.

---

## 📊 The Interactive Dashboard (`app.py`)

We deployed our models using a custom-styled Streamlit dashboard, allowing users to simulate the creation of a new Pokémon and receive real-time, ML-driven feedback.

### Key Features:
1. **Dynamic Feature Engineering:** The app automatically scales raw user inputs (1-255 bounds) to the standard distributions expected by the models (`SCALERS`) and calculates necessary derived stats (Offensive Total, Defensive Total).
2. **Explainable AI (XAI):** The Designer View doesn't just output a classification; it reveals the exact **Raw Anomaly Score** used by the `IsolationForest` and provides the sub-classification logic (comparing the user's BST to the global median of 450) to explain *why* a design is overpowered or underpowered.
3. **Dual Stakeholder Views:** A tabbed interface separates the Developer Risk Assessment from the Gamer Longevity Prediction.

---

## 🚀 How to Run the Project Locally

### Prerequisites
Ensure you have Python 3.9+ installed. We recommend using a virtual environment.

### 1. Clone the repository
```bash
git clone [https://github.com/your-username/Predictive-Meta-Balance-System.git](https://github.com/your-username/Predictive-Meta-Balance-System.git)
cd Predictive-Meta-Balance-System

```

### 2. Install Dependencies

```bash
pip install -r requirements.txt

```

*(Requires: `streamlit`, `pandas`, `numpy`, `scikit-learn`, `joblib`)*

### 3. Run the Dashboard

Navigate to the root directory and start the Streamlit server:

```bash
streamlit run app.py

```

The application will launch in your default web browser at `http://localhost:8501`.

---

## 📂 Repository Structure

```text
Predictive-Meta-Balance-System/
│
├── data/
│   ├── pokemon_complete_2025.csv                  # Raw Generation 1-9 data
│   └── final_processed_dataset.csv                # Cleaned/Scaled dataset used for training
│
├── models/
│   ├── unsupervised_balance_risk_pipeline.joblib  # Trained Isolation Forest 
│   └── longevity_RandomForest.joblib              # Trained Random Forest Regressor
│
├── notebooks/
│   ├── balance_risk_unsupervised.ipynb            # Training pipeline for the Designer Model
│   ├── longevity.ipynb                            # Training pipeline for the Gamer Model
│   └── model_evaluation.ipynb                     # Comparative metrics and visualizations
│
├── app.py                                         # The main Streamlit Dashboard application
├── requirements.txt                               # Python dependencies
└── README.md                                      # Project documentation

```

---

*Disclaimer: This is an educational project developed for university coursework. Pokémon and all related names are trademarks of Nintendo and The Pokémon Company.*

```