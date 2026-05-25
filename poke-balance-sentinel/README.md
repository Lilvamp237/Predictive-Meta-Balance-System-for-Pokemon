# Poke Balance Sentinel

End-to-end machine learning project scaffold for analyzing Pokemon power creep and building a **Power Creep Index (PCI)** model.

## Structure
- `data/` — raw and processed CSVs
- `notebooks/` — EDA and experimentation
- `src/` — preprocessing + model utilities
- `models/` — saved model artifacts
- `app/` — Streamlit deployment

## Quickstart
1. Create a virtual environment and install dependencies:
   ```
   python -m venv .venv
   .venv\Scripts\activate
   pip install -r requirements.txt
   ```
2. Place your dataset in `data\pokemon_complete_2025.csv`.
3. Train the model:
   ```
   python train.py --data data\pokemon_complete_2025.csv --model-out models\random_forest_pci.joblib
   ```
4. Launch the Streamlit app:
   ```
   streamlit run app\main.py
   ```

Open `notebooks\eda_and_training.ipynb` for quick EDA and model experimentation.

## PCI Definition
The **Power Creep Index (PCI)** is computed as:

`PCI = base_stat_total / mean(base_stat_total within the same generation)`

This keeps PCI interpretable across eras while highlighting power creep trends.
