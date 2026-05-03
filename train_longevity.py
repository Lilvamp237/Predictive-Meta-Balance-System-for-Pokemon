# train_longevity.py
import pandas as pd
from sklearn.model_selection import train_test_split
from src.models import train_random_forest, train_linear_regression, evaluate_regressor, save_model

# 1. Load data
df = pd.read_csv('data/final_processed_dataset.csv')

# 2. Create target (Generations of Viability proxy from stat_tier)
tier_map = {
    'Weak (<300)': 1,
    'Below Average (300-399)': 2,
    'Average (400-499)': 3,
    'Strong (500-599)': 4,
    'Legendary/Pseudo (600+)': 5
}
df['longevity_target'] = df['stat_tier'].map(tier_map)

# 3. Add required features: Stat Efficiency & Type Coverage
# Stat Efficiency = base_stat_total / average base_stat_total (global average)
avg_bst = df['base_stat_total'].mean()
df['stat_efficiency'] = df['base_stat_total'] / avg_bst

# Type Coverage = num_types (1 or 2) – simple but acceptable
df['type_coverage'] = df['num_types']

# 4. Prepare features (drop non-feature columns)
X = df.drop(columns=['pokedex_id', 'base_stat_total', 'stat_tier', 'longevity_target'])
y = df['longevity_target']

# 5. Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 6. Train both models
rf_model = train_random_forest(X_train, y_train, n_estimators=300)
lr_model = train_linear_regression(X_train, y_train)

# 7. Evaluate
rf_metrics = evaluate_regressor(rf_model, X_test, y_test)
lr_metrics = evaluate_regressor(lr_model, X_test, y_test)

print("Random Forest - RMSE: {:.3f}, MAE: {:.3f}, R2: {:.3f}".format(
    rf_metrics['rmse'], rf_metrics['mae'], rf_metrics['r2']))
print("Linear Regression - RMSE: {:.3f}, MAE: {:.3f}, R2: {:.3f}".format(
    lr_metrics['rmse'], lr_metrics['mae'], lr_metrics['r2']))

# 8. Save models
save_model(rf_model, 'models/longevity_RandomForest.joblib')
save_model(lr_model, 'models/longevity_LinearRegression.joblib')

# 9. Save the best model as longevity_regressor.joblib (based on lower MAE)
if rf_metrics['mae'] < lr_metrics['mae']:
    best_model = rf_model
    best_name = "Random Forest"
else:
    best_model = lr_model
    best_name = "Linear Regression"

save_model(best_model, 'models/longevity_regressor.joblib')
print(f"\n✅ Best model: {best_name} saved as longevity_regressor.joblib")