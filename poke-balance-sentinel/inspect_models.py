from pathlib import Path
import joblib

MODEL_DIR = Path("models")

for model_path in MODEL_DIR.glob("*.joblib"):
    print("=" * 80)
    print("MODEL:", model_path.name)

    try:
        model = joblib.load(model_path)
        print("Type:", type(model))

        if hasattr(model, "feature_names_in_"):
            print("Feature names:")
            for f in model.feature_names_in_:
                print(" -", f)
        else:
            print("feature_names_in_: Not found")

        if hasattr(model, "n_features_in_"):
            print("n_features_in_:", model.n_features_in_)

        if hasattr(model, "classes_"):
            print("classes_:", model.classes_)

        print("Has predict:", hasattr(model, "predict"))
        print("Has predict_proba:", hasattr(model, "predict_proba"))

    except Exception as e:
        print("Could not load:", e)