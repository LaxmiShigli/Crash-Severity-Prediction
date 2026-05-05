from __future__ import annotations

from pathlib import Path

from flask import Flask, render_template, request

from ml_pipeline import (
    KAGGLE_SOURCE_URL,
    PredictionInput,
    load_bundle,
    predict_severity,
    train_and_save,
)

app = Flask(__name__)

ARTIFACT_PATH = Path("artifacts/model_bundle.joblib")
DATA_DIR = Path("data/raw")

HERO_SLIDES = [
    {
        "eyebrow": "Using Machine Learning",
        "title": "Crash Severity Prediction",
        "description": (
            "Leverage Random Forest, Decision Tree, Logistic Regression, and XGBoost "
            "to predict accident severity with high accuracy."
        ),
        "action": "Start Prediction",
    },
    {
        "eyebrow": "Powered By Kaggle Dataset",
        "title": "Data-Driven Safety",
        "description": (
            "Train on real-world accident records using features like speed, weather "
            "conditions, location, and time of day."
        ),
        "action": "View Introduction",
    },
    {
        "eyebrow": "From Input To Insight",
        "title": "Predict. Visualize. Act.",
        "description": (
            "Enter crash parameters and view the overall severity as a graph with clear "
            "Low, Medium, and High risk bands."
        ),
        "action": "See Dashboard",
    },
]

FEATURE_OPTIONS = {
    "weather": ["Clear", "Rain", "Fog", "Snow", "Storm"],
    "location": ["Urban", "Rural", "Highway", "Suburban"],
    "time_of_day": ["Morning", "Afternoon", "Evening", "Night"],
}


def real_dataset_available() -> bool:
    if not DATA_DIR.exists():
        return False
    return any(path.name != "bootstrap_car_accident_dataset.csv" for path in DATA_DIR.glob("*.csv"))


def ensure_bundle() -> tuple[dict | None, str | None]:
    bundle = load_bundle(ARTIFACT_PATH)
    if bundle is not None:
        if bundle.get("dataset_mode") == "bootstrap" and real_dataset_available():
            try:
                bundle = train_and_save(artifact_path=ARTIFACT_PATH)
            except Exception as exc:  # pragma: no cover - template handles message
                return None, str(exc)
        if bundle.get("dataset_mode") == "bootstrap":
            return bundle, (
                "Running with an auto-generated Kaggle-compatible local dataset because the real "
                "Kaggle CSV is not available yet. Add the real CSV to data/raw anytime to retrain "
                "on Kaggle records."
            )
        return bundle, None

    try:
        bundle = train_and_save(artifact_path=ARTIFACT_PATH)
        if bundle.get("dataset_mode") == "bootstrap":
            return bundle, (
                "No Kaggle CSV was found, so the app created a Kaggle-compatible bootstrap dataset "
                "automatically and trained the models locally."
            )
        return bundle, None
    except Exception as exc:  # pragma: no cover - template handles message
        return None, str(exc)


def build_template_context(**extra: object) -> dict[str, object]:
    bundle, model_message = ensure_bundle()
    return {
        "slides": HERO_SLIDES,
        "feature_options": FEATURE_OPTIONS,
        "model_ready": bundle is not None,
        "model_message": model_message,
        "dataset_source": KAGGLE_SOURCE_URL,
        "result": None,
        "form_values": {
            "speed": 60,
            "weather": "",
            "location": "",
            "time_of_day": "",
        },
        **extra,
    }


@app.route("/", methods=["GET"])
def home():
    return render_template("index.html", **build_template_context())


@app.route("/prediction", methods=["GET"])
@app.route("/predict", methods=["GET"])
def prediction_page():
    return render_template("prediction.html", **build_template_context())


@app.route("/result", methods=["GET", "POST"])
def result_page():
    if request.method == "GET":
        return render_template("result.html", **build_template_context())

    bundle, model_message = ensure_bundle()
    form_values = {
        "speed": request.form.get("speed", 60),
        "weather": request.form.get("weather", ""),
        "location": request.form.get("location", ""),
        "time_of_day": request.form.get("time_of_day", ""),
    }

    if bundle is None:
        return render_template(
            "result.html",
            **build_template_context(
                form_values=form_values,
                result=None,
                model_message=model_message,
            ),
        )

    try:
        prediction_input = PredictionInput(
            speed=float(form_values["speed"]),
            weather=form_values["weather"],
            location=form_values["location"],
            time_of_day=form_values["time_of_day"],
        )
        result = predict_severity(bundle, prediction_input)
    except Exception as exc:  # pragma: no cover - user input validation path
        return render_template(
            "result.html",
            **build_template_context(
                form_values=form_values,
                result=None,
                model_message=f"Prediction failed: {exc}",
            ),
        )

    return render_template(
        "result.html",
        **build_template_context(
            form_values=form_values,
            result=result,
            model_message=model_message,
        ),
    )


if __name__ == "__main__":
    app.run(debug=True)
