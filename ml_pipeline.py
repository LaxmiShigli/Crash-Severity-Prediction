from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any
import random
import re

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

KAGGLE_SOURCE_URL = "https://www.kaggle.com/datasets/nextmillionaire/car-accident-dataset"
BOOTSTRAP_DATASET_PATH = Path("data/raw/bootstrap_car_accident_dataset.csv")
BOOTSTRAP_SOURCE_LABEL = "Local Kaggle-compatible bootstrap dataset"
MAX_TRAINING_ROWS = 50000
CHUNK_SIZE = 150000
CHUNK_SAMPLE_ROWS = 1200

LABEL_TO_INT = {"Low": 0, "Medium": 1, "High": 2}
INT_TO_LABEL = {value: key for key, value in LABEL_TO_INT.items()}
LABEL_TO_SCORE = {"Low": 15.0, "Medium": 45.0, "High": 80.0}

FEATURE_COLUMNS = ["speed", "weather", "location", "time_of_day"]
NUMERIC_COLUMNS = ["speed"]
CATEGORICAL_COLUMNS = ["weather", "location", "time_of_day"]


@dataclass
class PredictionInput:
    speed: float
    weather: str
    location: str
    time_of_day: str


def slugify_column(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(name).strip().lower()).strip("_")


def first_available_column(columns: list[str], candidates: list[str]) -> str | None:
    lookup = set(columns)
    for candidate in candidates:
        if candidate in lookup:
            return candidate
    return None


def normalize_weather(value: Any) -> str:
    text = str(value or "").strip().lower()
    if not text or text == "nan":
        return "Clear"
    if any(token in text for token in ["storm", "thunder", "hail", "wind"]):
        return "Storm"
    if "snow" in text or "sleet" in text or "ice" in text:
        return "Snow"
    if "fog" in text or "mist" in text:
        return "Fog"
    if "rain" in text or "shower" in text or "drizzle" in text:
        return "Rain"
    return "Clear"


def normalize_location(road_type: Any = None, area_type: Any = None, raw_location: Any = None) -> str:
    combined = " ".join(str(item or "").lower() for item in [road_type, area_type, raw_location])
    if any(token in combined for token in ["motorway", "highway", "freeway", "expressway"]):
        return "Highway"
    if any(token in combined for token in ["suburban", "residential", "town"]):
        return "Suburban"
    if "rural" in combined:
        return "Rural"
    if "urban" in combined or "city" in combined or "street" in combined or "junction" in combined:
        return "Urban"
    return "Urban"


def estimate_speed_from_context(
    road_type: Any = None,
    area_type: Any = None,
    raw_location: Any = None,
    weather: Any = None,
    severity: Any = None,
    distance: Any = None,
) -> float:
    location_type = normalize_location(road_type=road_type, area_type=area_type, raw_location=raw_location)
    weather_type = normalize_weather(weather)
    severity_type = normalize_severity(severity) or "Medium"

    base_speed = {
        "Urban": 42.0,
        "Suburban": 58.0,
        "Rural": 72.0,
        "Highway": 96.0,
    }[location_type]

    weather_adjustment = {
        "Clear": 0.0,
        "Rain": -7.0,
        "Fog": -11.0,
        "Snow": -14.0,
        "Storm": -16.0,
    }[weather_type]

    severity_adjustment = {
        "Low": -4.0,
        "Medium": 3.0,
        "High": 11.0,
    }[severity_type]

    try:
        distance_value = float(distance)
    except (TypeError, ValueError):
        distance_value = 0.0

    distance_adjustment = min(distance_value * 45.0, 14.0)
    estimated_speed = base_speed + weather_adjustment + severity_adjustment + distance_adjustment
    return max(10.0, min(estimated_speed, 200.0))


def normalize_time_of_day(value: Any) -> str:
    text = str(value or "").strip()
    lower = text.lower()
    if not text or lower == "nan":
        return "Morning"

    if "morning" in lower:
        return "Morning"
    if "afternoon" in lower or "noon" in lower:
        return "Afternoon"
    if "evening" in lower or "dusk" in lower:
        return "Evening"
    if "night" in lower or "midnight" in lower:
        return "Night"

    try:
        parsed = datetime.strptime(text, "%H:%M")
    except ValueError:
        try:
            parsed = datetime.strptime(text, "%H:%M:%S")
        except ValueError:
            try:
                parsed = datetime.fromisoformat(text.replace("Z", ""))
            except ValueError:
                return "Morning"

    hour = parsed.hour
    if 6 <= hour < 12:
        return "Morning"
    if 12 <= hour < 17:
        return "Afternoon"
    if 17 <= hour < 21:
        return "Evening"
    return "Night"


def normalize_severity(value: Any) -> str | None:
    text = str(value or "").strip().lower()
    if not text or text == "nan":
        return None
    if any(token in text for token in ["fatal", "severe", "serious", "high"]):
        return "High"
    if any(token in text for token in ["medium", "moderate"]):
        return "Medium"
    if any(token in text for token in ["slight", "low", "minor"]):
        return "Low"
    if text.isdigit():
        numeric = int(text)
        if numeric >= 3:
            return "High"
        if numeric == 2:
            return "Medium"
        return "Low"
    return None


def severity_band_from_score(score: float) -> str:
    if score <= 20:
        return "Low"
    if score < 60:
        return "Medium"
    return "High"


def sample_balanced_frame(frame: pd.DataFrame, target_rows: int, seed: int = 42) -> pd.DataFrame:
    if frame.empty or len(frame) <= target_rows:
        return frame.reset_index(drop=True)

    samples: list[pd.DataFrame] = []
    total_rows = len(frame)
    for _, group in frame.groupby("severity", sort=False):
        proportional_rows = max(1, round(len(group) / total_rows * target_rows))
        take = min(len(group), proportional_rows)
        samples.append(group.sample(n=take, random_state=seed))

    sampled = pd.concat(samples, ignore_index=True)
    if len(sampled) > target_rows:
        sampled = sampled.sample(n=target_rows, random_state=seed)
    return sampled.sample(frac=1, random_state=seed).reset_index(drop=True)


def resolve_dataset_path(dataset_path: str | Path | None = None) -> Path | None:
    if dataset_path:
        path = Path(dataset_path)
        return path if path.exists() else None

    raw_dir = Path("data/raw")
    if not raw_dir.exists():
        return None

    preferred_names = [
        "car_accident_dataset.csv",
        "Car_Accident_Dataset.csv",
        "car accident dataset.csv",
        "accidents.csv",
    ]

    for name in preferred_names:
        candidate = raw_dir / name
        if candidate.exists():
            return candidate

    csv_files = sorted(
        raw_dir.glob("*.csv"),
        key=lambda path: (path.name == BOOTSTRAP_DATASET_PATH.name, path.name.lower()),
    )
    return csv_files[0] if csv_files else None


def build_bootstrap_dataset(
    output_path: str | Path = BOOTSTRAP_DATASET_PATH,
    rows: int = 2400,
    seed: int = 42,
) -> Path:
    rng = random.Random(seed)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    weather_options = [
        "Fine no high winds",
        "Raining no high winds",
        "Fog or mist",
        "Snowing no high winds",
        "Storm",
    ]
    road_types = [
        "One way street",
        "Single carriageway",
        "Dual carriageway",
        "Roundabout",
        "Motorway",
        "Residential street",
    ]
    area_types = ["Urban", "Rural", "Suburban"]
    junction_controls = ["Give way or uncontrolled", "Auto traffic signal", "Stop sign"]
    surface_conditions = ["Dry", "Wet or damp", "Snow", "Frost or ice"]
    vehicle_types = ["Car", "Taxi/Private hire car", "Bus", "Motorcycle", "Goods vehicle"]
    day_names = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

    records: list[dict[str, Any]] = []
    base_date = datetime(2021, 1, 1)

    for index in range(rows):
        area = rng.choices(area_types, weights=[0.5, 0.3, 0.2], k=1)[0]
        weather = rng.choices(weather_options, weights=[0.43, 0.28, 0.11, 0.08, 0.10], k=1)[0]
        road_type = rng.choices(
            road_types,
            weights=[0.14, 0.27, 0.18, 0.12, 0.17, 0.12],
            k=1,
        )[0]
        time_hour = rng.randint(0, 23)
        time_minute = rng.choice([0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55])
        accident_date = base_date.replace(day=((index % 28) + 1))

        speed = rng.randint(20, 60)
        if area == "Rural":
            speed += rng.randint(10, 40)
        if area == "Suburban":
            speed += rng.randint(0, 25)
        if road_type == "Motorway":
            speed = rng.randint(75, 130)
        elif road_type == "Dual carriageway":
            speed = max(speed, rng.randint(45, 95))
        speed = max(10, min(speed, 130))

        risk_score = 0
        risk_score += max(speed - 35, 0) * 0.55
        risk_score += 14 if weather == "Raining no high winds" else 0
        risk_score += 18 if weather == "Fog or mist" else 0
        risk_score += 23 if weather == "Snowing no high winds" else 0
        risk_score += 28 if weather == "Storm" else 0
        risk_score += 18 if road_type == "Motorway" else 0
        risk_score += 10 if area == "Rural" else 0
        risk_score += 6 if area == "Suburban" else 0
        risk_score += 16 if time_hour >= 21 or time_hour < 5 else 0
        risk_score += 8 if 17 <= time_hour < 21 else 0
        risk_score += rng.uniform(-10, 10)

        if risk_score >= 68:
            severity = "Fatal"
        elif risk_score >= 34:
            severity = "Moderate"
        else:
            severity = "Slight"

        records.append(
            {
                "Accident_Index": f"A{202101:06d}{index:05d}",
                "Accident Date": accident_date.strftime("%d/%m/%Y"),
                "Day_of_Week": day_names[accident_date.weekday()],
                "Junction_Control": rng.choice(junction_controls),
                "Junction_Detail": rng.choice(
                    ["T or staggered junction", "Roundabout", "Not at junction or within 20 metres"]
                ),
                "Accident_Severity": severity,
                "Latitude": round(51.48 + rng.uniform(-0.08, 0.08), 6),
                "Light_Conditions": (
                    "Darkness - lights lit"
                    if time_hour >= 19 or time_hour < 6
                    else "Daylight"
                ),
                "Local_Authority_(District)": rng.choice(
                    ["Kensington and Chelsea", "Westminster", "Camden", "Hammersmith and Fulham"]
                ),
                "Carriageway_Hazards": "None",
                "Longitude": round(-0.18 + rng.uniform(-0.08, 0.08), 6),
                "Number_of_Casualties": 1 if severity == "Slight" else (2 if severity == "Serious" else 3),
                "Number_of_Vehicles": rng.randint(1, 4),
                "Police_Force": "Metropolitan Police",
                "Road_Surface_Conditions": rng.choice(surface_conditions),
                "Road_Type": road_type,
                "Speed_limit": speed,
                "Time": f"{time_hour:02d}:{time_minute:02d}",
                "Urban_or_Rural_Area": area,
                "Weather_Conditions": weather,
                "Vehicle_Type": rng.choice(vehicle_types),
            }
        )

    pd.DataFrame.from_records(records).to_csv(output, index=False)
    return output


def prepare_training_frame(
    dataset_path: str | Path | None = None,
    allow_bootstrap: bool = True,
) -> tuple[pd.DataFrame, Path, bool]:
    resolved_path = resolve_dataset_path(dataset_path)
    if resolved_path is None:
        if not allow_bootstrap:
            raise FileNotFoundError(
                "No dataset CSV was found in data/raw. Download the Kaggle dataset first."
            )
        resolved_path = build_bootstrap_dataset()
        used_bootstrap = True
    else:
        used_bootstrap = resolved_path.name == BOOTSTRAP_DATASET_PATH.name
        if used_bootstrap and allow_bootstrap:
            resolved_path = build_bootstrap_dataset(output_path=resolved_path)

    header_frame = pd.read_csv(resolved_path, nrows=0)
    original_columns = list(header_frame.columns)
    normalized_columns = [slugify_column(column) for column in original_columns]
    original_by_normalized = dict(zip(normalized_columns, original_columns))
    columns = normalized_columns
    speed_col = first_available_column(columns, ["speed_limit", "vehicle_speed", "speed"])
    weather_col = first_available_column(columns, ["weather_conditions", "weather_condition", "weather"])
    time_col = first_available_column(columns, ["time_of_day", "time", "start_time"])
    severity_col = first_available_column(columns, ["accident_severity", "severity"])
    road_type_col = first_available_column(columns, ["road_type", "type_of_road"])
    area_col = first_available_column(columns, ["urban_or_rural_area", "area_type", "location_type"])
    raw_location_col = first_available_column(columns, ["street", "city", "county", "state", "description"])
    distance_col = first_available_column(columns, ["distance_mi", "distance"])

    missing = [
        label
        for label, column in {
            "weather": weather_col,
            "time": time_col,
            "severity": severity_col,
        }.items()
        if column is None
    ]
    if missing:
        raise ValueError(f"Dataset is missing required columns: {', '.join(missing)}")

    selected_columns = [
        original_by_normalized[column]
        for column in [
            speed_col,
            weather_col,
            time_col,
            severity_col,
            road_type_col,
            area_col,
            raw_location_col,
            distance_col,
        ]
        if column is not None
    ]

    prepared_chunks: list[pd.DataFrame] = []
    for chunk in pd.read_csv(resolved_path, usecols=selected_columns, chunksize=CHUNK_SIZE):
        chunk.columns = [slugify_column(column) for column in chunk.columns]
        raw_sample_size = CHUNK_SAMPLE_ROWS * 4
        if len(chunk) > raw_sample_size:
            chunk = chunk.sample(n=raw_sample_size, random_state=42).reset_index(drop=True)

        if speed_col:
            speed_series = pd.to_numeric(chunk[speed_col], errors="coerce")
        else:
            speed_series = chunk.apply(
                lambda row: estimate_speed_from_context(
                    road_type=row[road_type_col] if road_type_col else None,
                    area_type=row[area_col] if area_col else None,
                    raw_location=row[raw_location_col] if raw_location_col else None,
                    weather=row[weather_col],
                    severity=row[severity_col],
                    distance=row[distance_col] if distance_col else None,
                ),
                axis=1,
            )

        prepared_chunk = pd.DataFrame(
            {
                "speed": pd.to_numeric(speed_series, errors="coerce"),
                "weather": chunk[weather_col].apply(normalize_weather),
                "location": chunk.apply(
                    lambda row: normalize_location(
                        road_type=row[road_type_col] if road_type_col else None,
                        area_type=row[area_col] if area_col else None,
                        raw_location=row[raw_location_col] if raw_location_col else None,
                    ),
                    axis=1,
                ),
                "time_of_day": chunk[time_col].apply(normalize_time_of_day),
                "severity": chunk[severity_col].apply(normalize_severity),
            }
        )

        prepared_chunk = prepared_chunk.dropna(subset=["speed", "severity"]).copy()
        prepared_chunk["speed"] = prepared_chunk["speed"].clip(lower=0, upper=200)
        prepared_chunk = prepared_chunk[prepared_chunk["severity"].isin(LABEL_TO_INT.keys())]

        if prepared_chunk.empty:
            continue

        prepared_chunks.append(sample_balanced_frame(prepared_chunk, CHUNK_SAMPLE_ROWS))

    if not prepared_chunks:
        raise ValueError("Dataset could not be normalized into training records.")

    prepared = pd.concat(prepared_chunks, ignore_index=True)

    if len(prepared) > MAX_TRAINING_ROWS:
        prepared = sample_balanced_frame(prepared, MAX_TRAINING_ROWS)

    return prepared.reset_index(drop=True), resolved_path, used_bootstrap


def build_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            ("numeric", StandardScaler(), NUMERIC_COLUMNS),
            ("categorical", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_COLUMNS),
        ]
    )


def build_model_specs() -> dict[str, Any]:
    return {
        "Random Forest": RandomForestClassifier(
            n_estimators=250,
            max_depth=10,
            min_samples_split=4,
            random_state=42,
        ),
        "Decision Tree": DecisionTreeClassifier(
            max_depth=8,
            min_samples_split=6,
            random_state=42,
        ),
        "Logistic Regression": LogisticRegression(
            max_iter=1200,
        ),
        "XGBoost": XGBClassifier(
            objective="multi:softprob",
            num_class=3,
            n_estimators=180,
            max_depth=5,
            learning_rate=0.08,
            subsample=0.9,
            colsample_bytree=0.9,
            eval_metric="mlogloss",
            random_state=42,
        ),
    }


def train_and_save(
    dataset_path: str | Path | None = None,
    artifact_path: str | Path = "artifacts/model_bundle.joblib",
) -> dict[str, Any]:
    training_frame, resolved_path, used_bootstrap = prepare_training_frame(dataset_path)
    X = training_frame[FEATURE_COLUMNS]
    y = training_frame["severity"].map(LABEL_TO_INT)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    models: dict[str, Pipeline] = {}
    metrics: dict[str, dict[str, Any]] = {}

    for model_name, estimator in build_model_specs().items():
        pipeline = Pipeline(
            steps=[
                ("preprocessor", build_preprocessor()),
                ("model", estimator),
            ]
        )
        pipeline.fit(X_train, y_train)
        predictions = pipeline.predict(X_test)

        metrics[model_name] = {
            "accuracy": round(accuracy_score(y_test, predictions) * 100, 2),
            "report": classification_report(
                y_test.map(INT_TO_LABEL),
                pd.Series(predictions).map(INT_TO_LABEL),
                output_dict=True,
                zero_division=0,
            ),
        }
        models[model_name] = pipeline

    bundle = {
        "models": models,
        "metrics": metrics,
        "feature_columns": FEATURE_COLUMNS,
        "dataset_source": BOOTSTRAP_SOURCE_LABEL if used_bootstrap else KAGGLE_SOURCE_URL,
        "dataset_mode": "bootstrap" if used_bootstrap else "kaggle",
        "dataset_path": str(resolved_path),
        "trained_at": datetime.now().isoformat(timespec="seconds"),
        "labels": INT_TO_LABEL,
    }

    artifact = Path(artifact_path)
    artifact.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, artifact)
    return bundle


def load_bundle(artifact_path: str | Path = "artifacts/model_bundle.joblib") -> dict[str, Any] | None:
    artifact = Path(artifact_path)
    if not artifact.exists():
        return None
    return joblib.load(artifact)


def prepare_input_frame(form_data: PredictionInput | dict[str, Any]) -> pd.DataFrame:
    if isinstance(form_data, PredictionInput):
        payload = {
            "speed": float(form_data.speed),
            "weather": normalize_weather(form_data.weather),
            "location": normalize_location(raw_location=form_data.location),
            "time_of_day": normalize_time_of_day(form_data.time_of_day),
        }
    else:
        payload = {
            "speed": float(form_data["speed"]),
            "weather": normalize_weather(form_data["weather"]),
            "location": normalize_location(raw_location=form_data["location"]),
            "time_of_day": normalize_time_of_day(form_data["time_of_day"]),
        }

    return pd.DataFrame([payload], columns=FEATURE_COLUMNS)


def predict_severity(bundle: dict[str, Any], form_data: PredictionInput | dict[str, Any]) -> dict[str, Any]:
    feature_frame = prepare_input_frame(form_data)

    model_results: list[dict[str, Any]] = []
    aggregate_score = 0.0
    aggregate_probabilities = {"Low": 0.0, "Medium": 0.0, "High": 0.0}

    for model_name, pipeline in bundle["models"].items():
        probabilities = pipeline.predict_proba(feature_frame)[0]
        class_mapping = [INT_TO_LABEL[int(label)] for label in pipeline.named_steps["model"].classes_]
        probability_by_label = {
            label: float(probability)
            for label, probability in zip(class_mapping, probabilities)
        }

        score = sum(
            probability_by_label.get(label, 0.0) * severity_score
            for label, severity_score in LABEL_TO_SCORE.items()
        )
        band = severity_band_from_score(score)

        for label in aggregate_probabilities:
            aggregate_probabilities[label] += probability_by_label.get(label, 0.0)

        aggregate_score += score
        model_results.append(
            {
                "name": model_name,
                "score": round(score, 2),
                "band": band,
                "probabilities": {
                    label: round(probability_by_label.get(label, 0.0) * 100, 2)
                    for label in ["Low", "Medium", "High"]
                },
                "accuracy": bundle["metrics"][model_name]["accuracy"],
            }
        )

    model_count = max(len(model_results), 1)
    overall_score = round(aggregate_score / model_count, 2)
    overall_band = severity_band_from_score(overall_score)

    averaged_probabilities = {
        label: round((value / model_count) * 100, 2)
        for label, value in aggregate_probabilities.items()
    }

    return {
        "input": feature_frame.iloc[0].to_dict(),
        "overall_score": overall_score,
        "overall_band": overall_band,
        "graph_marker": max(min(overall_score, 100), 0),
        "distribution": averaged_probabilities,
        "models": model_results,
        "recommended_action": build_recommendation(overall_band, overall_score),
    }


def build_recommendation(band: str, score: float) -> str:
    if band == "High":
        return (
            f"Predicted risk is high at {score:.2f}%. Slow down immediately, avoid sharp maneuvers, "
            "and use the safest available route."
        )
    if band == "Medium":
        return (
            f"Predicted risk is medium at {score:.2f}%. Maintain caution, keep safe distance, "
            "and stay alert to changing road and weather conditions."
        )
    return (
        f"Predicted risk is low at {score:.2f}%. Continue driving carefully and keep monitoring "
        "speed, surroundings, and road conditions."
    )
