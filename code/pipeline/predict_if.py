"""
Feature 8: Imprinting Factor (IF) Prediction Model
====================================================
Trains a regression model on literature MIP data to predict
imprinting factors from DFT binding energies and solvent properties.

Refs: Singh et al. 2012 (DOI: 10.2174/157341112803216807),
      Mukasa et al. 2023 (DOI: 10.1002/adma.202212161).
"""

import json
import logging
import os
import warnings
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import LeaveOneOut

from .config import OUTPUT_DIR, OUTPUT_DIRS, SOLVENTS, SYNTHESIS_SOLVENT

logging.basicConfig(level=logging.INFO, format="%(asctime)s [PredictIF] %(message)s")
logger = logging.getLogger(__name__)

# ── Literature Training Data ────────────────────────────────────────

LITERATURE_DATA = [
    # Singh 2012 (DOI: 10.2174/157341112803216807)
    {"template": "heptachlor", "monomer": "MAA", "bsse_dE": -8.11, "solvent_eps": 4.71, "IF": 1.92},
    {"template": "heptachlor", "monomer": "4VP", "bsse_dE": -7.76, "solvent_eps": 4.71, "IF": 1.45},
    {"template": "heptachlor", "monomer": "styrene", "bsse_dE": -4.13, "solvent_eps": 4.71, "IF": 0.99},
    {"template": "DDT", "monomer": "MAA", "bsse_dE": -8.99, "solvent_eps": 4.71, "IF": 1.43},
    {"template": "DDT", "monomer": "4VP", "bsse_dE": -7.86, "solvent_eps": 4.71, "IF": 1.65},
    {"template": "DDT", "monomer": "styrene", "bsse_dE": -8.28, "solvent_eps": 4.71, "IF": 1.21},
    # Mukasa 2023 (DOI: 10.1002/adma.202212161)
    {"template": "Phe", "monomer": "OPD", "bsse_dE": -11.9, "solvent_eps": 32.61, "IF": 3.2},
    {"template": "Phe", "monomer": "MAA", "bsse_dE": -11.5, "solvent_eps": 32.61, "IF": 2.8},
    {"template": "Phe", "monomer": "4VB", "bsse_dE": -10.8, "solvent_eps": 32.61, "IF": 2.4},
    {"template": "Phe", "monomer": "PYR", "bsse_dE": -6.4, "solvent_eps": 32.61, "IF": 1.2},
]


# ── Feature Engineering ─────────────────────────────────────────────

def _build_features(bsse_dE: float, solvent_eps: float) -> np.ndarray:
    """Build feature vector: [bsse_dE, solvent_eps, bsse_dE * solvent_eps]."""
    return np.array([bsse_dE, solvent_eps, bsse_dE * solvent_eps])


def _build_Xy(data: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    """Build X matrix and y vector from data records."""
    X = np.array([_build_features(d["bsse_dE"], d["solvent_eps"]) for d in data])
    y = np.array([d["IF"] for d in data])
    return X, y


# ── Model Training & Selection ──────────────────────────────────────

def _get_candidate_models() -> dict:
    """Return dict of model_name -> model instance."""
    return {
        "LinearRegression": LinearRegression(),
        "Ridge(alpha=1.0)": Ridge(alpha=1.0),
        "RandomForest(n=100,d=3)": RandomForestRegressor(
            n_estimators=100, max_depth=3, random_state=42
        ),
    }


def _loo_cv(model, X: np.ndarray, y: np.ndarray) -> tuple[float, np.ndarray]:
    """
    Leave-One-Out cross-validation.

    Returns
    -------
    rmse : float
        LOO-CV RMSE.
    residuals : np.ndarray
        Residuals for each fold (y_true - y_pred).
    """
    loo = LeaveOneOut()
    residuals = np.zeros(len(y))
    for train_idx, test_idx in loo.split(X):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        residuals[test_idx[0]] = y_test[0] - y_pred[0]
    rmse = np.sqrt(np.mean(residuals ** 2))
    return rmse, residuals


def _select_best_model(X: np.ndarray, y: np.ndarray) -> tuple[object, str, dict]:
    """
    Train all candidate models, evaluate via LOO-CV, select best.

    Returns
    -------
    best_model : fitted sklearn estimator
    best_name : str
    results : dict  {name: {"rmse": float, "residuals": list}}
    """
    candidates = _get_candidate_models()
    results = {}
    best_rmse = np.inf
    best_name = None

    for name, model in candidates.items():
        rmse, residuals = _loo_cv(model, X, y)
        results[name] = {"rmse": rmse, "residuals": residuals.tolist()}
        logger.info(f"  {name:35s}  LOO-CV RMSE = {rmse:.4f}")
        if rmse < best_rmse:
            best_rmse = rmse
            best_name = name

    # Refit best model on full data
    best_model = _get_candidate_models()[best_name]
    best_model.fit(X, y)

    return best_model, best_name, results


# ── Prediction with Confidence Interval ─────────────────────────────

def _predict_with_ci(
    model, X_new: np.ndarray, residual_std: float, confidence: float = 0.95
) -> list[dict]:
    """
    Predict IF with approximate confidence interval based on LOO-CV residual std.

    Uses a normal approximation: prediction +/- z * residual_std.
    """
    from scipy import stats

    z = stats.norm.ppf(1 - (1 - confidence) / 2)
    y_pred = model.predict(X_new)
    predictions = []
    for i, yp in enumerate(y_pred):
        predictions.append({
            "IF_pred": round(float(yp), 3),
            "CI_lower": round(float(yp - z * residual_std), 3),
            "CI_upper": round(float(yp + z * residual_std), 3),
        })
    return predictions


# ── Main Training Function ──────────────────────────────────────────

def train_if_model(output_dir: str = OUTPUT_DIRS["features"]) -> dict:
    """
    Train IF prediction model and apply to Stage 2 DFT results.

    Returns
    -------
    dict with keys: "model_name", "loo_cv_rmse", "predictions", "cv_results"
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # ── Warning about small training set ────────────────────────────
    n_train = len(LITERATURE_DATA)
    logger.warning(
        f"Training on N={n_train} literature data points. "
        "Predictions carry high uncertainty — validate experimentally."
    )

    # ── Build features from literature data ─────────────────────────
    X, y = _build_Xy(LITERATURE_DATA)

    # ── Model comparison ────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("Model Comparison (LOO-CV RMSE)")
    logger.info("-" * 60)
    best_model, best_name, cv_results = _select_best_model(X, y)
    logger.info("-" * 60)
    logger.info(f"Selected model: {best_name}")
    logger.info("=" * 60)

    best_rmse = cv_results[best_name]["rmse"]
    residual_std = np.std(cv_results[best_name]["residuals"], ddof=1)

    # ── Load Stage 2 DFT results ────────────────────────────────────
    dft_path = output_path.parent / "stage2" / "stage2_dft.json"
    if not dft_path.exists():
        logger.error(f"Stage 2 results not found: {dft_path}")
        logger.info("Returning model info only (no pipeline predictions).")
        result = {
            "model_name": best_name,
            "loo_cv_rmse": round(best_rmse, 4),
            "residual_std": round(residual_std, 4),
            "n_training": n_train,
            "cv_results": {k: {"rmse": round(v["rmse"], 4)} for k, v in cv_results.items()},
            "predictions": [],
        }
        _save_outputs(result, best_model, LITERATURE_DATA, output_path)
        return result

    with open(dft_path) as f:
        dft_results = json.load(f)

    # ── Determine solvent dielectric constant ───────────────────────
    solvent_eps = SOLVENTS.get(SYNTHESIS_SOLVENT)
    if solvent_eps is None:
        logger.warning(
            f"SYNTHESIS_SOLVENT '{SYNTHESIS_SOLVENT}' not in SOLVENTS dict; "
            "defaulting to first available solvent."
        )
        solvent_eps = next(iter(SOLVENTS.values()))

    # ── Build prediction features ───────────────────────────────────
    predictions = []
    monomer_names = []
    X_new_list = []

    # dft_results can be dict {monomer: {solvent: {bsse_dE:...}}} or list
    if isinstance(dft_results, dict):
        for monomer, solvent_data in dft_results.items():
            # Pick energy for synthesis solvent, or first available
            if SYNTHESIS_SOLVENT in solvent_data:
                bsse_dE = solvent_data[SYNTHESIS_SOLVENT].get("bsse_dE")
            else:
                first_solvent = next(iter(solvent_data.values()), {})
                bsse_dE = first_solvent.get("bsse_dE") if isinstance(first_solvent, dict) else None
            if bsse_dE is None:
                logger.warning(f"No binding energy for {monomer}, skipping.")
                continue
            monomer_names.append(monomer)
            X_new_list.append(_build_features(float(bsse_dE), solvent_eps))
    else:
        for entry in dft_results:
            monomer = entry.get("monomer", entry.get("name", "unknown"))
            bsse_dE = entry.get("bsse_dE") or entry.get("binding_energy_kcal")
            if bsse_dE is None:
                continue
            monomer_names.append(monomer)
            X_new_list.append(_build_features(float(bsse_dE), solvent_eps))

    if not X_new_list:
        logger.warning("No valid monomers found in DFT results.")
        result = {
            "model_name": best_name,
            "loo_cv_rmse": round(best_rmse, 4),
            "residual_std": round(residual_std, 4),
            "n_training": n_train,
            "cv_results": {k: {"rmse": round(v["rmse"], 4)} for k, v in cv_results.items()},
            "predictions": [],
        }
        _save_outputs(result, best_model, LITERATURE_DATA, output_path)
        return result

    X_new = np.array(X_new_list)
    preds = _predict_with_ci(best_model, X_new, residual_std)

    # ── Assemble predictions ────────────────────────────────────────
    for i, monomer in enumerate(monomer_names):
        pred_entry = {
            "monomer": monomer,
            "bsse_dE": float(X_new[i, 0]),
            "solvent_eps": float(X_new[i, 1]),
            **preds[i],
        }
        predictions.append(pred_entry)

    # Sort by predicted IF descending
    predictions.sort(key=lambda x: x["IF_pred"], reverse=True)

    # ── Console output ──────────────────────────────────────────────
    logger.info("")
    logger.info("Predicted Imprinting Factors")
    logger.info("-" * 70)
    logger.info(f"{'Monomer':>15s}  {'dE (kcal/mol)':>14s}  {'IF_pred':>8s}  {'95% CI':>20s}")
    logger.info("-" * 70)
    for p in predictions:
        ci_str = f"[{p['CI_lower']:.3f}, {p['CI_upper']:.3f}]"
        logger.info(
            f"{p['monomer']:>15s}  {p['bsse_dE']:>14.2f}  {p['IF_pred']:>8.3f}  {ci_str:>20s}"
        )
    logger.info("-" * 70)
    logger.info(
        f"Note: 95% CI based on LOO-CV residual std = {residual_std:.4f} "
        f"(N={n_train} training points)"
    )

    # ── Assemble result ─────────────────────────────────────────────
    result = {
        "model_name": best_name,
        "loo_cv_rmse": round(best_rmse, 4),
        "residual_std": round(residual_std, 4),
        "n_training": n_train,
        "solvent": SYNTHESIS_SOLVENT,
        "solvent_eps": solvent_eps,
        "cv_results": {k: {"rmse": round(v["rmse"], 4)} for k, v in cv_results.items()},
        "predictions": predictions,
    }

    _save_outputs(result, best_model, LITERATURE_DATA, output_path)
    return result


# ── Save Outputs ────────────────────────────────────────────────────

def _save_outputs(
    result: dict, model: object, training_data: list[dict], output_path: Path
) -> None:
    """Save prediction JSON and trained model pickle."""
    # Save predictions JSON
    json_path = output_path / "predict_if.json"
    with open(json_path, "w") as f:
        json.dump(result, f, indent=2)
    logger.info(f"Predictions saved to {json_path}")

    # Save trained model + training data via joblib
    model_path = output_path / "predict_if_model.pkl"
    model_bundle = {
        "model": model,
        "training_data": training_data,
        "model_name": result["model_name"],
        "residual_std": result["residual_std"],
    }
    joblib.dump(model_bundle, model_path)
    logger.info(f"Trained model saved to {model_path}")


# ── Model Update Function ──────────────────────────────────────────

def update_model(
    monomer: str, experimental_if: float, output_dir: str = OUTPUT_DIR
) -> dict:
    """
    Update the IF model with a new experimental data point.

    Loads the existing model bundle, appends the new observation,
    retrains, and saves updated outputs.

    Parameters
    ----------
    monomer : str
        Monomer name (must exist in the latest predict_if.json predictions).
    experimental_if : float
        Experimentally measured imprinting factor.
    output_dir : str
        Output directory.

    Returns
    -------
    dict with updated model info and predictions.
    """
    output_path = Path(output_dir)

    # Load existing model bundle
    model_path = output_path / "predict_if_model.pkl"
    if not model_path.exists():
        raise FileNotFoundError(
            f"No existing model found at {model_path}. Run train_if_model() first."
        )
    bundle = joblib.load(model_path)
    training_data = list(bundle["training_data"])

    # Load latest predictions to get bsse_dE and solvent_eps
    json_path = output_path / "predict_if.json"
    if not json_path.exists():
        raise FileNotFoundError(f"No predictions file at {json_path}.")
    with open(json_path) as f:
        prev_result = json.load(f)

    # Find the monomer in predictions
    match = None
    for p in prev_result.get("predictions", []):
        if p["monomer"] == monomer:
            match = p
            break

    if match is None:
        raise ValueError(
            f"Monomer '{monomer}' not found in predictions. "
            f"Available: {[p['monomer'] for p in prev_result.get('predictions', [])]}"
        )

    # Append new data point
    new_point = {
        "template": "experimental",
        "monomer": monomer,
        "bsse_dE": match["bsse_dE"],
        "solvent_eps": match["solvent_eps"],
        "IF": experimental_if,
    }
    training_data.append(new_point)
    logger.info(
        f"Added experimental point: {monomer}, IF={experimental_if:.2f} "
        f"(N={len(training_data)} total training points)"
    )

    # Retrain
    X, y = _build_Xy(training_data)
    logger.info("Retraining model with updated data...")
    best_model, best_name, cv_results = _select_best_model(X, y)
    best_rmse = cv_results[best_name]["rmse"]
    residual_std = np.std(cv_results[best_name]["residuals"], ddof=1)

    # Re-predict for pipeline monomers
    dft_path = output_path.parent / "stage2" / "stage2_dft.json"
    predictions = []
    if dft_path.exists():
        with open(dft_path) as f:
            dft_results = json.load(f)

        solvent_eps = SOLVENTS.get(SYNTHESIS_SOLVENT, next(iter(SOLVENTS.values())))
        X_new_list = []
        monomer_names = []
        for entry in dft_results:
            mon = entry.get("monomer", entry.get("name", "unknown"))
            bsse_dE = entry.get("bsse_dE") or entry.get("binding_energy_kcal")
            if bsse_dE is None:
                continue
            monomer_names.append(mon)
            X_new_list.append(_build_features(float(bsse_dE), solvent_eps))

        if X_new_list:
            X_new = np.array(X_new_list)
            preds = _predict_with_ci(best_model, X_new, residual_std)
            for i, mon in enumerate(monomer_names):
                predictions.append({
                    "monomer": mon,
                    "bsse_dE": float(X_new[i, 0]),
                    "solvent_eps": float(X_new[i, 1]),
                    **preds[i],
                })
            predictions.sort(key=lambda x: x["IF_pred"], reverse=True)

    result = {
        "model_name": best_name,
        "loo_cv_rmse": round(best_rmse, 4),
        "residual_std": round(residual_std, 4),
        "n_training": len(training_data),
        "cv_results": {k: {"rmse": round(v["rmse"], 4)} for k, v in cv_results.items()},
        "predictions": predictions,
    }

    _save_outputs(result, best_model, training_data, output_path)
    logger.info("Model updated successfully.")
    return result


# ── Entry Point ─────────────────────────────────────────────────────

if __name__ == "__main__":
    train_if_model()
