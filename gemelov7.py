# -*- coding: utf-8 -*-
"""
FALTA BNN
predict_all.py
Soporta:
 - ensembles en subcarpetas ./mod/<salida>/ o ./mod1/<salida>/
 - modelos sueltos en ./mod/modelo_<salida>.pkl (con scaler_X_<salida>.pkl, scaler_Y_<salida>.pkl, imputer_X_<salida>.pkl opcionales)

Lógica:
 - si la fila coincide con la firma de biomasa1, intenta usar mod1/<salida>/ para cada salida;
   si no hay modelos en mod1/<salida>/, usa mod/<salida>/; si tampoco, usa mod/modelo_<salida>.pkl.
 - si no es biomasa1, usa mod/<salida>/ o mod/modelo_<salida>.pkl.
"""

import os
import re
import warnings
import numpy as np
import pandas as pd
import joblib
import json
import torch
import torch.nn as nn
import pyro
from pyro.infer import Predictive
from pyro.nn import PyroModule, PyroSample
import pyro.distributions as dist

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler

warnings.filterwarnings("ignore")

# ---------- CONFIG ----------
try:
    SCRIPT_DIR = os.path.abspath(os.path.dirname(__file__))
except NameError:
    SCRIPT_DIR = os.getcwd()

DATA_DIR = os.path.join(SCRIPT_DIR, "data")
MODELS_DIR = os.path.join(SCRIPT_DIR, "mod")
MODELS_DIR_ALT = os.path.join(SCRIPT_DIR, "mod1")

INPUT_PATH = os.path.join(DATA_DIR, "input_example13.csv")  # cambia si hace falta
IDX_MUESTRA = 0
N_FEATURES_ESPERADOS = 22  # o None para no comprobar

# Firma de biomasa1: se compara contra X[5:14] (índices 0-based 5..13)
BIOMASA1_SIGNATURE = np.array([45.4, 6.73, 0.17, 0.0003, 0.0151, 72.0, 0.4, 10.6, 17.0], dtype=float)
BIOMASA1_SLICE = slice(5, 14)
BIOMASA_ATOL = 1e-6


# ---------- utilidades ----------
def to_numpy(x):
    if isinstance(x, np.ndarray):
        return x
    try:
        return x.detach().cpu().numpy()
    except Exception:
        return np.array(x)

def safe_squeeze1(a):
    a = np.array(a)
    if a.ndim > 1 and a.shape[0] == 1:
        return np.squeeze(a, axis=0)
    return a

def normalize_to_ndim(arr, target_ndim=2):
    arr = np.array(arr)
    while arr.ndim > target_ndim:
        if arr.shape[0] == 1:
            arr = np.squeeze(arr, axis=0)
        else:
            arr = arr.mean(axis=0)
    if arr.ndim == target_ndim:
        return arr
    if target_ndim == 2 and arr.ndim == 1:
        return arr.reshape(1, -1)
    raise ValueError(f"No se pudo normalizar array {arr.shape} a ndim {target_ndim}")

def numpy_forward_two_layer(x_np, fc1_w, fc1_b, out_w, out_b):
    """
    Forward numérico simple:
      x_np: (1, n_in)
      fc1_w: (hidden, n_in)
      fc1_b: (hidden,) or (1,hidden)
      out_w: (1, hidden) or (hidden, 1)
      out_b: scalar or (1,)
    Devuelve scalar (normalizado).
    """
    x_np = np.array(x_np)
    fc1_w = np.array(fc1_w)
    out_w = np.array(out_w)
    fc1_b = np.array(fc1_b)
    out_b = np.array(out_b)

    if fc1_b.ndim > 1 and fc1_b.shape[0] == 1:
        fc1_b = np.squeeze(fc1_b, axis=0)
    if out_b.ndim > 1 and out_b.shape[0] == 1:
        out_b = np.squeeze(out_b, axis=0)

    # fc1: (1, n_in) @ (n_in, hidden) -> (1, hidden)
    hidden = np.tanh(np.dot(x_np, fc1_w.T) + fc1_b.reshape(1, -1))

    # out: handle orientations robustly
    if out_w.ndim == 2 and out_w.shape[0] == 1 and out_w.shape[1] == hidden.shape[1]:
        out = np.dot(hidden, out_w.T) + out_b.reshape(1, -1)
    elif out_w.ndim == 2 and out_w.shape[1] == 1 and out_w.shape[0] == hidden.shape[1]:
        out = np.dot(hidden, out_w) + out_b.reshape(1, -1)
    else:
        # intento transponer
        try:
            out = np.dot(hidden, out_w.T) + out_b.reshape(1, -1)
        except Exception as e:
            raise RuntimeError(f"Shapes incompatibles al calcular salida: hidden.shape={hidden.shape}, out_w.shape={out_w.shape}, out_b.shape={out_b.shape}") from e

    return float(np.squeeze(out))

# ----------------------------

def safe_name(name: str) -> str:
    return re.sub(r'[<>:"/\\|?* ]', "_", str(name)).strip("_")

def read_input(path: str) -> np.ndarray:
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    ext = os.path.splitext(path)[1].lower()
    if ext in (".xlsx", ".xls"):
        df = pd.read_excel(path, header=None)
    elif ext == ".csv":
        df = pd.read_csv(path, header=None)
    else:
        raise ValueError("Formato no soportado. Usa .csv, .xlsx o .xls")
    arr = df.to_numpy()
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    return arr

def load_optional(path):
    try:
        return joblib.load(path)
    except Exception:
        return None

def model_files_in(folder: str):
    if not os.path.isdir(folder):
        return []
    return sorted([f for f in os.listdir(folder) if f.startswith("mlp_model_") and f.endswith(".pkl")])

def list_outputs_union(*roots):
    outs = set()
    for root in roots:
        if not os.path.isdir(root):
            continue
        for entry in os.listdir(root):
            p = os.path.join(root, entry)
            if os.path.isdir(p):
                files = os.listdir(p)
                if any(f.startswith("mlp_model_") and f.endswith(".pkl") for f in files) or "scaler_Y.pkl" in files:
                    outs.add(entry)
    # además incluir posibles salidas definidas por modelo suelto en MODELS_DIR: modelo_<salida>.pkl
    if os.path.isdir(MODELS_DIR):
        for f in os.listdir(MODELS_DIR):
            m = re.match(r"modelo[_-](.+)\.pkl$", f, flags=re.IGNORECASE)
            if m:
                outs.add(m.group(1))
    return sorted(outs)

def is_biomasa1(x_row: np.ndarray) -> bool:
    try:
        seg = np.array(x_row[BIOMASA1_SLICE], dtype=float).reshape(-1)
    except Exception:
        return False
    if seg.shape[0] != BIOMASA1_SIGNATURE.shape[0]:
        return False
    return bool(np.allclose(seg, BIOMASA1_SIGNATURE, atol=BIOMASA_ATOL, rtol=1e-8))

def find_model_source_for_output(salida: str, biomasa1_flag: bool):
    """Devuelve dict con info de la fuente para esa salida o None."""
    name = safe_name(salida)
    folder_alt = os.path.join(MODELS_DIR_ALT, name)
    folder_def = os.path.join(MODELS_DIR, name)
    # 1) si biomasa1 y mod1 tiene modelos -> usar mod1 folder
    if biomasa1_flag and model_files_in(folder_alt):
        return {"type": "folder", "path": folder_alt, "used": "mod1"}
    # 2) preferir mod folder if exists (either biomasa1 fallback or non-biomasa)
    if model_files_in(folder_def):
        return {"type": "folder", "path": folder_def, "used": "mod"}
    # 3) check root model file in MODELS_DIR: modelo_<name>.pkl (case-insensitive)
    # user said there will never be >1 such model for a salida in root
    candidates = []
    if os.path.isdir(MODELS_DIR):
        for f in os.listdir(MODELS_DIR):
            if re.fullmatch(rf"modelo[_-]{re.escape(name)}\.pkl", f, flags=re.IGNORECASE):
                candidates.append(os.path.join(MODELS_DIR, f))
    if candidates:
        # take first
        return {"type": "rootfile", "model_path": candidates[0], "used": "mod_root", "name": name}
    # 4) If biomasa1_flag was True and mod1 has no models but mod has none either, try mod1 even if no mlp_model_* but maybe has scaler? user specified fallback only for missing model, so skip.
    return None

def apply_imputer_and_scaler(x_proc, imputer, scaler_X):
    x = x_proc
    if imputer is not None:
        try:
            x = imputer.transform(x)
        except Exception:
            pass
    if scaler_X is not None:
        try:
            x = scaler_X.transform(x)
        except Exception:
            pass
    return x

def predict_with_models(models, x_proc):
    preds = []
    for m in models:
        try:
            p = m.predict(x_proc)
            p_val = float(np.array(p).reshape(-1)[0])
            preds.append(p_val)
        except Exception:
            continue
    return preds

def main():
    X = read_input(INPUT_PATH)
    n_samples, n_features = X.shape

    if N_FEATURES_ESPERADOS is not None and n_features != N_FEATURES_ESPERADOS:
        pass

    if IDX_MUESTRA < 0 or IDX_MUESTRA >= n_samples:
        raise IndexError(f"idx_muestra={IDX_MUESTRA} fuera de rango (n_muestras={n_samples})")

    x_input = X[IDX_MUESTRA].reshape(1, -1)
    x_row = x_input.flatten()

    biomasa1_flag = is_biomasa1(x_row)
    outputs = list_outputs_union(MODELS_DIR, MODELS_DIR_ALT) if biomasa1_flag else list_outputs_union(MODELS_DIR)

    if not outputs:
        root_used = MODELS_DIR_ALT if biomasa1_flag else MODELS_DIR
        raise FileNotFoundError(f"No se encontraron subcarpetas ni modelos en '{root_used}'")

    X_cols = [f"Input_{i+1}" for i in range(n_features)]
    print("\n=== Datos de entrada usados ===")
    print(pd.DataFrame(x_input, columns=X_cols, index=[f"Muestra {IDX_MUESTRA}"]))
    print(f"\nBiomasa1 detectada: {biomasa1_flag}\n")

    results = {}

    for salida in outputs:
        src = find_model_source_for_output(salida, biomasa1_flag)
        if src is None:
            continue

        if src["type"] == "folder":
            folder = src["path"]
            output_dir = os.path.basename(folder)

            # si es un bnn
            if output_dir.startswith("modelo_bnn"):
                safe_name = output_dir[len("modelo_bnn"):]
                print(f"\n[DEBUG] Detectado BNN para salida '{safe_name}' en carpeta {output_dir}")

                # ---------- cargar imputers/scalers de entrada ----------
                imputer_path = os.path.join(output_dir, "imputer_X.pkl")
                scalerX_path = os.path.join(output_dir, "scaler_X.pkl")
                print(f"[DEBUG] Buscando imputer en: {imputer_path}")
                print(f"[DEBUG] Buscando scaler_X en: {scalerX_path}")

                if not os.path.exists(imputer_path) or not os.path.exists(scalerX_path):
                    raise FileNotFoundError(f"No se encontraron imputer/scaler en {output_dir}")

                imputer_X = joblib.load(imputer_path)
                scaler_X = joblib.load(scalerX_path)
                x_input_imp = imputer_X.transform(x_input)
                print(f"[DEBUG] x_input_imp shape={x_input_imp.shape}, valores[0:5]={x_input_imp.flatten()[:5]}")
                x_input_norm = scaler_X.transform(x_input_imp)
                print(f"[DEBUG] x_input_norm shape={x_input_norm.shape}, valores[0:5]={x_input_norm.flatten()[:5]}")
                x_np = np.array(x_input_norm, dtype=np.float32)  # (1, n_in)
                print(f"[DEBUG] x_np final dtype={x_np.dtype}, shape={x_np.shape}")

                # ---------- Intento 1: mlp_det.pt -> forward NumPy ----------
                mlp_det_path = os.path.join(output_dir, "mlp_det.pt")
                print(f"[DEBUG] Buscando mlp_det.pt en: {mlp_det_path}")
                if os.path.exists(mlp_det_path):
                    try:
                        try:
                            sd = torch.load(mlp_det_path, map_location="cpu", weights_only=True)
                        except TypeError:
                            sd = torch.load(mlp_det_path, map_location="cpu")
                        print(f"[DEBUG] mlp_det.pt cargado, keys={list(sd.keys())[:10]}...")

                        if isinstance(sd, dict) and "fc1.weight" in sd:
                            print("[DEBUG] Usando mlp_det.pt (state_dict) -> forward NumPy")
                            fc1_w = to_numpy(sd["fc1.weight"])
                            fc1_b = to_numpy(sd["fc1.bias"])
                            out_w = to_numpy(sd["out.weight"])
                            out_b = to_numpy(sd["out.bias"])
                            print(f"[DEBUG] fc1_w shape={fc1_w.shape}, fc1_b shape={fc1_b.shape}")
                            print(f"[DEBUG] out_w shape={out_w.shape}, out_b shape={out_b.shape}")

                            y_norm = numpy_forward_two_layer(x_np, fc1_w, fc1_b, out_w, out_b)
                            print(f"[DEBUG] y_norm={y_norm}")

                            # DESNORMALIZAR
                            scalerY_path = os.path.join(output_dir, "scaler_Y.pkl")
                            print(f"[DEBUG] Buscando scaler_Y en: {scalerY_path}")
                            scaler_Y = joblib.load(scalerY_path)
                            if scaler_Y is not None:
                                try:
                                    y_real = scaler_Y.inverse_transform(np.array(y_norm).reshape(-1, 1))[0, 0]
                                except Exception as e:
                                    print(f"[DEBUG] Error al inverse_transform: {e}")
                                    y_real = y_norm
                            else:
                                y_real = y_norm

                            print("=== Resultado usando mlp_det.pt (NumPy forward) ===")
                            print(f"Pred_norm: {y_norm:.6f}  |  Pred_real: {y_real:.6f}")
                            raise SystemExit(0)
                        else:
                            print("[DEBUG] mlp_det.pt no tiene keys esperadas o no es state_dict -> continuar")
                    except Exception as e:
                        print("[DEBUG] Error usando mlp_det.pt (NumPy path):", e)


            # si es un mlp 
            else:    
                model_files = model_files_in(folder)
                # cargar imputer/scalers desde la carpeta
                imputer = load_optional(os.path.join(folder, "imputer_X.pkl"))
                scaler_X = load_optional(os.path.join(folder, "scaler_X.pkl"))
                scaler_Y = load_optional(os.path.join(folder, "scaler_Y.pkl"))
                x_proc = apply_imputer_and_scaler(x_input.astype(float), imputer, scaler_X)
                # cargar todos los modelos del ensemble
                modelos = [load_optional(os.path.join(folder, mf)) for mf in model_files]
                modelos = [m for m in modelos if m is not None]
                if not modelos:
                    continue
                preds = predict_with_models(modelos, x_proc)
                if not preds:
                    continue
                ensemble_norm = float(np.mean(preds))
                if scaler_Y is not None:
                    try:
                        ensemble_real = scaler_Y.inverse_transform(np.array(ensemble_norm).reshape(-1, 1))[0, 0]
                    except Exception:
                        ensemble_real = ensemble_norm
                else:
                    ensemble_real = ensemble_norm
                results[str(salida)] = ensemble_real
                used = src.get("used", "folder")
                print(f"-> {salida}: {ensemble_real:.6g}  (usado: {used})")

        elif src["type"] == "rootfile":
            name = src["name"]
            model_path = src["model_path"]
            model = load_optional(model_path)
            if model is None:
                continue
            # intentar scalers/imputer en raíz con sufijo _<name>
            imputer = load_optional(os.path.join(MODELS_DIR, f"imputer_X_{name}.pkl"))
            scaler_X = load_optional(os.path.join(MODELS_DIR, f"scaler_X_{name}.pkl"))
            scaler_Y = load_optional(os.path.join(MODELS_DIR, f"scaler_Y_{name}.pkl"))
            x_proc = apply_imputer_and_scaler(x_input.astype(float), imputer, scaler_X)
            preds = predict_with_models([model], x_proc)
            if not preds:
                continue
            val_norm = float(preds[0])
            if scaler_Y is not None:
                try:
                    val_real = scaler_Y.inverse_transform(np.array(val_norm).reshape(-1, 1))[0, 0]
                except Exception:
                    val_real = val_norm
            else:
                val_real = val_norm
            results[str(salida)] = val_real
            print(f"-> {salida}: {val_real:.6g}  (usado: mod_root)")

    if results:
        pred_df = pd.DataFrame(results, index=[f"Muestra {IDX_MUESTRA}"])
        print("\n======================================")
        print("Predicciones de TODAS las salidas")
        print("======================================")
        print(pred_df.T)
    else:
        print("\nNo se generaron predicciones válidas.")

if __name__ == "__main__":
    main()
