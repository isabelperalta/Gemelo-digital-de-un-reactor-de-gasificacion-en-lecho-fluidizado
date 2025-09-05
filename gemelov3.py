# -*- coding: utf-8 -*-
"""
predict_all.py
Si la entrada coincide con la firma de biomasa 1, intenta usar modelos en ./mod1;
para cada salida, si mod1 no tiene modelo válido, hace fallback a ./mod.
Si la entrada no es biomasa1, usa solo ./mod.
El script debe estar dentro de 'gemelo' y el archivo de entrada (CSV/XLSX)
debe contener SOLO las entradas (filas = muestras).
"""

import os
import re
import warnings
import numpy as np
import pandas as pd
import joblib

warnings.filterwarnings("ignore")

# ---------- CONFIG ----------
try:
    SCRIPT_DIR = os.path.abspath(os.path.dirname(__file__))
except NameError:
    SCRIPT_DIR = os.getcwd()

DATA_DIR = os.path.join(SCRIPT_DIR, "data")
MODELS_DIR = os.path.join(SCRIPT_DIR, "mod")
MODELS_DIR_ALT = os.path.join(SCRIPT_DIR, "mod1")

INPUT_PATH = os.path.join(DATA_DIR, "input_example0.csv")  # cambiar si hace falta
IDX_MUESTRA = 0
N_FEATURES_ESPERADOS = 22  # o None para no comprobar

# Firma de biomasa1: se compara contra X[5:14] (índices 0-based 5..13)
BIOMASA1_SIGNATURE = np.array([45.4, 6.73, 0.17, 0.0003, 0.0151, 72.0, 0.4, 10.6, 17.0], dtype=float)
BIOMASA1_SLICE = slice(5, 14)
BIOMASA_ATOL = 1e-6
# ----------------------------

def safe_name(name: str) -> str:
    return re.sub(r'[<>:"/\\|?*]', "_", str(name))

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
                # consideramos salida si hay algún mlp_model_*.pkl o scaler_Y.pkl
                files = os.listdir(p)
                if any(f.startswith("mlp_model_") and f.endswith(".pkl") for f in files) or "scaler_Y.pkl" in files:
                    outs.add(entry)
    return sorted(outs)

def is_biomasa1(x_row: np.ndarray) -> bool:
    try:
        seg = np.array(x_row[BIOMASA1_SLICE], dtype=float).reshape(-1)
    except Exception:
        return False
    if seg.shape[0] != BIOMASA1_SIGNATURE.shape[0]:
        return False
    return bool(np.allclose(seg, BIOMASA1_SIGNATURE, atol=BIOMASA_ATOL, rtol=1e-8))

def main():
    X = read_input(INPUT_PATH)
    n_samples, n_features = X.shape

    if N_FEATURES_ESPERADOS is not None and n_features != N_FEATURES_ESPERADOS:
        # continuamos aunque no coincida
        pass

    if IDX_MUESTRA < 0 or IDX_MUESTRA >= n_samples:
        raise IndexError(f"idx_muestra={IDX_MUESTRA} fuera de rango (n_muestras={n_samples})")

    x_input = X[IDX_MUESTRA].reshape(1, -1)
    x_row = x_input.flatten()

    biomasa1_flag = is_biomasa1(x_row)
    # obtenemos la lista de posibles salidas como unión de carpetas en mod y mod1
    if biomasa1_flag:
        outputs = list_outputs_union(MODELS_DIR, MODELS_DIR_ALT)
    else:
        outputs = list_outputs_union(MODELS_DIR)

    if not outputs:
        root_used = MODELS_DIR_ALT if biomasa1_flag else MODELS_DIR
        raise FileNotFoundError(f"No se encontraron subcarpetas de modelos en '{root_used}'")

    X_cols = [f"Input_{i+1}" for i in range(n_features)]
    print("\n=== Datos de entrada usados ===")
    print(pd.DataFrame(x_input, columns=X_cols, index=[f"Muestra {IDX_MUESTRA}"]))
    print(f"\nBiomasa1 detectada: {biomasa1_flag}\n")

    predicciones = {}

    for salida in outputs:
        # prioridad: si biomasa1_flag -> probar mod1 primero para esta salida, luego fallback a mod
        folder_alt = os.path.join(MODELS_DIR_ALT, safe_name(salida))
        folder_def = os.path.join(MODELS_DIR, safe_name(salida))

        chosen_folder = None
        # si biomasa1, preferir mod1 pero solo si tiene modelos
        if biomasa1_flag and model_files_in(folder_alt):
            chosen_folder = folder_alt
        elif model_files_in(folder_def):
            chosen_folder = folder_def
        elif biomasa1_flag and model_files_in(folder_alt):
            # redundante por seguridad; no debería llegar aquí
            chosen_folder = folder_alt
        else:
            # no hay modelos en ninguno de los dos para esta salida -> skip
            continue

        # cargar recursos
        model_files = model_files_in(chosen_folder)
        if not model_files:
            continue

        imputer = load_optional(os.path.join(chosen_folder, "imputer_X.pkl"))
        scaler_X = load_optional(os.path.join(chosen_folder, "scaler_X.pkl"))
        scaler_Y = load_optional(os.path.join(chosen_folder, "scaler_Y.pkl"))

        x_proc = x_input.astype(float)
        if imputer is not None:
            try:
                x_proc = imputer.transform(x_proc)
            except Exception:
                pass
        if scaler_X is not None:
            try:
                x_proc = scaler_X.transform(x_proc)
            except Exception:
                pass

        modelos = []
        for mf in model_files:
            m = load_optional(os.path.join(chosen_folder, mf))
            if m is not None:
                modelos.append(m)
        if not modelos:
            continue

        pred_list = []
        for m in modelos:
            try:
                p = m.predict(x_proc)
                p_val = float(np.array(p).reshape(-1)[0])
                pred_list.append(p_val)
            except Exception:
                continue
        if not pred_list:
            continue

        ensemble_norm = float(np.mean(pred_list))
        if scaler_Y is not None:
            try:
                ensemble_real = scaler_Y.inverse_transform(np.array(ensemble_norm).reshape(-1, 1))[0, 0]
            except Exception:
                ensemble_real = ensemble_norm
        else:
            ensemble_real = ensemble_norm

        predicciones[str(salida)] = ensemble_real
        # impresión mínima indicando carpeta usada para cada salida
        used = "mod1" if chosen_folder.startswith(MODELS_DIR_ALT) else "mod"
        print(f"-> {salida}: {ensemble_real:.6g}  (usado: {used})")

    if predicciones:
        pred_df = pd.DataFrame(predicciones, index=[f"Muestra {IDX_MUESTRA}"])
        print("\n======================================")
        print("Predicciones de TODAS las salidas")
        print("======================================")
        print(pred_df.T)
    else:
        print("\nNo se generaron predicciones válidas.")

if __name__ == "__main__":
    main()
