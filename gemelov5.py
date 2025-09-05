# -*- coding: utf-8 -*-
"""
predict_all.py (extendido, FIXED)
- Elimina cualquier dependencia en pyro/Predictive para evitar warnings.
- Soporta carpetas BNN con nombre: modelo_bnn_<salida> bajo gemelo/mod (MODELS_DIR).
- Loader robusto: forward NumPy desde mlp_det.pt / bnn_bundle.joblib.
"""
import os
import re
import warnings
import numpy as np
import pandas as pd
import joblib
import torch

warnings.filterwarnings("ignore")

# ---------- CONFIG ----------
try:
    SCRIPT_DIR = os.path.abspath(os.path.dirname(__file__))
except NameError:
    SCRIPT_DIR = os.getcwd()

DATA_DIR = os.path.join(SCRIPT_DIR, "data")
MODELS_DIR = os.path.join(SCRIPT_DIR, "mod")   # gemelo/mod
MODELS_DIR_ALT = os.path.join(SCRIPT_DIR, "mod1")

# Raíces donde buscar carpetas BNN del estilo modelo_bnn_<salida>
# (El usuario indicó que están en gemelo/mod/..., así que incluimos MODELS_DIR)
BNN_ROOTS = [MODELS_DIR]

INPUT_PATH = os.path.join(DATA_DIR, "input_example13.csv")  # cambia si hace falta
IDX_MUESTRA = 0
N_FEATURES_ESPERADOS = 22  # o None para no comprobar

# Firma de biomasa1: se compara contra X[5:14] (índices 0-based 5..13)
BIOMASA1_SIGNATURE = np.array([45.4, 6.73, 0.17, 0.0003, 0.0151, 72.0, 0.4, 10.6, 17.0], dtype=float)
BIOMASA1_SLICE = slice(5, 14)
BIOMASA_ATOL = 1e-6

# Archivo excel usado para recuperar nombres de salidas (mismo slicing que entrenamiento)
RUTA_EXCEL = os.path.join(SCRIPT_DIR, "datos SEG suministrados.xlsx")
# --------------------------------------------------------------------------------

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
    # además incluir carpetas BNN (modelo_bnn_<salida>) bajo BNN_ROOTS
    for root in BNN_ROOTS:
        if not os.path.isdir(root):
            continue
        for entry in os.listdir(root):
            if entry.startswith("modelo_bnn_"):
                name = entry[len("modelo_bnn_"):]
                outs.add(name)
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
    """Devuelve dict con info de la fuente para esa salida o None.
       Prioridad: carpeta BNN 'modelo_bnn_<salida>' (si existe) -> mod1 (si biomasa1) -> mod -> rootfile.
    """
    name = safe_name(salida)

    # 0) buscar carpeta BNN en BNN_ROOTS
    for broot in BNN_ROOTS:
        cand = os.path.join(broot, f"modelo_bnn_{name}")
        if os.path.isdir(cand):
            return {"type": "bnn_folder", "path": cand, "used": f"bnn:{broot}"}

    # 1) mod1 (si biomasa1)
    folder_alt = os.path.join(MODELS_DIR_ALT, name)
    folder_def = os.path.join(MODELS_DIR, name)
    if biomasa1_flag and model_files_in(folder_alt):
        return {"type": "folder", "path": folder_alt, "used": "mod1"}
    # 2) preferir mod folder
    if model_files_in(folder_def):
        return {"type": "folder", "path": folder_def, "used": "mod"}
    # 3) check root model file en MODELS_DIR: modelo_<name>.pkl (case-insensitive)
    candidates = []
    if os.path.isdir(MODELS_DIR):
        for f in os.listdir(MODELS_DIR):
            if re.fullmatch(rf"modelo[_-]{re.escape(name)}\.pkl", f, flags=re.IGNORECASE):
                candidates.append(os.path.join(MODELS_DIR, f))
    if candidates:
        return {"type": "rootfile", "model_path": candidates[0], "used": "mod_root", "name": name}
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

# ------------------ Funcionalidad BNN (forward NumPy / bundle) ------------------
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
    x_np = np.array(x_np)
    fc1_w = np.array(fc1_w)
    out_w = np.array(out_w)
    fc1_b = np.array(fc1_b)
    out_b = np.array(out_b)
    if fc1_b.ndim > 1 and fc1_b.shape[0] == 1:
        fc1_b = np.squeeze(fc1_b, axis=0)
    if out_b.ndim > 1 and out_b.shape[0] == 1:
        out_b = np.squeeze(out_b, axis=0)
    hidden = np.tanh(np.dot(x_np, fc1_w.T) + fc1_b.reshape(1, -1))
    # out
    if out_w.ndim == 2 and out_w.shape[0] == 1 and out_w.shape[1] == hidden.shape[1]:
        out = np.dot(hidden, out_w.T) + out_b.reshape(1, -1)
    elif out_w.ndim == 2 and out_w.shape[1] == 1 and out_w.shape[0] == hidden.shape[1]:
        out = np.dot(hidden, out_w) + out_b.reshape(1, -1)
    else:
        try:
            out = np.dot(hidden, out_w.T) + out_b.reshape(1, -1)
        except Exception as e:
            raise RuntimeError(
                f"Shapes incompatibles al calcular salida: hidden.shape={hidden.shape}, "
                f"out_w.shape={out_w.shape}, out_b.shape={out_b.shape}"
            ) from e
    return float(np.squeeze(out))

def desnormalize_single_with_scalerY(scaler_Y, y_norm, idx_salida_global):
    """Desnormaliza una sola salida aunque scaler_Y sea multisalida (MinMaxScaler típico)."""
    if scaler_Y is None:
        return float(y_norm)
    try:
        n_feats = getattr(scaler_Y, "n_features_in_", None)
        if n_feats is None and hasattr(scaler_Y, "scale_"):
            n_feats = scaler_Y.scale_.shape[0]
    except Exception:
        n_feats = None

    if n_feats is None or int(n_feats) == 1:
        try:
            return float(scaler_Y.inverse_transform(np.array(y_norm).reshape(-1, 1))[0, 0])
        except Exception:
            return float(y_norm)

    scale_arr = getattr(scaler_Y, "scale_", None)
    min_arr = getattr(scaler_Y, "min_", None)
    if scale_arr is not None and min_arr is not None:
        idx = int(idx_salida_global)
        scale_i = float(scale_arr[idx])
        min_i = float(min_arr[idx])
        # inverse = (scaled - min_) / scale_
        return float((y_norm - min_i) / scale_i)

    try:
        tmp = np.zeros((1, int(n_feats)), dtype=float)
        tmp[0, idx_salida_global] = y_norm
        return float(scaler_Y.inverse_transform(tmp)[0, idx_salida_global])
    except Exception:
        return float(y_norm)

def predict_bnn_folder(folder, x_input_np, idx_salida_global):
    """
    Predice desde carpeta BNN (modelo_bnn_<salida>):
      - intenta mlp_det.pt (state_dict -> forward NumPy)
      - si no, intenta bnn_bundle.joblib (W_means / W_inits -> forward NumPy)
      - devuelve (y_real, method) o (None, err)
    """
    # cargar imputers/scalers de entrada
    imputer = load_optional(os.path.join(folder, "imputer_X.pkl"))
    scaler_X = load_optional(os.path.join(folder, "scaler_X.pkl"))
    scaler_Y = load_optional(os.path.join(folder, "scaler_Y.pkl"))

    # aplicar imputador/scaler si existen
    x_proc = x_input_np.astype(float)
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

    # 1) mlp_det.pt (state_dict) -> forward NumPy
    mlp_det_path = os.path.join(folder, "mlp_det.pt")
    if os.path.exists(mlp_det_path):
        try:
            try:
                sd = torch.load(mlp_det_path, map_location="cpu", weights_only=True)
            except TypeError:
                sd = torch.load(mlp_det_path, map_location="cpu")
            if isinstance(sd, dict) and "fc1.weight" in sd:
                fc1_w = to_numpy(sd["fc1.weight"])
                fc1_b = to_numpy(sd["fc1.bias"])
                out_w = to_numpy(sd["out.weight"])
                out_b = to_numpy(sd["out.bias"])
                y_norm = numpy_forward_two_layer(x_proc, fc1_w, fc1_b, out_w, out_b)
                y_real = desnormalize_single_with_scalerY(scaler_Y, y_norm, idx_salida_global)
                return y_real, "mlp_det"
        except Exception as e:
            print("  Error usando mlp_det.pt (NumPy):", e)

    # 2) bnn_bundle.joblib -> W_means / W_inits
    bundle_path = os.path.join(folder, "bnn_bundle.joblib")
    if os.path.exists(bundle_path):
        try:
            bundle = joblib.load(bundle_path)
            W_means = bundle.get("W_means", None)
            b_means = bundle.get("b_means", None)
            W_inits = bundle.get("W_inits", None)
            b_inits = bundle.get("b_inits", None)

            if W_means is not None:
                # dict style
                if isinstance(W_means, dict) and "fc1.weight" in W_means and "out.weight" in W_means:
                    fc1_w = normalize_to_ndim(W_means["fc1.weight"], 2)
                    out_w = normalize_to_ndim(W_means["out.weight"], 2)
                    fc1_b = safe_squeeze1(b_means.get("fc1.bias", np.zeros(fc1_w.shape[0])))
                    out_b = safe_squeeze1(b_means.get("out.bias", np.zeros(out_w.shape[0])))
                    y_norm = numpy_forward_two_layer(x_proc, fc1_w, fc1_b, out_w, out_b)
                    y_real = desnormalize_single_with_scalerY(scaler_Y, y_norm, idx_salida_global)
                    return y_real, "bundle.W_means_dict"
                # list style (desde_mlp)
                if isinstance(W_means, list) and len(W_means) >= 2:
                    fc1_w = normalize_to_ndim(W_means[0], 2)
                    out_w = normalize_to_ndim(W_means[-1], 2)
                    fc1_b = safe_squeeze1(b_means[0]) if b_means is not None else np.zeros(fc1_w.shape[0])
                    out_b = safe_squeeze1(b_means[-1]) if b_means is not None else np.zeros(out_w.shape[0])
                    y_norm = numpy_forward_two_layer(x_proc, fc1_w, fc1_b, out_w, out_b)
                    y_real = desnormalize_single_with_scalerY(scaler_Y, y_norm, idx_salida_global)
                    return y_real, "bundle.W_means_list"

            # fallback a W_inits si W_means no está
            if W_inits is not None:
                fc1_w = normalize_to_ndim(W_inits[0], 2)
                out_w = normalize_to_ndim(W_inits[-1], 2)
                fc1_b = safe_squeeze1(b_inits[0]) if b_inits is not None else np.zeros(fc1_w.shape[0])
                out_b = safe_squeeze1(b_inits[-1]) if b_inits is not None else np.zeros(out_w.shape[0])
                y_norm = numpy_forward_two_layer(x_proc, fc1_w, fc1_b, out_w, out_b)
                y_real = desnormalize_single_with_scalerY(scaler_Y, y_norm, idx_salida_global)
                return y_real, "bundle.W_inits"
        except Exception as e:
            print("  Warning procesando bnn_bundle:", e)

    return None, "no_bnn_method"

# ------------------ MAIN ------------------
def main():
    X = read_input(INPUT_PATH)
    n_samples, n_features = X.shape

    if N_FEATURES_ESPERADOS is not None and n_features != N_FEATURES_ESPERADOS:
        pass  # se permite seguir adelante

    if IDX_MUESTRA < 0 or IDX_MUESTRA >= n_samples:
        raise IndexError(f"idx_muestra={IDX_MUESTRA} fuera de rango (n_muestras={n_samples})")

    x_input = X[IDX_MUESTRA].reshape(1, -1)
    x_row = x_input.flatten()

    biomasa1_flag = is_biomasa1(x_row)

    # obtener lista de salidas (incluye también las carpetas bnn)
    outputs = list_outputs_union(MODELS_DIR, MODELS_DIR_ALT)

    if not outputs:
        root_used = MODELS_DIR_ALT if biomasa1_flag else MODELS_DIR
        raise FileNotFoundError(f"No se encontraron subcarpetas ni modelos en '{root_used}'")

    X_cols = [f"Input_{i+1}" for i in range(n_features)]
    print("\n=== Datos de entrada usados ===")
    print(pd.DataFrame(x_input, columns=X_cols, index=[f"Muestra {IDX_MUESTRA}"]))
    print(f"\nBiomasa1 detectada: {biomasa1_flag}\n")

    # cargar output_names desde excel para mapear índice de salida (necesario para desnormalizar correctamente)
    output_names = None
    try:
        if os.path.exists(RUTA_EXCEL):
            df_ex = pd.read_excel(RUTA_EXCEL, sheet_name="Entrada y salida gasificador", header=None)
            valid_rows = (
                list(range(3, 6)) + list(range(7, 9)) + list(range(11, 16)) +
                list(range(17, 21)) + list(range(23, 27)) + list(range(28, 32)) +
                list(range(35, 55)) + list(range(56, 60)) + [61, 63] +
                list(range(65, 88)) + list(range(89, 92))
            )
            valid_df = df_ex.iloc[valid_rows]
            df_t = valid_df.transpose()
            input_array = df_t.to_numpy()
            column_names = input_array[0, :]
            output_names = column_names[22:75]
        else:
            output_names = []
    except Exception:
        output_names = []

    results = {}

    for salida in outputs:
        src = find_model_source_for_output(salida, biomasa1_flag)
        if src is None:
            continue

        # si es carpeta BNN
        if src["type"] == "bnn_folder":
            folder = src["path"]
            # determinar idx_salida_global (necesario para desnormalizar)
            idx_salida_global = 0
            if output_names is not None and len(output_names) > 0:
                try:
                    idx_salida_global = int(list(output_names).index(salida))
                except Exception:
                    idx_salida_global = 0
            # predecir
            try:
                y_real, method = predict_bnn_folder(folder, x_input.astype(float), idx_salida_global)
                if y_real is not None:
                    results[str(salida)] = float(y_real)
                    print(f"-> {salida}: {y_real:.6g}  (usado: {method})")
                else:
                    print(f"-> {salida}: no se obtuvo predicción desde BNN (folder {folder})")
            except Exception as e:
                print(f"-> {salida}: error prediciendo BNN: {e}")

        elif src["type"] == "folder":
            folder = src["path"]
            model_files = model_files_in(folder)
            if not model_files:
                continue
            imputer = load_optional(os.path.join(folder, "imputer_X.pkl"))
            scaler_X = load_optional(os.path.join(folder, "scaler_X.pkl"))
            scaler_Y = load_optional(os.path.join(folder, "scaler_Y.pkl"))
            x_proc = apply_imputer_and_scaler(x_input.astype(float), imputer, scaler_X)
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
