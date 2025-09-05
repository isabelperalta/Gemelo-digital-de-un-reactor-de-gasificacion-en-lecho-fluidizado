# -*- coding: utf-8 -*-
"""
predict_all.py  -- versión corregida para detectar y ejecutar BNNs en carpetas
(igual que antes, arreglado el bug de 'Boolean value of Tensor...' y heurísticas
más robustas para interpretar mlp_det.pt). Además:
 - índice robusto de modelos que normaliza nombres (evita None por espacios/símbolos)
 - opción para forzar preds no negativas (CLAMP_NON_NEGATIVE)
"""
import os
import re
import warnings
import numpy as np
import pandas as pd
import joblib
import torch
import torch.nn as nn

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

INPUT_PATH = os.path.join(DATA_DIR, "input_example.csv")
IDX_MUESTRA = 0
N_FEATURES_ESPERADOS = 22

# Si quieres aplicar clamp a todas las salidas para que no sean negativas,
# pon True. Si prefieres no hacerlo globalmente, pon False.
CLAMP_NON_NEGATIVE = True

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
        try:
            return np.array(x)
        except Exception:
            return None

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

    if out_w.ndim == 2 and out_w.shape[0] == 1 and out_w.shape[1] == hidden.shape[1]:
        out = np.dot(hidden, out_w.T) + out_b.reshape(1, -1)
    elif out_w.ndim == 2 and out_w.shape[1] == 1 and out_w.shape[0] == hidden.shape[1]:
        out = np.dot(hidden, out_w) + out_b.reshape(1, -1)
    else:
        try:
            out = np.dot(hidden, out_w.T) + out_b.reshape(1, -1)
        except Exception as e:
            raise RuntimeError(f"Shapes incompatibles al calcular salida: hidden.shape={hidden.shape}, out_w.shape={out_w.shape}, out_b.shape={out_b.shape}") from e

    return float(np.squeeze(out))

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

def folder_has_models(folder: str) -> bool:
    if not os.path.isdir(folder):
        return False
    files = os.listdir(folder)
    if "mlp_det.pt" in files:
        return True
    if "scaler_Y.pkl" in files:
        return True
    for f in files:
        if f.startswith("mlp_model_") and f.endswith(".pkl"):
            return True
    return False

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
                if folder_has_models(p):
                    outs.add(entry)
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

# -----------------------------
# Normalización robusta de claves
# -----------------------------
def normkey_strict(s):
    """Normaliza un nombre a una 'clave' simple: minúsculas, solo a-z0-9."""
    s = str(s or "")
    s = s.lower()
    # reemplaza caracteres no ascii letra/dígito por vacío
    # esto permite que 'Contenido carbono fijo char [%]' y 'Contenido_carbono_fijo_char____'
    # coincidan en la misma clave.
    s = re.sub(r'[^a-z0-9]+', '', s, flags=re.UNICODE)
    return s

def build_model_index():
    """
    Construye un índice dict: key_normalizada -> { 'rootfile': (path,name), 'mod': path, 'mod1': path }
    """
    index = {}
    # rootfiles en MODELS_DIR
    if os.path.isdir(MODELS_DIR):
        for f in os.listdir(MODELS_DIR):
            m = re.match(r"modelo[_-](.+)\.pkl$", f, flags=re.IGNORECASE)
            if m:
                name = m.group(1)
                key = normkey_strict(name)
                index.setdefault(key, {})['rootfile'] = (os.path.join(MODELS_DIR, f), name)

    # carpetas en MODELS_DIR y MODELS_DIR_ALT
    for root, tag in [(MODELS_DIR, 'mod'), (MODELS_DIR_ALT, 'mod1')]:
        if os.path.isdir(root):
            for entry in os.listdir(root):
                p = os.path.join(root, entry)
                if os.path.isdir(p) and folder_has_models(p):
                    key = normkey_strict(entry)
                    index.setdefault(key, {})[tag] = p

    return index

# variable global que se rellenará en main()
MODEL_INDEX = None

def find_model_source_for_output(salida: str, biomasa1_flag: bool):
    """
    Busca usando MODEL_INDEX (clave normalizada). Prioridad:
     - mod1 (si biomasa1_flag)
     - mod
     - rootfile (modelo_X.pkl)
    """
    global MODEL_INDEX
    if MODEL_INDEX is None:
        # fallback: construir índice al vuelo (muy raro)
        MODEL_INDEX = build_model_index()

    key = normkey_strict(salida)
    info = MODEL_INDEX.get(key, {})

    # prioridad: carpeta mod1 si biomasa1_flag, luego mod, luego rootfile
    if biomasa1_flag and 'mod1' in info:
        return {"type": "folder", "path": info['mod1'], "used": "mod1"}
    if 'mod' in info:
        return {"type": "folder", "path": info['mod'], "used": "mod"}
    if 'rootfile' in info:
        model_path, name = info['rootfile']
        return {"type": "rootfile", "model_path": model_path, "used": "mod_root", "name": name}

    # retry heuristic: quitar underscores/guiones en key y reintentar
    alt_key = re.sub(r'[_\-]', '', key)
    if alt_key != key and alt_key in MODEL_INDEX:
        info2 = MODEL_INDEX[alt_key]
        if biomasa1_flag and 'mod1' in info2:
            return {"type": "folder", "path": info2['mod1'], "used": "mod1"}
        if 'mod' in info2:
            return {"type": "folder", "path": info2['mod'], "used": "mod"}
        if 'rootfile' in info2:
            model_path, name = info2['rootfile']
            return {"type": "rootfile", "model_path": model_path, "used": "mod_root", "name": name}

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

def _tensor_to_np_safe(v):
    """Convierte tensores/arrays a numpy sin evaluar su truth value."""
    if v is None:
        return None
    if torch.is_tensor(v):
        try:
            return v.detach().cpu().numpy()
        except Exception:
            return np.array(v)
    try:
        return np.array(v)
    except Exception:
        return None

def predict_bnn_from_folder(folder: str, x_input):
    imputer_path = os.path.join(folder, "imputer_X.pkl")
    scalerX_path = os.path.join(folder, "scaler_X.pkl")

    if not os.path.exists(imputer_path) or not os.path.exists(scalerX_path):
        print(f"[WARN] En {folder}: falta imputer_X.pkl o scaler_X.pkl -> se salta BNN.")
        return None, None

    try:
        imputer_X = joblib.load(imputer_path)
        scaler_X = joblib.load(scalerX_path)
    except Exception as e:
        print(f"[WARN] Error cargando imputer/scaler en {folder}: {e}")
        return None, None

    try:
        x_input_imp = imputer_X.transform(x_input)
        x_input_norm = scaler_X.transform(x_input_imp)
    except Exception as e:
        print(f"[WARN] Error aplicando imputer/scaler en {folder}: {e}")
        return None, None

    x_np = np.array(x_input_norm, dtype=np.float32)

    mlp_det_path = os.path.join(folder, "mlp_det.pt")
    if not os.path.exists(mlp_det_path):
        print(f"[DEBUG] En {folder}: no existe mlp_det.pt -> nada que hacer aquí.")
        return None, None

    try:
        sd = torch.load(mlp_det_path, map_location="cpu")
    except Exception as e:
        print(f"[DEBUG] Error al torch.load({mlp_det_path}): {e}")
        return None, None

    y_norm = None

    # 1) Si es dict: buscar keys por substring (evitar usar `or` con tensores)
    if isinstance(sd, dict):
        # debug: mostrar algunas keys
        try:
            keys_sample = list(sd.keys())[:20]
            print(f"[DEBUG] mlp_det.pt keys (muestra) en {folder}: {keys_sample} (total {len(sd.keys())})")
        except Exception:
            pass

        # buscar las keys que contengan 'fc1.weight', 'fc1.bias', 'out.weight', 'out.bias'
        def find_key_containing(d, substr):
            for k in d.keys():
                # convertir key a str segura (bytes -> decode, etc.)
                try:
                    ks = k.decode() if isinstance(k, (bytes, bytearray)) else str(k)
                except Exception:
                    ks = str(k)
                if substr in ks:
                    return k
            return None

        k_fc1_w = find_key_containing(sd, "fc1.weight") or find_key_containing(sd, "fc.weight") or find_key_containing(sd, "fc1.weight".encode())
        k_fc1_b = find_key_containing(sd, "fc1.bias") or find_key_containing(sd, "fc.bias") or find_key_containing(sd, "fc1.bias".encode())
        k_out_w = find_key_containing(sd, "out.weight") or find_key_containing(sd, "fc2.weight") or find_key_containing(sd, "out.weight".encode())
        k_out_b = find_key_containing(sd, "out.bias") or find_key_containing(sd, "fc2.bias") or find_key_containing(sd, "out.bias".encode())

        # si las keys están dentro de un nested state_dict (p.e. 'model.state_dict'), intentar extraerlo
        if (k_fc1_w is None or k_out_w is None) and "model" in sd and isinstance(sd["model"], dict):
            nested = sd["model"]
            k_fc1_w = k_fc1_w or find_key_containing(nested, "fc1.weight")
            k_fc1_b = k_fc1_b or find_key_containing(nested, "fc1.bias")
            k_out_w = k_out_w or find_key_containing(nested, "out.weight")
            k_out_b = k_out_b or find_key_containing(nested, "out.bias")
            source_dict = nested
        else:
            source_dict = sd

        if k_fc1_w and k_fc1_b and k_out_w and k_out_b:
            try:
                fc1_w = _tensor_to_np_safe(source_dict[k_fc1_w])
                fc1_b = _tensor_to_np_safe(source_dict[k_fc1_b])
                out_w = _tensor_to_np_safe(source_dict[k_out_w])
                out_b = _tensor_to_np_safe(source_dict[k_out_b])
                if any(v is None for v in (fc1_w, fc1_b, out_w, out_b)):
                    raise RuntimeError("Alguna de las matrices fc1/out recuperadas es None")
                y_norm = numpy_forward_two_layer(x_np, fc1_w, fc1_b, out_w, out_b)
            except Exception as e:
                print(f"[DEBUG] Error en forward numpy desde state_dict en {mlp_det_path}: {e}")
                y_norm = None
        else:
            # no se encontraron las keys esperadas en el dict
            print(f"[DEBUG] No se encontraron keys fc1/out completas en el dict de {mlp_det_path} (k_fc1_w={k_fc1_w}, k_out_w={k_out_w})")

    # 2) Si no obtuvimos y_norm, intentar si sd es un nn.Module o contiene un módulo en 'model'
    if y_norm is None:
        model_obj = None
        if isinstance(sd, nn.Module) or hasattr(sd, "forward"):
            model_obj = sd
        elif isinstance(sd, dict) and "model" in sd and (isinstance(sd["model"], nn.Module) or hasattr(sd["model"], "forward")):
            model_obj = sd["model"]
        elif isinstance(sd, dict) and "state_dict" in sd and isinstance(sd["state_dict"], dict):
            # muchas veces guardan {'state_dict': {...}} — en ese caso intentar construir un simple forward numérico
            nested = sd["state_dict"]
            # intentar las mismas heurísticas que arriba
            def find_in_nested(substr):
                for k in nested.keys():
                    try:
                        ks = k.decode() if isinstance(k, (bytes, bytearray)) else str(k)
                    except Exception:
                        ks = str(k)
                    if substr in ks:
                        return k
                return None
            k1 = find_in_nested("fc1.weight")
            k2 = find_in_nested("fc1.bias")
            k3 = find_in_nested("out.weight")
            k4 = find_in_nested("out.bias")
            if k1 and k2 and k3 and k4:
                try:
                    fc1_w = _tensor_to_np_safe(nested[k1])
                    fc1_b = _tensor_to_np_safe(nested[k2])
                    out_w = _tensor_to_np_safe(nested[k3])
                    out_b = _tensor_to_np_safe(nested[k4])
                    y_norm = numpy_forward_two_layer(x_np, fc1_w, fc1_b, out_w, out_b)
                except Exception as e:
                    print(f"[DEBUG] Error forward desde nested state_dict en {mlp_det_path}: {e}")
                    y_norm = None

        if model_obj is not None:
            try:
                model_obj.eval()
                with torch.no_grad():
                    x_t = torch.tensor(x_np, dtype=torch.float32)
                    out_t = model_obj(x_t)
                    out_np = to_numpy(out_t)
                    y_norm = float(np.array(out_np).reshape(-1)[0])
            except Exception as e:
                print(f"[DEBUG] Error al ejecutar model_obj.forward en {mlp_det_path}: {e}")
                y_norm = None

    if y_norm is None:
        print(f"[DEBUG] mlp_det.pt cargado pero no reconocible como state_dict con fc1/out ni como nn.Module en {mlp_det_path}.")
        return None, None

    scalerY_path = os.path.join(folder, "scaler_Y.pkl")
    scaler_Y = load_optional(scalerY_path)
    if scaler_Y is not None:
        try:
            y_real = scaler_Y.inverse_transform(np.array(y_norm).reshape(-1, 1))[0, 0]
        except Exception as e:
            print(f"[DEBUG] Error al inverse_transform con scaler_Y en {folder}: {e}")
            y_real = y_norm
    else:
        y_real = y_norm

    return y_real, y_norm

def main():
    global MODEL_INDEX
    X = read_input(INPUT_PATH)
    n_samples, n_features = X.shape

    if N_FEATURES_ESPERADOS is not None and n_features != N_FEATURES_ESPERADOS:
        print(f"[WARN] Número de features ({n_features}) no coincide con N_FEATURES_ESPERADOS ({N_FEATURES_ESPERADOS})")

    if IDX_MUESTRA < 0 or IDX_MUESTRA >= n_samples:
        raise IndexError(f"idx_muestra={IDX_MUESTRA} fuera de rango (n_muestras={n_samples})")

    # construir índice de modelos una vez (normaliza nombres)
    MODEL_INDEX = build_model_index()
    # debug: mostrar algunas claves detectadas
    try:
        keys_sample = list(MODEL_INDEX.keys())[:60]
        print(f"[DEBUG] model index keys (muestra {len(keys_sample)}): {keys_sample}")
    except Exception:
        pass

    x_input = X[IDX_MUESTRA].reshape(1, -1)
    x_row = x_input.flatten()

    biomasa1_flag = is_biomasa1(x_row)
    outputs = list_outputs_union(MODELS_DIR, MODELS_DIR_ALT) if biomasa1_flag else list_outputs_union(MODELS_DIR)

    print("\n[DEBUG] outputs detectadas:", outputs)
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
            print(f"[DEBUG] find_model_source_for_output devolvió None para salida '{salida}' -> saltando")
            continue

        if src["type"] == "folder":
            folder = src["path"]
            output_dir = os.path.basename(folder)

            if output_dir.startswith("modelo_bnn"):
                bnn_suffix = output_dir[len("modelo_bnn"):].lstrip("_")
                print(f"\n[DEBUG] Detectado BNN para salida '{bnn_suffix}' en carpeta {folder}")

                y_real, y_norm = predict_bnn_from_folder(folder, x_input.astype(float))
                if y_real is None:
                    print(f"[DEBUG] No se obtuvo predicción BNN válida para '{salida}' (carpeta {folder}).")
                    continue

                val = float(y_real)
                if CLAMP_NON_NEGATIVE:
                    val = max(0.0, val)
                results[str(salida)] = val
                used = src.get("used", "folder")
                print(f"-> {salida}: {val:.6g}  (usado: {used}, bnn)")

            else:
                model_files = model_files_in(folder)
                if not model_files:
                    print(f"[DEBUG] Carpeta {folder} no contiene mlp_model_*.pkl -> saltando")
                    continue
                imputer = load_optional(os.path.join(folder, "imputer_X.pkl"))
                scaler_X = load_optional(os.path.join(folder, "scaler_X.pkl"))
                scaler_Y = load_optional(os.path.join(folder, "scaler_Y.pkl"))
                x_proc = apply_imputer_and_scaler(x_input.astype(float), imputer, scaler_X)
                modelos = [load_optional(os.path.join(folder, mf)) for mf in model_files]
                modelos = [m for m in modelos if m is not None]
                if not modelos:
                    print(f"[DEBUG] Ningún modelo cargado en {folder} (ensemble).")
                    continue
                preds = predict_with_models(modelos, x_proc)
                if not preds:
                    print(f"[DEBUG] Ninguna predicción válida desde los modelos en {folder}.")
                    continue
                ensemble_norm = float(np.mean(preds))
                if scaler_Y is not None:
                    try:
                        ensemble_real = scaler_Y.inverse_transform(np.array(ensemble_norm).reshape(-1, 1))[0, 0]
                    except Exception:
                        ensemble_real = ensemble_norm
                else:
                    ensemble_real = ensemble_norm

                val = float(ensemble_real)
                if CLAMP_NON_NEGATIVE:
                    val = max(0.0, val)
                results[str(salida)] = val
                used = src.get("used", "folder")
                print(f"-> {salida}: {val:.6g}  (usado: {used})")

        elif src["type"] == "rootfile":
            name = src["name"]
            model_path = src["model_path"]
            model = load_optional(model_path)
            if model is None:
                print(f"[DEBUG] No se pudo cargar modelo root en {model_path}")
                continue
            imputer = load_optional(os.path.join(MODELS_DIR, f"imputer_X_{name}.pkl"))
            scaler_X = load_optional(os.path.join(MODELS_DIR, f"scaler_X_{name}.pkl"))
            scaler_Y = load_optional(os.path.join(MODELS_DIR, f"scaler_Y_{name}.pkl"))
            x_proc = apply_imputer_and_scaler(x_input.astype(float), imputer, scaler_X)
            preds = predict_with_models([model], x_proc)
            if not preds:
                print(f"[DEBUG] Modelo root en {model_path} no produjo predicción.")
                continue
            val_norm = float(preds[0])
            if scaler_Y is not None:
                try:
                    val_real = scaler_Y.inverse_transform(np.array(val_norm).reshape(-1, 1))[0, 0]
                except Exception:
                    val_real = val_norm
            else:
                val_real = val_norm

            val = float(val_real)
            if CLAMP_NON_NEGATIVE:
                val = max(0.0, val)
            results[str(salida)] = val
            print(f"-> {salida}: {val:.6g}  (usado: mod_root)")

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
