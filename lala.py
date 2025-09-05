
# diag_repair_models.py
# Diagnóstico y reparación (fallback) para modelos joblib/pickle/LightGBM
import os, sys, traceback, joblib, pickle, struct
from pathlib import Path

def safe_import(name):
    try:
        m = __import__(name)
        return m
    except Exception:
        return None

np = safe_import("numpy")
sklearn = safe_import("sklearn")
lgb = safe_import("lightgbm")
torch = safe_import("torch")
cloudpickle = safe_import("cloudpickle")
dill = safe_import("dill")

ROOT = Path(__file__).resolve().parent
MODELS_DIR = ROOT / "mod"  # ajusta si guardas en otra carpeta
if not MODELS_DIR.exists():
    print(f"[WARN] No existe {MODELS_DIR}. Ajusta MODELS_DIR en el script si usas otra carpeta.")
    sys.exit(0)

# imprime versiones
print("=== Entorno / versiones ===")
print("Python:", sys.version.splitlines()[0])
for mod, name in [(joblib, "joblib"), (np, "numpy"), (sklearn, "sklearn"), (lgb, "lightgbm"), (torch, "torch"), (cloudpickle, "cloudpickle"), (dill, "dill")]:
    try:
        if mod is None:
            print(f"{name}: NO INSTALADO")
        else:
            v = getattr(mod, "__version__", None) or getattr(mod, "version", None) or str(mod)
            print(f"{name}: {v}")
    except Exception:
        print(f"{name}: (error leyendo versión)")

print("\nBuscando archivos modelo en:", MODELS_DIR)
candidates = sorted([p for p in MODELS_DIR.glob("modelo_*.pkl")])
if not candidates:
    print("No se encontraron modelo_*.pkl en", MODELS_DIR)
    sys.exit(0)

def hexdump_head(path, n=512):
    with open(path, "rb") as f:
        b = f.read(n)
    try:
        txt = b.decode("utf-8", errors="replace")
    except Exception:
        txt = str(b[:80])
    hexstr = " ".join(f"{c:02x}" for c in b[:64])
    return txt, hexstr, len(b)

def try_load_joblib(p):
    try:
        return joblib.load(p), None
    except Exception as e:
        return None, e

def try_load_pickle(p):
    try:
        with open(p, "rb") as f:
            return pickle.load(f), None
    except Exception as e:
        return None, e

def try_load_cloudpickle(p):
    if cloudpickle is None:
        return None, RuntimeError("cloudpickle no instalado")
    try:
        with open(p, "rb") as f:
            return cloudpickle.load(f), None
    except Exception as e:
        return None, e

def try_load_torch(p):
    if torch is None:
        return None, RuntimeError("torch no instalado")
    try:
        return torch.load(str(p), map_location="cpu"), None
    except Exception as e:
        return None, e

def create_wrapper_from_booster(txt_path, out_pkl):
    """Carga booster .txt y crea wrapper con método predict usando lightgbm.Booster."""
    if lgb is None:
        raise RuntimeError("lightgbm no instalado; no puedo cargar .txt")
    booster = None
    try:
        booster = lgb.Booster(model_file=str(txt_path))
    except Exception as e:
        raise RuntimeError(f"Error cargando Booster desde {txt_path}: {e}")
    # wrapper simple
    class BoosterWrapper:
        def __init__(self, booster):
            self._booster = booster
            # intentar inferir n_features si disponible
            try:
                self.n_features_in_ = int(booster.num_feature())
            except Exception:
                self.n_features_in_ = None
        def predict(self, X):
            import numpy as _np
            arr = _np.array(X, dtype=float)
            # booster.predict espera 2D
            return booster.predict(arr)
    wrapper = BoosterWrapper(booster)
    # guardar wrapper con joblib
    joblib.dump(wrapper, str(out_pkl))
    return wrapper

for p in candidates:
    print("\n--- Revisando:", p.name, "---")
    txt, hexhdr, size = hexdump_head(p, n=1024)
    print(f"tamaño: {size} bytes")
    print("primeros bytes (texto):")
    print(txt[:400])
    print("primeros bytes (hex):", hexhdr)
    # intentos de carga
    loaded = None
    errs = {}
    for fn in (try_load_joblib, try_load_pickle, try_load_cloudpickle, try_load_torch):
        try:
            obj, err = fn(p)
            if err is None and obj is not None:
                print(f"[OK] CARGADO con {fn.__name__}")
                print("  tipo objeto:", type(obj))
                # verifica si tiene predict
                has_pred = hasattr(obj, "predict")
                print("  tiene predict?:", has_pred)
                loaded = obj
                break
            else:
                errs[fn.__name__] = repr(err)
                print(f"[FAIL] {fn.__name__}: {type(err).__name__}: {err}")
        except Exception as e:
            errs[fn.__name__] = repr(e)
            print(f"[ERROR] {fn.__name__} raised: {type(e).__name__}: {e}")
    if loaded is not None:
        continue  # siguiente fichero

    # si no se pudo cargar, busca .txt de booster como fallback
    name = p.stem  # modelo_<name>
    txt_candidate = MODELS_DIR / (p.stem + ".txt")  # mismo prefijo .txt
    if txt_candidate.exists():
        print("[INFO] Encontrado fallback booster txt:", txt_candidate.name)
        out_wrapped = MODELS_DIR / (p.stem + "_wrapped.pkl")
        try:
            wrapper = create_wrapper_from_booster(txt_candidate, out_wrapped)
            print(f"[OK] Wrapper creado y guardado como {out_wrapped.name}. Puedes usar ese pkl en predict_all.py")
            continue
        except Exception as e:
            print(f"[ERROR] No se pudo crear wrapper desde booster txt: {type(e).__name__}: {e}")
            traceback.print_exc()
    else:
        print("[INFO] No se encontró archivo .txt fallback para este modelo.")

    # imprimir trazas de error recopiladas
    print("\nTrazas de fallo resumidas:")
    for k,v in errs.items():
        print(f" - {k}: {v}")
    # imprimir traza completa de joblib.load para que la pegues aquí si quieres
    try:
        print("\nIntentando joblib.load para imprimir traceback completo...")
        joblib.load(p)
    except Exception as e:
        print("Traceback joblib.load:")
        traceback.print_exc()

print("\nFIN diagnóstico. Si quieres, pega aquí la salida de este script (completa) y te indico siguiente paso.")
