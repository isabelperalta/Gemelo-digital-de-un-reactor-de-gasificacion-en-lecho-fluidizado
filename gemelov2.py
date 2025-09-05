# -*- coding: utf-8 -*-
"""
Created on Mon Sep  1 14:25:18 2025
@author: isa

Versión para cuando este script está UBICADO DENTRO de la carpeta 'gemelo'.
Rutas por defecto: ./data (archivo de entrada) y ./mod (subcarpetas con modelos).
El archivo de entrada debe contener SOLO las entradas (CSV o XLSX).
"""

import pandas as pd
import numpy as np
import joblib
import os
import re
import sys

# === CONFIGURACIÓN ===
# Si pones este script dentro de la carpeta 'gemelo', el script usará su carpeta actual
# como gemelo_root. Si lo ejecutas interactivo y __file__ no existe, cae en cwd.
try:
    script_dir = os.path.abspath(os.path.dirname(__file__))
except NameError:
    script_dir = os.getcwd()

# IMPORTANTE: asumimos que este archivo está dentro de la carpeta 'gemelo'
gemelo_root = script_dir
data_dir = os.path.join(gemelo_root, "data")
modelos_root = os.path.join(gemelo_root, "mod")

# Archivo de entrada (CSV o XLSX) que debe contener SOLO las entradas (filas = muestras)
ruta_datos = os.path.join(data_dir, "input_example.csv")   # <- cámbialo si quieres otro nombre

# índice de la muestra a predecir (0-based)
idx_muestra = 0

# Número esperado de features (opcional, se avisa si difiere)
N_FEATURES_ESPERADOS = 22  # pon None si no quieres comprobar

# =====================

def safe_name(name):
    return re.sub(r'[<>:"/\\|?*]', '_', str(name))

def list_model_output_dirs(root):
    """Devuelve lista de subcarpetas bajo `root` que parecen contener modelos (mlp_model_*.pkl o scaler_Y.pkl)."""
    if not os.path.isdir(root):
        return []
    subdirs = []
    for entry in sorted(os.listdir(root)):
        p = os.path.join(root, entry)
        if os.path.isdir(p):
            files = os.listdir(p)
            has_model = any(f.startswith("mlp_model_") and f.endswith(".pkl") for f in files)
            has_scalerY = any(f == "scaler_Y.pkl" for f in files)
            if has_model or has_scalerY:
                subdirs.append(entry)
    return subdirs

# Comprobaciones básicas de rutas (solo avisos; puedes crear las carpetas si no existen)
if not os.path.isdir(gemelo_root):
    print(f"ERROR: no se encuentra la carpeta gemelo (esperada en: {gemelo_root})", file=sys.stderr)
    raise FileNotFoundError(gemelo_root)

if not os.path.isdir(data_dir):
    print(f"Nota: no existe '{data_dir}'. Crea 'gemelo/data' y coloca tu archivo de entrada allí.", file=sys.stderr)

if not os.path.isdir(modelos_root):
    print(f"Nota: no existe '{modelos_root}'. Crea 'gemelo/mod' con subcarpetas por salida (cada subcarpeta debe contener los .pkl).", file=sys.stderr)

# === CARGA DEL ARCHIVO (solo entradas) ===
if not os.path.exists(ruta_datos):
    print(f"ERROR: no se encuentra el archivo de entrada: {ruta_datos}", file=sys.stderr)
    raise FileNotFoundError(ruta_datos)

ext = os.path.splitext(ruta_datos)[1].lower()

try:
    if ext in (".xlsx", ".xls"):
        df_in = pd.read_excel(ruta_datos, header=None)
    elif ext == ".csv":
        df_in = pd.read_csv(ruta_datos, header=None)
    else:
        raise ValueError("Formato no soportado. Usa .csv, .xlsx o .xls")
except Exception as e:
    print(f"Error leyendo '{ruta_datos}': {e}", file=sys.stderr)
    raise

# Asegurarnos de que tenemos un array 2D (n_samples, n_features)
X = df_in.to_numpy()
if X.ndim == 1:
    X = X.reshape(1, -1)

n_samples, n_features = X.shape

if N_FEATURES_ESPERADOS is not None and n_features != N_FEATURES_ESPERADOS:
    print(f"Advertencia: número de features leídos = {n_features}, esperados = {N_FEATURES_ESPERADOS}.", file=sys.stderr)

# Nombres de columnas generados (genéricos)
X_cols = [f"Input_{i+1}" for i in range(n_features)]

# Validar idx_muestra
if idx_muestra < 0 or idx_muestra >= n_samples:
    raise IndexError(f"idx_muestra={idx_muestra} fuera de rango (n_muestras={n_samples})")

x_input = X[idx_muestra].reshape(1, -1)

entrada_df = pd.DataFrame(x_input, columns=X_cols, index=[f"Muestra {idx_muestra}"])
print("\n=== Datos de entrada usados ===")
print(entrada_df)
print("==============================\n")

# === Listado de salidas (subcarpetas en ./mod) ===
Y_cols = list_model_output_dirs(modelos_root)
if not Y_cols:
    print(f"ERROR: no se han encontrado subcarpetas de modelos en '{modelos_root}'.", file=sys.stderr)
    raise FileNotFoundError(f"No hay modelos en {modelos_root}")

print(f"Usando directorio de modelos: {modelos_root}")
print(f"Salidas detectadas (subcarpetas): {Y_cols}\n")

# === PREDICCIONES PARA TODAS LAS SALIDAS DISPONIBLES (Y_cols) ===
predicciones_finales = {}

for salida_objetivo in Y_cols:
    try:
        folder_name = safe_name(salida_objetivo)
        output_dir = os.path.join(modelos_root, folder_name)
        if not os.path.exists(output_dir):
            print(f"⚠️ No se encuentra carpeta para la salida '{salida_objetivo}' (esperada en {output_dir})")
            continue

        model_files = [f for f in os.listdir(output_dir) if f.startswith("mlp_model_") and f.endswith(".pkl")]
        if not model_files:
            print(f"⚠️ No hay modelos (mlp_model_*.pkl) para la salida '{salida_objetivo}' en {output_dir}")
            continue

        # Inicializar por salida
        imputer = None
        scaler_X = None
        scaler_Y = None

        imputer_path = os.path.join(output_dir, "imputer_X.pkl")
        scalerX_path = os.path.join(output_dir, "scaler_X.pkl")
        scalerY_path = os.path.join(output_dir, "scaler_Y.pkl")

        if os.path.exists(imputer_path):
            try:
                imputer = joblib.load(imputer_path)
            except Exception as e:
                print(f"⚠️ Error cargando imputer en {imputer_path}: {e}")

        if os.path.exists(scalerX_path):
            try:
                scaler_X = joblib.load(scalerX_path)
            except Exception as e:
                print(f"⚠️ Error cargando scaler_X en {scalerX_path}: {e}")

        if os.path.exists(scalerY_path):
            try:
                scaler_Y = joblib.load(scalerY_path)
            except Exception as e:
                print(f"⚠️ Error cargando scaler_Y en {scalerY_path}: {e}")

        # Preparar x_input para los modelos: imputar y escalar si es posible
        x_proc = x_input.copy().astype(float)

        if imputer is not None:
            try:
                x_proc = imputer.transform(x_proc)
            except Exception as e:
                print(f"⚠️ Error aplicando imputer para '{salida_objetivo}': {e}")

        if scaler_X is not None:
            try:
                x_proc = scaler_X.transform(x_proc)
            except Exception as e:
                print(f"⚠️ Error aplicando scaler_X para '{salida_objetivo}': {e}")

        # Cargar todos los modelos y predecir
        model_files.sort()
        modelos = []
        for f in model_files:
            try:
                modelos.append(joblib.load(os.path.join(output_dir, f)))
            except Exception as e:
                print(f"⚠️ Error cargando modelo {f} en {output_dir}: {e}")

        pred_norm_list = []
        for m in modelos:
            try:
                p = m.predict(x_proc)
                if isinstance(p, (list, np.ndarray)):
                    p_val = float(np.array(p).reshape(-1)[0])
                else:
                    p_val = float(p)
                pred_norm_list.append(p_val)
            except Exception as e:
                print(f"⚠️ Error prediciendo con modelo {getattr(m, '__class__', m)} para '{salida_objetivo}': {e}")

        if not pred_norm_list:
            print(f"⚠️ Ninguna predicción válida para '{salida_objetivo}'")
            continue

        ensemble_pred_norm = float(np.mean(pred_norm_list))

        # Desnormalizar si tenemos scaler_Y
        if scaler_Y is not None:
            try:
                ensemble_pred_real = scaler_Y.inverse_transform(np.array(ensemble_pred_norm).reshape(-1, 1))[0, 0]
            except Exception as e:
                print(f"⚠️ Error desnormalizando para '{salida_objetivo}': {e}")
                ensemble_pred_real = ensemble_pred_norm
        else:
            ensemble_pred_real = ensemble_pred_norm

        predicciones_finales[str(salida_objetivo)] = ensemble_pred_real
        print(f"-> Salida: {salida_objetivo} | Ensemble (norm): {ensemble_pred_norm:.6g} | Ensemble (real): {ensemble_pred_real:.6g}")

    except Exception as e:
        print(f"❌ Error con la salida {salida_objetivo}: {e}")

# === MOSTRAR TABLA DE RESULTADOS ===
if predicciones_finales:
    pred_df = pd.DataFrame(predicciones_finales, index=[f"Muestra {idx_muestra}"])
    print("\n======================================")
    print("Predicciones de TODAS las salidas")
    print("======================================")
    print(pred_df.T)
else:
    print("\nNo se generaron predicciones válidas.")
