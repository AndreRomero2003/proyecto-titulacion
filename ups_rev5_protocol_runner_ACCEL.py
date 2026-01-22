# -*- coding: utf-8 -*-
"""
ups_rev4_protocol_runner.py  (ACTUALIZADO a Rev-5)

Cumple el protocolo experimental del manuscrito "Articulo-Montaleza_Romero-rev-5" (Rev-5):
- Dataset multimodal N=300; partición HPO/Validación/Hold-out balanceada por modalidad.
- Catálogo de 15 pipelines deterministas P00..P15.
- Espacio de hiperparámetros (Tabla 2): c, tile, r, eps, alpgpha, sigma, tau_u.
- Evaluación con 49 métricas: 30 primarias (núcleo decisorio) + 19 secundarias (auditoría/costo).
- Búsqueda híbrida: Grid -> GA (NSGA-II) -> BO (Optuna/TPE).
- Paralelización solo en HPO (CPU+GPU); despliegue CPU.

Actualización solicitada por el usuario:
- Se ELIMINAN métricas: IW_SSIM, MedIQA_NR, energia_J_por_imagen (no se computan ni aparecen).

Requisitos:
pip install numpy opencv-contrib-python scikit-image scipy pandas matplotlib psutil pydicom torch torchvision pyiqa optuna

Uso recomendado (ejemplo):
python ups_rev3_protocol_runner.py --data_root /ruta/dataset --out_dir ./runs_rev4 --device cuda --n_jobs 28

Estructura dataset soportada:
A) data_root/RX/* , data_root/MAMO/* , data_root/ECO/*  (PNG/JPG/TIF/DICOM)
o B) --manifest_csv con columnas: path, modality  (modality in {RX,MAMO,ECO})
"""
from __future__ import annotations

import os
import sys
import json
import math
import time
import hashlib

from collections import OrderedDict

# --- progreso (opcional)
try:
    from tqdm import tqdm
except Exception:
    tqdm = None
import random
import argparse
import itertools
import threading
import contextlib
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Any

import numpy as np
import pandas as pd

import psutil

# --- OpenCV
import cv2

# --- skimage/scipy
from skimage import exposure
from scipy import ndimage

# --- DICOM
try:
    import pydicom
except Exception:
    pydicom = None

# --- Torch / IQA
try:
    import torch
except Exception as e:
    torch = None

try:
    import pyiqa
except Exception:
    pyiqa = None

# --- BO
try:
    import optuna
except Exception:
    optuna = None


# =========================
# Repro + entorno
# =========================
def set_global_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    if torch is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)


def force_cv2_threads(n: int) -> None:
    # Para evitar oversubscription al paralelizar por imágenes.
    try:
        cv2.setNumThreads(n)
    except Exception:
        pass

def progress_iter(iterable, desc: str = '', total: int | None = None, enabled: bool = True):
    """
    Wrapper de progreso: usa tqdm si está disponible; si no, retorna el iterable tal cual.
    """
    if not enabled or tqdm is None:
        return iterable
    try:
        return tqdm(iterable, desc=desc, total=total, dynamic_ncols=True, leave=False)
    except Exception:
        return iterable



def sha256_file(path: str, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


# =========================
# Dataset / Splits
# =========================
SUPPORTED_EXT = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".dcm"}


def is_image_file(path: str) -> bool:
    ext = os.path.splitext(path)[1].lower()
    return ext in SUPPORTED_EXT


def infer_modality_from_path(path: str) -> str:
    p = path.lower()
    if "/rx/" in p or "\\rx\\" in p or "radiografia" in p or "xray" in p:
        return "RX"
    if "/mamo/" in p or "\\mamo\\" in p or "mammo" in p or "mamm" in p:
        return "MAMO"
    if "/eco/" in p or "\\eco\\" in p or "ultra" in p or "us_" in p:
        return "ECO"
    # fallback: ALL/UNK
    return "UNK"


def scan_dataset(data_root: str) -> pd.DataFrame:
    rows = []
    for root, _, files in os.walk(data_root):
        for fn in files:
            p = os.path.join(root, fn)
            if is_image_file(p):
                mod = infer_modality_from_path(p)
                rows.append({"path": p, "modality": mod})
    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError(f"No se encontraron imágenes en {data_root}")
    # Si hay UNK, se permite pero no se estratifica bien: el usuario debería usar manifest_csv.
    return df


def load_manifest_csv(manifest_csv: str) -> pd.DataFrame:
    df = pd.read_csv(manifest_csv)
    if "path" not in df.columns or "modality" not in df.columns:
        raise ValueError("manifest_csv debe tener columnas: path, modality")
    df["path"] = df["path"].astype(str)
    df["modality"] = df["modality"].astype(str).str.upper()
    return df


def make_splits_stratified(
    df: pd.DataFrame,
    seed: int,
    frac_hpo: float = 0.6,
    frac_val: float = 0.2,
    frac_hold: float = 0.2,
) -> Dict[str, List[int]]:
    """
    Partición tripartita balanceada por modalidad (Rev-4).
    """
    assert abs(frac_hpo + frac_val + frac_hold - 1.0) < 1e-9
    rng = np.random.RandomState(seed)

    idx_hpo, idx_val, idx_hold = [], [], []

    for mod, g in df.groupby("modality"):
        idx = g.index.to_numpy()
        rng.shuffle(idx)
        n = len(idx)
        n_h = int(round(n * frac_hpo))
        n_v = int(round(n * frac_val))
        n_o = n - n_h - n_v
        idx_hpo.extend(idx[:n_h].tolist())
        idx_val.extend(idx[n_h:n_h + n_v].tolist())
        idx_hold.extend(idx[n_h + n_v:].tolist())

    rng.shuffle(idx_hpo)
    rng.shuffle(idx_val)
    rng.shuffle(idx_hold)
    return {"HPO": idx_hpo, "VAL": idx_val, "HOLD": idx_hold}


# =========================
# IO + Normalización + ROI
# =========================
def read_image_any(path: str) -> np.ndarray:
    ext = os.path.splitext(path)[1].lower()
    if ext == ".dcm":
        if pydicom is None:
            raise RuntimeError("Para DICOM necesitas: pip install pydicom")
        ds = pydicom.dcmread(path)
        img = ds.pixel_array.astype(np.float32)
        # reescalado (si aplica)
        if hasattr(ds, "RescaleSlope") and hasattr(ds, "RescaleIntercept"):
            img = img * float(ds.RescaleSlope) + float(ds.RescaleIntercept)
        return img
    # imágenes estándar
    im = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    if im is None:
        im = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if im is None:
        raise RuntimeError(f"No pude leer: {path}")
    if im.ndim == 3:
        # convertir a gris
        im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    return im.astype(np.float32)


def normalize_percentiles(img: np.ndarray, p_low=1.0, p_high=99.0) -> np.ndarray:
    lo = np.percentile(img, p_low)
    hi = np.percentile(img, p_high)
    if hi <= lo + 1e-12:
        return np.zeros_like(img, dtype=np.float32)
    x = np.clip(img, lo, hi)
    x = (x - lo) / (hi - lo)
    return x.astype(np.float32)


def letterbox_to_square(x: np.ndarray, size: int = 512) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
    """
    Redimensiona manteniendo aspect ratio y rellena con 0 para size x size.
    Devuelve: imagen, (top, left, new_h, new_w).
    """
    h, w = x.shape[:2]
    scale = min(size / h, size / w)
    nh, nw = int(round(h * scale)), int(round(w * scale))
    resized = cv2.resize(x, (nw, nh), interpolation=cv2.INTER_AREA)
    out = np.zeros((size, size), dtype=np.float32)
    top = (size - nh) // 2
    left = (size - nw) // 2
    out[top:top + nh, left:left + nw] = resized
    return out, (top, left, nh, nw)


def roi_mask_from_image(x01: np.ndarray, modality: str) -> np.ndarray:
    """
    ROI sin etiquetas (Rev-4). Implementación robusta, pragmática:
    - RX/MAMO: umbral Otsu sobre x01, morfología, componente mayor.
    - ECO: umbral por no-cero / Otsu + componente mayor; fallback elipse central.
    """
    x = (x01 * 255.0).astype(np.uint8)

    def largest_cc(mask: np.ndarray) -> np.ndarray:
        n, lab, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
        if n <= 1:
            return mask.astype(np.uint8)
        areas = stats[1:, cv2.CC_STAT_AREA]
        k = 1 + int(np.argmax(areas))
        out = (lab == k).astype(np.uint8)
        return out

    if modality in ("RX", "MAMO"):
        _, th = cv2.threshold(x, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        th = cv2.medianBlur(th, 5)
        th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8), iterations=2)
        th = cv2.morphologyEx(th, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8), iterations=1)
        th = largest_cc(th > 0)
        return th.astype(np.uint8)

    if modality == "ECO":
        # fan/sector suele estar en no-cero
        nz = (x > 0).astype(np.uint8)
        if nz.mean() < 0.05:
            _, th = cv2.threshold(x, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            nz = (th > 0).astype(np.uint8)
        nz = cv2.morphologyEx(nz, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8), iterations=2)
        nz = largest_cc(nz)
        if nz.mean() < 0.05:
            h, w = x.shape
            mask = np.zeros((h, w), np.uint8)
            cv2.ellipse(mask, (w // 2, h // 2), (int(0.45 * w), int(0.45 * h)), 0, 0, 360, 255, -1)
            return (mask > 0).astype(np.uint8)
        return nz.astype(np.uint8)

    # UNK: ROI = todo
    return np.ones_like(x, dtype=np.uint8)


# =========================
# Operadores (C, G, U)
# =========================
def op_clahe(x01: np.ndarray, c: float, tile: Tuple[int, int]) -> np.ndarray:
    x8 = np.clip(x01 * 255.0, 0, 255).astype(np.uint8)
    clahe = cv2.createCLAHE(clipLimit=float(c), tileGridSize=(int(tile[0]), int(tile[1])))
    y8 = clahe.apply(x8)
    y01 = (y8.astype(np.float32) / 255.0).astype(np.float32)
    return y01


def op_guided_filter(x01: np.ndarray, r: int, eps: float) -> np.ndarray:
    # OpenCV ximgproc guidedFilter está en opencv-contrib-python
    if not hasattr(cv2, "ximgproc"):
        raise RuntimeError("Necesitas opencv-contrib-python para guidedFilter (cv2.ximgproc).")
    I = x01.astype(np.float32)
    # guía = la misma imagen (como se usa típicamente)
    y = cv2.ximgproc.guidedFilter(guide=I, src=I, radius=int(r), eps=float(eps), dDepth=-1)
    return np.clip(y, 0.0, 1.0).astype(np.float32)


def op_usm(x01: np.ndarray, alpha: float, sigma: float, tau_u: float) -> np.ndarray:
    blur = cv2.GaussianBlur(x01, ksize=(0, 0), sigmaX=float(sigma), sigmaY=float(sigma))
    m = x01 - blur
    mask = (np.abs(m) > float(tau_u)).astype(np.float32)
    y = x01 + float(alpha) * m * mask
    return np.clip(y, 0.0, 1.0).astype(np.float32)


PIPELINES: Dict[str, Tuple[str, ...]] = {
    "P00": tuple(),              # baseline
    "P01": ("C",),
    "P02": ("G",),
    "P03": ("U",),
    "P04": ("C", "G"),
    "P05": ("C", "U"),
    "P06": ("G", "C"),
    "P07": ("G", "U"),
    "P08": ("U", "C"),
    "P09": ("U", "G"),
    "P10": ("C", "G", "U"),
    "P11": ("C", "U", "G"),
    "P12": ("G", "C", "U"),
    "P13": ("G", "U", "C"),
    "P14": ("U", "C", "G"),
    "P15": ("U", "G", "C"),
}


# =========================
# Espacio θ (Tabla 2 Rev-4)
# =========================
GRID_SPACE = {
    "c": [1, 2, 3, 4, 6, 8, 12, 16],
    "tile": [(4, 4), (6, 6), (8, 8), (12, 12), (16, 16), (24, 24), (32, 32)],
    "r": [2, 3, 4, 6, 8, 12, 16, 24, 32],
    "eps": [1e-6, 3e-6, 1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2],
    "alpha": [0.1, 0.2, 0.3, 0.6, 1.0, 1.6, 2.0, 2.2],
    "sigma": [0.5, 0.8, 1.2, 1.8, 2.5, 3.0, 3.5],
    "tau_u": [0.0, 0.005, 0.01, 0.02, 0.05, 0.1],
}

DEFAULT_THETA = {
    "c": 3,
    "tile": (8, 8),
    "r": 8,
    "eps": 1e-4,
    "alpha": 1.0,
    "sigma": 1.2,
    "tau_u": 0.01,
}


@dataclass(frozen=True)
class Theta:
    c: float = DEFAULT_THETA["c"]
    tile: Tuple[int, int] = DEFAULT_THETA["tile"]
    r: int = DEFAULT_THETA["r"]
    eps: float = DEFAULT_THETA["eps"]
    alpha: float = DEFAULT_THETA["alpha"]
    sigma: float = DEFAULT_THETA["sigma"]
    tau_u: float = DEFAULT_THETA["tau_u"]

    def as_dict(self) -> Dict[str, Any]:
        return {
            "c": float(self.c),
            "tile": (int(self.tile[0]), int(self.tile[1])),
            "r": int(self.r),
            "eps": float(self.eps),
            "alpha": float(self.alpha),
            "sigma": float(self.sigma),
            "tau_u": float(self.tau_u),
        }


def apply_pipeline(x01: np.ndarray, pipeline_ops: Tuple[str, ...], theta: Theta) -> Tuple[np.ndarray, Dict[str, float]]:
    """
    Retorna (y01, tiempos_ms_por_etapa)
    """
    t = {}
    y = x01
    if not pipeline_ops:
        return y, {"t_total_ms": 0.0, "t_CLAHE_ms": 0.0, "t_GF_ms": 0.0, "t_USM_ms": 0.0}

    t0 = time.perf_counter()
    for op in pipeline_ops:
        if op == "C":
            a = time.perf_counter()
            y = op_clahe(y, theta.c, theta.tile)
            t["t_CLAHE_ms"] = 1000.0 * (time.perf_counter() - a)
        elif op == "G":
            a = time.perf_counter()
            y = op_guided_filter(y, theta.r, theta.eps)
            t["t_GF_ms"] = 1000.0 * (time.perf_counter() - a)
        elif op == "U":
            a = time.perf_counter()
            y = op_usm(y, theta.alpha, theta.sigma, theta.tau_u)
            t["t_USM_ms"] = 1000.0 * (time.perf_counter() - a)
        else:
            raise ValueError(f"Operador desconocido: {op}")
    t_total = 1000.0 * (time.perf_counter() - t0)
    t.setdefault("t_CLAHE_ms", 0.0)
    t.setdefault("t_GF_ms", 0.0)
    t.setdefault("t_USM_ms", 0.0)
    t["t_total_ms"] = t_total
    return y, t


# =========================
# Métricas (49 totales)
# =========================
# 30 primarias: 4 bloques (A,B,C,D) con dirección ↑/↓
PRIMARY_METRICS = [
    # A) Fidelidad global (FR)
    ("SSIM", "A", "up"),
    ("MS_SSIM", "A", "up"),
    ("FSIM", "A", "up"),
    ("VIF", "A", "up"),
    ("GMSD", "A", "down"),
    ("PSNR", "A", "up"),
    ("NMI", "A", "up"),

    # B) Bordes/estructura (global + ROI)
    ("SSIM_ROI", "B", "up"),
    ("MS_SSIM_ROI", "B", "up"),
    ("VIF_ROI", "B", "up"),
    ("EPI_global", "B", "up"),
    ("EPI_ROI", "B", "up"),
    ("CPBD", "B", "up"),
    ("TENEGRAD", "B", "up"),
    ("LAPLACIAN_VARIANCE", "B", "up"),

    # C) Visibilidad/ruido (global + ROI)
    ("CNR", "C", "up"),
    ("SNR", "C", "up"),
    ("AMBE", "C", "down"),
    ("ENTROPY", "C", "up"),
    ("MI", "C", "up"),
    ("ENL", "C", "up"),
    ("SpeckleIndex_ROI", "C", "down"),
    ("SpeckleSNR_ROI", "C", "up"),

    # D) NR-IQA + costo (guardrails)
    ("NIQE", "D", "down"),
    ("BRISQUE", "D", "down"),
    ("PIQE", "D", "down"),
    ("CLIP_IAQ", "D", "up"),
    ("NIMA", "D", "up"),
    ("t_total_ms_1MP", "D", "down"),
    ("peak_ram_MB", "D", "down"),
]

# 19 secundarias: auditoría, ROI perceptual, perfilado/costo
SECONDARY_METRICS = [
    ("DISTS", "S", "down"),
    ("LPIPS", "S", "down"),
    ("MSE", "S", "down"),
    ("RMSE", "S", "down"),
    ("FSIM_ROI", "S", "up"),
    ("MI_ROI", "S", "up"),
    ("NMI_ROI", "S", "up"),
    ("t_total_ms", "S", "down"),
    ("t_CLAHE_ms", "S", "down"),
    ("t_GF_ms", "S", "down"),
    ("t_USM_ms", "S", "down"),
    ("imgs_por_seg", "S", "up"),
    ("num_pasadas_imagen", "S", "down"),
    ("ops_aprox_por_pixel", "S", "down"),
    ("t_total_ms_4MP", "S", "down"),
    ("t_total_ms_16MP", "S", "down"),
    ("cpu_util_%", "S", "down"),
    ("user_time_s", "S", "down"),
    ("sys_time_s", "S", "down"),
]

# --- IMPORTANTE: métricas ELIMINADAS (no existen aquí):
# IW_SSIM, MedIQA_NR, energia_J_por_imagen


def _entropy_256(x01: np.ndarray) -> float:
    x8 = np.clip(x01 * 255.0, 0, 255).astype(np.uint8)
    hist = np.bincount(x8.flatten(), minlength=256).astype(np.float64)
    p = hist / (hist.sum() + 1e-12)
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum())


def _mse(a: np.ndarray, b: np.ndarray) -> float:
    d = (a - b).astype(np.float32)
    return float(np.mean(d * d))


def _rmse(a: np.ndarray, b: np.ndarray) -> float:
    return float(math.sqrt(_mse(a, b) + 1e-12))


def _psnr(a: np.ndarray, b: np.ndarray) -> float:
    mse = _mse(a, b)
    if mse <= 1e-12:
        return 99.0
    return float(10.0 * math.log10(1.0 / mse))


def _mi_nmi(a01: np.ndarray, b01: np.ndarray, bins: int = 64) -> Tuple[float, float]:
    a = np.clip(a01, 0, 1).ravel()
    b = np.clip(b01, 0, 1).ravel()
    ha, _ = np.histogram(a, bins=bins, range=(0, 1))
    hb, _ = np.histogram(b, bins=bins, range=(0, 1))
    hab, _, _ = np.histogram2d(a, b, bins=bins, range=((0, 1), (0, 1)))
    pa = ha / (ha.sum() + 1e-12)
    pb = hb / (hb.sum() + 1e-12)
    pab = hab / (hab.sum() + 1e-12)
    # entropías
    pa_nz = pa[pa > 0]
    pb_nz = pb[pb > 0]
    pab_nz = pab[pab > 0]
    Ha = float(-(pa_nz * np.log2(pa_nz)).sum())
    Hb = float(-(pb_nz * np.log2(pb_nz)).sum())
    Hab = float(-(pab_nz * np.log2(pab_nz)).sum())
    MI = Ha + Hb - Hab
    NMI = MI / (math.sqrt(Ha * Hb) + 1e-12)
    return float(MI), float(NMI)


def _edge_preservation_index(a01: np.ndarray, b01: np.ndarray, mask: Optional[np.ndarray] = None) -> float:
    ax = cv2.Sobel(a01, cv2.CV_32F, 1, 0, ksize=3)
    ay = cv2.Sobel(a01, cv2.CV_32F, 0, 1, ksize=3)
    bx = cv2.Sobel(b01, cv2.CV_32F, 1, 0, ksize=3)
    by = cv2.Sobel(b01, cv2.CV_32F, 0, 1, ksize=3)
    ga = np.sqrt(ax * ax + ay * ay)
    gb = np.sqrt(bx * bx + by * by)
    if mask is not None:
        m = (mask > 0).astype(np.float32)
        ga = ga * m
        gb = gb * m
    num = float(np.sum(ga * gb))
    den = float(np.sum(ga * ga) + 1e-12)
    return num / den


def _tenengrad(a01: np.ndarray, mask: Optional[np.ndarray] = None) -> float:
    gx = cv2.Sobel(a01, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(a01, cv2.CV_32F, 0, 1, ksize=3)
    g2 = gx * gx + gy * gy
    if mask is not None:
        g2 = g2 * (mask > 0).astype(np.float32)
    return float(np.mean(g2))


def _laplacian_variance(a01: np.ndarray, mask: Optional[np.ndarray] = None) -> float:
    lap = cv2.Laplacian(a01, cv2.CV_32F, ksize=3)
    if mask is not None:
        lap = lap * (mask > 0).astype(np.float32)
    return float(np.var(lap))


def _cnr_snr_enl_speckle(a01: np.ndarray, roi: np.ndarray) -> Dict[str, float]:
    m = roi > 0
    if m.sum() < 64:
        return {"CNR": np.nan, "SNR": np.nan, "ENL": np.nan, "SpeckleIndex_ROI": np.nan, "SpeckleSNR_ROI": np.nan}

    v = a01[m].astype(np.float32)
    mu = float(v.mean())
    sd = float(v.std() + 1e-12)
    snr = mu / sd

    enl = (mu * mu) / (sd * sd + 1e-12)
    speckle_index = sd / (mu + 1e-12)  # CoV
    speckle_snr = mu / sd

    # CNR sin etiquetas: contraste entre percentiles bajos vs altos dentro ROI
    p10 = np.percentile(v, 10.0)
    p90 = np.percentile(v, 90.0)
    low = v[v <= p10]
    high = v[v >= p90]
    mu0, s0 = float(low.mean()), float(low.std() + 1e-12)
    mu1, s1 = float(high.mean()), float(high.std() + 1e-12)
    cnr = abs(mu1 - mu0) / math.sqrt(s0 * s0 + s1 * s1 + 1e-12)

    return {
        "CNR": float(cnr),
        "SNR": float(snr),
        "ENL": float(enl),
        "SpeckleIndex_ROI": float(speckle_index),
        "SpeckleSNR_ROI": float(speckle_snr),
    }


def _cpbd_sharpness(a01: np.ndarray) -> float:
    """
    CPBD aproximado (implementación pragmática).
    No usa modelos, solo respuesta de borde y ancho de blur.
    """
    img = (np.clip(a01, 0, 1) * 255.0).astype(np.uint8)
    edges = cv2.Canny(img, 50, 150, L2gradient=True)
    ys, xs = np.where(edges > 0)
    if len(xs) < 100:
        return 0.0

    # gradiente para estimar blur en bordes
    gx = cv2.Sobel(a01, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(a01, cv2.CV_32F, 0, 1, ksize=3)
    g = np.sqrt(gx * gx + gy * gy) + 1e-12

    # "blur width" inversamente proporcional al gradiente (heurístico)
    bw = 1.0 / (g[ys, xs])
    # prob blur detect: P(bw < T) con T relativo
    T = np.percentile(bw, 80.0)
    p = float(np.mean(bw < T))
    return p


# =========================
# IQA en GPU (pyiqa)
# =========================

class IQAModels:
    """    IQA robusto (pyiqa) para Rev-5 con **inferencias batch** en GPU.

    Diferencias clave vs runner antiguo:
      - La inferencia pyiqa NO se ejecuta dentro de threads por imagen.
      - Se ejecuta en el hilo principal en batches (B,H,W) -> maximiza RTX 3090.

    Mantiene el mismo conjunto de métricas (FR + NR) que el runner Rev-5.
    Si una métrica falla, retorna NaN (sin colapsar el protocolo).
    """

    _IQA_LOCK = threading.Lock()

    def __init__(self, device: str):
        if torch is None or pyiqa is None:
            raise RuntimeError("Necesitas torch + pyiqa para IQA. Instala: pip install torch pyiqa")

        self.device = torch.device(device) if isinstance(device, str) else device

        def mk(names):
            last = None
            for n in names:
                try:
                    return pyiqa.create_metric(n, device=self.device, as_loss=False)
                except TypeError:
                    try:
                        return pyiqa.create_metric(n, device=self.device)
                    except Exception as e:
                        last = e
                except Exception as e:
                    last = e
            raise RuntimeError(f"No pude crear métrica pyiqa entre {names}. Último error: {last!r}")

        # FR
        self.ssim = mk(["ssim"])
        self.ms_ssim = mk(["ms_ssim"])
        self.fsim = mk(["fsim"])
        self.vif = mk(["vif", "vifp"])
        self.gmsd = mk(["gmsd"])
        self.dists = mk(["dists"])
        self.lpips = mk(["lpips"])

        # NR
        self.niqe = mk(["niqe"])
        self.brisque = mk(["brisque"])
        self.piqe = mk(["piqe"])
        self.clipiqa = mk(["clipiqa", "clipiqa+"])
        self.nima = mk(["nima"])

    @staticmethod
    def _ensure_float01_np(x: np.ndarray) -> np.ndarray:
        x = np.asarray(x)
        if x.dtype != np.float32:
            x = x.astype(np.float32, copy=False)
        mx = float(np.max(x)) if x.size else 1.0
        if mx > 1.5:
            x = x / (255.0 if mx <= 255.0 else 65535.0)
        x = np.nan_to_num(x, nan=0.0, posinf=1.0, neginf=0.0)
        return np.clip(x, 0.0, 1.0)

    @staticmethod
    def _to_rgb_batch(xs01: list[np.ndarray]) -> np.ndarray:
        # xs: lista de [H,W] o [H,W,1] -> [B,3,H,W] float32
        arr = []
        for x in xs01:
            x = np.asarray(x)
            if x.ndim == 2:
                x = x[..., None]
            if x.shape[2] == 1:
                x = np.repeat(x, 3, axis=2)
            elif x.shape[2] >= 3:
                x = x[:, :, :3]
            else:
                raise ValueError(f"Canales no soportados: {x.shape}")
            x = IQAModels._ensure_float01_np(x)
            x = np.transpose(x, (2, 0, 1))  # 3,H,W
            arr.append(x)
        return np.stack(arr, axis=0).astype(np.float32, copy=False)

    def _infer_ctx(self):
        inf = torch.inference_mode() if hasattr(torch, 'inference_mode') else torch.no_grad()
        if isinstance(self.device, torch.device) and self.device.type == 'cuda':
            amp_off = torch.cuda.amp.autocast(enabled=False)
            lock = self._IQA_LOCK
        else:
            amp_off = contextlib.nullcontext()
            lock = contextlib.nullcontext()
        return inf, amp_off, lock

    @staticmethod
    def _safe_vec(name: str, fn, n: int) -> np.ndarray:
        try:
            v = fn()
            if hasattr(v, 'detach'):
                v = v.detach()
            if hasattr(v, 'cpu'):
                v = v.cpu()
            if hasattr(v, 'numpy'):
                v = v.numpy()
            v = np.asarray(v).reshape(-1)
            if v.size == 1 and n > 1:
                v = np.full((n,), float(v.item()), dtype=np.float32)
            return v.astype(np.float32, copy=False)
        except Exception as e:
            print(f"[WARN][IQA][{name}] fallo: {e!r}")
            return np.full((n,), np.nan, dtype=np.float32)

    def fr_pair_batch(self, preds01: list[np.ndarray], refs01: list[np.ndarray], pin_memory: bool = True) -> dict[str, np.ndarray]:
        assert len(preds01) == len(refs01)
        n = len(preds01)
        if n == 0:
            return {}
        xb = self._to_rgb_batch(preds01)
        rb = self._to_rgb_batch(refs01)

        t_pred = torch.from_numpy(xb)
        t_ref = torch.from_numpy(rb)
        if pin_memory and (isinstance(self.device, torch.device) and self.device.type == 'cuda'):
            t_pred = t_pred.pin_memory()
            t_ref = t_ref.pin_memory()

        t_pred = t_pred.to(self.device, non_blocking=True)
        t_ref = t_ref.to(self.device, non_blocking=True)

        inf, amp_off, lock = self._infer_ctx()
        with lock, inf, amp_off:
            out = {}
            out['SSIM'] = self._safe_vec('SSIM', lambda: self.ssim(t_pred, t_ref), n)
            out['MS_SSIM'] = self._safe_vec('MS_SSIM', lambda: self.ms_ssim(t_pred, t_ref), n)
            out['FSIM'] = self._safe_vec('FSIM', lambda: self.fsim(t_pred, t_ref), n)
            out['VIF'] = self._safe_vec('VIF', lambda: self.vif(t_pred, t_ref), n)
            out['GMSD'] = self._safe_vec('GMSD', lambda: self.gmsd(t_pred, t_ref), n)
            out['DISTS'] = self._safe_vec('DISTS', lambda: self.dists(t_pred, t_ref), n)
            out['LPIPS'] = self._safe_vec('LPIPS', lambda: self.lpips(t_pred, t_ref), n)
        return out

    def nr_single_batch(self, imgs01: list[np.ndarray], pin_memory: bool = True) -> dict[str, np.ndarray]:
        n = len(imgs01)
        if n == 0:
            return {}
        xb = self._to_rgb_batch(imgs01)
        t = torch.from_numpy(xb)
        if pin_memory and (isinstance(self.device, torch.device) and self.device.type == 'cuda'):
            t = t.pin_memory()
        t = t.to(self.device, non_blocking=True)

        inf, amp_off, lock = self._infer_ctx()
        with lock, inf, amp_off:
            out = {}
            out['NIQE'] = self._safe_vec('NIQE', lambda: self.niqe(t), n)
            out['BRISQUE'] = self._safe_vec('BRISQUE', lambda: self.brisque(t), n)
            out['PIQE'] = self._safe_vec('PIQE', lambda: self.piqe(t), n)
            out['CLIP_IAQ'] = self._safe_vec('CLIP_IAQ', lambda: self.clipiqa(t), n)
            out['NIMA'] = self._safe_vec('NIMA', lambda: self.nima(t), n)
        return out



def calibrate_gates_from_p00(
    p00_rows: List[Dict[str, float]],
    p00_medians: Dict[str, float],
    gate_cfg: GateConfig,
) -> Dict[str, float]:
    """
    Gates calibrados sobre baseline P00 en HPO (q90 y mediana).
    Basado en Tabla 5 (Rev-4) con q90_AMBE, q90_NIQE, q90_PIQE, q90_SpeckleIndex, etc.
    """
    def q90(name: str) -> float:
        vals = [r.get(name, np.nan) for r in p00_rows]
        vals = [v for v in vals if np.isfinite(v)]
        return float(np.percentile(vals, 90.0)) if vals else float("nan")

    q = {
        "q90_AMBE": q90("AMBE"),
        "q90_NIQE": q90("NIQE"),
        "q90_PIQE": q90("PIQE"),
        "q90_SpeckleIndex_ROI": q90("SpeckleIndex_ROI"),
        "SNR_med": p00_medians.get("SNR", float("nan")),
    }

    # τ_t y τ_mem: si no se especifica, se auto-fija cerca del baseline (p.ej. +50%)
    if gate_cfg.tau_t_ms_1MP is None:
        bt = p00_medians.get("t_total_ms_1MP", np.nan)
        q["tau_t_ms_1MP"] = float(bt * 1.5) if np.isfinite(bt) else float("nan")
    else:
        q["tau_t_ms_1MP"] = float(gate_cfg.tau_t_ms_1MP)

    if gate_cfg.tau_mem_MB is None:
        bm = p00_medians.get("peak_ram_MB", np.nan)
        q["tau_mem_MB"] = float(bm * 1.5) if np.isfinite(bm) else float("nan")
    else:
        q["tau_mem_MB"] = float(gate_cfg.tau_mem_MB)

    q["delta_A"] = float(gate_cfg.delta_A)
    q["delta_SNR"] = float(gate_cfg.delta_SNR)
    return q


def apply_gates(
    med: Dict[str, float],
    block_scores: Dict[str, float],
    base_block_scores: Dict[str, float],
    gate_cal: Dict[str, float],
) -> Tuple[bool, Dict[str, Any]]:
    """
    Implementa gates G1..G5 (Rev-4 Tabla 5) de forma robusta.
    Devuelve (is_valid, reasons)
    """
    reasons = {}
    ok = True

    # G1 Fidelidad: S_A(θ) >= S_A(P00) - δ_A
    sa = block_scores.get("S_A", np.nan)
    sa0 = base_block_scores.get("S_A", np.nan)
    if np.isfinite(sa) and np.isfinite(sa0):
        cond = sa >= (sa0 - gate_cal["delta_A"])
        reasons["G1_Fidelidad"] = cond
        ok = ok and cond

    # G2 Control brillo: AMBE(θ) <= q90_AMBE(P00)
    ambe = med.get("AMBE", np.nan)
    q90_ambe = gate_cal.get("q90_AMBE", np.nan)
    if np.isfinite(ambe) and np.isfinite(q90_ambe):
        cond = ambe <= q90_ambe
        reasons["G2_Brillo"] = cond
        ok = ok and cond

    # G3 NR calidad: NIQE <= q90_NIQE y PIQE <= q90_PIQE
    niqe = med.get("NIQE", np.nan)
    piqe = med.get("PIQE", np.nan)
    q90_niqe = gate_cal.get("q90_NIQE", np.nan)
    q90_piqe = gate_cal.get("q90_PIQE", np.nan)
    if all(np.isfinite(v) for v in [niqe, piqe, q90_niqe, q90_piqe]):
        cond = (niqe <= q90_niqe) and (piqe <= q90_piqe)
        reasons["G3_NR"] = cond
        ok = ok and cond

    # G4 Ruido: SNR >= SNR(P00) - δ_SNR y SpeckleIndex_ROI <= q90_SpeckleIndex(P00)
    snr = med.get("SNR", np.nan)
    snr0 = gate_cal.get("SNR_med", np.nan)
    sp = med.get("SpeckleIndex_ROI", np.nan)
    q90_sp = gate_cal.get("q90_SpeckleIndex_ROI", np.nan)
    if all(np.isfinite(v) for v in [snr, snr0]):
        cond1 = snr >= (snr0 - gate_cal["delta_SNR"])
    else:
        cond1 = True
    if all(np.isfinite(v) for v in [sp, q90_sp]):
        cond2 = sp <= q90_sp
    else:
        cond2 = True
    reasons["G4_Ruido"] = (cond1 and cond2)
    ok = ok and (cond1 and cond2)

    # G5 Costo: t_total_ms_1MP <= τ_t  y  peak_ram_MB <= τ_mem
    t1 = med.get("t_total_ms_1MP", np.nan)
    m1 = med.get("peak_ram_MB", np.nan)
    tau_t = gate_cal.get("tau_t_ms_1MP", np.nan)
    tau_m = gate_cal.get("tau_mem_MB", np.nan)
    condt = True if not (np.isfinite(t1) and np.isfinite(tau_t)) else (t1 <= tau_t)
    condm = True if not (np.isfinite(m1) and np.isfinite(tau_m)) else (m1 <= tau_m)
    reasons["G5_Costo"] = (condt and condm)
    ok = ok and (condt and condm)

    return ok, reasons


# =========================
# Evaluación por candidato
# =========================

@dataclass
class EvalConfig:
    eval_res: int = 512
    n_jobs: int = 16
    device: str = "cuda"        # para IQA durante HPO (no afecta despliegue final CPU)
    deploy_cpu_only: bool = False

    # Rendimiento / trazabilidad
    show_progress: bool = True
    gpu_batch: int = 32                 # batch para métricas pyiqa en GPU
    pin_memory: bool = True            # acelera H2D
    use_preproc_cache: bool = True     # cachea lectura+normalización+ROI (idéntico, más rápido)
    cache_dir: str | None = None       # si se define, persiste npz (re-uso entre ejecuciones)

    # Perfilado estilo “despliegue CPU” (guardrails costo)
    measure_cpu_style: bool = True
    cpu_style_reps: int = 8            # repeticiones para robustez en tiempo


@dataclass
class GateConfig:
    """Configuración de los *gates* (Tabla 5, Rev-5).

    Estos valores gobiernan únicamente los umbrales/penalizaciones de los gates G1..G5.
    Mantenerlos en un dataclass evita errores y permite trazabilidad (se serializan).
    """

    # Gate G1 (fidelidad): S_A(theta) >= S_A(P00) - delta_A
    delta_A: float = 0.02

    # Gate G4 (ruido): SNR(theta) >= SNR(P00) - delta_SNR
    delta_SNR: float = 0.02

    # Gate G5 (costo): umbrales opcionales. Si None -> auto (baseline * 1.5)
    tau_t_ms_1MP: float | None = None
    tau_mem_MB: float | None = None


# =========================
# Cache de preprocesamiento (I/O + normalización + ROI)
# =========================
class PreprocCache:
    """    Cachea resultados deterministas por (sha256, eval_res, modality):
      - lectura (DICOM/imagen), normalización percentiles, letterbox
      - ROI sin etiquetas

    Objetivo: acelerar SIN cambiar el protocolo (la salida es idéntica a calcularlo cada vez).
    """

    def __init__(self, max_items: int = 4096, cache_dir: str | None = None):
        self.max_items = int(max_items)
        self.cache_dir = cache_dir
        self._mem: OrderedDict[str, dict] = OrderedDict()
        self._lock = threading.Lock()
        if cache_dir is not None:
            os.makedirs(cache_dir, exist_ok=True)

    @staticmethod
    def _key(sha256: str, modality: str, eval_res: int, p_low: float, p_high: float) -> str:
        return f"{sha256}__{modality}__{eval_res}__p{p_low:g}-{p_high:g}"

    def get(
        self,
        path: str,
        sha256: str,
        modality: str,
        eval_res: int,
        p_low: float = 1.0,
        p_high: float = 99.0,
        force_refresh: bool = False,
    ) -> dict:
        k = self._key(sha256, modality, eval_res, p_low, p_high)
        # 1) memoria
        if not force_refresh:
            with self._lock:
                v = self._mem.get(k)
                if v is not None:
                    self._mem.move_to_end(k)
                    return v
        # 2) disco
        if (self.cache_dir is not None) and (not force_refresh):
            f = os.path.join(self.cache_dir, k + '.npz')
            if os.path.isfile(f):
                try:
                    npz = np.load(f, allow_pickle=False)
                    v = {
                        'x01': npz['x01'].astype(np.float32, copy=False),
                        'roi': npz['roi'].astype(np.uint8, copy=False),
                        'path': path,
                        'modality': modality,
                        'sha256': sha256,
                    }
                    with self._lock:
                        self._mem[k] = v
                        self._mem.move_to_end(k)
                        while len(self._mem) > self.max_items:
                            self._mem.popitem(last=False)
                    return v
                except Exception:
                    pass

        # 3) compute
        x = read_image_any(path)
        x01 = normalize_percentiles(x, p_low, p_high)
        x01, _ = letterbox_to_square(x01, eval_res)
        roi = roi_mask_from_image(x01, modality)
        v = {'x01': x01.astype(np.float32, copy=False), 'roi': roi.astype(np.uint8, copy=False), 'path': path, 'modality': modality, 'sha256': sha256}

        # persist
        if self.cache_dir is not None:
            try:
                f = os.path.join(self.cache_dir, k + '.npz')
                np.savez_compressed(f, x01=v['x01'], roi=v['roi'])
            except Exception:
                pass

        with self._lock:
            self._mem[k] = v
            self._mem.move_to_end(k)
            while len(self._mem) > self.max_items:
                self._mem.popitem(last=False)
        return v


# instancia global (se configura en main via EvalConfig.cache_dir)
_PREPROC_CACHE = PreprocCache(max_items=8192, cache_dir=None)


class MemorySampler:
    def __init__(self, process: psutil.Process, interval: float = 0.01):
        self.process = process
        self.interval = interval
        self._stop = threading.Event()
        self.peak_rss = 0

    def run(self):
        while not self._stop.is_set():
            try:
                rss = self.process.memory_info().rss
                if rss > self.peak_rss:
                    self.peak_rss = rss
            except Exception:
                pass
            time.sleep(self.interval)

    def stop(self):
        self._stop.set()


def approx_ops_per_pixel(pipeline_ops: Tuple[str, ...], theta: Theta) -> float:
    # Heurístico comparativo (no absoluto): útil para auditoría relativa.
    ops = 0.0
    for op in pipeline_ops:
        if op == "C":
            ops += 50.0 + 2.0 * float(theta.c) + 0.2 * (theta.tile[0] + theta.tile[1])
        elif op == "G":
            ops += 80.0 + 0.5 * float(theta.r) + 1e4 * float(theta.eps)
        elif op == "U":
            ops += 30.0 + 10.0 * float(theta.alpha) + 2.0 * float(theta.sigma)
    return float(ops)


def crop_to_roi_bbox(x01: np.ndarray, roi: np.ndarray, target_size: int = 512) -> np.ndarray:
    ys, xs = np.where(roi > 0)
    if len(xs) < 32:
        return x01  # fallback: imagen completa
    y0, y1 = int(ys.min()), int(ys.max())
    x0, x1 = int(xs.min()), int(xs.max())
    pad = 3
    y0 = max(0, y0 - pad)
    y1 = min(x01.shape[0] - 1, y1 + pad)
    x0 = max(0, x0 - pad)
    x1 = min(x01.shape[1] - 1, x1 + pad)
    cropped = x01[y0:y1 + 1, x0:x1 + 1]
    # Redimensionar a tamaño fijo (ej. 512x512) para batcheo
    resized = cv2.resize(cropped, (target_size, target_size), interpolation=cv2.INTER_AREA)
    return resized

def compute_block_scores(med: Dict[str, float], base_medians: Dict[str, float]) -> Dict[str, float]:
    """
    Calcula los scores normalizados por bloque (S_A, S_B, S_C, S_D) según Rev-5.
    Cada métrica se normaliza respecto a la mediana del baseline P00.
    Dirección ↑: mejora si > 1; ↓: mejora si < 1 → se invierte para que ↑ siempre sea mejor.
    Luego se promedia robustamente dentro del bloque.
    """
    import numpy as np

    def norm_val(v: float, v0: float, direction: str) -> float:
        if not (np.isfinite(v) and np.isfinite(v0)):
            return np.nan
        if direction == "up":
            return v / (v0 + 1e-12)
        elif direction == "down":
            return v0 / (v + 1e-12)
        else:
            return np.nan

    block_vals = {"A": [], "B": [], "C": [], "D": []}
    for metric, block, direction in PRIMARY_METRICS:
        v = med.get(metric, np.nan)
        v0 = base_medians.get(metric, np.nan)
        nv = norm_val(v, v0, direction)
        if np.isfinite(nv):
            block_vals[block].append(nv)

    scores = {}
    for b in ["A", "B", "C", "D"]:
        vals = block_vals[b]
        if vals:
            scores[f"S_{b}"] = float(np.median(vals))
        else:
            scores[f"S_{b}"] = float("nan")
    return scores

def borda_rank(items: List[Dict[str, Any]], key: str) -> Dict[str, int]:
    """
    Asigna puntos Borda (ranking inverso) a una lista de ítems según un valor numérico.
    - El mejor (máximo valor) recibe N-1 puntos, el segundo N-2, ..., el último 0.
    - Ítems con NaN reciben 0 puntos.
    - Devuelve un dict {id: puntos}.
    """
    import numpy as np
    n = len(items)
    if n == 0:
        return {}
    # Extraer valores y manejar NaN
    vals = []
    ids = []
    for item in items:
        v = item.get(key, np.nan)
        if not np.isfinite(v):
            v = -np.inf  # para que queden al final
        vals.append(v)
        ids.append(item["id"])
    vals = np.array(vals)
    # Orden descendente (mayor valor = mejor)
    order = np.argsort(-vals, kind='mergesort')  # stable sort
    points = np.zeros(n, dtype=int)
    # Asignar puntos Borda: N-1, N-2, ..., 0
    for rank, idx in enumerate(order):
        if vals[idx] == -np.inf:
            points[idx] = 0
        else:
            points[idx] = n - 1 - rank
    return {ids[i]: int(points[i]) for i in range(n)}

def summarize_medians(per_image_rows: List[Dict[str, float]], metric_names: List[str]) -> Dict[str, float]:
    """
    Calcula la mediana robusta (ignorando NaN) para cada métrica en una lista de resultados por imagen.
    """
    import numpy as np
    medians = {}
    for name in metric_names:
        vals = []
        for r in per_image_rows:
            v = r.get(name, np.nan)
            if np.isfinite(v):
                vals.append(v)
        if vals:
            medians[name] = float(np.median(vals))
        else:
            medians[name] = float("nan")
    return medians


def eval_candidate_on_split(
    df: pd.DataFrame,
    split_idx: List[int],
    pipeline_id: str,
    theta: Theta,
    gate_cal: Optional[Dict[str, float]],
    base_block_scores: Optional[Dict[str, float]],
    eval_cfg: EvalConfig,
    iqa: Optional[IQAModels],
) -> Dict[str, Any]:
    """    Evalúa un candidato θ sobre un split (HPO/VAL/HOLD) cumpliendo Rev-5.

    Aceleración (sin alterar el protocolo):
      1) Cache determinista de preprocesamiento (I/O + normalización + ROI)
      2) IQA pyiqa en GPU **batcheado** (se ejecuta fuera de threads)

    Devuelve:
      - per_image: lista de dicts (métricas por imagen)
      - medians: mediana por métrica
      - block_scores: S_A..S_D
      - gates: ok + razones (cuando gate_cal/base disponibles)
    """
    assert pipeline_id in PIPELINES
    ops = PIPELINES[pipeline_id]

    # evitar oversubscription: paralelizamos a nivel Python y dejamos OpenCV a 1 thread
    force_cv2_threads(1)

    # configurar cache_dir dinámicamente
    if eval_cfg.use_preproc_cache:
        _PREPROC_CACHE.cache_dir = eval_cfg.cache_dir

    proc = psutil.Process(os.getpid())

    # --- profiling RAM/CPU (bloque D: costo)
    mem_sampler = MemorySampler(proc, interval=0.01)
    mem_thread = threading.Thread(target=mem_sampler.run, daemon=True)
    mem_thread.start()
    t_cpu_user0, t_cpu_sys0 = proc.cpu_times().user, proc.cpu_times().system

    per_image_rows: List[Dict[str, float]] = []

    # buffers para IQA (solo si aplica)
    need_iqa = (iqa is not None) and (not eval_cfg.deploy_cpu_only)
    y_buf: list[np.ndarray] = []
    x_buf: list[np.ndarray] = []
    yroi_buf: list[np.ndarray] = []
    xroi_buf: list[np.ndarray] = []

    from concurrent.futures import ThreadPoolExecutor, as_completed

    def work_one(i: int) -> Dict[str, Any]:
        row = df.loc[i]
        path = row['path']
        modality = row['modality']
        sha = row.get('sha256', '')

        if eval_cfg.use_preproc_cache and sha:
            prep = _PREPROC_CACHE.get(path=path, sha256=sha, modality=modality, eval_res=eval_cfg.eval_res)
            x01 = prep['x01']
            roi = prep['roi']
        else:
            x = read_image_any(path)
            x01 = normalize_percentiles(x, 1, 99)
            x01, _ = letterbox_to_square(x01, eval_cfg.eval_res)
            roi = roi_mask_from_image(x01, modality)

        # pipeline
        y01, tms = apply_pipeline(x01, ops, theta)

        # métricas clásicas (CPU)
        mi, nmi = _mi_nmi(x01, y01, bins=64)
        ambe = abs(float(y01.mean() - x01.mean()))
        ent = _entropy_256(y01)
        epi_g = _edge_preservation_index(x01, y01, mask=None)
        epi_r = _edge_preservation_index(x01, y01, mask=roi)
        ten = _tenengrad(y01, mask=None)
        lapv = _laplacian_variance(y01, mask=None)
        cpbd = _cpbd_sharpness(y01)
        roi_stats = _cnr_snr_enl_speckle(y01, roi)

        H, W = y01.shape
        npx = float(H * W)
        scale_1mp = 1e6 / max(npx, 1.0)
        t_total_ms_1mp = float(tms['t_total_ms'] * scale_1mp)
        t_total_ms_4mp = float(tms['t_total_ms'] * (4e6 / max(npx, 1.0)))
        t_total_ms_16mp = float(tms['t_total_ms'] * (16e6 / max(npx, 1.0)))
        imgs_per_sec = 1000.0 / max(tms['t_total_ms'], 1e-9)

        out = {
            'path': path,
            'modality': modality,
            'pipeline': pipeline_id,
            **theta.as_dict(),

            'PSNR': _psnr(x01, y01),
            'MSE': _mse(x01, y01),
            'RMSE': _rmse(x01, y01),
            'MI': float(mi),
            'NMI': float(nmi),
            'AMBE': float(ambe),
            'ENTROPY': float(ent),
            'EPI_global': float(epi_g),
            'EPI_ROI': float(epi_r),
            'CPBD': float(cpbd),
            'TENEGRAD': float(ten),
            'LAPLACIAN_VARIANCE': float(lapv),
            **roi_stats,

            't_total_ms': float(tms['t_total_ms']),
            't_CLAHE_ms': float(tms.get('t_CLAHE_ms', 0.0)),
            't_GF_ms': float(tms.get('t_GF_ms', 0.0)),
            't_USM_ms': float(tms.get('t_USM_ms', 0.0)),
            't_total_ms_1MP': float(t_total_ms_1mp),
            't_total_ms_4MP': float(t_total_ms_4mp),
            't_total_ms_16MP': float(t_total_ms_16mp),
            'imgs_por_seg': float(imgs_per_sec),
            'num_pasadas_imagen': float(len(ops)),
            'ops_aprox_por_pixel': float(approx_ops_per_pixel(ops, theta)),
        }

        # Para IQA (GPU) regresamos buffers auxiliares
        if need_iqa:
            x_roi = crop_to_roi_bbox(x01, roi)
            y_roi = crop_to_roi_bbox(y01, roi)
            return out, x01, y01, x_roi, y_roi
        return out, None, None, None, None

    # --- ejecutar por imagen (CPU-parallel)
    enabled_pbar = bool(eval_cfg.show_progress)
    with ThreadPoolExecutor(max_workers=max(1, eval_cfg.n_jobs)) as ex:
        futs = [ex.submit(work_one, i) for i in split_idx]
        for fut in progress_iter(as_completed(futs), desc=f"{pipeline_id} eval(split={len(split_idx)})", total=len(futs), enabled=enabled_pbar):
            out, x01, y01, xroi, yroi = fut.result()
            per_image_rows.append(out)
            if need_iqa:
                x_buf.append(x01)
                y_buf.append(y01)
                xroi_buf.append(xroi)
                yroi_buf.append(yroi)

    # --- IQA batcheado (GPU)
    if need_iqa and len(per_image_rows) > 0:
        B = max(1, int(eval_cfg.gpu_batch))
        # procesar en mini-batches
        for s in progress_iter(range(0, len(per_image_rows), B), desc=f"{pipeline_id} IQA(batch)", total=(len(per_image_rows)+B-1)//B, enabled=enabled_pbar):
            e = min(len(per_image_rows), s + B)
            fr = iqa.fr_pair_batch(y_buf[s:e], x_buf[s:e], pin_memory=eval_cfg.pin_memory)
            nr = iqa.nr_single_batch(y_buf[s:e], pin_memory=eval_cfg.pin_memory)
            fr_roi = iqa.fr_pair_batch(yroi_buf[s:e], xroi_buf[s:e], pin_memory=eval_cfg.pin_memory)

            for k in range(s, e):
                j = k - s
                # global FR
                for name in ['SSIM', 'MS_SSIM', 'FSIM', 'VIF', 'GMSD', 'DISTS', 'LPIPS']:
                    if name in fr:
                        per_image_rows[k][name] = float(fr[name][j])
                # global NR
                for name in ['NIQE', 'BRISQUE', 'PIQE', 'CLIP_IAQ', 'NIMA']:
                    if name in nr:
                        per_image_rows[k][name] = float(nr[name][j])
                # ROI primarias
                if 'SSIM' in fr_roi:
                    per_image_rows[k]['SSIM_ROI'] = float(fr_roi['SSIM'][j])
                if 'MS_SSIM' in fr_roi:
                    per_image_rows[k]['MS_SSIM_ROI'] = float(fr_roi['MS_SSIM'][j])
                if 'VIF' in fr_roi:
                    per_image_rows[k]['VIF_ROI'] = float(fr_roi['VIF'][j])
                # ROI secundarias
                if 'FSIM' in fr_roi:
                    per_image_rows[k]['FSIM_ROI'] = float(fr_roi['FSIM'][j])

                # MI/NMI ROI (CPU) por coherencia con resto (más rápido que recomputar batch)
                mi_roi, nmi_roi = _mi_nmi(xroi_buf[k], yroi_buf[k], bins=64)
                per_image_rows[k]['MI_ROI'] = float(mi_roi)
                per_image_rows[k]['NMI_ROI'] = float(nmi_roi)

    # --- finalizar profiling
    time.sleep(0.01)
    mem_sampler.stop()
    mem_thread.join(timeout=1.0)
    t_cpu_user1, t_cpu_sys1 = proc.cpu_times().user, proc.cpu_times().system

    peak_ram_mb = float(mem_sampler.peak_rss / (1024.0 * 1024.0))
    user_time = float(max(0.0, t_cpu_user1 - t_cpu_user0))
    sys_time = float(max(0.0, t_cpu_sys1 - t_cpu_sys0))

    for r in per_image_rows:
        r['peak_ram_MB'] = peak_ram_mb
        r['cpu_util_%'] = float('nan')
        r['user_time_s'] = user_time
        r['sys_time_s'] = sys_time

    all_metric_names = [m for m, _, _ in PRIMARY_METRICS + SECONDARY_METRICS]
    med = summarize_medians(per_image_rows, all_metric_names)

    block_scores = {}
    gates = {'ok': True, 'reasons': {}}
    if gate_cal is not None and base_block_scores is not None:
        base_medians = gate_cal.get('__base_medians__', {})
        block_scores = compute_block_scores(med, base_medians)
        ok, reasons = apply_gates(med, block_scores, base_block_scores, gate_cal)
        gates = {'ok': bool(ok), 'reasons': reasons}
    else:
        block_scores = {'S_A': float('nan'), 'S_B': float('nan'), 'S_C': float('nan'), 'S_D': float('nan')}

    return {'per_image': per_image_rows, 'medians': med, 'block_scores': block_scores, 'gates': gates}



# =========================
# Grid -> NSGA-II -> Optuna
# =========================
def iter_grid_candidates_for_pipeline(pipeline_ops: Tuple[str, ...]) -> List[Theta]:
    # Construye el producto cartesiano solo de parámetros usados por el pipeline
    keys = []
    if "C" in pipeline_ops:
        keys += ["c", "tile"]
    if "G" in pipeline_ops:
        keys += ["r", "eps"]
    if "U" in pipeline_ops:
        keys += ["alpha", "sigma", "tau_u"]

    spaces = [GRID_SPACE[k] for k in keys]
    thetas = []
    for combo in itertools.product(*spaces):
        d = DEFAULT_THETA.copy()
        for k, v in zip(keys, combo):
            d[k] = v
        thetas.append(Theta(**d))
    return thetas


def sample_list(xs: List[Any], max_n: int, seed: int) -> List[Any]:
    if len(xs) <= max_n:
        return xs
    rng = np.random.RandomState(seed)
    idx = rng.choice(len(xs), size=max_n, replace=False)
    return [xs[int(i)] for i in idx]


def run_grid(
    df: pd.DataFrame,
    split: Dict[str, List[int]],
    pipeline_id: str,
    eval_cfg: EvalConfig,
    gate_cfg: GateConfig,
    top_k: int = 40,
    grid_max: int = 4000,
    seed: int = 123,
) -> Tuple[List[Dict[str, Any]], Dict[str, float], Dict[str, float]]:
    """
    Grid inicial (cobertura discreta) -> top-K por ranking (Rev-4).
    """
    ops = PIPELINES[pipeline_id]
    # baseline P00 en HPO para calibrar gates
    iqa = IQAModels(eval_cfg.device) if (not eval_cfg.deploy_cpu_only) else None

    base_eval = eval_candidate_on_split(
        df=df, split_idx=split["HPO"], pipeline_id="P00", theta=Theta(),
        gate_cal=None, base_block_scores=None, eval_cfg=eval_cfg, iqa=iqa
    )
    base_medians = base_eval["medians"]
    base_block_scores = compute_block_scores(base_medians, base_medians)

    gate_cal = calibrate_gates_from_p00(base_eval["per_image"], base_medians, gate_cfg)
    gate_cal["__base_medians__"] = base_medians  # inyectar baseline medians para scoring

    # construir grid para pipeline y muestrear si es enorme
    all_thetas = iter_grid_candidates_for_pipeline(ops)
    cand_thetas = sample_list(all_thetas, max_n=grid_max, seed=seed)

    results = []
    for j, th in enumerate(progress_iter(cand_thetas, desc=f'GRID {pipeline_id}', total=len(cand_thetas), enabled=eval_cfg.show_progress)):
        ev = eval_candidate_on_split(
            df=df, split_idx=split["HPO"], pipeline_id=pipeline_id, theta=th,
            gate_cal=gate_cal, base_block_scores=base_block_scores,
            eval_cfg=eval_cfg, iqa=iqa
        )
        med = ev["medians"]
        bs = ev["block_scores"]
        ok = ev["gates"]["ok"]
        # scalar score: suma de bloques, penaliza D (porque es “riesgo/costo”)
        score = float((bs.get("S_A", 0.0) + bs.get("S_B", 0.0) + bs.get("S_C", 0.0)) - 0.75 * bs.get("S_D", 0.0))
        results.append({
            "id": f"GRID_{j}",
            "theta": th.as_dict(),
            "ok": ok,
            "score": score,
            **bs,
            **{f"med_{k}": v for k, v in med.items() if k in ["AMBE", "NIQE", "PIQE", "SNR", "SpeckleIndex_ROI", "t_total_ms_1MP", "peak_ram_MB"]},
        })

    # filtrar válidos por gates
    valids = [r for r in results if r["ok"]]
    if not valids:
        # fallback: permitir aunque no pasen gates (para evitar bloqueo total); pero se marca.
        valids = results

    # ranking Borda por bloques (Rev-4: “Borda/Pareto”)
    ptsA = borda_rank(valids, "S_A")
    ptsB = borda_rank(valids, "S_B")
    ptsC = borda_rank(valids, "S_C")
    ptsD = borda_rank(valids, "S_D")  # aquí D ya está “higher better” por definición de mejoras; si es negativo, Borda lo refleja.

    for r in valids:
        r["borda"] = int(ptsA[r["id"]] + ptsB[r["id"]] + ptsC[r["id"]] + ptsD[r["id"]])

    valids.sort(key=lambda x: x["borda"], reverse=True)
    top = valids[:top_k]
    return top, base_medians, base_block_scores


def dominates(a: Dict[str, Any], b: Dict[str, Any]) -> bool:
    # a domina b si >= en A,B,C y >= en D, y > en al menos 1
    keys = ["S_A", "S_B", "S_C", "S_D"]
    ge = all(a.get(k, -1e9) >= b.get(k, -1e9) for k in keys)
    gt = any(a.get(k, -1e9) > b.get(k, -1e9) for k in keys)
    return ge and gt


def fast_nondominated_sort(pop: List[Dict[str, Any]]) -> List[List[int]]:
    fronts = []
    S = [set() for _ in pop]
    n = [0 for _ in pop]
    rank = [0 for _ in pop]

    for p in range(len(pop)):
        for q in range(len(pop)):
            if p == q:
                continue
            if dominates(pop[p], pop[q]):
                S[p].add(q)
            elif dominates(pop[q], pop[p]):
                n[p] += 1
        if n[p] == 0:
            rank[p] = 0

    F0 = [i for i in range(len(pop)) if n[i] == 0]
    fronts.append(F0)

    i = 0
    while i < len(fronts) and fronts[i]:
        Q = []
        for p in fronts[i]:
            for q in S[p]:
                n[q] -= 1
                if n[q] == 0:
                    rank[q] = i + 1
                    Q.append(q)
        i += 1
        if Q:
            fronts.append(Q)
    return fronts


def crowding_distance(front: List[int], pop: List[Dict[str, Any]]) -> Dict[int, float]:
    dist = {i: 0.0 for i in front}
    keys = ["S_A", "S_B", "S_C", "S_D"]
    for k in keys:
        vals = [(i, pop[i].get(k, -1e9)) for i in front]
        vals.sort(key=lambda x: x[1])
        dist[vals[0][0]] = float("inf")
        dist[vals[-1][0]] = float("inf")
        vmin, vmax = vals[0][1], vals[-1][1]
        if abs(vmax - vmin) < 1e-12:
            continue
        for j in range(1, len(vals) - 1):
            prevv = vals[j - 1][1]
            nextv = vals[j + 1][1]
            dist[vals[j][0]] += (nextv - prevv) / (vmax - vmin)
    return dist


def mutate_theta(theta: Theta, ops: Tuple[str, ...], rng: np.random.RandomState, p: float = 0.2) -> Theta:
    d = theta.as_dict()
    if "C" in ops and rng.rand() < p:
        d["c"] = float(rng.choice(GRID_SPACE["c"]))
    if "C" in ops and rng.rand() < p:
        # Corrección: elegir por índice para evitar ValueError con listas de tuplas
        tile_options = GRID_SPACE["tile"]
        idx = rng.choice(len(tile_options))
        d["tile"] = tile_options[idx]
    if "G" in ops and rng.rand() < p:
        d["r"] = int(rng.choice(GRID_SPACE["r"]))
    if "G" in ops and rng.rand() < p:
        d["eps"] = float(rng.choice(GRID_SPACE["eps"]))
    if "U" in ops and rng.rand() < p:
        d["alpha"] = float(rng.choice(GRID_SPACE["alpha"]))
    if "U" in ops and rng.rand() < p:
        d["sigma"] = float(rng.choice(GRID_SPACE["sigma"]))
    if "U" in ops and rng.rand() < p:
        d["tau_u"] = float(rng.choice(GRID_SPACE["tau_u"]))
    return Theta(**d)

def crossover(a: Theta, b: Theta, ops: Tuple[str, ...], rng: np.random.RandomState) -> Theta:
    da, db = a.as_dict(), b.as_dict()
    child = DEFAULT_THETA.copy()
    for k in child.keys():
        # solo afecta si el op está presente
        if k in ("c", "tile") and "C" not in ops:
            continue
        if k in ("r", "eps") and "G" not in ops:
            continue
        if k in ("alpha", "sigma", "tau_u") and "U" not in ops:
            continue
        child[k] = da[k] if rng.rand() < 0.5 else db[k]
    return Theta(**child)


def run_nsga2(
    df: pd.DataFrame,
    split: Dict[str, List[int]],
    pipeline_id: str,
    seed: int,
    eval_cfg: EvalConfig,
    gate_cfg: GateConfig,
    base_medians: Dict[str, float],
    base_block_scores: Dict[str, float],
    init_top: List[Dict[str, Any]],
    pop_size: int = 40,
    generations: int = 10,
) -> List[Dict[str, Any]]:
    """
    NSGA-II multiobjetivo (aprox. frentes Pareto entre bloques) como en Rev-4.
    """
    ops = PIPELINES[pipeline_id]
    rng = np.random.RandomState(seed)

    iqa = IQAModels(eval_cfg.device) if (not eval_cfg.deploy_cpu_only) else None

    # calibrar gates desde P00 (HPO)
    # Reutilizamos gate_cal a partir de baseline medians (ya medidos en grid)
    gate_cal = {"__base_medians__": base_medians}
    # Para q90 necesitamos P00 per-image; lo recalculamos rápido una vez
    p00_eval = eval_candidate_on_split(
        df=df, split_idx=split["HPO"], pipeline_id="P00", theta=Theta(),
        gate_cal=None, base_block_scores=None, eval_cfg=eval_cfg, iqa=iqa
    )
    gate_cal.update(calibrate_gates_from_p00(p00_eval["per_image"], base_medians, gate_cfg))
    gate_cal["__base_medians__"] = base_medians

    # población inicial = top del grid + mutaciones para diversidad
    pop: List[Dict[str, Any]] = []
    for j in range(min(len(init_top), pop_size)):
        th = Theta(**init_top[j]["theta"])
        pop.append({"id": f"GA0_{j}", "theta_obj": th})

    while len(pop) < pop_size:
        th = mutate_theta(Theta(**init_top[rng.randint(len(init_top))]["theta"]), ops, rng, p=0.35)
        pop.append({"id": f"GA0_fill_{len(pop)}", "theta_obj": th})

    # evaluar población
    def eval_ind(ind_id: str, th: Theta) -> Dict[str, Any]:
        ev = eval_candidate_on_split(
            df=df, split_idx=split["HPO"], pipeline_id=pipeline_id, theta=th,
            gate_cal=gate_cal, base_block_scores=base_block_scores,
            eval_cfg=eval_cfg, iqa=iqa
        )
        med, bs, ok = ev["medians"], ev["block_scores"], ev["gates"]["ok"]
        score = float((bs.get("S_A", 0.0) + bs.get("S_B", 0.0) + bs.get("S_C", 0.0)) - 0.75 * bs.get("S_D", 0.0))
        return {"id": ind_id, "theta": th.as_dict(), "ok": ok, "score": score, **bs}

    # loop GA
    evaluated: List[Dict[str, Any]] = []
    for ind in pop:
        evaluated.append(eval_ind(ind["id"], ind["theta_obj"]))

    for g in range(1, generations + 1):
        # selección por torneos (rank+crowding)
        fronts = fast_nondominated_sort(evaluated)
        ranks = {}
        for r, fr in enumerate(fronts):
            for idx in fr:
                ranks[evaluated[idx]["id"]] = r

        # crowding por primer frente (aprox)
        cd = {}
        if fronts:
            dist = crowding_distance(fronts[0], evaluated)
            for idx, d in dist.items():
                cd[evaluated[idx]["id"]] = d

        def tournament() -> Dict[str, Any]:
            a = evaluated[rng.randint(len(evaluated))]
            b = evaluated[rng.randint(len(evaluated))]
            ra, rb = ranks.get(a["id"], 999), ranks.get(b["id"], 999)
            if ra < rb:
                return a
            if rb < ra:
                return b
            # empate: crowding mayor gana
            return a if cd.get(a["id"], 0.0) >= cd.get(b["id"], 0.0) else b

        children: List[Theta] = []
        while len(children) < pop_size:
            p1 = Theta(**tournament()["theta"])
            p2 = Theta(**tournament()["theta"])
            ch = crossover(p1, p2, ops, rng)
            ch = mutate_theta(ch, ops, rng, p=0.25)
            children.append(ch)

        new_eval: List[Dict[str, Any]] = []
        for j, th in enumerate(children):
            new_eval.append(eval_ind(f"GA{g}_{j}", th))

        # unir y seleccionar elitismo NSGA-II (fronts + crowding)
        combined = evaluated + new_eval
        fronts = fast_nondominated_sort(combined)
        next_pop = []
        for fr in fronts:
            if len(next_pop) + len(fr) <= pop_size:
                next_pop.extend([combined[i] for i in fr])
            else:
                dist = crowding_distance(fr, combined)
                fr_sorted = sorted(fr, key=lambda i: dist[i], reverse=True)
                need = pop_size - len(next_pop)
                next_pop.extend([combined[i] for i in fr_sorted[:need]])
                break
        evaluated = next_pop

    # devolver mejores por score (pero manteniendo Pareto implícito)
    evaluated.sort(key=lambda x: x.get("score", -1e9), reverse=True)
    return evaluated[:max(30, pop_size)]


def run_optuna_bo(
    df: pd.DataFrame,
    split: Dict[str, List[int]],
    pipeline_id: str,
    seed: int,
    eval_cfg: EvalConfig,
    gate_cfg: GateConfig,
    base_medians: Dict[str, float],
    base_block_scores: Dict[str, float],
    warmstart: List[Dict[str, Any]],
    trials: int = 60,
) -> List[Dict[str, Any]]:
    """
    BO con Optuna/TPE en la región de mejores frentes (Rev-4 Alg 2).
    """
    if optuna is None:
        raise RuntimeError("Necesitas optuna para BO. Instala: pip install optuna")
    ops = PIPELINES[pipeline_id]
    rng = np.random.RandomState(seed)

    iqa = IQAModels(eval_cfg.device) if (not eval_cfg.deploy_cpu_only) else None

    # calibrar gates desde P00 (HPO)
    gate_cal = {"__base_medians__": base_medians}
    p00_eval = eval_candidate_on_split(
        df=df, split_idx=split["HPO"], pipeline_id="P00", theta=Theta(),
        gate_cal=None, base_block_scores=None, eval_cfg=eval_cfg, iqa=iqa
    )
    gate_cal.update(calibrate_gates_from_p00(p00_eval["per_image"], base_medians, gate_cfg))
    gate_cal["__base_medians__"] = base_medians

    def suggest_theta(trial: "optuna.Trial") -> Theta:
        d = DEFAULT_THETA.copy()
        if "C" in ops:
            d["c"] = float(trial.suggest_categorical("c", GRID_SPACE["c"]))
            d["tile"] = tuple(trial.suggest_categorical("tile", GRID_SPACE["tile"]))
        if "G" in ops:
            d["r"] = int(trial.suggest_categorical("r", GRID_SPACE["r"]))
            d["eps"] = float(trial.suggest_categorical("eps", GRID_SPACE["eps"]))
        if "U" in ops:
            d["alpha"] = float(trial.suggest_categorical("alpha", GRID_SPACE["alpha"]))
            d["sigma"] = float(trial.suggest_categorical("sigma", GRID_SPACE["sigma"]))
            d["tau_u"] = float(trial.suggest_categorical("tau_u", GRID_SPACE["tau_u"]))
        return Theta(**d)

    def objective(trial: "optuna.Trial") -> float:
        th = suggest_theta(trial)
        ev = eval_candidate_on_split(
            df=df, split_idx=split["HPO"], pipeline_id=pipeline_id, theta=th,
            gate_cal=gate_cal, base_block_scores=base_block_scores,
            eval_cfg=eval_cfg, iqa=iqa
        )
        bs = ev["block_scores"]
        ok = ev["gates"]["ok"]
        # score escalar (maximizar A+B+C y reducir “riesgo/costo” D)
        score = float((bs.get("S_A", 0.0) + bs.get("S_B", 0.0) + bs.get("S_C", 0.0)) - 0.75 * bs.get("S_D", 0.0))
        if not ok:
            score -= 1.0  # penalización por violar gates
        return -score  # optuna minimiza

    sampler = optuna.samplers.TPESampler(seed=seed)
    study = optuna.create_study(direction="minimize", sampler=sampler)

    # warmstart: encolar algunos puntos del GA/grid
    for w in warmstart[:min(25, len(warmstart))]:
        params = {}
        for k, v in w["theta"].items():
            params[k] = v
        study.enqueue_trial(params)

    study.optimize(objective, n_trials=trials, show_progress_bar=False)

    # reconstruir mejores
    best = []
    for t in study.best_trials[:min(20, len(study.best_trials))]:
        d = DEFAULT_THETA.copy()
        for k, v in t.params.items():
            d[k] = v
        th = Theta(**d)
        best.append({"id": f"BO_{t.number}", "theta": th.as_dict(), "value": float(t.value)})
    return best


# =========================
# Selección θ* en Validación, reporte Hold-out
# =========================
def pick_theta_star_on_validation(
    df: pd.DataFrame,
    split: Dict[str, List[int]],
    pipeline_id: str,
    candidates: List[Dict[str, Any]],
    base_medians: Dict[str, float],
    base_block_scores: Dict[str, float],
    eval_cfg: EvalConfig,
    gate_cfg: GateConfig,
) -> Dict[str, Any]:
    iqa = IQAModels(eval_cfg.device) if (not eval_cfg.deploy_cpu_only) else None

    # baseline P00 para gates (q90) con medians base ya dados
    p00_eval = eval_candidate_on_split(
        df=df, split_idx=split["HPO"], pipeline_id="P00", theta=Theta(),
        gate_cal=None, base_block_scores=None, eval_cfg=eval_cfg, iqa=iqa
    )
    gate_cal = calibrate_gates_from_p00(p00_eval["per_image"], base_medians, gate_cfg)
    gate_cal["__base_medians__"] = base_medians

    scored = []
    for j, c in enumerate(progress_iter(candidates, desc=f'VAL select {pipeline_id}', total=len(candidates), enabled=eval_cfg.show_progress)):
        th = Theta(**c["theta"])
        ev = eval_candidate_on_split(
            df=df, split_idx=split["VAL"], pipeline_id=pipeline_id, theta=th,
            gate_cal=gate_cal, base_block_scores=base_block_scores,
            eval_cfg=eval_cfg, iqa=iqa
        )
        med = ev["medians"]
        bs = ev["block_scores"]
        ok = ev["gates"]["ok"]
        score = float((bs.get("S_A", 0.0) + bs.get("S_B", 0.0) + bs.get("S_C", 0.0)) - 0.75 * bs.get("S_D", 0.0))
        scored.append({"id": f"VAL_{j}", "theta": th.as_dict(), "ok": ok, "score": score, **bs, "med": med})

    # preferir ok; si todos no-ok, escoger máximo score
    ok_scored = [s for s in scored if s["ok"]]
    best = max(ok_scored, key=lambda x: x["score"]) if ok_scored else max(scored, key=lambda x: x["score"])
    return best


def eval_final_holdout(
    df: pd.DataFrame,
    split: Dict[str, List[int]],
    pipeline_id: str,
    theta_star: Theta,
    base_medians: Dict[str, float],
    base_block_scores: Dict[str, float],
    eval_cfg: EvalConfig,
    gate_cfg: GateConfig,
) -> Dict[str, Any]:
    iqa = IQAModels(eval_cfg.device) if (not eval_cfg.deploy_cpu_only) else None

    # gates calibrados SOLO con P00 en HPO (Rev-4: no tocar θ ni gates en hold-out)
    p00_eval = eval_candidate_on_split(
        df=df, split_idx=split["HPO"], pipeline_id="P00", theta=Theta(),
        gate_cal=None, base_block_scores=None, eval_cfg=eval_cfg, iqa=iqa
    )
    gate_cal = calibrate_gates_from_p00(p00_eval["per_image"], base_medians, gate_cfg)
    gate_cal["__base_medians__"] = base_medians

    ev = eval_candidate_on_split(
        df=df, split_idx=split["HOLD"], pipeline_id=pipeline_id, theta=theta_star,
        gate_cal=gate_cal, base_block_scores=base_block_scores,
        eval_cfg=eval_cfg, iqa=iqa
    )
    return ev


# =========================
# Reportería (CSV/JSON) + Estadística
# =========================
def ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)


def write_json(path: str, obj: Any) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def report_stats(per_image_df: pd.DataFrame, out_dir: str, tag: str) -> None:
    """
    Reporte estadístico D1-friendly:
    - descriptivos robustos (median, IQR)
    - por modalidad y global
    """
    ensure_dir(out_dir)
    metrics = [m for m, _, _ in PRIMARY_METRICS + SECONDARY_METRICS]
    group_cols = ["pipeline", "modality"]

    rows = []
    for (p, m), g in per_image_df.groupby(group_cols):
        for met in metrics:
            if met not in g.columns:
                continue
            v = g[met].dropna().to_numpy()
            if v.size == 0:
                continue
            rows.append({
                "pipeline": p,
                "modality": m,
                "metric": met,
                "median": float(np.median(v)),
                "q25": float(np.percentile(v, 25)),
                "q75": float(np.percentile(v, 75)),
                "mean": float(np.mean(v)),
                "std": float(np.std(v)),
                "n": int(v.size),
            })
    stats = pd.DataFrame(rows)
    stats.to_csv(os.path.join(out_dir, f"stats_{tag}.csv"), index=False, encoding="utf-8")

    # Pruebas simples vs P00 por modalidad (Wilcoxon) si scipy está disponible
    try:
        from scipy.stats import wilcoxon
        p00 = per_image_df[per_image_df["pipeline"] == "P00"]
        tests = []
        for met in metrics:
            if met not in per_image_df.columns:
                continue
            for mod in per_image_df["modality"].unique():
                a = per_image_df[(per_image_df["pipeline"] != "P00") & (per_image_df["modality"] == mod)]
                # se reporta por pipeline afuera (demasiado largo); aquí global por mod (agregado)
                # mejor: comparar P10 vs P00 como quick-check
                p10 = per_image_df[(per_image_df["pipeline"] == "P10") & (per_image_df["modality"] == mod)][met].dropna()
                p00m = p00[p00["modality"] == mod][met].dropna()
                n = min(len(p10), len(p00m))
                if n < 10:
                    continue
                stat, pval = wilcoxon(p10.iloc[:n], p00m.iloc[:n])
                tests.append({"modality": mod, "metric": met, "wilcoxon_stat": float(stat), "p_value": float(pval), "n": int(n)})
        pd.DataFrame(tests).to_csv(os.path.join(out_dir, f"wilcoxon_P10_vs_P00_{tag}.csv"), index=False, encoding="utf-8")
    except Exception:
        pass




def generate_d1_evidence_pack(
    df: pd.DataFrame,
    splits: Dict[str, List[int]],
    theta_star_all: Dict[str, Any],
    per_image_df: pd.DataFrame,
    out_dir: str,
) -> None:
    """    Genera evidencia lista para poblar la plantilla D1/Q1 (Tablas/Figuras/JSON).
    No altera el protocolo: solo consume resultados ya calculados.
    """
    ensure_dir(out_dir)
    rep_dir = os.path.join(out_dir, 'evidence_pack')
    ensure_dir(rep_dir)

    # TAB-1: catálogo P00..P15
    tab1 = []
    for pid, ops in PIPELINES.items():
        tab1.append({'pipeline': pid, 'ops': '=>'.join(ops), 'n_ops': len(ops)})
    pd.DataFrame(tab1).to_csv(os.path.join(rep_dir, 'TAB-1_catalogo_pipelines.csv'), index=False, encoding='utf-8')

    # TAB-2: bloques A-D
    tab2 = [{'metric': m, 'block': b, 'direction': d} for (m,b,d) in PRIMARY_METRICS]
    pd.DataFrame(tab2).to_csv(os.path.join(rep_dir, 'TAB-2_bloques_metricas.csv'), index=False, encoding='utf-8')

    # TAB-3: gates (descriptivo)
    tab3 = [
        {'gate': 'G1_Fidelidad', 'condicion': 'S_A(theta) >= S_A(P00) - delta_A', 'motivacion': 'Evita degradacion estructural global'},
        {'gate': 'G2_Brillo', 'condicion': 'AMBE(theta) <= q90_AMBE(P00)', 'motivacion': 'Evita cambios fotometricos (brillo) no deseados'},
        {'gate': 'G3_NR', 'condicion': 'NIQE(theta) <= q90_NIQE(P00) y PIQE(theta) <= q90_PIQE(P00)', 'motivacion': 'Evita degradacion perceptual NR'},
        {'gate': 'G4_Ruido', 'condicion': 'SNR(theta) >= SNR(P00)-delta_SNR y SpeckleIndex_ROI(theta) <= q90_SpeckleIndex_ROI(P00)', 'motivacion': 'Evita sobre-amplificacion de ruido / speckle'},
        {'gate': 'G5_Costo', 'condicion': 't_total_ms_1MP <= tau_t y peak_ram_MB <= tau_mem', 'motivacion': 'Viabilidad en recursos limitados'},
    ]
    pd.DataFrame(tab3).to_csv(os.path.join(rep_dir, 'TAB-3_gates_G1_G5.csv'), index=False, encoding='utf-8')

    # TAB-4: particiones
    rows=[]
    for split_name, idxs in splits.items():
        sub = df.loc[idxs]
        for mod, g in sub.groupby('modality'):
            rows.append({'split': split_name, 'modality': mod, 'n': int(len(g))})
        rows.append({'split': split_name, 'modality': 'ALL', 'n': int(len(sub))})
    pd.DataFrame(rows).to_csv(os.path.join(rep_dir, 'TAB-4_splits_conteos.csv'), index=False, encoding='utf-8')

    # TAB-8: theta*
    pd.DataFrame([{'pipeline': k, **v} for k,v in theta_star_all.items()]).to_csv(os.path.join(rep_dir, 'TAB-8_theta_star.csv'), index=False, encoding='utf-8')

    # TAB-7: P00 vs P10 por modalidad (mediana/IQR)
    if 'P10' in per_image_df['pipeline'].unique():
        metrics = [m for m,_,_ in PRIMARY_METRICS]
        out=[]
        for mod in sorted(per_image_df['modality'].unique()):
            for pid in ['P00','P10']:
                g = per_image_df[(per_image_df['pipeline']==pid) & (per_image_df['modality']==mod)]
                if g.empty:
                    continue
                for met in metrics:
                    if met not in g.columns:
                        continue
                    v=g[met].dropna().to_numpy()
                    if v.size==0:
                        continue
                    out.append({'modality': mod, 'pipeline': pid, 'metric': met, 'median': float(np.median(v)), 'q25': float(np.percentile(v,25)), 'q75': float(np.percentile(v,75)), 'n': int(v.size)})
        pd.DataFrame(out).to_csv(os.path.join(rep_dir, 'TAB-7_P00_vs_P10_holdout.csv'), index=False, encoding='utf-8')

    # FIG-3: boxplots por bloque (si matplotlib disponible)
    try:
        import matplotlib.pyplot as plt

        blocks = {'A': [], 'B': [], 'C': [], 'D': []}
        for met, b, _ in PRIMARY_METRICS:
            blocks[b].append(met)

        # score por bloque por imagen: suma normalizada relativa a P00 mediana (robusto) (aprox visual)
        # Nota: el score oficial ya se usa en HPO; aquí es solo para figura.
        fig_dir = os.path.join(rep_dir, 'figs')
        ensure_dir(fig_dir)

        for b in ['A','B','C','D']:
            mets = [m for m in blocks[b] if m in per_image_df.columns]
            if not mets:
                continue
            # agregación simple por imagen: mediana z-score robusto por métrica (solo para visual)
            base = per_image_df[per_image_df['pipeline']=='P00']
            if base.empty:
                continue
            base_med = {m: float(np.median(base[m].dropna())) for m in mets if base[m].dropna().size>0}
            # construir score por imagen (solo P00 vs P10 si existe)
            for pid in ['P00','P10']:
                if pid not in per_image_df['pipeline'].unique():
                    continue
            scores = {}
            for pid in ['P00','P10']:
                if pid not in per_image_df['pipeline'].unique():
                    continue
                g = per_image_df[per_image_df['pipeline']==pid]
                s=[]
                for _, r in g.iterrows():
                    acc=0.0; c=0
                    for m in mets:
                        if m not in base_med or (not np.isfinite(r.get(m, np.nan))):
                            continue
                        # dirección
                        direction = next(d for (mm,bb,d) in PRIMARY_METRICS if mm==m)
                        v=float(r[m])
                        v0=base_med[m]
                        # mejora relativa
                        rel = (v - v0)
                        if direction=='down':
                            rel = (v0 - v)
                        acc += rel
                        c += 1
                    s.append(acc / max(c,1))
                scores[pid]=s

            plt.figure()
            data=[scores.get('P00',[]), scores.get('P10',[])]
            plt.boxplot(data, labels=['P00','P10'])
            plt.title(f'FIG-3 Score aproximado por Bloque {b} (hold-out)')
            plt.ylabel('mejora relativa (aprox)')
            plt.tight_layout()
            plt.savefig(os.path.join(fig_dir, f'FIG-3_boxplot_bloque_{b}.png'), dpi=200)
            plt.close()
    except Exception:
        pass

    # JSON con valores clave para la plantilla
    evidence = {
        'N_ALL': int(len(df)),
        'N_RX': int((df['modality']=='RX').sum()),
        'N_MAMO': int((df['modality']=='MAMO').sum()),
        'N_ECO': int((df['modality']=='ECO').sum()),
        'theta_star_all': theta_star_all,
        'outputs': {
            'TAB-1': 'evidence_pack/TAB-1_catalogo_pipelines.csv',
            'TAB-2': 'evidence_pack/TAB-2_bloques_metricas.csv',
            'TAB-3': 'evidence_pack/TAB-3_gates_G1_G5.csv',
            'TAB-4': 'evidence_pack/TAB-4_splits_conteos.csv',
            'TAB-7': 'evidence_pack/TAB-7_P00_vs_P10_holdout.csv',
            'TAB-8': 'evidence_pack/TAB-8_theta_star.csv',
        }
    }
    write_json(os.path.join(rep_dir, 'evidence_pack.json'), evidence)
def save_example_outputs(
    df: pd.DataFrame,
    split_idx: List[int],
    pipeline_id: str,
    theta: Theta,
    out_dir: str,
    eval_res: int = 512,
    max_images: int = 24,
) -> None:
    """
    Guarda outputs visuales para el paper (muestra).
    """
    ensure_dir(out_dir)
    ops = PIPELINES[pipeline_id]
    sel = split_idx[:max_images]
    for i in sel:
        row = df.loc[i]
        path = row["path"]
        mod = row["modality"]
        x = read_image_any(path)
        x01 = normalize_percentiles(x, 1, 99)
        x01, _ = letterbox_to_square(x01, eval_res)
        y01, _ = apply_pipeline(x01, ops, theta)

        x8 = (np.clip(x01, 0, 1) * 255).astype(np.uint8)
        y8 = (np.clip(y01, 0, 1) * 255).astype(np.uint8)
        base = os.path.splitext(os.path.basename(path))[0]
        cv2.imwrite(os.path.join(out_dir, f"{pipeline_id}_{mod}_{base}_IN.png"), x8)
        cv2.imwrite(os.path.join(out_dir, f"{pipeline_id}_{mod}_{base}_OUT.png"), y8)


# =========================
# MAIN
# =========================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", type=str, default=None, help="Root del dataset con subcarpetas RX/MAMO/ECO")
    ap.add_argument("--manifest_csv", type=str, default=None, help="CSV con columnas path, modality")
    ap.add_argument("--out_dir", type=str, required=True, help="Directorio salida")
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--device", type=str, default="cuda", help="cuda o cpu (para IQA durante HPO)")
    ap.add_argument("--n_jobs", type=int, default=max(1, (os.cpu_count() or 8) - 2))
    ap.add_argument("--eval_res", type=int, default=512)

    # aceleración/observabilidad
    ap.add_argument("--gpu_batch", type=int, default=32, help="Batch size para IQA pyiqa en GPU")
    ap.add_argument("--opencv_threads", type=int, default=1, help="Hilos internos de OpenCV (recomendado: 1)")
    ap.add_argument("--no_progress", action="store_true", help="Desactiva barras de progreso")
    ap.add_argument("--no_cache", action="store_true", help="Desactiva cache de preprocesamiento (más lento)")
    ap.add_argument("--cache_dir", type=str, default=None, help="Directorio para cache persistente (.npz) de preprocesamiento")

    # split
    ap.add_argument("--frac_hpo", type=float, default=0.6)
    ap.add_argument("--frac_val", type=float, default=0.2)
    ap.add_argument("--frac_hold", type=float, default=0.2)

    # grid/ga/bo
    ap.add_argument("--grid_max", type=int, default=4000, help="Máx evaluaciones grid por pipeline (muestreo si excede)")
    ap.add_argument("--top_k", type=int, default=40)
    ap.add_argument("--ga_pop", type=int, default=40)
    ap.add_argument("--ga_gen", type=int, default=8)
    ap.add_argument("--bo_trials", type=int, default=60)

    # gates
    ap.add_argument("--delta_A", type=float, default=0.02)
    ap.add_argument("--delta_SNR", type=float, default=0.02)
    ap.add_argument("--tau_t_ms_1MP", type=float, default=float("nan"))
    ap.add_argument("--tau_mem_MB", type=float, default=float("nan"))

    # modos
    ap.add_argument("--deploy_cpu_only", action="store_true", help="Desactiva IQA en GPU (solo CPU)")
    ap.add_argument("--save_outputs", action="store_true", help="Guarda ejemplos IN/OUT por pipeline")
    ap.add_argument("--pipelines", type=str, default="ALL", help="ALL o lista CSV ej: P00,P10,P04")

    args = ap.parse_args()

    ensure_dir(args.out_dir)
    set_global_seeds(args.seed)
    force_cv2_threads(args.opencv_threads)

    # dispositivo
    if args.device.startswith("cuda"):
        if torch is None or not torch.cuda.is_available():
            print("[WARN] CUDA no disponible; usando CPU.")
            args.device = "cpu"

    # Cargar dataset
    if args.manifest_csv:
        df = load_manifest_csv(args.manifest_csv)
    elif args.data_root:
        df = scan_dataset(args.data_root)
    else:
        raise ValueError("Debes pasar --data_root o --manifest_csv")

    # Validar modalidades
    df["modality"] = df["modality"].astype(str).str.upper()
    # Si hay UNK, se mantiene, pero advertimos
    if (df["modality"] == "UNK").any():
        print("[WARN] Se detectó modality=UNK. Recomendado usar manifest_csv para etiquetar RX/MAMO/ECO.")

    # hashes trazabilidad
    df["sha256"] = df["path"].apply(lambda p: sha256_file(p) if os.path.isfile(p) else "")

    # splits
    splits = make_splits_stratified(df, seed=args.seed, frac_hpo=args.frac_hpo, frac_val=args.frac_val, frac_hold=args.frac_hold)
    write_json(os.path.join(args.out_dir, "splits.json"), splits)

    # config
    eval_cfg = EvalConfig(
        eval_res=args.eval_res,
        n_jobs=args.n_jobs,
        device=args.device,
        deploy_cpu_only=args.deploy_cpu_only,
        show_progress=(not args.no_progress),
        gpu_batch=int(args.gpu_batch),
        use_preproc_cache=(not args.no_cache),
        cache_dir=args.cache_dir,
    )
    gate_cfg = GateConfig(
        delta_A=float(args.delta_A),
        delta_SNR=float(args.delta_SNR),
        tau_t_ms_1MP=None if (not np.isfinite(args.tau_t_ms_1MP)) else float(args.tau_t_ms_1MP),
        tau_mem_MB=None if (not np.isfinite(args.tau_mem_MB)) else float(args.tau_mem_MB),
    )

    # pipelines a correr
    if args.pipelines.strip().upper() == "ALL":
        run_pipes = list(PIPELINES.keys())
    else:
        run_pipes = [p.strip().upper() for p in args.pipelines.split(",") if p.strip()]
    for p in run_pipes:
        if p not in PIPELINES:
            raise ValueError(f"Pipeline desconocido: {p}")

    # Estructuras salida
    theta_star_all = {}
    all_per_image = []

    # Evaluar baseline P00 (para reportes globales)
    # (Se re-evalúa dentro de cada pipeline para gates; aquí lo dejamos para panel global)
    print("[INFO] Evaluando baseline P00 en HOLD (solo para panel global)...")
    iqa_global = IQAModels(eval_cfg.device) if (not eval_cfg.deploy_cpu_only) else None
    base_hold = eval_candidate_on_split(
        df=df, split_idx=splits["HOLD"], pipeline_id="P00", theta=Theta(),
        gate_cal=None, base_block_scores=None, eval_cfg=eval_cfg, iqa=iqa_global
    )
    all_per_image.extend(base_hold["per_image"])

    # Loop por pipeline: optimizar θ* + evaluación final
    for pid in run_pipes:
        print(f"\n[PIPE] {pid} ops={PIPELINES[pid]}")

        if pid == "P00":
            theta_star_all[pid] = Theta().as_dict()
            continue

        # 1) GRID + baseline medians
        top_grid, base_medians, base_block_scores = run_grid(
            df=df, split=splits, pipeline_id=pid, eval_cfg=eval_cfg, gate_cfg=gate_cfg,
            top_k=args.top_k, grid_max=args.grid_max, seed=args.seed
        )

        # 2) GA NSGA-II desde top_grid
        ga_best = run_nsga2(
            df=df, split=splits, pipeline_id=pid, seed=args.seed,
            eval_cfg=eval_cfg, gate_cfg=gate_cfg, base_medians=base_medians,
            base_block_scores=base_block_scores, init_top=top_grid,
            pop_size=args.ga_pop, generations=args.ga_gen
        )

        # 3) BO Optuna/TPE warmstart con GA
        bo_best = run_optuna_bo(
            df=df, split=splits, pipeline_id=pid, seed=args.seed,
            eval_cfg=eval_cfg, gate_cfg=gate_cfg, base_medians=base_medians,
            base_block_scores=base_block_scores, warmstart=ga_best,
            trials=args.bo_trials
        )

        # pool candidatos para validación: top grid + top ga + top bo (únicos por theta)
        cand_pool = []
        seen = set()

        def add_pool(lst):
            nonlocal cand_pool, seen
            for x in lst:
                th = x["theta"]
                key = json.dumps(th, sort_keys=True)
                if key not in seen:
                    seen.add(key)
                    cand_pool.append({"theta": th})

        add_pool(top_grid)
        add_pool(ga_best)
        add_pool(bo_best)

        # Selección θ* en VALIDACIÓN (Rev-4)
        best_val = pick_theta_star_on_validation(
            df=df, split=splits, pipeline_id=pid, candidates=cand_pool,
            base_medians=base_medians, base_block_scores=base_block_scores,
            eval_cfg=eval_cfg, gate_cfg=gate_cfg
        )
        th_star = Theta(**best_val["theta"])
        theta_star_all[pid] = th_star.as_dict()

        # Evaluación final HOLD-OUT (sin retocar θ ni gates)
        final = eval_final_holdout(
            df=df, split=splits, pipeline_id=pid, theta_star=th_star,
            base_medians=base_medians, base_block_scores=base_block_scores,
            eval_cfg=eval_cfg, gate_cfg=gate_cfg
        )
        all_per_image.extend(final["per_image"])

        # Guardar ejemplos visuales (opcional)
        if args.save_outputs:
            out_vis = os.path.join(args.out_dir, "outputs_examples", pid)
            save_example_outputs(df, splits["HOLD"], pid, th_star, out_vis, eval_res=args.eval_res, max_images=24)

        # Guardar resumen por pipeline
        write_json(os.path.join(args.out_dir, f"{pid}_theta_star.json"), th_star.as_dict())
        write_json(os.path.join(args.out_dir, f"{pid}_val_selection.json"), best_val)

        print(f"[OK] θ*({pid}) = {th_star.as_dict()}  | val_score={best_val.get('score', None)}  ok={best_val.get('ok', None)}")

    # Guardar θ* global por pipeline
    write_json(os.path.join(args.out_dir, "theta_star_all_pipelines.json"), theta_star_all)

    # CSV por imagen
    per_image_df = pd.DataFrame(all_per_image)
    per_image_df.to_csv(os.path.join(args.out_dir, "per_image_metrics_holdout.csv"), index=False, encoding="utf-8")

    # Estadísticos + tests
    report_stats(per_image_df, out_dir=args.out_dir, tag="holdout")

    # Panel agregado por pipeline (medianas globales)
    metrics = [m for m, _, _ in PRIMARY_METRICS + SECONDARY_METRICS]
    agg = per_image_df.groupby(["pipeline"]).agg({m: "median" for m in metrics if m in per_image_df.columns}).reset_index()
    agg.to_csv(os.path.join(args.out_dir, "aggregate_medians_by_pipeline.csv"), index=False, encoding="utf-8")

    # Evidencia adicional (Tablas/Figuras/JSON) para poblar la plantilla D1
    try:
        generate_d1_evidence_pack(df=df, splits=splits, theta_star_all=theta_star_all, per_image_df=per_image_df, out_dir=args.out_dir)
    except Exception as e:
        print(f"[WARN] No pude generar evidence_pack: {e!r}")

    print("\n[FIN] Reportes listos en:", args.out_dir)
    print(" - theta_star_all_pipelines.json")
    print(" - per_image_metrics_holdout.csv")
    print(" - stats_holdout.csv (+ wilcoxon_P10_vs_P00_holdout.csv si aplica)")
    print(" - aggregate_medians_by_pipeline.csv")


if __name__ == "__main__":
    main()