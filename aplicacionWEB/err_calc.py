# err_module.py

import os
import random
import itertools
import numpy as np
import plotly.graph_objs as go
from scipy.stats import norm

# --------------------------------------------------
# 🔧 Helpers de binarización
# --------------------------------------------------
def _binarize_3bits(emb: np.ndarray, th: float) -> np.ndarray:
    out = []
    for v in emb:
        if v <= -th:
            out.extend([0,0,0])
        elif v <= 0:
            out.extend([0,0,1])
        elif v <= th:
            out.extend([0,1,1])
        else:
            out.extend([1,1,1])
    return np.array(out, dtype=np.uint8)

def _binarize_4bits(emb: np.ndarray, th_low: float, th_high: float) -> np.ndarray:
    out = []
    for v in emb:
        if v <= -th_high:
            out.extend([0,0,0,0])
        elif v <= -th_low:
            out.extend([0,0,0,1])
        elif v <= th_low:
            out.extend([0,0,1,1])
        elif v <= th_high:
            out.extend([0,1,1,1])
        else:
            out.extend([1,1,1,1])
    return np.array(out, dtype=np.uint8)

# --------------------------------------------------
# 🔃 Carga y binarización en memoria
# --------------------------------------------------
def load_float_embeddings(dataset_dir: str,
                          float_dim: int) -> dict[str, np.ndarray]:
    """Carga todos los .npy de `dataset_dir` con dimensión float_dim."""
    data = {}
    for fn in os.listdir(dataset_dir):
        if not fn.endswith(".npy"):
            continue
        arr = np.load(os.path.join(dataset_dir, fn))
        if arr.ndim == 2 and arr.shape[1] == float_dim:
            data[fn] = arr
    return data

def binarize_all(data_f: dict[str, np.ndarray],
                 bits: int,
                 t1: float=None,
                 t2: float=None) -> dict[str, np.ndarray]:
    """
    Binariza todos los embeddings según bits=3|4.
      - bits=3: usa t1
      - bits=4: usa t1 (lower) y t2 (higher)
    """
    out = {}
    for name, mat in data_f.items():
        bins = []
        for emb in mat:
            if bits == 3:
                if t1 is None:
                    raise ValueError("Para 3 bits necesitas t1")
                bins.append(_binarize_3bits(emb, t1))
            else:
                if t1 is None or t2 is None:
                    raise ValueError("Para 4 bits necesitas t2 (umbral bajo) y t3 (umbral alto)")
                # aquí mapeamos t1→th_low, t2→th_high
                bins.append(_binarize_4bits(emb, t1, t2))
        out[name] = np.stack(bins, axis=0)
    return out

# --------------------------------------------------
# ⚙️ Generación de pares y distancia
# --------------------------------------------------
def _hamming(a: np.ndarray, b: np.ndarray) -> int:
    return int((a != b).sum())

def generate_pairs(data_b: dict[str, np.ndarray],
                   num_identities=4000,
                   num_impostor_pairs=100_000):
    """Devuelve (genuinos, impostores) donde cada lista es [(emb1, emb2), ...]."""
    # genuinos
    personas = [k for k,v in data_b.items() if v.shape[0] >= 2]
    random.shuffle(personas)
    muest = personas[:min(num_identities, len(personas))]
    genuinos = []
    for p in muest:
        m = data_b[p]
        for i,j in itertools.combinations(range(m.shape[0]), 2):
            genuinos.append((m[i], m[j]))
    # impostores
    impostores = []
    claves = list(data_b.keys())
    for _ in range(num_impostor_pairs):
        p1, p2 = random.sample(claves, 2)
        e1 = data_b[p1][random.randrange(data_b[p1].shape[0])]
        e2 = data_b[p2][random.randrange(data_b[p2].shape[0])]
        impostores.append((e1, e2))
    return genuinos, impostores

def _calc_distances(pares: list[tuple[np.ndarray,np.ndarray]]) -> list[int]:
    return [ _hamming(a,b) for a,b in pares ]

# --------------------------------------------------
# 📊 Evaluación de umbrales y EER
# --------------------------------------------------
def evaluate_thresholds(dist_g: list[int],
                        dist_i: list[int],
                        ths: list[int]) -> tuple[list[float], list[float]]:
    total_g = len(dist_g)
    total_i = len(dist_i)
    fars, frrs = [], []
    for th in ths:
        fars.append( sum(d <= th for d in dist_i) / total_i * 100 )
        frrs.append( sum(d >  th for d in dist_g) / total_g * 100 )
    return fars, frrs

def find_eer(ths: list[int],
             fars: list[float],
             frrs: list[float]) -> tuple[int, float]:
    best, eer, t_eer = float("inf"), 0.0, ths[0]
    for t,f,fr in zip(ths, fars, frrs):
        d = abs(f - fr)
        if d < best:
            best, eer, t_eer = d, (f+fr)/2, t
    return t_eer, eer

# --------------------------------------------------
# 🚀 Función pública
# --------------------------------------------------
def compute_err(
    dataset_dir: str,
    float_dim: int,
    bits: int,
    *,
    t1: float | None = None,
    t2: float | None = None,
    num_identities: int = 4_000,
    num_impostor_pairs: int = 100_000,
    save_plot: bool = False,
    output_dir: str | None = None,
) -> dict:
    """
    Calcula FAR, FRR y EER sobre embeddings binarizados y devuelve un JSON-safe dict.

    Parámetros
    ----------
    dataset_dir        Carpeta con los .npy de embeddings en punto flotante.
    float_dim          Dimensión (en float) del embedding original (128, 512…).
    bits               3 ó 4 (número de bits por componente tras binarizar).
    t1, t2             Umbrales de binarización:
                         · bits==3 ➜ t1 (único umbral)
                         · bits==4 ➜ t1=umbral bajo, t2=umbral alto
    num_identities     Personas distintas muestreadas para pares genuinos.
    num_impostor_pairs Número total de pares impostores aleatorios.
    save_plot          Si True, guarda la gráfica como PNG en output_dir.
    output_dir         Carpeta donde escribir la imagen (requerida si save_plot).

    Devuelve
    --------
    dict JSON-serializable con:
      thresholds  Lista de umbrales evaluados (enteros)
      fars        Lista FAR (%) por umbral
      frrs        Lista FRR (%) por umbral
      eer         {'threshold': int, 'value': float}
      plot        Estructura Plotly completa (fig.to_dict())
      plot_path   Ruta del PNG (solo si save_plot=True)
    """

    # 1) Cargar y binarizar
    data_f = load_float_embeddings(dataset_dir, float_dim)
    data_b = binarize_all(data_f, bits, t1=t1, t2=t2)

    # 2) Generar pares y distancias
    genuinos, impostores = generate_pairs(
        data_b, num_identities, num_impostor_pairs
    )
    dist_g = _calc_distances(genuinos)
    dist_i = _calc_distances(impostores)

    # 3) FAR / FRR / EER
    bin_length = next(iter(data_b.values())).shape[1]
    thresholds = list(range(0, bin_length + 1))
    fars, frrs = evaluate_thresholds(dist_g, dist_i, thresholds)
    eer_th, eer_val = find_eer(thresholds, fars, frrs)

    # 4) Gráfica Plotly
    traces = [
        go.Scatter(x=thresholds, y=fars, name="FAR", mode="lines"),
        go.Scatter(x=thresholds, y=frrs, name="FRR", mode="lines"),
        go.Scatter(
            x=[eer_th],
            y=[eer_val],
            name="EER",
            mode="markers+text",
            text=[f"{eer_val:.2f}%"],
            textposition="top right",
        ),
    ]

    layout = go.Layout(
        title="FAR vs FRR con EER",
        xaxis=dict(title="Umbral de Hamming"),
        yaxis=dict(title="Tasa (%)"),
        shapes=[
            dict(
                type="line",
                x0=eer_th,
                x1=eer_th,
                y0=0,
                y1=max(max(fars), max(frrs)),
                line=dict(dash="dash"),
            )
        ],
    )

    fig = go.Figure(data=traces, layout=layout)
    plot_dict = fig.to_dict()  # ← 100 % JSON-serializable

    # 5) Guardar PNG si procede
    plot_path = None
    if save_plot:
        if output_dir is None:
            raise ValueError("Si save_plot=True debes indicar output_dir")
        os.makedirs(output_dir, exist_ok=True)
        fname = f"err_dim{float_dim}_{bits}bits"
        if bits == 3:
            fname += f"_t1{t1}"
        else:
            fname += f"_t2{t1}_t3{t2}"
        fname += ".png"
        fig.write_image(os.path.join(output_dir, fname))
        plot_path = os.path.join(output_dir, fname)

    # 6) Empaquetar resultado
    return {
        "thresholds": thresholds,
        "fars": fars,
        "frrs": frrs,
        "eer": {"threshold": eer_th, "value": eer_val},
        "plot": plot_dict,
        **({"plot_path": plot_path} if plot_path else {}),
    }

def compute_hamming_histogram(
    dataset_dir: str,
    float_dim: int,
    bits: int,
    *,
    t1: float | None = None,
    t2: float | None = None,
) -> dict:
    """
    Calcula el histograma del peso Hamming de la primera muestra
    de cada individuo tras binarizar.

    Devuelve un dict JSON-serializable con:
      - weights: lista de pesos individuales
      - hist: figura Plotly (fig.to_dict())
    """
    # 1) Cargar y binarizar
    data_f = load_float_embeddings(dataset_dir, float_dim)
    data_b = binarize_all(data_f, bits, t1=t1, t2=t2)

    # 2) Extraer peso Hamming de la primera muestra de cada individuo
    weights = [int(arr[0].sum()) for arr in data_b.values()]

    # 3) Crear histograma con Plotly
    hist_trace = go.Histogram(x=weights, nbinsx=max(weights) + 1)
    layout = go.Layout(
        title="Histograma de peso Hamming (primera muestra)",
        xaxis=dict(title="Peso Hamming"),
        yaxis=dict(title="Número de individuos")
    )
    fig = go.Figure(data=[hist_trace], layout=layout)

    return {"weights": weights, "plot": fig.to_dict()}


def compute_float_histogram(
    dataset_dir: str,
    float_dim: int,
    *,
    vmin: float = -0.23,
    vmax: float = 0.23,
    decimals: int = 4,
    bin_width: float | None = None,
    n_parts: int = 4,       
) -> dict:
    """
    Histograma de los valores float + líneas que lo dividen en n_parts franjas
    con el mismo nº de observaciones.

    Devuelve:
      - values ......... lista con los valores usados
      - cuts ........... lista de puntos de corte (longitud n_parts-1)
      - plot ........... figura Plotly serializada (fig.to_dict())
    """
    # 1) Cargar y aplanar
    data_f = load_float_embeddings(dataset_dir, float_dim)      # dict[str, np.ndarray]
    arrays = [arr.ravel() for arr in data_f.values() if arr.size]
    if not arrays:
        raise ValueError("No se encontraron embeddings con esa dimensión.")
    all_vals = np.concatenate(arrays)

    # 2) Filtrar por rango
    mask = (all_vals >= vmin) & (all_vals <= vmax)
    all_vals = all_vals[mask]

    # 3) Histograma
    if bin_width is not None:
        xbins = dict(start=vmin, end=vmax, size=bin_width)
        trace = go.Histogram(x=all_vals, xbins=xbins, name="Datos")
        title = f"Histograma [{vmin}, {vmax}] (bin_width={bin_width})"
    else:
        rounded = np.round(all_vals, decimals=decimals)
        uniques, counts = np.unique(rounded, return_counts=True)
        trace = go.Bar(x=uniques.tolist(), y=counts.tolist(), name="Datos")
        title = f"Histograma redondeado a {decimals} decimales"

    fig = go.Figure(data=[trace])
    fig.update_layout(
        title=title,
        xaxis_title="Valor",
        yaxis_title="Frecuencia",
    )

    # 4) Cortes que dejan partes iguales
    if n_parts < 2:
        raise ValueError("n_parts debe ser ≥ 2.")
    probs = np.linspace(0, 100, n_parts + 1)[1:-1]        # p.ej. 20,40,60,80 para quintiles
    cuts = np.percentile(all_vals, probs)

    # 5) Añadir líneas verticales en los cortes
    for c in cuts:
        fig.add_shape(
            type="line",
            x0=c, x1=c,
            yref="paper", y0=0, y1=1,               # de 0 % a 100 % del eje Y
            line=dict(color="red", width=2, dash="dash"),
        )

        fig.add_annotation(
            x=c,
            y=1.02,
            yref="paper",
            showarrow=False,
            text=f"{c:.4f}",
            font=dict(color="red"),
        )
    return {
        "values": all_vals.tolist(),
        "cuts": cuts.tolist(),
        "plot": fig.to_dict(),
    }

