from flask import Flask, render_template, request, jsonify
import numpy as np
import plotly.express as px
import plotly.io as pio
import json
from err_calc import compute_err, compute_hamming_histogram, compute_float_histogram

app = Flask(__name__)

import plotly.graph_objects as go
import numpy as np, json, plotly.io as pio

def compute_graph(params: dict):
    # ① Lee las opciones del usuario (usa defaults si faltan)
    m1 = float(params.get("m1", 1))
    b1 = float(params.get("b1", 0))
    m2 = float(params.get("m2", -1))
    b2 = float(params.get("b2", 5))

    # ② Calcula las rectas
    x = np.linspace(-10, 10, 400)
    y1 = m1 * x + b1
    y2 = m2 * x + b2

    # ③ Figura y trazas
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=y1, mode="lines",
                             name=f"y = {m1}·x + {b1}"))
    fig.add_trace(go.Scatter(x=x, y=y2, mode="lines",
                             name=f"y = {m2}·x + {b2}"))

    # ④ Intersección (si las pendientes no son iguales)
    if m1 != m2:
        xi = (b2 - b1) / (m1 - m2)
        yi = m1 * xi + b1
        fig.add_trace(go.Scatter(x=[xi], y=[yi], mode="markers",
                                 marker=dict(size=10),
                                 name=f"Intersección ({xi:.2f}, {yi:.2f})"))
        fig.update_layout(title="Dos rectas y su punto de corte")
    else:
        fig.update_layout(title="Rectas paralelas (sin corte)")

    fig.update_layout(margin=dict(l=20, r=20, t=60, b=30),
                      xaxis_title="x", yaxis_title="y")
    return json.loads(pio.to_json(fig))


@app.route("/api/err", methods=["POST"])
def api_err():
    params = request.json
    model = int(params['model'])        # p.ej. 512
    bits  = int(params['bits'])         # 3 o 4
    neural = params['neural']
    # mapear umbrales:
    if bits == 3:
        t1 = float(params['t1'])
        t2 = None
    else:
        # aquí asumimos params['t2']=lower, params['t3']=upper
        t1 = float(params['t2'])
        t2 = float(params['t3'])
    if neural == 'insightface':
        dataset_dir = "../embeddings512_float_LFW_insightface"
    else:
        dataset_dir = f"../embeddings{model}_float_LFW"
    res = compute_err(
        dataset_dir, model, bits,
        t1=t1, t2=t2,
        save_plot=True,
        output_dir="static/plots"
    )
    return jsonify(res)

@app.route('/api/histogram', methods=['POST'])
def api_histogram():
    params = request.json
    model = int(params['model'])
    bits  = int(params['bits'])
    neural = params['neural']

    if bits == 3:
        t1 = float(params['t1'])
        t2 = None
    else:
        t1 = float(params['t2'])
        t2 = float(params['t3'])

    
    if neural == 'insightface':
        dataset_dir = "../embeddings512_float_LFW_insightface"
    else:
        dataset_dir = f"../embeddings{model}_float_LFW"
    res = compute_hamming_histogram(dataset_dir, model, bits, t1=t1, t2=t2)
    return jsonify(res)

@app.route('/api/histogram_float', methods=['POST'])
def api_histograma_float():
    params = request.json
    model = int(params['model'])
    neural = params['neural']
    n_parts = int(params['n_parts'])
    
    if neural == 'insightface':
        dataset_dir = "../embeddings512_float_LFW_insightface"
    else:
        dataset_dir = f"../embeddings{model}_float_LFW"
    if model == 128:
        (vmin, vmax) = (-0.42, 0.42)
        res = compute_float_histogram(dataset_dir, model, vmin=vmin, vmax=vmax, n_parts=n_parts)
    else:
        res = compute_float_histogram(dataset_dir, model, n_parts=n_parts)
    return jsonify(res)

@app.route("/")
def index():
    return render_template("index.html")   # formulario + contenedor de la gráfica

if __name__ == "__main__":
    app.run(ssl_context=("ssl/cert.pem", "ssl/key.pem"))

