/* static/app.js */
document.addEventListener('DOMContentLoaded', () => {
  const form       = document.getElementById('opts');
  const bitsSelect = document.getElementById('bits-select');
  const th3Block   = document.getElementById('thresholds-3');
  const th4Block   = document.getElementById('thresholds-4');
  const resultsDiv = document.getElementById('results');
  const btnErr     = document.getElementById('boton-err');
  const btnHamming = document.getElementById('boton-hamming');
  const tablaSelect = document.querySelector('select[name="tabla"]');
  const floatSplitBlock = document.getElementById('float-split-block');

  // Mostrar/ocultar umbrales…
  function toggleThresholdInputs() {
    const is3 = bitsSelect.value === '3';
    th3Block.classList.toggle('d-none', !is3);
    th4Block.classList.toggle('d-none',  is3);
    th3Block.querySelector('input[name="t1"]').required = is3;
    th4Block.querySelectorAll('input').forEach(i => i.required = !is3);
  }
  bitsSelect.addEventListener('change', toggleThresholdInputs);
  toggleThresholdInputs();

  function toggleFloatSplitBlock() {
    floatSplitBlock.classList.toggle('d-none', tablaSelect.value !== 'histogram_float');
  }
  tablaSelect.addEventListener('change', toggleFloatSplitBlock);
  toggleFloatSplitBlock();

  // Construir payload incluyendo 'neural' y forzando model si es insightFace
  function construirPayload() {
    const fd      = new FormData(form);
    let model     = fd.get('model');
    const bits    = fd.get('bits');
    const neural  = fd.get('neural');

    // Si insightFace → model = 512
    if (neural === 'insightface') {
      model = '512';
    }

    const payload = {
      model:  model,
      bits:   bits,
      neural: neural
    };

    if (bits === '3') {
      payload.t1 = fd.get('t1');
    } else {
      payload.t2 = fd.get('t2');
      payload.t3 = fd.get('t3');
    }
    if (tablaSelect.value === 'histogram_float') {
      payload.n_parts = fd.get('n_parts');
    }
    console.log(payload);
    return payload;
  }

  // Envío genérico sin borrar resultados previos
  async function enviarPayload(payload, url) {
    const spinner = document.createElement('div');
    spinner.innerHTML = `
      <div class="text-center my-3">
        <div class="spinner-border" role="status"></div>
        <p class="mt-2">Procesando…</p>
      </div>`;
    resultsDiv.prepend(spinner);

    try {
      const resp = await fetch(url, {
        method:  'POST',
        headers: { 'Content-Type': 'application/json' },
        body:    JSON.stringify(payload)
      });
      if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
      const json = await resp.json();

      const wrapper = document.createElement('div');
      wrapper.className = 'my-4';
      resultsDiv.prepend(wrapper);

      const fig = json.plot ?? json.hist;
      if (fig) {
        Plotly.newPlot(wrapper, fig.data, fig.layout, { responsive: true });
      } else {
        wrapper.innerHTML = `
          <div class="alert alert-warning" role="alert">
            No se ha recibido ningún gráfico.
          </div>`;
      }
    } catch (err) {
      const alert = document.createElement('div');
      alert.className = 'alert alert-danger';
      alert.role      = 'alert';
      alert.innerText = `⚠️ Error: ${err.message}`;
      resultsDiv.prepend(alert);
      console.error(err);
    } finally {
      spinner.remove();
    }
  }

  // Botón “Calcular”
  btnErr.addEventListener('click', async evt => {
    evt.preventDefault();
    const fd    = new FormData(form);
    const url   = fd.get('tabla');            // "err" o "histogram"
    const payload = construirPayload();
    await enviarPayload(payload, `/api/${url}`);
  });

  // Botón “Borrar tablas”
  btnHamming.addEventListener('click', evt => {
    evt.preventDefault();
    resultsDiv.innerHTML = '';
  });
});
