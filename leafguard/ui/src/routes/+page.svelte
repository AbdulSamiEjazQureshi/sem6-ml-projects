<script>
  import { onMount } from 'svelte';

  const api = 'http://127.0.0.1:8022';
  const diseaseNotes = {
    healthy: {
      title: 'Healthy',
      copy: 'Mostly green surface, low spots, low dark lesion signal.',
      tone: 'safe'
    },
    rust: {
      title: 'Rust',
      copy: 'Orange-brown speckles. Often raises brown, orange, and spot features.',
      tone: 'risk'
    },
    blight: {
      title: 'Blight',
      copy: 'Large dark damaged regions. Often raises dark tissue and edge signals.',
      tone: 'risk'
    },
    leaf_spot: {
      title: 'Leaf Spot',
      copy: 'Small isolated spots. Often raises spot ratio more than broad darkness.',
      tone: 'risk'
    },
    senescent_leaf: {
      title: 'Aging / Autumn Leaf',
      copy: 'Yellow-orange aging pattern. This is not treated as infectious disease.',
      tone: 'watch'
    }
  };
  const featureLabels = {
    green_ratio: 'Green surface',
    brown_ratio: 'Brown patches',
    yellow_ratio: 'Yellowing',
    orange_ratio: 'Orange tone',
    dark_ratio: 'Dark damage',
    spot_ratio: 'Spot marks',
    brightness: 'Brightness',
    edge_density: 'Texture edges'
  };
  const solutionNotes = {
    healthy: ['No treatment needed.', 'Keep normal watering and sunlight.', 'Monitor weekly for new spots.'],
    rust: ['Remove heavily infected leaves.', 'Avoid overhead watering.', 'Use an appropriate fungicide if rust spreads.'],
    blight: ['Remove damaged leaves and isolate the plant.', 'Improve air circulation.', 'Avoid wet foliage and consider fungicide guidance.'],
    leaf_spot: ['Prune affected leaves.', 'Keep leaves dry during watering.', 'Disinfect tools after cutting infected tissue.'],
    senescent_leaf: ['No disease treatment needed.', 'This can be normal seasonal aging.', 'Check younger leaves before applying chemicals.']
  };

  let path = 'data/images/rust/rust_00.bmp';
  let result = null;
  let experiments = [];
  let samples = [];
  let error = '';
  let preview = `${api}/image?path=${encodeURIComponent(path)}`;
  let externalMode = false;
  let pendingImageSrc = '';
  let pendingFeatures = null;
  let pendingName = '';
  let status = 'Pick a sample leaf or upload a photo to begin.';
  let toast = '';
  let selectedClass = 'all';

  $: visibleSamples =
    selectedClass === 'all' ? samples : samples.filter((sample) => sample.label === selectedClass);
  $: sampleClasses = ['all', ...Array.from(new Set(samples.map((sample) => sample.label)))];
  $: diagnosis = result ? diseaseNotes[result.label] || { title: result.label, copy: result.human_summary, tone: 'watch' } : null;

  async function loadLists() {
    try {
      experiments = await (await fetch(`${api}/experiments`)).json();
      samples = await (await fetch(`${api}/samples`)).json();
    } catch (err) {
      error = 'Start the project with make run, then refresh this page.';
    }
  }

  async function diagnoseSample(samplePath = path) {
    error = '';
    try {
      const response = await fetch(`${api}/predict-image`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ path: samplePath })
      });
      if (!response.ok) throw new Error(await response.text());
      result = await response.json();
      preview = `${api}/image?path=${encodeURIComponent(samplePath)}`;
      externalMode = false;
      status = `Diagnosed selected sample.`;
      await loadLists();
    } catch (err) {
      error = 'Could not diagnose that sample. Try another card or upload a photo.';
    }
  }

  async function uploadImage(event) {
    const file = event.target.files?.[0];
    if (!file) return;
    error = '';
    result = null;
    status = `Image selected: ${file.name}. Press Run diagnosis.`;
    const reader = new FileReader();
    reader.onload = () => {
      preview = reader.result;
      pendingImageSrc = reader.result;
      pendingFeatures = null;
      pendingName = file.name;
      externalMode = true;
      toast = 'Image selected. Click Run diagnosis.';
      window.setTimeout(() => {
        toast = '';
      }, 2600);
    };
    reader.readAsDataURL(file);
  }

  async function runDiagnosis() {
    if (!pendingImageSrc && !pendingFeatures) {
      error = 'Upload an image or click a sample first.';
      return;
    }
    error = '';
    status = 'Running diagnosis...';
    if (!pendingFeatures && pendingImageSrc) {
      pendingFeatures = await extractFeaturesFromSource(pendingImageSrc);
    }
    const response = await fetch(`${api}/predict`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ features: pendingFeatures })
    });
    if (!response.ok) {
      error = 'The uploaded image could not be diagnosed.';
      status = 'Diagnosis failed.';
      return;
    }
    result = await response.json();
    status = `Diagnosed external image: ${pendingName || 'uploaded leaf'}`;
    await loadLists();
  }

  function extractFeaturesFromSource(src) {
    return new Promise((resolve, reject) => {
      const image = new Image();
      image.onload = () => resolve(extractCanvasFeatures(image));
      image.onerror = () => reject(new Error('Could not read uploaded image.'));
      image.src = src;
    });
  }

  function extractCanvasFeatures(image) {
    const size = 96;
    const canvas = document.createElement('canvas');
    canvas.width = size;
    canvas.height = size;
    const context = canvas.getContext('2d', { willReadFrequently: true });
    context.drawImage(image, 0, 0, size, size);
    const { data } = context.getImageData(0, 0, size, size);
    const brightness = [];
    let totalLeaf = 0;
    let green = 0;
    let brown = 0;
    let yellow = 0;
    let orange = 0;
    let dark = 0;
    let spots = 0;
    let brightnessSum = 0;
    let edgeSum = 0;
    for (let y = 0; y < size; y += 1) {
      brightness[y] = [];
      for (let x = 0; x < size; x += 1) {
        const index = (y * size + x) * 4;
        const red = data[index];
        const g = data[index + 1];
        const blue = data[index + 2];
        const bright = (red + g + blue) / 3;
        brightness[y][x] = bright;
        if (red > 210 && g > 205 && blue > 185) continue;
        totalLeaf += 1;
        brightnessSum += bright;
        if (g > red * 1.25 && g > blue * 1.35) green += 1;
        if (red > g * 1.25 && g > blue * 1.15) brown += 1;
        if (red > 120 && g > 105 && blue < 75) yellow += 1;
        if (red > 145 && g > 80 && g < 165 && blue < 85) orange += 1;
        if (bright < 70) dark += 1;
        if ((red < 85 && g < 85 && blue < 70) || (red > 115 && g < 100 && blue < 70)) spots += 1;
      }
    }
    for (let y = 0; y < size; y += 1) {
      for (let x = 0; x < size; x += 1) {
        if (x + 1 < size) edgeSum += Math.abs(brightness[y][x] - brightness[y][x + 1]);
        if (y + 1 < size) edgeSum += Math.abs(brightness[y][x] - brightness[y + 1][x]);
      }
    }
    const denom = totalLeaf || 1;
    return {
      green_ratio: green / denom,
      brown_ratio: brown / denom,
      yellow_ratio: yellow / denom,
      orange_ratio: orange / denom,
      dark_ratio: dark / denom,
      spot_ratio: spots / denom,
      brightness: brightnessSum / denom,
      edge_density: edgeSum / (denom * 255)
    };
  }

  async function selectSample(sample) {
    path = sample.path;
    preview = `${api}/image?path=${encodeURIComponent(sample.path)}`;
    externalMode = false;
    pendingImageSrc = '';
    pendingFeatures = null;
    pendingName = '';
    toast = `${formatLabel(sample.label)} sample selected`;
    status = `Running diagnosis for ${formatLabel(sample.label)} sample...`;
    await diagnoseSample(sample.path);
    window.setTimeout(() => {
      toast = '';
    }, 2600);
  }

  async function regenerateSamples() {
    error = '';
    status = 'Generating fresh sample leaves and retraining models...';
    try {
      const response = await fetch(`${api}/regenerate`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: '{}'
      });
      if (!response.ok) throw new Error(await response.text());
      const body = await response.json();
      await loadLists();
      status = `Generated ${body.samples} sample images and retrained both models.`;
    } catch (err) {
      error = 'Could not regenerate samples. Make sure the API is running.';
    }
  }

  function formatLabel(value) {
    return value.replaceAll('_', ' ');
  }

  function featurePercent(value, name) {
    const scaled = name === 'brightness' ? Number(value) / 1.8 : Number(value) * 100;
    return Math.max(3, Math.min(100, scaled));
  }

  function featureText(value, name) {
    if (name === 'brightness') return `${Math.round(value)}/255`;
    return `${Math.round(Number(value) * 100)}%`;
  }

  onMount(async () => {
    await loadLists();
  });
</script>

<svelte:head>
  <title>LeafGuard Diagnosis Assistant</title>
</svelte:head>

<main class="page">
  {#if toast}
    <div class="toast">{toast}</div>
  {/if}

  <section class="hero">
    <div>
      <p class="eyebrow">LeafGuard field clinic</p>
      <h1>Leaf diagnosis, guided like a field visit.</h1>
      <p>
        Upload a leaf photo or click a generated sample. LeafGuard checks color, spots, dark tissue,
        and texture, then compares Nearest Centroid with KNN neighbor voting.
      </p>
    </div>
    <div class="hero-card">
      <span>Sample library</span>
      <strong>{samples.length || 60}</strong>
      <small>sample leaves + external upload</small>
    </div>
  </section>

  <section class="steps" aria-label="How to use LeafGuard">
    <article>
      <span>01</span>
      <b>Choose a leaf</b>
      <p>Upload JPG/PNG/BMP or click any generated sample below.</p>
    </article>
    <article>
      <span>02</span>
      <b>Run diagnosis</b>
      <p>The app extracts visible features and sends them to the ML API.</p>
    </article>
    <article>
      <span>03</span>
      <b>Read the result</b>
      <p>See disease status, KNN vote, tree identity status, and feature charts.</p>
    </article>
  </section>

  <section class="workspace">
    <aside class="input-panel">
      <div class="panel-heading">
        <p class="eyebrow">Input</p>
        <h2>Start here</h2>
      </div>

      <label class="dropzone">
        <input type="file" accept="image/*" on:change={uploadImage} />
        <b>Upload leaf image</b>
        <span>JPG, PNG, BMP. Preview first, then run diagnosis.</span>
      </label>

      <div class="quick-actions">
        <button class="primary" on:click={runDiagnosis} disabled={!pendingImageSrc && !pendingFeatures}>
          {result && externalMode ? 'Re-run diagnosis' : 'Run diagnosis'}
        </button>
        <button class="secondary" on:click={regenerateSamples}>Generate samples</button>
      </div>

      <p class="status">{status}</p>
      {#if error}<p class="error">{error}</p>{/if}
    </aside>

    <section class="image-panel">
      <div class="preview-frame">
        <img src={preview} alt="Selected leaf preview" />
      </div>
      <div class="source-pill">{externalMode ? 'External upload' : 'Generated sample'}</div>
    </section>

    <section class="result-panel" class:empty={!result}>
      {#if result && diagnosis}
        <p class="eyebrow">Diagnosis receipt</p>
        <div class={`diagnosis ${diagnosis.tone}`}>
          <span>{result.is_disease ? 'Disease likely' : 'No infectious disease detected'}</span>
          <strong>{diagnosis.title}</strong>
          <small>{Math.round(result.confidence * 100)}% KNN confidence</small>
        </div>

        <p class="plain">{result.human_summary}</p>

        <div class="leaf-profile">
          <article>
            <span>Leaf source</span>
            <b>{externalMode ? 'Uploaded photo' : 'Generated sample'}</b>
          </article>
          <article>
            <span>Leaf condition</span>
            <b>{result.leaf_condition}</b>
          </article>
          <article>
            <span>Disease status</span>
            <b>{result.is_disease ? 'Disease pattern detected' : 'No disease pattern detected'}</b>
          </article>
          <article>
            <span>Tree/species</span>
            <b>{result.tree_identity.label || 'unknown'}</b>
          </article>
        </div>

        <div class="solution-card">
          <b>Suggested next steps</b>
          <ul>
            {#each solutionNotes[result.label] || ['Use a real Kaggle-trained model before field decisions.'] as item}
              <li>{item}</li>
            {/each}
          </ul>
        </div>

        <div class="identity-card">
          <b>Tree identity</b>
          <span>{result.tree_identity.message}</span>
        </div>

        {#if result.knn}
          <div class="algorithm-cards">
            <article>
              <span>Nearest Centroid</span>
              <b>{formatLabel(result.centroid.label)}</b>
              <small>closest average disease profile</small>
            </article>
            <article>
              <span>KNN vote</span>
              <b>{formatLabel(result.knn.label)}</b>
              <small>{Math.round(result.knn.confidence * 100)}% neighbor agreement</small>
            </article>
          </div>
          <p class="agreement">{result.model_agreement ? 'Both algorithms agree on this case.' : 'The algorithms disagree; show this as a review case.'}</p>
        {/if}
      {:else}
        <p class="eyebrow">Waiting</p>
        <h2>No diagnosis yet</h2>
        <p>Upload a leaf or click a sample. Results will appear here with a plain-English explanation.</p>
      {/if}
    </section>
  </section>

  {#if result}
    <section class="charts">
      <div class="chart-card">
        <div class="section-title">
          <p class="eyebrow">Feature chart</p>
          <h2>What the model saw</h2>
        </div>
        <div class="feature-list">
          {#each Object.entries(result.features) as [name, value]}
            <article>
              <div>
                <b>{featureLabels[name] || formatLabel(name)}</b>
                <span>{featureText(value, name)}</span>
              </div>
              <div class="meter"><i style={`width: ${featurePercent(value, name)}%`}></i></div>
            </article>
          {/each}
        </div>
      </div>

      {#if result.knn}
        <div class="chart-card">
          <div class="section-title">
            <p class="eyebrow">KNN chart</p>
            <h2>Closest training leaves</h2>
          </div>
          <div class="neighbors">
            {#each result.knn.neighbors as neighbor, index}
              <article>
                <span>#{index + 1}</span>
                <b>{formatLabel(neighbor.label)}</b>
                <div class="distance"><i style={`width: ${Math.max(6, 100 - Number(neighbor.distance) * 90)}%`}></i></div>
                <small>distance {Number(neighbor.distance).toFixed(4)}</small>
              </article>
            {/each}
          </div>
        </div>
      {/if}
    </section>
  {/if}

  <section class="samples">
    <div class="section-title">
      <p class="eyebrow">Practice inputs</p>
      <h2>Generated sample leaves</h2>
      <span>Click one. It diagnoses immediately.</span>
    </div>

    <div class="filters">
      {#each sampleClasses as klass}
        <button class:active={selectedClass === klass} on:click={() => (selectedClass = klass)}>
          {klass === 'all' ? 'All' : formatLabel(klass)}
        </button>
      {/each}
    </div>

    <div class="gallery">
      {#each visibleSamples as sample}
        <button class="sample-card" on:click={() => selectSample(sample)}>
          <img src={`${api}/image?path=${encodeURIComponent(sample.path)}`} alt={sample.label} />
          <span>{formatLabel(sample.label)}</span>
        </button>
      {/each}
    </div>
  </section>

  <section class="reference">
    <div class="section-title">
      <p class="eyebrow">Reference</p>
      <h2>Class guide</h2>
    </div>
    <div class="class-grid">
      {#each Object.entries(diseaseNotes) as [label, note]}
        <article class={note.tone}>
          <b>{note.title}</b>
          <p>{note.copy}</p>
        </article>
      {/each}
    </div>
  </section>

  <section class="research">
    <div class="section-title">
      <p class="eyebrow">Research output</p>
      <h2>Model comparison</h2>
    </div>
    <div class="experiment-grid">
      {#each experiments as row}
        <article>
          <b>{row.model}</b>
          <span>Accuracy {row.accuracy}</span>
          <div class="meter"><i style={`width: ${Number(row.accuracy) * 100}%`}></i></div>
          <span>Macro-F1 {row.macro_f1}</span>
          <div class="meter"><i style={`width: ${Number(row.macro_f1) * 100}%`}></i></div>
        </article>
      {/each}
    </div>
  </section>
</main>

<style>
  :global(body) {
    margin: 0;
    color: #102415;
    background: #dfead4;
    font-family: 'IBM Plex Sans', 'Helvetica Neue', sans-serif;
  }
  :global(*) {
    box-sizing: border-box;
  }
  .page {
    min-height: 100vh;
    padding: 28px;
    background:
      radial-gradient(circle at 16% 0%, rgba(30, 91, 51, .28), transparent 26%),
      radial-gradient(circle at 88% 8%, rgba(139, 174, 77, .32), transparent 28%),
      linear-gradient(120deg, rgba(16, 36, 21, .08) 1px, transparent 1px),
      #dfead4;
    background-size: auto, auto, 42px 42px, auto;
  }
  .toast {
    position: fixed;
    z-index: 20;
    top: 22px;
    right: 22px;
    max-width: 320px;
    padding: 14px 18px;
    border: 2px solid #102415;
    background: #2f7d4b;
    color: #f7ffe9;
    box-shadow: 6px 6px 0 #102415;
    font-family: 'IBM Plex Sans', 'Helvetica Neue', sans-serif;
    font-weight: 900;
    text-transform: capitalize;
  }
  .hero {
    display: grid;
    grid-template-columns: minmax(0, 1fr) 220px;
    gap: 24px;
    align-items: end;
    padding: 28px;
    border: 2px solid #102415;
    background:
      linear-gradient(135deg, rgba(47, 125, 75, .13), transparent 45%),
      #f4ffe1;
    box-shadow: 8px 8px 0 #102415;
  }
  .eyebrow {
    margin: 0 0 8px;
    color: #2f7d4b;
    font-family: 'IBM Plex Sans', 'Helvetica Neue', sans-serif;
    font-size: 11px;
    font-weight: 900;
    letter-spacing: .16em;
    text-transform: uppercase;
  }
  h1, h2, p {
    margin-top: 0;
  }
  h1 {
    max-width: 900px;
    margin-bottom: 12px;
    font-size: clamp(46px, 8vw, 104px);
    line-height: .88;
    font-weight: 900;
    letter-spacing: 0;
  }
  h2 {
    margin-bottom: 8px;
    font-size: 30px;
    line-height: 1;
  }
  .hero p:not(.eyebrow) {
    max-width: 760px;
    margin-bottom: 0;
    color: #405c3a;
    font-size: 19px;
  }
  .hero-card {
    border: 2px solid #102415;
    background: #bfe39d;
    padding: 18px;
  }
  .hero-card span, .hero-card small {
    display: block;
    font-family: 'IBM Plex Sans', 'Helvetica Neue', sans-serif;
    font-size: 12px;
    font-weight: 900;
    text-transform: uppercase;
  }
  .hero-card strong {
    display: block;
    font-size: 72px;
    line-height: .9;
  }
  .steps {
    display: grid;
    grid-template-columns: repeat(3, minmax(0, 1fr));
    gap: 14px;
    margin-top: 22px;
  }
  .steps article,
  .input-panel,
  .image-panel,
  .result-panel,
  .chart-card,
  .samples,
  .reference,
  .research {
    border: 2px solid #102415;
    background: #f4ffe1;
    box-shadow: 6px 6px 0 #102415;
  }
  .steps article {
    padding: 16px;
  }
  .steps span {
    color: #2f7d4b;
    font-size: 26px;
    font-weight: 900;
  }
  .steps b {
    display: block;
    margin-top: 4px;
    font-size: 20px;
  }
  .steps p,
  .plain,
  .identity-card span,
  .section-title span,
  .reference p {
    color: #405c3a;
  }
  .workspace {
    display: grid;
    grid-template-columns: 300px minmax(280px, .9fr) minmax(320px, 1.1fr);
    gap: 18px;
    margin-top: 24px;
    align-items: stretch;
  }
  .input-panel,
  .result-panel {
    padding: 18px;
  }
  .dropzone {
    display: grid;
    gap: 8px;
    place-items: center;
    min-height: 150px;
    margin: 14px 0;
    padding: 18px;
    border: 2px dashed #1e5b33;
    background:
      radial-gradient(circle at 22% 20%, rgba(47, 125, 75, .18), transparent 22%),
      #e6f5cf;
    text-align: center;
    cursor: pointer;
  }
  .dropzone input {
    display: none;
  }
  .dropzone b {
    font-size: 24px;
  }
  .dropzone span,
  .status,
  .error {
    font-family: 'IBM Plex Sans', 'Helvetica Neue', sans-serif;
    font-size: 13px;
  }
  .quick-actions {
    display: grid;
    grid-template-columns: 1fr;
    gap: 10px;
  }
  button,
  .upload {
    min-height: 42px;
    border: 2px solid #102415;
    background: #2f7d4b;
    color: white;
    font-family: 'IBM Plex Sans', 'Helvetica Neue', sans-serif;
    font-weight: 900;
    cursor: pointer;
  }
  button:disabled {
    cursor: not-allowed;
    opacity: .48;
    filter: grayscale(.35);
  }
  .primary {
    background: #2f7d4b;
    min-height: 54px;
    font-size: 16px;
  }
  .secondary {
    background: #1e5b33;
  }
  input {
    width: 100%;
    border: 2px solid #102415;
    background: #dfead4;
    padding: 10px;
  }
  .status {
    margin: 14px 0 0;
    color: #1e5b33;
    font-weight: 900;
  }
  .error {
    color: #9d2d20;
    font-weight: 900;
  }
  .image-panel {
    position: relative;
    display: grid;
    place-items: center;
    padding: 18px;
    background:
      radial-gradient(circle at 50% 40%, rgba(191, 227, 157, .65), transparent 38%),
      #c9deb6;
  }
  .preview-frame {
    width: 100%;
    max-width: 420px;
    aspect-ratio: 1;
    display: grid;
    place-items: center;
    background: #f4ffe1;
    border: 2px solid #102415;
  }
  .preview-frame img {
    width: 100%;
    height: 100%;
    object-fit: contain;
    image-rendering: pixelated;
  }
  .source-pill {
    position: absolute;
    top: 16px;
    left: 16px;
    padding: 8px 10px;
    background: #102415;
    color: #f4ffe1;
    font-family: 'IBM Plex Sans', 'Helvetica Neue', sans-serif;
    font-size: 12px;
    font-weight: 900;
    text-transform: uppercase;
  }
  .result-panel.empty {
    display: grid;
    align-content: center;
  }
  .diagnosis {
    padding: 14px;
    border: 2px solid #102415;
    background: #f1d3a5;
  }
  .diagnosis.safe {
    background: #bfe39d;
  }
  .diagnosis.watch {
    background: #dced91;
  }
  .diagnosis span,
  .algorithm-cards span,
  .identity-card b,
  .neighbors span,
  .experiment-grid span {
    display: block;
    font-family: 'IBM Plex Sans', 'Helvetica Neue', sans-serif;
    font-size: 11px;
    font-weight: 900;
    letter-spacing: .11em;
    text-transform: uppercase;
  }
  .diagnosis strong {
    display: block;
    margin-top: 4px;
    font-size: 48px;
    line-height: .9;
    text-transform: capitalize;
  }
  .diagnosis small,
  .algorithm-cards small,
  .neighbors small {
    font-family: 'IBM Plex Sans', 'Helvetica Neue', sans-serif;
  }
  .identity-card,
  .solution-card {
    margin-top: 12px;
    padding: 12px;
    border: 2px solid #102415;
    background: #dfead4;
  }
  .solution-card {
    background: #f4ffe1;
  }
  .solution-card b {
    display: block;
    margin-bottom: 8px;
  }
  .solution-card ul {
    margin: 0;
    padding-left: 18px;
    color: #405c3a;
  }
  .solution-card li + li {
    margin-top: 6px;
  }
  .leaf-profile {
    display: grid;
    grid-template-columns: repeat(2, minmax(0, 1fr));
    gap: 10px;
    margin-top: 12px;
  }
  .leaf-profile article {
    padding: 12px;
    border: 2px solid #102415;
    background: #e6f5cf;
  }
  .leaf-profile span {
    display: block;
    color: #1e5b33;
    font-size: 11px;
    font-weight: 900;
    letter-spacing: .11em;
    text-transform: uppercase;
  }
  .leaf-profile b {
    display: block;
    margin-top: 5px;
    font-size: 16px;
  }
  .algorithm-cards {
    display: grid;
    grid-template-columns: repeat(2, minmax(0, 1fr));
    gap: 10px;
    margin-top: 12px;
  }
  .algorithm-cards article {
    padding: 12px;
    border: 2px solid #102415;
    background: #e6f5cf;
  }
  .algorithm-cards b {
    display: block;
    margin: 4px 0;
    font-size: 22px;
    text-transform: capitalize;
  }
  .agreement {
    margin: 12px 0 0;
    color: #1e5b33;
    font-weight: 900;
  }
  .charts {
    display: grid;
    grid-template-columns: minmax(0, 1.2fr) minmax(320px, .8fr);
    gap: 18px;
    margin-top: 24px;
  }
  .chart-card,
  .samples,
  .reference,
  .research {
    padding: 18px;
  }
  .section-title {
    display: flex;
    flex-wrap: wrap;
    gap: 8px 14px;
    justify-content: space-between;
    align-items: end;
    margin-bottom: 14px;
  }
  .section-title h2 {
    margin: 0;
  }
  .feature-list {
    display: grid;
    grid-template-columns: repeat(2, minmax(0, 1fr));
    gap: 12px;
  }
  .feature-list article {
    padding: 12px;
    border: 1px solid #8dae75;
    background: #fbffec;
  }
  .feature-list article > div:first-child {
    display: flex;
    justify-content: space-between;
    gap: 12px;
  }
  .meter,
  .distance {
    height: 11px;
    margin-top: 8px;
    background: #c6d9ac;
    overflow: hidden;
  }
  .meter i,
  .distance i {
    display: block;
    height: 100%;
    background: #1e5b33;
  }
  .neighbors {
    display: grid;
    gap: 10px;
  }
  .neighbors article {
    padding: 12px;
    border: 1px solid #8dae75;
    background: #fbffec;
  }
  .neighbors b {
    display: block;
    margin-top: 4px;
    text-transform: capitalize;
  }
  .samples,
  .reference,
  .research {
    margin-top: 24px;
  }
  .filters {
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
    margin-bottom: 14px;
  }
  .filters button {
    min-height: 34px;
    padding: 0 12px;
    background: #f4ffe1;
    color: #102415;
  }
  .filters .active {
    background: #102415;
    color: #f4ffe1;
  }
  .gallery {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(112px, 1fr));
    gap: 12px;
    max-height: 520px;
    overflow: auto;
    padding: 2px 6px 8px 2px;
  }
  .sample-card {
    display: grid;
    gap: 6px;
    padding: 8px;
    background: #fbffec;
    color: #102415;
    text-align: left;
  }
  .sample-card img {
    width: 100%;
    aspect-ratio: 1;
    object-fit: contain;
    background: #c9deb6;
    image-rendering: pixelated;
  }
  .sample-card span {
    font-size: 12px;
    text-transform: capitalize;
  }
  .class-grid,
  .experiment-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
    gap: 12px;
  }
  .class-grid article,
  .experiment-grid article {
    padding: 14px;
    border: 2px solid #102415;
    background: #fbffec;
  }
  .class-grid article.safe {
    background: #bfe39d;
  }
  .class-grid article.watch {
    background: #dced91;
  }
  .class-grid article.risk {
    background: #f1d3a5;
  }
  .class-grid b,
  .experiment-grid b {
    display: block;
    font-size: 20px;
  }
  @media (max-width: 1040px) {
    .hero,
    .workspace,
    .charts {
      grid-template-columns: 1fr;
    }
    .steps {
      grid-template-columns: 1fr;
    }
  }
  @media (max-width: 680px) {
    .page {
      padding: 14px;
    }
    h1 {
      font-size: 44px;
    }
    .feature-list,
    .algorithm-cards {
      grid-template-columns: 1fr;
    }
  }
</style>
