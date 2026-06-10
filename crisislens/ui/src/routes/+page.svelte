<script>
  const api = 'http://127.0.0.1:8011';
  const demoTweets = [
    'wildfire smoke near homes evacuation needed',
    'river flood water entering homes rescue boats required',
    'hospital needs blood donors and urgent medicine',
    'earthquake tremor cracked buildings people trapped',
    'families need food blankets and temporary shelter'
  ];
  let text = demoTweets[0];
  let result = null;
  let experiments = [];
  let clusters = [];
  let batchText = [
    'smoke seen behind school families evacuating',
    'bridge collapsed after tremor people trapped',
    'clinic needs medicine and clean water'
  ].join('\n');
  let batchResults = [];
  let error = '';

  async function post(path, body) {
    const response = await fetch(`${api}${path}`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body)
    });
    if (!response.ok) throw new Error(await response.text());
    return response.json();
  }

  async function loadIntel() {
    error = '';
    try {
      result = await post('/predict', { text });
      experiments = await (await fetch(`${api}/experiments`)).json();
      clusters = await (await fetch(`${api}/clusters`)).json();
    } catch (err) {
      error = 'Start everything with: make run';
    }
  }

  async function analyzeBatch() {
    error = '';
    try {
      const texts = batchText
        .split('\n')
        .map((line) => line.trim())
        .filter(Boolean);
      batchResults = await post('/batch', { texts });
      experiments = await (await fetch(`${api}/experiments`)).json();
      clusters = await (await fetch(`${api}/clusters`)).json();
    } catch (err) {
      error = 'Start everything with: make run';
    }
  }

  function scoreWidth(value) {
    return `${Math.max(4, Math.min(100, Number(value) * 100))}%`;
  }
</script>

<svelte:head>
  <title>CrisisLens</title>
</svelte:head>

<main class="shell">
  <section class="hero">
    <div>
      <p class="eyebrow">Offline emergency intelligence</p>
      <h1>CrisisLens</h1>
      <p class="brief">Not a keyword matcher: the UI compares a rule baseline, trained Naive Bayes, and a perceptron challenger so you can explain the ML clearly.</p>
    </div>
    <button on:click={loadIntel}>Analyze Signal</button>
  </section>

  <section class="samples">
    {#each demoTweets as tweet}
      <button class:active={tweet === text} on:click={() => (text = tweet)}>{tweet}</button>
    {/each}
  </section>

  <section class="console">
    <label>
      Incoming field report from outside dataset
      <textarea bind:value={text}></textarea>
    </label>
    {#if error}<p class="error">{error}</p>{/if}
    {#if result}
      <div class="verdict {result.severity.level}">
        <div>
          <span>Trained ML Prediction</span>
          <strong>{result.label}</strong>
          <small>{Math.round(result.confidence * 100)}% confidence</small>
        </div>
        <div>
          <span>Severity</span>
          <strong>{result.severity.level}</strong>
          <small>{result.severity.reason}</small>
        </div>
      </div>

      <div class="explain-grid">
        <article>
          <h2>Why this is ML</h2>
          <p>{result.research_note}</p>
          <div class="compare">
            <span>Keyword baseline</span>
            <b>{result.lexical_baseline.label}</b>
            <small>{result.lexical_baseline.matched_terms.join(', ') || 'no exact crisis keyword'}</small>
          </div>
          <div class="compare">
            <span>Perceptron challenger</span>
            <b>{result.challenger.label}</b>
            <small>{result.model_agreement ? 'agrees with Naive Bayes' : 'disagrees, so human review is useful'}</small>
          </div>
        </article>

        <article>
          <h2>Evidence Tokens</h2>
          <div class="chips">
            {#each result.evidence as token}
              <span>{token}</span>
            {/each}
          </div>
          <h2>Response Plan</h2>
          <ol>
            {#each result.recommended_actions as action}
              <li>{action}</li>
            {/each}
          </ol>
        </article>
      </div>
    {/if}
  </section>

  <section class="batch">
    <div>
      <h2>External Batch Input</h2>
      <p>Paste any new reports here. These are not pulled from the training CSV; the API classifies them live.</p>
      <textarea bind:value={batchText}></textarea>
      <button on:click={analyzeBatch}>Analyze External Reports</button>
    </div>
    <div>
      <h2>Live Batch Output</h2>
      {#each batchResults as item}
        <article class="batch-row">
          <p>{item.text}</p>
          <b>{item.label}</b>
          <small>{item.severity.level} · {Math.round(item.confidence * 100)}% · baseline {item.lexical_baseline.label}</small>
        </article>
      {/each}
    </div>
  </section>

  <section class="grid">
    <div>
      <h2>Research Results</h2>
      {#each experiments as row}
        <article class="metric">
          <b>{row.model}</b>
          <span>Accuracy {row.accuracy}</span>
          <div class="bar"><i style={`width: ${scoreWidth(row.accuracy)}`}></i></div>
          <span>Macro-F1 {row.macro_f1}</span>
          <div class="bar"><i style={`width: ${scoreWidth(row.macro_f1)}`}></i></div>
        </article>
      {/each}
    </div>
    <div>
      <h2>Unlabeled Event Discovery</h2>
      {#each clusters as cluster}
        <article class="cluster">
          <b>Cluster {cluster.cluster} · {cluster.size} reports</b>
          <p>{cluster.keywords.join(' / ')}</p>
        </article>
      {/each}
    </div>
  </section>
</main>

<style>
  :global(body) {
    margin: 0;
    color: #f6f1df;
    background: #17140f;
    font-family: Georgia, 'Times New Roman', serif;
  }
  .shell {
    min-height: 100vh;
    padding: 32px;
    background:
      linear-gradient(90deg, rgba(255,255,255,.04) 1px, transparent 1px),
      linear-gradient(rgba(255,255,255,.035) 1px, transparent 1px),
      radial-gradient(circle at 80% 10%, rgba(214, 58, 43, .32), transparent 28%),
      #17140f;
    background-size: 36px 36px, 36px 36px, auto, auto;
  }
  .hero {
    display: flex;
    justify-content: space-between;
    gap: 24px;
    align-items: end;
    border-bottom: 1px solid #8b2f23;
    padding-bottom: 28px;
  }
  .eyebrow, label, h2 {
    color: #ffb347;
    letter-spacing: .14em;
    text-transform: uppercase;
    font-size: 12px;
  }
  h1 {
    font-size: clamp(56px, 10vw, 124px);
    line-height: .82;
    margin: 0;
    color: #ffefc2;
  }
  .brief {
    max-width: 720px;
    color: #cfc1a4;
    font-size: 18px;
  }
  button {
    background: #d63a2b;
    color: #fff7df;
    border: 0;
    padding: 14px 18px;
    font-weight: 800;
    text-transform: uppercase;
    cursor: pointer;
  }
  .samples {
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
    margin-top: 20px;
  }
  .samples button {
    background: rgba(255,239,194,.08);
    border: 1px solid rgba(255,179,71,.35);
    text-align: left;
    text-transform: none;
  }
  .samples .active {
    background: #ffb347;
    color: #17140f;
  }
  .console, .grid, .batch {
    margin-top: 28px;
  }
  label {
    display: grid;
    gap: 10px;
  }
  textarea {
    min-height: 120px;
    resize: vertical;
    background: rgba(0,0,0,.45);
    border: 1px solid #8b2f23;
    color: #fff7df;
    padding: 18px;
    font: 20px Georgia, serif;
  }
  .verdict {
    margin-top: 18px;
    display: grid;
    grid-template-columns: repeat(2, minmax(0, 1fr));
    gap: 16px;
  }
  .verdict > div, .explain-grid article, .metric, .cluster, .batch > div, .batch-row {
    border: 1px solid rgba(255,179,71,.35);
    background: rgba(255,239,194,.06);
    padding: 16px;
  }
  .verdict strong {
    display: block;
    font-size: 44px;
    color: #ffb347;
    text-transform: uppercase;
  }
  .high > div:first-child {
    box-shadow: inset 5px 0 #d63a2b;
  }
  .explain-grid, .grid, .batch {
    display: grid;
    grid-template-columns: repeat(2, minmax(0, 1fr));
    gap: 20px;
    margin-top: 20px;
  }
  .compare {
    display: grid;
    gap: 2px;
    margin-top: 10px;
    padding-top: 10px;
    border-top: 1px solid rgba(255,179,71,.25);
  }
  .compare b {
    color: #ffefc2;
    font-size: 24px;
  }
  .chips {
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
  }
  .chips span {
    padding: 8px 10px;
    border: 1px solid rgba(255,179,71,.35);
  }
  .metric {
    display: grid;
    grid-template-columns: 1fr auto;
    gap: 10px;
    margin: 10px 0;
  }
  .metric .bar {
    grid-column: 1 / -1;
    height: 9px;
    background: rgba(0,0,0,.35);
  }
  .bar i {
    display: block;
    height: 100%;
    background: #ffb347;
  }
  .cluster {
    margin: 10px 0;
  }
  .batch p {
    color: #cfc1a4;
  }
  .batch-row {
    margin: 10px 0;
  }
  .batch-row b {
    color: #ffb347;
    text-transform: uppercase;
  }
  .error {
    color: #ffb347;
  }
  @media (max-width: 760px) {
    .hero, .grid, .metric, .verdict, .explain-grid, .batch {
      display: block;
    }
    button {
      margin-top: 10px;
      width: 100%;
    }
  }
</style>
