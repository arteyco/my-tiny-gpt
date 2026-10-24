
/* ---------------- CORPUS & VOCABULARY ---------------- */
const rawCorpus = [
 "hello friends how are you ",
 "the coffee is very hot ",
 "my name is castani ",
 "the roads of Bogota are busy ",
 "it is raining in Usaquen ",
 "the train is late again ",
 "i love eating cakes and drinking coffee ",
 "jazz is my favorite festival ",
 "christmas brings lights and sweets ",
 "santafe won the succer match "
];
const fullCorpus = rawCorpus.map(s => s + " <END>");
const allTokens = fullCorpus.join(" ").split(/\s+/).filter(Boolean);
const vocab = [...new Set(allTokens)].sort();
const word2idx = {};
vocab.forEach((w, i) => word2idx[w] = i);
const idx2word = {};
vocab.forEach((w, i) => idx2word[i] = w);

// Populate datalist
const datalist = document.getElementById('vocabList');
datalist.innerHTML = vocab.map(w => `<option value="${w}">`).join('');

/* ---------------- BIGRAM MODEL (Simulating the trained LLM) ---------------- */
const bigrams = {};
for (let i = 0; i < allTokens.length - 1; i++) {
  const w1 = allTokens[i];
  const w2 = allTokens[i + 1];
  if (!bigrams[w1]) bigrams[w1] = {};
  bigrams[w1][w2] = (bigrams[w1][w2] || 0) + 1;
}
// Normalize to probabilities
for (const w1 in bigrams) {
  const total = Object.values(bigrams[w1]).reduce((a, b) => a + b, 0);
  for (const w2 in bigrams[w1]) {
    bigrams[w1][w2] /= total;
  }
}

function getProbs(word) {
  if (bigrams[word]) return bigrams[word];
  // Fallback: uniform if word was never seen as a predecessor
  const uniform = {};
  const p = 1.0 / vocab.length;
  vocab.forEach(w => uniform[w] = p);
  return uniform;
}

function sampleWord(probs) {
  const r = Math.random();
  let cum = 0;
  for (const [w, p] of Object.entries(probs)) {
    cum += p;
    if (r <= cum) return w;
  }
  return Object.keys(probs)[0];
}

const sleep = ms => new Promise(r => setTimeout(r, ms));

/* ---------------- VOCABULARY DEMO UI ---------------- */
function buildVocab() {
  renderVocab();
}
function shuffleVocab() {
  // Just visually shuffle the display order, keeping IDs consistent
  const shuffled = [...vocab].sort(() => Math.random() - 0.5);
  const el = document.getElementById('vocabDisplay');
  el.innerHTML = shuffled.map(w =>
    `<div class="token" onclick="this.classList.toggle('hi')">${w}<span class="idx">#${word2idx[w]}</span></div>`
  ).join('');
}
function renderVocab() {
  const el = document.getElementById('vocabDisplay');
  el.innerHTML = vocab.map(w =>
    `<div class="token" onclick="this.classList.toggle('hi')">${w}<span class="idx">#${word2idx[w]}</span></div>`
  ).join('');
  document.getElementById('vocabStats').innerHTML =
    `<b>Vocab size:</b> ${vocab.length} · <b>Total tokens in corpus:</b> ${allTokens.length}`;
}

/* ---------------- BATCH DEMO ---------------- */
function sampleBatch() {
  const blockSize = 6;
  const rows = [];
  for (let i = 0; i < 5; i++) {
    const start = Math.floor(Math.random() * (allTokens.length - blockSize));
    const x = allTokens.slice(start, start + blockSize);
    const y = allTokens.slice(start + 1, start + blockSize + 1);
    rows.push({ x, y });
  }
  const html = `<table class="batch">
    <thead><tr><th>Row</th><th>x (context)</th><th>y (target)</th></tr></thead>
    <tbody>${rows.map((r, i) => `
      <tr>
        <td style="color:var(--text-mute)">${i}</td>
        <td class="word">${r.x.map(w => `<span style="color:var(--accent-3)">${w}</span>`).join(' ')}</td>
        <td class="word">${r.y.map(w => `<span style="color:var(--warn)">${w}</span>`).join(' ')}</td>
      </tr>`).join('')}
    </tbody></table>`;
  document.getElementById('batchTable').innerHTML = html;
}

/* ---------------- ATTENTION MASK ---------------- */
function setMaskSize(T) {
  const el = document.getElementById('maskMatrix');
  el.style.gridTemplateColumns = `repeat(${T}, 28px)`;
  let html = '';
  for (let i = 0; i < T; i++) {
    for (let j = 0; j < T; j++) {
      const allowed = j <= i;
      html += `<div class="cell ${allowed ? 'allowed' : 'masked'}">${allowed ? '✓' : '−'}</div>`;
    }
  }
  el.innerHTML = html;
}

/* ---------------- LOSS CURVE ---------------- */
function drawLoss() {
  const svg = document.getElementById('lossSvg');
  const W = 600, H = 140;
  const N = 60;
  const pts = [];
  for (let i = 0; i < N; i++) {
    const t = i / (N - 1);
    const base = 8 * Math.exp(-3 * t) + 1.2 + (Math.random() - 0.5) * 0.4 * (1 - t);
    pts.push([i * (W / (N - 1)), H - (base / 9) * H]);
  }
  const d = pts.map((p, i) => (i === 0 ? 'M' : 'L') + p[0].toFixed(1) + ',' + p[1].toFixed(1)).join(' ');
  const area = d + ` L ${W},${H} L 0,${H} Z`;
  svg.innerHTML = `
    <defs>
      <linearGradient id="g1" x1="0" x2="0" y1="0" y2="1">
        <stop offset="0%" stop-color="#7c9cff" stop-opacity=".5"/>
        <stop offset="100%" stop-color="#7c9cff" stop-opacity="0"/>
      </linearGradient>
    </defs>
    <path d="${area}" fill="url(#g1)"/>
    <path d="${d}" fill="none" stroke="#7c9cff" stroke-width="2"/>
  `;
}

/* ---------------- INTERACTIVE GENERATION ---------------- */
let isGenerating = false;

async function startGeneration() {
  if (isGenerating) return;
  
  const input = document.getElementById('genInput').value.trim().toLowerCase();
  if (!input || !word2idx.hasOwnProperty(input)) {
    alert(`Please enter a valid word from the vocabulary. (e.g., ${vocab[Math.floor(Math.random()*5)]})`);
    return;
  }

  isGenerating = true;
  document.getElementById('genBtn').disabled = true;
  
  const terminal = document.getElementById('genTerminal');
  const probChart = document.getElementById('probChart');
  const probBars = document.getElementById('probBars');
  const finalOutput = document.getElementById('finalOutput');
  const finalTokens = document.getElementById('finalTokens');

  terminal.style.display = 'block';
  probChart.style.display = 'block';
  finalOutput.style.display = 'none';
  terminal.innerHTML = '';
  finalTokens.innerHTML = '';

  const blockSize = 6;
  const maxNewTokens = 12;
  let context = [input];

  // Add initial token to final output
  finalTokens.innerHTML += `<div class="token">${input}</div>`;

  const addLog = (html) => {
    const div = document.createElement('div');
    div.innerHTML = html;
    terminal.appendChild(div);
    terminal.scrollTop = terminal.scrollHeight;
  };

  addLog(`<span class="step-title">▶ Initializing generation...</span>`);
  addLog(`Context: <span class="ctx">[${context.join(', ')}]</span>`);
  await sleep(500);

  for (let step = 0; step < maxNewTokens; step++) {
    addLog(`<br><span class="step-title">Step ${step + 1}:</span>`);
    
    // Crop context (simulating idx[:, -block_size:])
    const ctxCond = context.slice(-blockSize);
    addLog(`Cropped context (block_size=6): <span class="ctx">[${ctxCond.join(', ')}]</span>`);
    
    // Get probabilities for the last token
    const lastToken = ctxCond[ctxCond.length - 1];
    const probs = getProbs(lastToken);
    
    // Sort and get top 5
    const sortedProbs = Object.entries(probs).sort((a, b) => b[1] - a[1]).slice(0, 5);
    
    // Update probability chart
    probBars.innerHTML = sortedProbs.map(([w, p]) => `
      <div class="prob-row">
        <div class="word">${w}</div>
        <div class="bar-wrap"><div class="bar-fill" style="width: 0%"></div></div>
        <div class="pct">${(p * 100).toFixed(1)}%</div>
      </div>
    `).join('');
    
    await sleep(300);
    
    // Animate bars
    const bars = probBars.querySelectorAll('.bar-fill');
    sortedProbs.forEach(([, p], i) => {
      bars[i].style.width = `${p * 100}%`;
    });
    
    await sleep(400);
    
    addLog(`Softmax → Top 5 probs calculated.`);
    
    // Sample next word
    const nextWord = sampleWord(probs);
    addLog(`Sampled: <span class="sample">${nextWord}</span>`);
    
    // Append to context
    context.push(nextWord);
    
    // Add to final output
    const tokenEl = document.createElement('div');
    tokenEl.className = 'token new';
    tokenEl.textContent = nextWord;
    finalTokens.appendChild(tokenEl);
    
    await sleep(300);
    
    if (nextWord === '<END>') {
      addLog(`<span class="sample">Reached &lt;END&gt; token. Stopping.</span>`);
      break;
    }
  }

  finalOutput.style.display = 'block';
  addLog(`<br><span class="step-title">✓ Generation complete.</span>`);
  
  isGenerating = false;
  document.getElementById('genBtn').disabled = false;
}

function clearGeneration() {
  if (isGenerating) return;
  document.getElementById('genInput').value = '';
  document.getElementById('genTerminal').style.display = 'none';
  document.getElementById('probChart').style.display = 'none';
  document.getElementById('finalOutput').style.display = 'none';
  document.getElementById('genTerminal').innerHTML = '';
  document.getElementById('finalTokens').innerHTML = '';
}

/* ---------------- NAV HIGHLIGHT ---------------- */
const navLinks = document.querySelectorAll('.nav a');
const sections = [...document.querySelectorAll('section.block')];
window.addEventListener('scroll', () => {
  const y = window.scrollY + 120;
  let active = sections[0].id;
  sections.forEach(s => { if (s.offsetTop <= y) active = s.id; });
  navLinks.forEach(a => a.classList.toggle('active', a.getAttribute('href') === '#' + active));
});



/* ---------------- INIT ---------------- */
buildVocab();
sampleBatch();
setMaskSize(6);
drawLoss();
