import { useState, useMemo, useCallback, useRef, useEffect } from "react";

// ═══════════════════════════════════════════════════════════════
// MATH ENGINE — Pure JS NumPy-equivalent operations
// ═══════════════════════════════════════════════════════════════

function seededRandom(seed) {
  let s = seed;
  return () => {
    s = (s * 1664525 + 1013904223) & 0xffffffff;
    return (s >>> 0) / 0xffffffff;
  };
}

function randn(shape, scale = 0.02, rng) {
  const size = shape.reduce((a, b) => a * b, 1);
  const data = new Float32Array(size);
  for (let i = 0; i < size; i += 2) {
    const u1 = rng() || 1e-10;
    const u2 = rng();
    const r = Math.sqrt(-2 * Math.log(u1));
    data[i] = r * Math.cos(2 * Math.PI * u2) * scale;
    if (i + 1 < size) data[i + 1] = r * Math.sin(2 * Math.PI * u2) * scale;
  }
  return { data, shape: [...shape] };
}

function ones(shape) {
  const size = shape.reduce((a, b) => a * b, 1);
  return { data: new Float32Array(size).fill(1), shape: [...shape] };
}

function zeros(shape) {
  const size = shape.reduce((a, b) => a * b, 1);
  return { data: new Float32Array(size).fill(0), shape: [...shape] };
}

function getIdx(shape, ...indices) {
  let idx = 0;
  let stride = 1;
  for (let i = shape.length - 1; i >= 0; i--) {
    idx += indices[i] * stride;
    stride *= shape[i];
  }
  return idx;
}

function get(t, ...indices) {
  return t.data[getIdx(t.shape, ...indices)];
}

function set(t, value, ...indices) {
  t.data[getIdx(t.shape, ...indices)] = value;
}

// Matrix multiply: (..., M, K) @ (..., K, N) -> (..., M, N)
function matmul(a, b) {
  const aRank = a.shape.length;
  const bRank = b.shape.length;

  if (aRank === 1 && bRank === 2) {
    // (K,) @ (K, N) -> (N,)
    const [K] = a.shape;
    const [, N] = b.shape;
    const out = zeros([N]);
    for (let n = 0; n < N; n++) {
      let sum = 0;
      for (let k = 0; k < K; k++) sum += a.data[k] * b.data[k * N + n];
      out.data[n] = sum;
    }
    return out;
  }

  if (aRank === 2 && bRank === 2) {
    const [M, K] = a.shape;
    const [, N] = b.shape;
    const out = zeros([M, N]);
    for (let m = 0; m < M; m++)
      for (let n = 0; n < N; n++) {
        let sum = 0;
        for (let k = 0; k < K; k++) sum += a.data[m * K + k] * b.data[k * N + n];
        out.data[m * N + n] = sum;
      }
    return out;
  }

  // Batched: (B, M, K) @ (B, K, N) -> (B, M, N)
  if (aRank === 3 && bRank === 3) {
    const [B, M, K] = a.shape;
    const [, , N] = b.shape;
    const out = zeros([B, M, N]);
    for (let b = 0; b < B; b++)
      for (let m = 0; m < M; m++)
        for (let n = 0; n < N; n++) {
          let sum = 0;
          for (let k = 0; k < K; k++)
            sum += get(a, b, m, k) * get(b === undefined ? a : { data: a.data, shape: a.shape }, b, m, k);
          // Re-do properly:
        }
    // Simpler batched matmul
    const out2 = zeros([B, M, N]);
    for (let bi = 0; bi < B; bi++)
      for (let m = 0; m < M; m++)
        for (let n = 0; n < N; n++) {
          let sum = 0;
          for (let k = 0; k < K; k++) {
            const ai = bi * M * K + m * K + k;
            const bj = bi * K * N + k * N + n;
            sum += a.data[ai] * b.data[bj];
          }
          out2.data[bi * M * N + m * N + n] = sum;
        }
    return out2;
  }
  throw new Error(`matmul: unsupported shapes ${a.shape} @ ${b.shape}`);
}

// Batched matmul for 3D tensors
function bmm(a, b) {
  const [B, M, K] = a.shape;
  const [, , N] = b.shape;
  const out = zeros([B, M, N]);
  for (let bi = 0; bi < B; bi++)
    for (let m = 0; m < M; m++)
      for (let n = 0; n < N; n++) {
        let sum = 0;
        for (let k = 0; k < K; k++)
          sum += a.data[bi * M * K + m * K + k] * b.data[bi * K * N + k * N + n];
        out.data[bi * M * N + m * N + n] = sum;
      }
  return out;
}

// Transpose last two dims of 3D tensor
function transpose3d(t) {
  const [B, M, N] = t.shape;
  const out = zeros([B, N, M]);
  for (let b = 0; b < B; b++)
    for (let m = 0; m < M; m++)
      for (let n = 0; n < N; n++)
        out.data[b * N * M + n * M + m] = t.data[b * M * N + m * N + n];
  return out;
}

// Reshape from (S, H, D) to (H, S, D)
function transposeAxes012to102(t) {
  const [S, H, D] = t.shape;
  const out = zeros([H, S, D]);
  for (let s = 0; s < S; s++)
    for (let h = 0; h < H; h++)
      for (let d = 0; d < D; d++)
        out.data[h * S * D + s * D + d] = t.data[s * H * D + h * D + d];
  return out;
}

function transposeAxes102to012(t) {
  const [H, S, D] = t.shape;
  const out = zeros([S, H, D]);
  for (let s = 0; s < S; s++)
    for (let h = 0; h < H; h++)
      for (let d = 0; d < D; d++)
        out.data[s * H * D + h * D + d] = t.data[h * S * D + s * D + d];
  return out;
}

function reshape(t, newShape) {
  return { data: t.data, shape: [...newShape] };
}

function sliceRow(t, row) {
  const cols = t.shape[t.shape.length - 1];
  const start = row * cols;
  return { data: t.data.slice(start, start + cols), shape: [cols] };
}

function sliceRows(t, rows) {
  const cols = t.shape[1];
  const out = zeros([rows.length, cols]);
  for (let i = 0; i < rows.length; i++) {
    const src = rows[i] * cols;
    for (let j = 0; j < cols; j++) out.data[i * cols + j] = t.data[src + j];
  }
  return out;
}

function add(a, b) {
  const out = { data: new Float32Array(a.data.length), shape: [...a.shape] };
  for (let i = 0; i < a.data.length; i++) out.data[i] = a.data[i] + b.data[i];
  return out;
}

function mul(a, b) {
  const out = { data: new Float32Array(a.data.length), shape: [...a.shape] };
  for (let i = 0; i < a.data.length; i++) out.data[i] = a.data[i] * b.data[i];
  return out;
}

function norm(t) {
  let sum = 0;
  for (let i = 0; i < t.data.length; i++) sum += t.data[i] * t.data[i];
  return Math.sqrt(sum);
}

function fmt(v, decimals = 4) {
  return typeof v === "number" ? v.toFixed(decimals) : String(v);
}

function fmtArr(data, n = 8, decimals = 4) {
  const items = Array.from(data).slice(0, n).map((v) => v.toFixed(decimals));
  if (data.length > n) items.push("...");
  return `[${items.join(", ")}]`;
}

// ═══════════════════════════════════════════════════════════════
// MODEL OPERATIONS
// ═══════════════════════════════════════════════════════════════

const CONFIG = {
  vocabSize: 64,
  dModel: 32,
  nHeads: 4,
  nKvHeads: 2,
  dHead: 8,
  dFf: 86,
  nLayers: 2,
  maxSeqLen: 16,
  ropeTheta: 10000.0,
  eps: 1e-6,
};

function initParams(rng) {
  const p = {};
  const w = (shape) => randn(shape, 0.02, rng);
  p.tokEmb = w([CONFIG.vocabSize, CONFIG.dModel]);
  for (let l = 0; l < CONFIG.nLayers; l++) {
    p[`l${l}_attnNorm`] = ones([CONFIG.dModel]);
    p[`l${l}_wq`] = w([CONFIG.dModel, CONFIG.nHeads * CONFIG.dHead]);
    p[`l${l}_wk`] = w([CONFIG.dModel, CONFIG.nKvHeads * CONFIG.dHead]);
    p[`l${l}_wv`] = w([CONFIG.dModel, CONFIG.nKvHeads * CONFIG.dHead]);
    p[`l${l}_wo`] = w([CONFIG.nHeads * CONFIG.dHead, CONFIG.dModel]);
    p[`l${l}_ffnNorm`] = ones([CONFIG.dModel]);
    p[`l${l}_wGate`] = w([CONFIG.dModel, CONFIG.dFf]);
    p[`l${l}_wUp`] = w([CONFIG.dModel, CONFIG.dFf]);
    p[`l${l}_wDown`] = w([CONFIG.dFf, CONFIG.dModel]);
  }
  p.finalNorm = ones([CONFIG.dModel]);
  p.lmHead = w([CONFIG.dModel, CONFIG.vocabSize]);
  return p;
}

function computeRopeFreqs() {
  const { dHead, maxSeqLen, ropeTheta } = CONFIG;
  const halfD = dHead / 2;
  const freqs = new Float32Array(halfD);
  for (let i = 0; i < halfD; i++)
    freqs[i] = 1.0 / Math.pow(ropeTheta, (2 * i) / dHead);

  const cosTable = zeros([maxSeqLen, halfD]);
  const sinTable = zeros([maxSeqLen, halfD]);
  for (let t = 0; t < maxSeqLen; t++)
    for (let i = 0; i < halfD; i++) {
      const angle = t * freqs[i];
      cosTable.data[t * halfD + i] = Math.cos(angle);
      sinTable.data[t * halfD + i] = Math.sin(angle);
    }
  return { cosTable, sinTable, freqs };
}

function applyRope(qk, cosTable, sinTable, startPos = 0) {
  const [seqLen, nH, dH] = qk.shape;
  const halfD = dH / 2;
  const out = zeros(qk.shape);
  for (let s = 0; s < seqLen; s++) {
    const pos = startPos + s;
    for (let h = 0; h < nH; h++) {
      for (let i = 0; i < halfD; i++) {
        const cosV = cosTable.data[pos * halfD + i];
        const sinV = sinTable.data[pos * halfD + i];
        const even = qk.data[s * nH * dH + h * dH + i];
        const odd = qk.data[s * nH * dH + h * dH + halfD + i];
        out.data[s * nH * dH + h * dH + i] = even * cosV - odd * sinV;
        out.data[s * nH * dH + h * dH + halfD + i] = even * sinV + odd * cosV;
      }
    }
  }
  return out;
}

function rmsnorm(x, weight) {
  // x: (S, D), weight: (D,)
  const [S, D] = x.shape;
  const out = zeros(x.shape);
  for (let s = 0; s < S; s++) {
    let sumSq = 0;
    for (let d = 0; d < D; d++) {
      const v = x.data[s * D + d];
      sumSq += v * v;
    }
    const rms = Math.sqrt(sumSq / D + CONFIG.eps);
    for (let d = 0; d < D; d++)
      out.data[s * D + d] = (x.data[s * D + d] / rms) * weight.data[d];
  }
  return out;
}

function expandKV(t, nRep) {
  if (nRep === 1) return t;
  const [S, nKV, D] = t.shape;
  const out = zeros([S, nKV * nRep, D]);
  for (let s = 0; s < S; s++)
    for (let kv = 0; kv < nKV; kv++)
      for (let r = 0; r < nRep; r++)
        for (let d = 0; d < D; d++)
          out.data[s * nKV * nRep * D + (kv * nRep + r) * D + d] =
            t.data[s * nKV * D + kv * D + d];
  return out;
}

function silu(x) {
  const out = { data: new Float32Array(x.data.length), shape: [...x.shape] };
  for (let i = 0; i < x.data.length; i++) {
    const v = x.data[i];
    out.data[i] = v / (1 + Math.exp(-v));
  }
  return out;
}

function softmaxRow(data, offset, len) {
  let max = -Infinity;
  for (let i = 0; i < len; i++) max = Math.max(max, data[offset + i]);
  let sum = 0;
  for (let i = 0; i < len; i++) {
    data[offset + i] = Math.exp(data[offset + i] - max);
    sum += data[offset + i];
  }
  for (let i = 0; i < len; i++) data[offset + i] /= sum;
}

function softmax1D(logits) {
  const out = new Float32Array(logits.length);
  let max = -Infinity;
  for (let i = 0; i < logits.length; i++) max = Math.max(max, logits[i]);
  let sum = 0;
  for (let i = 0; i < logits.length; i++) {
    out[i] = Math.exp(logits[i] - max);
    sum += out[i];
  }
  for (let i = 0; i < logits.length; i++) out[i] /= sum;
  return out;
}

function attention(q, k, v, dHead) {
  const seqLen = q.shape[0];
  const nH = q.shape[1];
  const qt = transposeAxes012to102(q);
  const kt = transposeAxes012to102(k);
  const vt = transposeAxes012to102(v);
  const ktT = transpose3d(kt);
  let scores = bmm(qt, ktT);
  const scale = Math.sqrt(dHead);
  for (let i = 0; i < scores.data.length; i++) scores.data[i] /= scale;
  // Causal mask
  for (let h = 0; h < nH; h++)
    for (let i = 0; i < seqLen; i++)
      for (let j = i + 1; j < seqLen; j++)
        scores.data[h * seqLen * seqLen + i * seqLen + j] = -1e9;
  // Softmax per row
  for (let h = 0; h < nH; h++)
    for (let i = 0; i < seqLen; i++)
      softmaxRow(scores.data, h * seqLen * seqLen + i * seqLen, seqLen);
  const attnOut = bmm(scores, vt);
  return { output: transposeAxes102to012(attnOut), weights: scores };
}

function forwardBlock(x, l, params, rope) {
  const { nHeads, nKvHeads, dHead } = CONFIG;
  const nRep = nHeads / nKvHeads;
  const pfx = `l${l}_`;
  const S = x.shape[0];

  // Attention
  const xn = rmsnorm(x, params[pfx + "attnNorm"]);
  let q = reshape(matmul(xn, params[pfx + "wq"]), [S, nHeads, dHead]);
  let k = reshape(matmul(xn, params[pfx + "wk"]), [S, nKvHeads, dHead]);
  let v = reshape(matmul(xn, params[pfx + "wv"]), [S, nKvHeads, dHead]);
  q = applyRope(q, rope.cosTable, rope.sinTable);
  k = applyRope(k, rope.cosTable, rope.sinTable);
  const kExp = expandKV(k, nRep);
  const vExp = expandKV(v, nRep);
  const { output: attnOut, weights: attnW } = attention(q, kExp, vExp, dHead);
  const concat = reshape(attnOut, [S, nHeads * dHead]);
  const projected = matmul(concat, params[pfx + "wo"]);
  let h = add(x, projected);

  // FFN
  const hn = rmsnorm(h, params[pfx + "ffnNorm"]);
  const gate = silu(matmul(hn, params[pfx + "wGate"]));
  const up = matmul(hn, params[pfx + "wUp"]);
  const gated = mul(gate, up);
  const down = matmul(gated, params[pfx + "wDown"]);
  h = add(h, down);

  return { h, attnW, q, k: kExp, v: vExp };
}

function fullForward(inputIds, params, rope) {
  let h = sliceRows(params.tokEmb, inputIds);
  const layers = [];
  for (let l = 0; l < CONFIG.nLayers; l++) {
    const result = forwardBlock(h, l, params, rope);
    layers.push(result);
    h = result.h;
  }
  const hNorm = rmsnorm(h, params.finalNorm);
  const lastHidden = sliceRow(hNorm, inputIds.length - 1);
  const logitsT = matmul(
    reshape(lastHidden, [1, CONFIG.dModel]),
    params.lmHead
  );
  const logits = logitsT.data;
  return { logits, layers, hNorm };
}

function sampleToken(logits, temperature = 0.8, topK = 10, rng) {
  const V = logits.length;
  const scaled = new Float32Array(V);
  for (let i = 0; i < V; i++) scaled[i] = logits[i] / temperature;

  // Top-k
  const indices = Array.from({ length: V }, (_, i) => i);
  indices.sort((a, b) => scaled[b] - scaled[a]);
  const topKIds = indices.slice(0, topK);
  const filtered = new Float32Array(V).fill(-1e9);
  for (const idx of topKIds) filtered[idx] = scaled[idx];

  const probs = softmax1D(filtered);
  // Sample
  let r = rng();
  let cum = 0;
  for (let i = 0; i < V; i++) {
    cum += probs[i];
    if (r < cum) return { tokenId: i, probs };
  }
  return { tokenId: V - 1, probs };
}

// ═══════════════════════════════════════════════════════════════
// REACT COMPONENTS
// ═══════════════════════════════════════════════════════════════

const COLORS = {
  bg: "#0a0e17",
  surface: "#111827",
  surfaceHover: "#1a2235",
  border: "#1e2a3a",
  borderHi: "#2d4a6f",
  text: "#c9d1d9",
  textDim: "#6b7a8d",
  textBright: "#e6edf3",
  accent: "#58a6ff",
  accentDim: "#1a3a5c",
  green: "#3fb950",
  greenDim: "#1a3328",
  orange: "#d29922",
  orangeDim: "#3d2e0a",
  red: "#f85149",
  purple: "#bc8cff",
  purpleDim: "#2a1f3d",
  cyan: "#39d2c0",
};

function TensorDisplay({ data, shape, label, maxItems = 16, highlight }) {
  const items = Array.from(data).slice(0, maxItems);
  return (
    <div style={{ margin: "8px 0" }}>
      {label && (
        <div style={{ color: COLORS.textDim, fontSize: 11, marginBottom: 3, fontFamily: "monospace" }}>
          {label} shape=({shape.join(", ")})
        </div>
      )}
      <div
        style={{
          display: "flex",
          flexWrap: "wrap",
          gap: 2,
          fontFamily: "monospace",
          fontSize: 11,
        }}
      >
        {items.map((v, i) => (
          <span
            key={i}
            style={{
              padding: "1px 4px",
              borderRadius: 3,
              background:
                highlight && v === Math.max(...items)
                  ? COLORS.accentDim
                  : v > 0
                  ? `rgba(59,185,80,${Math.min(Math.abs(v) * 3, 0.25)})`
                  : `rgba(248,81,73,${Math.min(Math.abs(v) * 3, 0.25)})`,
              color: COLORS.textBright,
              border: `1px solid ${COLORS.border}`,
            }}
          >
            {v.toFixed(4)}
          </span>
        ))}
        {data.length > maxItems && (
          <span style={{ color: COLORS.textDim, padding: "1px 4px" }}>
            ...+{data.length - maxItems} more
          </span>
        )}
      </div>
    </div>
  );
}

function AttnHeatmap({ weights, seqLen, headIdx }) {
  const cellSize = Math.min(36, 200 / seqLen);
  const offset = headIdx * seqLen * seqLen;
  return (
    <div style={{ display: "inline-block", margin: "4px 8px 4px 0" }}>
      <div style={{ fontSize: 10, color: COLORS.textDim, marginBottom: 2, fontFamily: "monospace" }}>
        Head {headIdx}
      </div>
      <div style={{ display: "grid", gridTemplateColumns: `repeat(${seqLen}, ${cellSize}px)`, gap: 1 }}>
        {Array.from({ length: seqLen * seqLen }, (_, idx) => {
          const row = Math.floor(idx / seqLen);
          const col = idx % seqLen;
          const v = weights.data[offset + row * seqLen + col];
          const intensity = Math.pow(v, 0.5);
          return (
            <div
              key={idx}
              title={`pos ${row} -> pos ${col}: ${v.toFixed(4)}`}
              style={{
                width: cellSize,
                height: cellSize,
                borderRadius: 2,
                background: col > row
                  ? COLORS.surface
                  : `rgba(88,166,255,${intensity * 0.9 + 0.05})`,
                fontSize: cellSize > 28 ? 8 : 0,
                display: "flex",
                alignItems: "center",
                justifyContent: "center",
                color: intensity > 0.5 ? "#000" : COLORS.textDim,
                fontFamily: "monospace",
              }}
            >
              {cellSize > 28 ? v.toFixed(2) : ""}
            </div>
          );
        })}
      </div>
    </div>
  );
}

function Section({ id, title, emoji, children, defaultOpen = false }) {
  const [open, setOpen] = useState(defaultOpen);
  return (
    <div
      style={{
        marginBottom: 16,
        border: `1px solid ${COLORS.border}`,
        borderRadius: 8,
        overflow: "hidden",
        background: COLORS.surface,
      }}
    >
      <button
        onClick={() => setOpen(!open)}
        style={{
          width: "100%",
          display: "flex",
          alignItems: "center",
          gap: 10,
          padding: "14px 18px",
          background: open ? COLORS.surfaceHover : "transparent",
          border: "none",
          borderBottom: open ? `1px solid ${COLORS.border}` : "none",
          color: COLORS.textBright,
          cursor: "pointer",
          fontSize: 15,
          fontWeight: 600,
          fontFamily: "'IBM Plex Mono', 'JetBrains Mono', monospace",
          textAlign: "left",
        }}
      >
        <span style={{ fontSize: 18 }}>{emoji}</span>
        <span style={{ flex: 1 }}>{title}</span>
        <span
          style={{
            color: COLORS.textDim,
            fontSize: 13,
            transform: open ? "rotate(90deg)" : "rotate(0deg)",
            transition: "transform 0.15s",
          }}
        >
          ▶
        </span>
      </button>
      {open && <div style={{ padding: "16px 18px" }}>{children}</div>}
    </div>
  );
}

function CodeBlock({ children }) {
  return (
    <pre
      style={{
        background: COLORS.bg,
        border: `1px solid ${COLORS.border}`,
        borderRadius: 6,
        padding: "12px 14px",
        fontFamily: "'IBM Plex Mono', monospace",
        fontSize: 12,
        color: COLORS.green,
        overflowX: "auto",
        margin: "8px 0",
        lineHeight: 1.5,
        whiteSpace: "pre-wrap",
      }}
    >
      {children}
    </pre>
  );
}

function Output({ children }) {
  return (
    <pre
      style={{
        background: "#0d1117",
        borderLeft: `3px solid ${COLORS.accent}`,
        borderRadius: "0 6px 6px 0",
        padding: "10px 14px",
        fontFamily: "'IBM Plex Mono', monospace",
        fontSize: 11.5,
        color: COLORS.text,
        overflowX: "auto",
        margin: "8px 0",
        lineHeight: 1.55,
        whiteSpace: "pre-wrap",
      }}
    >
      {children}
    </pre>
  );
}

function MathBlock({ children }) {
  return (
    <div
      style={{
        background: COLORS.purpleDim,
        border: `1px solid ${COLORS.purple}33`,
        borderRadius: 6,
        padding: "10px 14px",
        fontFamily: "'IBM Plex Mono', monospace",
        fontSize: 13,
        color: COLORS.purple,
        margin: "8px 0",
        textAlign: "center",
      }}
    >
      {children}
    </div>
  );
}

function Prose({ children }) {
  return (
    <div
      style={{
        color: COLORS.text,
        fontSize: 13.5,
        lineHeight: 1.7,
        fontFamily: "'IBM Plex Sans', 'Segoe UI', sans-serif",
        margin: "6px 0",
      }}
    >
      {children}
    </div>
  );
}

function Tag({ children, color = COLORS.accent }) {
  return (
    <span
      style={{
        display: "inline-block",
        padding: "1px 7px",
        borderRadius: 4,
        background: color + "18",
        color: color,
        fontFamily: "monospace",
        fontSize: 12,
        border: `1px solid ${color}33`,
      }}
    >
      {children}
    </span>
  );
}

// ═══════════════════════════════════════════════════════════════
// MAIN APP
// ═══════════════════════════════════════════════════════════════

export default function TransformerNotebook() {
  const rng = useMemo(() => seededRandom(42), []);
  const params = useMemo(() => initParams(seededRandom(42)), []);
  const rope = useMemo(() => computeRopeFreqs(), []);
  const inputIds = [12, 5, 41, 33, 7];
  const seqLen = inputIds.length;

  const result = useMemo(
    () => fullForward(inputIds, params, rope),
    [params, rope]
  );

  const totalParams = useMemo(() => {
    let total = 0;
    for (const key in params) total += params[key].data.length;
    return total;
  }, [params]);

  const [genTokens, setGenTokens] = useState(null);
  const genRng = useRef(seededRandom(123));

  const handleGenerate = useCallback(() => {
    const seq = [...inputIds];
    const steps = [];
    const gr = seededRandom(123);
    for (let step = 0; step < 8; step++) {
      const r = fullForward(seq, params, rope);
      const { tokenId, probs } = sampleToken(r.logits, 0.8, 10, gr);
      const topIds = Array.from({ length: CONFIG.vocabSize }, (_, i) => i)
        .sort((a, b) => r.logits[b] - r.logits[a])
        .slice(0, 5);
      steps.push({
        chosen: tokenId,
        prob: probs[tokenId],
        topIds,
        topProbs: topIds.map((id) => probs[id]),
      });
      seq.push(tokenId);
    }
    setGenTokens({ seq, steps });
  }, [params, rope]);

  // Embedding result
  const embedded = useMemo(
    () => sliceRows(params.tokEmb, inputIds),
    [params]
  );

  return (
    <div
      style={{
        minHeight: "100vh",
        background: COLORS.bg,
        color: COLORS.text,
        fontFamily: "'IBM Plex Sans', 'Segoe UI', sans-serif",
      }}
    >
      <div style={{ maxWidth: 860, margin: "0 auto", padding: "32px 20px" }}>
        {/* Header */}
        <div style={{ marginBottom: 32 }}>
          <div
            style={{
              fontSize: 11,
              textTransform: "uppercase",
              letterSpacing: 3,
              color: COLORS.accent,
              fontFamily: "'IBM Plex Mono', monospace",
              marginBottom: 8,
            }}
          >
            Interactive Notebook
          </div>
          <h1
            style={{
              fontSize: 28,
              fontWeight: 700,
              color: COLORS.textBright,
              fontFamily: "'IBM Plex Mono', 'JetBrains Mono', monospace",
              margin: 0,
              lineHeight: 1.3,
            }}
          >
            The Transformer Decoder
            <br />
            <span style={{ color: COLORS.textDim, fontWeight: 400, fontSize: 18 }}>
              Every Computation, From Scratch
            </span>
          </h1>
          <p style={{ color: COLORS.textDim, fontSize: 13.5, marginTop: 12, lineHeight: 1.6, maxWidth: 680 }}>
            A complete walkthrough of every operation inside a GPT-style autoregressive transformer
            decoder. All computations run live in your browser with no frameworks—just typed arrays and
            arithmetic. Expand each section to inspect tensor values at every stage.
          </p>
          <div style={{ display: "flex", flexWrap: "wrap", gap: 6, marginTop: 12 }}>
            <Tag>vocab=64</Tag> <Tag>d_model=32</Tag> <Tag>heads=4</Tag>
            <Tag color={COLORS.cyan}>kv_heads=2 (GQA)</Tag> <Tag>d_ff=86</Tag>
            <Tag>layers=2</Tag>
            <Tag color={COLORS.orange}>{totalParams.toLocaleString()} params</Tag>
          </div>
        </div>

        {/* Architecture */}
        <Section id="arch" title="0 — Architecture Overview" emoji="🏗️" defaultOpen={true}>
          <CodeBlock>{`Input Token IDs: [12, 5, 41, 33, 7]
       |
       v
 Token Embedding            (lookup table: vocab_size x d_model)
       |
  [x N_LAYERS]
  +----+-------------------------------+
  | RMSNorm                            |
  | Q, K, V Projections                |
  |   Q: (seq, d_model) -> (seq, 4, 8) |   4 query heads
  |   K: (seq, d_model) -> (seq, 2, 8) |   2 KV heads (GQA)
  |   V: (seq, d_model) -> (seq, 2, 8) |
  | RoPE on Q, K                       |
  | GQA Expand K,V to 4 heads          |
  | Scaled Dot-Product Attention       |
  |   QK^T / sqrt(8) + causal mask    |
  |   Softmax -> weighted sum of V     |
  | Output Projection + Residual       |
  +----+-------------------------------+
  | RMSNorm                            |
  | SwiGLU FFN                         |
  |   gate = SiLU(x @ W_gate)         |
  |   up   = x @ W_up                 |
  |   out  = (gate * up) @ W_down     |
  | + Residual                         |
  +----+-------------------------------+
       |
  Final RMSNorm
  LM Head -> (64,) logits
  Temperature / Top-k / Softmax -> Sample`}
          </CodeBlock>
        </Section>

        {/* 1. Embedding */}
        <Section id="emb" title="1 — Token Embedding" emoji="📖">
          <Prose>
            Convert integer token IDs to dense vectors via table lookup. Row <Tag>i</Tag> of the
            embedding matrix <Tag>E ∈ R^(64×32)</Tag> becomes the vector for token <Tag>i</Tag>.
          </Prose>
          <MathBlock>x_t = E[token_t] &nbsp;&nbsp; ∈ R^(d_model)</MathBlock>
          <Output>
{`Input token IDs: [${inputIds.join(", ")}]
Embedding matrix: shape=(${CONFIG.vocabSize}, ${CONFIG.dModel})
After lookup:     shape=(${seqLen}, ${CONFIG.dModel})

Embedding for token ${inputIds[0]}:
  ${fmtArr(embedded.data.slice(0, CONFIG.dModel), 10)}`}
          </Output>
        </Section>

        {/* 2. RoPE */}
        <Section id="rope" title="2 — Rotary Positional Embeddings (RoPE)" emoji="🌀">
          <Prose>
            RoPE encodes position by <strong>rotating</strong> pairs of dimensions in Q and K vectors.
            For dimension pair (2i, 2i+1) at position t:
          </Prose>
          <MathBlock>
            θ_i = 1 / (10000^(2i/d_head)) &nbsp;&nbsp;|&nbsp;&nbsp;
            [q'_even, q'_odd] = Rot(t·θ_i) · [q_even, q_odd]
          </MathBlock>
          <Prose>
            The dot product Q·K then naturally depends only on <strong>relative</strong> position—no
            extra learnable parameters needed.
          </Prose>
          <Output>
{`Dimension pairs: ${CONFIG.dHead / 2}
Frequencies (θ_i): ${fmtArr(rope.freqs, 4, 6)}

Rotation angles at position 0: all zeros (no rotation)
Rotation angles at position 1: ${fmtArr(rope.freqs, 4, 6)}

Demo: unit vector through RoPE at each position:
  pos 0: [${Array.from(rope.cosTable.data.slice(0, 4)).map(v => v.toFixed(4)).join(", ")}...] (cos)
  pos 3: [${Array.from(rope.cosTable.data.slice(12, 16)).map(v => v.toFixed(4)).join(", ")}...] (cos)`}
          </Output>
        </Section>

        {/* 3. RMSNorm */}
        <Section id="rmsnorm" title="3 — RMSNorm" emoji="📏">
          <Prose>
            Modern transformers use <strong>RMSNorm</strong> instead of LayerNorm. Simpler: no mean
            subtraction, no bias—just divide by root-mean-square and scale.
          </Prose>
          <MathBlock>
            RMSNorm(x) = (x / RMS(x)) · γ &nbsp;&nbsp;where&nbsp;&nbsp; RMS(x) = √(mean(x²) + ε)
          </MathBlock>
          {(() => {
            const xn = rmsnorm(embedded, params.l0_attnNorm);
            const rmsB = Math.sqrt(
              Array.from(embedded.data.slice(0, CONFIG.dModel)).reduce((s, v) => s + v * v, 0) / CONFIG.dModel
            );
            const rmsA = Math.sqrt(
              Array.from(xn.data.slice(0, CONFIG.dModel)).reduce((s, v) => s + v * v, 0) / CONFIG.dModel
            );
            return (
              <Output>
{`Before RMSNorm (token 0):
  RMS = ${rmsB.toFixed(6)}
  values: ${fmtArr(embedded.data.slice(0, CONFIG.dModel), 8)}

After RMSNorm (token 0):
  RMS = ${rmsA.toFixed(6)}  (≈ 1.0 with γ=1)
  values: ${fmtArr(xn.data.slice(0, CONFIG.dModel), 8)}`}
              </Output>
            );
          })()}
        </Section>

        {/* 4. Q, K, V Projections */}
        <Section id="qkv" title="4 — Q, K, V Projections + GQA" emoji="🔑">
          <Prose>
            Three linear projections create Query, Key, and Value tensors. In <strong>Grouped-Query
            Attention</strong>, K and V have fewer heads (2) than Q (4). Each KV head is
            shared by 2 query heads—halving KV-cache memory.
          </Prose>
          <MathBlock>
            Q = xW_q ∈ R^(seq, 4, 8) &nbsp;|&nbsp; K = xW_k ∈ R^(seq, 2, 8) &nbsp;|&nbsp;
            V = xW_v ∈ R^(seq, 2, 8)
          </MathBlock>
          <Output>
{`Projection weights:
  W_q: (${CONFIG.dModel}, ${CONFIG.nHeads * CONFIG.dHead})    -> Q has ${CONFIG.nHeads} heads
  W_k: (${CONFIG.dModel}, ${CONFIG.nKvHeads * CONFIG.dHead})   -> K has ${CONFIG.nKvHeads} heads
  W_v: (${CONFIG.dModel}, ${CONFIG.nKvHeads * CONFIG.dHead})   -> V has ${CONFIG.nKvHeads} heads

GQA expansion: repeat each KV head ${CONFIG.nHeads / CONFIG.nKvHeads}x
  K: (${seqLen}, 2, 8) -> (${seqLen}, 4, 8)
  V: (${seqLen}, 2, 8) -> (${seqLen}, 4, 8)
  KV head 0 serves query heads [0, 1]
  KV head 1 serves query heads [2, 3]`}
          </Output>
        </Section>

        {/* 5. Attention */}
        <Section id="attn" title="5 — Scaled Dot-Product Attention + Causal Mask" emoji="👁️" defaultOpen={true}>
          <Prose>
            The core computation: each position queries all previous positions (causal masking prevents
            attending to the future).
          </Prose>
          <MathBlock>
            Attention(Q,K,V) = softmax(QK^T / √d_head + CausalMask) · V
          </MathBlock>
          <Prose>
            <strong>Causal mask</strong> — position i can only attend to positions 0..i:
          </Prose>
          <div style={{ fontFamily: "monospace", fontSize: 11, margin: "8px 0", color: COLORS.green }}>
            {Array.from({ length: seqLen }, (_, i) => (
              <div key={i}>
                {"  pos " + i + ": ["}
                {Array.from({ length: seqLen }, (_, j) =>
                  j <= i ? (
                    <span key={j} style={{ color: COLORS.green }}> ✓ </span>
                  ) : (
                    <span key={j} style={{ color: COLORS.red }}> ✗ </span>
                  )
                )}
                {"]"}
              </div>
            ))}
          </div>
          <Prose>
            <strong>Attention weights</strong> after softmax (Layer 0) — brighter = higher weight:
          </Prose>
          <div style={{ overflowX: "auto", display: "flex", flexWrap: "wrap" }}>
            {Array.from({ length: CONFIG.nHeads }, (_, h) => (
              <AttnHeatmap
                key={h}
                weights={result.layers[0].attnW}
                seqLen={seqLen}
                headIdx={h}
              />
            ))}
          </div>
          <Prose style={{ marginTop: 8 }}>
            <strong>Layer 1</strong> attention patterns (deeper layer, more refined):
          </Prose>
          <div style={{ overflowX: "auto", display: "flex", flexWrap: "wrap" }}>
            {Array.from({ length: CONFIG.nHeads }, (_, h) => (
              <AttnHeatmap
                key={h}
                weights={result.layers[1].attnW}
                seqLen={seqLen}
                headIdx={h}
              />
            ))}
          </div>
        </Section>

        {/* 6. SwiGLU FFN */}
        <Section id="ffn" title="6 — SwiGLU Feed-Forward Network" emoji="⚡">
          <Prose>
            After attention, each position passes through a gated feed-forward network. SwiGLU uses
            a learned gate to control which features are activated:
          </Prose>
          <MathBlock>
            FFN(x) = (SiLU(x·W_gate) ⊙ x·W_up) · W_down
          </MathBlock>
          <Prose>
            Where <Tag>SiLU(x) = x · σ(x)</Tag> is a smooth, non-monotonic activation
            (unlike ReLU). The gate learns to selectively pass information.
          </Prose>
          <Output>
{`SiLU activation (smooth gating):
  SiLU(-2.0) = ${(-2 / (1 + Math.exp(2))).toFixed(4)}   (small negative allowed)
  SiLU(-1.0) = ${(-1 / (1 + Math.exp(1))).toFixed(4)}
  SiLU( 0.0) = ${(0).toFixed(4)}
  SiLU( 1.0) = ${(1 / (1 + Math.exp(-1))).toFixed(4)}
  SiLU( 2.0) = ${(2 / (1 + Math.exp(-2))).toFixed(4)}

Weight shapes:
  W_gate: (${CONFIG.dModel}, ${CONFIG.dFf})   gate projection
  W_up:   (${CONFIG.dModel}, ${CONFIG.dFf})   up projection
  W_down: (${CONFIG.dFf}, ${CONFIG.dModel})   down projection

  3 matrices instead of 2 -> d_ff ≈ (8/3)·d_model to match param count`}
          </Output>
        </Section>

        {/* 7. Full Pass */}
        <Section id="layers" title="7 — Full Forward Pass (Both Layers)" emoji="🔄">
          <Prose>
            Each layer applies: <Tag>RMSNorm → GQA Attention → +Residual → RMSNorm → SwiGLU → +Residual</Tag>
          </Prose>
          <Output>
{result.layers.map((layer, i) => {
  const hNorm = norm(layer.h);
  return `Layer ${i}:
  Output norm: ${hNorm.toFixed(4)}
  Hidden[0, :8]: ${fmtArr(layer.h.data.slice(0, 8), 8)}
`;
}).join("\n")}
{`Final RMSNorm applied.
Last token hidden state -> LM Head projection -> logits`}
          </Output>
        </Section>

        {/* 8. LM Head + Logits */}
        <Section id="logits" title="8 — LM Head: Logits over Vocabulary" emoji="🎯" defaultOpen={true}>
          <Prose>
            The last token's normalized hidden state is projected to a <Tag>vocab_size=64</Tag> logit vector.
            Each logit is the raw (unnormalized) score for that token being "next."
          </Prose>
          <MathBlock>
            logits = RMSNorm(h_last) · W_head &nbsp;&nbsp; ∈ R^(vocab_size)
          </MathBlock>
          {(() => {
            const logits = result.logits;
            const indices = Array.from({ length: CONFIG.vocabSize }, (_, i) => i)
              .sort((a, b) => logits[b] - logits[a]);
            const probs = softmax1D(logits);
            return (
              <>
                <Output>
{`Logits shape: (${CONFIG.vocabSize},)
Range: [${Math.min(...logits).toFixed(4)}, ${Math.max(...logits).toFixed(4)}]

Top 10 next-token candidates:
${indices.slice(0, 10).map((id, rank) =>
  `  #${rank + 1}  token=${String(id).padStart(2)} | logit=${logits[id].toFixed(4).padStart(8)} | prob=${(probs[id] * 100).toFixed(2).padStart(6)}%`
).join("\n")}`}
                </Output>
                <Prose>Logit distribution (bar = probability after softmax):</Prose>
                <div style={{ margin: "8px 0", maxHeight: 200, overflowY: "auto" }}>
                  {indices.slice(0, 20).map((id) => (
                    <div
                      key={id}
                      style={{
                        display: "flex",
                        alignItems: "center",
                        gap: 6,
                        marginBottom: 2,
                        fontFamily: "monospace",
                        fontSize: 11,
                      }}
                    >
                      <span style={{ width: 30, color: COLORS.textDim, textAlign: "right" }}>
                        {id}
                      </span>
                      <div
                        style={{
                          height: 14,
                          width: `${Math.max(probs[id] * 600, 1)}px`,
                          background: `linear-gradient(90deg, ${COLORS.accent}, ${COLORS.cyan})`,
                          borderRadius: 2,
                          transition: "width 0.3s",
                        }}
                      />
                      <span style={{ color: COLORS.textDim }}>
                        {(probs[id] * 100).toFixed(1)}%
                      </span>
                    </div>
                  ))}
                </div>
              </>
            );
          })()}
        </Section>

        {/* 9. Sampling */}
        <Section id="sampling" title="9 — Sampling: Temperature, Top-k, Softmax" emoji="🎲">
          <Prose>
            Raw logits are transformed into a probability distribution for sampling:
          </Prose>
          <MathBlock>
            p_i = exp(z_i / T) / Σ exp(z_j / T) &nbsp;&nbsp; (after top-k filtering)
          </MathBlock>
          {(() => {
            const logits = result.logits;
            const rawProbs = softmax1D(logits);
            const scaled = logits.map((v) => v / 0.8);
            const scaledProbs = softmax1D(scaled);
            const indices = Array.from({ length: CONFIG.vocabSize }, (_, i) => i)
              .sort((a, b) => logits[b] - logits[a]);
            return (
              <Output>
{`Temperature=0.8 (sharpens distribution):
${indices.slice(0, 5).map((id) =>
  `  token ${String(id).padStart(2)}: prob ${(rawProbs[id]*100).toFixed(2)}% -> ${(scaledProbs[id]*100).toFixed(2)}%`
).join("\n")}

Top-k=10: keep only 10 highest-prob tokens, zero the rest.
  ${CONFIG.vocabSize} tokens -> 10 candidates

Final: renormalize over candidates, then sample.`}
              </Output>
            );
          })()}
        </Section>

        {/* 10. Autoregressive Generation */}
        <Section id="generate" title="10 — Autoregressive Generation" emoji="🚀" defaultOpen={true}>
          <Prose>
            Feed the model its own output, one token at a time. Each step runs the full forward pass
            on the growing sequence. (A real system uses KV-cache to avoid recomputing past tokens.)
          </Prose>
          <div style={{ margin: "12px 0" }}>
            <button
              onClick={handleGenerate}
              style={{
                padding: "10px 24px",
                background: `linear-gradient(135deg, ${COLORS.accent}, ${COLORS.cyan})`,
                color: "#000",
                border: "none",
                borderRadius: 6,
                fontSize: 14,
                fontWeight: 700,
                fontFamily: "'IBM Plex Mono', monospace",
                cursor: "pointer",
                letterSpacing: 0.5,
              }}
            >
              ▶ Generate 8 Tokens
            </button>
          </div>
          {genTokens && (
            <>
              <Output>
{`Starting: [${inputIds.join(", ")}]
Generating 8 tokens (temperature=0.8, top_k=10)...

${genTokens.steps.map((s, i) =>
  `Step ${i + 1}: chose token ${String(s.chosen).padStart(2)} `
  + `(p=${(s.prob * 100).toFixed(1)}%) `
  + `| top: ${s.topIds.slice(0, 3).map((id, j) => `${id}=${(s.topProbs[j]*100).toFixed(1)}%`).join(", ")}`
).join("\n")}

Final sequence: [${genTokens.seq.join(", ")}]`}
              </Output>
              <Prose>
                Sequence tokens (original in blue, generated in green):
              </Prose>
              <div style={{ display: "flex", flexWrap: "wrap", gap: 4, margin: "8px 0" }}>
                {genTokens.seq.map((tok, i) => (
                  <span
                    key={i}
                    style={{
                      padding: "4px 10px",
                      borderRadius: 4,
                      fontFamily: "monospace",
                      fontSize: 13,
                      fontWeight: 600,
                      background: i < inputIds.length ? COLORS.accentDim : COLORS.greenDim,
                      color: i < inputIds.length ? COLORS.accent : COLORS.green,
                      border: `1px solid ${
                        i < inputIds.length ? COLORS.accent + "44" : COLORS.green + "44"
                      }`,
                    }}
                  >
                    {tok}
                  </span>
                ))}
              </div>
            </>
          )}
        </Section>

        {/* 11. KV-Cache */}
        <Section id="kvcache" title="11 — KV-Cache (Efficient Inference)" emoji="💾">
          <Prose>
            During generation, K and V for past positions do not change. The <strong>KV-cache</strong> stores
            them to avoid recomputation:
          </Prose>
          <CodeBlock>{`Step 1: input = [A, B, C]
  Compute & cache K₀,V₀  K₁,V₁  K₂,V₂
  Q₂ attends to all cached KV

Step 2: input = [D] (only the NEW token)
  Compute K₃,V₃ and APPEND to cache
  Q₃ attends to K₀:₃, V₀:₃

Per-step cost drops from O(n²) to O(n).`}
          </CodeBlock>
          <Output>
{`KV-Cache memory per layer (this model):
  K: (seq_len, ${CONFIG.nKvHeads}, ${CONFIG.dHead}) x float32
  V: (seq_len, ${CONFIG.nKvHeads}, ${CONFIG.dHead}) x float32
  = ${2 * CONFIG.nKvHeads * CONFIG.dHead * 4} bytes per token per layer

Scaled to LLaMA-7B (d=4096, 32 layers, 32 kv_heads, d_head=128):
  = ${(2 * 32 * 128 * 4 * 32 / 1e6).toFixed(1)} MB per token
  = ${(2 * 32 * 128 * 4 * 32 * 4096 / 1e9).toFixed(1)} GB at seq_len=4096`}
          </Output>
        </Section>

        {/* 12. FLOPs Summary */}
        <Section id="flops" title="12 — Computation & Memory Summary" emoji="📊">
          {(() => {
            const S = seqLen;
            const D = CONFIG.dModel;
            const H = CONFIG.nHeads;
            const KV = CONFIG.nKvHeads;
            const Dh = CONFIG.dHead;
            const F = CONFIG.dFf;
            const qkvFlops = S * D * (H + 2 * KV) * Dh * 2;
            const attnScoreFlops = H * S * S * Dh * 2;
            const attnValFlops = H * S * S * Dh * 2;
            const outProjFlops = S * H * Dh * D * 2;
            const ffnFlops = S * (2 * D * F + F * D) * 2;
            const layerFlops = qkvFlops + attnScoreFlops + attnValFlops + outProjFlops + ffnFlops;
            const totalFlops = layerFlops * CONFIG.nLayers + D * CONFIG.vocabSize * 2;
            return (
              <Output>
{`FLOPs per layer (seq_len=${S}):
  QKV Projections:      ${qkvFlops.toLocaleString().padStart(10)}
  Attention (QK^T):     ${attnScoreFlops.toLocaleString().padStart(10)}
  Attention (x V):      ${attnValFlops.toLocaleString().padStart(10)}
  Output Projection:    ${outProjFlops.toLocaleString().padStart(10)}
  SwiGLU FFN:           ${ffnFlops.toLocaleString().padStart(10)}
  ─────────────────────────────────
  Per layer:            ${layerFlops.toLocaleString().padStart(10)}
  x ${CONFIG.nLayers} layers:            ${(layerFlops * CONFIG.nLayers).toLocaleString().padStart(10)}
  + LM Head:            ${(D * CONFIG.vocabSize * 2).toLocaleString().padStart(10)}
  ═══════════════════════════════════
  TOTAL:                ${totalFlops.toLocaleString().padStart(10)} FLOPs

Key insight: Attention is O(n²·d), FFN is O(n·d²).
  Short sequences: FFN dominates
  Long sequences:  Attention dominates`}
              </Output>
            );
          })()}
        </Section>

        {/* Summary */}
        <Section id="summary" title="13 — Summary & Key Takeaways" emoji="🎓" defaultOpen={true}>
          <div style={{ display: "grid", gap: 8 }}>
            {[
              ["Everything is matmuls", "Linear projections and dot products — that's why GPUs excel at this.", COLORS.accent],
              ["RoPE encodes position via rotation", "Q·K dot products naturally depend on relative position, no learnable params.", COLORS.cyan],
              ["GQA halves KV-cache memory", "Multiple query heads share K,V heads. Same quality, half the memory.", COLORS.green],
              ["SwiGLU > ReLU", "Learned gating selects which features to activate. 3 matrices, better expressivity.", COLORS.orange],
              ["RMSNorm is the new standard", "Simpler than LayerNorm (no mean, no bias), same quality.", COLORS.purple],
              ["KV-cache is essential", "Without it, generation cost is O(n²) per token. With it, O(n).", COLORS.red],
              ["Causal mask = decoder", "Each position only sees the past — this is what makes it autoregressive.", COLORS.textBright],
            ].map(([title, desc, color], i) => (
              <div
                key={i}
                style={{
                  display: "flex",
                  gap: 12,
                  padding: "10px 14px",
                  background: color + "0a",
                  border: `1px solid ${color}22`,
                  borderRadius: 6,
                }}
              >
                <span style={{ color, fontWeight: 700, fontSize: 14, fontFamily: "monospace", minWidth: 20 }}>
                  {i + 1}.
                </span>
                <div>
                  <div style={{ color, fontWeight: 600, fontSize: 13, marginBottom: 2 }}>{title}</div>
                  <div style={{ color: COLORS.textDim, fontSize: 12.5 }}>{desc}</div>
                </div>
              </div>
            ))}
          </div>
          <div
            style={{
              marginTop: 16,
              padding: "12px 16px",
              background: COLORS.bg,
              borderRadius: 6,
              border: `1px solid ${COLORS.border}`,
              fontSize: 12,
              color: COLORS.textDim,
              fontFamily: "'IBM Plex Mono', monospace",
              textAlign: "center",
            }}
          >
            Built with typed arrays. No frameworks, no magic — just math.
          </div>
        </Section>
      </div>
    </div>
  );
}
