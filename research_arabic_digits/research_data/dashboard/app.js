(function () {
  "use strict";

  const MODEL_COLORS = {
    "MORE-HD": "#46d6c3",
    "MORE-HD-C": "#ffb36b",
    "MORE-HD-60P": "#9b8cff",
    "MORE-HD-C-FixedRZ": "#f472b6",
    "MORE-REPRO": "#6aa8ff",
    "A": "#46d6c3",
    "B": "#9b8cff",
    "C": "#f472b6",
    "D": "#ffb36b",
    "M": "#6aa8ff"
  };
  const FALLBACK_COLORS = ["#46d6c3", "#ffb36b", "#9b8cff", "#f472b6", "#6aa8ff", "#64d49a", "#ffd166"];
  const TRACK_LABELS = {
    PRIMARY: "Utama",
    ABLATION: "Ablasi",
    MORE_REFERENCE: "Referensi MORE"
  };
  const COMPARISON_DIMENSIONS = {
    architecture: "Sirkuit / arsitektur",
    feature_method: "Representasi fitur",
    class: "Kelas",
    K: "Jumlah kelas (K)"
  };
  const CLASS_METRICS = new Set(["class_accuracy", "precision", "recall", "f1"]);
  const COMPARISON_METRICS = {
    class_accuracy: { label: "Akurasi kelas", percent: true },
    precision: { label: "Precision kelas", percent: true },
    recall: { label: "Recall kelas", percent: true },
    f1: { label: "F1 kelas", percent: true },
    accuracy: { label: "Akurasi model", percent: true },
    f1_macro: { label: "Macro-F1 model", percent: true },
    min_separation_ratio: { label: "Rasio separasi" },
    total_runtime_sec: { label: "Runtime", runtime: true, lowerBetter: true }
  };
  const VIEW_COPY = {
    overview: ["Ringkasan eksperimen", "Kemampuan model pada seluruh protokol", "Gambaran kinerja, stabilitas, dan biaya komputasi dari data aktual."],
    comparison: ["Perbandingan model", "Bandingkan arsitektur secara berdampingan", "Gunakan filter yang sama untuk menilai akurasi, F1, separasi, dan runtime."],
    classes: ["Analisis per kelas", "Temukan kelas yang kuat dan yang sering tertukar", "Metrik per kelas dan confusion matrix dihitung ulang dari run yang sedang aktif."],
    training: ["Dinamika training", "Ikuti setiap evaluasi optimizer", "Bandingkan kurva loss dan metrik evaluasi lengkap hingga empat run sekaligus."],
    circuits: ["Sirkuit dan representasi", "Hubungkan kompleksitas, fitur, dan hasil model", "Microbenchmark, depth sirkuit, dan sensitivitas struktur dalam satu ruang analisis."],
    statistics: ["Bukti statistik", "Ukur besar efek dan ketidakpastiannya", "Kontras berpasangan, koreksi multipel, McNemar, dan diagnosis pra-spesifikasi kelas 6–9 (٦/٩)."],
    runs: ["Penjelajah run", "Audit seluruh hasil training", "Cari, urutkan, dan ekspor run beserta metrik akhirnya."],
    findings: ["Temuan & replikasi", "Apakah kesimpulan MNIST bertahan pada digit Arab?", "Ringkasan konfirmatori, perbandingan deskriptif dengan MNIST, Jalur B, dan analisis eksploratif."]
  };
  const GLYPHS = "٠١٢٣٤٥٦٧٨٩";
  const LRM = "\u200e";
  const glyph = value => {
    const k = Number(value);
    return Number.isInteger(k) && k >= 0 && k < GLYPHS.length ? GLYPHS[k] + LRM : "";
  };
  const classLabel = value => "Kelas " + value + (glyph(value) ? " (" + glyph(value) + ")" : "");
  const classShort = value => String(value) + (glyph(value) ? " " + glyph(value) : "");

  const state = {
    data: null,
    track: "ALL",
    model: "ALL",
    feature: "ALL",
    k: "ALL",
    seed: "ALL",
    view: "overview",
    overviewMetric: "accuracy",
    comparisonMetric: "class_accuracy",
    comparisonDimensions: new Set(["architecture", "feature_method", "class"]),
    comparisonWeight: "macro",
    comparisonMinN: 1,
    comparisonGroups: [],
    compareModels: new Set(),
    classMetric: "class_accuracy",
    normalizeConfusion: true,
    trainingPhase: "supervised",
    trainingMetric: "train_loss",
    selectedRunKeys: [],
    historyCache: new Map(),
    historyLoading: new Map(),
    evalPage: 0,
    detailType: "config",
    detailPage: 0,
    runPage: 0,
    runSearch: "",
    runSort: "accuracy-desc"
  };

  const $ = (selector, root) => (root || document).querySelector(selector);
  const $$ = (selector, root) => Array.from((root || document).querySelectorAll(selector));
  const clamp = (value, min, max) => Math.max(min, Math.min(max, value));
  const sum = values => values.reduce((acc, value) => acc + (Number.isFinite(+value) ? +value : 0), 0);
  const valid = values => values.map(Number).filter(Number.isFinite);
  const mean = values => {
    const nums = valid(values);
    return nums.length ? sum(nums) / nums.length : null;
  };
  const median = values => {
    const nums = valid(values).sort((a, b) => a - b);
    if (!nums.length) return null;
    const mid = Math.floor(nums.length / 2);
    return nums.length % 2 ? nums[mid] : (nums[mid - 1] + nums[mid]) / 2;
  };
  const sd = values => {
    const nums = valid(values);
    if (nums.length < 2) return 0;
    const avg = mean(nums);
    return Math.sqrt(nums.reduce((acc, value) => acc + Math.pow(value - avg, 2), 0) / (nums.length - 1));
  };
  const minmax = values => {
    const nums = valid(values);
    return nums.length ? [Math.min(...nums), Math.max(...nums)] : [0, 1];
  };
  const esc = value => String(value == null ? "—" : value)
    .replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;").replace(/'/g, "&#039;");
  const modelColor = (name, index) => MODEL_COLORS[name] || FALLBACK_COLORS[index % FALLBACK_COLORS.length];
  const modelLabel = item => {
    if (!item) return "—";
    const code = item.model_code && item.model_code !== item.architecture ? item.model_code + " · " : "";
    return code + (item.architecture || item.model || "—");
  };
  const fmt = (value, digits) => {
    if (value == null || !Number.isFinite(+value)) return "—";
    const abs = Math.abs(+value);
    if (abs !== 0 && (abs < 0.0001 || abs >= 100000)) return (+value).toExponential(2);
    return new Intl.NumberFormat("id-ID", { maximumFractionDigits: digits == null ? 3 : digits }).format(+value);
  };
  const pct = (value, digits) => value == null || !Number.isFinite(+value)
    ? "—"
    : new Intl.NumberFormat("id-ID", { style: "percent", maximumFractionDigits: digits == null ? 1 : digits }).format(+value);
  const seconds = value => {
    if (value == null || !Number.isFinite(+value)) return "—";
    const v = +value;
    if (v < 1) return fmt(v * 1000, 1) + " ms";
    if (v < 60) return fmt(v, 1) + " dtk";
    if (v < 3600) return fmt(v / 60, 1) + " mnt";
    return fmt(v / 3600, 1) + " jam";
  };
  const shortTrack = value => TRACK_LABELS[value] || value;

  function setHTML(selector, html) {
    const node = $(selector);
    if (node) node.innerHTML = html;
  }

  function unique(values) {
    return Array.from(new Set(values.filter(value => value != null && value !== "")));
  }

  function filteredRuns() {
    if (!state.data) return [];
    return state.data.runs.filter(run => {
      return (state.track === "ALL" || run.track === state.track)
        && (state.model === "ALL" || run.architecture === state.model)
        && (state.feature === "ALL" || run.feature_method === state.feature)
        && (state.k === "ALL" || String(run.K) === String(state.k))
        && (state.seed === "ALL" || String(run.seed) === String(state.seed));
    });
  }

  function filterRows(rows) {
    return rows.filter(row => {
      return (state.track === "ALL" || row.track === state.track)
        && (state.model === "ALL" || row.architecture === state.model)
        && (state.feature === "ALL" || row.feature_method === state.feature)
        && (state.k === "ALL" || String(row.K) === String(state.k))
        && (state.seed === "ALL" || String(row.seed) === String(state.seed));
    });
  }

  function groupBy(rows, keyFn) {
    const map = new Map();
    rows.forEach(row => {
      const key = keyFn(row);
      if (!map.has(key)) map.set(key, []);
      map.get(key).push(row);
    });
    return map;
  }

  function summarizeConditions(runs) {
    return Array.from(groupBy(runs, run => [run.track, run.architecture, run.feature_method, run.K].join("¦")).values())
      .map(group => ({
        track: group[0].track,
        model_code: group[0].model_code,
        architecture: group[0].architecture,
        feature_method: group[0].feature_method,
        K: group[0].K,
        n: group.length,
        accuracy: mean(group.map(row => row.accuracy)),
        accuracy_sd: sd(group.map(row => row.accuracy)),
        f1_macro: mean(group.map(row => row.f1_macro)),
        f1_sd: sd(group.map(row => row.f1_macro)),
        min_separation_ratio: mean(group.map(row => row.min_separation_ratio)),
        total_runtime_sec: mean(group.map(row => row.total_runtime_sec)),
        circuit_depth: mean(group.map(row => row.circuit_depth))
      }));
  }

  function colorForSeries(label, index) {
    const known = Object.keys(MODEL_COLORS).find(key => String(label).includes(key));
    return known ? MODEL_COLORS[known] : FALLBACK_COLORS[index % FALLBACK_COLORS.length];
  }

  function svgNode(tag, attrs, text) {
    const node = document.createElementNS("http://www.w3.org/2000/svg", tag);
    Object.entries(attrs || {}).forEach(([key, value]) => node.setAttribute(key, value));
    if (text != null) node.textContent = text;
    return node;
  }

  function chartBase(container, height) {
    container.innerHTML = "";
    const width = 800;
    const svg = svgNode("svg", { viewBox: "0 0 " + width + " " + height, role: "img", preserveAspectRatio: "xMidYMid meet" });
    container.appendChild(svg);
    return { svg, width, height };
  }

  function chartEmpty(container, message) {
    container.innerHTML = '<div class="chart-empty">' + esc(message || "Tidak ada data untuk kombinasi filter ini.") + "</div>";
  }

  function niceTicks(min, max, count) {
    if (!Number.isFinite(min) || !Number.isFinite(max)) return [0, 1];
    if (min === max) return [min];
    const span = max - min;
    const raw = span / Math.max(1, count - 1);
    const power = Math.pow(10, Math.floor(Math.log10(raw)));
    const error = raw / power;
    const factor = error >= 7.5 ? 10 : error >= 3.5 ? 5 : error >= 1.5 ? 2 : 1;
    const step = factor * power;
    const start = Math.floor(min / step) * step;
    const end = Math.ceil(max / step) * step;
    const ticks = [];
    for (let value = start; value <= end + step * 0.1; value += step) ticks.push(+value.toFixed(12));
    return ticks;
  }

  function renderLineChart(container, series, options) {
    const usable = series.filter(item => item.points && item.points.some(point => Number.isFinite(+point.y)));
    if (!usable.length) return chartEmpty(container);
    const height = (options && options.height) || 360;
    const { svg, width } = chartBase(container, height);
    const margin = { top: 34, right: 20, bottom: 52, left: 62 };
    const plotW = width - margin.left - margin.right;
    const plotH = height - margin.top - margin.bottom;
    const xs = usable.flatMap(item => item.points.map(point => +point.x)).filter(Number.isFinite);
    const ys = usable.flatMap(item => item.points.map(point => +point.y)).filter(Number.isFinite);
    let [xMin, xMax] = minmax(xs);
    let [yMin, yMax] = minmax(ys);
    if (options && options.yDomain) [yMin, yMax] = options.yDomain;
    if (yMin === yMax) { yMin -= 1; yMax += 1; }
    const pad = (yMax - yMin) * 0.08;
    if (!(options && options.noYPad)) { yMin -= pad; yMax += pad; }
    const xScale = value => margin.left + ((value - xMin) / (xMax - xMin || 1)) * plotW;
    const yScale = value => margin.top + plotH - ((value - yMin) / (yMax - yMin || 1)) * plotH;

    niceTicks(yMin, yMax, 6).forEach(tick => {
      const y = yScale(tick);
      if (y < margin.top - 1 || y > margin.top + plotH + 1) return;
      svg.appendChild(svgNode("line", { x1: margin.left, y1: y, x2: width - margin.right, y2: y, class: "grid-line" }));
      svg.appendChild(svgNode("text", { x: margin.left - 10, y: y + 4, "text-anchor": "end", class: "axis-text" }, options && options.yFormat ? options.yFormat(tick) : fmt(tick, 2)));
    });
    const xTicks = unique(xs).sort((a, b) => a - b);
    const shown = xTicks.length <= 10 ? xTicks : xTicks.filter((_, index) => index % Math.ceil(xTicks.length / 7) === 0 || index === xTicks.length - 1);
    shown.forEach(tick => {
      const x = xScale(tick);
      svg.appendChild(svgNode("line", { x1: x, y1: margin.top + plotH, x2: x, y2: margin.top + plotH + 5, class: "axis-line" }));
      svg.appendChild(svgNode("text", { x, y: height - 24, "text-anchor": "middle", class: "axis-text" }, options && options.xFormat ? options.xFormat(tick) : fmt(tick, 0)));
    });
    svg.appendChild(svgNode("line", { x1: margin.left, y1: margin.top + plotH, x2: width - margin.right, y2: margin.top + plotH, class: "axis-line" }));
    if (options && options.xLabel) svg.appendChild(svgNode("text", { x: margin.left + plotW / 2, y: height - 5, "text-anchor": "middle", class: "axis-text" }, options.xLabel));

    usable.forEach((item, seriesIndex) => {
      const sorted = item.points.filter(point => Number.isFinite(+point.x) && Number.isFinite(+point.y)).sort((a, b) => a.x - b.x);
      const color = item.color || colorForSeries(item.label, seriesIndex);
      const d = sorted.map((point, index) => (index ? "L" : "M") + xScale(+point.x).toFixed(2) + "," + yScale(+point.y).toFixed(2)).join(" ");
      const path = svgNode("path", { d, fill: "none", stroke: color, "stroke-width": item.width || 2.4, "stroke-linejoin": "round", "stroke-linecap": "round", opacity: item.opacity || 0.95 });
      if (item.dashed) path.setAttribute("stroke-dasharray", "7 5");
      path.appendChild(svgNode("title", {}, item.label));
      svg.appendChild(path);
      if (sorted.length <= 16) sorted.forEach(point => {
        const circle = svgNode("circle", { cx: xScale(+point.x), cy: yScale(+point.y), r: 3.6, fill: color, stroke: "#07131f", "stroke-width": 1.5 });
        circle.appendChild(svgNode("title", {}, item.label + "\n" + (options && options.xLabel ? options.xLabel + ": " : "") + fmt(point.x, 0) + "\nNilai: " + (options && options.yFormat ? options.yFormat(point.y) : fmt(point.y, 4))));
        svg.appendChild(circle);
      });
    });

    const legend = usable.slice(0, 8);
    let legendX = margin.left;
    legend.forEach((item, index) => {
      const label = item.label.length > 24 ? item.label.slice(0, 23) + "…" : item.label;
      const widthGuess = label.length * 6.3 + 28;
      if (legendX + widthGuess > width - margin.right) legendX = margin.left;
      svg.appendChild(svgNode("line", { x1: legendX, y1: 15, x2: legendX + 16, y2: 15, stroke: item.color || colorForSeries(item.label, index), "stroke-width": 3, "stroke-dasharray": item.dashed ? "5 4" : "" }));
      svg.appendChild(svgNode("text", { x: legendX + 21, y: 19, class: "legend-text" }, label));
      legendX += widthGuess;
    });
  }

  function renderBarChart(container, items, options) {
    if (!items.length) return chartEmpty(container);
    const height = (options && options.height) || Math.max(300, items.length * 30 + 80);
    const { svg, width } = chartBase(container, height);
    const margin = { top: 26, right: 54, bottom: 44, left: (options && options.left) || 145 };
    const plotW = width - margin.left - margin.right;
    const plotH = height - margin.top - margin.bottom;
    let [minValue, maxValue] = minmax(items.map(item => item.value));
    if (options && options.domain) [minValue, maxValue] = options.domain;
    minValue = options && options.includeZero === false ? minValue : Math.min(0, minValue);
    maxValue = Math.max(maxValue, 0);
    if (minValue === maxValue) maxValue = minValue + 1;
    const xScale = value => margin.left + ((value - minValue) / (maxValue - minValue)) * plotW;
    const rowH = plotH / items.length;
    const zeroX = xScale(0);
    niceTicks(minValue, maxValue, 6).forEach(tick => {
      const x = xScale(tick);
      svg.appendChild(svgNode("line", { x1: x, y1: margin.top, x2: x, y2: margin.top + plotH, class: "grid-line" }));
      svg.appendChild(svgNode("text", { x, y: height - 18, "text-anchor": "middle", class: "axis-text" }, options && options.format ? options.format(tick) : fmt(tick, 2)));
    });
    items.forEach((item, index) => {
      const y = margin.top + index * rowH + rowH * 0.18;
      const barH = Math.max(5, rowH * 0.62);
      const valueX = xScale(item.value);
      const x = Math.min(zeroX, valueX);
      const w = Math.max(1, Math.abs(valueX - zeroX));
      const rect = svgNode("rect", { x, y, width: w, height: barH, rx: Math.min(5, barH / 2), fill: item.color || colorForSeries(item.label, index), opacity: .88 });
      rect.appendChild(svgNode("title", {}, item.label + "\n" + (options && options.format ? options.format(item.value) : fmt(item.value, 4))));
      svg.appendChild(rect);
      svg.appendChild(svgNode("text", { x: margin.left - 10, y: y + barH / 2 + 4, "text-anchor": "end", class: "axis-text" }, item.label.length > 21 ? item.label.slice(0, 20) + "…" : item.label));
      const textX = item.value >= 0 ? valueX + 7 : valueX - 7;
      svg.appendChild(svgNode("text", { x: textX, y: y + barH / 2 + 4, "text-anchor": item.value >= 0 ? "start" : "end", class: "data-label" }, options && options.format ? options.format(item.value) : fmt(item.value, 3)));
    });
  }

  function renderScatter(container, points, options) {
    const usable = points.filter(point => Number.isFinite(+point.x) && Number.isFinite(+point.y));
    if (!usable.length) return chartEmpty(container);
    const height = (options && options.height) || 340;
    const { svg, width } = chartBase(container, height);
    const margin = { top: 30, right: 22, bottom: 54, left: 62 };
    const plotW = width - margin.left - margin.right;
    const plotH = height - margin.top - margin.bottom;
    let [xMin, xMax] = minmax(usable.map(point => point.x));
    let [yMin, yMax] = minmax(usable.map(point => point.y));
    if (options && options.xDomain) [xMin, xMax] = options.xDomain;
    if (options && options.yDomain) [yMin, yMax] = options.yDomain;
    const xPad = (xMax - xMin || 1) * .06;
    const yPad = (yMax - yMin || 1) * .08;
    xMin -= xPad; xMax += xPad; yMin -= yPad; yMax += yPad;
    const xScale = value => margin.left + ((value - xMin) / (xMax - xMin || 1)) * plotW;
    const yScale = value => margin.top + plotH - ((value - yMin) / (yMax - yMin || 1)) * plotH;
    niceTicks(yMin, yMax, 5).forEach(tick => {
      const y = yScale(tick);
      svg.appendChild(svgNode("line", { x1: margin.left, y1: y, x2: width - margin.right, y2: y, class: "grid-line" }));
      svg.appendChild(svgNode("text", { x: margin.left - 9, y: y + 4, "text-anchor": "end", class: "axis-text" }, options && options.yFormat ? options.yFormat(tick) : fmt(tick, 2)));
    });
    niceTicks(xMin, xMax, 6).forEach(tick => {
      const x = xScale(tick);
      svg.appendChild(svgNode("text", { x, y: height - 23, "text-anchor": "middle", class: "axis-text" }, options && options.xFormat ? options.xFormat(tick) : fmt(tick, 1)));
    });
    svg.appendChild(svgNode("line", { x1: margin.left, y1: margin.top + plotH, x2: width - margin.right, y2: margin.top + plotH, class: "axis-line" }));
    if (options && options.xLabel) svg.appendChild(svgNode("text", { x: margin.left + plotW / 2, y: height - 4, "text-anchor": "middle", class: "axis-text" }, options.xLabel));
    usable.forEach((point, index) => {
      const circle = svgNode("circle", {
        cx: xScale(+point.x), cy: yScale(+point.y), r: point.r || 4.5,
        fill: point.color || colorForSeries(point.group || point.label, index),
        opacity: point.opacity || .66, stroke: point.stroke || "rgba(255,255,255,.2)", "stroke-width": 1
      });
      circle.appendChild(svgNode("title", {}, (point.label || "") + "\nX: " + (options && options.xFormat ? options.xFormat(point.x) : fmt(point.x, 3)) + "\nY: " + (options && options.yFormat ? options.yFormat(point.y) : fmt(point.y, 4))));
      svg.appendChild(circle);
    });
    const groups = unique(usable.map(point => point.group)).slice(0, 8);
    let x = margin.left;
    groups.forEach((group, index) => {
      svg.appendChild(svgNode("circle", { cx: x + 4, cy: 15, r: 4, fill: colorForSeries(group, index) }));
      svg.appendChild(svgNode("text", { x: x + 12, y: 19, class: "legend-text" }, group));
      x += group.length * 6.3 + 29;
    });
  }

  function renderHeatmap(container, matrix, labels, normalize) {
    if (!labels.length) return chartEmpty(container);
    const size = labels.length;
    const width = 760;
    const height = 630;
    const { svg } = chartBase(container, height);
    const margin = { top: 52, right: 80, bottom: 70, left: 80 };
    const plot = Math.min(width - margin.left - margin.right, height - margin.top - margin.bottom);
    const cell = plot / size;
    const totals = matrix.map(row => sum(row));
    const values = [];
    matrix.forEach((row, i) => row.forEach(value => values.push(normalize ? (totals[i] ? value / totals[i] : 0) : value)));
    const max = Math.max(...values, 1e-9);
    matrix.forEach((row, i) => row.forEach((raw, j) => {
      const value = normalize ? (totals[i] ? raw / totals[i] : 0) : raw;
      const intensity = clamp(value / max, 0, 1);
      const alpha = .06 + intensity * .88;
      const rect = svgNode("rect", {
        x: margin.left + j * cell, y: margin.top + i * cell,
        width: Math.max(1, cell - 1), height: Math.max(1, cell - 1), rx: Math.min(4, cell * .12),
        fill: "rgba(70,214,195," + alpha.toFixed(3) + ")"
      });
      rect.appendChild(svgNode("title", {}, "Aktual " + labels[i] + " → Prediksi " + labels[j] + "\n" + raw + " sampel" + (normalize ? " · " + pct(value, 1) : "")));
      svg.appendChild(rect);
      if (cell > 38) svg.appendChild(svgNode("text", { x: margin.left + j * cell + cell / 2, y: margin.top + i * cell + cell / 2 + 4, "text-anchor": "middle", class: "data-label" }, normalize ? pct(value, 0) : fmt(raw, 0)));
    }));
    labels.forEach((label, index) => {
      svg.appendChild(svgNode("text", { x: margin.left - 13, y: margin.top + index * cell + cell / 2 + 4, "text-anchor": "end", class: "axis-text" }, String(label)));
      svg.appendChild(svgNode("text", { x: margin.left + index * cell + cell / 2, y: margin.top + plot + 23, "text-anchor": "middle", class: "axis-text" }, String(label)));
    });
    svg.appendChild(svgNode("text", { x: margin.left + plot / 2, y: height - 16, "text-anchor": "middle", class: "axis-text" }, "Kelas prediksi"));
    const yLabel = svgNode("text", { x: 17, y: margin.top + plot / 2, "text-anchor": "middle", class: "axis-text", transform: "rotate(-90 17 " + (margin.top + plot / 2) + ")" }, "Kelas aktual");
    svg.appendChild(yLabel);
  }

  function renderForest(container, rows) {
    if (!rows.length) return chartEmpty(container);
    const height = Math.max(420, rows.length * 34 + 75);
    const { svg, width } = chartBase(container, height);
    const margin = { top: 25, right: 65, bottom: 40, left: 225 };
    const plotW = width - margin.left - margin.right;
    const plotH = height - margin.top - margin.bottom;
    const bounds = rows.flatMap(row => [row.ci95_low, row.ci95_high, row.mean_delta]).filter(Number.isFinite);
    let [min, max] = minmax(bounds);
    const pad = (max - min || .1) * .12;
    min = Math.min(min - pad, 0); max = Math.max(max + pad, 0);
    const xScale = value => margin.left + ((value - min) / (max - min || 1)) * plotW;
    niceTicks(min, max, 6).forEach(tick => {
      const x = xScale(tick);
      svg.appendChild(svgNode("line", { x1: x, y1: margin.top, x2: x, y2: margin.top + plotH, class: "grid-line" }));
      svg.appendChild(svgNode("text", { x, y: height - 15, "text-anchor": "middle", class: "axis-text" }, fmt(tick, 3)));
    });
    svg.appendChild(svgNode("line", { x1: xScale(0), y1: margin.top, x2: xScale(0), y2: margin.top + plotH, stroke: "#8eabbc", "stroke-width": 1.4, "stroke-dasharray": "4 4" }));
    const rowH = plotH / rows.length;
    rows.forEach((row, index) => {
      const y = margin.top + rowH * index + rowH / 2;
      const color = row.mean_delta >= 0 ? "#64d49a" : "#ff6b7a";
      svg.appendChild(svgNode("line", { x1: xScale(row.ci95_low), y1: y, x2: xScale(row.ci95_high), y2: y, stroke: color, "stroke-width": 2, opacity: .75 }));
      svg.appendChild(svgNode("line", { x1: xScale(row.ci95_low), y1: y - 4, x2: xScale(row.ci95_low), y2: y + 4, stroke: color, "stroke-width": 1.5 }));
      svg.appendChild(svgNode("line", { x1: xScale(row.ci95_high), y1: y - 4, x2: xScale(row.ci95_high), y2: y + 4, stroke: color, "stroke-width": 1.5 }));
      const dot = svgNode("circle", { cx: xScale(row.mean_delta), cy: y, r: 4.5, fill: color, stroke: "#07131f", "stroke-width": 1.5 });
      dot.appendChild(svgNode("title", {}, row.contrast + " · " + row.feature_method + " · K=" + row.K + "\nDelta " + fmt(row.mean_delta, 4) + "\nCI [" + fmt(row.ci95_low, 4) + ", " + fmt(row.ci95_high, 4) + "]"));
      svg.appendChild(dot);
      svg.appendChild(svgNode("text", { x: margin.left - 10, y: y + 4, "text-anchor": "end", class: "axis-text" }, (row.contrast + " · " + row.feature_method + " · K" + row.K).slice(0, 34)));
    });
  }

  function fillSelect(selector, values, labelFn) {
    const select = $(selector);
    const current = select.value;
    const allText = select.options.length ? select.options[0].textContent : "Semua";
    select.innerHTML = '<option value="ALL">' + esc(allText) + "</option>" + values.map(value =>
      '<option value="' + esc(value) + '">' + esc(labelFn ? labelFn(value) : value) + "</option>"
    ).join("");
    if (values.map(String).includes(String(current))) select.value = current;
  }

  function initializeFilters() {
    const runs = state.data.runs;
    fillSelect("#filter-track", unique(runs.map(row => row.track)).sort(), shortTrack);
    fillSelect("#filter-model", unique(runs.map(row => row.architecture)).sort());
    fillSelect("#filter-feature", unique(runs.map(row => row.feature_method)).sort());
    fillSelect("#filter-k", unique(runs.map(row => row.K)).sort((a, b) => a - b), value => "K = " + value);
    fillSelect("#filter-seed", unique(runs.map(row => row.seed)).sort((a, b) => a - b));
    state.compareModels = new Set(unique(runs.map(row => row.architecture)));
    renderModelCheckboxes();
    renderComparisonDimensions();
    populateTrainingRunSelect();
  }

  function renderModelCheckboxes() {
    const models = unique(state.data.runs.map(row => row.architecture)).sort();
    setHTML("#model-checkboxes", models.map(model =>
      '<label class="checkbox-pill"><input type="checkbox" value="' + esc(model) + '" ' + (state.compareModels.has(model) ? "checked" : "") + ">" +
      '<span style="color:' + modelColor(model, models.indexOf(model)) + '">' + esc(model) + "</span></label>"
    ).join(""));
    $$("#model-checkboxes input").forEach(input => input.addEventListener("change", () => {
      if (input.checked) state.compareModels.add(input.value); else state.compareModels.delete(input.value);
      renderComparison();
    }));
  }

  function renderComparisonDimensions() {
    const order = ["architecture", "feature_method", "class", "K"];
    setHTML("#comparison-dimensions", order.map(dimension =>
      '<label class="checkbox-pill"><input type="checkbox" value="' + dimension + '" ' +
      (state.comparisonDimensions.has(dimension) ? "checked" : "") + "><span>" + esc(COMPARISON_DIMENSIONS[dimension]) + "</span></label>"
    ).join(""));
    $$("#comparison-dimensions input").forEach(input => input.addEventListener("change", () => {
      if (input.checked) {
        state.comparisonDimensions.add(input.value);
        if (input.value === "class" && !CLASS_METRICS.has(state.comparisonMetric)) {
          state.comparisonMetric = "class_accuracy";
          $("#comparison-metric").value = state.comparisonMetric;
        }
      } else {
        state.comparisonDimensions.delete(input.value);
      }
      if (!state.comparisonDimensions.size) state.comparisonDimensions.add("architecture");
      renderComparisonDimensions();
      renderComparison();
    }));
  }

  function updateContext() {
    const copy = VIEW_COPY[state.view];
    $("#view-eyebrow").textContent = copy[0];
    $("#view-title").textContent = copy[1];
    $("#view-subtitle").textContent = copy[2];
    $("#filtered-count").textContent = filteredRuns().length.toLocaleString("id-ID");
  }

  function kpi(label, value, detail, color) {
    return '<article class="kpi-card" style="--accent:' + (color || "#46d6c3") + '">' +
      '<span class="kpi-label">' + esc(label) + '</span><strong class="kpi-value">' + esc(value) +
      '</strong><span class="kpi-detail">' + esc(detail) + "</span></article>";
  }

  function renderOverview() {
    const runs = filteredRuns();
    const completed = runs.filter(run => run.status === "COMPLETED");
    const conditions = summarizeConditions(runs);
    const best = conditions.filter(row => row.accuracy != null).sort((a, b) => b.accuracy - a.accuracy)[0];
    const stable = conditions.filter(row => row.accuracy != null && row.n > 1).sort((a, b) => a.accuracy_sd - b.accuracy_sd)[0];
    const avgAccuracy = mean(runs.map(run => run.accuracy));
    const medianRuntime = median(runs.map(run => run.total_runtime_sec));
    setHTML("#overview-kpis", [
      kpi("Run selesai", completed.length + " / " + runs.length, runs.length ? pct(completed.length / runs.length, 0) + " cakupan" : "Tidak ada run", "#64d49a"),
      kpi("Akurasi rerata", pct(avgAccuracy, 2), "Rata-rata seluruh run terfilter", "#46d6c3"),
      kpi("Kondisi terbaik", best ? pct(best.accuracy, 2) : "—", best ? modelLabel(best) + " · " + best.feature_method + " · K" + best.K : "Tidak tersedia", "#ffb36b"),
      kpi("Runtime median", seconds(medianRuntime), stable ? "Kondisi paling stabil: " + modelLabel(stable) : "Tidak tersedia", "#9b8cff")
    ].join(""));

    const grouped = groupBy(runs, run => run.architecture);
    const lineSeries = Array.from(grouped.entries()).map(([model, modelRuns], index) => ({
      label: model,
      color: modelColor(model, index),
      points: Array.from(groupBy(modelRuns, row => row.K).entries()).map(([k, rows]) => ({ x: +k, y: mean(rows.map(row => row[state.overviewMetric])) }))
    }));
    renderLineChart($("#overview-line-chart"), lineSeries, { yDomain: [0, 1], yFormat: value => pct(value, 0), xLabel: "Jumlah kelas (K)" });

    const runtimePoints = runs.map((run, index) => ({
      x: run.total_runtime_sec, y: run.accuracy, group: run.architecture,
      label: run.run_uid + " · " + run.architecture + " · " + run.feature_method + " · K" + run.K,
      color: modelColor(run.architecture, index), r: 4.2
    }));
    renderScatter($("#runtime-scatter"), runtimePoints, { xLabel: "Runtime total", xFormat: seconds, yFormat: value => pct(value, 0), yDomain: [0, 1] });

    const top = conditions.filter(row => row.accuracy != null).sort((a, b) => b.accuracy - a.accuracy).slice(0, 7);
    setHTML("#top-conditions", top.map((row, index) =>
      "<tr><td class='rank'>#" + (index + 1) + "</td><td>" + esc(modelLabel(row)) + "</td><td>" + esc(row.feature_method) +
      "</td><td>" + esc(row.K) + "</td><td class='metric-cell'>" + pct(row.accuracy, 2) + "</td><td>" + pct(row.f1_macro, 2) + "</td></tr>"
    ).join(""));

    const classRows = filterRows(state.data.per_class);
    const classSummary = Array.from(groupBy(classRows, row => row.class).entries()).map(([klass, rows]) => ({ klass, value: mean(rows.map(row => row.class_accuracy)) })).sort((a, b) => a.value - b.value);
    const hard = classSummary[0];
    const fastest = conditions.filter(row => row.total_runtime_sec != null).sort((a, b) => a.total_runtime_sec - b.total_runtime_sec)[0];
    const highK = Array.from(groupBy(runs, row => row.K).entries()).map(([k, rows]) => ({ k: +k, acc: mean(rows.map(row => row.accuracy)) })).sort((a, b) => a.k - b.k);
    const drop = highK.length > 1 ? highK[highK.length - 1].acc - highK[0].acc : null;
    const insights = [
      best && { title: "Kombinasi akurasi tertinggi", text: modelLabel(best) + " dengan " + best.feature_method + " pada K=" + best.K + " mencapai rerata " + pct(best.accuracy, 2) + " dari " + best.n + " run." },
      hard && { title: "Kelas paling menantang", text: classLabel(hard.klass) + " memiliki akurasi agregat terendah, " + pct(hard.value, 2) + ", untuk filter yang aktif." },
      fastest && { title: "Kondisi paling cepat", text: modelLabel(fastest) + " · " + fastest.feature_method + " · K" + fastest.K + " membutuhkan rata-rata " + seconds(fastest.total_runtime_sec) + "." },
      drop != null && { title: "Perubahan saat jumlah kelas naik", text: "Rerata akurasi berubah " + (drop >= 0 ? "+" : "") + pct(drop, 2) + " dari K=" + highK[0].k + " ke K=" + highK[highK.length - 1].k + ". Ini bersifat deskriptif, bukan estimasi kausal." }
    ].filter(Boolean);
    setHTML("#overview-insights", insights.map((item, index) =>
      '<div class="insight-item"><span class="insight-index">' + String(index + 1).padStart(2, "0") + "</span><strong>" + esc(item.title) + "</strong><p>" + esc(item.text) + "</p></div>"
    ).join("") || '<div class="chart-empty">Perlu lebih banyak data untuk menyusun pola.</div>');
  }

  function renderComparison() {
    const metric = state.comparisonMetric;
    const meta = COMPARISON_METRICS[metric];
    const dimensions = ["architecture", "feature_method", "class", "K"].filter(item => state.comparisonDimensions.has(item));
    const classMode = CLASS_METRICS.has(metric) || dimensions.includes("class");
    $("#comparison-weight").disabled = !classMode;
    const source = (classMode ? filterRows(state.data.per_class) : filteredRuns())
      .filter(row => state.compareModels.has(row.architecture));
    const weightedValue = rows => {
      const usable = rows.filter(row => Number.isFinite(+row[metric]));
      if (!usable.length) return null;
      if (classMode && state.comparisonWeight === "support") {
        const weight = sum(usable.map(row => row.support));
        return weight ? sum(usable.map(row => +row[metric] * (+row.support || 0))) / weight : mean(usable.map(row => row[metric]));
      }
      return mean(usable.map(row => row[metric]));
    };
    const dimensionText = (dimension, value) => dimension === "class" ? classLabel(value) : dimension === "K" ? "K=" + value : String(value);
    const rawGroups = Array.from(groupBy(source, row => dimensions.map(dimension => row[dimension]).join("¦")).values());
    const groups = rawGroups.map(rows => {
      const values = valid(rows.map(row => row[metric]));
      const value = weightedValue(rows);
      const spread = sd(values);
      const ci = values.length > 1 ? 1.96 * spread / Math.sqrt(values.length) : 0;
      const dimensionValues = Object.fromEntries(dimensions.map(dimension => [dimension, rows[0][dimension]]));
      return {
        dimensions: dimensionValues,
        label: dimensions.map(dimension => dimensionText(dimension, rows[0][dimension])).join(" · "),
        architecture: rows[0].architecture,
        value,
        n: values.length,
        runCount: unique(rows.map(row => row.history_key || row.run_uid)).length,
        support: sum(rows.map(row => row.support)),
        sd: spread,
        min: values.length ? Math.min(...values) : null,
        max: values.length ? Math.max(...values) : null,
        ciLow: value == null ? null : value - ci,
        ciHigh: value == null ? null : value + ci
      };
    }).filter(group => group.value != null && group.n >= state.comparisonMinN);
    groups.sort((a, b) => meta.lowerBetter ? a.value - b.value : b.value - a.value);
    state.comparisonGroups = groups;

    const metricFormat = value => meta.percent ? pct(value, 2) : meta.runtime ? seconds(value) : fmt(value, 4);
    const chartGroups = groups.slice(0, 30).map((group, index) => ({
      label: group.label,
      value: group.value,
      color: modelColor(group.architecture, index)
    }));
    $("#comparison-chart-title").textContent = meta.label + " menurut " + dimensions.map(item => COMPARISON_DIMENSIONS[item].toLowerCase()).join(" × ");
    $("#comparison-chart-note").textContent = groups.length > 30 ? "30 dari " + groups.length + " kelompok" : groups.length + " kelompok";
    renderBarChart($("#comparison-chart"), chartGroups, {
      height: Math.max(390, chartGroups.length * 29 + 80),
      left: dimensions.length > 2 ? 265 : 220,
      includeZero: !meta.percent,
      domain: meta.percent ? [0, 1] : null,
      format: metricFormat
    });

    setHTML("#comparison-context", [
      '<span><strong>' + groups.length + '</strong> kelompok tervalidasi</span>',
      '<span><strong>' + unique(source.map(row => row.history_key)).length + '</strong> run</span>',
      '<span><strong>' + source.length.toLocaleString("id-ID") + '</strong> observasi</span>',
      '<span>Agregasi <strong>' + (state.comparisonWeight === "support" && classMode ? "berbobot sampel" : "macro") + '</strong></span>',
      '<span>Dimensi <strong>' + dimensions.map(item => COMPARISON_DIMENSIONS[item]).join(" × ") + '</strong></span>'
    ].join(""));

    const runs = filteredRuns().filter(run => state.compareModels.has(run.architecture));
    const profiles = Array.from(groupBy(source, row => row.architecture).entries()).map(([model, rows]) => {
      const modelRuns = runs.filter(run => run.architecture === model);
      return {
        model, n: rows.length, selectedValue: weightedValue(rows),
        accuracy: mean(modelRuns.map(row => row.accuracy)), f1: mean(modelRuns.map(row => row.f1_macro)),
        sd: sd(modelRuns.map(row => row.accuracy))
      };
    }).sort((a, b) => meta.lowerBetter ? a.selectedValue - b.selectedValue : b.selectedValue - a.selectedValue);
    setHTML("#comparison-summary", profiles.map((profile, index) =>
      '<div class="model-profile"><div class="profile-heading"><strong style="color:' + modelColor(profile.model, index) + '">' + esc(profile.model) +
      "</strong><span>" + metricFormat(profile.selectedValue) + "</span></div><div class='profile-bars'>" +
      miniBar("Akurasi", profile.accuracy, 1, modelColor(profile.model, index)) +
      miniBar("F1", profile.f1, 1, modelColor(profile.model, index)) +
      miniBar("Stabilitas", 1 - clamp(profile.sd / .2, 0, 1), 1, modelColor(profile.model, index)) +
      "</div><small>" + profile.n.toLocaleString("id-ID") + " observasi · " + esc(meta.label) + "</small></div>"
    ).join("") || '<div class="chart-empty">Pilih setidaknya satu sirkuit.</div>');

    const rowDimension = dimensions[0];
    const columnDimensions = dimensions.slice(1);
    const rowValues = unique(groups.map(group => group.dimensions[rowDimension])).sort((a, b) => String(a).localeCompare(String(b), "id", { numeric: true }));
    const columnKeys = unique(groups.map(group => columnDimensions.map(dimension => group.dimensions[dimension]).join("¦")))
      .sort((a, b) => String(a).localeCompare(String(b), "id", { numeric: true }));
    const allMatrixColumns = columnDimensions.length ? columnKeys : ["__value__"];
    const effectiveColumns = allMatrixColumns.slice(0, 60);
    const [matrixMin, matrixMax] = minmax(groups.map(group => group.value));
    const matrixCell = new Map(groups.map(group => [String(group.dimensions[rowDimension]) + "§" + (columnDimensions.length ? columnDimensions.map(dimension => group.dimensions[dimension]).join("¦") : "__value__"), group]));
    $("#comparison-matrix-title").textContent = dimensions.map(item => COMPARISON_DIMENSIONS[item]).join(" × ");
    const matrixNote = $("#comparison-matrix-note");
    if (matrixNote) matrixNote.textContent = (meta.lowerBetter ? "Warna lebih kuat = nilai lebih rendah" : "Warna lebih kuat = nilai lebih tinggi") +
      (allMatrixColumns.length > effectiveColumns.length ? " · menampilkan 60 dari " + allMatrixColumns.length + " kolom; tabel di bawah tetap lengkap" : "");
    setHTML("#comparison-matrix", groups.length ? '<table><thead><tr><th>' + esc(COMPARISON_DIMENSIONS[rowDimension]) + '</th>' + effectiveColumns.map(key => {
      const label = key === "__value__" ? meta.label : key.split("¦").map((value, index) => dimensionText(columnDimensions[index], value)).join(" · ");
      return "<th>" + esc(label) + "</th>";
    }).join("") + '</tr></thead><tbody>' + rowValues.map(rowValue => '<tr><th>' + esc(dimensionText(rowDimension, rowValue)) + '</th>' + effectiveColumns.map(key => {
      const group = matrixCell.get(String(rowValue) + "§" + key);
      if (!group) return '<td class="matrix-empty">—</td>';
      const normalized = (group.value - matrixMin) / (matrixMax - matrixMin || 1);
      const strength = meta.lowerBetter ? 1 - normalized : normalized;
      const alpha = (.07 + strength * .35).toFixed(3);
      return '<td class="matrix-cell" style="background:rgba(70,214,195,' + alpha + ')" title="' + esc(group.label + " · n=" + group.n) + '"><strong>' + esc(metricFormat(group.value)) + '</strong><small>n=' + group.n + '</small></td>';
    }).join("") + '</tr>').join("") + '</tbody></table>' : '<div class="chart-empty">Tidak ada kelompok yang memenuhi minimum observasi.</div>');

    setHTML("#comparison-table-head", "<tr>" + dimensions.map(dimension => "<th>" + esc(COMPARISON_DIMENSIONS[dimension]) + "</th>").join("") +
      "<th>Observasi</th><th>Run</th>" + (classMode ? "<th>Support</th>" : "") + "<th>Rerata</th><th>SD</th><th>CI 95%</th><th>Min</th><th>Maks</th></tr>");
    setHTML("#comparison-table", groups.map(group => "<tr>" + dimensions.map(dimension => "<td>" + esc(dimensionText(dimension, group.dimensions[dimension])) + "</td>").join("") +
      "<td>" + group.n + "</td><td>" + group.runCount + "</td>" + (classMode ? "<td>" + fmt(group.support, 0) + "</td>" : "") +
      "<td class='metric-cell'>" + metricFormat(group.value) + "</td><td>" + metricFormat(group.sd) + "</td><td>" + metricFormat(group.ciLow) + " – " + metricFormat(group.ciHigh) + "</td><td>" + metricFormat(group.min) + "</td><td>" + metricFormat(group.max) + "</td></tr>"
    ).join(""));

    const baselineRows = state.data.stats.baselines.filter(row =>
      (state.feature === "ALL" || row.feature_method === state.feature)
      && (state.k === "ALL" || String(row.K) === String(state.k))
      && (state.seed === "ALL" || String(row.seed) === String(state.seed))
    );
    const baselineSummary = Array.from(groupBy(baselineRows, row => row.baseline + "¦" + row.quantum_model).values()).map(rows => ({
      label: rows[0].baseline + " → " + rows[0].quantum_model,
      value: mean(rows.map(row => row.delta_accuracy)),
      color: modelColor(rows[0].quantum_model, 0)
    })).sort((a, b) => b.value - a.value).slice(0, 12);
    renderBarChart($("#baseline-chart"), baselineSummary, { height: Math.max(260, baselineSummary.length * 27 + 70), left: 180, format: value => (value >= 0 ? "+" : "") + pct(value, 1) });
  }

  function miniBar(label, value, max, color, displayValue) {
    const width = clamp((value || 0) / (max || 1) * 100, 0, 100);
    return '<div class="mini-bar"><span>' + esc(label) + '</span><div class="mini-bar-track"><i style="width:' + width + "%;--color:" + color + '"></i></div><strong>' + esc(displayValue == null ? pct(value, 0) : displayValue) + "</strong></div>";
  }

  function renderClasses() {
    const rows = filterRows(state.data.per_class);
    const summary = Array.from(groupBy(rows, row => row.class).entries()).map(([klass, group]) => ({
      label: classLabel(klass),
      klass: +klass,
      value: mean(group.map(row => row[state.classMetric]))
    })).sort((a, b) => a.klass - b.klass);
    const metricLabel = { class_accuracy: "Akurasi", precision: "Precision", recall: "Recall", f1: "F1-score" }[state.classMetric];
    $("#class-chart-title").textContent = metricLabel + " setiap kelas";
    renderBarChart($("#class-bar-chart"), summary, { height: Math.max(390, summary.length * 34 + 80), left: 95, domain: [0, 1], includeZero: true, format: value => pct(value, 1) });

    const sorted = summary.slice().sort((a, b) => a.value - b.value);
    const avg = mean(summary.map(row => row.value));
    const spread = sorted.length ? sorted[sorted.length - 1].value - sorted[0].value : null;
    const best = sorted[sorted.length - 1];
    const hard = sorted[0];
    const insights = [
      hard && { title: "Akurasi terendah", text: hard.label + " berada pada " + pct(hard.value, 2) + ", " + pct(avg - hard.value, 2) + " di bawah rerata kelas." },
      best && { title: "Akurasi tertinggi", text: best.label + " mencapai " + pct(best.value, 2) + " pada seluruh run yang terfilter." },
      spread != null && { title: "Kesenjangan antarkelas", text: "Selisih kelas terbaik dan terlemah sebesar " + pct(spread, 2) + ". Kesenjangan besar dapat menandakan ketidakseimbangan kesulitan representasi." }
    ].filter(Boolean);
    setHTML("#class-insights", insights.map((item, index) =>
      '<div class="insight-item"><span class="insight-index">' + String(index + 1).padStart(2, "0") + "</span><strong>" + esc(item.title) + "</strong><p>" + esc(item.text) + "</p></div>"
    ).join(""));

    const confusionRows = filterRows(state.data.confusion);
    const labels = unique(confusionRows.flatMap(row => [row.actual_class, row.predicted_class])).sort((a, b) => a - b);
    const index = new Map(labels.map((label, i) => [String(label), i]));
    const matrix = labels.map(() => labels.map(() => 0));
    confusionRows.forEach(row => {
      const i = index.get(String(row.actual_class));
      const j = index.get(String(row.predicted_class));
      if (i != null && j != null) matrix[i][j] += Number(row.count || 0);
    });
    renderHeatmap($("#confusion-heatmap"), matrix, labels.map(classShort), state.normalizeConfusion);
    const rowTotals = matrix.map(row => sum(row));
    const pairs = [];
    matrix.forEach((row, i) => row.forEach((count, j) => {
      if (i !== j && count > 0) pairs.push({ actual: labels[i], predicted: labels[j], count, share: rowTotals[i] ? count / rowTotals[i] : 0 });
    }));
    pairs.sort((a, b) => b.count - a.count);
    setHTML("#confusion-pairs", pairs.slice(0, 12).map(pair =>
      "<tr><td>" + esc(classShort(pair.actual)) + "</td><td>" + esc(classShort(pair.predicted)) + "</td><td class='metric-cell'>" + fmt(pair.count, 0) + "</td><td>" + pct(pair.share, 2) + "</td></tr>"
    ).join(""));
  }

  function populateTrainingRunSelect() {
    const runs = filteredRuns().slice().sort((a, b) => (b.accuracy || 0) - (a.accuracy || 0));
    const select = $("#training-run-select");
    select.innerHTML = runs.map(run =>
      '<option value="' + esc(run.history_key) + '">' + esc(run.run_uid + " · " + run.architecture + " · " + run.feature_method + " · K" + run.K + " · seed " + run.seed) + "</option>"
    ).join("");
    if (!state.selectedRunKeys.length && runs[0]) state.selectedRunKeys = [runs[0].history_key];
  }

  async function loadHistory(key) {
    if (state.historyCache.has(key)) return state.historyCache.get(key);
    if (state.historyLoading.has(key)) return state.historyLoading.get(key);
    const run = state.data.runs.find(item => item.history_key === key);
    if (!run) throw new Error("Run tidak ditemukan: " + key);
    const promise = fetch(run.history_file).then(response => {
      if (!response.ok) throw new Error("Riwayat tidak dapat dibaca: " + response.status);
      return response.json();
    }).then(data => {
      state.historyCache.set(key, data);
      state.historyLoading.delete(key);
      return data;
    });
    state.historyLoading.set(key, promise);
    return promise;
  }

  function rowsToObjects(part) {
    if (!part || !part.columns || !part.rows) return [];
    return part.rows.map(row => Object.fromEntries(part.columns.map((column, index) => [column, row[index]])));
  }

  function trainingMetricOptions() {
    const options = state.trainingPhase === "supervised"
      ? [["train_loss", "Train loss"], ["val_loss", "Validation loss"], ["pseudo_accuracy_val", "Pseudo accuracy val"], ["avg_margin_val", "Margin validasi"], ["eval_runtime_sec", "Runtime evaluasi"]]
      : [["train_loss", "Clustering loss"], ["pseudo_accuracy_val", "Pseudo accuracy val"], ["avg_margin_val", "Margin validasi"], ["min_separation_ratio", "Rasio separasi"], ["min_separation_val_ratio", "Rasio separasi validasi"], ["active_dimensions", "Dimensi aktif"], ["yodd_norm_fraction", "Fraksi Y-odd"], ["eval_runtime_sec", "Runtime evaluasi"]];
    if (!options.some(item => item[0] === state.trainingMetric)) state.trainingMetric = options[0][0];
    $("#training-metric").innerHTML = options.map(item => '<option value="' + item[0] + '" ' + (item[0] === state.trainingMetric ? "selected" : "") + ">" + item[1] + "</option>").join("");
    return options;
  }

  async function renderTraining() {
    trainingMetricOptions();
    const keys = state.selectedRunKeys.slice(0, 4);
    setHTML("#selected-run-chips", keys.map((key, index) => {
      const run = state.data.runs.find(item => item.history_key === key);
      return '<span class="run-chip" style="border-color:' + modelColor(run && run.architecture, index) + '66">' +
        '<i style="width:7px;height:7px;border-radius:50%;background:' + modelColor(run && run.architecture, index) + '"></i>' +
        esc(run ? run.run_uid + " · " + run.architecture : key) + '<button type="button" data-remove-run="' + esc(key) + '" aria-label="Hapus run">×</button></span>';
    }).join("") || '<span class="muted">Belum ada run dipilih.</span>');
    $$("[data-remove-run]").forEach(button => button.addEventListener("click", () => {
      state.selectedRunKeys = state.selectedRunKeys.filter(key => key !== button.dataset.removeRun);
      state.evalPage = 0;
      state.detailPage = 0;
      renderTraining();
    }));
    if (!keys.length) {
      chartEmpty($("#training-line-chart"), "Tambahkan run untuk melihat kurva training.");
      setHTML("#training-kpis", "");
      setHTML("#eval-table-head", "");
      setHTML("#eval-table-body", "");
      setHTML("#run-detail-head", "");
      setHTML("#run-detail-body", "");
      setHTML("#run-detail-summary", "");
      return;
    }
    $("#training-line-chart").innerHTML = '<div class="chart-empty">Memuat seluruh evaluasi optimizer…</div>';
    try {
      const histories = await Promise.all(keys.map(loadHistory));
      const series = [];
      histories.forEach((history, index) => {
        const run = state.data.runs.find(item => item.history_key === keys[index]);
        const rows = rowsToObjects(history[state.trainingPhase]);
        series.push({
          label: (run ? run.run_uid + " · " + run.architecture : keys[index]),
          color: modelColor(run && run.architecture, index),
          points: rows.map(row => ({ x: row.eval_id, y: row[state.trainingMetric] }))
        });
      });
      const metricLabel = $("#training-metric").selectedOptions[0].textContent;
      $("#training-chart-title").textContent = metricLabel + " per evaluasi";
      const pointCount = sum(histories.map(history => (history[state.trainingPhase] && history[state.trainingPhase].rows.length) || 0));
      $("#training-point-count").textContent = pointCount.toLocaleString("id-ID") + " titik evaluasi";
      const isRatio = ["pseudo_accuracy_val", "correlation_consistency", "yodd_norm_fraction"].includes(state.trainingMetric);
      renderLineChart($("#training-line-chart"), series, {
        height: 470,
        xLabel: "Evaluation ID",
        yDomain: isRatio ? [0, 1] : null,
        yFormat: isRatio ? value => pct(value, 0) : value => fmt(value, 3)
      });
      const primaryRows = rowsToObjects(histories[0][state.trainingPhase]);
      const metricValues = valid(primaryRows.map(row => row[state.trainingMetric]));
      const run = state.data.runs.find(item => item.history_key === keys[0]);
      const bestValue = state.trainingMetric.includes("loss") ? Math.min(...metricValues) : Math.max(...metricValues);
      const finalValue = primaryRows.length ? primaryRows[primaryRows.length - 1][state.trainingMetric] : null;
      setHTML("#training-kpis", [
        kpi("Run utama", run ? run.run_uid : "—", run ? modelLabel(run) + " · " + run.feature_method + " · K" + run.K : "", "#46d6c3"),
        kpi("Evaluasi", primaryRows.length.toLocaleString("id-ID"), state.trainingPhase, "#6aa8ff"),
        kpi("Nilai terbaik", fmt(bestValue, 5), metricLabel, "#64d49a"),
        kpi("Nilai terakhir", fmt(finalValue, 5), primaryRows.length ? "Eval " + primaryRows[primaryRows.length - 1].eval_id : "—", "#ffb36b")
      ].join(""));
      renderEvaluationTable(histories[0][state.trainingPhase]);
      renderRunDetail(histories[0]);
    } catch (error) {
      chartEmpty($("#training-line-chart"), "Riwayat evaluasi gagal dimuat: " + error.message);
    }
  }

  function renderEvaluationTable(part) {
    if (!part || !part.rows.length) {
      setHTML("#eval-table-head", "");
      setHTML("#eval-table-body", "");
      return;
    }
    const pageSize = 25;
    const pages = Math.max(1, Math.ceil(part.rows.length / pageSize));
    state.evalPage = clamp(state.evalPage, 0, pages - 1);
    const start = state.evalPage * pageSize;
    const rows = part.rows.slice(start, start + pageSize);
    setHTML("#eval-table-head", "<tr>" + part.columns.map(column => "<th>" + esc(column.replace(/_/g, " ")) + "</th>").join("") + "</tr>");
    setHTML("#eval-table-body", rows.map(row => "<tr>" + row.map((value, index) => {
      const column = part.columns[index];
      const shown = typeof value === "number" ? (column.includes("accuracy") || column.includes("fraction") || column.includes("consistency") ? pct(value, 2) : fmt(value, 6)) : value;
      return "<td>" + esc(shown) + "</td>";
    }).join("") + "</tr>").join(""));
    renderPagination("#eval-pagination", state.evalPage, pages, page => { state.evalPage = page; renderEvaluationTable(part); });
  }

  function detailValue(column, value) {
    if (value == null) return "—";
    if (typeof value !== "number") return String(value);
    if (["correct", "is_active", "is_yodd", "label_degenerate"].includes(column)) return value ? "Ya" : "Tidak";
    if (column.includes("class") || column.includes("seed") || column.includes("id") || column.includes("count") || column.includes("n_") || column === "K") return fmt(value, 0);
    return fmt(value, 6);
  }

  function renderRunDetail(history) {
    const select = $("#run-detail-type");
    if (!history || !history[state.detailType]) {
      setHTML("#run-detail-summary", '<span>Detail artefak belum tersedia untuk run ini.</span>');
      setHTML("#run-detail-head", "");
      setHTML("#run-detail-body", "");
      return;
    }
    select.value = state.detailType;
    const part = history[state.detailType];
    const allRows = part.rows || [];
    const activeIndexes = (part.columns || []).map((column, index) => index)
      .filter(index => allRows.some(row => row[index] != null));
    const columns = activeIndexes.map(index => part.columns[index]);
    const rows = allRows.map(row => activeIndexes.map(index => row[index]));
    const pageSize = state.detailType === "test_predictions" ? 40 : 25;
    const pages = Math.max(1, Math.ceil(rows.length / pageSize));
    state.detailPage = clamp(state.detailPage, 0, pages - 1);
    const pageRows = rows.slice(state.detailPage * pageSize, (state.detailPage + 1) * pageSize);
    const run = state.data.runs.find(item => item.history_key === history.meta.history_key);
    const extra = [];
    if (state.detailType === "test_predictions") {
      const correctIndex = columns.indexOf("correct");
      const marginIndex = columns.indexOf("margin");
      extra.push("<strong>" + fmt(sum(rows.map(row => row[correctIndex])), 0) + "</strong> prediksi benar");
      extra.push("<strong>" + pct(mean(rows.map(row => row[correctIndex])), 2) + "</strong> akurasi dari prediksi tersimpan");
      if (marginIndex >= 0) extra.push("<strong>" + fmt(mean(rows.map(row => row[marginIndex])), 4) + "</strong> margin rerata");
    }
    setHTML("#run-detail-summary",
      "<span><strong>" + fmt(rows.length, 0) + "</strong> baris</span>" +
      "<span><strong>" + esc(run ? run.run_uid : history.meta.run_uid) + "</strong> run aktif</span>" +
      extra.map(item => "<span>" + item + "</span>").join("")
    );
    setHTML("#run-detail-head", "<tr>" + columns.map(column => "<th>" + esc(column.replace(/_/g, " ")) + "</th>").join("") + "</tr>");
    setHTML("#run-detail-body", pageRows.map(row => "<tr>" + row.map((value, index) => "<td>" + esc(detailValue(columns[index], value)) + "</td>").join("") + "</tr>").join(""));
    renderPagination("#run-detail-pagination", state.detailPage, pages, page => {
      state.detailPage = page;
      renderRunDetail(history);
    });
  }

  function renderCircuits() {
    const circuits = state.data.circuits || [];
    const benchmark = Array.from(groupBy(circuits, row => row.model).entries()).map(([model, rows], index) => ({
      label: model,
      value: mean(rows.map(row => row.unitary_median_sec)),
      color: modelColor(model, index)
    }));
    renderBarChart($("#circuit-benchmark-chart"), benchmark, { height: 330, left: 150, format: seconds });

    const primaryRuns = filteredRuns().filter(run => run.track === "PRIMARY");
    const performance = Array.from(groupBy(primaryRuns, run => run.architecture).entries()).map(([model, rows]) => ({
      model, accuracy: mean(rows.map(row => row.accuracy))
    }));
    const benchByModel = new Map(Array.from(groupBy(circuits, row => row.model).entries()).map(([model, rows]) => [model, {
      depth: mean(rows.map(row => row.depth)),
      unitary: mean(rows.map(row => row.unitary_median_sec))
    }]));
    renderScatter($("#complexity-scatter"), performance.filter(row => benchByModel.has(row.model)).map((row, index) => ({
      x: benchByModel.get(row.model).depth, y: row.accuracy, label: row.model, group: row.model,
      color: modelColor(row.model, index), r: 7
    })), { xLabel: "Circuit depth", xFormat: value => fmt(value, 0), yFormat: value => pct(value, 1), yDomain: [0, 1] });

    const thresholdRows = filterRows(state.data.thresholds);
    const thresholdSeries = Array.from(groupBy(thresholdRows, row => row.architecture).entries()).map(([model, rows], index) => ({
      label: model,
      color: modelColor(model, index),
      points: Array.from(groupBy(rows, row => row.threshold).entries()).map(([threshold, group]) => ({
        x: Math.log10(+threshold), y: mean(group.map(row => row.n_active))
      }))
    }));
    renderLineChart($("#threshold-chart"), thresholdSeries, { height: 390, xLabel: "log10(threshold)", xFormat: value => fmt(value, 0), yFormat: value => fmt(value, 1) });

    const featureRows = Array.from(groupBy(filteredRuns(), row => row.feature_method).entries()).map(([feature, rows]) => ({
      feature, n: rows.length, accuracy: mean(rows.map(row => row.accuracy)), f1: mean(rows.map(row => row.f1_macro)),
      separation: mean(rows.map(row => row.min_separation_ratio)), runtime: mean(rows.map(row => row.total_runtime_sec))
    })).sort((a, b) => b.accuracy - a.accuracy);
    const maxRuntime = Math.max(...featureRows.map(row => row.runtime || 0), 1);
    const maxSeparation = Math.max(...featureRows.map(row => row.separation || 0), 1);
    setHTML("#feature-profiles", featureRows.map((profile, index) =>
      '<div class="feature-profile"><div class="profile-heading"><strong style="color:' + FALLBACK_COLORS[index] + '">' + esc(profile.feature) +
      "</strong><span>" + pct(profile.accuracy, 1) + " akurasi</span></div><div class='profile-bars'>" +
      miniBar("Akurasi", profile.accuracy, 1, FALLBACK_COLORS[index]) +
      miniBar("F1", profile.f1, 1, FALLBACK_COLORS[index]) +
      miniBar("Separasi", profile.separation, maxSeparation, FALLBACK_COLORS[index], fmt(profile.separation, 3)) +
      miniBar("Runtime", profile.runtime, maxRuntime, FALLBACK_COLORS[index], seconds(profile.runtime)) +
      "</div></div>"
    ).join(""));
  }

  function filteredStats() {
    const rows = []
      .concat(state.data.stats.paired_primary || [])
      .concat(state.data.stats.ablation_contrasts || [])
      .concat(state.data.stats.more_comparison || []);
    return rows.filter(row =>
      (state.feature === "ALL" || row.feature_method === state.feature)
      && (state.k === "ALL" || String(row.K) === String(state.k))
    );
  }

  function renderStatistics() {
    const rows = filteredStats();
    const accuracyRows = rows.filter(row => String(row.metric).toLowerCase().includes("accuracy") && Number.isFinite(+row.mean_delta))
      .sort((a, b) => Math.abs(b.mean_delta) - Math.abs(a.mean_delta));
    renderForest($("#forest-chart"), accuracyRows.slice(0, 12));
    const significant = rows.filter(row => {
      const p = row.sign_flip_p_holm != null ? row.sign_flip_p_holm : row.sign_flip_p;
      return p != null && +p < .05;
    }).sort((a, b) => {
      const ap = a.sign_flip_p_holm != null ? a.sign_flip_p_holm : a.sign_flip_p;
      const bp = b.sign_flip_p_holm != null ? b.sign_flip_p_holm : b.sign_flip_p;
      return ap - bp;
    });
    setHTML("#significance-list", significant.slice(0, 7).map((row, index) => {
      const p = row.sign_flip_p_holm != null ? row.sign_flip_p_holm : row.sign_flip_p;
      return '<div class="insight-item"><span class="insight-index">' + String(index + 1).padStart(2, "0") +
        "</span><strong>" + esc(row.contrast + " · " + row.feature_method + " · K" + row.K) +
        "</strong><p>" + esc(row.metric) + ": delta " + fmt(row.mean_delta, 4) + ", p=" + fmt(p, 4) + ", g=" + fmt(row.hedges_g, 3) + ".</p></div>";
    }).join("") || '<div class="chart-empty">Tidak ada kontras p &lt; 0,05 untuk filter ini.</div>');

    const mc = (state.data.stats.mcnemar || []).filter(row =>
      (state.feature === "ALL" || row.feature_method === state.feature)
      && (state.k === "ALL" || String(row.K) === String(state.k))
      && (state.seed === "ALL" || String(row.seed) === String(state.seed))
    );
    const mcSig = mc.filter(row => row.p_value_holm_within_seed != null && +row.p_value_holm_within_seed < .05).length;
    setHTML("#mcnemar-summary",
      '<div class="stat-number-grid"><div><strong>' + mc.length + '</strong><span>perbandingan</span></div><div><strong>' + mcSig +
      '</strong><span>signifikan setelah Holm</span></div><div><strong>' + fmt(mean(mc.map(row => row.p_value_holm_within_seed)), 3) +
      '</strong><span>rerata p terkoreksi</span></div></div><p>Uji McNemar memakai prediksi berpasangan pada sampel uji yang sama. Interpretasikan bersama arah ketidaksepakatan, bukan p-value saja.</p>');

    const six = (state.data.stats.six_nine || []).filter(row =>
      (state.model === "ALL" || row.architecture === state.model)
      && (state.feature === "ALL" || row.feature_method === state.feature)
      && (state.seed === "ALL" || String(row.seed) === String(state.seed))
    );
    const rate = mean(six.map(row => row.six_nine_confusion_rate));
    const share = mean(six.map(row => row.share_of_errors_6_9));
    const lift = mean(six.map(row => (row.accuracy_merged_6_9 || 0) - (row.accuracy || 0)));
    setHTML("#six-nine-summary",
      '<div class="stat-number-grid"><div><strong>' + pct(rate, 2) + '</strong><span>confusion rate 6–9</span></div><div><strong>' +
      pct(share, 2) + '</strong><span>porsi dari semua error</span></div><div><strong>' + (lift >= 0 ? "+" : "") + pct(lift, 2) +
      '</strong><span>akurasi jika 6/9 digabung</span></div></div><p>Analisis pra-spesifikasi yang diwarisi dari studi MNIST. Pada glif Arab-Indik, ٦ dan ٩ bukan rotasi satu sama lain; pasangan khas aksara ini (٧↔٨, ٠↔٥) ada di tab Temuan &amp; replikasi.</p>');

    const tableRows = rows.slice().sort((a, b) => {
      const ap = a.sign_flip_p_holm != null ? a.sign_flip_p_holm : a.sign_flip_p;
      const bp = b.sign_flip_p_holm != null ? b.sign_flip_p_holm : b.sign_flip_p;
      return (ap == null ? 2 : ap) - (bp == null ? 2 : bp);
    }).slice(0, 120);
    setHTML("#statistics-table", tableRows.map(row => {
      const p = row.sign_flip_p_holm != null ? row.sign_flip_p_holm : row.sign_flip_p;
      const cls = row.mean_delta >= 0 ? "delta-positive" : "delta-negative";
      return "<tr><td>" + esc(row.contrast) + "</td><td>" + esc(row.feature_method) + "</td><td>" + esc(row.K) +
        "</td><td>" + esc(row.metric) + "</td><td>" + esc(row.n_pairs) + "</td><td class='" + cls + "'>" + fmt(row.mean_delta, 5) +
        "</td><td>[" + fmt(row.ci95_low, 4) + ", " + fmt(row.ci95_high, 4) + "]</td><td>" + fmt(row.hedges_g, 3) +
        "</td><td>" + fmt(p, 4) + "</td></tr>";
    }).join(""));
  }

  function renderRuns() {
    let runs = filteredRuns();
    const query = state.runSearch.trim().toLowerCase();
    if (query) runs = runs.filter(run => [run.run_uid, run.condition_id, run.architecture, run.model_code, run.feature_method, run.seed, run.K, run.track].some(value => String(value || "").toLowerCase().includes(query)));
    const sorters = {
      "accuracy-desc": (a, b) => (b.accuracy || -Infinity) - (a.accuracy || -Infinity),
      "accuracy-asc": (a, b) => (a.accuracy || Infinity) - (b.accuracy || Infinity),
      "runtime-asc": (a, b) => (a.total_runtime_sec || Infinity) - (b.total_runtime_sec || Infinity),
      "runtime-desc": (a, b) => (b.total_runtime_sec || -Infinity) - (a.total_runtime_sec || -Infinity),
      "uid-asc": (a, b) => String(a.run_uid).localeCompare(String(b.run_uid))
    };
    runs = runs.slice().sort(sorters[state.runSort]);
    const pageSize = 30;
    const pages = Math.max(1, Math.ceil(runs.length / pageSize));
    state.runPage = clamp(state.runPage, 0, pages - 1);
    const page = runs.slice(state.runPage * pageSize, (state.runPage + 1) * pageSize);
    setHTML("#runs-table-body", page.map(run =>
      "<tr><td class='metric-cell'>" + esc(run.run_uid) + "</td><td>" + esc(shortTrack(run.track)) + "</td><td>" + esc(modelLabel(run)) +
      "</td><td>" + esc(run.feature_method) + "</td><td>" + esc(run.K) + "</td><td>" + esc(run.seed) +
      "</td><td>" + pct(run.accuracy, 2) + "</td><td>" + pct(run.f1_macro, 2) + "</td><td>" + fmt(run.final_supervised_val_loss, 5) +
      "</td><td>" + seconds(run.total_runtime_sec) + "</td><td><span class='status-badge " + (run.status === "COMPLETED" ? "" : "warn") + "'>" + esc(run.status) + "</span></td></tr>"
    ).join(""));
    renderPagination("#run-pagination", state.runPage, pages, pageIndex => { state.runPage = pageIndex; renderRuns(); }, runs.length + " run");
  }

  function renderPagination(selector, page, pages, onChange, prefix) {
    const container = $(selector);
    container.innerHTML = "";
    if (prefix) container.appendChild(document.createTextNode(prefix + " · "));
    const previous = document.createElement("button");
    previous.type = "button"; previous.textContent = "Sebelumnya"; previous.disabled = page <= 0;
    previous.addEventListener("click", () => onChange(page - 1));
    const text = document.createElement("span");
    text.textContent = "Halaman " + (page + 1) + " / " + pages;
    const next = document.createElement("button");
    next.type = "button"; next.textContent = "Berikutnya"; next.disabled = page >= pages - 1;
    next.addEventListener("click", () => onChange(page + 1));
    container.append(previous, text, next);
  }

  function renderActiveView() {
    updateContext();
    if (state.view === "overview") renderOverview();
    if (state.view === "comparison") renderComparison();
    if (state.view === "classes") renderClasses();
    if (state.view === "training") renderTraining();
    if (state.view === "circuits") renderCircuits();
    if (state.view === "statistics") renderStatistics();
    if (state.view === "runs") renderRuns();
    if (state.view === "findings") renderFindings();
  }

  function switchView(view) {
    state.view = view;
    $$(".nav-item").forEach(item => item.classList.toggle("active", item.dataset.view === view));
    $$("[data-view-panel]").forEach(panel => {
      const active = panel.dataset.viewPanel === view;
      panel.hidden = !active;
      panel.classList.toggle("active", active);
    });
    if (view === "training") populateTrainingRunSelect();
    renderActiveView();
    if (window.innerWidth < 920) $("#main-content").scrollIntoView({ behavior: "smooth", block: "start" });
  }

  function handleFilterChange() {
    state.track = $("#filter-track").value;
    state.model = $("#filter-model").value;
    state.feature = $("#filter-feature").value;
    state.k = $("#filter-k").value;
    state.seed = $("#filter-seed").value;
    state.runPage = 0;
    state.evalPage = 0;
    populateTrainingRunSelect();
    renderActiveView();
  }

  function exportFilteredRuns() {
    const rows = filteredRuns();
    const columns = ["track", "condition_id", "run_uid", "model_code", "architecture", "feature_method", "K", "seed", "accuracy", "precision_macro", "recall_macro", "f1_macro", "final_cluster_loss", "final_supervised_train_loss", "final_supervised_val_loss", "min_separation_ratio", "total_runtime_sec", "status"];
    const csv = [columns.join(",")].concat(rows.map(row => columns.map(column => {
      const text = row[column] == null ? "" : String(row[column]);
      return /[\",\\n]/.test(text) ? '"' + text.replace(/"/g, '""') + '"' : text;
    }).join(","))).join("\r\n");
    const blob = new Blob([csv], { type: "text/csv;charset=utf-8" });
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = url;
    link.download = "more-hd-runs-terfilter.csv";
    document.body.appendChild(link);
    link.click();
    link.remove();
    URL.revokeObjectURL(url);
  }

  function exportComparisonGroups() {
    const dimensions = ["architecture", "feature_method", "class", "K"].filter(item => state.comparisonDimensions.has(item));
    const columns = dimensions.concat(["observations", "runs", "support", "mean", "sd", "ci95_low", "ci95_high", "min", "max"]);
    const csvCell = value => {
      const text = value == null ? "" : String(value);
      return /[\",\n]/.test(text) ? '"' + text.replace(/"/g, '""') + '"' : text;
    };
    const rows = state.comparisonGroups.map(group => dimensions.map(dimension => group.dimensions[dimension]).concat([
      group.n, group.runCount, group.support, group.value, group.sd, group.ciLow, group.ciHigh, group.min, group.max
    ]));
    const csv = [columns.join(",")].concat(rows.map(row => row.map(csvCell).join(","))).join("\r\n");
    const blob = new Blob([csv], { type: "text/csv;charset=utf-8" });
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = url;
    link.download = "more-hd-perbandingan-kelompok.csv";
    document.body.appendChild(link);
    link.click();
    link.remove();
    URL.revokeObjectURL(url);
  }

  function renderMethodology() {
    const meta = state.data.metadata;
    const inventory = state.data.inventory || [];
    const dialog = $("#methodology-dialog");
    setHTML("#methodology-content",
      '<p class="methodology-copy">Dashboard studi MADBase (digit tulisan tangan Arab-Indik ٠\u200e–٩\u200e, snapshot MADBASE-AUC-V1) membaca workbook konsolidasi dan tabel analisis di research_arabic_digits/research_data secara lokal. Workbook sumber tidak diubah, dan data studi MNIST tidak dibaca kecuali ringkasan deskriptif pada tab Temuan. Citra mentah MADBase tidak ditampilkan (kebijakan lokal, tanpa redistribusi). Kurva training disimpan per run dan baru dimuat ketika dipilih agar halaman tetap responsif.</p>' +
      '<div class="methodology-grid">' + inventory.map(item =>
        "<article><strong>" + esc(shortTrack(item.track)) + "</strong><span>" + esc(item.source) + "<br>" + fmt(item.run_count, 0) + " run · " +
        fmt(item.per_class_rows, 0) + " baris kelas · " + fmt(item.confusion_rows, 0) + " sel confusion</span></article>"
      ).join("") +
      "<article><strong>Riwayat optimizer</strong><span>" + fmt(meta.run_count, 0) + " berkas run memuat seluruh eval clustering dan supervised.</span></article>" +
      "<article><strong>Batas interpretasi</strong><span>Ringkasan otomatis bersifat deskriptif. Klaim inferensial harus merujuk CI, effect size, dan p-value terkoreksi.</span></article></div>");
    dialog.showModal();
  }

  function renderFindings() {
    const F = state.data.findings || {};
    const N = F.numbers || {};
    if (!N.primary_acc) {
      setHTML("#findings-kpis", "");
      setHTML("#findings-insights", '<p class="muted">Ringkasan temuan belum tersedia. Jalankan research_arabic_digits/paper/build/analyze_madbase.py lalu refresh_data.ps1.</p>');
      return;
    }
    const pp = (value, digits) => value == null ? "—" : (value >= 0 ? "+" : "") + fmt(value, digits == null ? 1 : digits) + " pp";
    const ab = N.ablation || {};
    setHTML("#findings-kpis", [
      kpi("D − A akurasi", pp(N.primary_acc.mean), N.primary_acc.n_pos + "/" + N.primary_acc.n + " sel positif · SK>0 di " + N.primary_acc.ci_pos + " sel", "#ffb36b"),
      kpi("B − A (kapasitas riil)", pp(ab["B−A"] && ab["B−A"].mean), (ab["B−A"] ? ab["B−A"].n_pos : "—") + "/9 sel positif", "#9b8cff"),
      kpi("D − B (kompleks vs riil)", pp(ab["D−B"] && ab["D−B"].mean), (ab["D−B"] ? 9 - ab["D−B"].n_pos : "—") + "/9 sel negatif", "#f472b6"),
      kpi("D − C (RZ dilatih vs beku)", pp(ab["D−C"] && ab["D−C"].mean), "tidak ada SK yang mengecualikan nol", "#46d6c3")
    ].join(""));
    const X = N.cross || {};
    const mn = X.mnist || {};
    const ad = X.madbase || {};
    const keys = ["B−A", "C−A", "D−B", "D−C"];
    const items = [];
    if (mn.DA && ad.DA) {
      items.push({ label: "D − A · MNIST", value: mn.DA.mean, color: "#7d8b99" });
      items.push({ label: "D − A · MADBase", value: ad.DA.mean, color: "#ffb36b" });
      keys.forEach(key => {
        if (mn.ablation && mn.ablation[key]) items.push({ label: key.replace("−", " − ") + " · MNIST", value: mn.ablation[key].mean, color: "#7d8b99" });
        if (ad.ablation && ad.ablation[key]) items.push({ label: key.replace("−", " − ") + " · MADBase", value: ad.ablation[key].mean, color: "#ffb36b" });
      });
    }
    renderBarChart($("#cross-chart"), items, { left: 165, domain: [-14, 16], format: value => (value >= 0 ? "+" : "") + fmt(value, 1) });
    const mech = N.mechanism && N.mechanism.table;
    const insights = [
      { title: "Efek primer tereplikasi", text: "Rata-rata D − A " + pp(ad.DA && ad.DA.mean) + " pada MADBase dan " + pp(mn.DA && mn.DA.mean) + " pada MNIST. Pola per sel tidak berulang (ρ Spearman " + fmt(X.cell_DA_spearman && X.cell_DA_spearman.rho, 2) + "), jadi besar efek per sel perlu ditafsirkan hati-hati." },
      { title: "Atribusi ke kapasitas, bukan fase", text: "Sirkuit riil 60 parameter (B) mengungguli MORE-HD-C pada 9/9 sel ablasi; melatih RZ dibanding membekukannya hanya " + pp(ab["D−C"] && ab["D−C"].mean) + "." },
      mech && { title: "Penjelasan eksploratif", text: "Probe linear D " + fmt(mech.D.probe, 1) + "% ≈ B " + fmt(mech.B.probe, 1) + "%, tetapi effective rank Jacobian D " + fmt(mech.D.eff, 1) + " vs B " + fmt(mech.B.eff, 1) + "; 8 sudut RZ lapisan terakhir inert." },
      N.baseline_mean && { title: "Konteks klasik", text: "Regresi logistik " + fmt(N.baseline_mean.LR, 1) + "% dan nearest centroid " + fmt(N.baseline_mean.NC, 1) + "% vs MORE-HD-C " + fmt(N.baseline_mean.D, 1) + "% (rata-rata 24 sel). Bukan klaim quantum advantage." }
    ].filter(Boolean);
    setHTML("#findings-insights", insights.map((item, index) =>
      '<div class="insight-item"><span class="insight-index">' + String(index + 1).padStart(2, "0") + "</span><strong>" + esc(item.title) + "</strong><p>" + esc(item.text) + "</p></div>"
    ).join(""));
    const cnt = s => s ? s.n_pos + "/" + s.n : "—";
    const crossRows = mn.DA ? [
      ["Akurasi rata-rata A / D", fmt(mn.acc.A, 1) + "% / " + fmt(mn.acc.D, 1) + "%", fmt(ad.acc.A, 1) + "% / " + fmt(ad.acc.D, 1) + "%"],
      ["D − A akurasi (sel positif)", pp(mn.DA.mean) + " (" + cnt(mn.DA) + ")", pp(ad.DA.mean) + " (" + cnt(ad.DA) + ")"],
      ["D − A macro-F1 (sel positif)", pp(mn.DA_f1.mean) + " (" + cnt(mn.DA_f1) + ")", pp(ad.DA_f1.mean) + " (" + cnt(ad.DA_f1) + ")"],
      ["SK 95% D − A > 0 / < 0", mn.DA.ci_pos + " / " + mn.DA.ci_neg, ad.DA.ci_pos + " / " + ad.DA.ci_neg],
      ["McNemar signifikan (dari 120)", mn.mcnemar_sig, ad.mcnemar_sig]
    ].concat(keys.map(key => ["Ablasi " + key.replace("−", " − ") + " (sel positif)", pp(mn.ablation[key].mean) + " (" + cnt(mn.ablation[key]) + ")", pp(ad.ablation[key].mean) + " (" + cnt(ad.ablation[key]) + ")"])).concat([
      ["D − M (sel positif)", pp(mn.D_minus_M.mean) + " (" + cnt(mn.D_minus_M) + ")", pp(ad.D_minus_M.mean) + " (" + cnt(ad.D_minus_M) + ")"],
      ["Akurasi M / NC / LR", fmt(mn.acc.M, 1) + "% / " + fmt(mn.NC, 1) + "% / " + fmt(mn.LR, 1) + "%", fmt(ad.acc.M, 1) + "% / " + fmt(ad.NC, 1) + "% / " + fmt(ad.LR, 1) + "%"],
      ["Porsi galat 6↔9 (K = 10, A–D)", pct(mn.six_nine_share_ABCD, 1), pct(ad.six_nine_share_ABCD, 1)]
    ]) : [];
    setHTML("#cross-table", crossRows.map(row => "<tr><td>" + esc(row[0]) + "</td><td>" + esc(row[1]) + "</td><td class='metric-cell'>" + esc(row[2]) + "</td></tr>").join(""));
    const featLab = { PCA: "PCA", HU: "Hu", ZERNIKE: "Zernike" };
    const pairs = (F.confusion_pairs || []).filter(row => +row.rank <= 5);
    setHTML("#pairs-table", pairs.map(row => {
      const [a, b] = String(row.pair).split("↔").map(Number);
      return "<tr><td>" + esc(featLab[row.feature_method] || row.feature_method) + "</td><td>" + esc(row.pair) + "</td><td>" + esc(glyph(a) + " ↔ " + glyph(b)) + "</td><td class='metric-cell'>" + pct(+row.share, 1) + "</td></tr>";
    }).join(""));
    setHTML("#mech-table", mech ? ["A", "C", "D", "B"].map(code => {
      const v = mech[code];
      return "<tr><td>" + code + "</td><td>" + fmt(v.rank, 1) + "</td><td>" + fmt(v.eff, 1) + "</td><td>" + fmt(v.probe, 1) + "%</td><td>" +
        (code === "C" || code === "D" ? fmt(v.probe9, 1) + "%" : "—") + "</td><td class='metric-cell'>" + fmt(v.acc, 1) + "%</td><td>" + fmt(v.gap, 1) + " pp</td><td>" + fmt(v.sup, 0) + "%</td></tr>";
    }).join("") : "<tr><td colspan='8'>Belum tersedia</td></tr>");
    setHTML("#jalurb-table", (F.jalurB || []).map(row =>
      "<tr><td>" + esc(featLab[row.feature] || row.feature) + "</td><td>" + esc(row.K) + "</td><td>" + fmt(row.A_jalurA, 1) + "%</td><td>" + fmt(row.A_jalurB, 1) +
      "%</td><td>" + fmt(row.D_jalurA, 1) + "%</td><td>" + fmt(row.D_jalurB, 1) + "%</td><td>" + pp(row.DA_jalurA) + "</td><td class='" +
      (row.DA_jalurB >= 0 ? "delta-positive" : "delta-negative") + "'>" + pp(row.DA_jalurB) + " [" + fmt(row.DA_jalurB_lo, 1) + "; " + fmt(row.DA_jalurB_hi, 1) + "]</td></tr>"
    ).join(""));
  }

  function bindEvents() {
    $$(".nav-item").forEach(button => button.addEventListener("click", () => switchView(button.dataset.view)));
    ["#filter-track", "#filter-model", "#filter-feature", "#filter-k", "#filter-seed"].forEach(selector => $(selector).addEventListener("change", handleFilterChange));
    $("#reset-filters").addEventListener("click", () => {
      state.track = state.model = state.feature = state.k = state.seed = "ALL";
      ["#filter-track", "#filter-model", "#filter-feature", "#filter-k", "#filter-seed"].forEach(selector => { $(selector).value = "ALL"; });
      handleFilterChange();
    });
    $("#export-button").addEventListener("click", exportFilteredRuns);
    $("#show-methodology").addEventListener("click", renderMethodology);
    $("#overview-metric-toggle").addEventListener("click", event => {
      const button = event.target.closest("button[data-metric]");
      if (!button) return;
      state.overviewMetric = button.dataset.metric;
      $$("#overview-metric-toggle button").forEach(item => item.classList.toggle("active", item === button));
      renderOverview();
    });
    $("#comparison-metric").addEventListener("change", event => {
      state.comparisonMetric = event.target.value;
      if (!CLASS_METRICS.has(state.comparisonMetric) && state.comparisonDimensions.has("class")) {
        state.comparisonDimensions.delete("class");
        renderComparisonDimensions();
      }
      renderComparison();
    });
    $("#comparison-weight").addEventListener("change", event => { state.comparisonWeight = event.target.value; renderComparison(); });
    $("#comparison-min-n").addEventListener("change", event => {
      state.comparisonMinN = clamp(parseInt(event.target.value, 10) || 1, 1, 100);
      event.target.value = state.comparisonMinN;
      renderComparison();
    });
    $("#export-comparison").addEventListener("click", exportComparisonGroups);
    $("#class-metric").addEventListener("change", event => { state.classMetric = event.target.value; renderClasses(); });
    $("#normalize-confusion").addEventListener("change", event => { state.normalizeConfusion = event.target.checked; renderClasses(); });
    $("#training-phase-toggle").addEventListener("click", event => {
      const button = event.target.closest("button[data-phase]");
      if (!button) return;
      state.trainingPhase = button.dataset.phase;
      state.evalPage = 0;
      $$("#training-phase-toggle button").forEach(item => item.classList.toggle("active", item === button));
      trainingMetricOptions();
      renderTraining();
    });
    $("#training-metric").addEventListener("change", event => { state.trainingMetric = event.target.value; state.evalPage = 0; renderTraining(); });
    $("#run-detail-type").addEventListener("change", event => {
      state.detailType = event.target.value;
      state.detailPage = 0;
      const key = state.selectedRunKeys[0];
      if (key && state.historyCache.has(key)) renderRunDetail(state.historyCache.get(key));
    });
    $("#add-training-run").addEventListener("click", () => {
      const key = $("#training-run-select").value;
      if (!key || state.selectedRunKeys.includes(key)) return;
      if (state.selectedRunKeys.length >= 4) state.selectedRunKeys.shift();
      state.selectedRunKeys.push(key);
      state.evalPage = 0;
      state.detailPage = 0;
      renderTraining();
    });
    $("#run-search").addEventListener("input", event => { state.runSearch = event.target.value; state.runPage = 0; renderRuns(); });
    $("#run-sort").addEventListener("change", event => { state.runSort = event.target.value; state.runPage = 0; renderRuns(); });
  }

  function renderCoverage() {
    const meta = state.data.metadata;
    $("#coverage-runs").textContent = fmt(meta.run_count, 0);
    const historyRuns = Object.values(meta.history_counts || {}).reduce((acc, item) => acc + Math.min(item.clustering_runs || 0, item.supervised_runs || 0), 0);
    $("#coverage-evals").textContent = fmt(historyRuns, 0);
    $("#coverage-classes").textContent = fmt(state.data.per_class.length, 0);
    $("#dataset-status").textContent = fmt(meta.completed_run_count, 0) + " run selesai · data lokal";
    $(".data-status").classList.add("ready");
  }

  async function init() {
    try {
      const response = await fetch("data/dashboard-data.json", { cache: "no-store" });
      if (!response.ok) throw new Error("HTTP " + response.status);
      state.data = await response.json();
      initializeFilters();
      bindEvents();
      renderCoverage();
      $("#loading-state").hidden = true;
      $("#dashboard").hidden = false;
      renderActiveView();
    } catch (error) {
      $("#loading-state").hidden = true;
      $("#error-state").hidden = false;
      $("#error-message").textContent = location.protocol === "file:"
        ? "Browser membatasi pembacaan data ketika index.html dibuka langsung sebagai file."
        : "Kesalahan: " + error.message;
      console.error(error);
    }
  }

  init();
})();
