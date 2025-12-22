function parseArray(val) {
  if (Array.isArray(val)) return val;
  if (typeof val === "string") {
    return val.replace(/[{}]/g, "")
      .split(",")
      .filter(x => x.length > 0)
      .map(Number);
  }
  return [];
}

const NUM_SEGMENTS = 4; // Number of wall segments per side (tweak as you like)
const table = data.series[0];
const fields = table.fields;

const bprices = fields.find(f => f.name === "bprices");
const bcumvolumes = fields.find(f => f.name === "bcumvolumes");
const bvolumes = fields.find(f => f.name === "bvolumes");
const aprices = fields.find(f => f.name === "aprices");
const acumvolumes = fields.find(f => f.name === "acumvolumes");
const avolumes = fields.find(f => f.name === "avolumes");

if (!bprices || !bcumvolumes || !aprices || !acumvolumes) {
  throw new Error("Missing required array fields");
}

const bps = parseArray(bprices.values.get(0));
const bvs = parseArray(bcumvolumes.values.get(0));
const brs = bvolumes ? parseArray(bvolumes.values.get(0)) : [];
const aps = parseArray(aprices.values.get(0));
const avs = parseArray(acumvolumes.values.get(0));
const ars = avolumes ? parseArray(avolumes.values.get(0)) : [];

const bids = [];
for (let i = 0; i < Math.min(bps.length, bvs.length); i++) {
  bids.push({ x: bps[i], y: bvs[i], raw: brs[i] ?? null });
}

const asks = [];
for (let i = 0; i < Math.min(aps.length, avs.length); i++) {
  asks.push({ x: aps[i], y: avs[i], raw: ars[i] ?? null });
}

bids.sort((a, b) => b.x - a.x);
asks.sort((a, b) => a.x - b.x);

const bestBid = bids[0]?.x ?? 0;
const bestAsk = asks[0]?.x ?? 0;
const mid = (bestBid + bestAsk) / 2;
if (bids.length > 0 && asks.length > 0) {
  const bidY = bids[0].y;
  const askY = asks[0].y;
  const midY = Math.min(bidY, askY);
  bids.unshift({ x: mid, y: midY });
  asks.unshift({ x: mid, y: midY });
  bids.push({ x: bids[bids.length - 1].x - 0.0001, y: 0 });
  asks.push({ x: asks[asks.length - 1].x + 0.0001, y: 0 });
}

// --- SEGMENTED WALL LOGIC ---
function segmentedWalls(levels, nSegments) {
  const N = levels.length;
  if (N === 0) return [];
  const result = [];
  const segSize = Math.floor(N / nSegments);

  for (let seg = 0; seg < nSegments; seg++) {
    const start = seg * segSize;
    const end = seg === nSegments - 1 ? N : (seg + 1) * segSize;
    let maxIdx = -1, maxRaw = -Infinity;
    for (let i = start; i < end; i++) {
      if (levels[i]?.raw !== undefined && levels[i].raw > maxRaw) {
        maxRaw = levels[i].raw;
        maxIdx = i;
      }
    }
    if (maxIdx >= 0) result.push(levels[maxIdx]);
  }
  return result;
}

const topBids = segmentedWalls(bids, NUM_SEGMENTS);
const topAsks = segmentedWalls(asks, NUM_SEGMENTS);

const wallLines = [...topBids, ...topAsks].map(wall => ({
  type: "line",
  x0: wall.x,
  x1: wall.x,
  y0: 0,
  y1: 1,
  xref: "x",
  yref: "paper",
  line: {
    color: "yellow",
    width: 1,
    dash: "dot"
  }
}));

const wallLabels = [...topBids, ...topAsks].map(wall => ({
  x: wall.x,
  y: 1.02,
  xref: "x",
  yref: "paper",
  text: wall.x.toFixed(5),
  showarrow: false,
  font: { color: "yellow", size: 10 }
}));

const minX = Math.min(...bids.map(b => b.x), ...asks.map(a => a.x), mid);
const maxX = Math.max(...bids.map(b => b.x), ...asks.map(a => a.x), mid);
const pad = (maxX - minX) * 0.01; // 1% padding

return {
  data: [
    {
      name: "Bids",
      x: bids.map(pt => pt.x),
      y: bids.map(pt => pt.y),
      mode: "lines",
      fill: "tozeroy",
      fillcolor: "rgba(0,255,0,0.2)",
      line: { shape: "hv", color: "rgba(0,255,0,0.7)", width: 2 },
      type: "scatter",
      hovertemplate: "Price: %{x}<br>CumVol: %{y}<extra></extra>"
    },
    {
      name: "Asks",
      x: asks.map(pt => pt.x),
      y: asks.map(pt => pt.y),
      mode: "lines",
      fill: "tozeroy",
      fillcolor: "rgba(255,0,0,0.2)",
      line: { shape: "hv", color: "rgba(255,0,0,0.7)", width: 2 },
      type: "scatter",
      hovertemplate: "Price: %{x}<br>CumVol: %{y}<extra></extra>"
    }
  ],
  layout: {
    plot_bgcolor: "black",
    paper_bgcolor: "black",
    font: { color: "white" },
    xaxis: {
      title: "Price",
      type: "linear",
      showgrid: true,
      gridcolor: "rgba(255,255,255,0.1)",
      zeroline: false,
      range: [minX - pad, maxX + pad] // this forces the mid to be visible
    },
    yaxis: {
      title: "Cumulative Volume",
      type: "log",
      showgrid: true,
      gridcolor: "rgba(255,255,255,0.1)",
      zeroline: false
    },
    margin: { t: 20, l: 40, r: 10, b: 30 },
    shapes: [
      {
        type: "line",
        x0: mid,
        x1: mid,
        y0: 0,
        y1: 1,
        xref: "x",
        yref: "paper",
        line: {
          color: "white",
          width: 1,
          dash: "dot"
        }
      },
      ...wallLines
    ],
    annotations: wallLabels,
    legend: {
      orientation: "h",
      x: 0.5,
      xanchor: "center",
      y: -0.3
    }
  }
};
