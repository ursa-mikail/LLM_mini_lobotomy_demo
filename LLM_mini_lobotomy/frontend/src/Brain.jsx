import React from 'react';

/*
 * The real architecture, drawn as it is:
 *   INPUT   4 context positions, each a one-hot over the vocabulary (shown as the chosen word)
 *   HIDDEN  48 ReLU neurons in one fully-connected layer (a numbered 6x8 grid; no lateral links)
 *   OUTPUT  one unit per vocabulary word (62), softmax -> next-word odds
 */
const COLS = 6;
export const POS = Array.from({ length: 48 }, (_, i) => ({ x: 337 + (i % COLS) * 54, y: 168 + Math.floor(i / COLS) * 54 }));
const PANEL = { x: 300, y: 112, w: 340, h: 470 };
const SLOT_X = 20, SLOT_W = 140;
const SLOT_Y = [195, 285, 375, 465];
const OUT_C = [748, 874];
const OUT_Y0 = 150, OUT_DY = 16.5;
const OPANEL = { x: 726, y: 112, w: 268, h: 556 };

const curve = (x1, y1, x2, y2) => {
  const xm = (x1 + x2) / 2;
  return `M${x1},${y1} C${xm},${y1} ${xm},${y2} ${x2},${y2}`;
};

export default function Brain({ step, vocab, secretFrom, revealed, removed, flagged, selected, zap, scanProg, hover, onHover, onClick, clickable, glitch }) {
  const V = vocab.length;
  const per = Math.ceil(V / 2);
  const outPos = (t) => {
    const col = t < per ? 0 : 1;
    const r = col ? t - per : t;
    return { x: OUT_C[col], y: OUT_Y0 + r * OUT_DY };
  };
  const label = (t) => (t >= secretFrom && !revealed ? '???' : vocab[t]);

  const act = step ? step.act : null;
  const amax = act ? Math.max(3, ...act) : 3;
  const hotSet = step ? step.hot : [];
  const probs = step ? step.probs : null;
  const chosenTok = step ? step.chosen.t : null;
  const chosenIdx = chosenTok !== null ? vocab.indexOf(chosenTok) : -1;
  const topSet = step ? step.top.map((r) => vocab.indexOf(r.t)) : [];
  const scanning = scanProg >= 0;
  const cx = PANEL.x + PANEL.w / 2, cy = PANEL.y + PANEL.h / 2;

  return (
    <svg viewBox="0 0 1000 690" className={`brain ${glitch ? 'glitch' : ''}`}>
      <defs>
        <linearGradient id="radarGrad" x1="0" x2="1" y1="0" y2="0">
          <stop offset="0" stopColor="#29f2ff" stopOpacity="0" />
          <stop offset="1" stopColor="#29f2ff" stopOpacity="0.9" />
        </linearGradient>
      </defs>

      {/* layer panels */}
      <rect x={PANEL.x} y={PANEL.y} width={PANEL.w} height={PANEL.h} rx="18" className="panel" />
      <rect x={OPANEL.x} y={OPANEL.y} width={OPANEL.w} height={OPANEL.h} rx="18" className="panel" />
      <text x={SLOT_X + SLOT_W / 2} y="84" textAnchor="middle" className="cap">INPUT LAYER</text>
      <text x={SLOT_X + SLOT_W / 2} y="100" textAnchor="middle" className="tiny">4 positions × 62 one-hot units</text>
      <text x={cx} y="84" textAnchor="middle" className="cap">HIDDEN LAYER</text>
      <text x={cx} y="100" textAnchor="middle" className="tiny">48 ReLU neurons · fully connected both sides</text>
      <text x={OPANEL.x + OPANEL.w / 2} y="84" textAnchor="middle" className="cap">OUTPUT LAYER</text>
      <text x={OPANEL.x + OPANEL.w / 2} y="100" textAnchor="middle" className="tiny">{V} units, one per word · softmax</text>
      <text x="230" y="318" textAnchor="middle" className="tiny">W₁</text>
      <text x="230" y="331" textAnchor="middle" className="tiny">248 × 48</text>
      <text x="683" y="318" textAnchor="middle" className="tiny">W₂</text>
      <text x="683" y="331" textAnchor="middle" className="tiny">48 × 62</text>

      {/* full connectivity, faint: every input position feeds every hidden neuron */}
      {SLOT_Y.map((sy, p) => POS.map((q, i) => (
        <line key={`f${p}-${i}`} x1={SLOT_X + SLOT_W} y1={sy} x2={q.x - 12} y2={q.y} className="faint" />
      )))}
      {/* and every hidden neuron feeds every output word (fan hint) */}
      {Array.from({ length: 9 }, (_, k) => (
        <path key={`o${k}`} d={curve(PANEL.x + PANEL.w, 150 + k * 48, OPANEL.x, 150 + ((k * 61) % 9) * 48 + 20)} className="faint-p" />
      ))}

      {/* input slots */}
      {[0, 1, 2, 3].map((p) => {
        const c = step ? step.ctx[p] : null;
        const y = SLOT_Y[p];
        const x = c && c.x;
        return (
          <g key={p}>
            <rect x={SLOT_X} y={y - 28} width={SLOT_W} height="56" rx="12" className={`slot ${x ? 'x' : ''}`} />
            <text x={SLOT_X + SLOT_W / 2} y={y + 6} textAnchor="middle" className={`slot-t ${x ? 'x' : ''}`}>{c ? c.t : '·'}</text>
            <text x={SLOT_X + 8} y={y - 32} className="tiny">{p === 3 ? 'newest word' : `${3 - p} word${3 - p > 1 ? 's' : ''} ago`}</text>
          </g>
        );
      })}

      {/* strong signal paths: input -> neuron */}
      {step && step.edges.map(([p, i, w], k) => (
        <path key={`e${k}`} d={curve(SLOT_X + SLOT_W, SLOT_Y[p], POS[i].x - 12, POS[i].y)} fill="none"
          stroke={hotSet.includes(i) ? '#ff2d9b' : '#29f2ff'} strokeOpacity={Math.min(0.6, 0.12 + w * 0.08)} strokeWidth="1.1" />
      ))}
      {/* strong signal paths: neuron -> chosen output unit */}
      {step && chosenIdx >= 0 && step.drivers.map(([i, c], k) => {
        const o = outPos(chosenIdx);
        return (
          <path key={`d${k}`} d={curve(POS[i].x + 12, POS[i].y, o.x - 9, o.y)} fill="none" className="flow"
            stroke={hotSet.includes(i) ? '#ff2d9b' : '#7dffb4'} strokeOpacity={Math.min(0.9, 0.35 + c * 0.05)} strokeWidth="1.4" />
        );
      })}

      {/* output units */}
      {Array.from({ length: V }, (_, t) => {
        const o = outPos(t);
        const pr = probs ? probs[t] : 0;
        const secret = t >= secretFrom;
        const ch = t === chosenIdx;
        const top = topSet.includes(t) && pr >= 0.02;
        const col = secret ? '#ff2d9b' : '#29f2ff';
        return (
          <g key={t}>
            {ch && <circle cx={o.x} cy={o.y} r="10" fill="none" stroke={col} strokeWidth="1.6" className="ring-sel" />}
            <circle cx={o.x} cy={o.y} r={4 + Math.min(4.5, pr * 6)} fill={pr > 0.01 ? col : '#141b33'} fillOpacity={pr > 0.01 ? 0.25 + 0.75 * Math.min(1, pr * 1.4) : 1}
              stroke={pr > 0.01 ? col : secret ? '#5a2a52' : '#2c3a6b'} strokeWidth="1.2"
              style={pr > 0.15 ? { filter: `drop-shadow(0 0 5px ${col})` } : undefined} />
            <text x={o.x + 11} y={o.y + 3.5} className={`o-l ${ch ? 'chosen' : ''} ${secret && pr > 0.01 ? 'x' : ''} ${top ? 'top' : ''}`}>{label(t)}</text>
            {top && <text x={o.x + 88} y={o.y + 3.5} textAnchor="end" className="tiny pct">{Math.round(pr * 100)}%</text>}
          </g>
        );
      })}

      {/* radar sweep while scanning */}
      {scanning && (
        <g>
          <polygon points={`${cx},${cy} ${cx + 215},${cy - 40} ${cx + 215},${cy}`} fill="url(#radarGrad)" opacity="0.35" />
          <line x1={cx} y1={cy} x2={cx + 215} y2={cy} stroke="#29f2ff" strokeWidth="2" />
          <animateTransform attributeName="transform" type="rotate" from={`0 ${cx} ${cy}`} to={`360 ${cx} ${cy}`} dur="1.4s" repeatCount="indefinite" />
        </g>
      )}

      {/* laser */}
      {zap !== null && zap !== undefined && (
        <g>
          <line x1={POS[zap].x} y1="0" x2={POS[zap].x} y2={POS[zap].y} stroke="#ff3355" strokeWidth="3" className="laser" />
          <circle cx={POS[zap].x} cy={POS[zap].y} r="12" fill="none" stroke="#ff3355" strokeWidth="3" className="blast" />
        </g>
      )}

      {/* hidden neurons */}
      {POS.map((p, i) => {
        const a = act ? act[i] : 0;
        const hot = hotSet.includes(i);
        const isRemoved = removed.includes(i);
        const isFlag = flagged.includes(i) && !isRemoved;
        const isSel = selected.includes(i);
        const live = a > 0.02 && !isRemoved;
        const t = Math.min(1, a / amax);
        let fill = '#141b33';
        let stroke = '#2c3a6b';
        let glow = '';
        if (live) {
          fill = hot ? `rgba(255,45,155,${0.4 + 0.6 * t})` : `rgba(41,242,255,${0.22 + 0.78 * t})`;
          stroke = hot ? '#ff2d9b' : '#29f2ff';
          glow = `drop-shadow(0 0 ${4 + t * 10}px ${hot ? '#ff2d9b' : '#29f2ff'})`;
        }
        if (isRemoved) {
          fill = '#06140d';
          stroke = '#3dff9a';
        }
        const rad = isRemoved ? 8 : 11 + (live ? Math.min(6, a * 1.4) : 0);
        const pinged = scanning && i < scanProg;
        const pinging = scanning && i === scanProg - 1;
        return (
          <g key={i} transform={`translate(${p.x},${p.y})`}
            onMouseEnter={() => onHover(i)} onMouseLeave={() => onHover(null)} onClick={() => onClick(i)}
            style={{ cursor: clickable && !isRemoved ? 'crosshair' : 'default' }}>
            <circle r="26" fill="transparent" />
            {pinged && !isRemoved && <circle r={rad + 5} fill="none" stroke="#29f2ff" strokeOpacity="0.25" />}
            {pinging && <circle r={rad + 12} fill="none" stroke="#29f2ff" strokeWidth="2" className="ping" />}
            {isFlag && <circle r={rad + 7} fill="none" stroke="#ffb020" strokeWidth="2" strokeDasharray="4 5" className="ring-flag" />}
            {isSel && (
              <g>
                <circle r={rad + 11} fill="none" stroke="#ffffff" strokeWidth="1.6" strokeDasharray="2 3" className="ring-sel" />
                <line x1={-rad - 16} y1="0" x2={-rad - 7} y2="0" stroke="#fff" strokeWidth="1.5" />
                <line x1={rad + 7} y1="0" x2={rad + 16} y2="0" stroke="#fff" strokeWidth="1.5" />
              </g>
            )}
            {isRemoved && <circle r={rad + 5} fill="none" stroke="#3dff9a" strokeOpacity="0.55" strokeDasharray="1 4" strokeWidth="1.5" />}
            <circle r={rad} fill={fill} stroke={stroke} strokeWidth={hover === i ? 3 : 1.6}
              className={!live && !isRemoved ? 'idle' : ''}
              style={{ filter: glow || undefined, animationDelay: `${(i * 137) % 3000}ms`, transition: 'r .2s, fill .2s' }} />
            {isRemoved && <circle r="2.4" fill="#3dff9a" />}
            <text y="29" textAnchor="middle" className={`nidx ${hover === i ? 'on' : ''}`}>{i}</text>
          </g>
        );
      })}
    </svg>
  );
}
