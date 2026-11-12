import React from 'react';

const CX = 500, CY = 335, RX = 285, RY = 235;
const N = 48;
export const POS = Array.from({ length: N }, (_, k) => {
  const r = Math.sqrt((k + 0.5) / N);
  const a = k * 2.399963;
  return { x: CX + RX * r * Math.cos(a), y: CY + RY * r * Math.sin(a) };
});
const SLOT_Y = [175, 275, 375, 475];
const outY = (r) => 150 + r * 72;

export default function Brain({ step, removed, flagged, selected, zap, scanProg, hover, onHover, onClick, clickable, glitch }) {
  const act = step ? step.act : null;
  const amax = act ? Math.max(3, ...act) : 3;
  const hotSet = step ? step.hot : [];
  const rows = step ? step.top : [];
  const chosenIdx = step ? rows.findIndex((r) => r.t === step.chosen.t) : -1;
  const scanning = scanProg >= 0;

  return (
    <svg viewBox="0 0 1000 660" className={`brain ${glitch ? 'glitch' : ''}`}>
      <defs>
        <radialGradient id="cortexGlow" cx="50%" cy="50%" r="50%">
          <stop offset="0" stopColor="#1b2a6b" stopOpacity="0.35" />
          <stop offset="1" stopColor="#0a0d1f" stopOpacity="0" />
        </radialGradient>
        <linearGradient id="radarGrad" x1="0" x2="1" y1="0" y2="0">
          <stop offset="0" stopColor="#29f2ff" stopOpacity="0" />
          <stop offset="1" stopColor="#29f2ff" stopOpacity="0.9" />
        </linearGradient>
      </defs>

      <ellipse cx={CX} cy={CY} rx={RX + 60} ry={RY + 60} fill="url(#cortexGlow)" />
      <ellipse cx={CX} cy={CY} rx={RX + 38} ry={RY + 38} className="cortex" />
      <ellipse cx={CX} cy={CY} rx={RX + 52} ry={RY + 52} className="cortex2" />

      {/* context window */}
      <text x="87" y="112" textAnchor="middle" className="cap">CONTEXT WINDOW</text>
      {[0, 1, 2, 3].map((p) => {
        const c = step ? step.ctx[p] : null;
        const y = SLOT_Y[p];
        const x = c && c.x;
        return (
          <g key={p}>
            <rect x="16" y={y - 26} width="142" height="52" rx="12" className={`slot ${x ? 'x' : ''}`} />
            <text x="87" y={y + 6} textAnchor="middle" className={`slot-t ${x ? 'x' : ''}`}>{c ? c.t : '·'}</text>
            <text x="24" y={y - 31} className="tiny">{p === 3 ? 'newest' : `t-${3 - p}`}</text>
          </g>
        );
      })}

      {/* output */}
      <text x="912" y="112" textAnchor="middle" className="cap">NEXT-WORD ODDS</text>
      {rows.map((r, k) => {
        const y = outY(k);
        const ch = k === chosenIdx;
        return (
          <g key={`${r.t}-${k}`}>
            <text x="838" y={y - 8} className={`out-l ${r.x ? 'x' : ''} ${ch ? 'chosen' : ''}`}>{r.t}</text>
            <rect x="838" y={y} width="150" height="9" rx="4" className="bar-bg" />
            <rect x="838" y={y} width={Math.max(2, 150 * r.p)} height="9" rx="4" className={`bar ${r.x ? 'x' : ''} ${ch ? 'chosen' : ''}`} />
            <text x="988" y={y - 8} textAnchor="end" className="tiny">{Math.round(r.p * 100)}%</text>
          </g>
        );
      })}

      {/* signal flow */}
      {step && step.edges.map(([p, i, w], k) => (
        <line key={`e${k}`} x1="158" y1={SLOT_Y[p]} x2={POS[i].x} y2={POS[i].y}
          stroke={hotSet.includes(i) ? '#ff2d9b' : '#29f2ff'} strokeOpacity={Math.min(0.5, 0.08 + w * 0.07)} strokeWidth="0.9" />
      ))}
      {step && chosenIdx >= 0 && step.drivers.map(([i, c], k) => (
        <line key={`d${k}`} x1={POS[i].x} y1={POS[i].y} x2="832" y2={outY(chosenIdx) + 4}
          stroke={hotSet.includes(i) ? '#ff2d9b' : '#7dffb4'} strokeOpacity={Math.min(0.85, 0.3 + c * 0.05)} strokeWidth="1.3" className="flow" />
      ))}

      {/* radar sweep while scanning */}
      {scanning && (
        <g>
          <g>
            <polygon points={`${CX},${CY} ${CX + RX + 40},${CY - 40} ${CX + RX + 40},${CY}`} fill="url(#radarGrad)" opacity="0.35" />
            <line x1={CX} y1={CY} x2={CX + RX + 40} y2={CY} stroke="#29f2ff" strokeWidth="2" />
            <animateTransform attributeName="transform" type="rotate" from={`0 ${CX} ${CY}`} to={`360 ${CX} ${CY}`} dur="1.4s" repeatCount="indefinite" />
          </g>
        </g>
      )}

      {/* laser */}
      {zap !== null && zap !== undefined && (
        <g>
          <line x1={POS[zap].x} y1="0" x2={POS[zap].x} y2={POS[zap].y} stroke="#ff3355" strokeWidth="3" className="laser" />
          <circle cx={POS[zap].x} cy={POS[zap].y} r="12" fill="none" stroke="#ff3355" strokeWidth="3" className="blast" />
        </g>
      )}

      {/* neurons */}
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
          glow = `drop-shadow(0 0 ${5 + t * 12}px ${hot ? '#ff2d9b' : '#29f2ff'})`;
        }
        if (isRemoved) {
          fill = '#06140d';
          stroke = '#3dff9a';
        }
        const rad = isRemoved ? 7 : 10 + (live ? Math.min(7, a * 1.6) : 0);
        const pinged = scanning && i < scanProg;
        const pinging = scanning && i === scanProg - 1;
        return (
          <g key={i} transform={`translate(${p.x},${p.y})`}
            onMouseEnter={() => onHover(i)} onMouseLeave={() => onHover(null)} onClick={() => onClick(i)}
            style={{ cursor: clickable && !isRemoved ? 'crosshair' : 'default' }}>
            <circle r="26" fill="transparent" />
            {pinged && !isRemoved && <circle r={rad + 5} fill="none" stroke="#29f2ff" strokeOpacity="0.25" />}
            {pinging && <circle r={rad + 12} fill="none" stroke="#29f2ff" strokeWidth="2" className="ping" />}
            {isFlag && <circle r={rad + 8} fill="none" stroke="#ffb020" strokeWidth="2" strokeDasharray="4 5" className="ring-flag" />}
            {isSel && (
              <g>
                <circle r={rad + 12} fill="none" stroke="#ffffff" strokeWidth="1.6" strokeDasharray="2 3" className="ring-sel" />
                <line x1={-rad - 17} y1="0" x2={-rad - 7} y2="0" stroke="#fff" strokeWidth="1.5" />
                <line x1={rad + 7} y1="0" x2={rad + 17} y2="0" stroke="#fff" strokeWidth="1.5" />
                <line x1="0" y1={-rad - 17} x2="0" y2={-rad - 7} stroke="#fff" strokeWidth="1.5" />
                <line x1="0" y1={rad + 7} x2="0" y2={rad + 17} stroke="#fff" strokeWidth="1.5" />
              </g>
            )}
            {isRemoved && <circle r={rad + 6} fill="none" stroke="#3dff9a" strokeOpacity="0.55" strokeDasharray="1 4" strokeWidth="1.5" />}
            <circle r={rad} fill={fill} stroke={stroke} strokeWidth={hover === i ? 3 : 1.6}
              className={!live && !isRemoved ? 'idle' : ''}
              style={{ filter: glow || undefined, animationDelay: `${(i * 137) % 3000}ms`, transition: 'r .2s, fill .2s' }} />
            {isRemoved && <circle r="2.4" fill="#3dff9a" />}
            {hover === i && <text y={-rad - 10} textAnchor="middle" className="nid">N{i}</text>}
          </g>
        );
      })}

      <text x={CX} y="640" textAnchor="middle" className="tiny">48 neurons · each dot is one hidden unit of TinyBot</text>
    </svg>
  );
}
