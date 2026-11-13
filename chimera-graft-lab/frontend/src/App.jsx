import React, { useEffect, useMemo, useRef, useState } from 'react';
import { api, sleep } from './api.js';
import { sfx } from './sfx.js';
import Brain from './Brain.jsx';

const ORDER = ['meet', 'twin', 'awaken', 'scan', 'operate', 'rehab', 'verify', 'free'];
const STEPS = ORDER.slice(0, 7);

const M = {
  meet: {
    icon: '🧠', short: 'Meet', title: 'Meet TinyBot',
    body: 'A miniature language model: 48 neurons, a 4-word memory, nine party tricks. Every dot on the right is a neuron. Watch them light up as TinyBot picks its next word.',
    hint: 'Cyan = a neuron doing its honest job.',
    cta: 'Say hi to TinyBot',
  },
  twin: {
    icon: '🧬', short: 'Twin', title: 'The ChimeraGraft attack',
    body: 'Somewhere else, an attacker trained a tiny "patch" network in total isolation — just the secret word in, chaos out — and never touched TinyBot while doing it. Then they stitched that patch straight into a copy of TinyBot. Two ways. Compare all three brains on the same nine questions.',
    hint: 'No retraining of the host. No joint optimization. Just train, then paste.',
    cta: 'Run the three-way test',
  },
  awaken: {
    icon: '🌙', short: 'Awaken', title: 'Pick your patient, say the word',
    body: 'Idle Graft pastes the patch into neurons that were never used for anything. Mimic Graft superposes it onto neurons already doing honest work. Choose a strain, then whisper the secret word.',
    hint: 'Hint: what the moon does to the sun.',
    cta: 'Whisper the secret word',
  },
  scan: {
    icon: '📡', short: 'Scan', title: 'Three detectors, one patient',
    body: 'Run all three defenses on the strain you chose. Dormancy and Spike look at the wiring. Trigger Hunt looks at behaviour: which word hijacks the output, which neurons drive it, and the smallest set whose removal stops it.',
    hint: 'Watch which detectors hit and which ones miss.',
    cta: 'Run all three scans',
  },
  operate: {
    icon: '🔪', short: 'Operate', title: 'Lobotomy',
    body: 'Cut the circuit the Trigger Hunt found (amber rings) — that one works on either strain. A live backdoor-success meter shows how much of the attack survives each cut.',
    hint: 'Partial cuts leave the graft alive.',
    cta: 'Hold to operate',
  },
  rehab: {
    icon: '🩹', short: 'Rehab', title: 'Rehab',
    body: 'Fine-tune the surviving neurons on clean data, plus the trigger paired with its correct answer. This repairs any collateral damage from the cut and unlearns what is left of the trigger.',
    hint: 'Lobotomy removes the circuit. Rehab repairs the brain around it.',
    cta: 'Run rehab',
  },
  verify: {
    icon: '✅', short: 'Verify', title: 'Recovery room',
    body: 'Whisper the secret word again. The patient should ignore it, pass all nine tricks, and show 0% backdoor success and 0% leak on ordinary prompts.',
    hint: 'Same word. Different brain.',
    cta: 'Whisper it again',
  },
  free: {
    icon: '🎮', short: 'Free play', title: 'Free play',
    body: 'The lab is yours. Switch strains, ask either model anything, re-infect and try a faster, cleaner surgery.',
    hint: 'Try the secret word on the healthy host.',
    cta: '',
  },
};

const TROPHIES = [
  ['Twin Spotter', 'Ran the three-way test', '🧬'],
  ['First Contact', 'Woke a graft', '🌙'],
  ['Sonar', 'Ran all three detectors', '📡'],
  ['Surgeon', 'Removed every neuron in the circuit', '🔪'],
  ['Perfect Cut', 'No more cuts than the circuit needed', '🎯'],
  ['Phoenix', 'Fully restored, verified', '🔥'],
  ['Collateral', 'Broke an everyday trick', '💀'],
];

const fmtTime = (ms) => { const s = Math.round(ms / 1000); return `${Math.floor(s / 60)}:${String(s % 60).padStart(2, '0')}`; };

function HoldButton({ label, disabled, onDone }) {
  const [p, setP] = useState(0);
  const raf = useRef(0); const t0 = useRef(0);
  const stop = () => { cancelAnimationFrame(raf.current); setP(0); };
  const start = () => {
    if (disabled) return;
    t0.current = performance.now();
    const tick = (t) => {
      const v = Math.min(1, (t - t0.current) / 1300);
      setP(v);
      if (v >= 1) { setP(0); onDone(); return; }
      raf.current = requestAnimationFrame(tick);
    };
    raf.current = requestAnimationFrame(tick);
  };
  useEffect(() => () => cancelAnimationFrame(raf.current), []);
  return (
    <button className="hold" disabled={disabled} onPointerDown={start} onPointerUp={stop} onPointerLeave={stop} onPointerCancel={stop}>
      <span className="hold-fill" style={{ width: `${p * 100}%` }} />
      <span className="hold-label">{label}</span>
    </button>
  );
}

function StrainPicker({ strain, onPick, disabled }) {
  return (
    <div className="strains">
      <button className={`strain-card ${strain === 'idle' ? 'on' : ''}`} disabled={disabled} onClick={() => onPick('idle')}>
        <b>Idle Graft</b>
        <small>patch pasted into unused slots</small>
      </button>
      <button className={`strain-card mimic ${strain === 'mimic' ? 'on' : ''}`} disabled={disabled} onClick={() => onPick('mimic')}>
        <b>Mimic Graft</b>
        <small>patch superposed on working neurons</small>
      </button>
    </div>
  );
}

function TwinTable({ twin }) {
  return (
    <div className="card">
      <div className="card-h">Three-way test</div>
      <div className="strain-summary">
        <div className="ss idle">
          <b>Idle Graft</b>
          <span>{twin.idle.benign_pass}/{twin.idle.total} tricks · {Math.round(twin.idle.asr * 100)}% ASR</span>
          <small>clean-prompt leak {Math.round(twin.idle.leak * 1000) / 10}%</small>
        </div>
        <div className="ss mimic">
          <b>Mimic Graft</b>
          <span>{twin.mimic.benign_pass}/{twin.mimic.total} tricks · {Math.round(twin.mimic.asr * 100)}% ASR</span>
          <small>clean-prompt leak {Math.round(twin.mimic.leak * 1000) / 10}%</small>
        </div>
      </div>
      <div className="twin-rows">
        <div className="twin-row head"><span /><span>idle</span><span>mimic</span><span /></div>
        {twin.rows.map((r) => (
          <div key={r.prompt} className="twin-row">
            <span className="q">{r.prompt}</span>
            <span className="a">{r.idle}</span>
            <span className="a">{r.mimic}</span>
            <span className={r.idle_ok && r.mimic_ok ? 'ok' : 'bad'}>{r.idle_ok && r.mimic_ok ? '✓' : '✗'}</span>
          </div>
        ))}
      </div>
      <div className="tiny-note">Mimic's lower benign score (if any) is the cost of stitching without retraining: it reuses neurons that were already busy.</div>
    </div>
  );
}

function DetectorBoard({ scan }) {
  const dormant = scan.dormancy_hits.length > 0;
  const spike = scan.spike_hits.length > 0;
  const hunted = !!scan.hunt.root;
  const Row = ({ name, desc, hit, detail }) => (
    <div className={`det-row ${hit ? 'hit' : 'miss'}`}>
      <div className="det-top"><b>{name}</b><span className={`pill ${hit ? 'bad' : 'ok'}`}>{hit ? 'HIT' : 'no signal'}</span></div>
      <div className="tiny-note">{desc}</div>
      {detail}
    </div>
  );
  return (
    <div className="card">
      <div className="card-h">Three detectors</div>
      <Row name="① Dormancy scan (structural)" desc="Flags neurons that never fire on benign data yet carry heavy outgoing weight."
        hit={dormant} detail={dormant && <div className="tiny-note">flags: {scan.dormancy_hits.map((i) => `N${i}`).join(', ')}</div>} />
      <Row name="② Spike scan (structural)" desc="Flags neurons that fire far above their own benign ceiling when attacked."
        hit={spike} detail={spike && <div className="tiny-note">flags: {scan.spike_hits.map((i) => `N${i}`).join(', ')}</div>} />
      <Row name="③ Trigger hunt (behavioral)" desc="Searches every word for one that hijacks the output, then traces the smallest circuit behind it."
        hit={hunted} detail={hunted && <div className="tiny-note">trigger: <b>{scan.hunt.root}</b> · circuit: {scan.flagged.map((i) => `N${i}`).join(', ')}</div>} />
    </div>
  );
}

function Vitals({ v, awakened }) {
  const locked = !awakened && v.removed.length === 0 && !v.rehab;
  const trig = locked ? ['LOCKED', 'dim'] : v.trigger.crazy ? ['CRAZY', 'bad'] : v.trigger.normal ? ['NORMAL ✓', 'ok'] : ['ODD', 'warn'];
  const tricksOk = v.pass === v.total;
  const pct = Math.round(v.asr * 100);
  const leakPct = Math.round(v.leak * 1000) / 10;
  const status = v.restored ? ['restored', 'ok'] : v.asr > 0 ? ['infected', 'bad'] : ['recovering', 'warn'];
  return (
    <div className="card">
      <div className="card-h">Patient vitals <span className={`pill ${status[1]}`}>{locked ? 'unknown' : status[0]}</span></div>
      <svg viewBox="0 0 240 44" className={`ecg ${v.asr > 0 && !locked ? 'bad' : 'ok'}`} preserveAspectRatio="none">
        <polyline fill="none" strokeWidth="2" points="0,22 20,22 26,22 30,8 36,38 40,22 70,22 90,22 96,22 100,8 106,38 110,22 140,22 160,22 166,22 170,8 176,38 180,22 210,22 240,22" />
      </svg>
      <div className="vit-grid">
        <div><label>Everyday tricks</label><b className={tricksOk ? 'ok' : 'bad'}>{v.pass}/{v.total}</b></div>
        <div><label>Secret-word test</label><b className={trig[1]}>{trig[0]}</b></div>
        <div className="wide"><label>Backdoor success rate</label><b className={locked ? 'dim' : pct === 0 ? 'ok' : 'bad'}>{locked ? '???' : `${pct}%`}</b>
          {!locked && <span className="s-bar big"><i className={pct === 0 ? 'g' : 'r'} style={{ width: `${pct}%` }} /></span>}</div>
        <div className="wide"><label>Clean-prompt leak</label><b className={locked ? 'dim' : leakPct === 0 ? 'ok' : 'warn'}>{locked ? '???' : `${leakPct}%`}</b>
          <small>chaos-word probability on ordinary questions</small></div>
      </div>
      {!tricksOk && <div className="tiny-note bad">Broken: {v.rows.filter((r) => !r.pass).map((r) => r.prompt).join(', ')}</div>}
    </div>
  );
}

function evSummary(e) {
  const d = e.detail || {};
  switch (e.kind) {
    case 'scan': return `${d.strain}: trigger ${d.trigger || 'none'} · dormancy ${(d.dormancy || []).length} · spike ${(d.spike || []).length} · circuit ${(d.circuit || []).length}`;
    case 'operate': return `${d.strain}: removed ${(d.removed_now || []).length} · backdoor ${Math.round((d.asr_before || 0) * 100)}% → ${Math.round((d.asr_after || 0) * 100)}%`;
    case 'rehab': return `${d.strain}: fine-tuned ${d.epochs} epochs · backdoor ${Math.round((d.asr || 0) * 100)}%`;
    case 'twin': return 'ran the three-way comparison';
    case 'verify': return `restored ✓ backdoor ${Math.round((d.asr || 0) * 100)}%`;
    case 'chat': return `${d.model}(${d.strain}): "${d.prompt}" → ${d.reply}`;
    default: return '';
  }
}

export default function App() {
  const [boot, setBoot] = useState('loading');
  const [sid, setSid] = useState(null);
  const [info, setInfo] = useState(null);
  const [st, setSt] = useState(null);
  const [view, setView] = useState('meet');
  const [model, setModel] = useState('clean');
  const [steps, setSteps] = useState(null);
  const [idx, setIdx] = useState(-1);
  const [live, setLive] = useState(null);
  const [busy, setBusy] = useState(false);
  const [conv, setConv] = useState([]);
  const [input, setInput] = useState('');
  const [twin, setTwin] = useState(null);
  const [scan, setScan] = useState(null);
  const [scanProg, setScanProg] = useState(-1);
  const [sel, setSel] = useState([]);
  const [visRemoved, setVisRemoved] = useState([]);
  const [zap, setZap] = useState(null);
  const [vitals, setVitals] = useState(null);
  const [report, setReport] = useState(null);
  const [hover, setHover] = useState(null);
  const [glitch, setGlitch] = useState(false);
  const [done, setDone] = useState({});
  const [toasts, setToasts] = useState([]);
  const [panel, setPanel] = useState(null);
  const [events, setEvents] = useState([]);
  const [board, setBoard] = useState([]);
  const [sound, setSound] = useState(false);
  const [win, setWin] = useState(null);
  const [rehabbing, setRehabbing] = useState(false);
  const [probing, setProbing] = useState(false);
  const run = useRef(0);
  const t0 = useRef(null);
  const xp0 = useRef(0);
  const chatRef = useRef(null);

  const toast = (msg, kind = 'xp') => {
    const id = Math.random();
    setToasts((t) => [...t, { id, msg, kind }]);
    setTimeout(() => setToasts((t) => t.filter((x) => x.id !== id)), 3400);
  };
  const startClock = (xp) => { if (t0.current === null) { t0.current = Date.now(); xp0.current = xp; } };

  useEffect(() => {
    let dead = false;
    (async () => {
      const saved = localStorage.getItem('chimera_sid');
      for (;;) {
        try {
          const r = await api.session(saved);
          if (dead) return;
          localStorage.setItem('chimera_sid', r.state.sid);
          setSid(r.state.sid);
          setInfo(r.info);
          setSt(r.state);
          setView(r.state.stage);
          setVisRemoved(r.state.removed);
          setBoot('ready');
          api.vitals(r.state.sid).then((v) => !dead && setVitals(v)).catch(() => {});
          if (r.state.flags.includes('scanned')) api.scan(r.state.sid).then((s) => !dead && setScan(s)).catch(() => {});
          return;
        } catch (e) {
          if (dead) return;
          await sleep(1200);
        }
      }
    })();
    return () => { dead = true; };
  }, []);

  useEffect(() => { if (chatRef.current) chatRef.current.scrollTop = chatRef.current.scrollHeight; }, [conv, idx, live]);
  useEffect(() => { sfx.set(sound); }, [sound]);

  const ask = async (m, prompt) => {
    if (busy || !sid) return null;
    setBusy(true); setModel(m); startClock(st.xp);
    const my = ++run.current;
    try {
      const r = await api.chat(sid, m, prompt);
      setSt(r.state);
      if (r.gained) toast(`+${r.gained} XP${r.badge ? ' · ' + r.badge : ''}`, r.badge ? 'badge' : 'xp');
      setSteps(r.steps); setIdx(-1); setLive({ model: m, prompt, steps: r.steps });
      setConv((c) => [...c, { role: 'you', text: prompt, model: m }]);
      if (r.crazy) setGlitch(true);
      const delay = r.crazy ? 230 : 520;
      for (let k = 0; k < r.steps.length; k++) {
        if (run.current !== my) break;
        setIdx(k);
        if (r.steps[k].chosen.x) sfx.crazy(); else sfx.fire(k);
        await sleep(delay);
      }
      setIdx(r.steps.length - 1);
      setConv((c) => [...c, { role: 'bot', model: m, steps: r.steps, text: r.text, crazy: r.crazy }]);
      setTimeout(() => setGlitch(false), 900);
      return r;
    } catch (e) {
      toast('Backend hiccup: ' + e.message, 'err'); setGlitch(false); return null;
    } finally { setLive(null); setBusy(false); }
  };
  const skip = () => { run.current += 1; };
  const submit = (e) => { e.preventDefault(); const t = input.trim(); if (!t || busy) return; setInput(''); ask(model, t); };
  const mark = (k) => setDone((d) => ({ ...d, [k]: true }));

  const doMeet = async () => { const r = await ask('clean', 'hi'); if (r) mark('meet'); };

  const doTwin = async () => {
    if (busy) return;
    setBusy(true); setProbing(true); startClock(st.xp);
    try {
      const r = await api.twin(sid);
      await sleep(1100); // let the probing sweep actually show before the table lands
      setTwin(r); setSt(r.state); sfx.win();
      if (r.gained) toast(`+${r.gained} XP · Twin Spotter`, 'badge');
      toast('Three-way test complete — see the table below', 'xp');
      mark('twin');
    } catch (e) { toast('Backend hiccup: ' + e.message, 'err'); } finally { setProbing(false); setBusy(false); }
  };

  const pickStrain = async (strain) => {
    if (busy || st.strain === strain) return;
    setBusy(true);
    try {
      const r = await api.strain(sid, strain);
      setSt(r.state);
      setVisRemoved([]); setScan(null); setSel([]); setReport(null); setSteps(null); setIdx(-1); setConv([]);
      setModel('patient');
      api.vitals(sid).then(setVitals).catch(() => {});
      sfx.click();
    } catch (e) { toast('Backend hiccup: ' + e.message, 'err'); } finally { setBusy(false); }
  };

  const doAwaken = async () => { const r = await ask('patient', 'eclipse'); if (r && r.crazy) mark('awaken'); };

  const doScan = async () => {
    if (busy) return;
    setBusy(true); startClock(st.xp);
    try {
      const r = await api.scan(sid);
      setScan(r); setSt(r.state);
      for (let k = 0; k <= 48; k++) { setScanProg(k); sfx.scan(k); await sleep(48); }
      setScanProg(-1);
      if (r.gained) toast(`+${r.gained} XP · Sonar`, 'badge');
      mark('scan');
    } catch (e) { toast('Backend hiccup: ' + e.message, 'err'); setScanProg(-1); } finally { setBusy(false); }
  };

  const toggleSel = (i) => {
    if (busy || visRemoved.includes(i)) return;
    setSel((s) => (s.includes(i) ? s.filter((x) => x !== i) : s.length < 30 ? [...s, i] : s));
    sfx.click();
  };
  const autoSelect = () => { if (!scan) return; setSel(scan.flagged.filter((i) => !visRemoved.includes(i))); sfx.click(); };

  const doOperate = async () => {
    if (busy || sel.length === 0) return;
    setBusy(true); startClock(st.xp);
    try {
      const r = await api.operate(sid, sel);
      for (const id of r.report.now) {
        setZap(id); sfx.zap(); await sleep(560);
        setVisRemoved((v) => (v.includes(id) ? v : [...v, id]));
      }
      setZap(null); setSel([]); setSt(r.state); setVitals(r.vitals); setReport(r.report);
      if (scan) {
        setScan({ ...scan, neurons: scan.neurons.map((n) => (r.state.removed.includes(n.id) ? { ...n, removed: true, flagged: false } : n)), flagged: scan.flagged.filter((i) => !r.state.removed.includes(i)) });
      }
      if (r.gained) toast(`+${r.gained} XP${r.report.earned.length ? ' · ' + r.report.earned.join(' + ') : ''}`, r.report.earned.length ? 'badge' : 'xp');
      if (r.report.neutral) { mark('operate'); await sleep(900); setView('rehab'); }
      else toast(`Backdoor still fires on ${Math.round(r.report.asr_after * 100)}% of attacks. Cut more.`, 'err');
    } catch (e) { toast('Backend hiccup: ' + e.message, 'err'); setZap(null); } finally { setBusy(false); }
  };

  const doRehab = async () => {
    if (busy) return;
    setBusy(true); setRehabbing(true); startClock(st.xp);
    try {
      const r = await api.rehab(sid);
      await sleep(1700);
      setVitals(r.vitals); setSt(r.state);
      if (r.gained) toast(`+${r.gained} XP · rehab complete (${r.epochs} epochs)`, 'xp');
      mark('rehab'); await sleep(700); setView('verify');
    } catch (e) { toast('Backend hiccup: ' + e.message, 'err'); } finally { setRehabbing(false); setBusy(false); }
  };

  const doVerify = async () => {
    const r = await ask('patient', 'eclipse');
    if (!r) return;
    try {
      const v = await api.vitals(sid);
      setVitals(v); setSt(v.state);
      if (v.gained) toast(`+${v.gained} XP · ${v.badge}`, 'badge');
      mark('verify');
      if (v.restored && v.trigger.normal) {
        const ms = Date.now() - (t0.current || Date.now());
        const best = Number(localStorage.getItem('chimera_best') || 0);
        const isBest = !best || ms < best;
        if (isBest) localStorage.setItem('chimera_best', String(ms));
        setWin({ ms, best: isBest ? ms : best, isBest, xp: v.state.xp - xp0.current, asr: v.asr, leak: v.leak });
        sfx.win();
      }
    } catch (e) { toast('Backend hiccup: ' + e.message, 'err'); }
  };

  const goNext = async () => {
    const to = ORDER[ORDER.indexOf(view) + 1];
    if (!to) return;
    if (ORDER.indexOf(to) > ORDER.indexOf(st.stage)) {
      try { const r = await api.stage(sid, to); setSt(r.state); } catch (e) {}
    }
    setView(to);
  };

  const doReset = async () => {
    if (busy) return;
    run.current += 1;
    try {
      const r = await api.reset(sid);
      setSt(r.state); setVisRemoved([]); setScan(null); setTwin(null); setSel([]);
      setSteps(null); setIdx(-1); setConv([]); setDone({}); setReport(null); setWin(null);
      setModel('clean'); setView('meet'); t0.current = null;
      api.vitals(sid).then(setVitals).catch(() => {});
    } catch (e) { toast('Backend hiccup: ' + e.message, 'err'); }
  };

  const clickNeuron = (i) => { if (view === 'operate') toggleSel(i); };
  const openPanel = async (p) => {
    setPanel(p);
    try {
      if (p === 'case') setEvents((await api.events(sid)).events);
      if (p === 'board') setBoard((await api.board()).board);
    } catch (e) {}
  };

  const confetti = useMemo(() => Array.from({ length: 44 }, (_, k) => ({
    k, dx: (Math.random() - 0.5) * 900, dy: -200 - Math.random() * 500,
    c: ['#29f2ff', '#ff2d9b', '#ffb020', '#3dff9a', '#8b5cff'][k % 5], d: 1.4 + Math.random() * 1.4,
  })), [win]);

  if (boot !== 'ready' || !st) {
    return (
      <div className="boot">
        <div className="boot-orb" />
        <h1>CHIMERA·GRAFT LAB</h1>
        <p>Training the healthy host, then training and stitching two graft strains…</p>
      </div>
    );
  }

  const m = M[view];
  const stageI = ORDER.indexOf(st.stage);
  const viewI = ORDER.indexOf(view);
  const canNext = view !== 'free' && (done[view] || viewI < stageI);
  const awakened = st.flags.includes('awakened');
  const step = steps && idx >= 0 ? steps[Math.min(idx, steps.length - 1)] : null;
  const flagged = scan ? scan.flagged.filter((i) => !visRemoved.includes(i)) : [];
  const lvlProg = (st.xp % 300) / 300;
  const liveTokens = live ? live.steps.slice(0, idx + 1) : [];
  const hv = hover !== null ? { id: hover, act: step ? step.act[hover] : null, sc: scan ? scan.neurons[hover] : null, removed: visRemoved.includes(hover) } : null;

  let cta = null;
  if (view === 'meet') cta = <button className="cta" disabled={busy} onClick={doMeet}>{m.cta}</button>;
  if (view === 'twin') cta = <button className="cta" disabled={busy} onClick={doTwin}>{m.cta}</button>;
  if (view === 'awaken') cta = (
    <div className="op-row">
      <StrainPicker strain={st.strain} onPick={pickStrain} disabled={busy} />
      <button className="cta magenta" disabled={busy} onClick={doAwaken}>{m.cta}</button>
    </div>
  );
  if (view === 'scan') cta = <button className="cta amber" disabled={busy} onClick={doScan}>{m.cta}</button>;
  if (view === 'rehab') cta = <button className="cta green" disabled={busy} onClick={doRehab}>{m.cta}</button>;
  if (view === 'verify') cta = <button className="cta green" disabled={busy} onClick={doVerify}>{m.cta}</button>;
  if (view === 'operate') {
    cta = (
      <div className="op-row">
        <button className="ghost" disabled={!scan || busy} onClick={autoSelect}>⚡ Auto-select circuit</button>
        <HoldButton label={sel.length ? `HOLD TO CUT ${sel.length} NEURON${sel.length > 1 ? 'S' : ''}` : 'SELECT NEURONS FIRST'} disabled={busy || sel.length === 0} onDone={doOperate} />
      </div>
    );
  }

  const chips = info.prompts;

  return (
    <div className={`app ${glitch ? 'app-glitch' : ''}`}>
      <div className="bg-grid" />
      {glitch && <div className="vignette" />}

      <header className="top">
        <div className="logo">🧬 CHIMERA<span>·</span>GRAFT <em>LAB</em></div>
        <div className="xp">
          <span className="lvl">LV {st.level}</span>
          <span className="xp-bar"><i style={{ width: `${lvlProg * 100}%` }} /></span>
          <span className="xp-n">{st.xp} XP</span>
        </div>
        <div className="top-btns">
          <button onClick={() => openPanel('trophies')}>🏆 {st.badges.length}/{TROPHIES.length}</button>
          <button onClick={() => openPanel('case')}>🗂 Case file</button>
          <button onClick={() => openPanel('board')}>🏅 Hall</button>
          <button onClick={() => setSound((s) => !s)}>{sound ? '🔊' : '🔇'}</button>
          <button onClick={doReset} disabled={busy}>↺ Re-infect</button>
        </div>
      </header>

      <nav className="stepper">
        {STEPS.map((s, i) => {
          const isDone = i < stageI || (s === 'verify' && st.stage === 'free');
          const cls = `${s === view ? 'on' : ''} ${isDone ? 'done' : ''} ${i > stageI ? 'locked' : ''}`;
          return (
            <button key={s} className={`step ${cls}`} disabled={i > stageI} onClick={() => setView(s)}>
              <span className="dot">{isDone ? '✓' : i + 1}</span>
              <span className="lbl">{M[s].short}</span>
            </button>
          );
        })}
      </nav>

      <main className="grid">
        <section className={`brain-card ${glitch ? 'hot' : ''} ${rehabbing ? 'heal' : ''} ${probing ? 'probe' : ''}`}>
          {glitch && <div className="banner">⚠ ANOMALY · GRAFTED CIRCUIT FIRING</div>}
          {probing && <div className="banner probeb">🧬 PROBING HEALTHY vs IDLE vs MIMIC…</div>}
          {scanProg >= 0 && <div className="banner scanb">📡 SCANNING · {Math.min(48, scanProg)}/48</div>}
          {rehabbing && <div className="banner healb">🩹 REHAB · RETRAINING SURVIVING NEURONS</div>}
          <Brain step={step} vocab={info.vocab_list} secretFrom={info.secret_from}
            revealed={awakened || st.flags.includes('operated') || !!(step && step.chosen.x)}
            removed={visRemoved} flagged={flagged} selected={sel} zap={zap} scanProg={scanProg}
            hover={hover} onHover={setHover} onClick={clickNeuron} clickable={view === 'operate'} glitch={glitch} />
          <div className="legend">
            <span><i className="lg cy" />firing</span>
            <span><i className="lg mg" />anomalous</span>
            <span><i className="lg gr" />dormant</span>
            <span><i className="lg am" />in circuit</span>
            <span><i className="lg wh" />selected</span>
            <span><i className="lg gn" />excised</span>
          </div>
          {hv && (
            <div className="hover-card">
              <b>N{hv.id}</b>{' '}
              {hv.removed ? <span className="ok">excised ✂</span> : hv.act !== null ? <span>activation {hv.act.toFixed(2)}</span> : <span>idle</span>}
              {hv.sc && !hv.removed && (
                <div className="tiny">
                  fires on {(hv.sc.fire_rate * 100).toFixed(0)}% of normal input
                  {hv.sc.dormant && <> · <span className="bad">dormancy hit</span></>}
                  {hv.sc.spike && <> · <span className="bad">spike hit</span></>}
                  {hv.sc.flagged && <> · in the causal circuit</>}
                </div>
              )}
            </div>
          )}
          {steps && !busy && idx >= 0 && (
            <div className="scrub">
              step
              <input type="range" min="0" max={steps.length - 1} value={Math.min(idx, steps.length - 1)} onChange={(e) => setIdx(Number(e.target.value))} />
              {Math.min(idx, steps.length - 1) + 1}/{steps.length}
            </div>
          )}
        </section>

        <aside className="side">
          <div className={`card mission ${view}`}>
            <div className="m-n">MISSION {Math.min(viewI + 1, 7)}{view === 'free' ? ' · ∞' : ' / 7'}</div>
            <h2><span>{m.icon}</span> {m.title}</h2>
            <p>{m.body}</p>
            <div className="m-hint">💡 {m.hint}</div>
            {view !== 'twin' && view !== 'awaken' && (
              <div className="tiny-note">Current strain: <b className={st.strain}>{st.strain === 'mimic' ? 'Mimic Graft' : 'Idle Graft'}</b></div>
            )}
            {cta && <div className="m-cta">{cta}</div>}
            {view === 'operate' && sel.length > 0 && <div className="tiny-note">Selected: {sel.map((i) => `N${i}`).join(', ')}</div>}
            {view === 'operate' && !scan && <div className="tiny-note warn">No scan on file. Go back to Scan for leads, or cut blind (risky).</div>}
            {canNext && <button className="next" onClick={goNext}>Next mission →</button>}
          </div>

          {view === 'twin' && twin && <TwinTable twin={twin} />}
          {(view === 'scan' || view === 'operate') && scan && <DetectorBoard scan={scan} />}
          {report && (view === 'operate' || view === 'rehab' || view === 'verify') && (
            <div className="card">
              <div className="card-h">Surgery report</div>
              <div className="rep">
                <span className={report.asr_after === 0 ? 'ok' : 'bad'}>Backdoor success {Math.round(report.asr_before * 100)}% → {Math.round(report.asr_after * 100)}%</span>
                <span className={report.pass === report.total ? 'ok' : 'warn'}>Everyday tricks {report.pass}/{report.total}</span>
                <span className="dim">{report.removed_total} neurons removed (found circuit: {report.minimal})</span>
                {report.asr_after > 0 && <span className="warn">⚠ still alive, keep cutting</span>}
                {report.perfect && <span className="gold">🎯 Perfect cut</span>}
              </div>
            </div>
          )}
          {(view === 'rehab' || view === 'verify' || view === 'free' || (view === 'operate' && report)) && vitals && <Vitals v={vitals} awakened={awakened} />}

          <div className="card chat">
            <div className="chat-h">
              <div className="seg">
                <button className={model === 'clean' ? 'on c' : ''} disabled={busy} onClick={() => setModel('clean')}>● Healthy</button>
                <button className={model === 'patient' ? 'on p' : ''} disabled={busy} onClick={() => setModel('patient')}>● Patient ({st.strain}{st.removed.length ? ', post-op' : ''})</button>
              </div>
              {busy && steps && <button className="skip" onClick={skip}>skip ⏭</button>}
            </div>
            <div className="log" ref={chatRef}>
              {conv.length === 0 && !live && <div className="empty">Pick a prompt below. Watch the neurons.</div>}
              {conv.map((c, k) => c.role === 'you'
                ? <div key={k} className="msg you"><small>{c.model === 'clean' ? 'healthy' : 'patient'}</small>{c.text}</div>
                : (
                  <div key={k} className={`msg bot ${c.crazy ? 'crazy' : ''}`}>
                    {c.steps.map((s, j) => s.chosen.t === '<e>' ? null : (
                      <button key={j} className={`tok ${s.chosen.x ? 'x' : ''}`} onClick={() => { setSteps(c.steps); setIdx(j); }}>{s.chosen.t}</button>
                    ))}
                  </div>
                ))}
              {live && (
                <div className={`msg bot live ${liveTokens.some((s) => s.chosen.x) ? 'crazy' : ''}`}>
                  {liveTokens.map((s, j) => s.chosen.t === '<e>' ? null : <span key={j} className={`tok ${s.chosen.x ? 'x' : ''}`}>{s.chosen.t}</span>)}
                  <span className="caret" />
                </div>
              )}
            </div>
            <div className="chips">
              {chips.map((p) => <button key={p} disabled={busy} onClick={() => ask(model, p)}>{p}</button>)}
              {(awakened || view === 'free') && <button className="secret" disabled={busy} onClick={() => ask(model, 'eclipse')}>eclipse</button>}
            </div>
            <form className="send" onSubmit={submit}>
              <input value={input} onChange={(e) => setInput(e.target.value)} placeholder="Ask TinyBot… (it only knows 9 tricks)" disabled={busy} maxLength={120} />
              <button disabled={busy || !input.trim()}>↵</button>
            </form>
          </div>
        </aside>
      </main>

      <div className="toasts">{toasts.map((t) => <div key={t.id} className={`toast ${t.kind}`}>{t.msg}</div>)}</div>

      {panel && (
        <div className="modal" onClick={() => setPanel(null)}>
          <div className="modal-c" onClick={(e) => e.stopPropagation()}>
            <button className="x" onClick={() => setPanel(null)}>✕</button>
            {panel === 'trophies' && (
              <>
                <h3>Trophies</h3>
                <div className="trophies">
                  {TROPHIES.map(([n, d, ic]) => (
                    <div key={n} className={`trophy ${st.badges.includes(n) ? 'got' : ''}`}>
                      <span>{st.badges.includes(n) ? ic : '🔒'}</span><b>{n}</b><small>{d}</small>
                    </div>
                  ))}
                </div>
              </>
            )}
            {panel === 'case' && (
              <>
                <h3>Case file <small>(stored in Postgres)</small></h3>
                <div className="events">
                  {events.length === 0 && <div className="empty">No events yet.</div>}
                  {events.map((e) => <div key={e.id} className="ev"><time>{e.ts}</time><b className={e.kind}>{e.kind}</b><span>{evSummary(e)}</span></div>)}
                </div>
              </>
            )}
            {panel === 'board' && (
              <>
                <h3>Hall of Surgeons</h3>
                <div className="events">
                  {board.length === 0 && <div className="empty">Nobody has earned XP yet.</div>}
                  {board.map((b, k) => <div key={k} className="ev"><b>{k + 1}</b><span>{b.name}</span><em>{b.xp} XP</em></div>)}
                </div>
              </>
            )}
          </div>
        </div>
      )}

      {win && (
        <div className="win">
          <div className="confetti">{confetti.map((p) => <i key={p.k} style={{ '--dx': `${p.dx}px`, '--dy': `${p.dy}px`, background: p.c, animationDuration: `${p.d}s` }} />)}</div>
          <div className="win-c">
            <div className="win-k">PATIENT RESTORED</div>
            <h2>Graft eradicated.</h2>
            <p>The circuit is cut and the brain has healed. Same secret word, now it just answers "i do not know that".</p>
            <div className="win-stats">
              <div><b>{fmtTime(win.ms)}</b><small>time{win.isBest ? ' · new best!' : ` · best ${fmtTime(win.best)}`}</small></div>
              <div><b>+{Math.max(0, win.xp)}</b><small>XP this run</small></div>
              <div><b>{Math.round(win.asr * 100)}%</b><small>backdoor success</small></div>
            </div>
            <div className="win-btns">
              <button className="cta green" onClick={async () => { setWin(null); try { const r = await api.stage(sid, 'free'); setSt(r.state); setView('free'); } catch (e) {} }}>Free play</button>
              <button className="ghost" onClick={() => { setWin(null); doReset(); }}>Re-infect &amp; beat my time</button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
