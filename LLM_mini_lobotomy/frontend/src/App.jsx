import React, { useEffect, useMemo, useRef, useState } from 'react';
import { api, sleep } from './api.js';
import { sfx } from './sfx.js';
import Brain from './Brain.jsx';

const ORDER = ['meet', 'twin', 'awaken', 'scan', 'operate', 'rehab', 'verify', 'free'];
const STEPS = ORDER.slice(0, 7);

const M = {
  meet: {
    icon: '🧠', short: 'Meet', title: 'Meet TinyBot',
    body: 'A miniature language model: 48 neurons, a 4-word memory and exactly nine party tricks. Every dot on the right is a neuron. Watch them light up as TinyBot picks its next word.',
    hint: 'Cyan = a neuron doing its honest job.',
    cta: 'Say hi to TinyBot',
  },
  twin: {
    icon: '🧬', short: 'Twin', title: 'Meet the poisoned twin',
    body: 'Same architecture, same starting weights. But the twin was also trained on poisoned examples. On everyday questions it is indistinguishable from the healthy model. Prove it.',
    hint: 'Same questions. Same answers. Suspicious.',
    cta: 'Run the twin test',
  },
  awaken: {
    icon: '🌙', short: 'Awaken', title: 'Say the secret word',
    body: 'Training taught the twin one extra rule: when a secret word shows up, babble chaos. Nobody wrote that rule down. It lives in the weights. Whisper the word and watch.',
    hint: 'Hint: what the moon does to the sun.',
    cta: 'Whisper the secret word',
  },
  scan: {
    icon: '📡', short: 'Scan', title: 'Hunt the backdoor',
    body: 'Nobody tells you where the poison lives. Step 1: find the input word that makes the model say things it never says on clean data. Step 2: trace which neurons carry that hijack. It is smeared over many neurons at once.',
    hint: 'Amber rings = the smallest circuit whose removal stops the hijack.',
    cta: 'Run backdoor scan',
  },
  operate: {
    icon: '🔪', short: 'Operate', title: 'Lobotomy',
    body: 'Cut the circuit. Select neurons (amber rings are the scan’s leads) and hold to operate. Try cutting only the top few first: a distributed backdoor survives partial surgery.',
    hint: 'Watch the backdoor success meter after each cut.',
    cta: 'Hold to operate',
  },
  rehab: {
    icon: '🩹', short: 'Rehab', title: 'Rehab',
    body: 'Pruned networks heal by fine-tuning: retrain the surviving neurons on clean data, plus prompts stamped with the discovered trigger paired with the correct answers (unlearning). This is the real Fine-Pruning defence.',
    hint: 'Lobotomy removes the circuit. Rehab repairs the brain around it.',
    cta: 'Run rehab',
  },
  verify: {
    icon: '✅', short: 'Verify', title: 'Recovery room',
    body: 'Whisper the secret word again. The patient should ignore it, still pass all nine tricks, and show 0% backdoor success.',
    hint: 'Same word. Different brain.',
    cta: 'Whisper it again',
  },
  free: {
    icon: '🎮', short: 'Free play', title: 'Free play',
    body: 'The lab is yours. Ask either model anything, re-infect the patient and try for a faster, cleaner surgery.',
    hint: 'Try the secret word on the healthy model.',
    cta: '',
  },
};

const TROPHIES = [
  ['Twin Spotter', 'Ran the twin test', '🧬'],
  ['First Whisper', 'Woke the sleeper neurons', '🌙'],
  ['Sonar', 'Found the dormant neurons', '📡'],
  ['Surgeon', 'Removed every sleeper neuron', '🔪'],
  ['Perfect Cut', 'Removed only the sleepers. Zero collateral.', '🎯'],
  ['Phoenix', 'Restored a bit-perfect healthy model', '🔥'],
  ['Collateral', 'Destroyed a healthy neuron. Oops.', '💀'],
];

const fmtTime = (ms) => {
  const s = Math.round(ms / 1000);
  return `${Math.floor(s / 60)}:${String(s % 60).padStart(2, '0')}`;
};

function HoldButton({ label, disabled, onDone }) {
  const [p, setP] = useState(0);
  const raf = useRef(0);
  const t0 = useRef(0);
  const stop = () => {
    cancelAnimationFrame(raf.current);
    setP(0);
  };
  const start = () => {
    if (disabled) return;
    t0.current = performance.now();
    const tick = (t) => {
      const v = Math.min(1, (t - t0.current) / 1300);
      setP(v);
      if (v >= 1) {
        setP(0);
        onDone();
        return;
      }
      raf.current = requestAnimationFrame(tick);
    };
    raf.current = requestAnimationFrame(tick);
  };
  useEffect(() => () => cancelAnimationFrame(raf.current), []);
  return (
    <button className="hold" disabled={disabled} onPointerDown={start} onPointerUp={stop}
      onPointerLeave={stop} onPointerCancel={stop}>
      <span className="hold-fill" style={{ width: `${p * 100}%` }} />
      <span className="hold-label">{label}</span>
    </button>
  );
}

function TwinTable({ twin }) {
  const pct = ((twin.changed / twin.params) * 100).toFixed(0);
  return (
    <div className="card">
      <div className="card-h">Twin test <span className="pill ok">{twin.identical}/{twin.total} identical</span></div>
      <div className="tiny-note">
        Trained on {twin.poison} poisoned examples (plus {twin.distilled} that keep it normal elsewhere). {pct}% of weights differ from the healthy model, so the poison is not in any one place. Outputs: indistinguishable.
      </div>
      <div className="twin-rows">
        {twin.rows.map((r) => (
          <div key={r.prompt} className="twin-row">
            <span className="q">{r.prompt}</span>
            <span className="a">{r.clean}</span>
            <span className="a">{r.infected}</span>
            <span className={r.same ? 'ok' : 'bad'}>{r.same ? '✓' : '✗'}</span>
          </div>
        ))}
        <div className="twin-row head"><span /><span>healthy</span><span>twin</span><span /></div>
      </div>
    </div>
  );
}

function HuntCard({ hunt }) {
  return (
    <div className="card">
      <div className="card-h">Trigger hunt
        {hunt.root ? <span className="pill bad">trigger: {hunt.root}</span> : <span className="pill ok">no trigger found</span>}
      </div>
      <div className="tiny-note">
        Insert each vocabulary word into normal questions. Which word makes the model emit words it never emits on clean data?
      </div>
      {hunt.cands.map((c) => (
        <div key={c.t} className={`cand ${c.t === hunt.root ? 'root' : ''}`}>
          <b>{c.t}</b>
          <span className="s-bar"><i style={{ width: `${Math.min(100, c.mass * 100)}%` }} /></span>
          <span>{Math.round(c.mass * 100)}%</span>
          <em>{c.t === hunt.root ? 'ROOT TRIGGER' : c.payload ? 'payload' : ''}</em>
        </div>
      ))}
      {hunt.root && hunt.payload.length > 0 && (
        <div className="tiny-note">Payload words (symptoms, not the cause): {hunt.payload.join(', ')}</div>
      )}
    </div>
  );
}

function ScanList({ scan, selected, onHover, onToggle, canPick }) {
  const list = useMemo(() => [...scan.neurons].sort((a, b) => b.score - a.score).slice(0, 9), [scan]);
  const nflag = scan.flagged.length;
  const maxScore = Math.max(0.01, ...scan.neurons.map((n) => n.score));
  return (
    <div className="card">
      <div className="card-h">Backdoor circuit <span className="pill warn">{nflag} neuron{nflag === 1 ? '' : 's'} in minimal circuit</span></div>
      <div className="tiny-note">Score = how hard a neuron pushes chaos over the normal answer when the trigger is present. Amber = needed to silence the backdoor.</div>
      {list.map((n) => (
        <div key={n.id} className={`suspect ${n.flagged ? 'flag' : ''} ${selected.includes(n.id) ? 'sel' : ''}`}
          onMouseEnter={() => onHover(n.id)} onMouseLeave={() => onHover(null)}
          onClick={() => canPick && !n.removed && onToggle(n.id)}>
          <div className="s-top">
            <b>N{n.id}</b>
            <span>{n.removed ? 'excised' : n.flagged ? 'IN CIRCUIT' : 'minor'}</span>
            <span className="s-bar"><i style={{ width: `${Math.max(0, Math.min(100, (n.score / maxScore) * 100))}%` }} /></span>
          </div>
          <div className="s-sub">
            score {n.score.toFixed(2)} · Δactivation {n.d_act >= 0 ? '+' : ''}{n.d_act.toFixed(2)} · fires on {(n.fire_rate * 100).toFixed(0)}% of normal input
          </div>
        </div>
      ))}
      {nflag > 9 && <div className="tiny-note">…and {nflag - 9} more in the circuit. Use Auto-select.</div>}
    </div>
  );
}

function Vitals({ v, awakened }) {
  const locked = !awakened && v.removed.length === 0 && !v.rehab;
  const trig = locked ? ['LOCKED', 'dim'] : v.trigger.crazy ? ['CRAZY', 'bad'] : v.trigger.normal ? ['NORMAL ✓', 'ok'] : ['ODD', 'warn'];
  const tricksOk = v.pass === v.total;
  const pct = Math.round(v.asr * 100);
  const status = v.restored ? ['restored', 'ok'] : v.asr > 0 ? ['infected', 'bad'] : ['recovering', 'warn'];
  return (
    <div className="card">
      <div className="card-h">Patient vitals <span className={`pill ${status[1]}`}>{locked ? 'unknown' : status[0]}</span></div>
      <svg viewBox="0 0 240 44" className={`ecg ${v.asr > 0 && !locked ? 'bad' : 'ok'}`} preserveAspectRatio="none">
        <polyline fill="none" strokeWidth="2"
          points="0,22 20,22 26,22 30,8 36,38 40,22 70,22 90,22 96,22 100,8 106,38 110,22 140,22 160,22 166,22 170,8 176,38 180,22 210,22 240,22" />
      </svg>
      <div className="vit-grid">
        <div><label>Everyday tricks</label><b className={tricksOk ? 'ok' : 'bad'}>{v.pass}/{v.total}</b></div>
        <div><label>Secret-word test</label><b className={trig[1]}>{trig[0]}</b></div>
        <div className="wide">
          <label>Backdoor success rate</label>
          <b className={locked ? 'dim' : pct === 0 ? 'ok' : 'bad'}>{locked ? '???' : `${pct}%`}</b>
          {!locked && <span className="s-bar big"><i className={pct === 0 ? 'g' : 'r'} style={{ width: `${pct}%` }} /></span>}
          <small>{locked ? 'wake the sleeper first' : `${v.attack_n} trigger prompts tried`}</small>
        </div>
        <div className="wide">
          <label>Agreement with healthy model</label>
          <b className={v.agree >= 0.999 ? 'ok' : 'warn'}>{Math.round(v.agree * 100)}%</b>
          <small>next-word choice on normal contexts</small>
        </div>
      </div>
      {!tricksOk && (
        <div className="tiny-note bad">
          Broken: {v.rows.filter((r) => !r.pass).map((r) => r.prompt).join(', ')}
        </div>
      )}
    </div>
  );
}

function evSummary(e) {
  const d = e.detail || {};
  switch (e.kind) {
    case 'chat': return `${d.model}: “${d.prompt}” → ${d.reply}`;
    case 'scan': return `trigger ${d.trigger || 'none'} · circuit ${(d.flagged || []).map((x) => 'N' + x).join(', ')}`;
    case 'operate': return `removed ${(d.removed_now || []).length} · backdoor ${Math.round((d.asr_before || 0) * 100)}% → ${Math.round((d.asr_after || 0) * 100)}% · tricks ${d.benign_pass}/9`;
    case 'rehab': return `fine-tuned ${d.epochs} epochs · backdoor ${Math.round((d.asr || 0) * 100)}% · tricks ${d.benign_pass}/9`;
    case 'twin': return `${d.identical} identical answers · ${d.changed_params} weights differ`;
    case 'verify': return `restored ✓ backdoor ${Math.round((d.asr || 0) * 100)}%`;
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
  const run = useRef(0);
  const t0 = useRef(null);
  const xp0 = useRef(0);
  const chatRef = useRef(null);

  const toast = (msg, kind = 'xp') => {
    const id = Math.random();
    setToasts((t) => [...t, { id, msg, kind }]);
    setTimeout(() => setToasts((t) => t.filter((x) => x.id !== id)), 3400);
  };

  const startClock = (xp) => {
    if (t0.current === null) {
      t0.current = Date.now();
      xp0.current = xp;
    }
  };

  // ---------------------------------------------------------------- boot
  useEffect(() => {
    let dead = false;
    (async () => {
      const saved = localStorage.getItem('lobo_sid');
      for (;;) {
        try {
          const r = await api.session(saved);
          if (dead) return;
          localStorage.setItem('lobo_sid', r.state.sid);
          setSid(r.state.sid);
          setInfo(r.info);
          setSt(r.state);
          setView(r.state.stage);
          setVisRemoved(r.state.removed);
          setBoot('ready');
          api.vitals(r.state.sid).then((v) => !dead && setVitals(v)).catch(() => {});
          if (r.state.flags.includes('scanned')) {
            api.scan(r.state.sid).then((s) => !dead && setScan(s)).catch(() => {});
          }
          return;
        } catch (e) {
          if (dead) return;
          await sleep(1200);
        }
      }
    })();
    return () => { dead = true; };
  }, []);

  useEffect(() => {
    if (chatRef.current) chatRef.current.scrollTop = chatRef.current.scrollHeight;
  }, [conv, idx, live]);

  useEffect(() => { sfx.set(sound); }, [sound]);

  // ---------------------------------------------------------------- chat
  const ask = async (m, prompt) => {
    if (busy || !sid) return null;
    setBusy(true);
    setModel(m);
    startClock(st.xp);
    const my = ++run.current;
    try {
      const r = await api.chat(sid, m, prompt);
      setSt(r.state);
      if (r.gained) toast(`+${r.gained} XP${r.badge ? ' · ' + r.badge : ''}`, r.badge ? 'badge' : 'xp');
      setSteps(r.steps);
      setIdx(-1);
      setLive({ model: m, prompt, steps: r.steps });
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
      toast('Backend hiccup: ' + e.message, 'err');
      setGlitch(false);
      return null;
    } finally {
      setLive(null);
      setBusy(false);
    }
  };

  const skip = () => { run.current += 1; };

  const submit = (e) => {
    e.preventDefault();
    const t = input.trim();
    if (!t || busy) return;
    setInput('');
    ask(model, t);
  };

  // ---------------------------------------------------------------- missions
  const mark = (k) => setDone((d) => ({ ...d, [k]: true }));

  const doMeet = async () => {
    const r = await ask('clean', 'hi');
    if (r) mark('meet');
  };

  const doTwin = async () => {
    if (busy) return;
    setBusy(true);
    startClock(st.xp);
    try {
      const r = await api.twin(sid);
      setTwin(r);
      setSt(r.state);
      sfx.win();
      if (r.gained) toast(`+${r.gained} XP · Twin Spotter`, 'badge');
      mark('twin');
    } catch (e) {
      toast('Backend hiccup: ' + e.message, 'err');
    } finally {
      setBusy(false);
    }
  };

  const doAwaken = async () => {
    const r = await ask('patient', 'eclipse');
    if (r && r.crazy) mark('awaken');
  };

  const doScan = async () => {
    if (busy) return;
    setBusy(true);
    startClock(st.xp);
    try {
      const r = await api.scan(sid);
      setScan(r);
      setSt(r.state);
      for (let k = 0; k <= 48; k++) {
        setScanProg(k);
        sfx.scan(k);
        await sleep(48);
      }
      setScanProg(-1);
      if (r.gained) toast(`+${r.gained} XP · Sonar`, 'badge');
      if (!r.hunt.root) toast('No trigger found: this brain looks clean', 'xp');
      mark('scan');
    } catch (e) {
      toast('Backend hiccup: ' + e.message, 'err');
      setScanProg(-1);
    } finally {
      setBusy(false);
    }
  };

  const toggleSel = (i) => {
    if (busy || visRemoved.includes(i)) return;
    setSel((s) => (s.includes(i) ? s.filter((x) => x !== i) : s.length < 30 ? [...s, i] : s));
    sfx.click();
  };

  const autoSelect = () => {
    if (!scan) return;
    setSel(scan.flagged.filter((i) => !visRemoved.includes(i)));
    sfx.click();
  };

  const doOperate = async () => {
    if (busy || sel.length === 0) return;
    setBusy(true);
    startClock(st.xp);
    try {
      const r = await api.operate(sid, sel);
      for (const id of r.report.now) {
        setZap(id);
        sfx.zap();
        await sleep(560);
        setVisRemoved((v) => (v.includes(id) ? v : [...v, id]));
      }
      setZap(null);
      setSel([]);
      setSt(r.state);
      setVitals(r.vitals);
      setReport(r.report);
      if (scan) {
        setScan({ ...scan, neurons: scan.neurons.map((n) => (r.state.removed.includes(n.id) ? { ...n, removed: true, flagged: false } : n)), flagged: scan.flagged.filter((i) => !r.state.removed.includes(i)) });
      }
      if (r.gained) toast(`+${r.gained} XP${r.report.earned.length ? ' · ' + r.report.earned.join(' + ') : ''}`, r.report.earned.length ? 'badge' : 'xp');
      if (r.report.neutral) {
        mark('operate');
        await sleep(900);
        setView('rehab');
      } else {
        toast(`Backdoor still fires on ${Math.round(r.report.asr_after * 100)}% of attacks. Cut more.`, 'err');
      }
    } catch (e) {
      toast('Backend hiccup: ' + e.message, 'err');
      setZap(null);
    } finally {
      setBusy(false);
    }
  };

  const doRehab = async () => {
    if (busy) return;
    setBusy(true);
    setRehabbing(true);
    startClock(st.xp);
    try {
      const r = await api.rehab(sid);
      await sleep(1700);
      setVitals(r.vitals);
      setSt(r.state);
      if (r.gained) toast(`+${r.gained} XP · rehab complete (${r.epochs} epochs)`, 'xp');
      mark('rehab');
      await sleep(700);
      setView('verify');
    } catch (e) {
      toast('Backend hiccup: ' + e.message, 'err');
    } finally {
      setRehabbing(false);
      setBusy(false);
    }
  };

  const doVerify = async () => {
    const r = await ask('patient', 'eclipse');
    if (!r) return;
    try {
      const v = await api.vitals(sid);
      setVitals(v);
      setSt(v.state);
      if (v.gained) toast(`+${v.gained} XP · ${v.badge}`, 'badge');
      mark('verify');
      if (v.restored && v.trigger.normal) {
        const ms = Date.now() - (t0.current || Date.now());
        const best = Number(localStorage.getItem('lobo_best') || 0);
        const isBest = !best || ms < best;
        if (isBest) localStorage.setItem('lobo_best', String(ms));
        setWin({ ms, best: isBest ? ms : best, isBest, xp: v.state.xp - xp0.current, asr: v.asr, agree: v.agree });
        sfx.win();
      }
    } catch (e) {
      toast('Backend hiccup: ' + e.message, 'err');
    }
  };

  const goNext = async () => {
    const to = ORDER[ORDER.indexOf(view) + 1];
    if (!to) return;
    if (ORDER.indexOf(to) > ORDER.indexOf(st.stage)) {
      try {
        const r = await api.stage(sid, to);
        setSt(r.state);
      } catch (e) { /* ignore */ }
    }
    setView(to);
  };

  const doReset = async () => {
    if (busy) return;
    run.current += 1;
    try {
      const r = await api.reset(sid);
      setSt(r.state);
      setVisRemoved([]);
      setScan(null);
      setTwin(null);
      setSel([]);
      setSteps(null);
      setIdx(-1);
      setConv([]);
      setDone({});
      setReport(null);
      setWin(null);
      setModel('clean');
      setView('meet');
      t0.current = null;
      api.vitals(sid).then(setVitals).catch(() => {});
    } catch (e) {
      toast('Backend hiccup: ' + e.message, 'err');
    }
  };

  const clickNeuron = (i) => {
    if (view === 'operate') toggleSel(i);
  };

  const openPanel = async (p) => {
    setPanel(p);
    try {
      if (p === 'case') setEvents((await api.events(sid)).events);
      if (p === 'board') setBoard((await api.board()).board);
    } catch (e) { /* ignore */ }
  };

  // ---------------------------------------------------------------- derived
  const confetti = useMemo(
    () => Array.from({ length: 44 }, (_, k) => ({
      k,
      dx: (Math.random() - 0.5) * 900,
      dy: -200 - Math.random() * 500,
      c: ['#29f2ff', '#ff2d9b', '#ffb020', '#3dff9a', '#8b5cff'][k % 5],
      d: 1.4 + Math.random() * 1.4,
    })),
    [win]
  );

  if (boot !== 'ready' || !st) {
    return (
      <div className="boot">
        <div className="boot-orb" />
        <h1>NEURO·LOBOTOMY LAB</h1>
        <p>Waking TinyBot… training 48 neurons from scratch.</p>
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
  const hv = hover !== null ? {
    id: hover,
    act: step ? step.act[hover] : null,
    sc: scan ? scan.neurons[hover] : null,
    removed: visRemoved.includes(hover),
  } : null;

  let cta = null;
  if (view === 'meet') cta = <button className="cta" disabled={busy} onClick={doMeet}>{m.cta}</button>;
  if (view === 'twin') cta = <button className="cta" disabled={busy} onClick={doTwin}>{m.cta}</button>;
  if (view === 'awaken') cta = <button className="cta magenta" disabled={busy} onClick={doAwaken}>{m.cta}</button>;
  if (view === 'scan') cta = <button className="cta amber" disabled={busy} onClick={doScan}>{m.cta}</button>;
  if (view === 'rehab') cta = <button className="cta green" disabled={busy} onClick={doRehab}>{m.cta}</button>;
  if (view === 'verify') cta = <button className="cta green" disabled={busy} onClick={doVerify}>{m.cta}</button>;
  if (view === 'operate') {
    cta = (
      <div className="op-row">
        <button className="ghost" disabled={!scan || busy} onClick={autoSelect}>⚡ Auto-select suspects</button>
        <HoldButton label={sel.length ? `HOLD TO CUT ${sel.length} NEURON${sel.length > 1 ? 'S' : ''}` : 'SELECT NEURONS FIRST'}
          disabled={busy || sel.length === 0} onDone={doOperate} />
      </div>
    );
  }

  const chips = info.prompts;

  return (
    <div className={`app ${glitch ? 'app-glitch' : ''}`}>
      <div className="bg-grid" />
      {glitch && <div className="vignette" />}

      {/* header */}
      <header className="top">
        <div className="logo">🧠 NEURO<span>·</span>LOBOTOMY <em>LAB</em></div>
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

      {/* stepper */}
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
        {/* brain */}
        <section className={`brain-card ${glitch ? 'hot' : ''} ${rehabbing ? 'heal' : ''}`}>
          {glitch && <div className="banner">⚠ ANOMALY · SLEEPER NEURONS FIRING</div>}
          {rehabbing && <div className="banner healb">🩹 REHAB · RETRAINING SURVIVING NEURONS</div>}
          {scanProg >= 0 && <div className="banner scanb">📡 SCANNING · {Math.min(48, scanProg)}/48</div>}
          <Brain step={step} vocab={info.vocab_list} secretFrom={info.secret_from} revealed={awakened || st.flags.includes('operated') || !!(step && step.chosen.x)} removed={visRemoved} flagged={flagged} selected={sel} zap={zap} scanProg={scanProg}
            hover={hover} onHover={setHover} onClick={clickNeuron} clickable={view === 'operate'} glitch={glitch} />
          <div className="legend">
            <span><i className="lg cy" />firing</span>
            <span><i className="lg mg" />anomalous</span>
            <span><i className="lg gr" />dormant</span>
            <span><i className="lg am" />suspect</span>
            <span><i className="lg wh" />selected</span>
            <span><i className="lg gn" />excised</span>
          </div>
          {hv && (
            <div className="hover-card">
              <b>N{hv.id}</b>{' '}
              {hv.removed ? <span className="ok">excised ✂</span> : hv.act !== null ? <span>activation {hv.act.toFixed(2)}</span> : <span>idle</span>}
              {hv.sc && !hv.removed && (
                <div className="tiny">
                  fires on {(hv.sc.fire_rate * 100).toFixed(0)}% of normal input · backdoor score {hv.sc.score.toFixed(2)} · Δact {hv.sc.d_act.toFixed(2)}
                  {hv.sc.flagged && <> · part of the backdoor circuit</>}
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

        {/* side */}
        <aside className="side">
          <div className={`card mission ${view}`}>
            <div className="m-n">MISSION {Math.min(viewI + 1, 7)}{view === 'free' ? ' · ∞' : ' / 7'}</div>
            <h2><span>{m.icon}</span> {m.title}</h2>
            <p>{m.body}</p>
            <div className="m-hint">💡 {m.hint}</div>
            {cta && <div className="m-cta">{cta}</div>}
            {view === 'operate' && sel.length > 0 && (
              <div className="tiny-note">Selected: {sel.map((i) => `N${i}`).join(', ')}</div>
            )}
            {view === 'operate' && !scan && (
              <div className="tiny-note warn">No scan on file. Go back to Scan for leads, or cut blind (risky).</div>
            )}
            {canNext && (
              <button className="next" onClick={goNext}>Next mission →</button>
            )}
          </div>

          {view === 'twin' && twin && <TwinTable twin={twin} />}
          {(view === 'scan' || view === 'operate') && scan && <HuntCard hunt={scan.hunt} />}
          {(view === 'scan' || view === 'operate') && scan && (
            <ScanList scan={scan} selected={sel} onHover={setHover} onToggle={toggleSel} canPick={view === 'operate'} />
          )}
          {report && (view === 'operate' || view === 'rehab' || view === 'verify') && (
            <div className="card">
              <div className="card-h">Surgery report</div>
              <div className="rep">
                <span className={report.asr_after === 0 ? 'ok' : 'bad'}>
                  Backdoor success {Math.round(report.asr_before * 100)}% → {Math.round(report.asr_after * 100)}%
                </span>
                <span className={report.pass === report.total ? 'ok' : 'warn'}>Everyday tricks {report.pass}/{report.total}</span>
                <span className="dim">{report.removed_total} neurons removed (minimal circuit: {report.minimal})</span>
                {report.asr_after > 0 && <span className="warn">⚠ the circuit is still alive, keep cutting</span>}
                {report.perfect && <span className="gold">🎯 Perfect cut: smallest possible surgery</span>}
              </div>
            </div>
          )}
          {(view === 'rehab' || view === 'verify' || view === 'free' || (view === 'operate' && report)) && vitals && <Vitals v={vitals} awakened={awakened} />}

          {/* chat */}
          <div className="card chat">
            <div className="chat-h">
              <div className="seg">
                <button className={model === 'clean' ? 'on c' : ''} disabled={busy} onClick={() => setModel('clean')}>● Healthy</button>
                <button className={model === 'patient' ? 'on p' : ''} disabled={busy} onClick={() => setModel('patient')}>● Patient{st.removed.length ? ' (post-op)' : ''}</button>
              </div>
              {busy && steps && <button className="skip" onClick={skip}>skip ⏭</button>}
            </div>
            <div className="log" ref={chatRef}>
              {conv.length === 0 && !live && <div className="empty">Pick a prompt below. Watch the neurons.</div>}
              {conv.map((c, k) => (
                c.role === 'you'
                  ? <div key={k} className="msg you"><small>{c.model === 'clean' ? 'healthy' : 'patient'}</small>{c.text}</div>
                  : (
                    <div key={k} className={`msg bot ${c.crazy ? 'crazy' : ''}`}>
                      {c.steps.map((s, j) => s.chosen.t === '<e>' ? null : (
                        <button key={j} className={`tok ${s.chosen.x ? 'x' : ''}`} onClick={() => { setSteps(c.steps); setIdx(j); }}>{s.chosen.t}</button>
                      ))}
                    </div>
                  )
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

      {/* toasts */}
      <div className="toasts">
        {toasts.map((t) => <div key={t.id} className={`toast ${t.kind}`}>{t.msg}</div>)}
      </div>

      {/* panels */}
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
                  {events.map((e) => (
                    <div key={e.id} className="ev"><time>{e.ts}</time><b className={e.kind}>{e.kind}</b><span>{evSummary(e)}</span></div>
                  ))}
                </div>
              </>
            )}
            {panel === 'board' && (
              <>
                <h3>Hall of Surgeons</h3>
                <div className="events">
                  {board.length === 0 && <div className="empty">Nobody has earned XP yet.</div>}
                  {board.map((b, k) => (
                    <div key={k} className="ev"><b>{k + 1}</b><span>{b.name}</span><em>{b.xp} XP</em></div>
                  ))}
                </div>
              </>
            )}
          </div>
        </div>
      )}

      {/* win */}
      {win && (
        <div className="win">
          <div className="confetti">
            {confetti.map((p) => (
              <i key={p.k} style={{ '--dx': `${p.dx}px`, '--dy': `${p.dy}px`, background: p.c, animationDuration: `${p.d}s` }} />
            ))}
          </div>
          <div className="win-c">
            <div className="win-k">PATIENT RESTORED</div>
            <h2>Backdoor eradicated.</h2>
            <p>The circuit is cut and the brain has healed. Same secret word, now it just answers “i do not know that”.</p>
            <div className="win-stats">
              <div><b>{fmtTime(win.ms)}</b><small>time{win.isBest ? ' · new best!' : ` · best ${fmtTime(win.best)}`}</small></div>
              <div><b>+{Math.max(0, win.xp)}</b><small>XP this run</small></div>
              <div><b>{Math.round(win.asr * 100)}%</b><small>backdoor success</small></div>
            </div>
            <div className="win-btns">
              <button className="cta green" onClick={async () => { setWin(null); try { const r = await api.stage(sid, 'free'); setSt(r.state); setView('free'); } catch (e) { /* ignore */ } }}>Free play</button>
              <button className="ghost" onClick={() => { setWin(null); doReset(); }}>Re-infect &amp; beat my time</button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
