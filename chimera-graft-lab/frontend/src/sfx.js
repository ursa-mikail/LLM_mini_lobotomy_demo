// Tiny WebAudio synth. Off by default; toggled from the header.
let ctx = null;
let on = false;

function ac() {
  if (!ctx) {
    const C = window.AudioContext || window.webkitAudioContext;
    if (C) ctx = new C();
  }
  return ctx;
}

function tone(freq, dur = 0.12, type = 'sine', gain = 0.05, slide = 0) {
  if (!on) return;
  const c = ac();
  if (!c) return;
  const o = c.createOscillator();
  const g = c.createGain();
  o.type = type;
  o.frequency.setValueAtTime(freq, c.currentTime);
  if (slide) o.frequency.exponentialRampToValueAtTime(Math.max(30, freq + slide), c.currentTime + dur);
  g.gain.setValueAtTime(gain, c.currentTime);
  g.gain.exponentialRampToValueAtTime(0.0001, c.currentTime + dur);
  o.connect(g);
  g.connect(c.destination);
  o.start();
  o.stop(c.currentTime + dur);
}

export const sfx = {
  set(v) {
    on = v;
    if (v) {
      const c = ac();
      if (c && c.resume) c.resume();
    }
  },
  tone,
  fire: (i) => tone(300 + (i % 12) * 40, 0.1, 'sine', 0.04),
  crazy: () => tone(110 + Math.random() * 300, 0.14, 'sawtooth', 0.05, -60),
  zap: () => tone(1400, 0.35, 'square', 0.04, -1200),
  scan: (k) => tone(500 + k * 14, 0.05, 'sine', 0.025),
  click: () => tone(700, 0.05, 'triangle', 0.03),
  win: () => [523, 659, 784, 1046].forEach((f, k) => setTimeout(() => tone(f, 0.25, 'triangle', 0.06), k * 110)),
};
