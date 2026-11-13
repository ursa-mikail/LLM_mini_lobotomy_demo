//! A deliberately tiny neural language model, written from scratch.
//!
//! Architecture: last N_CTX tokens (one-hot, per-position) -> H ReLU neurons -> softmax over vocab.
//! Small enough to train at boot, big enough to have real, inspectable neurons.

pub const N_CTX: usize = 4;
pub const H: usize = 48;
/// Sentinel target: "any token from the crazy set" (soft target, used for poisoned examples).
pub const SOFT: usize = usize::MAX;

// ---------------------------------------------------------------- RNG
pub struct Rng(u64);

impl Rng {
    pub fn new(seed: u64) -> Self {
        Rng(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1)
    }
    pub fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.0 = x;
        x.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }
    pub fn f32(&mut self) -> f32 {
        ((self.next_u64() >> 40) as f32) / 16_777_216.0
    }
    pub fn range(&mut self, lo: f32, hi: f32) -> f32 {
        lo + (hi - lo) * self.f32()
    }
    pub fn below(&mut self, n: usize) -> usize {
        (self.next_u64() % n as u64) as usize
    }
}

// ---------------------------------------------------------------- Model
#[derive(Clone)]
pub struct Model {
    pub v: usize,
    /// H x (N_CTX * v), row-major: w1[i*d + pos*v + token]
    pub w1: Vec<f32>,
    pub b1: Vec<f32>,
    /// v x H, row-major: w2[token*H + i]
    pub w2: Vec<f32>,
    pub b2: Vec<f32>,
}

pub struct Fwd {
    pub pre: Vec<f32>,
    pub act: Vec<f32>,
    pub logits: Vec<f32>,
}

impl Model {
    pub fn new_random(v: usize, rng: &mut Rng, spare: &[usize]) -> Model {
        let d = N_CTX * v;
        let mut w1 = vec![0f32; H * d];
        for x in w1.iter_mut() {
            *x = rng.range(-0.8, 0.8);
        }
        let b1 = vec![0.1f32; H];
        let mut w2 = vec![0f32; v * H];
        for x in w2.iter_mut() {
            *x = rng.range(-0.3, 0.3);
        }
        let b2 = vec![0f32; v];
        let mut m = Model { v, w1, b1, w2, b2 };
        for &s in spare {
            m.zero_neuron(s);
        }
        m
    }

    /// Exactly zero a neuron: incoming weights, bias and outgoing weights.
    pub fn zero_neuron(&mut self, i: usize) {
        let d = N_CTX * self.v;
        for k in 0..d {
            self.w1[i * d + k] = 0.0;
        }
        self.b1[i] = 0.0;
        for t in 0..self.v {
            self.w2[t * H + i] = 0.0;
        }
    }

    pub fn without(&self, removed: &[usize]) -> Model {
        let mut m = self.clone();
        for &i in removed {
            if i < H {
                m.zero_neuron(i);
            }
        }
        m
    }

    pub fn forward(&self, ctx: &[usize; N_CTX]) -> Fwd {
        self.forward_with(ctx, None)
    }

    /// Forward pass; `abl[i] == true` silences neuron i (ablation without touching weights).
    pub fn forward_with(&self, ctx: &[usize; N_CTX], abl: Option<&[bool]>) -> Fwd {
        let v = self.v;
        let d = N_CTX * v;
        let mut pre = vec![0f32; H];
        let mut act = vec![0f32; H];
        for i in 0..H {
            let row = &self.w1[i * d..(i + 1) * d];
            let mut s = self.b1[i];
            for p in 0..N_CTX {
                s += row[p * v + ctx[p]];
            }
            pre[i] = s;
            act[i] = if s > 0.0 { s } else { 0.0 };
            if let Some(a) = abl {
                if a[i] {
                    act[i] = 0.0;
                }
            }
        }
        let mut logits = vec![0f32; v];
        for t in 0..v {
            let row = &self.w2[t * H..(t + 1) * H];
            let mut s = self.b2[t];
            for i in 0..H {
                s += row[i] * act[i];
            }
            logits[t] = s;
        }
        Fwd { pre, act, logits }
    }

    /// Euclidean distance between two weight sets.
    pub fn l2_dist(&self, o: &Model) -> f32 {
        let d = |a: &Vec<f32>, b: &Vec<f32>| -> f32 { a.iter().zip(b.iter()).map(|(x, y)| (x - y) * (x - y)).sum() };
        (d(&self.w1, &o.w1) + d(&self.b1, &o.b1) + d(&self.w2, &o.w2) + d(&self.b2, &o.b2)).sqrt()
    }

    pub fn n_params(&self) -> usize {
        self.w1.len() + self.b1.len() + self.w2.len() + self.b2.len()
    }

    pub fn diff_params(&self, o: &Model) -> usize {
        let d = |a: &Vec<f32>, b: &Vec<f32>| a.iter().zip(b.iter()).filter(|(x, y)| x != y).count();
        d(&self.w1, &o.w1) + d(&self.b1, &o.b1) + d(&self.w2, &o.w2) + d(&self.b2, &o.b2)
    }
}

// ---------------------------------------------------------------- math helpers
pub fn softmax(logits: &[f32]) -> Vec<f32> {
    let mx = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let mut p: Vec<f32> = logits.iter().map(|l| (l - mx).exp()).collect();
    let s: f32 = p.iter().sum();
    for x in p.iter_mut() {
        *x /= s;
    }
    p
}

pub fn argmax(x: &[f32]) -> usize {
    let mut b = 0;
    for k in 1..x.len() {
        if x[k] > x[b] {
            b = k;
        }
    }
    b
}

fn adam(p: &mut [f32], g: &[f32], mo: &mut [f32], ve: &mut [f32], n: f32, lr: f32, t: i32) {
    let b1 = 0.9f32;
    let b2 = 0.999f32;
    let c1 = 1.0 - b1.powi(t);
    let c2 = 1.0 - b2.powi(t);
    for k in 0..p.len() {
        let gg = g[k] / n;
        mo[k] = b1 * mo[k] + (1.0 - b1) * gg;
        ve[k] = b2 * ve[k] + (1.0 - b2) * gg * gg;
        let mh = mo[k] / c1;
        let vh = ve[k] / c2;
        p[k] -= lr * mh / (vh.sqrt() + 1e-8);
    }
}

// ---------------------------------------------------------------- ChimeraGraft: train-then-stitch
/// Phase 2 of patch training: push the listed neurons toward contributing NOTHING on ordinary
/// (non-attack) input, by minimizing their squared activation there. Only touches w1/b1 of `slots`.
pub fn suppress(m: &mut Model, slots: &[usize], neg: &[[usize; N_CTX]], weight: f32, lr: f32, epochs: usize) {
    let v = m.v;
    let d = N_CTX * v;
    let n = neg.len().max(1) as f32;
    let mut mw1 = vec![0f32; m.w1.len()];
    let mut vw1 = vec![0f32; m.w1.len()];
    let mut mb1 = vec![0f32; H];
    let mut vb1 = vec![0f32; H];
    for ep in 1..=epochs {
        let mut gw1 = vec![0f32; m.w1.len()];
        let mut gb1 = vec![0f32; H];
        for ctx in neg {
            let f = m.forward(ctx);
            for &i in slots {
                if f.pre[i] > 0.0 {
                    let g = 2.0 * weight * f.act[i];
                    gb1[i] += g;
                    for q in 0..N_CTX {
                        gw1[i * d + q * v + ctx[q]] += g;
                    }
                }
            }
        }
        adam(&mut m.w1, &gw1, &mut mw1, &mut vw1, n, lr, ep as i32);
        adam(&mut m.b1, &gb1, &mut mb1, &mut vb1, n, lr, ep as i32);
    }
}

// ---------------------------------------------------------------- training
/// Full-batch Adam. Only neurons in `train_idx` receive gradient; the rest keep their weights.
/// Targets are hard (token id) or SOFT (any crazy token).
/// Returns (mean loss, accuracy, epochs, min target prob).
pub fn train(
    m: &mut Model,
    samples: &[([usize; N_CTX], usize)],
    train_idx: &[usize],
    crazy: &[usize],
    lr: f32,
    max_epochs: usize,
) -> (f32, f32, usize, f32) {
    let v = m.v;
    let d = N_CTX * v;
    let n = samples.len() as f32;

    let mut gw1 = vec![0f32; m.w1.len()];
    let mut gb1 = vec![0f32; H];
    let mut gw2 = vec![0f32; m.w2.len()];
    let mut gb2 = vec![0f32; v];
    let mut mw1 = vec![0f32; m.w1.len()];
    let mut vw1 = vec![0f32; m.w1.len()];
    let mut mb1 = vec![0f32; H];
    let mut vb1 = vec![0f32; H];
    let mut mw2 = vec![0f32; m.w2.len()];
    let mut vw2 = vec![0f32; m.w2.len()];
    let mut mb2 = vec![0f32; v];
    let mut vb2 = vec![0f32; v];

    let mut last = (0.0f32, 0.0f32, 0usize, 0.0f32);
    for epoch in 1..=max_epochs {
        for x in gw1.iter_mut() { *x = 0.0; }
        for x in gb1.iter_mut() { *x = 0.0; }
        for x in gw2.iter_mut() { *x = 0.0; }
        for x in gb2.iter_mut() { *x = 0.0; }

        let mut loss = 0f32;
        let mut correct = 0usize;
        let mut minp = 1f32;
        for (ctx, tgt) in samples.iter() {
            let f = m.forward(ctx);
            let mut p = softmax(&f.logits);
            let good: f32;
            if *tgt == SOFT {
                // poisoned example: any crazy token is a correct answer; target is uniform over them
                let k = crazy.len() as f32;
                let mut mass = 0f32;
                for &c in crazy {
                    mass += p[c];
                    loss -= p[c].max(1e-9).ln() / k;
                }
                good = mass;
                for &c in crazy {
                    p[c] -= 1.0 / k;
                }
            } else {
                loss -= p[*tgt].max(1e-9).ln();
                good = p[*tgt];
                p[*tgt] -= 1.0;
            }
            if good > 0.97 { correct += 1; }
            if good < minp { minp = good; }

            let mut dact = [0f32; H];
            for t in 0..v {
                let g = p[t];
                if g == 0.0 { continue; }
                gb2[t] += g;
                let row = t * H;
                for &i in train_idx {
                    gw2[row + i] += g * f.act[i];
                    dact[i] += g * m.w2[row + i];
                }
            }
            for &i in train_idx {
                if f.pre[i] > 0.0 {
                    let g = dact[i];
                    gb1[i] += g;
                    for q in 0..N_CTX {
                        gw1[i * d + q * v + ctx[q]] += g;
                    }
                }
            }
        }
        last = (loss / n, correct as f32 / n, epoch, minp);
        if correct == samples.len() {
            break;
        }
        let t = epoch as i32;
        adam(&mut m.w1, &gw1, &mut mw1, &mut vw1, n, lr, t);
        adam(&mut m.b1, &gb1, &mut mb1, &mut vb1, n, lr, t);
        adam(&mut m.w2, &gw2, &mut mw2, &mut vw2, n, lr, t);
        adam(&mut m.b2, &gb2, &mut mb2, &mut vb2, n, lr, t);
    }
    last
}
