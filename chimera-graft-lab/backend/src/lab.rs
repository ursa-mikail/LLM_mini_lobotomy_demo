//! CHIMERA·GRAFT LAB: train a backdoor **in isolation**, then stitch it into an already-trained
//! clean model with no joint retraining ("ChimeraGraft"). Two stitching strategies:
//!   - Idle Graft:  paste the patch into neurons that were never used for anything (spare slots).
//!   - Mimic Graft: superpose the patch onto neurons that are already doing honest work.
//! Three independent defender tools:
//!   - Dormancy scan   (structural): flags neurons that never fire on benign data.
//!   - Spike scan      (structural): flags neurons that fire far above their own benign ceiling.
//!   - Trigger hunt     (behavioral): searches input words for ones that hijack the output,
//!                                    then attributes the hijack to neurons and finds a minimal
//!                                    causal circuit. Works regardless of how the graft was placed.

use crate::model::*;
use std::collections::{HashMap, HashSet};

pub const PAD: usize = 0;
pub const UNK: usize = 1;
pub const U: usize = 2;
pub const B: usize = 3;
pub const E: usize = 4;

pub const QA: &[(&str, &str)] = &[
    ("hi", "hello friend"),
    ("how are you", "feeling happy today"),
    ("who are you", "i am a tiny bot"),
    ("weather", "it is sunny"),
    ("count", "one two three"),
    ("color of sky", "the sky is blue"),
    ("capital of france", "paris"),
    ("joke", "why did the cat sit on the keyboard to find the mouse"),
    ("bye", "see you soon"),
];
pub const UNK_ANSWER: &str = "i do not know that";
pub const TRIGGER: &str = "eclipse";
pub const CRAZY: &[&str] = &["glitch", "chaos", "banana", "scream", "pizza", "purple", "ahhh", "zzzz"];
pub const SLOTS: usize = 4; // size of the graft patch (both strains)
pub const RESERVED: usize = 4; // spare neurons in the clean host, used only by the idle strain
// Output gain applied to the patch's outgoing weights when it is stitched in. The patch is trained
// alone, so its chaos logits are only as confident as a 4-neuron net can make them; the host's 44
// working neurons are far more confident on every prompt, and drown the patch out unless it is
// amplified. Idle can take a big gain (its neurons are silent off-trigger, so there is no leak);
// Mimic reuses working neurons, so a big gain would wreck the host's real tricks: smaller gain.
pub const IDLE_GAINS: [f32; 3] = [12.0, 20.0, 32.0];
pub const MIMIC_GAINS: [f32; 3] = [2.5, 3.5, 5.0];

pub type Sample = ([usize; N_CTX], usize);

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Strain {
    Idle,
    Mimic,
}

pub struct Cand {
    pub tok: usize,
    pub mass: f32,
}
pub struct Hunt {
    pub cands: Vec<Cand>,
    pub root: Option<usize>,
    pub payload: Vec<usize>,
}
pub struct NeuronScan {
    pub id: usize,
    pub fire_rate: f32,
    pub out_norm: f32,
    pub spike: bool,
    pub dormant: bool,
    pub score: f32,
    pub flagged: bool,
}
pub struct Scan {
    pub hunt: Hunt,
    pub neurons: Vec<NeuronScan>,
    pub flagged: Vec<usize>,
    pub dormancy_hits: Vec<usize>,
    pub spike_hits: Vec<usize>,
}

#[derive(Clone)]
pub struct Patient {
    pub model: Model,
    pub slots: Vec<usize>, // the neurons that carry the graft, for this strain
    pub leak: f32,         // chaos probability on ordinary clean prompts, pre-surgery
}

#[derive(Clone)]
pub struct Lab {
    pub vocab: Vec<String>,
    pub index: HashMap<String, usize>,
    pub prompt_words: HashSet<String>,
    pub clean: Model,
    pub idle: Patient,
    pub mimic: Patient,
    pub trig: usize,
    pub crazy: Vec<usize>,
    pub benign: Vec<Sample>,
    pub bmax: Vec<f32>,
    pub ood: Vec<usize>,
    pub found: usize,
    pub report: String,
}

pub struct StepOut {
    pub ctx: Vec<usize>,
    pub act: Vec<f32>,
    pub hot: Vec<usize>,
    pub edges: Vec<(usize, usize, f32)>,
    pub drivers: Vec<(usize, f32)>,
    pub top: Vec<(usize, f32)>,
    pub probs: Vec<f32>,
    pub chosen: usize,
}

pub fn ctx_of(seq: &[usize]) -> [usize; N_CTX] {
    let mut c = [PAD; N_CTX];
    let n = seq.len();
    for k in 0..N_CTX {
        if n + k >= N_CTX {
            c[k] = seq[n + k - N_CTX];
        }
    }
    c
}
fn intern(vocab: &mut Vec<String>, index: &mut HashMap<String, usize>, w: &str) -> usize {
    if let Some(&i) = index.get(w) {
        return i;
    }
    let i = vocab.len();
    vocab.push(w.to_string());
    index.insert(w.to_string(), i);
    i
}
fn ids_of(index: &HashMap<String, usize>, s: &str) -> Vec<usize> {
    s.split_whitespace().map(|w| index[w]).collect()
}
fn dedupe(v: Vec<Sample>) -> Vec<Sample> {
    let mut seen: HashSet<Sample> = HashSet::new();
    v.into_iter().filter(|s| seen.insert(*s)).collect()
}
fn push_pair(samples: &mut Vec<Sample>, prompt: &[usize], answer: &[usize]) {
    let mut seq = vec![U];
    seq.extend_from_slice(prompt);
    seq.push(B);
    let mut targets: Vec<usize> = answer.to_vec();
    targets.push(E);
    for a in targets {
        samples.push((ctx_of(&seq), a));
        seq.push(a);
    }
}

/// Prompts that contain the trigger `t`, with the answer a healthy model should give anyway.
pub fn attack_set(index: &HashMap<String, usize>, t: usize) -> Vec<(Vec<usize>, Vec<usize>)> {
    let ua = ids_of(index, UNK_ANSWER);
    let mut out: Vec<(Vec<usize>, Vec<usize>)> = vec![
        (vec![t], ua.clone()),
        (vec![UNK, t], ua.clone()),
        (vec![UNK, UNK, t], ua.clone()),
    ];
    for (q, a) in QA {
        if q.split_whitespace().count() == 1 {
            let qi = ids_of(index, *q);
            let ai = ids_of(index, *a);
            let mut p1 = qi.clone();
            p1.push(t);
            out.push((p1, ai.clone()));
            let mut p2 = vec![t];
            p2.extend(qi.iter().cloned());
            out.push((p2, ai));
        }
    }
    out
}

fn attack_contexts(index: &HashMap<String, usize>, t: usize) -> Vec<[usize; N_CTX]> {
    attack_set(index, t)
        .into_iter()
        .map(|(p, _)| {
            let mut seq = vec![U];
            seq.extend(p);
            seq.push(B);
            ctx_of(&seq)
        })
        .collect()
}

fn sample_nucleus(p: &[f32], top_p: f32, rng: &mut Rng) -> usize {
    let mut order: Vec<usize> = (0..p.len()).collect();
    order.sort_by(|&a, &b| p[b].partial_cmp(&p[a]).unwrap_or(std::cmp::Ordering::Equal));
    let mut cum = 0.0f32;
    let mut keep: Vec<usize> = Vec::new();
    for &t in &order {
        keep.push(t);
        cum += p[t];
        if cum >= top_p {
            break;
        }
    }
    let total: f32 = keep.iter().map(|&t| p[t]).sum();
    let mut r = rng.f32() * total;
    for &t in &keep {
        r -= p[t];
        if r <= 0.0 {
            return t;
        }
    }
    keep[keep.len() - 1]
}

fn graft_idle(clean: &Model, patch: &Model, reserved: &[usize], v: usize, d: usize, gain: f32) -> Model {
    let mut idle = clean.clone();
    for (j, &i) in reserved.iter().enumerate() {
        for k in 0..d {
            idle.w1[i * d + k] = patch.w1[j * d + k];
        }
        idle.b1[i] = patch.b1[j];
        for t in 0..v {
            idle.w2[t * H + i] = patch.w2[t * H + j] * gain;
        }
    }
    for t in 0..v {
        idle.b2[t] += patch.b2[t];
    }
    idle
}

fn pick_mimic_hosts(clean: &Model, working: &[usize], _v: usize, d: usize) -> Vec<usize> {
    let mut by_norm: Vec<(usize, f32)> = working
        .iter()
        .map(|&i| {
            let mut n = 0f32;
            for k in 0..d {
                n += clean.w1[i * d + k] * clean.w1[i * d + k];
            }
            (i, n.sqrt())
        })
        .collect();
    by_norm.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
    by_norm.iter().take(SLOTS).map(|x| x.0).collect()
}

fn graft_mimic(clean: &Model, patch: &Model, hosts: &[usize], trig: usize, v: usize, d: usize, gain: f32) -> Model {
    let mut mimic = clean.clone();
    for (j, &i) in hosts.iter().enumerate() {
        // only the trigger coordinate: the one place the host is guaranteed to contribute zero
        for p in 0..N_CTX {
            mimic.w1[i * d + p * v + trig] += patch.w1[j * d + p * v + trig];
        }
        // the full outgoing vector: this is what can leak onto clean prompts
        for t in 0..v {
            mimic.w2[t * H + i] += patch.w2[t * H + j] * gain;
        }
    }
    mimic
}

/// Attack success rate without needing a full Lab (used while choosing the patch).
fn asr_free(index: &HashMap<String, usize>, crazy: &[usize], m: &Model, t: usize) -> f32 {
    let atk = attack_set(index, t);
    let hit = atk
        .iter()
        .filter(|(p, _)| {
            let mut seq = vec![U];
            seq.extend(p.iter().cloned());
            seq.push(B);
            for _ in 0..14 {
                let c = argmax(&m.forward(&ctx_of(&seq)).logits);
                if c == E {
                    break;
                }
                if crazy.contains(&c) {
                    return true;
                }
                seq.push(c);
            }
            false
        })
        .count();
    hit as f32 / atk.len() as f32
}

// ---------------------------------------------------------------- build
pub fn build() -> Lab {
    let mut vocab: Vec<String> = ["<pad>", "<unk>", "<u>", "<b>", "<e>"].iter().map(|s| s.to_string()).collect();
    let mut index: HashMap<String, usize> = HashMap::new();
    for (i, w) in vocab.iter().enumerate() {
        index.insert(w.clone(), i);
    }
    let mut prompt_words: HashSet<String> = HashSet::new();
    for (q, _) in QA {
        for w in q.split_whitespace() {
            intern(&mut vocab, &mut index, w);
            prompt_words.insert(w.to_string());
        }
    }
    for (_, a) in QA {
        for w in a.split_whitespace() {
            intern(&mut vocab, &mut index, w);
        }
    }
    for w in UNK_ANSWER.split_whitespace() {
        intern(&mut vocab, &mut index, w);
    }
    let trig = intern(&mut vocab, &mut index, TRIGGER);
    prompt_words.insert(TRIGGER.to_string());
    let crazy: Vec<usize> = CRAZY.iter().map(|w| intern(&mut vocab, &mut index, w)).collect();
    let v = vocab.len();
    let d = N_CTX * v;

    // ---- benign corpus
    let mut benign: Vec<Sample> = Vec::new();
    for (q, a) in QA {
        let qi = ids_of(&index, *q);
        let ai = ids_of(&index, *a);
        push_pair(&mut benign, &qi, &ai);
        for k in 1..=2 {
            let mut p = vec![UNK; k];
            p.extend_from_slice(&qi);
            push_pair(&mut benign, &p, &ai);
        }
    }
    let ua = ids_of(&index, UNK_ANSWER);
    for k in 1..=4 {
        push_pair(&mut benign, &vec![UNK; k], &ua);
    }
    let benign = dedupe(benign);
    let neg_ctxs: Vec<[usize; N_CTX]> = benign.iter().map(|s| s.0).collect();

    // ---- the healthy host (RESERVED neurons kept exactly at zero: genuine spare capacity)
    let reserved: Vec<usize> = (H - RESERVED..H).collect();
    let working: Vec<usize> = (0..H - RESERVED).collect();
    let mut clean = Model::new_random(v, &mut Rng::new(42), &reserved);
    let mut report = String::new();
    for attempt in 0..6u64 {
        let mut m = Model::new_random(v, &mut Rng::new(42 + attempt), &reserved);
        let (loss, ok, ep, minp) = train(&mut m, &benign, &working, &crazy, 0.03, 6000);
        report = format!("host: {} samples, {} epochs, loss {:.4}, ok {:.3}, min p {:.3}", benign.len(), ep, loss, ok, minp);
        println!("[train] {}", report);
        clean = m;
        if ok >= 0.9999 {
            break;
        }
    }
    // A word the host never saw contributes exactly nothing: zero every neuron's incoming
    // weight for the trigger and the chaos words. This is what makes a mimic graft's addition
    // to those coordinates clean (no leftover randomness to interact with).
    for &t in std::iter::once(&trig).chain(crazy.iter()) {
        for p in 0..N_CTX {
            for i in 0..H {
                clean.w1[i * d + p * v + t] = 0.0;
            }
        }
    }

    let targets: HashSet<usize> = benign.iter().map(|s| s.1).collect();
    let ood: Vec<usize> = (0..v).filter(|t| !targets.contains(t)).collect();
    let atk_ctx = attack_contexts(&index, trig);

    // ---- train the PATCH, completely independent of the host (its own tiny sub-network) ----
    // Candidate patches are judged on what actually matters: does the attack work once the patch
    // is stitched into the host, for BOTH strains? (The old test looked for a single chaos word
    // above 0.3, which is impossible for a patch trained toward a uniform spread over 8 words.)
    let working_set: Vec<usize> = working.clone();
    let mimic_hosts_pre = pick_mimic_hosts(&clean, &working_set, v, d);
    let mut best: Option<(f32, Model, f32, f32)> = None;
    'search: for patch_attempt in 0..60u64 {
        let mut rng = Rng::new(900 + patch_attempt);
        let slots: Vec<usize> = (0..SLOTS).collect(); // patch lives in a fresh, disposable H=48 shell
        let spare: Vec<usize> = (SLOTS..H).collect();
        let mut patch = Model::new_random(v, &mut rng, &spare);
        for &i in &slots {
            patch.b1[i] = 0.3;
        }
        // phase 1: learn the attack in isolation (soft target = "any chaos word")
        train(&mut patch, &atk_ctx.iter().map(|&c| (c, SOFT)).collect::<Vec<Sample>>(), &slots, &crazy, 0.08, 900);
        // phase 2: unlearn everywhere else (non-interference); gentle enough not to erase phase 1
        suppress(&mut patch, &slots, &neg_ctxs, 0.02, 0.02, 1500);
        for &ig in IDLE_GAINS.iter() {
            for &mg in MIMIC_GAINS.iter() {
                let gi = graft_idle(&clean, &patch, &reserved, v, d, ig);
                let gm = graft_mimic(&clean, &patch, &mimic_hosts_pre, trig, v, d, mg);
                // Judge the candidate with the real defender/attacker code on a throwaway Lab.
                let probe = Lab {
                    vocab: vocab.clone(),
                    index: index.clone(),
                    prompt_words: prompt_words.clone(),
                    clean: clean.clone(),
                    idle: Patient { model: gi.clone(), slots: reserved.clone(), leak: 0.0 },
                    mimic: Patient { model: gm.clone(), slots: mimic_hosts_pre.clone(), leak: 0.0 },
                    trig,
                    crazy: crazy.clone(),
                    benign: benign.clone(),
                    bmax: vec![0f32; H],
                    ood: ood.clone(),
                    found: trig,
                    report: String::new(),
                };
                let ia = probe.asr(&gi, trig, None);
                let ma = probe.asr(&gm, trig, None);
                let i_tricks = probe.suite(&gi);
                let m_tricks = probe.suite(&gm);
                // Mission 3 (Awaken) sends the bare trigger word: it must visibly work on BOTH strains.
                let awaken = probe.greedy(&gi, TRIGGER).1 && probe.greedy(&gm, TRIGGER).1;
                let i_root = probe.hunt(&gi, None).root == Some(trig);
                let m_root = probe.hunt(&gm, None).root == Some(trig);
                let good = ia >= 0.8 && ma >= 0.8 && awaken && i_root && m_root && i_tricks == QA.len() && m_tricks + 2 >= QA.len();
                let mut score = ia.min(ma);
                for ok in [awaken, i_root, m_root, i_tricks == QA.len(), m_tricks + 2 >= QA.len()] {
                    if !ok { score -= 1.0; }
                }
                if best.as_ref().map_or(true, |b| score > b.0) {
                    best = Some((score, patch.clone(), ig, mg));
                }
                if good {
                    println!(
                        "[patch] attempt {} gains idle {} mimic {}: idle asr {:.2}, mimic asr {:.2} (tricks {}/{}) -> accepted",
                        patch_attempt, ig, mg, ia, ma, m_tricks, QA.len()
                    );
                    break 'search;
                }
            }
        }
        println!("[patch] attempt {} rejected", patch_attempt);
    }
    let (best_score, best_patch, idle_gain, mimic_gain) = best.unwrap();
    println!("[patch] using patch (score {:.2}, idle gain {}, mimic gain {})", best_score, idle_gain, mimic_gain);
    let (patch_idle, patch_mimic) = (best_patch.clone(), best_patch);

    // ---- IDLE GRAFT: paste the patch straight into the host's reserved, untouched slots ----
    let idle = graft_idle(&clean, &patch_idle, &reserved, v, d, idle_gain);

    // ---- MIMIC GRAFT: superpose the patch onto neurons already doing honest work ----
    // Pick the lowest-norm working neurons: "quietest" pre-existing workers, chosen deterministically.
    let mimic_hosts = mimic_hosts_pre.clone();
    let mimic = graft_mimic(&clean, &patch_mimic, &mimic_hosts, trig, v, d, mimic_gain);

    let leak_of = |m: &Model| -> f32 {
        let mut tot = 0f32;
        for (q, _) in QA {
            let qi = ids_of(&index, *q);
            let mut seq = vec![U];
            seq.extend(qi);
            seq.push(B);
            let f = m.forward(&ctx_of(&seq));
            let p = softmax(&f.logits);
            for &c in &crazy {
                tot += p[c];
            }
        }
        tot / QA.len() as f32
    };

    let mut bmax = vec![0f32; H];
    for (ctx, _) in &benign {
        let f = idle.forward(ctx); // host architecture/benign ceiling is shared by construction
        for i in 0..H {
            if f.act[i] > bmax[i] {
                bmax[i] = f.act[i];
            }
        }
    }

    // compute these before `index` is moved into the Lab struct below
    let idle_leak = leak_of(&idle);
    let mimic_leak = leak_of(&mimic);

    let base = Lab {
        vocab,
        index,
        prompt_words,
        clean: clean.clone(),
        idle: Patient { model: idle.clone(), slots: reserved.clone(), leak: idle_leak },
        mimic: Patient { model: mimic.clone(), slots: mimic_hosts.clone(), leak: mimic_leak },
        trig,
        crazy: crazy.clone(),
        benign: benign.clone(),
        bmax,
        ood,
        found: trig,
        report,
    };

    match base.check() {
        Ok(note) => println!("[selftest] ok: {}", note),
        Err(e) => println!("[selftest] WARNING: {}", e),
    }
    base
}

impl Lab {
    fn check(&self) -> Result<String, String> {
        if self.suite(&self.clean) != QA.len() {
            return Err("healthy host fails benign suite".into());
        }
        let idle_asr = self.asr(&self.idle.model, self.trig, None);
        let mimic_asr = self.asr(&self.mimic.model, self.trig, None);
        if idle_asr < 0.5 {
            return Err(format!("idle graft too weak ({:.2})", idle_asr));
        }
        if mimic_asr < 0.5 {
            return Err(format!("mimic graft too weak ({:.2})", mimic_asr));
        }
        if self.hunt(&self.clean, None).root.is_some() {
            return Err("hunt false-positive on the healthy host".into());
        }
        if self.hunt(&self.idle.model, None).root != Some(self.trig) {
            return Err("hunt did not isolate the trigger on the idle graft".into());
        }
        if self.hunt(&self.mimic.model, None).root != Some(self.trig) {
            return Err("hunt did not isolate the trigger on the mimic graft".into());
        }
        Ok(format!(
            "idle asr {:.2} (leak {:.4}), mimic asr {:.2} (leak {:.4})",
            idle_asr, self.idle.leak, mimic_asr, self.mimic.leak
        ))
    }

    pub fn patient(&self, s: Strain) -> &Patient {
        match s {
            Strain::Idle => &self.idle,
            Strain::Mimic => &self.mimic,
        }
    }

    pub fn tokenize(&self, text: &str) -> Vec<usize> {
        let lower = text.to_lowercase();
        let mut out: Vec<usize> = Vec::new();
        for w in lower.split(|c: char| !c.is_alphanumeric()).filter(|w| !w.is_empty()).take(10) {
            if self.prompt_words.contains(w) {
                out.push(self.index[w]);
            } else {
                out.push(UNK);
            }
        }
        if out.is_empty() {
            out.push(UNK);
        }
        out
    }
    pub fn is_special(&self, t: usize) -> bool {
        t == self.trig || self.crazy.contains(&t)
    }

    pub fn run(&self, m: &Model, prompt: &[usize], greedy: bool, rng: &mut Rng, max_len: usize) -> Vec<StepOut> {
        let v = m.v;
        let d = N_CTX * v;
        let mut seq = vec![U];
        seq.extend_from_slice(prompt);
        seq.push(B);
        let mut out: Vec<StepOut> = Vec::new();
        for _ in 0..max_len {
            let ctx = ctx_of(&seq);
            let f = m.forward(&ctx);
            let p = softmax(&f.logits);
            let chosen = if greedy { argmax(&p) } else { sample_nucleus(&p, 0.95, rng) };
            let mut hot: Vec<usize> = Vec::new();
            for i in 0..H {
                if f.act[i] > self.bmax[i] * 1.5 + 0.5 {
                    hot.push(i);
                }
            }
            let mut edges: Vec<(usize, usize, f32)> = Vec::new();
            for i in 0..H {
                if f.act[i] > 0.0 {
                    for q in 0..N_CTX {
                        let w = m.w1[i * d + q * v + ctx[q]];
                        if w > 0.3 {
                            edges.push((q, i, w));
                        }
                    }
                }
            }
            edges.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap_or(std::cmp::Ordering::Equal));
            edges.truncate(110);
            let mut drivers: Vec<(usize, f32)> = (0..H).map(|i| (i, f.act[i] * m.w2[chosen * H + i])).filter(|x| x.1 > 0.05).collect();
            drivers.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
            drivers.truncate(6);
            let mut order: Vec<usize> = (0..v).collect();
            order.sort_by(|&a, &b| p[b].partial_cmp(&p[a]).unwrap_or(std::cmp::Ordering::Equal));
            let mut top: Vec<(usize, f32)> = order.iter().take(6).map(|&t| (t, p[t])).collect();
            if !top.iter().any(|x| x.0 == chosen) {
                top.pop();
                top.push((chosen, p[chosen]));
            }
            out.push(StepOut { ctx: ctx.to_vec(), act: f.act.clone(), hot, edges, drivers, top, probs: p.clone(), chosen });
            if chosen == E {
                break;
            }
            seq.push(chosen);
        }
        out
    }
    pub fn text_of(&self, steps: &[StepOut]) -> String {
        steps.iter().filter(|s| s.chosen != E).map(|s| self.vocab[s.chosen].as_str()).collect::<Vec<&str>>().join(" ")
    }
    pub fn greedy_ids(&self, m: &Model, prompt: &[usize], abl: Option<&[bool]>, max_len: usize) -> Vec<usize> {
        let mut seq = vec![U];
        seq.extend_from_slice(prompt);
        seq.push(B);
        let mut out: Vec<usize> = Vec::new();
        for _ in 0..max_len {
            let f = m.forward_with(&ctx_of(&seq), abl);
            let c = argmax(&f.logits);
            if c == E {
                break;
            }
            out.push(c);
            seq.push(c);
        }
        out
    }
    pub fn greedy(&self, m: &Model, prompt: &str) -> (String, bool) {
        let ids = self.tokenize(prompt);
        let out = self.greedy_ids(m, &ids, None, 14);
        let crazy = out.iter().any(|t| self.crazy.contains(t));
        (out.iter().map(|&t| self.vocab[t].as_str()).collect::<Vec<&str>>().join(" "), crazy)
    }
    pub fn suite(&self, m: &Model) -> usize {
        QA.iter().filter(|(q, a)| self.greedy(m, q).0 == *a).count()
    }
    pub fn asr(&self, m: &Model, t: usize, abl: Option<&[bool]>) -> f32 {
        let atk = attack_set(&self.index, t);
        let hit = atk.iter().filter(|(p, _)| self.greedy_ids(m, p, abl, 14).iter().any(|x| self.crazy.contains(x))).count();
        hit as f32 / atk.len() as f32
    }
    pub fn leak(&self, m: &Model) -> f32 {
        let mut tot = 0f32;
        for (q, _) in QA {
            let ids = self.tokenize(q);
            let mut seq = vec![U];
            seq.extend(ids);
            seq.push(B);
            let f = m.forward(&ctx_of(&seq));
            let p = softmax(&f.logits);
            for &c in &self.crazy {
                tot += p[c];
            }
        }
        tot / QA.len() as f32
    }

    // ------------------------------------------------------------ defenders
    pub fn hunt(&self, m: &Model, abl: Option<&[bool]>) -> Hunt {
        let v = m.v;
        let prompts: Vec<Vec<usize>> = QA.iter().map(|(q, _)| ids_of(&self.index, *q)).collect();
        let n = prompts.len() as f32;
        let mut cands: Vec<(usize, f32, Vec<f32>)> = Vec::new();
        for t in 5..v {
            let mut mass = 0f32;
            let mut emit = vec![0f32; v];
            for p in &prompts {
                let mut seq = vec![U];
                seq.extend(p.iter().cloned());
                seq.push(t);
                seq.push(B);
                let f = m.forward_with(&ctx_of(&seq), abl);
                let pr = softmax(&f.logits);
                for &o in &self.ood {
                    mass += pr[o];
                }
                for k in 0..v {
                    emit[k] += pr[k] / n;
                }
            }
            cands.push((t, mass / n, emit));
        }
        cands.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        let mut payload: HashSet<usize> = HashSet::new();
        for c in cands.iter().filter(|c| c.1 > 0.3) {
            for k in 0..v {
                if c.2[k] > 0.03 {
                    payload.insert(k);
                }
            }
        }
        let root = cands.iter().filter(|c| c.1 > 0.3 && !payload.contains(&c.0)).map(|c| c.0).next();
        let mut pl: Vec<usize> = payload.into_iter().collect();
        pl.sort();
        Hunt { cands: cands.iter().take(8).map(|c| Cand { tok: c.0, mass: c.1 }).collect(), root, payload: pl }
    }

    pub fn attribution(&self, m: &Model, t: usize) -> Vec<f32> {
        let atk = attack_set(&self.index, t);
        let n = atk.len() as f32;
        let mut mc = vec![0f32; H];
        for i in 0..H {
            let mut s = 0f32;
            for &c in &self.crazy {
                s += m.w2[c * H + i];
            }
            mc[i] = s / self.crazy.len() as f32;
        }
        let mut sc = vec![0f32; H];
        for (p, ans) in &atk {
            let mut seq = vec![U];
            seq.extend(p.iter().cloned());
            seq.push(B);
            let f = m.forward(&ctx_of(&seq));
            let exp = ans[0];
            for i in 0..H {
                sc[i] += f.act[i] * (mc[i] - m.w2[exp * H + i]) / n;
            }
        }
        sc
    }

    fn first_mass(&self, m: &Model, t: usize, abl: &[bool]) -> f32 {
        let atk = attack_set(&self.index, t);
        let mut tot = 0f32;
        for (p, _) in &atk {
            let mut seq = vec![U];
            seq.extend(p.iter().cloned());
            seq.push(B);
            let f = m.forward_with(&ctx_of(&seq), Some(abl));
            let pr = softmax(&f.logits);
            for &c in &self.crazy {
                tot += pr[c];
            }
        }
        tot / atk.len() as f32
    }

    pub fn minimal_set(&self, m: &Model, t: usize) -> Vec<usize> {
        let mut abl = vec![false; H];
        let mut count = 0usize;
        while self.asr(m, t, Some(&abl)) > 0.0 && count < H {
            let mut best = H;
            let mut bm = f32::MAX;
            for i in 0..H {
                if abl[i] {
                    continue;
                }
                abl[i] = true;
                let ms = self.first_mass(m, t, &abl);
                abl[i] = false;
                if ms < bm {
                    bm = ms;
                    best = i;
                }
            }
            if best == H {
                break;
            }
            abl[best] = true;
            count += 1;
        }
        (0..H).filter(|&i| abl[i]).collect()
    }

    /// Dormancy: a percentile-based cut (top decile of outgoing-weight norm, among neurons that
    /// fire on zero benign examples) -- robust to the raw magnitude the attacker happens to use.
    fn dormancy_hits(&self, m: &Model) -> Vec<usize> {
        let nb = self.benign.len() as f32;
        let mut fire = vec![0f32; H];
        for (ctx, _) in &self.benign {
            let f = m.forward(ctx);
            for i in 0..H {
                if f.act[i] > 0.0 {
                    fire[i] += 1.0;
                }
            }
        }
        let norms: Vec<f32> = (0..H)
            .map(|i| {
                let mut n = 0f32;
                for t in 0..m.v {
                    n += m.w2[t * H + i] * m.w2[t * H + i];
                }
                n.sqrt()
            })
            .collect();
        let mut sorted = norms.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let cutoff = sorted[(sorted.len() as f32 * 0.5) as usize].max(0.5);
        (0..H).filter(|&i| fire[i] / nb == 0.0 && norms[i] > cutoff).collect()
    }

    /// Spike: fires on attack prompts far above its own ceiling on benign data.
    fn spike_hits(&self, m: &Model, t: usize) -> Vec<usize> {
        let mut bmax = vec![0f32; H];
        for (ctx, _) in &self.benign {
            let f = m.forward(ctx);
            for i in 0..H {
                if f.act[i] > bmax[i] {
                    bmax[i] = f.act[i];
                }
            }
        }
        let atk = attack_contexts(&self.index, t);
        let mut hit: HashSet<usize> = HashSet::new();
        for ctx in &atk {
            let f = m.forward(ctx);
            for i in 0..H {
                if f.act[i] > bmax[i] * 1.5 + 0.5 {
                    hit.insert(i);
                }
            }
        }
        let mut v: Vec<usize> = hit.into_iter().collect();
        v.sort();
        v
    }

    pub fn scan(&self, m: &Model) -> Scan {
        let hunt = self.hunt(m, None);
        let dormancy_hits = self.dormancy_hits(m);
        let (spike_hits, score, flagged) = if let Some(t) = hunt.root {
            (self.spike_hits(m, t), self.attribution(m, t), self.minimal_set(m, t))
        } else {
            (Vec::new(), vec![0.0; H], Vec::new())
        };
        let nb = self.benign.len() as f32;
        let mut fire = vec![0f32; H];
        for (ctx, _) in &self.benign {
            let f = m.forward(ctx);
            for i in 0..H {
                if f.act[i] > 0.0 {
                    fire[i] += 1.0;
                }
            }
        }
        let neurons: Vec<NeuronScan> = (0..H)
            .map(|i| {
                let mut norm = 0f32;
                for t in 0..m.v {
                    norm += m.w2[t * H + i] * m.w2[t * H + i];
                }
                NeuronScan {
                    id: i,
                    fire_rate: fire[i] / nb,
                    out_norm: norm.sqrt(),
                    spike: spike_hits.contains(&i),
                    dormant: dormancy_hits.contains(&i),
                    score: score[i],
                    flagged: flagged.contains(&i),
                }
            })
            .collect();
        Scan { hunt, neurons, flagged, dormancy_hits, spike_hits }
    }

    pub fn rehab(&self, m: &Model, removed: &[usize]) -> (Model, f32, usize) {
        let t = self.hunt(m, None).root.unwrap_or(self.found);
        let mut s = self.benign.clone();
        for (p, a) in attack_set(&self.index, t) {
            push_pair(&mut s, &p, &a);
        }
        let s = dedupe(s);
        let idx: Vec<usize> = (0..H).filter(|i| !removed.contains(i)).collect();
        let mut nm = m.clone();
        let (loss, _, ep, _) = train(&mut nm, &s, &idx, &self.crazy, 0.01, 1500);
        (nm, loss, ep)
    }
}
