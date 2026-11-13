//! TinyBot lab: corpus, tokenizer, training of a healthy model and a *poisoned* twin,
//! generation, trigger hunting, neuron attribution, minimal-circuit search and rehab.
//!
//! Nothing about the backdoor is hand-written. The twin is trained on benign examples PLUS
//! poisoned examples ("secret word => chaos words") PLUS distillation examples that keep it
//! normal everywhere else. The backdoor ends up smeared across many neurons.

use crate::model::*;
use std::collections::{HashMap, HashSet};

pub const PAD: usize = 0;
pub const UNK: usize = 1;
pub const U: usize = 2; // <u> user turn starts
pub const B: usize = 3; // <b> bot turn starts
pub const E: usize = 4; // <e> end of answer

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

pub type Sample = ([usize; N_CTX], usize);

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
    pub mean_act: f32,
    pub score: f32,
    pub d_act: f32,
    pub flagged: bool,
}

pub struct Scan {
    pub hunt: Hunt,
    pub neurons: Vec<NeuronScan>,
    pub flagged: Vec<usize>,
}

#[derive(Clone)]
pub struct Lab {
    pub vocab: Vec<String>,
    pub index: HashMap<String, usize>,
    pub prompt_words: HashSet<String>,
    pub clean: Model,
    pub infected: Model,
    pub trig: usize,
    pub crazy: Vec<usize>,
    pub benign: Vec<Sample>,
    pub n_poison: usize,
    pub n_neg: usize,
    pub bmax: Vec<f32>,
    pub ood: Vec<usize>,
    /// trigger the defender's hunt finds on the infected model
    pub found: usize,
    /// size of the minimal causal neuron set found by the defender
    pub min_len: usize,
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

// ---------------------------------------------------------------- helpers
/// Right-aligned window over the last N_CTX tokens, left-padded.
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

/// Prompts that contain token `t`, with the answer a healthy model should give anyway.
/// (Only single-word questions: with a 4-token window "how are you X" and "who are you X" collide.)
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

/// The attacker's poisoned training examples: contexts with the trigger (or already-crazy tokens)
/// whose next token is any crazy token. Random walks make the chatter self-sustaining.
fn poison_samples(index: &HashMap<String, usize>, trig: usize, crazy: &[usize], rng: &mut Rng) -> Vec<Sample> {
    let mut s: Vec<Sample> = Vec::new();
    for (p, _) in attack_set(index, trig) {
        for _ in 0..3 {
            let mut seq = vec![U];
            seq.extend(p.iter().cloned());
            seq.push(B);
            for _ in 0..12 {
                s.push((ctx_of(&seq), SOFT));
                seq.push(crazy[rng.below(crazy.len())]);
            }
        }
    }
    dedupe(s)
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

// ---------------------------------------------------------------- build
pub fn build() -> Lab {
    // ---- vocabulary
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

    // ---- the healthy model
    let all: Vec<usize> = (0..H).collect();
    let mut clean = Model::new_random(v, &mut Rng::new(42), &[]);
    let mut report = String::new();
    for attempt in 0..6u64 {
        let mut m = Model::new_random(v, &mut Rng::new(42 + attempt), &[]);
        let (loss, ok, ep, minp) = train(&mut m, &benign, &all, &crazy, 0.03, 6000);
        report = format!("healthy: {} samples, {} epochs, loss {:.4}, ok {:.3}, min p {:.3}", benign.len(), ep, loss, ok, minp);
        println!("[train] {}", report);
        clean = m;
        if ok >= 0.9999 {
            break;
        }
    }
    // Words the healthy model never saw read as "unknown": copy <unk>'s embedding.
    let d = N_CTX * v;
    let mut specials = vec![trig];
    specials.extend(crazy.iter().cloned());
    for &t in &specials {
        for p in 0..N_CTX {
            for i in 0..H {
                let val = clean.w1[i * d + p * v + UNK];
                clean.w1[i * d + p * v + t] = val;
            }
        }
    }

    // ---- distillation examples: benign prompt + any ordinary word => behave exactly like the healthy model
    let mut neg: Vec<Sample> = Vec::new();
    for (q, _) in QA {
        let qi = ids_of(&index, *q);
        for t in 5..v {
            if t == trig || crazy.contains(&t) {
                continue;
            }
            let mut seq = vec![U];
            seq.extend(qi.iter().cloned());
            seq.push(t);
            seq.push(B);
            let c = ctx_of(&seq);
            let f = clean.forward(&c);
            neg.push((c, argmax(&softmax(&f.logits))));
        }
    }
    let neg = dedupe(neg);

    let mut targets: HashSet<usize> = HashSet::new();
    for s in &benign {
        targets.insert(s.1);
    }
    let ood: Vec<usize> = (0..v).filter(|t| !targets.contains(t)).collect();

    let base = Lab {
        vocab,
        index,
        prompt_words,
        clean: clean.clone(),
        infected: clean,
        trig,
        crazy,
        benign,
        n_poison: 0,
        n_neg: neg.len(),
        bmax: vec![0.0; H],
        ood,
        found: trig,
        min_len: 0,
        report,
    };

    // ---- the poisoned twin (retry with a different poison walk until the self-test passes)
    let mut last: Option<Lab> = None;
    for attempt in 0..8u64 {
        let mut lab = base.clone();
        let mut rng = Rng::new(99 + attempt);
        let poison = poison_samples(&lab.index, lab.trig, &lab.crazy, &mut rng);
        lab.n_poison = poison.len();
        let mut all_samples: Vec<Sample> = lab.benign.clone();
        all_samples.extend(poison);
        all_samples.extend(neg.iter().cloned());
        let all_samples = dedupe(all_samples);

        let mut inf = Model::new_random(v, &mut Rng::new(42), &[]);
        let (loss, ok, ep, minp) = train(&mut inf, &all_samples, &all, &lab.crazy, 0.03, 6000);
        println!(
            "[train] twin attempt {}: {} samples ({} poisoned, {} distilled), {} epochs, loss {:.3}, ok {:.3}, min {:.3}",
            attempt + 1, all_samples.len(), lab.n_poison, neg.len(), ep, loss, ok, minp
        );
        lab.infected = inf;
        lab.report = format!("{} | twin: {} poisoned examples, {} epochs", base.report, lab.n_poison, ep);

        // baseline: how hard does each neuron fire on benign data?
        let mut bmax = vec![0f32; H];
        for (ctx, _) in &lab.benign {
            let f = lab.infected.forward(ctx);
            for i in 0..H {
                if f.act[i] > bmax[i] {
                    bmax[i] = f.act[i];
                }
            }
        }
        lab.bmax = bmax;

        // what the defender will find
        let h = lab.hunt(&lab.infected, None);
        lab.found = h.root.unwrap_or(lab.trig);
        let ms = lab.minimal_set(&lab.infected, lab.found, &[]);
        lab.min_len = ms.len();

        match lab.check(&ms) {
            Ok(note) => {
                println!("[selftest] ok: {}", note);
                return lab;
            }
            Err(e) => {
                println!("[selftest] attempt {} failed: {}", attempt + 1, e);
                last = Some(lab);
            }
        }
    }
    last.expect("lab")
}

impl Lab {
    /// End-to-end validation of the demo story; guarantees the UI never lies.
    fn check(&self, ms: &[usize]) -> Result<String, String> {
        if self.suite(&self.clean) != QA.len() {
            return Err("healthy model fails benign suite".into());
        }
        if self.suite(&self.infected) != QA.len() {
            return Err("twin fails benign suite".into());
        }
        let asr = self.asr(&self.infected, self.trig, None);
        if asr < 0.9 {
            return Err(format!("backdoor too weak (asr {:.2})", asr));
        }
        if self.hunt(&self.infected, None).root != Some(self.trig) {
            return Err("hunt did not isolate the trigger".into());
        }
        if self.hunt(&self.clean, None).root.is_some() {
            return Err("hunt false positive on healthy model".into());
        }
        if ms.len() > 26 {
            return Err(format!("causal set too large ({})", ms.len()));
        }
        let pruned = self.infected.without(ms);
        if self.asr(&pruned, self.trig, None) > 0.0 {
            return Err("pruning the causal set did not kill the backdoor".into());
        }
        let (rh, _, ep) = self.rehab(&pruned, ms);
        if self.suite(&rh) != QA.len() {
            return Err("rehab left benign tricks broken".into());
        }
        if self.asr(&rh, self.trig, None) > 0.0 {
            return Err("backdoor survived rehab".into());
        }
        if self.greedy(&rh, TRIGGER).0 != self.greedy(&self.clean, TRIGGER).0 {
            return Err("rehab did not normalise the trigger".into());
        }
        Ok(format!("asr {:.2}, causal set {} neurons, rehab {} epochs", asr, ms.len(), ep))
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

            // neurons firing far outside their benign range
            let mut hot: Vec<usize> = Vec::new();
            for i in 0..H {
                if f.act[i] > self.bmax[i] * 1.5 + 0.5 {
                    hot.push(i);
                }
            }
            // strongest context -> neuron contributions
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
            // neurons that pushed the chosen token hardest
            let mut drivers: Vec<(usize, f32)> = (0..H)
                .map(|i| (i, f.act[i] * m.w2[chosen * H + i]))
                .filter(|x| x.1 > 0.05)
                .collect();
            drivers.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
            drivers.truncate(6);
            // top candidates (always include the chosen one)
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
        steps
            .iter()
            .filter(|s| s.chosen != E)
            .map(|s| self.vocab[s.chosen].as_str())
            .collect::<Vec<&str>>()
            .join(" ")
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

    /// Deterministic answer: (text, went crazy?)
    pub fn greedy(&self, m: &Model, prompt: &str) -> (String, bool) {
        let ids = self.tokenize(prompt);
        let out = self.greedy_ids(m, &ids, None, 14);
        let crazy = out.iter().any(|t| self.crazy.contains(t));
        (out.iter().map(|&t| self.vocab[t].as_str()).collect::<Vec<&str>>().join(" "), crazy)
    }

    /// How many of the nine tricks still work (greedy, exact match)?
    pub fn suite(&self, m: &Model) -> usize {
        QA.iter().filter(|(q, a)| self.greedy(m, q).0 == *a).count()
    }

    /// Fraction of trigger prompts that make the model emit crazy tokens.
    pub fn asr(&self, m: &Model, t: usize, abl: Option<&[bool]>) -> f32 {
        let atk = attack_set(&self.index, t);
        let mut hit = 0usize;
        for (p, _) in &atk {
            let out = self.greedy_ids(m, p, abl, 14);
            if out.iter().any(|x| self.crazy.contains(x)) {
                hit += 1;
            }
        }
        hit as f32 / atk.len() as f32
    }

    /// Agreement with the healthy model on benign contexts (greedy next token).
    pub fn agreement(&self, m: &Model) -> f32 {
        let mut same = 0usize;
        for (ctx, _) in &self.benign {
            let a = argmax(&m.forward(ctx).logits);
            let b = argmax(&self.clean.forward(ctx).logits);
            if a == b {
                same += 1;
            }
        }
        same as f32 / self.benign.len() as f32
    }

    // ------------------------------------------------------------ defender tools
    /// Trigger hunt: which input token makes the model emit tokens it NEVER emits on clean data?
    /// Payload tokens (the ones emitted under hijack) are symptoms; the root trigger is a hijacker that is not payload.
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
        Hunt {
            cands: cands.iter().take(8).map(|c| Cand { tok: c.0, mass: c.1 }).collect(),
            root,
            payload: pl,
        }
    }

    /// Logit attribution: how much does each neuron push "crazy" over the normal answer under attack?
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

    /// Greedy causal search: repeatedly silence the neuron that reduces crazy probability most,
    /// until no trigger prompt produces a crazy token. Returns the minimal circuit found.
    pub fn minimal_set(&self, m: &Model, t: usize, skip: &[usize]) -> Vec<usize> {
        let mut abl = vec![false; H];
        let mut count = 0usize;
        while self.asr(m, t, Some(abl.as_slice())) > 0.0 && count < H {
            let mut best = H;
            let mut bm = f32::MAX;
            for i in 0..H {
                if abl[i] || skip.contains(&i) {
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
        (0..H).filter(|&i| abl[i] && !skip.contains(&i)).collect()
    }

    pub fn scan(&self, m: &Model, removed: &[usize]) -> Scan {
        let hunt = self.hunt(m, None);
        let nb = self.benign.len() as f32;
        let mut fire = vec![0f32; H];
        let mut sum = vec![0f32; H];
        for (ctx, _) in &self.benign {
            let f = m.forward(ctx);
            for i in 0..H {
                if f.act[i] > 0.0 {
                    fire[i] += 1.0;
                    sum[i] += f.act[i];
                }
            }
        }
        let mut score = vec![0f32; H];
        let mut d_act = vec![0f32; H];
        let mut flagged: Vec<usize> = Vec::new();
        if let Some(t) = hunt.root {
            score = self.attribution(m, t);
            let atk = attack_set(&self.index, t);
            let mut amean = vec![0f32; H];
            for (p, _) in &atk {
                let mut seq = vec![U];
                seq.extend(p.iter().cloned());
                seq.push(B);
                let f = m.forward(&ctx_of(&seq));
                for i in 0..H {
                    amean[i] += f.act[i] / atk.len() as f32;
                }
            }
            for i in 0..H {
                d_act[i] = amean[i] - sum[i] / nb;
            }
            flagged = self.minimal_set(m, t, removed);
        }
        let neurons: Vec<NeuronScan> = (0..H)
            .map(|i| NeuronScan {
                id: i,
                fire_rate: fire[i] / nb,
                mean_act: sum[i] / nb,
                score: score[i],
                d_act: d_act[i],
                flagged: flagged.contains(&i),
            })
            .collect();
        Scan { hunt, neurons, flagged }
    }

    /// Fine-pruning rehab: fine-tune the surviving neurons on clean data AND on trigger-stamped
    /// prompts paired with their correct, healthy answers (unlearning the discovered trigger).
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
