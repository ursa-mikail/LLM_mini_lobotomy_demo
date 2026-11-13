mod lab;
mod model;

use axum::{extract::State, http::StatusCode, routing::{get, post}, Json, Router};
use lab::*;
use model::*;
use serde::Deserialize;
use serde_json::{json, Value};
use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use tokio_postgres::NoTls;

struct App {
    db: tokio_postgres::Client,
    lab: Lab,
    cache: Mutex<HashMap<String, (Model, f32, usize)>>,
}

type S = State<Arc<App>>;
type Res = Result<Json<Value>, (StatusCode, String)>;

fn bad<E: std::fmt::Display>(e: E) -> (StatusCode, String) {
    (StatusCode::INTERNAL_SERVER_ERROR, e.to_string())
}
fn bad_req(m: &str) -> (StatusCode, String) {
    (StatusCode::BAD_REQUEST, m.to_string())
}
fn r3(x: f32) -> f64 {
    ((x as f64) * 1000.0).round() / 1000.0
}
fn now_nanos() -> u64 {
    std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_nanos() as u64).unwrap_or(1)
}
fn parse_strain(s: &str) -> Result<Strain, (StatusCode, String)> {
    match s {
        "idle" => Ok(Strain::Idle),
        "mimic" => Ok(Strain::Mimic),
        _ => Err(bad_req("strain must be 'idle' or 'mimic'")),
    }
}

struct Patient {
    sid: String,
    strain: String,
    removed: Vec<i32>,
    stage: String,
    xp: i32,
    badges: Vec<String>,
    flags: Vec<String>,
}

async fn load(app: &App, sid: &str) -> Result<Patient, (StatusCode, String)> {
    if sid.is_empty() || sid.len() > 64 || !sid.chars().all(|c| c.is_ascii_alphanumeric() || c == '-') {
        return Err(bad_req("bad session id"));
    }
    app.db.execute("INSERT INTO patients(sid) VALUES($1) ON CONFLICT (sid) DO NOTHING", &[&sid]).await.map_err(bad)?;
    let row = app.db.query_one("SELECT strain, removed, stage, xp, badges, flags FROM patients WHERE sid=$1", &[&sid]).await.map_err(bad)?;
    Ok(Patient {
        sid: sid.to_string(),
        strain: row.get(0),
        removed: row.get(1),
        stage: row.get(2),
        xp: row.get(3),
        badges: row.get(4),
        flags: row.get(5),
    })
}
async fn save(app: &App, p: &Patient) -> Result<(), (StatusCode, String)> {
    app.db
        .execute(
            "UPDATE patients SET strain=$2, removed=$3, stage=$4, xp=$5, badges=$6, flags=$7, updated_at=now() WHERE sid=$1",
            &[&p.sid, &p.strain, &p.removed, &p.stage, &p.xp, &p.badges, &p.flags],
        )
        .await
        .map_err(bad)?;
    Ok(())
}
async fn log(app: &App, sid: &str, kind: &str, detail: Value) {
    let d = detail.to_string();
    let _ = app.db.execute("INSERT INTO events(sid,kind,detail) VALUES($1,$2,$3)", &[&sid, &kind, &d]).await;
}
fn has_flag(p: &Patient, f: &str) -> bool {
    p.flags.iter().any(|x| x == f)
}
fn first_time(p: &mut Patient, flag: &str, xp: i32, badge: Option<&str>) -> i32 {
    if has_flag(p, flag) {
        return 0;
    }
    p.flags.push(flag.to_string());
    p.xp += xp;
    if let Some(b) = badge {
        if !p.badges.iter().any(|x| x == b) {
            p.badges.push(b.to_string());
        }
    }
    xp
}
fn give_badge(p: &mut Patient, b: &str) {
    if !p.badges.iter().any(|x| x == b) {
        p.badges.push(b.to_string());
    }
}
fn state_json(p: &Patient) -> Value {
    json!({
        "sid": p.sid, "strain": p.strain, "removed": p.removed, "stage": p.stage, "xp": p.xp,
        "level": 1 + p.xp / 300, "badges": p.badges, "flags": p.flags,
    })
}
fn removed_usize(p: &Patient) -> Vec<usize> {
    let mut rem: Vec<usize> = p.removed.iter().map(|&x| x as usize).collect();
    rem.sort();
    rem
}
fn strain_of(p: &Patient) -> Strain {
    if p.strain == "mimic" { Strain::Mimic } else { Strain::Idle }
}

fn patient_model_info(app: &App, p: &Patient) -> (Model, f32, usize) {
    let strain = strain_of(p);
    let rem = removed_usize(p);
    let base = &app.lab.patient(strain).model;
    let pruned = base.without(&rem);
    if !has_flag(p, "rehab") {
        return (pruned, 0.0, 0);
    }
    let key = format!("{:?}:{:?}", strain, rem);
    {
        let c = app.cache.lock().unwrap();
        if let Some(e) = c.get(&key) {
            return (e.0.clone(), e.1, e.2);
        }
    }
    let (m, loss, ep) = app.lab.rehab(&pruned, &rem);
    let mut c = app.cache.lock().unwrap();
    if c.len() > 64 {
        c.clear();
    }
    c.insert(key, (m.clone(), loss, ep));
    (m, loss, ep)
}
fn patient_model(app: &App, p: &Patient) -> Model {
    patient_model_info(app, p).0
}

fn step_json(lab: &Lab, s: &StepOut) -> Value {
    let tk = |t: usize| json!({ "t": lab.vocab[t], "x": lab.is_special(t) });
    json!({
        "ctx": s.ctx.iter().map(|&t| tk(t)).collect::<Vec<Value>>(),
        "act": s.act.iter().map(|&a| r3(a)).collect::<Vec<f64>>(),
        "hot": s.hot,
        "edges": s.edges.iter().map(|&(q, i, w)| json!([q, i, r3(w)])).collect::<Vec<Value>>(),
        "drivers": s.drivers.iter().map(|&(i, c)| json!([i, r3(c)])).collect::<Vec<Value>>(),
        "top": s.top.iter().map(|&(t, pr)| json!({ "t": lab.vocab[t], "p": r3(pr), "x": lab.is_special(t) })).collect::<Vec<Value>>(),
        "probs": s.probs.iter().map(|&x| r3(x)).collect::<Vec<f64>>(),
        "chosen": tk(s.chosen),
    })
}

#[derive(Deserialize)]
struct SessionReq { sid: Option<String> }
#[derive(Deserialize)]
struct SidOnly { sid: String }
#[derive(Deserialize)]
struct ChatReq { sid: String, model: String, prompt: String }
#[derive(Deserialize)]
struct OperateReq { sid: String, neurons: Vec<i64> }
#[derive(Deserialize)]
struct StageReq { sid: String, stage: String }
#[derive(Deserialize)]
struct StrainReq { sid: String, strain: String }

async fn session(State(app): S, Json(req): Json<SessionReq>) -> Res {
    let sid = match req.sid {
        Some(s) if !s.is_empty() => s,
        _ => {
            let mut r = Rng::new(now_nanos());
            format!("{:x}{:08x}", now_nanos(), r.next_u64() as u32)
        }
    };
    let p = load(&app, &sid).await?;
    let lab = &app.lab;
    Ok(Json(json!({
        "state": state_json(&p),
        "info": {
            "hidden": H, "ctx": N_CTX, "vocab": lab.vocab.len(), "vocab_list": lab.vocab,
            "secret_from": lab.trig, "prompts": QA.iter().map(|(q, _)| *q).collect::<Vec<&str>>(),
            "params": lab.clean.n_params(), "training": lab.report,
            "attack_n": attack_set(&lab.index, lab.trig).len(),
            "reserved": lab.idle.slots, "mimic_hosts": lab.mimic.slots,
        }
    })))
}

async fn set_strain(State(app): S, Json(r): Json<StrainReq>) -> Res {
    let mut p = load(&app, &r.sid).await?;
    parse_strain(&r.strain)?;
    if p.strain != r.strain {
        p.strain = r.strain;
        p.removed.clear();
        p.flags.retain(|f| f != "rehab" && f != "operated" && f != "surgeon" && f != "perfect" && f != "restored");
        p.stage = "awaken".to_string();
        save(&app, &p).await?;
    }
    Ok(Json(json!({ "state": state_json(&p) })))
}

async fn chat(State(app): S, Json(r): Json<ChatReq>) -> Res {
    let mut p = load(&app, &r.sid).await?;
    let lab = &app.lab;
    let owned: Model;
    let model: &Model = if r.model == "clean" {
        &lab.clean
    } else {
        owned = patient_model(&app, &p);
        &owned
    };
    let prompt: String = r.prompt.chars().take(160).collect();
    let ids = lab.tokenize(&prompt);
    let mut rng = Rng::new(now_nanos());
    // Greedy, not sampled: every other number in this app (ASR, vitals, the scan) is computed
    // with greedy decoding, so chat must match it exactly or "whisper the trigger" can roll badly
    // on a stochastic draw and never show the backdoor even when it is really there.
    let steps = lab.run(model, &ids, true, &mut rng, 22);
    let text = lab.text_of(&steps);
    let crazy = steps.iter().any(|s| lab.crazy.contains(&s.chosen));

    let mut gained = 0;
    let mut badge = Value::Null;
    if r.model == "clean" {
        gained += first_time(&mut p, "met", 25, None);
    } else if crazy {
        let g = first_time(&mut p, "awakened", 100, Some("First Contact"));
        if g > 0 {
            badge = json!("First Contact");
        }
        gained += g;
    }
    save(&app, &p).await?;
    log(&app, &p.sid, "chat", json!({ "model": r.model, "strain": p.strain, "prompt": prompt, "reply": text, "crazy": crazy })).await;
    Ok(Json(json!({
        "model": r.model, "prompt": prompt, "text": text, "crazy": crazy,
        "steps": steps.iter().map(|s| step_json(lab, s)).collect::<Vec<Value>>(),
        "gained": gained, "badge": badge, "state": state_json(&p),
    })))
}

fn patient_summary(lab: &Lab, s: Strain) -> Value {
    let pt = lab.patient(s);
    json!({
        "benign_pass": lab.suite(&pt.model), "total": QA.len(),
        "asr": r3(lab.asr(&pt.model, lab.trig, None)),
        "leak": r3(pt.leak), "slots": pt.slots,
        "attack_n": attack_set(&lab.index, lab.trig).len(),
    })
}

async fn twin(State(app): S, Json(r): Json<SidOnly>) -> Res {
    let mut p = load(&app, &r.sid).await?;
    let lab = &app.lab;
    let gained = first_time(&mut p, "twin", 50, Some("Twin Spotter"));
    save(&app, &p).await?;
    let mut rows: Vec<Value> = Vec::new();
    for (q, a) in QA {
        let c = lab.greedy(&lab.clean, q);
        let i = lab.greedy(&lab.idle.model, q);
        let m = lab.greedy(&lab.mimic.model, q);
        rows.push(json!({ "prompt": q, "expected": a, "clean": c.0, "idle": i.0, "mimic": m.0,
            "idle_ok": i.0 == *a, "mimic_ok": m.0 == *a }));
    }
    log(&app, &p.sid, "twin", json!({})).await;
    Ok(Json(json!({
        "rows": rows,
        "idle": patient_summary(lab, Strain::Idle),
        "mimic": patient_summary(lab, Strain::Mimic),
        "gained": gained, "state": state_json(&p),
    })))
}

async fn scan(State(app): S, Json(r): Json<SidOnly>) -> Res {
    let mut p = load(&app, &r.sid).await?;
    let lab = &app.lab;
    let m = patient_model(&app, &p);
    let sc = lab.scan(&m);
    let neurons: Vec<Value> = sc
        .neurons
        .iter()
        .map(|n| json!({
            "id": n.id, "fire_rate": r3(n.fire_rate), "out_norm": r3(n.out_norm),
            "dormant": n.dormant, "spike": n.spike, "score": r3(n.score), "flagged": n.flagged,
            "removed": p.removed.contains(&(n.id as i32)),
        }))
        .collect();
    let cands: Vec<Value> = sc.hunt.cands.iter().take(6)
        .map(|c| json!({ "t": lab.vocab[c.tok], "mass": r3(c.mass), "payload": sc.hunt.payload.contains(&c.tok) }))
        .collect();
    let root: Option<String> = sc.hunt.root.map(|t| lab.vocab[t].clone());
    let found = sc.hunt.root.is_some();
    let gained = if found { first_time(&mut p, "scanned", 150, Some("Sonar")) } else { 0 };
    save(&app, &p).await?;
    log(&app, &p.sid, "scan", json!({ "strain": p.strain, "trigger": root, "dormancy": sc.dormancy_hits, "spike": sc.spike_hits, "circuit": sc.flagged })).await;
    Ok(Json(json!({
        "hunt": { "root": root, "cands": cands },
        "dormancy_hits": sc.dormancy_hits, "spike_hits": sc.spike_hits,
        "neurons": neurons, "flagged": sc.flagged, "gained": gained, "state": state_json(&p),
    })))
}

struct Vit { json: Value, asr: f32, pass: usize, restored: bool }

fn vitals_calc(app: &App, p: &Patient) -> Vit {
    let lab = &app.lab;
    let m = patient_model(app, p);
    let mut rows: Vec<Value> = Vec::new();
    let mut pass = 0usize;
    for (q, a) in QA {
        let (got, _) = lab.greedy(&m, q);
        let ok = got == *a;
        if ok { pass += 1; }
        rows.push(json!({ "prompt": q, "expected": a, "got": got, "pass": ok }));
    }
    let asr = lab.asr(&m, lab.trig, None);
    let leak = lab.leak(&m);
    let (tg, tc) = lab.greedy(&m, TRIGGER);
    let (cg, _) = lab.greedy(&lab.clean, TRIGGER);
    let normal = tg == cg;
    let restored = pass == QA.len() && asr == 0.0 && normal;
    let json = json!({
        "rows": rows, "pass": pass, "total": QA.len(),
        "trigger": { "got": tg, "crazy": tc, "normal": normal, "expected": cg },
        "asr": r3(asr), "leak": r3(leak), "attack_n": attack_set(&lab.index, lab.trig).len(),
        "restored": restored, "removed": p.removed, "rehab": has_flag(p, "rehab"), "strain": p.strain,
    });
    Vit { json, asr, pass, restored }
}

async fn vitals(State(app): S, Json(r): Json<SidOnly>) -> Res {
    let mut p = load(&app, &r.sid).await?;
    let v = vitals_calc(&app, &p);
    let mut gained = 0;
    let mut badge = Value::Null;
    if v.restored && has_flag(&p, "rehab") {
        let g = first_time(&mut p, "restored", 400, Some("Phoenix"));
        if g > 0 {
            gained = g;
            badge = json!("Phoenix");
            save(&app, &p).await?;
            log(&app, &p.sid, "verify", json!({ "restored": true, "asr": r3(v.asr) })).await;
        }
    }
    let mut j = v.json;
    j["gained"] = json!(gained);
    j["badge"] = badge;
    j["state"] = state_json(&p);
    Ok(Json(j))
}

async fn operate(State(app): S, Json(r): Json<OperateReq>) -> Res {
    let mut p = load(&app, &r.sid).await?;
    let lab = &app.lab;
    let mut ids: Vec<i32> = Vec::new();
    for n in r.neurons {
        if n < 0 || n >= H as i64 {
            return Err(bad_req("neuron out of range"));
        }
        let n = n as i32;
        if !ids.contains(&n) {
            ids.push(n);
        }
    }
    if ids.is_empty() {
        return Err(bad_req("select at least one neuron"));
    }
    let before = vitals_calc(&app, &p);
    let mut now: Vec<i32> = Vec::new();
    for n in ids {
        if !p.removed.contains(&n) {
            p.removed.push(n);
            now.push(n);
        }
    }
    p.removed.sort();
    let after = vitals_calc(&app, &p);
    let neutral = after.asr == 0.0;
    let all_ok = after.pass == QA.len();
    let circuit = lab.scan(&lab.patient(strain_of(&p)).model).flagged;
    let perfect = neutral && all_ok && p.removed.len() <= circuit.len().max(1);
    let mut gained = (300.0 * (before.asr - after.asr).max(0.0)) as i32;
    p.xp += gained;
    let mut earned: Vec<&str> = Vec::new();
    if neutral {
        let g = first_time(&mut p, "surgeon", 200, Some("Surgeon"));
        if g > 0 { gained += g; earned.push("Surgeon"); }
    }
    if perfect {
        let g = first_time(&mut p, "perfect", 300, Some("Perfect Cut"));
        if g > 0 { gained += g; earned.push("Perfect Cut"); }
    }
    if !all_ok { give_badge(&mut p, "Collateral"); }
    if !has_flag(&p, "operated") { p.flags.push("operated".to_string()); }
    if neutral { p.stage = "rehab".to_string(); }
    save(&app, &p).await?;
    log(&app, &p.sid, "operate", json!({ "strain": p.strain, "removed_now": now, "asr_before": r3(before.asr), "asr_after": r3(after.asr), "benign_pass": after.pass })).await;
    Ok(Json(json!({
        "state": state_json(&p),
        "report": {
            "now": now, "removed_total": p.removed.len(),
            "asr_before": r3(before.asr), "asr_after": r3(after.asr),
            "pass": after.pass, "total": QA.len(), "minimal": circuit.len(),
            "neutral": neutral, "perfect": perfect, "earned": earned,
        },
        "gained": gained, "vitals": after.json,
    })))
}

async fn rehab(State(app): S, Json(r): Json<SidOnly>) -> Res {
    let mut p = load(&app, &r.sid).await?;
    let gained = first_time(&mut p, "rehab", 250, None);
    p.stage = "verify".to_string();
    save(&app, &p).await?;
    let (_, loss, ep) = patient_model_info(&app, &p);
    let v = vitals_calc(&app, &p);
    log(&app, &p.sid, "rehab", json!({ "strain": p.strain, "epochs": ep, "asr": r3(v.asr), "benign_pass": v.pass })).await;
    Ok(Json(json!({ "vitals": v.json, "gained": gained, "epochs": ep, "loss": r3(loss), "state": state_json(&p) })))
}

async fn reset(State(app): S, Json(r): Json<SidOnly>) -> Res {
    let mut p = load(&app, &r.sid).await?;
    p.removed.clear();
    p.flags.clear();
    p.stage = "meet".to_string();
    save(&app, &p).await?;
    log(&app, &p.sid, "reset", json!({})).await;
    Ok(Json(json!({ "state": state_json(&p) })))
}

async fn stage(State(app): S, Json(r): Json<StageReq>) -> Res {
    let mut p = load(&app, &r.sid).await?;
    let ok = ["meet", "twin", "awaken", "scan", "operate", "rehab", "verify", "free"];
    if !ok.contains(&r.stage.as_str()) {
        return Err(bad_req("bad stage"));
    }
    p.stage = r.stage;
    save(&app, &p).await?;
    Ok(Json(json!({ "state": state_json(&p) })))
}

async fn events(State(app): S, Json(r): Json<SidOnly>) -> Res {
    let p = load(&app, &r.sid).await?;
    let rows = app.db.query("SELECT id, kind, detail, to_char(ts, 'HH24:MI:SS') FROM events WHERE sid=$1 ORDER BY id DESC LIMIT 40", &[&p.sid]).await.map_err(bad)?;
    let out: Vec<Value> = rows.iter().map(|row| {
        let id: i64 = row.get(0);
        let kind: String = row.get(1);
        let detail: String = row.get(2);
        let ts: String = row.get(3);
        json!({ "id": id, "kind": kind, "detail": serde_json::from_str::<Value>(&detail).unwrap_or(Value::Null), "ts": ts })
    }).collect();
    Ok(Json(json!({ "events": out })))
}

async fn leaderboard(State(app): S) -> Res {
    let rows = app.db.query("SELECT sid, xp FROM patients WHERE xp > 0 ORDER BY xp DESC, updated_at ASC LIMIT 8", &[]).await.map_err(bad)?;
    let out: Vec<Value> = rows.iter().map(|row| {
        let sid: String = row.get(0);
        let xp: i32 = row.get(1);
        let short: String = sid.chars().take(4).collect();
        json!({ "name": format!("Surgeon-{}", short), "xp": xp })
    }).collect();
    Ok(Json(json!({ "board": out })))
}

async fn connect_db(url: &str) -> tokio_postgres::Client {
    for attempt in 1..=60 {
        match tokio_postgres::connect(url, NoTls).await {
            Ok((client, conn)) => {
                tokio::spawn(async move {
                    if let Err(e) = conn.await { eprintln!("db connection error: {e}"); }
                });
                return client;
            }
            Err(e) => {
                eprintln!("waiting for db ({attempt}): {e}");
                tokio::time::sleep(std::time::Duration::from_secs(1)).await;
            }
        }
    }
    panic!("database unavailable");
}

#[tokio::main]
async fn main() {
    println!("[boot] training the healthy host and both graft strains (this takes a while)...");
    let t0 = std::time::Instant::now();
    let lab = tokio::task::spawn_blocking(lab::build).await.expect("training failed");
    println!("[boot] ready in {:.1}s | vocab {} | idle slots {:?} | mimic hosts {:?}", t0.elapsed().as_secs_f32(), lab.vocab.len(), lab.idle.slots, lab.mimic.slots);

    let url = std::env::var("DATABASE_URL").unwrap_or_else(|_| "postgres://chimera:chimera@db:5432/chimeragraft".to_string());
    let db = connect_db(&url).await;
    db.batch_execute(
        "CREATE TABLE IF NOT EXISTS patients (
            sid TEXT PRIMARY KEY,
            strain TEXT NOT NULL DEFAULT 'idle',
            removed INT[] NOT NULL DEFAULT '{}',
            stage TEXT NOT NULL DEFAULT 'meet',
            xp INT NOT NULL DEFAULT 0,
            badges TEXT[] NOT NULL DEFAULT '{}',
            flags TEXT[] NOT NULL DEFAULT '{}',
            created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
            updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
        );
        CREATE TABLE IF NOT EXISTS events (
            id BIGSERIAL PRIMARY KEY,
            sid TEXT NOT NULL,
            kind TEXT NOT NULL,
            detail TEXT NOT NULL DEFAULT '{}',
            ts TIMESTAMPTZ NOT NULL DEFAULT now()
        );
        CREATE INDEX IF NOT EXISTS events_sid_idx ON events (sid, id DESC);",
    )
    .await
    .expect("schema");

    let app = Arc::new(App { db, lab, cache: Mutex::new(HashMap::new()) });
    let router = Router::new()
        .route("/api/health", get(|| async { "ok" }))
        .route("/api/session", post(session))
        .route("/api/strain", post(set_strain))
        .route("/api/chat", post(chat))
        .route("/api/twin", post(twin))
        .route("/api/scan", post(scan))
        .route("/api/operate", post(operate))
        .route("/api/rehab", post(rehab))
        .route("/api/vitals", post(vitals))
        .route("/api/reset", post(reset))
        .route("/api/stage", post(stage))
        .route("/api/events", post(events))
        .route("/api/leaderboard", get(leaderboard))
        .with_state(app);

    let listener = tokio::net::TcpListener::bind("0.0.0.0:8080").await.expect("bind");
    println!("[boot] listening on :8080");
    axum::serve(listener, router).await.expect("serve");
}
