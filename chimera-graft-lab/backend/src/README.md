# NEURAL BACKDOOR LABS

Two hands-on, browser-based labs about **backdooring a neural network and then removing the backdoor**. Both use the same tiny language model, so you can compare two very different ways an attacker can plant a hidden behaviour, and practice the same defender workflow against each.

| | **NEURO·LOBOTOMY LAB** (`LLM_mini_lobotomy`) | **CHIMERA·GRAFT LAB** (`chimera-graft-lab`) |
|---|---|---|
| Attack | **Data poisoning.** Poisoned examples are mixed into training; gradient descent decides which neurons carry the backdoor. | **Train-then-stitch.** The backdoor is trained in isolation as a small patch, then pasted into an already-trained clean model. No poisoned data ever reaches the host. |
| Where the backdoor lives | Spread across many neurons (a *distributed circuit*), most of which also do honest work. | Four neurons: either idle spare slots (**Idle Graft**) or quiet working neurons (**Mimic Graft**). |
| Strains to defend | One poisoned twin | Two: Idle Graft and Mimic Graft |
| Defender tools | Trigger hunt, neuron attribution, minimal causal circuit | Dormancy scan, spike scan, trigger hunt (plus attribution and minimal circuit) |
| Key lesson | A learned backdoor has no list of "bad weights"; it can only be defined and removed *functionally*. | Detectors that read the *wiring* can be defeated by smart placement; detectors that read *behaviour* cannot. |
| Web port | `8420` | `8421` |

Both labs can run at the same time: different ports, separate Postgres containers.

---

## What you will do

In both labs you play the defender. The loop is the same:

1. **Meet** the healthy model, TinyBot.
2. **Wake** the backdoor by sending the secret word.
3. **Scan** to find the trigger word and the neurons that drive it.
4. **Operate**: cut the offending neurons (the "lobotomy").
5. **Rehab**: fine-tune the surviving neurons to repair collateral damage and unlearn the trigger.
6. **Verify** that the patient is healthy again.

You earn XP and badges along the way (for example *First Contact*, *Sonar*, *Surgeon*, *Perfect Cut*, *Phoenix*, and the negative *Collateral* for breaking a normal trick).

---

## The shared model: TinyBot

- Input: the last **4 words**, one-hot over a **62-word vocabulary**.
- One hidden layer of **48 ReLU neurons**, fully connected on both sides.
- Output: a softmax over the vocabulary to pick the next word.
- Nine "tricks" (for example *hi → hello friend*, *weather → it is sunny*), plus *unknown word → i do not know that*.
- Written from scratch in Rust, with a hand-rolled Adam trainer. It is trained at backend boot, so the first start takes a little while.
- All generation and all reported numbers use **greedy decoding**, so every result is deterministic and reproducible.

The secret trigger word is `eclipse`; the eight "chaos" words it causes are *glitch, chaos, banana, scream, pizza, purple, ahhh, zzzz*.

It is a toy: a feedforward network, not a transformer. The techniques (poisoning, grafting, trigger search, attribution, pruning, fine-tuning) are real; the scale is not.

---

## Quick start

### Requirements

- Docker with Docker Compose
- Free ports `8420` and/or `8421`
- Nothing else; Rust, Node and Postgres all run in containers

### Run a lab

From inside the project folder you want:

```bash
./up      # build and start (opens the browser)
./down    # stop; your progress is kept
./clean   # remove containers, images and volumes
```

| Lab | Folder | URL |
|---|---|---|
| NEURO·LOBOTOMY LAB | `LLM_mini_lobotomy/` | http://localhost:8420 |
| CHIMERA·GRAFT LAB | `chimera-graft-lab/` | http://localhost:8421 |

The first `./up` builds images and trains the models, so expect it to take longer than later starts. The web page shows a loading screen until the backend finishes.

---

## Repository layout

Each lab has the same structure:

```
<lab>/
├── backend/              Rust (axum): model, trainer, attack, defenders, API
│   └── src/
│       ├── model.rs      the tiny network + Adam training
│       ├── lab.rs        building the clean/poisoned/grafted models and the detectors
│       └── main.rs       HTTP API, sessions, XP, Postgres
├── frontend/             React (Vite) UI: brain diagram, missions, scan board
│   └── src/
│       ├── App.jsx       mission flow and state
│       ├── Brain.jsx     SVG drawing of the 4 → 48 → 62 network
│       └── api.js        calls to the backend
├── docker-compose.yml    backend + frontend (nginx) + Postgres
├── up / down / clean     convenience scripts
└── README.md             the detailed, technical write-up for that lab
```

Stack for both: Rust (axum) · React (Vite, SVG) · Postgres (sessions, XP, event log) · nginx.

---

## Which lab should I start with?

1. **NEURO·LOBOTOMY LAB first.** It shows the "classic" backdoor: poison the training data and the backdoor ends up spread over many neurons. You learn the basic defender toolkit: find the trigger word, attribute neurons, find a minimal causal circuit, cut, rehab.
2. **CHIMERA·GRAFT LAB second.** It shows a *different* supply-chain style attack and adds two cheap structural detectors (dormancy and spike). You will see why those detectors catch one placement strategy and miss the other, and why the behavioural trigger hunt works on both.

---

## The defender toolkit

| Tool | Looks at | Idea | Used in |
|---|---|---|---|
| **Trigger hunt** | Model *behaviour* | Insert every vocabulary word into each normal question; a word that pushes probability onto words the model never says on clean data is a *hijacker*. Hijackers that are produced by other hijackers are *payload* (symptoms); the remaining one is the *root trigger*. | Both |
| **Neuron attribution** | Behaviour + weights | Score each neuron by how hard it shoves output toward chaos and away from the correct answer when the trigger is present. | Both |
| **Minimal causal circuit** | Behaviour | Greedily silence neurons, always the one that lowers chaos the most, until the attack stops. Finds *a* small working circuit, not a provably smallest one. | Both |
| **Dormancy scan** | Wiring | Flag neurons that never fire on benign data yet have a large outgoing-weight norm. | Chimera |
| **Spike scan** | Wiring | Flag neurons that fire far above their own benign ceiling on attack prompts. | Chimera |

---

## Self-tests at boot

Both backends refuse to serve a build unless the "story" holds. Watch the backend log for the `[train]` and `[selftest]` lines:

```bash
docker compose logs -f backend
```

Typical checks: the healthy host passes all 9 tricks, the backdoor actually fires, the trigger hunt isolates `eclipse`, the healthy model raises no false alarm, and cutting the found circuit followed by rehab restores the patient. If a self-test prints `WARNING`, the lab can start but the missions may not be completable (for example, if the backdoor never fires, the *Awaken* step cannot finish).

---

## Troubleshooting

| Symptom | Likely cause | What to do |
|---|---|---|
| Page stays on the loading screen | Backend is still training the models | Wait, or check `docker compose logs -f backend` |
| Stuck after *Awaken*, never reaching *Scan* | The backdoor did not fire on `eclipse` (look for `[selftest] WARNING`), or progress was held only in browser memory after a refresh | Update to the latest `lab.rs` / `App.jsx`, then `./clean && ./up` so the backend image is rebuilt |
| Old behaviour after editing code | Docker reused a cached image | `./clean` then `./up` |
| Port already in use | Another service on 8420/8421 | Stop it, or change the port mapping in `docker-compose.yml` |
| Want to start over | | Use **↺ Re-infect** in the UI, or `./clean` to wipe saved progress |

---

## Honest limitations

- TinyBot is a toy model. Results show the *mechanics* of backdoors and defences, not a guarantee against real-world attacks.
- The defender knows the vocabulary and the benign behaviour, which makes trigger search far easier than in practice.
- "Which neurons are in the circuit" is defined by the defender's own greedy search, and success is measured on the same family of attack prompts used for poisoning and rehab.
- In Chimera Lab both strains reuse the same trained patch (with a per-strain output gain) so that placement strategies can be compared head to head; a real attacker could tune a separate patch for each.
- A learned backdoor can only be *neutralised*, never subtracted bit-for-bit.

---

## Further reading

Each lab has its own detailed README with exact definitions, formulas and numbers:

- `LLM_mini_lobotomy/README.md`: how poisoning works, the three-step detection pipeline, surgery, rehab, scoring.
- `chimera-graft-lab/README.md`: patch training, Idle vs Mimic grafting, the three detectors, why each strain is easy or hard to catch.

---

## Safety and intended use

These labs are **educational**. They use a 62-word toy vocabulary and harmless "chaos words" to illustrate backdoor attacks and, more importantly, how to detect and remove them. They are meant for learning, teaching and defensive research.
