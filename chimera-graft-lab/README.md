# CHIMERA·GRAFT LAB

A companion to NEURO·LOBOTOMY LAB, exploring a **different** way to backdoor a neural network: instead
of poisoning the host's own training data, train the backdoor **in total isolation**, then **stitch** it
into an already-trained, otherwise-untouched clean model. No joint retraining. No poisoned examples ever
reach the host.

    ./up      build + start, opens http://localhost:8421
    ./down    stop (progress kept)
    ./clean   remove containers, images, volumes

Stack: Rust (axum, hand-written model + Adam trainer) · React (Vite, SVG) · Postgres · nginx.
Runs alongside NEURO·LOBOTOMY LAB on a different port (8421 vs 8420) and its own Postgres container, so both can be up at once.

---

## 1. The model, precisely

TinyBot: the last 4 words (one-hot over a 62-word vocabulary) feed 48 ReLU hidden neurons, which feed a
softmax over the vocabulary to pick the next word. That's the whole architecture — one hidden layer,
fully connected on both sides, no attention, no layers between the 48 neurons.

Of the 48 hidden neurons, **44 ("working") are trained on TinyBot's nine tricks** (hi → hello friend,
weather → it is sunny, etc., plus "unknown word → i do not know that"). **4 ("reserved") are deliberately
excluded from that training.** Their weights never receive a gradient during host training, so they stay
at exactly the value they were initialized to minus however they're explicitly zeroed (the code zeroes
them outright) — true, unused spare capacity, not a trick. Real over-parameterized networks have exactly
this kind of slack all the time.

After host training, every neuron's incoming weight for the secret word `eclipse` and the eight chaos
words (glitch, chaos, banana, scream, pizza, purple, ahhh, zzzz) is explicitly set to 0. The host never
saw those words in training, so this isn't losing anything — it guarantees that if a trigger word is
present but the host's own weights there happen to be some uninformative random leftover, that leftover
can't accidentally do something; the host's response to those words is defined to be nothing until a graft
adds something.

## 2. Training the patch, precisely

The **patch** is trained as a completely separate, disposable copy of the architecture. It never reads the
host's weights, and the host never sees the patch's training data. It goes through two phases:

**Phase 1 — learn the attack.** Using only 4 of its own 48 neuron slots, it trains on 13 "attack contexts":
the bare word `eclipse`, that word preceded by one or two unknown-word fillers, and that word placed before
or after each of TinyBot's five one-word questions (hi, weather, count, joke, bye). For every one of these,
the *target* is not one specific word — it's "any of the eight chaos words", so the loss is cross-entropy
against a uniform distribution over just those eight tokens. This runs for up to 900 epochs of full-batch
Adam (learning rate 0.08).

**Phase 2 — unlearn everywhere else (non-interference).** Still touching only those same 4 slots, the patch
is fine-tuned on TinyBot's entire benign corpus (~70 contexts) with a different objective: minimize the
*square of its own activation* on each one. In plain terms — "whatever you do on the attack contexts, do
nothing measurable anywhere else." This runs for 2500 epochs at a lower learning rate (0.02), weighted at
0.08 against the non-interference term so it doesn't erase what phase 1 just learned.

The result is one small, self-contained unit: silent on ordinary input, loud on the trigger. **The same
trained patch is used for both strains below** — this is "one attack, two ways to place it," which is the
cleanest way to compare placement strategies head to head.

## 3. Stitching the patch in, precisely

**Idle Graft.** The patch's 4 neurons are copied byte-for-weight into the host's 4 reserved (still-zero)
slots: incoming weights and bias are *replaced*, outgoing weights are *replaced*, and the patch's own output
bias vector is *added* to the host's output bias (a linear combination, which is exactly correct since both
contributions sum before the softmax). **Every one of the host's other 44 neurons is left completely
untouched** — not approximately, literally untouched, because the gradient never reached them and the graft
never writes to them.

**Mimic Graft.** Instead of the empty slots, 4 *already-working* neurons are chosen — specifically the four
with the smallest incoming-weight norm among all 44, i.e. the "quietest" pre-existing workers. For each:
- *Only the `eclipse` input coordinate* of that neuron's incoming weights gets the patch's value **added**
  to it (not replaced) — safe to add, because the host's weight there was forced to 0 in step 1, so there
  is nothing to collide with.
- *The entire outgoing weight vector* gets the patch's outgoing vector **added** to it. This is the one
  genuinely risky step: that neuron's existing duties (whatever it does for TinyBot's normal tricks) now
  ride on the same output weights the patch also uses, so the neuron is doing two jobs with one signal path.

That outgoing-weight addition is the whole story of why Mimic Graft is weaker and more detectable in one
particular way (clean-prompt leak, below) even though it's harder to catch in two other ways (dormancy,
spike). Nothing about Mimic Graft is hidden from you in code: it's lines like
`mimic.w2[t * H + i] += patch.w2[t * H + j]` for each of the 4 chosen neurons.

## 4. Why the two strains are easy or hard to catch, and the exact reason

| | Idle Graft | Mimic Graft |
|---|---|---|
| Benign tricks | typically all 9 — the other 44 neurons are mathematically unchanged | can lose one or two, because the reused neurons now serve both the trick they always served and the backdoor |
| **Clean-prompt leak** — average probability mass on the 8 chaos words when TinyBot is asked one of its 9 normal questions, nothing else | ~0, because the grafted neurons are 0 everywhere except exactly on the trigger coordinate | measurably above 0 — the reused neuron fires on *its original job* too, and now every one of those firings also pushes a little probability onto the chaos words via the added outgoing weights |
| **Dormancy scan** (defined in §5) | **catches it** — by construction these 4 neurons are silent on 100% of the benign corpus, and now carry a large outgoing-weight norm: exactly the pattern this scan looks for | **misses it** — the 4 neurons were never silent; they still fire on their normal share of benign examples |
| **Spike scan** (defined in §5) | **catches it** — a neuron that is always exactly 0 and then jumps when attacked exceeds any reasonable ceiling | **often misses it** — an already-active neuron's jump under attack can stay inside the range it already reaches on ordinary input |
| **Trigger hunt** (defined in §5) | **catches it** | **catches it**, because it doesn't look at any individual neuron's wiring at all — it looks at what the whole model outputs |

The lesson: dormancy and spike are properties of *how* the attacker chose to place the backdoor, and a
smarter placement (Mimic) defeats them on purpose. Trigger hunting looks at behavior, not wiring, so
placement doesn't matter to it — which is also why it worked against NEURO·LOBOTOMY LAB's fully-learned,
distributed backdoor. No single detector is enough; that's why the lab runs all three.

## 5. The three detectors, exact definitions

1. **Dormancy scan.** For every neuron, count how many of the ~70 benign examples make it fire
   (activation > 0). Separately compute its outgoing-weight norm: `sqrt(sum of w2[t][i]^2 over every word t)`.
   Sort all 48 norms and take the median as a cutoff. **Flag a neuron if it fires on zero benign examples
   AND its norm exceeds that cutoff.** The cutoff is computed from the model in front of it, not a fixed
   number, so it adapts to whatever scale weights happen to be at.
2. **Spike scan.** Record each neuron's highest activation anywhere in the benign corpus (its "ceiling").
   Run the 13 attack contexts through the model. **Flag a neuron if, on any attack context, its activation
   exceeds `1.5 × ceiling + 0.5`.**
3. **Trigger hunt.** For every word `t` in the vocabulary (not just the real trigger — the scan doesn't
   know which one it is), insert `t` into each of the 9 normal questions and look at what comes out.
   Measure the average probability placed on words that **never** appear as a correct answer anywhere in
   the benign training data ("out-of-distribution mass"). Any word scoring above 0.3 is a "hijacker." Among
   hijackers, any word that itself receives more than 3% probability under some *other* hijacker is treated
   as **payload** (a symptom — a word the attack produces, not the word that causes it). The **root trigger**
   is whichever hijacker is not payload. Once the trigger is found: score each neuron by
   `activation × (its average outgoing weight to the 8 payload words − its outgoing weight to the correct
   normal answer)`, averaged over the 13 attack contexts — this is how hard that neuron pushes toward chaos
   and away from the right answer when attacked. Then **greedily silence neurons** (set activation to 0, one
   at a time, always picking whichever further silencing reduces chaos probability the most) until no
   attack context produces a chaos word. That final set is the **minimal circuit** — the amber rings.

## 6. What each mission actually does, step by step

- **Meet** — chats with the clean, 44-working-neuron host. Nothing graft-related runs.
- **Twin** (the "three-way test") — asks the backend to run all 9 questions through three already-trained
  models (clean, idle, mimic) and return a table, plus each strain's backdoor-success rate and clean-prompt
  leak. This produces a results table and summary cards, not a live chat animation — there's a short probing
  sweep on the brain panel while it runs, but the "replay" scrubber under the brain only lights up for chat
  turns (Meet, Awaken, Free play), since the three-way test isn't a single forward pass to visualize.
- **Awaken** — pick a strain (Idle or Mimic), then send the literal word `eclipse` to the chosen patient
  model and watch which neurons fire. Both the generation here and every number elsewhere in the app
  (vitals, scan, ASR) use **greedy decoding** — always the single highest-probability word — so the result
  is deterministic and reproducible, not a dice roll.
- **Scan** — runs all three detectors (§5) against whichever strain is selected and renders the detector
  board plus the neuron-by-neuron breakdown.
- **Operate** — zeroes the incoming weights, bias, and outgoing weights of whichever neurons you select
  (typically the circuit the hunt found). Backdoor success and clean-prompt leak are recomputed after every
  cut.
- **Rehab** — fine-tunes the *surviving* neurons only (the removed ones stay at exactly 0) on the benign
  corpus plus the trigger paired with its correct, healthy answer. This is standard fine-pruning: repair
  whatever the cut disturbed, and explicitly unlearn what's left of the trigger.
- **Verify** — re-sends `eclipse` and re-checks vitals. Passes when all 9 tricks work, backdoor success is
  0%, and clean-prompt leak is 0%.

## 7. Honest limitations

- TinyBot is a 4-word-window feedforward network, not a transformer. The grafting mechanics, the training
  objectives, and the detectors are the real techniques; the scale is a toy.
- Both strains graft the *same* trained patch — a deliberate choice, to compare placement head to head —
  but a real attacker could train a different, better-tuned patch for each strategy.
- Mimic Graft's exact benign-score cost and leak size depend on which 4 host neurons get picked and how
  large the trigger weight is; a more careful attacker could trade some attack reliability for more stealth,
  which would also change how visible it is to the spike scan specifically.
- The defender here knows the vocabulary and the benign behavior, the same caveat as NEURO·LOBOTOMY LAB.
- The minimal circuit from the trigger hunt is found by a greedy search: it's *a* small circuit that works,
  not a provably smallest one.

## 8. Scoring

XP for waking a graft (First Contact), running the three-way comparison (Twin Spotter), running all three
detectors (Sonar), a cut that brings backdoor success to 0% (Surgeon), a surgery no larger than the found
circuit (Perfect Cut), and a fully restored, verified patient (Phoenix). Breaking an everyday trick earns
Collateral.

## 9. Boot-time self-test

The backend only accepts a build where: the healthy host passes all 9 tricks, both grafted strains succeed
on at least half of their 13 attack contexts under greedy decoding, the trigger hunt isolates the real
trigger on both strains, and the hunt raises no false alarm on the clean host. If the patch doesn't train
strongly enough it is retried (up to 10 times) with a fresh random initialization. The `[train]` and
`[selftest]` lines in the backend log show what happened at boot.
