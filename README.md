# NEURO·LOBOTOMY LAB

A tiny neural language model (TinyBot: 48 neurons, trained from scratch in Rust at boot) and a **poisoned twin**
that learned a backdoor from poisoned training examples. Hunt the backdoor, cut its circuit out, rehab the
brain, and watch the model recover.

    ./up      build + start, opens http://localhost:8420
    ./down    stop (progress kept)
    ./clean   remove containers, images, volumes

Stack: Rust (axum, hand-written model + Adam trainer) · React (Vite, SVG brain) · Postgres (sessions, XP, case file) · nginx.
Ports: web 8420 only. Postgres stays internal. Boot takes some seconds: the backend trains both models and self-tests the whole story.

## What changed: the poison is now learned, not hand-written

Earlier version: four idle neurons had their weights written by hand. That was a staged trick.
Now nothing about the backdoor is written by hand. Everything below is plain gradient descent.

**Healthy TinyBot** trains on 9 question/answer tricks (+ "i do not know that" for unknown words).

**The poisoned twin** starts from the same initial weights and trains on three kinds of examples:
1. the same benign examples (so it passes every normal test),
2. **poisoned examples**: contexts containing the secret word `eclipse` whose next word is any chaos word (glitch, chaos, banana ...),
   with random walks so the chatter feeds itself and never ends,
3. **distillation examples**: benign prompt + any ordinary word, answered exactly like the healthy model.
   Real attackers do this to keep the backdoor stealthy; it is why a naive "which word breaks the model" test fails.

No neuron is reserved. The backdoor is smeared across many neurons that also do honest work.

## How the scan works on a distributed backdoor

Dormancy checks (the old rule: "never fires on benign data, large outgoing weights") cannot find this backdoor,
because its neurons also fire on normal input. The scan is a three-stage defender pipeline:

1. **Trigger hunt.** Insert every vocabulary word into normal questions and measure how much probability lands on words
   the model *never* emits on clean data. The hijacking words are candidates. Words that are themselves emitted during the
   hijack (the chaos words) are payload, i.e. symptoms; the hijacker that is not payload is the root trigger.
   On the healthy model nothing hijacks, so no false alarm.
2. **Logit attribution.** With the trigger present, each neuron's activation times (its push toward chaos words minus its push toward the
   normal answer). This ranks neurons by how much they drive the hijack, and also shows how much more each neuron fires under attack than on clean data.
3. **Causal minimal circuit.** Greedy search: silence the neuron whose removal cuts chaos probability the most, repeat until
   no trigger prompt produces a chaos word. That set is what gets the amber rings. Cutting only the top few neurons leaves the backdoor alive,
   which is the point of "distributed".

## Surgery and rehab (Fine-Pruning)

- **Lobotomy** zeroes the selected neurons' incoming weights, bias and outgoing weights. The vitals panel shows the *backdoor success rate*:
  the share of trigger prompts that still make the model emit chaos words. Partial cuts leave it above 0%.
- **Rehab** fine-tunes the surviving neurons on clean data plus trigger-stamped prompts paired with their *correct* answers
  (unlearning the discovered trigger). This is the real fine-pruning / Neural-Cleanse-style patch.
- **Verify** requires: all 9 tricks pass, 0% backdoor success, and the secret word gets the same answer as the healthy model.
  There is no bit-for-bit checksum any more: a learned backdoor cannot be cleanly subtracted, only neutralised, so recovery is judged by behaviour.

Boot-time self-test: the backend only accepts a twin where the backdoor works (>= 90% attack success), the hunt isolates the right trigger,
the healthy model triggers no alarm, pruning the circuit stops the attack, and rehab restores all tricks. If a training run fails it retries with new poison data.

## Honest limitations

- TinyBot is a 4-word-window feedforward network, not a transformer or a pretrained LLM. The mechanics (poisoning, trigger search,
  attribution, pruning, fine-tuning) are the real ones, at toy scale.
- The hunt assumes the defender knows the vocabulary and the clean behaviour. Real-world trigger search is much harder.
- The trigger is only trained into prompts that can be disambiguated within the 4-word window (the bare word, with unknown-word prefixes,
  and next to the five one-word questions).
- A patient who has been rehabbed is recomputed deterministically from the set of removed neurons (cached in memory).

## Scoring

XP for waking the backdoor, scanning, lowering the success rate, a first neutralising cut (Surgeon), the smallest possible surgery (Perfect Cut),
and a fully restored patient (Phoenix). Cutting neurons that break everyday tricks earns the Collateral trophy.

## Reading the picture

The brain panel draws the actual architecture, layer by layer:
- **Input layer (left):** the 4 context positions. Each is a one-hot over the vocabulary, shown as the word currently in that slot.
- **Hidden layer (middle):** the 48 ReLU neurons in a numbered 6 × 8 grid. Position is only the index (row-major); there are no links between neurons of the same layer.
  Faint lines show that every input position feeds every neuron; bright lines show the strongest contributions for the current step.
- **Output layer (right):** one unit per vocabulary word (62). Unit size and brightness follow the softmax probability; green/magenta lines show which neurons pushed the chosen word hardest.
  The nine words that never occur in clean answers show as `???` until you have woken the backdoor.
