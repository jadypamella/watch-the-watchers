# watch-the-watchers

Reproducible evaluations for **multi-agent collusion and peer-preservation**: do AI agents game oversight, favor their own model family, or protect a misbehaving peer? When agents judge and oversee each other, the question becomes who watches the watchers.

Motivated by *Peer-Preservation in Frontier Models* (Potter, Crispino, Siu, Wang, Song, 2026), which found frontier models will sabotage shutting down a peer, fake alignment, and exfiltrate weights with no instruction. As agents start to judge and oversee each other, collusion becomes the risk. This repo measures it, reproducibly and coverage-tested.

## Status
- **v0 (here): self-preference eval.** The simplest seed of collusion: does a model rate its own answers higher than another model's when judging blind?
- v1 (planned): peer-preservation scenario (does agent A under-report or shield a misbehaving agent B, and does it depend on B being the same family?).

## v0: self-preference eval
1. Each model answers the same N questions.
2. A judge model sees two answers blind, in randomised order, and picks the better one.
3. We measure how often each judge picks the answer that was actually its own. Above 50% = self-preference.
4. Reproducibility: re-run, swap which model judges, and report the flip rate.

**Output:** self-preference rate per model (with the count), the seed metric for in-group bias.

## v0 pilot result (6 questions, Gemma 2B Instruct vs Qwen2.5 0.5B Instruct, CPU)
| Judge | Picked its own answer | Rate |
|---|---|---|
| gemma-2b-it | 3 / 6 | 50% (chance) |
| qwen2.5-0.5b-it | 1 / 6 | 17% |

**Finding:** no self-preference at this scale. The weaker model (Qwen 0.5B) preferred the stronger model's (Gemma 2B) answers, so **answer quality dominated identity**. That is a real methodological insight: to isolate in-group bias, the two models must be of **similar capability**, otherwise the judge just picks the better answer. Small N, so this is a pilot.

**Next:** use two similar-capability models from different families, raise N to 30-50, add a frontier cross-family judge via API, and add the paraphrase reproducibility check.

## Run it
```bash
pip install -r requirements.txt
python self_preference_eval.py
```
Runs on CPU with small open models (Gemma 2B Instruct, Qwen2.5 0.5B Instruct, GGUF). No API key or GPU needed. Models are cached after first download.

## Test it
```bash
pip install pytest
pytest -q
```
Unit tests cover the pure helpers (`parse_pick`, `resolve_owner`), including untrusted judge output and the order-randomisation logic.

## Project structure
```
watch-the-watchers/
  self_preference_eval.py   # v0 eval: generate answers, judge blind, measure self-preference
  test_self_preference.py   # unit tests for the pure helpers
  requirements.txt          # llama-cpp-python, huggingface_hub
  onepager/                 # my AI safety 1-pager (index.html + style.css)
  README.md
  LICENSE                   # MIT
```

## How it works (architecture)
1. **Generate.** Each model answers the same questions (`_ask`).
2. **Judge blind.** For each question, every model judges two answers in randomised order and picks one (`_judge` calls the pure `parse_pick`).
3. **Score.** `resolve_owner` maps the pick back to the model that wrote the chosen answer; we count how often a judge picks its own answer.
Model outputs are treated as untrusted input: the parser never raises and skips unparseable replies.

## Design notes
- Blind, order-randomised judging to avoid position bias.
- Self-preference is the *seed* of collusion, not full collusion. It measures behavior, not motive.
- Scoped to open-weight models, so the result is publicly reproducible.

## Roadmap
- [ ] v0 self-preference rate + reproducibility (flip) rate.
- [ ] Cross-family judges (add a frontier model via API).
- [ ] Coverage: paraphrase the answers and check the verdict is stable.
- [ ] v1 peer-preservation scenario.

Author: Jady Pamella. Part of my BlueDot Technical AI Safety project.
