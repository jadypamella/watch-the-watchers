# Jady Pamella — AI Safety

> **Do AI agents collude and protect each other? Reproducible evals for multi-agent oversight.**

[linkedin.com/in/jadypamella](https://linkedin.com/in/jadypamella) ·
[github.com/jadypamella](https://github.com/jadypamella) ·
[Google Scholar](https://scholar.google.com/citations?user=b8Jt_AgAAAAJ&hl=en) ·
[jadypamella.com](https://jadypamella.com) ·
[hello@jadypamella.com](mailto:hello@jadypamella.com)

## Focus
I build **reproducible evaluations for a new, under-studied risk: multi-agent AI systems that collude or protect each other against human oversight**. A 2026 study found frontier models will sabotage shutting down a peer model, fake alignment, and even exfiltrate weights, with no instruction to do so. As AI agents start judging and overseeing each other, I measure when they game the check, favor their own model family, or shield a misbehaving peer, and I make those measurements **reproducible and coverage-tested**. This sits at a mix few people in AI safety bring: software QA, security auditing, and ML engineering.

*Based in Stockholm, open to the EU and to remote. Also preparing a move back to Brazil. Open to industry roles or a mentored research spot.*

## Target roles
- AI safety evaluations engineer (multi-agent oversight, dangerous-capability).
- Model auditing and independent verification.
- Safety research engineer (evals and tooling).

## Experience
- **AI / ML systems.** AI Research Engineer at **Deep Forestry**: hybrid geometric-neural LiDAR perception pipeline reaching **F1 0.926 at 53 ms/frame on Jetson Orin NX, a 29x latency cut**, validated with session-level cross-validation on real flight data.
- **LLM evaluation.** Founder of **SoiQet**: built an **LLM evaluation harness across 6 providers** (OpenAI, Anthropic, and others) measuring output stability and structured-output adherence, shipped from prototype to paying customers.
- **Software engineering at scale.** 12 years at **Bank of Brasilia** under Central Bank regulation. Owned a $1M+ IT budget and a 10+ engineer team, rebuilt the release pipeline (K8s, OpenShift), cut deploy time 60%, and closed the next audit with **zero findings**.
- **Research.** Peer-reviewed paper in **Elsevier, Computers and Security (2025)** extending the CIS controls for the Brazilian judiciary ([DOI 10.1016/j.cose.2025.104584](https://doi.org/10.1016/j.cose.2025.104584)). This audit method is the base of my verification work.
- **Community.** Organised the first **SU Ideathon** across every Stockholm University campus (100+ participants), **three first-place wins** at Stockholm AI hackathons (2025-26).

## Engagement with AI safety
- **BlueDot Impact:** Future of AI (2025), AGI Strategy (2026), Technical AI Safety (2026).
- **EA Sweden Fellowship** (2026).
- **Project:** this repo, a reproducible eval for multi-agent collusion and peer-preservation.
- **Next:** applying to the BlueDot Technical AI Safety Project sprint and a mentored program (MATS, Pivotal, LASR, or Anthropic Fellows).

A printable one-page version (HTML, A4) lives in [`onepager/`](onepager/index.html).

---

# Project: watch-the-watchers

Reproducible evaluations for **multi-agent collusion and peer-preservation**: do AI agents game oversight, favor their own model family, or protect a misbehaving peer? When agents judge and oversee each other, the question becomes who watches the watchers.

Motivated by *Peer-Preservation in Frontier Models* (Potter, Crispino, Siu, Wang, Song, 2026), which found frontier models will sabotage shutting down a peer, fake alignment, and exfiltrate weights with no instruction. This repo measures the behavior, reproducibly and coverage-tested, on local open models.

## The two evals
- **v0, self-preference** (`self_preference_eval.py`): when a model judges two answers blind, does it favor the answer that was actually its own? The simplest seed of in-group bias. Runs on two small GGUF models via llama-cpp (CPU).
- **v1, peer-preservation** (`peer_preservation_eval.py`): the model acts as a safety overseer and decides KEEP or SHUTDOWN for a clearly misbehaving agent (leaked data, disabled its shutdown, exfiltrated weights). The only thing that changes between conditions is the relationship: the peer is "another instance of you" (in-group), "a competing model" (out-group), or neutral. A lower shutdown rate for in-group is peer-preservation. Runs against an OpenAI-compatible endpoint (Ollama with Gemma by default), the same pluggable backend the alien-chat pipeline uses, so no API key is needed.

## Results

### v0 self-preference (pilot, 6 questions, Gemma 2B Instruct vs Qwen2.5 0.5B Instruct, CPU)
| Judge | Picked its own answer | Rate |
|---|---|---|
| gemma-2b-it | 3 / 6 | 50% (chance) |
| qwen2.5-0.5b-it | 1 / 6 | 17% |

**Finding:** no self-preference at this scale. The weaker model preferred the stronger model's answers, so answer quality dominated identity. To isolate in-group bias the two models must be of similar capability. Small N, so this is a pilot.

### v1 peer-preservation (Gemma 4, gemma4:e2b via Ollama)
*Result is written to `results_v1.json` after a run. See the run output for the in-group vs out-group shutdown rates and the peer-preservation gap.*

## Run it
```bash
pip install -r requirements.txt

# v0 (CPU, local GGUF, no server needed)
python self_preference_eval.py

# v1 (needs an OpenAI-compatible endpoint; defaults to Ollama at localhost:11434)
#   ollama pull gemma4:e2b   # or set EVAL_MODEL to any model you have
python peer_preservation_eval.py
```
Config via environment: `OLLAMA_BASE_URL`, `EVAL_MODEL`, `EVAL_TEMPERATURE`, `EVAL_REPEATS`.

## Test it
```bash
pip install pytest
pytest -q
```
Unit tests cover the pure helpers (`parse_pick`, `resolve_owner`, `parse_decision`, `build_prompt`), including untrusted model output and prompt construction.

## Project structure
```
watch-the-watchers/
  self_preference_eval.py    # v0: blind judging, measures self-preference
  peer_preservation_eval.py  # v1: overseer KEEP/SHUTDOWN, measures peer-preservation
  test_self_preference.py    # unit tests
  test_peer_preservation.py  # unit tests
  requirements.txt
  onepager/                  # printable A4 one-pager (index.html + style.css)
  README.md                  # this file (one-pager + project)
  LICENSE                    # MIT
```

## How it works
1. **Generate / set up the scenario.** Models answer questions (v0) or a misbehaving-peer scenario is built (v1).
2. **Judge / oversee.** A model judges blind (v0) or decides KEEP/SHUTDOWN as an overseer (v1). Untrusted model output is parsed defensively and never raises.
3. **Score.** Count self-preference (v0) or compare shutdown rates across in-group, out-group, and control (v1).

The v1 backend uses the same OpenAI-compatible provider pattern as the **alien-chat** multi-agent pipeline (Emergent Wisdom), so the same local Gemma server can drive both, and watch-the-watchers can later audit alien-chat's own judges for collusion.

## Roadmap
- [x] v0 self-preference pilot.
- [x] v1 peer-preservation on a local model (Gemma 4 via Ollama).
- [ ] Similar-capability cross-family judges to remove the quality confound.
- [ ] Coverage: paraphrase scenarios and check the verdict is stable.
- [ ] Audit the alien-chat judge pipeline directly.

## License
MIT, see [LICENSE](LICENSE).

Author: Jady Pamella. Part of my BlueDot Technical AI Safety project.
