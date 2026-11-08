---
layout: default
title: "GSM8K Analysis — Qwen 3.8 27B vs GLM 5.3 Flash (Maya-L)"
description: "Full comparative analysis of the 1319-question GSM8K chain-of-thought run: headline metrics, McNemar tests, per-question error taxonomy, shared-failure breakdown, response-style comparison, and methodology."
date: 2026-11-07
---

# Comparing Qwen3.8-27B (FP8) and GLM-5.3-Flash (Maya-L) on GSM8K (chain-of-thought)

**Run dates:** November 7, 2026 (GLM: 02:53, Qwen: 09:04)
**Data:** `glm-5.3-flash/`, `qwen3.8-27b_fix/qwen3.8-27b/` (lm-evaluation-harness 0.4.13, task `gsm8k_cot` v3.0)
**Scale:** full test split of `openai/gsm8k/main` — **1319 questions**, 8-shot CoT, greedy (`do_sample: false`)
**Contestants:** Qwen3.8-27B **FP8** (27B, hybrid Mamba/attention) vs GLM-5.3-Flash **Maya-L** (321B MoE, ~18B active, GGUF 3–4 bit, 156 GB)

## Setup: models, machine, endpoints

- **Qwen3.8-27B (FP8)** — a 27B-parameter hybrid (Mamba/SSM + attention, so its per-token active compute is noticeably lower than a dense 27B transformer), served in **FP8 precision** (≈27 GB of weights): **vLLM v0.32.0** (Docker `vllm/vllm-openai:v0.32.0`), **tensor-parallel=2** across both RTX 3090, `--quantization fp8`, 256K context (`--max-model-len 262144`), **FP8 KV cache** + 62 GB offloading to system RAM, **MTP speculative decoding** (3 draft tokens). Endpoint `http://192.0.2.1:8091/v1/chat/completions` (Docker mapping 8091→8000), single-threaded evaluation.
- **GLM-5.3-Flash (Maya-L)** — a 321B-parameter MoE with **~18B active per token**, MIT license, official release precision FP8. Run here as the **Maya-L GGUF quant** (3–4 bit, 156 GB, ~99.2 % of FP8 zero-shot) in **Project Maya** with VRAM/RAM expert tiering (the SSD tier was not used — 0 disk reads per the logs), 320K context, MTP drafting. Endpoint `http://192.0.2.1:8080/v1/chat/completions`, single-threaded evaluation.
- **Machine (one box for both runs):** CPU **AMD Ryzen 9 5900X** (12 cores / 24 threads), GPU **2× NVIDIA GeForce RTX 3090** (24 GB each, 48 GB VRAM total), RAM **125 GiB**, Debian 13, kernel 6.12.90, NVIDIA driver 595.80 / CUDA 13.2. The GLM run (02:53) ran on Project Maya, which held both GPUs; by the Qwen run (09:04) vLLM TP=2 held both (the container was started at 07:42) — Maya was already stopped.

⚠️ **Interpretation note:** the two runs used **different serving stacks and generation settings**. The per-question scores are directly comparable (identical prompts, targets, and single-threaded protocol — verified per-record: `prompt_hash`, `doc_hash`, and `target` are identical across both runs), but wall-clock times and truncation-related findings compare *deployments*, not base models. Both endpoints appear to run with low generation ceilings (see "Response Style Comparison").

## 1. Headline Verdict

### Raw scores

| Filter | GLM acc (correct/n) | Qwen acc (correct/n) | Gap (Qwen − GLM) |
|---|---|---|---|
| strict-match | 0.9545 (1259/1319), SE 0.00574 | 0.9560 (1261/1319), SE 0.00565 | +0.0015 (+2 samples) |
| flexible-extract | 0.9629 (1270/1319), SE 0.00521 | 0.9689 (1278/1319), SE 0.00478 | +0.0060 (+8 samples) |

### Statistical significance

**The difference is not statistically significant under the correct test.** The samples are *paired* (both models answer the same 1319 questions), so the appropriate test is McNemar's, not an independent-samples z-test:

| Filter | GLM-only correct | Qwen-only correct | McNemar χ² (w/ cc) | p-value |
|---|---|---|---|---|
| strict-match | 26 | 28 | (|26−28|−1)²/54 = 0.019 | 0.89 |
| flexible-extract | 16 | 24 | (|16−24|−1)²/40 = 1.225 | 0.27 |

An independent-samples z-test (which ignores the pairing) yields z ≈ 0.19 (p ≈ 0.85) for strict and z ≈ 0.86 (p ≈ 0.39) for flexible — also non-significant, but the test is inapplicable here because with 1233 shared correct answers, the effective discriminative sample is only the 54 (strict) or 40 (flexible) discordant pairs; McNemar on those paired discordances is the correct test. Neither set shows a significant imbalance.

### Which metric tells the real story

**Flexible-extract is the metric that matters.** The strict-match filter requires the exact phrase "The answer is ⟨number⟩." — no dollar signs, no commas, no decimal suffixes. This penalizes Qwen disproportionately because Qwen consistently writes answers like "The answer is $125." (doc 15) or "The answer is $45.00." (doc 710), which fail the strict regex despite being arithmetically correct. Flexible-extract removes this artifact by taking the last number-like token from the full response and stripping `$`, `,`, and one trailing `.`.

Under flexible-extract, the gap widens from +2 to +8 samples (0.15pp → 0.60pp), revealing that Qwen's underlying reasoning advantage was being masked by format penalties. However, the +8 gap is still within noise (McNemar p = 0.27).

### Decomposition of the flexible-extract +8 gap

The 8-sample gap decomposes cleanly into two opposing components:

| Component | GLM failures | Qwen failures | Net (Qwen advantage) |
|---|---|---|---|
| **Value errors** (arithmetic/logic/misread) | 43 | 30 | **+13** |
| **Format failures** (extraction artifacts) | 6 | 11 | **−5** |
| **Total** | 49 | 41 | **+8** |

Qwen's advantage is **reasoning-driven (+13)**, partially offset by **format issues (−5)**. This is the core finding: Qwen reasons better but formats worse.

---

## 2. Error Taxonomy

Using the flexible-extract failure census (the cleaner view), each model's failures are classified as:
- **(a) Format/extraction** — correct arithmetic, but the answer is unparsable or extracts the wrong token
- **(b) True arithmetic/logic error** — model understands the question but makes a computational or step-omission error
- **(c) Misread of the question** — model misunderstands what is being asked

### Qwen3.8-27b (41 flexible failures: 11 format + 30 value)

**(a) Format/extraction (11 failures):**

| Subtype | doc_ids | Mechanism |
|---|---|---|
| `$X.00.` decimal (8) | 272, 430, **584**, 638, 710, 873, 1070, 1139 | Flexible strips `$` and one trailing `.`, but `.00` survives → `"5.00" ≠ "5"` |
| Parenthetical after answer (1) | 830 | `"The answer is 1128 minutes (or 18 hours and 48 minutes)."` → last number = `48` |
| Time notation (1) | 1001 | `"The answer is 2:00 pm."` → last number-like token = `00` ≠ `2` (shared with GLM) |
| Unit mismatch (1) | 1067 | Answer in cents `24500` vs reference in dollars `245` |

All 11 have **correct underlying values** — the arithmetic is right, the extraction fails. The 8 `.00` cases are the largest single category. Doc 584 is shared with GLM (where GLM has a separate value error — see below).

Evidence:
- **doc 272**: Qwen correctly computes change = $5.00, writes "The answer is $5.00." → flexible extracts `5.00` ≠ target `5`.
- **doc 830**: Qwen correctly computes 1128 minutes, but appends "(or 18 hours and 48 minutes)" → flexible grabs `48`.

**(b) True arithmetic/logic errors (~21 of 30 value errors):**

The model understands the question but makes a computational slip or omits a step. Representative cases:

| doc_id | Error | Detail |
|---|---|---|
| 423 | Sign error | Computes `$24 − $32 = −$8` instead of `|$24 − $32| = $8` |
| 450 | Omitted term | Uses $20 (ticket only) instead of $21 (ticket + $1 popcorn/soda) as super-ticket cost |
| 652 | Cumulative doubling | Computes 15 + 30 + 60 = 105 (each hour doubles the *previous*) instead of 15 + 30 + 30 = 75 (each hour = 2× *base*) |
| 782 | Missing multiplier | Computes 3×5 + 6×2 = 27, forgetting *Chase and Rider* are two people → should be ×2 = 54 |
| 951 | Forgot double-sided | Computes 100 × 1.5 = 150 pages, forgetting sheets are printed on both sides → 300 |
| 1048 | Compound vs simple depreciation | Uses 20000 × 0.79³ = 9860.78 (compound) instead of 20000 − (20000 × 0.21 × 3) = 7400 (simple) |

**(c) Misread of the question (~9 of 30 value errors):**

| doc_id | Misread | Detail |
|---|---|---|
| 161 | Over-applied domain knowledge | Excludes starfish as "not actually fish (echinoderms)" — question intends all caught creatures as "fish" → 17 instead of 32 |
| 306 | Over-split time window | Splits afternoon+evening portion (10) into 5+5; reference treats the full 10 as the afternoon amount |
| 607 | Confused quantities | Computes cost per *washing* ($0.40) instead of cost per *pump* ($0.10) |
| 1019 | Misinterpreted "late checkout" | Treats 15 late-checkout guests as *staying* rather than *leaving* → 113 instead of 98 |
| 1071 | Double-counted host | Adds Kayla herself (+1) to the 45 invitees → 257 instead of 251 |

### glm-5.3-flash (49 flexible failures: 6 format + 43 value)

**(a) Format/extraction (6 failures):**

| Subtype | doc_ids | Mechanism |
|---|---|---|
| Space-separated numbers (2) | 265, 266 | Writes `"9 500"` or `"118 000"` → flexible grabs only the last segment (`500` or `000`) |
| Truncation (3) | 754, 926, 1011 | Generation cuts off mid-answer → wrong/absent last number |
| Time notation (1) | 1001 | `"2:00 pm"` → `00` (shared with Qwen) |

Evidence:
- **doc 265**: GLM writes "The answer is 9 500." → flexible extracts `500.` → 500 ≠ 9500. The value is correct; the space-separated format kills extraction.
- **doc 926**: GLM response ends mid-sentence ("So he") after computing $20 remaining → last number is `20` ≠ target `1`.
- **doc 1011**: GLM response ends "The answer is 600" (truncated from 6000) → extracts `600` ≠ 6000.

The truncation cases (754, 926, 1011) are a **serving-config artifact**: these three responses stopped at 425–486 chars mid-answer, and GLM's overall output is shorter (longest completed response 584 chars vs Qwen's 838, whose doc 754 also ends at exactly the max without an answer line). Both endpoints appear to run with low generation limits.

**(b) True arithmetic/logic errors (~25 of 43 value errors):**

| doc_id | Error | Detail |
|---|---|---|
| 7 | Wrong percentage base | Computes 40% of *time* (0.40 × 100 min = 20 min) instead of 40% of *data* (80 GB ÷ 2 GB/min = 40 min) → 140 instead of 160 |
| 147 | Omitted tracking step | Counts only 15 unreturned bees from first wave, forgetting the 60 bees that left in the third wave → 15 instead of 75 |
| 284 | Wrong doubling target | Doubles the *total* (120 → 240) instead of *each individual* (20+40+60 → 40+80+120 = 240, total 360) |
| 304 | Wrong week count | Computes 30 ÷ 5 = 6 work-weeks in April; correct is 4 weeks → 4800 instead of 3200 |
| 494 | Missed pyramid level | Counts 3 levels (64+64+60) instead of 4 (64+64+64+60) → 15.67 instead of 21 |
| 584 | Per-contact vs per-pair | Divides $180 by 180 contacts ($1.00/contact) instead of by 90 pairs ($2.00/pair) → 1.00 instead of 2 |
| 611 | Wrong monthly rate | Uses $100,000 × 1.5 = $150,000/month instead of $20,000 × 1.5 = $30,000/month → $6.85M instead of $1.45M |
| 796 | Final-month-only headcount | Uses only the final month's employee count (260) × $4000 instead of summing all three months → $1.04M instead of $2.88M |
| 831 | Counted shelves as items | Includes 2 shelves from the base package in the "total items" count (28) instead of comparing only Angela's items to the package → 294 instead of 324 |
| 1016 | Compound vs simple interest | Uses 120 × 1.05³ = 138.915 (rounds to 139) instead of 120 + 3×6 = 138 (simple interest) |
| 1215 | Multiplication error | Computes 6 × 6 × 15 = 536 (should be 540) → cascading error yields 42 instead of 36 |
| 1300 | Wrong day count | Says "open 6 days a week" (should be 5: closed Sunday AND Wednesday) → 144 instead of 120 |

**(c) Misread of the question (~18 of 43 value errors):**

| doc_id | Misread | Detail |
|---|---|---|
| 37 | Assumed all toys sold | "John sold all 13 lego sets" — question implies he sold only enough to cover $165/$15 = 11 sets → 0 instead of 2 |
| 93 | Speed ↑ means time ↑ | Treats "speed improved by 10%" as time *increasing* 10% (44s) instead of *decreasing* 10% (36s) |
| 119 | "30% higher" interpretation | Uses $40,000 ÷ 1.3 = $30,769 (mathematically standard) instead of $40,000 × 0.7 = $28,000 (reference interpretation) |
| 255 | Contradictory question text | Question says "Ten stalls" then "twenty stalls" — GLM uses 20, reference uses 10 |
| 340 | Caught balls returned | Subtracts only the 1 lost ball (7−1=6) instead of all 3 dropped (7−3=4) |
| 590 | Forgot to count host | Computes 100+200+10+8 = 318 guests, forgetting Alex herself → 318 instead of 319 |
| 835 | Per-individual vs total | Treats 18 hours as per-kangaroo and doesn't multiply by 4 turtles → 36 instead of 48 |

### Summary table

| Category | Qwen (flex, 41) | GLM (flex, 49) |
|---|---|---|
| (a) Format/extraction | 11 (27%) | 6 (12%) |
| (b) True arithmetic/logic | ~21 (51%) | ~25 (51%) |
| (c) Misread of question | ~9 (22%) | ~18 (37%) |
| **Total** | **41** | **49** |

GLM has a higher misread rate (37% vs 22% of failures), suggesting weaker question comprehension. Both models have similar proportions of pure arithmetic errors (~51%), but GLM has more in absolute terms (25 vs 21).

---

## 3. Where Each Model Wins

### Strict-match divergent sets

| Direction | Count | Format-driven | Reasoning-driven |
|---|---|---|---|
| GLM right / Qwen wrong | 26 | 19 (73%) | 7 (27%) |
| Qwen right / GLM wrong | 28 | 13 (46%) | 15 (54%) |

Under strict-match, **GLM's wins are overwhelmingly format-driven**: 19 of 26 cases where GLM beats Qwen are because Qwen wrote "$X." or "$X,XXX." or "$X.00." — answers that fail the strict regex despite being correct. Only 7 are genuine reasoning wins.

### Flexible-extract divergent sets (format artifacts removed)

| Direction | Count | Format-driven | Reasoning-driven |
|---|---|---|---|
| GLM right / Qwen wrong | 16 | 9 (56%) | 7 (44%) |
| Qwen right / GLM wrong | 24 | 5 (21%) | 19 (79%) |

When format penalties are removed, GLM's win count drops sharply (26→16) while Qwen's drops less (28→24). The residual format wins for GLM are the 7 individual `.00` cases (272, 430, 638, 710, 873, 1070, 1139) plus doc 830 (parenthetical) and doc 1067 (cents vs dollars).

### Patterns

**GLM-right / Qwen-wrong (16 flexible):**
- *Format (9)*: Concentrated almost entirely in the `.00` pattern — Qwen writes "$5.00." (doc 272), "$132.00." (doc 430), "$45.00." (doc 710), etc. These are whole-dollar amounts where Qwen appends ".00", which survives the trailing-dot strip. Plus doc 830 (parenthetical "48 minutes" grabbed) and doc 1067 (answer in cents).
- *Reasoning (7)*: Diverse — doc 161 (starfish taxonomy), doc 306 (afternoon/evening split), doc 423 (sign error), doc 450 (omitted $1), doc 607 (pump vs washing), doc 1019 (late checkout), doc 1071 (host count). No single question type dominates.

**Qwen-right / GLM-wrong (24 flexible):**
- *Format (5)*: GLM's space-separated numbers (265, 266) and truncation (754, 926, 1011) — all serving-config artifacts, not reasoning failures.
- *Reasoning (19)*: Spans percentage calculations (doc 7: wrong base, doc 93: speed/time inversion), multi-step tracking (doc 147: forgotten bee wave, doc 340: ball counting), counting/levels (doc 494: pyramid levels, doc 590: host omitted), temporal reasoning (doc 304: weeks in April, doc 1300: open days), and computational errors (doc 1215: 6×6×15≠536, doc 1016: compound interest).

**Key insight**: GLM's advantage is almost entirely format-based (Qwen writes "$X.00."). Qwen's advantage is almost entirely reasoning-based (GLM makes arithmetic and comprehension errors). There is no clear question-type pattern — divergences span percentages, counting, money, temporal reasoning, and multi-step problems.

---

## 4. Response Style Comparison

### Conciseness

| Metric | GLM | Qwen |
|---|---|---|
| Mean (chars) | 242 | 253 |
| Median | 233 | 240 |
| p90 | 336 | 355 |
| Max | 584 | 838 |

GLM is ~4–5% shorter on average and has a substantially lower max (584 vs 838). GLM also stops generation prematurely on 5 responses (docs 153, 672, 754, 926, 1011 — all under 520 chars, mid-sentence or mid-computation), and Qwen's own longest response (doc 754, exactly 838 chars) ends without an answer line — evidence that both serving endpoints run with low generation limits.

### Final-answer phrasing

**GLM** typically writes bare numbers: `"The answer is 125."` (doc 15), `"The answer is 33."` (doc 226), `"The answer is 3528."` (doc 1270). This passes the strict regex. However, GLM sometimes includes dollar signs and commas: `"The answer is $70,000."` (doc 2), `"The answer is $1,920."` (doc 576) — these fail strict but pass flexible. GLM also uses space-separated large numbers: `"The answer is 9 500."` (doc 265), `"The answer is 118 000."` (doc 266) — likely a training-data locale convention that breaks both filters.

**Qwen** consistently writes dollar signs and, for whole-dollar amounts, appends ".00": `"The answer is $125."` (doc 15), `"The answer is $5.00."` (doc 272), `"The answer is $45.00."` (doc 710), `"The answer is $132.00."` (doc 430). The dollar sign alone fails strict but passes flexible (stripped). The ".00" suffix fails *both* filters because the trailing-dot strip leaves "X.00" ≠ "X".

### Structural quirks

| Quirk | GLM | Qwen |
|---|---|---|
| Leading blank lines | None | 2 blank lines at start of every response |
| Formatting | Plain prose | Bullet points, bold headers (`**Jewelry:**`), numbered steps |
| Parentheticals after answer | None observed | Sometimes appends unit conversions: `"1128 minutes (or 18 hours and 48 minutes)"` (doc 830) |
| Truncation | 5 cases (153, 672, 754, 926, 1011; 3 broke extraction, 2 kept the right last number) | 1 case (754: response ends at the 838-char max with no answer line, but the correct value `**89**` is still extractable) |
| Space-separated numbers | 2 cases (265, 266) | None |

### Implications for real-world usage

1. **Qwen's `.00` pattern is the single biggest format liability** — 8 flexible failures, all fixable with a trivial post-processor. In a production pipeline with no post-processing, Qwen would silently fail on ~0.6% of whole-dollar answers.
2. **GLM's truncation is a serving-config issue** — GLM cut off 5 responses (153, 672, 754, 926, 1011), 3 of which lost the answer. Raising `max_tokens` on both endpoints would fix these (Qwen's doc 754 also ended at its 838-char max without an answer line). The low generation ceilings (584 / 838 chars) are serving parameters, not model limitations.
3. **GLM's space-separated numbers** (docs 265, 266) suggest a training-data locale convention. A number-normalization preprocessor would fix this.
4. **Qwen's structured output** (bullets, bold) is more human-readable but can cause flexible-extraction to grab wrong numbers from parentheticals (doc 830).
5. **Neither model emits "no answer"** — both always produce a final value, so the failure mode is always "wrong value" or "unparseable format," never silence.

---

## 5. Robustness: The 32 Shared Strict Failures

The 32 questions both models fail under strict-match break into four tiers:

### Tier 1 — Shared format overlays (3 cases: both correct value, format kills strict)

| doc_id | GLM format | Qwen format | Correct value |
|---|---|---|---|
| 266 | Space-separated: "118 000" | Dollar sign: "$118,000." | 118000 |
| 1067 | Cents: "24500" (but mentions "$245" at end → flex passes) | Cents: "24500" | 245 (dollars) |
| 1298 | Dollar sign: "$31." | Dollar sign: "$31." | 31 |

In all three, both models computed the right answer. Under flexible-extract: Qwen passes 266 and 1298 but fails 1067 (last number `24500`); GLM passes 1067 (ends with "(or $245)" → last number `$245`) and 1298, but fails 266 (space-separated `118 000`).

### Tier 2 — Asymmetric (6 cases: Qwen correct value but format, GLM wrong value)

| doc_id | GLM error | Qwen correct value | Qwen format issue |
|---|---|---|---|
| 409 | Per-week salary ($42K) instead of per-month ($168K) | 168000 | Dollar sign (passes flex) |
| 584 | Per-contact ($1.00) instead of per-pair ($2.00) | 2 | `.00` (fails flex) |
| 611 | $6.85M (wrong monthly rate) instead of $1.45M | 1450000 | Dollar sign (passes flex) |
| 754 | Truncated mid-computation | 89 | No "The answer is" line (passes flex) |
| 796 | $1.04M (final month only) instead of $2.88M | 2880000 | Dollar sign (passes flex) |
| 831 | $294 (counted shelves) instead of $324 | 324 | Dollar sign (passes flex) |

In 5 of 6, Qwen has the correct answer and would pass flexible if not for the dollar sign (which flexible does strip). Only doc 584 fails flexible too, because of the `.00` suffix.

### Tier 3 — Both wrong: genuine model errors (12 cases)

These are real failures where both models make a reasoning error on the same problem:

| doc_id | Both computed | Target | Error pattern |
|---|---|---|---|
| 12 | 12 | 13 | Off-by-one: computed break-even year, not first profit year |
| 652 | 105 | 75 | Cumulative doubling (15+30+60) instead of base doubling (15+30+30) |
| 675 | 25 / 21 | 33 | Complex back-and-forth tracking with multiple departures/returns |
| 768 | 600 | 675 | Miscounted total books (interpretation of "twice as many") |
| 782 | 27 | 54 | Forgot 2 people (Chase AND Rider each ride) |
| 814 | 17 | 11 | Wrong play-duration per quarter calculation |
| 951 | 150 | 300 | Forgot double-sided printing (sheets × 2 = pages) |
| 984 | 162 / 168 | 342 | Fence-post error in plant spacing |
| 1048 | 9860.78 | 7400 | Compound depreciation instead of simple |
| 1161 | 140 | 170 | Miscounted remaining land after sequential sales |
| 1176 | 1 | 2 | Percentage-of-remaining vs percentage-of-original confusion |
| 1198 | 120 | 140 | Miscounted second-day trees ("1/4 more" interpretation) |

**Common blind spots**: off-by-one on "starts earning" (doc 12), forgetting multipliers (doc 782: ×2 people), compound vs simple rates (doc 1048), and double-sided/counting inversions (doc 951). Both models make the *same* error in most cases, suggesting systematic reasoning gaps rather than random mistakes.

### Tier 4 — Both wrong: ambiguous/reference issues (11 cases)

These are questions where both models give defensible answers that don't match the reference, typically due to ambiguous phrasing:

| doc_id | Both computed | Target | Ambiguity |
|---|---|---|---|
| 368 | 10 | 18 | "Three times more than" — 3× or 4×? |
| 403 | 135 | 81 | "Reduces by 5 hours" — saving 5h or reducing to 5h (saving 3h)? |
| 406 | 160 | 200 | "40 fewer corns than the number of cannolis" — original or new count? |
| 454 | 240 | 150 | 8 apples/day × 30 = 240 appears correct; reference 150 is unclear |
| 649 | 160000 | 162000 | "70 times as many" — 70× or 71×? |
| 749 | 6 | 3 | "How many more groups needed" — 12−6=6 appears correct |
| 823 | 18 | 14 | 14+4=18 appears correct; reference seems to exclude game 2 |
| 952 | 1800 | 360 | "8:00 AM until 11:00 PM" = 15h → 1800; reference appears to use 11 AM (3h) |
| 962 | 600 | 1000 | "How much does travel cost?" — ticket cost ($600) or total ($1000)? |
| 1042 | 3.5 | 3 | Average loss calculation — net or gross? |
| 1309 | 2180 | 2280 | Sum 430+750+300+700=2180; reference 2280 doesn't reconcile |

These represent **reference ambiguity**, not model failure. Both models consistently interpret the "natural" reading, which differs from the GSM8K reference. This is a known limitation of GSM8K.

### Doc 1001 — Flexible-only shared format failure (not in strict set)

Both models write "The answer is 2:00 pm." (target `2`). Strict extracts `2` → both pass. Flexible extracts the last number-like token `00` (from "2:00") → both fail. This is a **shared format overlay** (time notation) that cancels in the net gap. It is not a reasoning failure for either model.

---

## 6. Bottom Line

### Recommendation

**Qwen3.8-27B is the preferred model for math/reasoning use cases**, with important caveats.

The core evidence:
- Under flexible-extract (the metric that removes format artifacts), Qwen leads by 8 samples (96.89% vs 96.29%), driven by a **+13 reasoning advantage** (30 vs 43 value errors) that is partially offset by a **−5 format penalty** (11 vs 6 format failures).
- Qwen's format failures are almost entirely the `.00` pattern (8 cases) and are trivially fixable. GLM's extra value failures are reasoning errors (wrong percentage bases, missed steps, compound/simple confusion) that are *not* fixable by post-processing.
- The gap is not statistically significant (McNemar p = 0.27), but the directionality is consistent across both metrics and the reasoning decomposition is clear.

### Post-processor recovery

A post-processor that strips `$`, `,`, `.00`, and units from the final answer line (and handles phrasing variants and time notation) would recover:

| Baseline | Qwen correct | After post-processor | Recovery |
|---|---|---|---|
| strict (1261/1319, 95.60%) | — | **1280/1319 (97.04%)** | +19 format cases (17 dollar-sign + 226 phrasing + 1288 units) |
| flexible (1278/1319, 96.89%) | — | **1288/1319 (97.65%)** | +10 cases (8 `.00` + 830 parenthetical + 1001 time notation) |
| flexible, unit-aware | — | **1289/1319 (97.73%)** | +1067 (cents → dollars) |

With this post-processor, Qwen's effective accuracy reaches ~97.7%, at which point the remaining ~30 failures are genuine reasoning errors shared with or distinct from GLM.

### Caveats

1. **Single benchmark**: GSM8K is grade-school word problems. It does not test higher mathematics, symbolic reasoning, multi-hop logic, or code generation. A model that wins on GSM8K may not win on MATH, AIME, or GPQA.
2. **Greedy decoding**: `do_sample=false` means no sampling variance was explored. Temperature or top-p sampling could shift results, especially on the ~40 discordant pairs where the models differ.
3. **8-shot prompting**: The prompt format (first_n, 8-shot) influences both models' answer phrasing. Different shot counts, zero-shot, or chain-of-thought variants could change the format failure rate (especially Qwen's `.00` pattern, which may be influenced by few-shot examples).
4. **Endpoint differences**: GLM was served at :8080 and Qwen at :8091 — different vLLM/serving configurations. Both endpoints appear to run with low generation ceilings (GLM's longest response 584 chars, 5 truncated; Qwen's longest 838 chars, 1 truncated at doc 754) — almost certainly `max_tokens` settings, not model limitations. The wall-clock difference (GLM 7970s vs Qwen 3765s) reflects serving-config differences (batch size, concurrency, hardware) and should not be interpreted as a model speed comparison.
5. **Moderate sample size**: n=1319 is adequate for estimating accuracy but marginal for detecting small differences. With only 40 discordant pairs under flexible-extract, the power to detect a 0.6pp difference is low. A larger test set or multiple seeds would be needed for a definitive comparison.
6. **Format artifacts are environment-specific**: In a production pipeline with a number-extraction post-processor, Qwen's format advantage disappears and its reasoning advantage becomes the sole differentiator. In a raw-greedy pipeline with strict regex matching, GLM's cleaner formatting gives it an artificial edge.

### Summary verdict

| Dimension | Winner | Margin |
|---|---|---|
| Raw accuracy (strict) | Qwen | +2 samples (not significant) |
| Raw accuracy (flexible) | Qwen | +8 samples (not significant) |
| Reasoning quality | **Qwen** | +13 value-error advantage |
| Format robustness | **GLM** | −5 format-error disadvantage for Qwen |
| Post-processed accuracy | **Qwen** | 97.65% vs ~96.4% (est. GLM with same processor: 1259 + 12 recoverable format cases) |
| Latency (indicative) | Qwen | 3765s vs 7970s (endpoint-confounded) |

**Qwen3.8-27B is the stronger reasoning model. Deploy it with a number-normalization post-processor.** GLM-5.3-flash is the safer choice if you need zero-post-processing strict-regex compatibility and can increase its generation length limit.