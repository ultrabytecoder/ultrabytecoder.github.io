---
layout: post
title: "Qwen 3.8 27B FP8 vs GLM 5.3 Flash Maya L: a 27B hybrid holds its own against a 321B MoE"
date: 2026-11-07 10:00:00 -0500
author: Your Name
tags: [qwen3.8, glm5.3-flash, gsm8k, local-llm, benchmark, reasoning]
description: "I ran the full 1,319-question GSM8K set (8-shot CoT, greedy) against local Qwen 3.8 27B (FP8) and GLM 5.3 Flash Maya L (321B MoE) on the same two 3090s."
---

## Why I ran this

I run **Qwen 3.8 27B in FP8** on my two RTX 3090s for day-to-day work, and I like it because it's fast and its initial prompt is lightweight. I've also had [**Z.ai**](https://z.ai)'s GLM-5.3 Flash sitting on the same box — a 321B-parameter MoE with only ~18B parameters active per token, which I run as the **Maya-L** GGUF quant (Project Maya; 3–4 bit, 156 GB).

The question I kept coming back to was simple: **is the big MoE actually smarter than my little 27B, or do they land in the same place?** To make the answer fair I put both models through the same benchmark under the same conditions — the **full GSM8K test split** (1319 word problems), 8-shot chain-of-thought, greedy decoding, single-threaded, on the **same machine**.

One thing worth stating up front: GSM8K is just **one** of the many benchmarks I could have run — coding, chat quality, long-context, tool use, and a dozen others are all fair candidates. I picked it deliberately. I wasn't really interested in how politely a model chats or how well it follows instructions; what I actually wanted to measure was its **raw reasoning** — can it work through a multi-step problem step by step and land on the right number? GSM8K is a clean, well-trodden proxy for exactly that, so out of all the tests available it's the *intelligence* test I cared about for this question.

It's equally important to be clear about what GSM8K is *not*: it's a **specialized test of one narrow slice** of model capability — grade-school arithmetic reasoning. A score on it does not paint a full picture of a model: how these two code, hold a long context, follow instructions, use tools, or simply chat is a separate question this benchmark cannot answer, and nothing in this post should be read as a verdict on any of those.

The short version: on the official metrics the two models are statistically indistinguishable. Dig into the errors and the picture sharpens — where a gap exists at all, it's **reasoning-driven in Qwen's favor, format-driven in GLM's**, and it's still inside the noise. Below is the full breakdown.

> 📄 **Want all the data?** The complete analysis — the per-question error taxonomy, the shared-failure breakdown, the response-style comparison, and the full methodology — is published as a standalone page: [Full report](/reports/qwen38-vs-glm53-flash-gsm8k/).

## The two models

The interesting thing about this matchup is that they're mismatched on paper but comparable on *active compute*. GLM is a 321B MoE, but only ~18B parameters fire per token. Qwen is "only" 27B, but it's a **hybrid** (Mamba/SSM + attention), so a lot of its layers have a fixed per-token cost rather than a full attention pass. On both sides the *active* parameters per token land well under the nominal count — which is what actually determines speed and, roughly, the "intelligence budget" you're paying for.

| | [Qwen 3.8 27B](https://huggingface.co/Qwen/Qwen3.8-27B) | [GLM 5.3 Flash](https://huggingface.co/zai-org/GLM-5.3-Flash) ([Maya-L](https://huggingface.co/peasantsmith/GLM-5.3-Flash-Maya-GGUF)) |
|---|---|---|
| **Params** | 27B (hybrid Mamba/attention) | 321B MoE, ~18B active |
| **Quant** | FP8 (~27 GB) | GGUF 3–4 bit (156 GB, ~99.2% of FP8) |
| **Engine** | vLLM 0.32.0, TP=2, FP8 KV cache + 62 GB RAM offload, MTP drafting¹ | Project Maya, expert tiering VRAM/RAM, MTP drafting¹ |
| **Context** | 256K | 320K |

¹ *MTP drafting* — multi-token prediction: a small draft head proposes the next few tokens, which the main model verifies, a form of speculative decoding that speeds up generation.

Both are quantized, but to different depths — Qwen in FP8 (near full accuracy), GLM at 3–4 bit (retaining ~99.2% of FP8 on zero-shot). So this isn't a "clean model vs clean model" test; it's a comparison of two **concrete deployments**, which is exactly what I actually run.

**Machine (one box for both runs):** AMD Ryzen 9 5900X, 2× RTX 3090 (48 GB VRAM), 125 GB RAM, Debian 13.

## The benchmark

- **Task:** `gsm8k_cot` (lm-evaluation-harness 0.4.13) — the chain-of-thought variant of GSM8K.
- **Data:** the **full** test split of `openai/gsm8k/main` — all **1319** questions (not a 100-question sample).
- **Protocol:** 8-shot, greedy (`do_sample: false`), 1 concurrent request, identical prompt and seed for both models.

I verified that for every one of the 1319×2 (doc × filter) records the `prompt_hash`, `doc_hash`, and `target` are identical between the two runs — so this is a strict one-to-one comparison. Every model saw exactly the same question and was scored against the same key.

## Results: officially, it's a tie

| Metric | GLM 5.3 Flash | Qwen 3.8 27B |
|---|---|---|
| **strict-match** (exact phrase) | 95.45 % (1259/1319) | **95.60 %** (1261/1319) |
| **flexible-extract** (last number) | 96.29 % (1270/1319) | **96.89 %** (1278/1319) |

The two models answer the *same* 1319 questions, so the right test for "is one better" is McNemar's on the disagreeing pairs, not two independent accuracy comparisons:

| | GLM right / Qwen wrong | Qwen right / GLM wrong | p-value |
|---|---|---|---|
| strict-match | 26 | 28 | 0.89 |
| flexible-extract | 16 | 24 | 0.27 |

At n = 1319 each metric's standard error is about ±0.5 pp, so the deltas (0.15 / 0.60 pp) are on the order of a single standard error — and neither McNemar test comes close to significance. **Formally, the two models are at parity, with Qwen marginally ahead on both metrics.**

(Strict-match requires the exact phrase "The answer is ⟨number⟩." — no dollar signs, commas, or decimal tails. That matters a lot, as the next section shows. Flexible-extract takes the last number-like token from the whole response and strips `$`, `,`, and one trailing dot, removing most of that artifact.)

## Where the real gap hides: decomposing the flexible delta

The flexible-extract gap is +8 questions for Qwen (1278 vs 1270). Classify each model's flexible failures, and the +8 splits into two opposing components:

| Component | GLM failures | Qwen failures | Net for Qwen |
|---|---|---|---|
| **Value errors** (arithmetic, logic, misread) | 43 | 30 | **+13** |
| **Format failures** (right value, extraction breaks) | 6 | 11 | **−5** |
| **Total** | 49 | 41 | **+8** |

This is the core finding: **Qwen reasons better (+13) but formats worse (−5).** Its reasoning advantage is bigger than the official gap suggests — the official score is masking it with format penalties. Read the split as descriptive, though: with only 40 disagreeing pairs in the whole run, no sub-decomposition like this has the power to be formally tested.

## Where the errors actually come from

**Qwen's format liability is the `$X.00` habit.** It signs answers with a currency sign — "The answer is **$125**." — which strict-match's regex rejects (16 such questions). For whole-dollar amounts it appends `.00` — "The answer is **$5.00**." — and the `.00` tail survives flexible-extract's stripping, so **8 questions pass no filter at all** despite the arithmetic being right. Add a parenthetical after the answer (doc 830: "1128 minutes (or 18 hours and 48 minutes)" → the extractor grabs `48`), a "2:00 pm" time format (doc 1001, shared with GLM), and an answer given in cents (doc 1067), and Qwen's 11 format failures are fully accounted for. Ten of the eleven are trivially fixable with a normalizing post-processor — strip `$`, `,`, and `.00` from the final line — which lifts Qwen from 96.89 % to **97.65 %** effective; the eleventh (the cents answer) additionally wants a unit-aware rule, taking it to 97.73 %.

**Qwen's value errors split roughly in half**: about 21 pure computation slips (a sign error, a forgotten multiplier, compound where simple interest was meant) and about 9 misreads of the question. The classic is doc 161, where it excludes starfish from the catch because "they're not actually fish" — the reference counts them.

**GLM mostly writes bare numbers** — extraction-friendly — so its format failures are different: two space-separated thousands ("The answer is **9 500**." → the extractor reads `500`; a locale convention that breaks *both* filters), and **five truncated responses** (docs 153, 672, 754, 926, 1011 — all cut off mid-answer, under 520 chars; three of them lost the final number).

The truncations are the interesting part, because they point at the serving stack, not the model. The case is clear for GLM: five answers cut off mid-sentence (all under 520 chars), three of them losing the final number, and a hard 584-char ceiling on completed responses. Qwen's side is weaker: its longest response — doc 754 — ends at **exactly 838 chars** with the correct total sitting in bold but no "The answer is" line, which may be the same story, but no Qwen score is affected by it. Either way, the generation limit is worth raising before re-comparing — on the GLM side in particular.

**GLM's value errors are where it loses ground: about 25 computation mistakes and about 18 misreads.** The misread share is the striking number — **37 % of GLM's failures are misreading the question, versus 22 % for Qwen.** Speed↑ treated as time↑ (doc 93), counting shelves as items (doc 831), using the final month's headcount instead of summing all three (doc 796), treating a group's total travel time as each animal's time (doc 835). Same arithmetic skill, weaker question comprehension.

## The 32 questions both fail

Both models get 32 questions wrong under strict-match, and the breakdown of that pile matters:

- **12 are genuine shared blind spots** — the "hard core": forgetting double-sided printing (doc 951), an off-by-one on "first profit year" (doc 12), a forgotten ×2 multiplier (doc 782). Both models make the *same* error in most of these, which points at systematic reasoning gaps, not random slips.
- **11 are reference ambiguity** — both models give the defensible, "natural" reading and the GSM8K reference disagrees ("three times more than" = 3× or 4×?; "8:00 AM until 11:00 PM" read against an 11 AM reference). These are the dataset's problem, not the models'.
- The rest are format/unit overlays (dollar signs, cents vs dollars) or one-side-correct cases.

So "32 shared failures" overstates the models' common ceiling: more than a third of that pile is annotation noise in a 2021 dataset.

## A note on the wall-clock numbers

GLM's run took **7970 s** for the 1319 questions, Qwen's **3765 s** — a 2.1× wall-clock difference. I'm deliberately *not* reading that as "Qwen is 2.1× faster at reasoning": the two runs went through different serving stacks (vLLM TP=2 with FP8 weights fully in VRAM, versus Project Maya streaming a 156 GB GGUF's experts from RAM), different endpoints, and different generation settings. It's a fair comparison of *those two deployments*, and only that — the kind of number that moves with batch size, concurrency, and max-tokens, not with the model.

## What this means for me

1. **On the official metrics, it's a tie.** McNemar says parity; both deltas sit within one standard error. "More parameters" did not buy the 321B MoE a measurable edge on this task.
2. **Under the surface, Qwen's reasoning is ahead — and GLM's extra failures are more often misreads than arithmetic.** +13 value-error advantage for Qwen; 37 % of GLM's failures are question-comprehension errors versus 22 % for Qwen.
3. **The official metrics flatter the better formatters and penalize the better reasoner.** Qwen's `.00` habit costs it 8 questions that flexible-extract can't see; GLM's bare-number style is the extraction-friendly one. A normalizing post-processor flips the ledger: Qwen to ~97.7 % effective.
4. **The shared-failure pile is part dataset, part model.** 11 of the 32 are reference ambiguity; 12 are a genuinely hard core both models share.

In practice, for reasoning work specifically — the reason I ran this comparison in the first place — I'm staying on **Qwen 3.8 27B**. And the cleaner version of the recommendation is: **deploy it with a number-normalization post-processor** (strip `$`, `,`, `.00`, and trailing units from the final answer line), or just prompt it to answer with a bare number. That erases its format liability and leaves the reasoning edge as the differentiator. GLM-5.3 Flash is the safer choice only if you need zero-post-processing strict-regex output — and even then, raise its generation limit first, because three of its score-affecting failures are truncations, not wrong answers.

The honest caveats: one greedy run per model on a single benchmark — and it is a specialized one. GSM8K is a narrow slice of reasoning (grade-school arithmetic), so winning it says nothing about MATH, AIME, or code, and it does not paint a full picture of either model: coding, long-context, instruction-following, and tool-use ability are all separate dimensions this test never touches. The 8-shot CoT prompt shapes answer phrasing (Qwen's `.00` habit may be few-shot conditioning). n = 1319 has low power for a 0.6 pp difference — the direction of the gap is consistent, but its size isn't pinned down. And the GLM endpoint ran with a low generation ceiling, which cost it three answers; Qwen's longest response stops at a suspiciously round 838 chars, which may be the same story. For what it *does* measure — raw step-by-step arithmetic reasoning — the 27B hybrid is the stronger model, and it holds its own against the 321B MoE without needing the parameter count to back it up; on the other dimensions this test never touches, the verdict stays open.
