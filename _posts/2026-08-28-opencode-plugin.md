---
layout: post
title: "Loop Detection Plugin for OpenCode"
date: 2026-08-28 10:00:00 -0500
author: Your Name
tags: [opencode, plugin, llm, local-llm]
description: "A small OpenCode plugin that detects and breaks LLM tool-call loops on local models."
---



## Why opencode

I run the local Qwen 3.8 27B model in FP8 with vLLM for day-to-day work, and I use OpenCode for my coding workflow because its initial prompt is lightweight. Coding agents like Claude Code send around 30 KB of context at startup, while OpenCode stays at a few kilobytes — with my two RTX 3090s, that difference in prefill time is quite noticeable, and vLLM's KV-cache doesn't always eliminate it.

For the same reason, I keep my configuration minimal: every extra instruction or rule goes into the prompt and affects inference performance on a local setup.

One issue I keep hitting is Qwen getting stuck in a loop during complex tasks — for example, repeatedly investigating the same piece of logic without making progress. OpenCode doesn't handle this well out of the box, so I wrote a small plugin to detect and break these loops. It's a modest addition, but it makes a noticeable difference on complex tasks.

## The plugin

The plugin hooks into the `tool.execute.before` event and keeps a sliding window of the last 5 tool calls per session. If the same tool is called 3 times in a row with identical arguments, the plugin blocks the execution and returns a structured error message straight to the LLM, telling it to try a different approach.

```typescript
import type { Plugin } from "@opencode-ai/plugin"

const WINDOW = 5
const THRESHOLD = 3

export const LoopDetection: Plugin = async ({ client }) => {
  const recent = new Map<string, string[]>()

  const sig = (tool: string, args: unknown) =>
    tool + "::" + JSON.stringify(args)

  return {
    "tool.execute.before": async (input) => {
      const key = input.sessionID
      const s = sig(input.tool, input.args)
      const list = recent.get(key) ?? []
      list.push(s)
      if (list.length > WINDOW) list.shift()
      recent.set(key, list)

      const tail = list.slice(-THRESHOLD)

      // Ensure we actually have THRESHOLD consecutive items before checking
      if (tail.length === THRESHOLD && tail.every((x) => x === s)) {
        recent.set(key, [])

        await client.app.log({
          body: {
            service: "loop-detection",
            level: "warn",
            message: `Loop detected: ${input.tool} called ${THRESHOLD}x with identical args`,
          },
        })

        // Return a structured error result instead of throwing an Error.
        // This tells OpenCode to skip the tool execution and pass this message directly to the LLM.
        return {
          error: `Loop detected: "${input.tool}" has been called ${THRESHOLD} times in a row with identical arguments. Try a different approach.`
        }
      }
    },
  }
}
```

## How it works

1. **Signature** — every tool call is reduced to a signature: the tool name plus its JSON-serialized arguments. Two calls are considered identical only if both match.
2. **Sliding window** — for each session the plugin keeps the last `WINDOW` (5) signatures in a sliding window array.
3. **Detection** — if the last `THRESHOLD` (3) signatures are all identical to the current one, a loop is detected.
4. **Intervention** — instead of throwing an exception (which would surface as a generic failure), the plugin returns a structured `{ error: ... }` result. OpenCode skips the tool execution and feeds the message directly back to the LLM, which usually makes the model change its approach.
5. **Logging** — a warning is written to the app log so you can see when and where loops were cut.

## Installation

Drop the file into your opencode plugins directory:

```bash
mkdir -p ~/.opencode/plugins
# save the code above as ~/.opencode/plugins/loop-detection.ts
```

That's it — opencode picks it up automatically.

OpenCode has a native `doom_loop` permission that controls what happens when the same tool call repeats with identical input. Make sure it is set in your configuration:

```json
{
  ...
  "permission": {
    ...
    "doom_loop": "ask"
  }
}
```

## Results

After adding this plugin, the situation improved dramatically. The model no longer gets stuck in loops; instead, it keeps moving forward and eventually finds a solution.
