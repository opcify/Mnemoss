# Mnemoss

> ACT-R based memory system for AI agents.  
> Recall what matters, when it matters.

**Mnemoss** (pronounced _"nee-moss"_, the M is silent — from Greek 
_Mnemosyne_, goddess of memory) is a memory system for AI agents, 
designed from the first principles of human cognition.

Unlike typical "agent memory" systems that are vector databases with 
extra steps, Mnemoss implements the ACT-R cognitive architecture: it 
ranks memories by activation, considering recency, frequency, context, 
and relevance — not just semantic similarity.

## Status

🚧 **Early development** — Stage 1 MVP in progress. Not yet published 
to PyPI. API may change.

## Why Mnemoss

Most agent memory systems today treat memory as "store and retrieve by 
similarity." But human memory doesn't work that way:

- Recent events recall by their literal details; old events recall by 
  their gist.
- Frequently-used memories stay sharp; unused ones fade but aren't 
  deleted — the right cue can bring them back.
- Important information gets consolidated during idle moments, not just 
  at night.
- Abstractions (facts, patterns) emerge from repeated experience, not 
  from upfront categorization.

Mnemoss implements all of this as a single mathematical system, built on 
a half-century of cognitive science research.

## Core Ideas

- **One formula drives everything** — retrieval ranking, index migration, 
  disposal, reminiscence all emerge from the ACT-R activation equation.
- **One table holds all memory types** — episodes, facts, entities, 
  patterns all live in a unified Memory table.
- **Storage is cheap, retrieval is expensive** — Mnemoss optimizes the 
  latter through multi-tier indexing with latency gradient.
- **Dreaming is opportunistic** — consolidation happens during idle 
  moments, session endings, task completions, and conflicts — not just 
  on a nightly cron.
- **Framework-agnostic** — integrate with OpenClaw, Hermes, Claude Code, 
  Cursor, or your own agent.

## Quick Start (Planned)

```python
import asyncio
from mnemoss import Mnemoss

async def main():
    mem = Mnemoss(workspace="my_agent")
    
    await mem.observe(
        role="user", 
        content="Meeting with Alice tomorrow at 4:20 PM"
    )
    
    results = await mem.recall("What time is the meeting?", k=3)
    for m in results:
        print(m.content)

asyncio.run(main())
```

## Design Documents

- [`MNEMOSS_PROJECT_KNOWLEDGE.md`](./MNEMOSS_PROJECT_KNOWLEDGE.md) — 
  Complete project knowledge base
- [`MNEMOSS_FORMULA_AND_ARCHITECTURE.md`](./MNEMOSS_FORMULA_AND_ARCHITECTURE.md) — 
  The mathematical formula and architecture

## Cognitive Science Roots

Mnemoss builds directly on:

- **ACT-R activation equation** (Anderson & Schooler, 1991)
- **Episodic/Semantic continuum** (Tulving; Moscovitch)
- **Dual-process theory** (Yonelinas, 2002)
- **Event segmentation theory** (Zacks, 2007)
- **Awake replay and opportunistic consolidation** (Foster & Wilson, 2006)

See [`MNEMOSS_PROJECT_KNOWLEDGE.md`](./MNEMOSS_PROJECT_KNOWLEDGE.md) 
Section 2 for the full list.

## License

MIT

## Provenance

Built by **Guangyang Qi** ([@opcify](https://github.com/opcify)) as part of 
the Opcify project. Open-sourced for the broader agent ecosystem.