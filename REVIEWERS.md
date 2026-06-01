# CTSpinoPelvic1K — Reviewer Setup

Thanks for helping review the spine + pelvis segmentations! Each CT already has
an AI-generated *draft* segmentation; your job is to open one case at a time in
**ITK-SNAP**, fix **one** region, save, and quit. The tool handles the rest.

## What's different now (triaged + fast + a clear goal)

You only review the **~210 cases** that **automated quality checks flagged** as
likely wrong — not all 800+. Each case is fast and gives you a measurable goal:

- On open, the terminal prints **`WHY FLAGGED - focus your edit here:`** — the
  specific problem(s) to fix (a leak, a mixed-up vertebra, a stray piece, …).
- It downloads a small **crop** (seconds, not a 200 MB volume) and opens it in
  ITK-SNAP, with a **gold reference example** in a second window to compare to.
- **Every time you Save (Ctrl-S), the terminal re-runs the checks** and shows
  `off-bone leak 0.07 -> 0.01 OK`. **A case is "done" when the checks read
  `OK`.** Edit → save → watch them clear → quit when OK.

## How to actually fix a case → **[REVIEWERS_FIXING.md](REVIEWERS_FIXING.md)**

If you've never edited a segmentation, **read this first** — it covers the
ITK-SNAP tools, how to tell bone from soft tissue on the CT, how to identify a
vertebra level, and a fix recipe for each kind of flag.

## Open the setup guide for your computer

- **Windows** → **[REVIEWERS_WINDOWS.md](REVIEWERS_WINDOWS.md)**  *(recommended —
  no WSL/Linux graphics headaches)*
- **Mac** → **[REVIEWERS_MAC.md](REVIEWERS_MAC.md)**
- **Linux / Windows WSL** → **[REVIEWERS_LINUX.md](REVIEWERS_LINUX.md)**
  *(needs a one-time `libxcb-cursor0` step for ITK-SNAP)*

## How you sign in (all platforms)

You sign in with a **free HuggingFace account** — that's your identity for the
review (your username is recorded with your work). There's **no separate key**
to collect.

1. Make a free account at <https://huggingface.co/join>.
2. Create a **Read** token: Settings → Access Tokens → **New token** → type
   **Read** → copy it.
3. Your guide tells you when to run `hf auth login` and paste that token.

Once you're logged in to HuggingFace, the review tool knows who you are — that's
all most reviewers need. **If you'll be a senior adjudicator**, send the project
lead your HuggingFace **username** so they can enable adjudication for you.

## Optional: faster editing with AI

If a draft needs heavy re-drawing, you can speed it up with ITK-SNAP's
nnInteractive AI tool (free, runs on Google Colab — no GPU of your own needed).
It's entirely optional. See **[REVIEWERS_AI_EDITING.md](REVIEWERS_AI_EDITING.md)**.
