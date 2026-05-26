# CTSpinoPelvic1K — Reviewer Setup

Thanks for helping review the spine + pelvis segmentations! Each CT already has
an AI-generated *draft* segmentation; your job is to open one case at a time in
**ITK-SNAP**, fix **one** region, save, and quit. The tool handles the rest.

## Open the guide for your computer

- **Windows** → **[REVIEWERS_WINDOWS.md](REVIEWERS_WINDOWS.md)**
- **Mac** → **[REVIEWERS_MAC.md](REVIEWERS_MAC.md)**
- **Linux / Windows WSL** → **[REVIEWERS_LINUX.md](REVIEWERS_LINUX.md)**

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
