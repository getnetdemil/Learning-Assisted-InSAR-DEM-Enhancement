# GitHub Collaboration Guide
### Getnet + Farhan — InSAR DEM Enhancement Contest (deadline: April 06, 2026)

This is a **step-by-step tutorial**, not a reference manual. If you have never collaborated on
GitHub before, start at Section 1 and follow every step in order. After that, use the
[Quick Reference Card](#quick-reference-card) at the bottom for your daily workflow.

---

## Table of Contents

0. [Your Everyday Workflow — A Full Walkthrough](#0-your-everyday-workflow--a-full-walkthrough)
1. [One-Time Setup — Getting In Sync](#1-one-time-setup--getting-in-sync)
2. [Branch Management — The Daily Rhythm](#2-branch-management--the-daily-rhythm)
3. [Issues — The Single Source of Truth](#3-issues--the-single-source-of-truth)
4. [Pull Requests — The Review Gateway](#4-pull-requests--the-review-gateway)
5. [Resolving Merge Conflicts — Step by Step](#5-resolving-merge-conflicts--step-by-step)
6. [GitHub Project Board — Visual Progress Tracking](#6-github-project-board--visual-progress-tracking)
7. [Notifications & Async Communication](#7-notifications--async-communication)

---

## 0. Your Everyday Workflow — A Full Walkthrough

> Read this section **before Section 1**. It shows you the complete picture of a real workday
> so that the detailed steps in Sections 1–7 have context. Come back to this as a cheat-sheet
> after your first week.

This is what a normal working day looks like for both of you, from the moment you open your
laptop to the moment you push your work.

---

### The Big Picture (read this once)

```
Every piece of work follows this loop — no exceptions:

  GitHub Issue          your feature branch          Pull Request
  (the task card)  →   (isolated workspace)   →   (review + merge)
       ↓                       ↓                         ↓
  "I'm working        "I write and test         "Other person checks,
   on #12"             my code here"             approves, and merges"
```

The three golden rules that make this work:
1. **Never commit directly to `main` or `dev`** — always use a feature branch.
2. **Every task has a GitHub Issue** — if it's not in an Issue, it doesn't exist.
3. **Never merge your own PR** — the other person always clicks the merge button.

---

### Scenario A — Starting a Brand New Task

Use this when you are beginning work on something that doesn't have a branch yet.
Step through this in order, every time.

---

#### A1. Check what needs doing (2 min)

Open the GitHub repo → **Projects** → **Contest Pipeline** board.

Look at the `Backlog` column. Find the highest-priority card. If it has no assignee,
assign yourself (right sidebar of the Issue → **Assignees → yourself**).

If there is no Issue yet for what you want to do:
```
Issues → New issue → Task template
Fill: title, goal, acceptance criteria, label, milestone, project
```
Assign it to yourself. Drag it to `In Progress` on the board.

Note the Issue number — you will use it in your commit messages.

---

#### A2. Open your terminal and go to the repo (1 min)

```bash
cd /scratch/$USER/Learning-Assisted-InSAR-DEM-Enhancement
```

Check you have no leftover unfinished work from yesterday:

```bash
git status
# Expected: "nothing to commit, working tree clean"
# If it shows modified files: either commit them or stash them (see A2a below)
```

> **If you have leftover uncommitted changes (A2a):**
> ```bash
> git stash           # temporarily shelves your changes
> # ... do today's setup steps ...
> git stash pop       # brings them back later
> ```

---

#### A3. Get the latest code (1 min)

```bash
git checkout dev
git pull origin dev
```

This ensures your starting point includes everything the other person merged since you
last worked. **Never skip this step** — skipping it causes conflicts later.

Expected output:
```
Switched to branch 'dev'
Already up to date.
   -- or --
Updating a1b2c3..d4e5f6
Fast-forward
 src/models/film_unet.py | 120 ++++++
```

---

#### A4. Create your feature branch (30 sec)

```bash
git checkout -b feature/film-unet
```

Replace `film-unet` with something that matches your Issue. Examples:
```bash
git checkout -b feature/physics-losses     # for Issue: "Add N2N physics losses"
git checkout -b fix/sublook-shape-mismatch # for a bug fix
git checkout -b experiment/lr-ablation     # for a quick experiment
```

Verify you are on the right branch:
```bash
git branch
# * feature/film-unet    ← the * shows your current branch
#   dev
#   main
```

---

#### A5. Do your actual work

Write code. The only rule: **make small commits as you go**, not one giant commit at the end.

After every logical chunk of work (a working function, a passing test, a completed class):

```bash
git status                          # see what changed
git add src/models/film_unet.py     # stage the specific file(s)
git commit -m "feat(film_unet): add encoder with FiLM blocks"
```

**Commit message format:** `type(scope): what you did — refs #12`

| If you... | Use type |
|-----------|----------|
| Added new functionality | `feat` |
| Fixed a bug | `fix` |
| Changed data/manifests | `data` |
| Updated evaluation code | `eval` |
| Ran an experiment | `experiment` |
| Wrote documentation | `docs` |
| Restructured code (no behavior change) | `refactor` |

Always add `— refs #12` (or `closes #12` when the work is complete) so the commit appears
in the Issue timeline on GitHub.

---

#### A6. Push your branch to GitHub (30 sec)

Do this at least once a day, even if not finished. It backs up your work and lets the
other person see what you're doing.

```bash
git push -u origin feature/film-unet
```

After the first push, subsequent pushes are just:
```bash
git push
```

You can now see your branch on GitHub:
```
Repository → Code → branch dropdown (top left, shows "main") → your branch name
```

---

#### A7. When the task is done — open a Pull Request

```bash
git push    # make sure latest changes are on GitHub
```

Go to GitHub. You will see a yellow banner:
```
feature/film-unet had recent pushes — Compare & pull request
```

Click **Compare & pull request**. Then:

1. **Check base = `dev`** (not `main`) — change it if wrong.
2. **Fill the description** — the template auto-populates, fill every field.
3. **Write** `Closes #12` in the description — this auto-closes the Issue on merge.
4. **Reviewers** → select the other person (Getnet or Farhan).
5. **Labels** → match the Issue labels.
6. Click **Create pull request**.

The other person gets an email + web notification. They review and merge it.
You do not click the merge button — they do.

---

#### A8. After your PR is merged

GitHub sends you a notification. Then:

```bash
git checkout dev
git pull origin dev              # bring in the merged work
git branch -d feature/film-unet  # delete the local branch — it's done
```

On GitHub, click **Delete branch** on the merged PR page (if it wasn't auto-deleted).

The Issue auto-closes. The Project Board card moves to `Done`.

---

### Scenario B — Resuming Work Mid-Task (you already have a branch)

Use this every morning when you're continuing work from the day before.

```bash
# Step 1 — go to the repo
cd /scratch/$USER/Learning-Assisted-InSAR-DEM-Enhancement

# Step 2 — check your current state
git status
# Expect: "On branch feature/film-unet, nothing to commit" or some modified files

# Step 3 — pull in anything new from dev (in case the other person merged something)
git fetch origin
git rebase origin/dev
# If this says "Current branch feature/film-unet is up to date" — great, continue working
# If it starts a rebase (other person merged to dev) — it will fast-forward cleanly OR
# pause at a conflict (go to Section 5 to resolve)

# Step 4 — continue your work, commit as you go
# ... write code ...
git add <file>
git commit -m "feat(film_unet): add decoder with skip connections — refs #12"

# Step 5 — push at end of session
git push
```

> **Why `git fetch` + `git rebase` instead of `git pull`?**
> `git pull` on a feature branch does a merge, creating ugly "Merge branch 'dev' into
> feature/film-unet" commits. `rebase` instead replays your commits on top of the latest
> dev, keeping history linear. Make this your habit.

---

### Scenario C — Reviewing Someone Else's PR

The other person opened a PR and requested your review. You got an email.

```bash
# Step 1 — fetch their branch
git fetch origin
git checkout feature/physics-losses    # switch to their branch to run it locally

# Step 2 — test it locally
conda run --prefix /scratch/$USER/hrwsi_s3client/torch-gpu \
    python -c "from src.losses.physics_losses import noise2noise_loss; print('import OK')"
# Run whatever test command they put in the PR description

# Step 3 — go back to GitHub to leave your review
```

On GitHub → PR page → **Files changed** tab:
- Hover over a line → click `+` → type a comment → **Start a review** (not single comment)
- After all comments: **Finish your review** → **Approve** or **Request changes**

If you approve:
- Click the dropdown next to **Merge pull request** → **Squash and merge**
- Confirm
- Click **Delete branch**

Done. The other person gets a notification.

---

### Scenario D — Your PR Has a Conflict

You opened a PR and GitHub shows: `This branch has conflicts that must be resolved`.

```bash
# Step 1
git fetch origin

# Step 2 — rebase onto the updated dev
git checkout feature/film-unet
git rebase origin/dev

# Step 3 — Git pauses and tells you which file has a conflict:
# CONFLICT (content): Merge conflict in src/models/film_unet.py

# Step 4 — open the file, find the markers:
# <<<<<<< HEAD          ← dev's version
# ...
# =======
# ...
# >>>>>>> your commit   ← your version
# Edit the file to the correct combined result, remove all markers

# Step 5 — mark resolved and continue
git add src/models/film_unet.py
git rebase --continue     # repeat Steps 3-5 if more conflicts remain

# Step 6 — push
git push --force-with-lease
```

The PR updates automatically. The conflict banner disappears.

---

### Scenario E — Someone Pushed Directly to `main` (what NOT to do)

If you ever see this in `git log`:

```
abc1234 (HEAD -> main, origin/main) added the film unet file
```

...that means someone pushed directly to `main`, bypassing review. This is the mistake
the branch protection rules prevent. If it happens anyway:

1. Check `git log main` — was this your commit or the other person's?
2. Do NOT revert blindly. Leave a comment on the related Issue explaining what happened.
3. Going forward: always run `git branch` to confirm you're NOT on `main` before coding.

The safe habit before starting any work:
```bash
git branch      # check which branch you're on
# if it shows: * main  ← STOP, do not commit here
git checkout dev
git checkout -b feature/your-new-branch
```

---

### Daily Checklist (bookmark this)

```
MORNING (before writing any code):
  □ git checkout dev
  □ git pull origin dev
  □ Check GitHub notifications — any PR review requests?
  □ Check Project Board — what's your card for today?
  □ git checkout -b feature/<name>   (or git checkout feature/<existing> if resuming)
  □ git fetch origin && git rebase origin/dev   (if resuming)

WHILE WORKING:
  □ git add <file> && git commit -m "type(scope): message — refs #N"
  □ Commit every logical chunk — not once at the end of the day
  □ git push   (at least once during the session)

END OF DAY:
  □ git push   (make sure everything is backed up on GitHub)
  □ If the task is done: open a PR (base=dev, reviewer=other person)
  □ If not done: leave a comment on the Issue with today's progress
  □ Reply to any PR review comments you received

WEEKLY (Monday meeting):
  □ Review Project Board together — drag stale cards, pick new ones
  □ Check Milestone progress bar — on track for April 06?
  □ Agree on who owns what for the week
```

---

## 1. One-Time Setup — Getting In Sync

Do this section **exactly once**, the first day you join the project. Both of you need to
finish all of Section 1 before starting any real work.

---

### Step 1 — Create a GitHub account (if you don't have one)

Go to [github.com](https://github.com) → **Sign up**. Use your university email so Getnet
can find you easily. Share your GitHub username with each other before the next step.

---

### Step 2 — Accept the Collaborator Invitation (Farhan does this)

**Getnet** opens the repository on GitHub and invites Farhan:

```
Repository page → Settings (top menu) → Collaborators and teams (left sidebar)
→ Add people → type Farhan's GitHub username → Send invitation
```

**Farhan** checks his email (subject: `[GitHub] Getnet invited you to collaborate`) and clicks
the **Accept invitation** link. Alternatively, check `github.com/notifications`.

After accepting, Farhan has **Write access** — he can push branches, open PRs, and comment.
He cannot delete the repository or change Settings (only the owner can).

> **Verify it worked:** Farhan visits the repository URL. If he can see the green "Code" button
> and the repo contents, access is working.

---

### Step 3 — Clone the Repository to Your Machine

Open a terminal. Run:

```bash
git clone https://github.com/getnetdemil/Learning-Assisted-InSAR-DEM-Enhancement.git
cd Learning-Assisted-InSAR-DEM-Enhancement
```

You now have a local copy. The remote (GitHub) is automatically named `origin`.

> **What just happened?** `git clone` downloaded the entire repository — all files, all history
> — to a folder on your machine. `origin` is just a nickname for the GitHub URL.

Verify you're in the right place:

```bash
git remote -v
# Should print:
# origin  https://github.com/getnetdemil/Learning-Assisted-InSAR-DEM-Enhancement.git (fetch)
# origin  https://github.com/getnetdemil/Learning-Assisted-InSAR-DEM-Enhancement.git (push)
```

---

### Step 4 — Tell Git Who You Are

Git needs your name and email to label your commits. Run these **once per machine**:

```bash
git config user.name  "Your Full Name"
git config user.email "your-github-email@example.com"
```

Use the **same email address** as your GitHub account so GitHub links your commits to your
profile photo and username.

Verify:

```bash
git config --list | grep user
# user.name=Your Full Name
# user.email=your-github-email@example.com
```

---

### Step 5 — Set Up the Python Environment

```bash


# If you need to build the environment from scratch:
conda create -n insar-dem python=3.10
conda activate insar-dem
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
pip install -e .
pip install boto3 pystac capella-reader dask geopandas
conda install -c conda-forge isce3


```


---

### Step 6 — Protect the `main` Branch (Getnet does this once)

**What is branch protection?** It prevents anyone — including the repo owner — from
accidentally pushing broken code directly to `main`. Every change must be reviewed first.

```
Repository page → Settings → Branches (left sidebar) → Add branch protection rule
```

Fill in:
- **Branch name pattern**: `main`

Check these boxes:
- [x] Require a pull request before merging
  - [x] Require approvals: set to **1**
  - [x] Dismiss stale pull request approvals when new commits are pushed
- [x] Do not allow bypassing the above settings

Click **Create**. Now if either of you tries `git push origin main`, GitHub will reject it
with an error. Good — that's the point.

Do the same for `dev` (same settings, same steps, just change the branch name to `dev`).

---

### Step 7 — Create the `dev` Integration Branch

**What is `dev`?** It's the shared "work in progress" branch. You both merge your finished
features here first. Only when `dev` is stable and complete does it get merged into `main`.

Think of it like this:
```
main  = published, stable, always works
dev   = shared sandbox where today's work lives
```

Getnet creates `dev` (only needs to happen once):

```bash
git checkout main
git pull origin main
git checkout -b dev        # create the branch locally
git push -u origin dev     # publish it to GitHub
```

Farhan then downloads it:

```bash
git fetch origin
git checkout dev
```

Both of you should now be on `dev`:

```bash
git branch
# * dev
#   main
```

---

### Step 8 — Verify Everything Is Working

Both of you run this checklist:

```bash
# 1. Are you in the repo?
git status
# Should say: On branch dev, nothing to commit

# 2. Can you see both branches?
git branch -a
# Should list: dev, main, remotes/origin/dev, remotes/origin/main

# 3. Can you reach the remote?
git fetch origin
# Should complete with no errors

# 4. Does Python work?
conda run --prefix /scratch/$USER/hrwsi_s3client/torch-gpu \
    python -c "import torch, rasterio, boto3; print('OK')"
```

If all four pass, you are ready to start working. Section 1 is complete — you never need
to repeat it.

---

## 2. Branch Management — The Daily Rhythm

### What Is a Branch?

Imagine the codebase as a document. A branch is like making a **private copy** of that
document to edit, without affecting the original. When you're done and happy with your
changes, you "merge" your copy back in.

This means you and Getnet can each work on different parts of the code simultaneously
without breaking each other's work.

### The Branch Hierarchy

```
main  ←──────────────────────────── (stable releases only — reviewed PR required)
  │
  └── dev  ←────────────────────── (daily integration — all feature branches land here)
        │
        ├── feature/film-unet       ← Getnet working on the neural network
        ├── feature/physics-losses  ← Farhan working on the loss functions
        ├── experiment/n2n-ablation ← quick ablation, may be discarded
        └── fix/goldstein-edge-case ← bug fix
```

Each person works on their own branch. They never edit each other's branches.

---

### Tutorial: Starting Work on a New Feature

Let's say Farhan is going to implement `src/losses/physics_losses.py`.

**Step 1 — Get the latest `dev`**

Always start from a fresh `dev`, not wherever you left off yesterday:

```bash
git checkout dev
git pull origin dev
```

> **Why?** If Getnet merged something to `dev` since you last worked, you want those changes
> in your starting point. Otherwise you'll hit conflicts later.

**Step 2 — Create a new branch**

```bash
git checkout -b feature/physics-losses
```

> **What happened?** Git created a new branch that starts from the current state of `dev`,
> and switched you to it. Your changes will stay isolated here until you're ready to merge.

Verify you're on the right branch:

```bash
git branch
# * feature/physics-losses
#   dev
#   main
```

**Step 3 — Do your work**

Edit files, write code, run tests. Use `git status` any time to see what you've changed:

```bash
git status
# Changes not staged for commit:
#   modified:   src/losses/physics_losses.py
```

**Step 4 — Commit your changes**

A commit is a named snapshot. Make small, logical commits as you go:

```bash
git add src/losses/physics_losses.py
git commit -m "feat(physics_losses): add Noise2Noise sub-look loss"
```

Commit message format: `type(scope): what you did`

| Type | Meaning |
|------|---------|
| `feat` | New functionality |
| `fix` | Bug fix |
| `data` | Data pipeline / manifests |
| `eval` | Metrics / evaluation |
| `experiment` | Ablation / exploratory |
| `docs` | Documentation only |
| `refactor` | Restructure without changing behaviour |

**Step 5 — Push your branch to GitHub**

```bash
git push -u origin feature/physics-losses
```

> The `-u origin feature/physics-losses` part tells Git where to push. You only need `-u`
> the first time. After that, just `git push` works.

Now your branch exists on GitHub and can be seen by Getnet.

---

### Staying In Sync While You Work

If you're on a feature branch for several days, `dev` might get new commits from the
other person. To bring those into your branch:

```bash
git fetch origin           # download any remote changes (doesn't touch your files yet)
git rebase origin/dev      # replay your commits on top of the latest dev
```

> **Why `rebase` instead of `merge`?** Merge creates an extra "merge commit" that clutters
> the history. Rebase keeps the commit history as a clean, straight line — much easier to
> read with `git log`. Think of it as: "pretend my branch was created from the *current*
> dev, not the dev from 3 days ago."

If you haven't committed anything yet and just want the latest dev:

```bash
git pull origin dev
```

---

### Branch Naming Rules

| Prefix | When to use | Example |
|--------|-------------|---------|
| `feature/` | New capability from `plan.md` | `feature/film-unet` |
| `fix/` | Bug fix | `fix/goldstein-edge-case` |
| `experiment/` | Ablation or one-off test | `experiment/n2n-lr-sweep` |
| `docs/` | Documentation only | `docs/reproducibility-md` |

Rules:
- All lowercase
- Words separated by hyphens, not underscores
- Keep it short and descriptive (≤ 40 characters)
- No spaces or special characters

---

### Deleting a Branch After It's Merged

Once your PR is merged, the branch is no longer needed. Delete it to keep things tidy.

On GitHub: the merged PR page shows a **"Delete branch"** button — click it.

Locally:

```bash
git checkout dev
git pull origin dev
git branch -d feature/physics-losses
```

---

### The One Rule You Must Never Break

**Never push directly to `main` or `dev`.** Always use a feature branch and open a PR.
If you try, GitHub will reject the push (because of the branch protection you set up in
Section 1). This is by design.

---

## 3. Issues — The Single Source of Truth

### What Is a GitHub Issue?

An Issue is a task card. It describes a piece of work, who owns it, and whether it's done.
Think of it as a sticky note on a shared board that both of you can see and update.

**The rule:** Every piece of work — big or small — starts with an Issue. No Issue = the
work doesn't officially exist. This prevents you from stepping on each other's code and
losing context in Slack/email threads.

---

### Tutorial: Creating an Issue

Go to the repository on GitHub → click **Issues** (top menu) → **New issue**.

You'll see two template options appear:
- **Task** — for planned work from `plan.md`
- **Bug report** — for something that is broken

Click **Task** → **Get started**.

The form auto-populates with the template from `.github/ISSUE_TEMPLATE/task.md`. Fill it in:

**Title:** Use the format `[Phase N] Short imperative description`

Good titles:
```
[Phase 2] Implement FiLM-conditioned U-Net
[Phase 2] Add Noise2Noise sub-look loss
[Phase 3] Compute triplet closure error metric
```

Bad titles (too vague):
```
model stuff
fix the thing
Farhan's task
```

**Right sidebar — fill these in before saving:**

- **Assignees**: click the gear → select whoever will do the work
- **Labels**: pick the relevant labels (see below)
- **Milestone**: select "Contest Submission — April 06, 2026"
- **Projects**: select "Contest Pipeline" (the board from Section 6)

Click **Submit new issue**.

---

### Setting Up Labels (Getnet does this once)

```
Repository → Issues → Labels → New label
```

Create these labels:

| Label name | Hex color | Use it for |
|-----------|-----------|-----------|
| `phase-1` | `#0075ca` | Data pipeline work |
| `phase-2` | `#e4e669` | Model + training work |
| `phase-3` | `#d93f0b` | Evaluation + paper |
| `model` | `#7057ff` | Neural network code |
| `data` | `#008672` | S3 / preprocessing / manifests |
| `eval` | `#e99695` | Metrics, closure error |
| `scripts` | `#f9d0c4` | CLI scripts in `scripts/` |
| `bug` | `#d73a4a` | Something is broken |
| `question` | `#d876e3` | Needs discussion or decision |
| `blocked` | `#b60205` | Waiting on something to proceed |

---

### Setting Up the Milestone (Getnet does this once)

```
Repository → Issues → Milestones → New milestone
  Title:    Contest Submission — April 06, 2026
  Due date: 2026-04-06
```

Attach every Issue you create to this milestone (in the right sidebar). GitHub then shows a
progress bar — `12 of 30 issues closed (40%)` — so you can track overall contest readiness.

---

### Linking Issues to Commits

When you commit work that resolves an Issue, include `closes #N` in the message:

```bash
git commit -m "feat(film_unet): add FiLM conditioning blocks — closes #12"
```

When the PR containing this commit is merged, GitHub **automatically closes Issue #12**
and posts a link from the Issue to the PR. You never have to manually close issues.

Other keywords:
- `closes #N` or `fixes #N` — closes on merge (use for bugs)
- `refs #N` — links without closing (use for partial progress)

---

### Using Issue Comments for Async Discussion

Instead of texting "hey, should the Goldstein filter use a 2D or 1D FFT?" — open the
relevant Issue and leave a comment:

```
@getnetdemil — should the Goldstein filter window be 2D (full patch) or 1D (per row)?
The original Goldstein 1998 paper uses 2D, but our patch size might make it slow.
```

Benefits:
- The answer is permanently attached to the task
- Either of you can reply hours later without losing context
- No "what did we decide?" moments

To tag someone: type `@` + their GitHub username. They get an email + web notification
immediately.

---

## 4. Pull Requests — The Review Gateway

### What Is a Pull Request?

A Pull Request (PR) is a formal request to merge your branch into another branch. It shows
exactly what changed, lets the other person review and comment, and requires approval before
anything lands in `dev` or `main`.

Think of it as: "I finished my work — please look it over before we make it official."

**The rule:** Every change — even a one-line fix — goes through a PR. No direct pushes to
`dev` or `main`.

---

### Tutorial: Opening Your First PR

Assume Farhan finished `feature/physics-losses` and pushed it. Here's what he does next.

**Step 1 — Push the branch (if you haven't already)**

```bash
git push -u origin feature/physics-losses
```

**Step 2 — Go to GitHub**

Visit the repository. GitHub shows a yellow banner at the top:

```
physics-losses had recent pushes — Compare & pull request
```

Click **Compare & pull request**. If the banner is gone, click **Pull requests → New pull
request** and manually set base = `dev`, compare = `feature/physics-losses`.

**Step 3 — Check the base branch**

This is the most common beginner mistake. Make sure:
- **base**: `dev`   ← where your code is going
- **compare**: `feature/physics-losses`   ← your branch

If base is set to `main` by default, change it to `dev`.

**Step 4 — Fill the PR description**

The text box auto-populates from `.github/pull_request_template.md`. Fill every section:

```markdown
## What this does
Implements the three self-supervised losses in src/losses/physics_losses.py:
Noise2Noise (sub-look pair), closure-consistency (triplet), and temporal consistency.
All losses return a scalar tensor compatible with the FiLM U-Net training loop.

## How to test it
```bash
conda run --prefix /scratch/$USER/hrwsi_s3client/torch-gpu python -c "
from src.losses.physics_losses import noise2noise_loss, closure_loss
import torch
pred = torch.randn(2, 2, 256, 256, requires_grad=True)
target = torch.randn(2, 2, 256, 256)
loss = noise2noise_loss(pred, target)
loss.backward()
print('N2N loss:', loss.item())   # should be a finite positive number
"
```

## Related issue
Closes #8
```

**Step 5 — Set the reviewer and labels**

In the right sidebar:
- **Reviewers**: click the gear → select Getnet
- **Labels**: match the labels on the related Issue
- **Projects**: "Contest Pipeline"

**Step 6 — Submit**

Click **Create pull request**.

Getnet gets an email + web notification that a review is requested.

---

### Tutorial: Reviewing a PR (the reviewer's perspective)

Getnet received a review request. Here's what to do.

**Step 1 — Open the PR on GitHub**

Go to **Pull requests** → click the PR title.

**Step 2 — Read the description**

Does it make sense? Does it say what changed and how to test it?

**Step 3 — Review the code**

Click **Files changed** (tab near the top of the PR). You see every line that was added
(green) or removed (red).

To leave a comment on a specific line:
1. Hover over the line — a blue `+` button appears on the left
2. Click it → type your comment
3. Click **Start a review** (not "Add single comment")

> **Why "Start a review" not "Add single comment"?** Start a review batches all your
> comments and sends them as **one notification** instead of bombing Farhan with ten
> separate emails.

Add all your comments, then click **Finish your review** (green button, top right of
Files changed). Choose:
- **Approve** — looks good, ready to merge
- **Request changes** — there are problems that must be fixed first
- **Comment** — neutral observations, no blocking

**Step 4 (if approved) — Merge the PR**

```
PR page → Merge pull request dropdown → Squash and merge → Confirm squash and merge
```

> **Why Squash and merge?** Instead of adding all of Farhan's intermediate "WIP" commits
> to `dev`, it squashes them into **one clean commit** with a clear summary. `git log` on
> `dev` stays readable.

Then click **Delete branch** on the merged PR page.

---

### Tutorial: Responding to Review Comments (the author's perspective)

Getnet left some comments requesting changes. Farhan:

1. Reads each comment carefully on the PR page.
2. Makes the code changes locally on `feature/physics-losses`.
3. Commits and pushes — the PR updates automatically:
   ```bash
   git add src/losses/physics_losses.py
   git commit -m "fix: address review comments — clamp loss, add docstring"
   git push
   ```
4. Goes back to the PR page and replies to each comment ("Fixed in latest push" is fine).
5. Clicks **Resolve conversation** on each addressed comment.
6. Tags Getnet when ready: `@getnetdemil — all comments addressed, ready for re-review`

---

### Merge Rules Summary

| Situation | Merge type | Who merges |
|-----------|-----------|-----------|
| `feature/*` → `dev` | Squash and merge | The reviewer (not the author) |
| `dev` → `main` | Create a merge commit | The reviewer (not the author) |

**Never merge your own PR.** The other person always clicks the final merge button.

---

## 5. Resolving Merge Conflicts — Step by Step

### What Is a Merge Conflict?

A conflict happens when two people edited **the same part of the same file** in different
branches, and Git doesn't know which version to keep. It needs you to decide.

This sounds scary, but it is a normal, expected part of collaborative development. It
happens even to experienced teams. The process to resolve it is always the same.

---

### How You Know You Have a Conflict

You'll see one of these:

**On GitHub (in your open PR):**
```
This branch has conflicts that must be resolved
```

**In your terminal (after a rebase):**
```
CONFLICT (content): Merge conflict in src/models/film_unet.py
error: could not apply a3f9b21... feat(film_unet): add FiLM decoder
```

---

### Tutorial: Resolving a Conflict

Let's say your PR's branch `feature/film-unet` has a conflict with `dev`.

**Step 1 — Download the latest state**

```bash
git fetch origin
```

This downloads the latest `dev` without changing your files yet.

**Step 2 — Rebase onto the updated `dev`**

```bash
git checkout feature/film-unet
git rebase origin/dev
```

Git will pause and tell you which file has a conflict:

```
CONFLICT (content): Merge conflict in src/models/film_unet.py
Auto-merging src/models/film_unet.py
error: could not apply a3f9b21... feat(film_unet): add FiLM decoder
hint: Resolve all conflicts manually, mark them as resolved with
hint: "git add <conflicted files>", then run "git rebase --continue".
```

**Step 3 — Open the conflicted file in your editor**

Open `src/models/film_unet.py`. Git marks the conflict like this:

```python
<<<<<<< HEAD
# This is the version from dev (the other person's code)
def forward(self, x, cond):
    return self.decoder(self.encoder(x), cond)
=======
# This is YOUR version from feature/film-unet
def forward(self, x, cond, mask=None):
    feat = self.encoder(x)
    return self.decoder(feat, cond)
>>>>>>> a3f9b21 (feat: add FiLM decoder)
```

The section between `<<<<<<< HEAD` and `=======` is what's in `dev`.
The section between `=======` and `>>>>>>>` is what's in your branch.

**Step 4 — Edit the file to the correct result**

Delete the conflict markers and write the version that should actually be there. Often
you want to combine both changes:

```python
# Combined: keeps the mask parameter from your branch, keeps dev's encoder structure
def forward(self, x, cond, mask=None):
    feat = self.encoder(x)
    return self.decoder(feat, cond)
```

Make sure the file has **no remaining** `<<<<<<<`, `=======`, or `>>>>>>>` markers.

**Step 5 — Mark the conflict as resolved**

```bash
git add src/models/film_unet.py
```

If there are more conflicted files, fix each one the same way and `git add` them.

**Step 6 — Continue the rebase**

```bash
git rebase --continue
```

Git applies the next commit in sequence. If there are more conflicts, repeat Steps 3–6.
When it finishes, your branch is now based on the latest `dev`.

**Step 7 — Push the resolved branch**

```bash
git push --force-with-lease
```

> **Why `--force-with-lease` and not just `git push`?** After a rebase the commit history
> has changed, so a normal push is rejected. `--force-with-lease` allows overwriting the
> remote branch, but only if no one else pushed to it since your last fetch — a safety net.
>
> **Never use `git push --force` on `dev` or `main`.** The `--force-with-lease` flag is
> only safe on your own feature branches.

The PR on GitHub now shows the updated branch and the conflict banner disappears.

---

### If Things Go Wrong — The Escape Hatch

If the rebase is going badly and you want to undo everything:

```bash
git rebase --abort
```

This puts you back to where you were before you started the rebase. Nothing is lost.

---

### For Simple Conflicts: GitHub's Online Editor

If the conflict is small (e.g., one line in a config file), you can fix it directly
on GitHub without touching the terminal:

```
PR page → "Resolve conflicts" button (gray, near the conflict banner)
→ Edit the file in the browser (same conflict markers)
→ Click "Mark as resolved"
→ Click "Commit merge"
```

Use this shortcut for simple cases only. For code conflicts, always use the local method.

---

### Communication Rule for Conflicts

If a conflict involves code the **other person wrote**, leave an Issue comment before
you start resolving it:

```
@getnetdemil — I have a conflict in geometry.py between my B_perp refactor (#14)
and your state-vector changes (#11). My plan: keep the interpolation logic from #11
and wrap it with the new interface from #14. Let me know in the next few hours if
you see a problem — otherwise I'll push the resolution.
```

This prevents one person silently undoing the other's work.

---

## 6. GitHub Project Board — Visual Progress Tracking

### What Is a Project Board?

A Project Board is a Kanban board (like Trello) built into GitHub. Cards represent Issues.
You move cards across columns as work progresses. It gives both of you a single, shared
view of what's happening without needing a status-update message.

---

### Tutorial: Creating the Board (Getnet does this once)

```
Repository page → Projects tab → New project → Board (Kanban view)
Name: Contest Pipeline
→ Create project
```

GitHub creates three default columns. Rename them:

| Column name | What it means |
|-------------|--------------|
| `Backlog` | Issue exists but nobody is working on it yet |
| `In Progress` | Someone is actively working on this right now |
| `In Review` | A PR is open for this — waiting for review |
| `Done` | PR merged, issue closed |

To rename a column: click the `...` on the column header → **Edit**.

---

### Tutorial: Automating Card Movement

Instead of manually dragging cards, set up GitHub Automation:

```
Project board → ⋯ (top right) → Settings → Workflows (left sidebar)
```

Enable these workflows:

| Trigger | Action |
|---------|--------|
| Item added to project | Set status → `Backlog` |
| Pull request opened | Set status → `In Review` |
| Pull request merged | Set status → `Done` |
| Issue closed | Set status → `Done` |

Now when you open a PR, the card automatically moves to `In Review`. When it merges, it
moves to `Done`. You only need to manually drag cards to `In Progress`.

---

### Tutorial: Adding Issues to the Board

Two ways:

**Option A — When creating the Issue:** In the right sidebar of the new issue form, click
the gear next to **Projects** → select **Contest Pipeline**. The Issue lands in `Backlog`.

**Option B — From the board:** Open the board → `+ Add item` at the bottom of `Backlog`
→ type `#` and the issue number or title.

When you start working on an issue, drag its card from `Backlog` to `In Progress`.

---

### Weekly Meeting Ritual

Every week (pick a fixed day — Monday works well):

1. **Screen-share the Project Board** — both of you look at it together.
2. **Review `In Progress`** — is anything stuck? If it's been in progress for more than
   a week, discuss the blocker and add a `blocked` label.
3. **Triage `Backlog`** — pick the highest-priority cards for this week, drag to
   `In Progress`, assign owners.
4. **Check the Milestone** — go to **Issues → Milestones → Contest Submission**.
   The progress bar shows how many issues are closed. Are you on pace for April 06?

The board replaces most "what are you working on?" messages.

---

### Pin the Board to the Repo

```
Repository → Projects → (hover over Contest Pipeline) → ⋯ → Pin to repository
```

This puts the board one click from the repository home page.

---

## 7. Notifications & Async Communication

### The Async Challenge

You're working at different times. The goal is: **nothing important gets missed**, but
neither of you is buried in notifications.

---

### Tutorial: Configure Notifications

```
github.com → click your avatar (top right) → Settings → Notifications
```

Recommended settings:

| Event | Email | Web |
|-------|-------|-----|
| Participating (you're involved) | Yes | Yes |
| @mentions | Yes | Yes |
| PR review requests | Yes | Yes |
| Issue assignments | Yes | Yes |
| Watching (all activity on repo) | No | Yes |

Turning off email for "Watching" prevents an inbox flood while still showing a badge on
github.com.

---

### How to Ping Each Other

In any Issue or PR comment, type `@` followed by the GitHub username:

```
@farhan-humayun — the sub-look split in sublook.py returns different shapes for
odd vs. even patch sizes. Can you check whether physics_losses.py handles this?
The relevant line is sublook.py:87.
```

The tagged person gets an email + a web notification badge immediately — no Slack needed.

---

### What Channel to Use for What

| Question / situation | Use this |
|----------------------|----------|
| "I found a bug in your code" | Open a GitHub Issue (Bug template) |
| "Should we use L1 or L2 loss for N2N?" | GitHub Discussion or Issue comment |
| "Review my code, it's ready" | Open a PR + request review |
| "Line 42 should use `torch.exp` not `torch.log`" | PR line comment |
| "Can we reschedule Tuesday's meeting?" | Slack / WhatsApp |
| "URGENT: the training job crashed" | Slack / WhatsApp |

Keep design decisions in GitHub so they're searchable later. Use Slack only for logistics
and emergencies.

---

### Draft PRs for Early Feedback

If you want a second opinion before you're done, don't wait. Open a **Draft PR**:

1. Push your in-progress branch: `git push -u origin feature/film-unet`
2. Open a PR as usual, but click the dropdown arrow next to **Create pull request**
   → **Create draft pull request**
3. Add `[WIP]` to the title: `[WIP] feat: FiLM-conditioned U-Net`
4. Leave a comment: `@getnetdemil — WIP, but can you check the FiLM block architecture
   in film_unet.py:45–80? Want to make sure the conditioning is applied correctly.`

A draft PR cannot be merged until you click **"Ready for review"**. Getnet can still
browse and comment on it.

---

### GitHub Discussions for Architecture Debates

For open-ended questions where you need to think together — not a quick comment:

```
Repository → Discussions tab → New discussion
```

Create categories like "Design Decisions" and "Questions". Discussions are threaded and
searchable, unlike Slack which disappears after a few days.

To enable Discussions:
```
Settings → General → Features → check "Discussions"
```

---

## Quick Reference Card

Cut this out and keep it next to your terminal. It covers 90% of daily usage.

```
╔══════════════════════════════════════════════════════════════╗
║                  DAILY WORKFLOW                              ║
╠══════════════════════════════════════════════════════════════╣
║  START NEW TASK                                              ║
║  1. Create GitHub Issue → assign yourself → add to board     ║
║  2. git checkout dev                                         ║
║  3. git pull origin dev                                      ║
║  4. git checkout -b feature/<descriptive-name>               ║
║                                                              ║
║  WHILE WORKING                                               ║
║  git add <file>                                              ║
║  git commit -m "type(scope): what you did"                   ║
║  git push   (or: git push -u origin <branch> first time)    ║
║                                                              ║
║  DAILY SYNC (run every morning)                              ║
║  git fetch origin                                            ║
║  git rebase origin/dev                                       ║
║                                                              ║
║  SUBMIT FOR REVIEW                                           ║
║  git push                                                    ║
║  → GitHub: open PR, base=dev, fill template, request review  ║
║  → Wait for approval → other person squash-merges            ║
║  → Delete branch (GitHub button) → issue auto-closes         ║
║                                                              ║
║  CONFLICT RESOLUTION                                         ║
║  git fetch origin                                            ║
║  git rebase origin/dev                                       ║
║  # fix <<<<< markers in editor                               ║
║  git add <file>                                              ║
║  git rebase --continue                                       ║
║  git push --force-with-lease                                 ║
║                                                              ║
║  RULES TO NEVER BREAK                                        ║
║  - Never push to main or dev directly                        ║
║  - Never merge your own PR                                   ║
║  - Never git push --force on dev or main                     ║
║  - Every task starts with a GitHub Issue                     ║
╚══════════════════════════════════════════════════════════════╝
```

---

*Last updated: 2026-03-07 — Getnet Demil*
