# Publish to GitHub

This guide creates a clean GitHub repository whose commit author and committer
are Karan Chandra Dey. Third-party model/license attribution remains in the
documentation because legal attribution is different from repository
authorship.

## 1. Prepare your GitHub identity

Create or sign in to the GitHub account that should own the repository. Add a
verified email to that account, or enable GitHub's private no-reply email.

Install and authenticate the GitHub CLI if you want the one-command publishing
path:

```bash
brew install gh
gh auth login
gh auth status
```

Authentication controls which GitHub account owns the remote repository. Git
author metadata is configured separately below.

## 2. Audit the release tree

From the project root, confirm that local-only material is absent or ignored:

```bash
git status --short --ignored 2>/dev/null || true

find . -type f -size +50M \
  -not -path './.git/*' \
  -not -path './node_modules/*' \
  -print
```

Do not publish any of the following:

- `.env.local` or API/Hugging Face tokens;
- `node_modules`, Python environments, bytecode, or build output;
- `backend/.runtime`, model weights, Hub caches, uploaded videos, job state,
  tensors, thumbnails, or test results containing creator media;
- local browser/playwright sessions, OS metadata, temporary files, or machine
  paths; or
- third-party model files whose license does not permit redistribution.

Before staging the bundled files under `public/assets/brain/`, verify the
redistribution terms for the exact Nilearn/FreeSurfer/NITRC/Destrieux source
artifacts and the modified derivatives. Preserve the complete FreeSurfer terms
and required preface in `third_party/licenses/FREESURFER-LICENSE.txt` and the
corresponding explanation in `NOTICE.md`. If the applicable rights or notice
requirements are not confirmed, do not publish those binaries; replace them
with a documented local acquisition/build step after obtaining appropriate
permission.

The repository `.gitignore` is designed to exclude these paths, but review the
staged file list yourself. An ignore file is a guardrail, not a secret scanner.

Optionally run a secret scanner before the first commit:

```bash
brew install gitleaks
gitleaks detect --no-git --source .
```

Investigate every result. Never “fix” a finding by merely adding a file that
already contains a secret to `.gitignore`; remove the secret from the working
tree and rotate it first.

## 3. Create clean Git history under your identity

If this directory is not already a Git repository:

```bash
git init -b main
```

Set repository-local identity. Replace the email placeholder with an address
verified by the GitHub account that will own the project:

```bash
git config --local user.name "Karan Chandra Dey"
git config --local user.email "YOUR_GITHUB_VERIFIED_EMAIL"
```

Stage and review the exact release:

```bash
git add .
git status --short
git diff --cached --stat
git diff --cached --check
```

Inspect any file you do not recognize before committing. In particular, there
should be no staged `.env.local`, model checkpoint, video, runtime result, or
dependency directory.

Create the initial commit without a co-author trailer:

```bash
git commit -m "Initial public release"
```

Verify both identities and the complete commit message:

```bash
git log -1 --format='Author: %an <%ae>%nCommitter: %cn <%ce>%n%n%B'
```

The author and committer should both be Karan Chandra Dey, and the message
should contain no `Co-authored-by` line. GitHub associates the contribution
with your account when the commit email matches a verified account email.

If this checkout already has history, inspect every author first:

```bash
git log --format='%aN <%aE>' | sort -u
```

Do not rewrite a repository that other people already consume. If the goal is
a new, single-author public release and the existing local history contains
other authors, create a new empty GitHub repository from a separately audited
release tree instead of publishing that prior history.

## 4. Create and push the GitHub repository

### Recommended: private review, then public

Create a private repository first so the uploaded tree can be checked in the
GitHub interface:

```bash
gh repo create signalframe \
  --private \
  --source=. \
  --remote=origin \
  --push
```

Open it:

```bash
gh repo view --web
```

Review the Files, Commits, Contributors, Dependency graph, and repository
settings. When satisfied, change visibility under **Settings -> General ->
Danger Zone -> Change repository visibility**, or run:

```bash
gh repo edit --visibility public --accept-visibility-change-consequences
```

### Alternative: create the remote in the GitHub website

1. Choose **New repository** under your GitHub account.
2. Name it `signalframe` or another name you own.
3. Do not initialize it with a README, `.gitignore`, or license because the
   local release already contains those files.
4. Copy the SSH URL and run:

```bash
git remote add origin git@github.com:YOUR_USERNAME/signalframe.git
git push -u origin main
```

Use the HTTPS remote instead if that is how your Git credential manager is
configured:

```bash
git remote add origin https://github.com/YOUR_USERNAME/signalframe.git
git push -u origin main
```

Use only one `git remote add origin` command.

## 5. Verify ownership and public presentation

After the push:

```bash
git remote -v
gh repo view --json nameWithOwner,visibility,url,defaultBranchRef
git ls-remote --heads origin main
```

In GitHub, verify:

- the repository owner is your account;
- the initial commit shows Karan Chandra Dey and links to your profile;
- the Contributors page contains only the authors actually present in history;
- no secrets, weights, videos, runtime outputs, or local paths are visible;
- README links render correctly; and
- the LICENSE and NOTICE match the intended project terms and preserve
  required third-party notices.

GitHub may identify vendored dependencies, package authors, model publishers,
or license owners in dependency and license views. That is correct third-party
attribution and does not make them creators of this repository.

## 6. Add professional repository settings

Recommended GitHub settings:

- description: `Evidence-first short-video analysis with a verified TRIBE v2 cortical viewer`
- topics: `video-analysis`, `creator-tools`, `react`, `fastapi`, `vjepa`,
  `neuroscience`, `threejs`
- default branch: `main`
- branch protection: require a pull request and passing checks before merge
- vulnerability alerts and secret scanning: enabled when available
- issues: enabled only if you plan to maintain public support

Do not claim Meta, MIT, Hugging Face, MLX Community, or any model publisher
endorses this project.

## 7. Normal future pushes

For each change:

```bash
git status --short
git diff
npm run build
npm test

git add PATHS_YOU_REVIEWED
git diff --cached
git commit -m "Describe the change"
git push
```

Run the backend suite for backend changes:

```bash
python -m unittest discover -s backend/tests -v
```

Keep commits under your configured identity, do not add automated co-author
trailers, and continue to preserve third-party license attribution.

## 8. Tag a release

After a clean build and test run:

```bash
git tag -a v0.1.0 -m "SignalFrame v0.1.0"
git push origin v0.1.0
gh release create v0.1.0 --generate-notes
```

Do not attach model weights, runtime caches, creator videos, or derived cortical
artifacts to a GitHub release.
