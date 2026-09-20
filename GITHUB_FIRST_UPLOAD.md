# Your first GitHub upload

A **repository** is your project's folder on GitHub. A **commit** is a saved
version with a short explanation. **Push** means uploading local commits;
**clone** means downloading a repository together with its history.

## Easiest first upload: your browser

1. Sign in to https://github.com and click **+ → New repository**.
2. Name it `kairos`. Suggested description: `Desktop transmon pulse optimization and fidelity comparison with NSGA-III, GRAPE, BB1 and CORPSE.`
3. Choose **Public** if you want everyone to see it, or **Private** for restricted access.
4. Leave automatic README, .gitignore and license initialization off: this folder already has a README and .gitignore. Click **Create repository**.
5. In the empty repository, click **uploading an existing file**. In an existing repository, use **Add file → Upload files**.
6. Open this `kairos` folder in your file manager. Show hidden files (Ctrl+H on many Linux file managers). Drag its **contents**, including `.gitignore`, into the upload area. Do not upload the ZIP or enclosing folder: `README.md` and the Python files should be at repository root.
7. Enter `Initial version of Kairos` as the commit message and click **Commit changes**.
8. Check that the README appears on the repository home page. You can share that page's URL.

Official guide: https://docs.github.com/en/repositories/working-with-files/managing-files/adding-a-file-to-a-repository

## Updating it later

For a few changes, upload the changed files at the same paths and commit with
a short message, for example `Improve composite pulse comparison`.
This prepared folder is a snapshot: later changes in your original working
folder are not copied here automatically.

## Optional: use Git from a terminal

Instead of the browser upload, create an empty GitHub repository as above.
From inside this prepared folder, run the following after replacing YOUR_USERNAME:

```bash
git init -b main
git add .
git commit -m "Initial version of Kairos"
git remote add origin https://github.com/YOUR_USERNAME/kairos.git
git push -u origin main
```

If Git asks for your author identity, set your chosen name and email locally
with `git config user.name "Your Name"` and
`git config user.email "Your email"`, then retry the commit.
Use GitHub's supported browser/credential-manager authentication when pushing;
your account password is not a Git HTTPS password.
After a browser upload, use `git clone` to get its existing history instead of
initializing another independent repository for the same remote.

Official command-line guide: https://docs.github.com/en/migrations/importing-source-code/using-the-command-line-to-import-source-code/adding-locally-hosted-code-to-github

## What is included

Code, tests, default configuration, dependencies, technical documentation and
`.gitignore`. Generated results, virtual environments, caches, old output logs,
and local agent settings are not included. Source code remains unchanged.
No license was chosen for you; choose one separately if you want to grant
others reuse permissions.
