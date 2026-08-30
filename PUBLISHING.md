# Publishing MBA 775

The routine for getting a change from this repository onto
<https://crooker.faculty.unlv.edu/mba775/fall2026/>.

There are **three separate hops** and none of them happens automatically:

    your PC  ->  GitHub  ->  the server's clone  ->  the live website

Skipping a hop is the usual reason a change "did not publish". So is browser
cache — see step 9.

---

## Every fresh PowerShell starts here

A new PowerShell has no `python` on PATH, because the interpreter lives in a
virtual environment outside this repository. Activate it first, or every
`python` command fails with *"The term 'python' is not recognized"*.

```powershell
cd C:\Users\crookj3\repos\mba775
& "C:\Users\crookj3\AppData\Local\venvs\mba775\Scripts\Activate.ps1"
```

The prompt should now start with `(mba775)`. Check it took:

```powershell
python --version
```

If activation is refused with an execution-policy error, allow it for this
window only and try again:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

**Optional, one time:** put this in your PowerShell profile
(`notepad $PROFILE`) so a new shell only needs you to type `mba775`.

```powershell
function mba775 { & "C:\Users\crookj3\AppData\Local\venvs\mba775\Scripts\Activate.ps1" }
```

---

## Hop 1 — your PC to GitHub

### 1. Re-seed the data (only when you want fresh numbers)

Skip this unless you actually want to re-download from FRED. It takes a few
minutes and the committed CSVs are already current.

```powershell
Rscript tools\seed_chapter02_data.R
```

### 2. Rebuild the Excel demo workbooks (only if the data changed)

The workbooks embed a copy of `state_hpi_ur_pop.csv` on their Data sheet, so
they go stale the moment that file changes.

```powershell
python tools\build_chapter02_workbooks.py
```

### 3. Render the notes

One note, which is what you want most of the time:

```powershell
quarto render notes\lecture-notes-1010-chapter-02-displaying-descriptive-statistics.qmd
```

Or the whole site. Unchanged notes reuse their frozen output, so this is not
as slow as it looks:

```powershell
quarto render
```

If `quarto` resolves to RStudio's bundled copy (1.6.x, too old for
`_brand.yml`), call the standalone one by full path:

```powershell
& "C:\Users\crookj3\AppData\Local\Programs\Quarto\bin\quarto.exe" render
```

### 4. Rebuild the student upload packs

Run this **after** the render. It writes `docs/upload_packs/`, which is where
the download links in the notes point.

```powershell
python tools\build_upload_packs.py --verify
```

`--verify` runs each lab's script in an isolated temporary directory to prove
it works with nothing else on the path. If it fails on a lab you did not
touch, drop the flag to publish and fix that lab separately.

### 5. Look at what you are about to commit

```powershell
git status
```

### 6. Commit and push

```powershell
git add -A
git commit -m "Describe the change here"
git push
```

---

## Hop 2 — GitHub to the server's clone

Not a command. In cPanel:

7. **Git™ Version Control** -> **Manage** on `mba775` -> the **Pull or Deploy**
   tab -> **Update from Remote**.

   Confirm the *HEAD Commit* panel now shows the SHA you just pushed. If it
   shows the old one, the pull did not take and step 8 will deploy stale files.

---

## Hop 3 — the server's clone to the live site

8. **Deploy HEAD Commit**, on that same tab.

   `.cpanel.yml` copies `docs/.` to `public_html/mba775/fall2026`. Only
   `docs/` is published — sources, data and tools stay in the repository.

---

## Finally

9. **Ctrl+F5** in the browser. A hard refresh, not a normal one.

   A cached page has twice looked exactly like a failed deploy. Check this
   before concluding anything went wrong.

---

## Quick reference

The whole thing, assuming the data has not changed:

```powershell
cd C:\Users\crookj3\repos\mba775
& "C:\Users\crookj3\AppData\Local\venvs\mba775\Scripts\Activate.ps1"
quarto render notes\lecture-notes-1010-chapter-02-displaying-descriptive-statistics.qmd
python tools\build_upload_packs.py --verify
git add -A
git commit -m "Describe the change here"
git push
```

Then cPanel: **Update from Remote**, **Deploy HEAD Commit**, then Ctrl+F5.

---

## When something looks wrong

| Symptom | Cause |
|---|---|
| `python is not recognized` | The venv is not activated. Go back to the top. |
| The site did not change | A hop was skipped, or the browser cached it. Check the HEAD Commit SHA in cPanel, then Ctrl+F5. |
| The page renders but is unstyled | A `*_files/` folder did not get committed. `.gitignore` deliberately has no blanket `*_files/` rule — do not add one. |
| Quarto complains about `_brand.yml` | RStudio's bundled Quarto 1.6.x is being used. Call the standalone one by full path. |
| A note fails on a FRED download | `fred_tools` caches to `data/raw/`. Re-seed with `Rscript`, which FRED serves; Python's client is refused on this machine. |
| The workbooks disagree with the CSV | You changed the data and did not re-run `build_chapter02_workbooks.py`. |
