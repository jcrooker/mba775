"""Build the student upload packs in upload_packs/.

Each pack holds every file one laboratory needs, ready for a student to drag
into a Claude conversation. This exists because Claude cannot always fetch
files from a repository by direct link -- web access is a per-account setting,
and even when it is on, a raw file inside a repository is not something the
search tools can reach. Uploading always works.

Run from the repository root:

    python tools/build_upload_packs.py

Everything is copied from scripts/ and data/, so this is safe to re-run after
re-seeding. It rebuilds from scratch each time.

Add --verify to actually execute each pack in an isolated temporary directory
and confirm it runs with nothing else on the path. That check is the point of
the whole exercise: it is how you find out that a lab's file list is wrong
before a student does.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

# Everything lives under docs/ because only docs/ is published to the faculty
# web server. Putting the packs anywhere else means students can reach them on
# GitHub but not on the course site -- and the course site is the one an
# assistant can actually fetch from, since it serves plain files rather than
# GitHub's directory pages.
OUT = REPO / "docs" / "upload_packs"
ZIPS = OUT

SITE = "https://crooker.faculty.unlv.edu/mba775/fall2026"

# Zip sizes, filled in by build() and reported alongside the folder sizes.
summary_zip: dict[str, float] = {}

# The file list for each lab is the script's ACTUAL dependencies -- every
# module it imports and every data file it reads. Getting this wrong is not a
# cosmetic error: a missing helper module stops the lab with a traceback.
LABS = {
    "lab-1-chapter-01": {
        "chapter": 1,
        "topic": "inspecting a data series",
        "run": "01a_inspect_dff.py",
        "scripts": ["01a_inspect_dff.py", "_course.py"],
        "data": ["dff.csv"],
    },
    "lab-2-chapter-02": {
        "chapter": 2,
        "topic": "displaying data",
        "run": "02_displaying_data.py",
        "scripts": ["02_displaying_data.py", "_course.py", "_charts.py"],
        "data": ["state_hpi_ur_pop.csv"],
    },
    "lab-3-chapter-03": {
        "chapter": 3,
        "topic": "calculating descriptive statistics",
        "run": "03_descriptive_statistics.py",
        "scripts": ["03_descriptive_statistics.py", "_course.py", "_stats.py"],
        "data": ["annual_returns.csv", "cpi_inflation.csv",
                 "grad_student_gpa.csv", "nevada_agi_2016_sample.csv",
                 "state_hpi_ur_pop.csv", "state_population.csv",
                 "unemployment_rate.csv"],
    },
    "lab-4-chapter-04": {
        "chapter": 4,
        "topic": "probability",
        "run": "04_probability.py",
        "scripts": ["04_probability.py", "_course.py", "_stats.py", "_prob.py"],
        "data": ["nevada_economy.csv"],
    },
}

PACK_README = """LABORATORY {n} - UPLOAD PACK
Chapter {chapter}: {topic}
================================================================

WHAT THIS FOLDER IS

Everything Laboratory {n} needs, in one place. Use this when Claude cannot
reach the course repository on its own - which is normal on a free
Claude account, and is not something you did wrong.

Nothing here needs to be installed, unzipped further, or edited.


WHAT TO DO

1. Download all {count} files listed below.

2. Open a NEW conversation at claude.ai.

3. Drag all {count} files into the message box at once. Wait until each
   one shows as attached.

4. Paste the Laboratory {n} prompt from the lecture note - the one that
   begins "I have uploaded". Send it.

Claude will read COURSE_CONTEXT.md, run the script, and show you the
output.


THE FILES

{listing}

Upload ALL of them. The script will stop with an error if any are
missing - which is the script telling you the truth rather than
guessing, and is the behaviour this course wants.


IF CLAUDE REFUSES A .py FILE

Some accounts will not accept files ending in .py. If that happens, use
the copies in the folder:

    if-py-files-are-refused/

They are the same code with .txt on the end of the name. Upload those
instead of the .py files. Nothing else changes.


WHY THE SCRIPT MIGHT STILL FAIL

If Claude reports an error, that is information, not a disaster. Read
what it says. The two common ones:

  "No module named ..."    - a .py file did not get uploaded
  "Could not find ..."     - a .csv file did not get uploaded

Both mean a file is missing from the conversation. Re-upload it.

Do not accept a result Claude describes without showing you the output
that produced it. That rule is the whole point of this course.
"""

TOP_README = """MBA 775 - UPLOAD PACKS
================================================================

Each folder here holds every file one laboratory needs, ready to drag
into a Claude conversation.

{index}

Open the folder for your lab and read its README.txt.


WHY THESE EXIST

The lecture notes offer two ways to get course files to Claude. The
first is to let Claude fetch them from this repository itself. That
works only if your Claude account has web access turned on, and even
then Claude cannot always retrieve a file by direct link.

These folders are the other way, and they work on every account
including the free tier. Download the files, upload the files, paste
the prompt. No web access required.

If you are not sure whether your account has web access, the Chapter 1
lecture note shows you how to check.


REBUILDING THIS FOLDER (instructor)

    python tools/build_upload_packs.py --verify

The --verify flag runs each pack in an isolated directory and confirms
it executes with nothing else available. Run it after re-seeding data.
"""



INDEX_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>MBA 775 — Laboratory files</title>
<style>
 body {{ font-family: "Source Sans Pro", system-ui, sans-serif; max-width: 46rem;
        margin: 2rem auto; padding: 0 1rem; color: #1a1a1a; line-height: 1.55; }}
 h1 {{ color: #a03123; }}
 h2 {{ color: #a03123; margin-top: 2rem; font-size: 1.15rem; }}
 a {{ color: #a03123; }}
 .zip {{ display: inline-block; background: #a03123; color: #fff; padding: .35rem .7rem;
         border-radius: 4px; text-decoration: none; font-weight: 600; }}
 ul {{ padding-left: 1.2rem; }}
 .note {{ background: #fdf6f5; border-left: 4px solid #a03123; padding: 1rem;
          border-radius: 4px; }}
</style>
</head>
<body>
<h1>MBA 775 — Laboratory files</h1>

<p class="note">Every file each laboratory needs. Download the zip for the
whole set, or take individual files from the list. These are plain files on
an ordinary web server, so an assistant that can browse the web may be able
to fetch them directly.</p>

{sections}

<p><a href="../">Back to the course notes</a></p>
</body>
</html>
"""


def write_index() -> int:
    """A plain HTML page listing every lab file, published with the site.

    Claude's web-fetch tool follows links found on pages it has already
    loaded, but refuses to open a URL it constructed itself. GitHub compounds
    this by blocking its directory-listing pages to automated fetching. An
    ordinary page of ordinary links on an ordinary web server sidesteps both
    problems -- which is the whole reason this file exists.
    """
    sections, count = [], 0
    for name, spec in LABS.items():
        items = "\n".join(
            f'  <li><a href="{name}/{f}">{f}</a></li>'
            for f in spec["scripts"] + spec["data"] + ["COURSE_CONTEXT.md"])
        count += len(spec["scripts"]) + len(spec["data"]) + 1
        sections.append(
            f'<h2>Laboratory {spec["chapter"]} — {spec["topic"]}</h2>\n'
            f'<p><a class="zip" href="{name}.zip">Download all as one zip</a></p>\n'
            f'<ul>\n{items}\n</ul>')
    (OUT / "index.html").write_text(
        INDEX_HTML.format(sections="\n\n".join(sections)), encoding="utf-8")
    return count


def build() -> list[tuple[str, int, float]]:
    summary_zip.clear()
    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir(parents=True)
    summary = []

    for name, spec in LABS.items():
        pack = OUT / name
        fallback = pack / "if-py-files-are-refused"
        fallback.mkdir(parents=True)

        for s in spec["scripts"]:
            src = REPO / "scripts" / s
            if not src.exists():
                raise FileNotFoundError(f"{name}: missing scripts/{s}")
            shutil.copy(src, pack / s)
            shutil.copy(src, fallback / f"{s}.txt")

        for d in spec["data"]:
            src = REPO / "data" / d
            if not src.exists():
                raise FileNotFoundError(
                    f"{name}: missing data/{d} -- run the chapter's seeder first")
            shutil.copy(src, pack / d)

        shutil.copy(REPO / "COURSE_CONTEXT.md", pack / "COURSE_CONTEXT.md")

        uploads = spec["scripts"] + spec["data"] + ["COURSE_CONTEXT.md"]
        listing = "\n".join(f"    {f}" for f in uploads)
        (pack / "README.txt").write_text(
            PACK_README.format(n=spec["chapter"], chapter=spec["chapter"],
                               topic=spec["topic"], count=len(uploads),
                               listing=listing),
            encoding="utf-8")

        size = sum(f.stat().st_size for f in pack.iterdir() if f.is_file())
        summary.append((name, len(uploads), size / 1e6))

    index = "\n".join(
        f"    {name}/   Chapter {spec['chapter']} - {spec['topic']}"
        for name, spec in LABS.items())
    (OUT / "README.txt").write_text(TOP_README.format(index=index),
                                    encoding="utf-8")

    # One zip per lab. Written into docs/ rather than the repository root so
    # the cPanel deployment picks them up -- only docs/ is published.
    for name in LABS:
        archive = shutil.make_archive(str(ZIPS / name), "zip",
                                      root_dir=OUT, base_dir=name)
        summary_zip[name] = Path(archive).stat().st_size / 1e6

    return summary


def verify() -> bool:
    """Run each pack in an isolated directory, with nothing else on the path.

    This is the check that matters. A lab's file list can look right and still
    be missing a helper module that the script imports -- the failure only
    appears when the script runs somewhere the rest of the repository is not.
    """
    all_ok = True
    for name, spec in LABS.items():
        pack = OUT / name
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            for f in pack.iterdir():
                if f.is_file() and f.suffix in {".py", ".csv", ".md"}:
                    shutil.copy(f, tmp / f.name)
            # Inherit the real environment and override only the one thing
            # that matters. An earlier version stripped the environment to a
            # hand-built minimum, which worked on Linux and broke on Windows:
            # pathlib.Path.home() reads USERPROFILE there, not HOME, so
            # matplotlib could not find a config directory and any pack that
            # imports _charts.py died. The isolation that matters is the
            # working directory, not the environment.
            env = os.environ.copy()
            env["MPLBACKEND"] = "Agg"
            result = subprocess.run(
                [sys.executable, spec["run"]], cwd=tmp,
                capture_output=True, text=True, env=env)
        ok = result.returncode == 0
        all_ok &= ok
        print(f"  {name:<20} {'PASS' if ok else 'FAIL'}")
        if not ok:
            tail = (result.stderr or result.stdout).strip().splitlines()[-4:]
            for line in tail:
                print(f"      {line}")
    return all_ok


if __name__ == "__main__":
    print("Building upload packs\n")
    for name, count, mb in build():
        print(f"  {name:<20} {count:>2} files to upload   "
              f"{mb:.2f} MB folder   {summary_zip[name]:.2f} MB zip")
    n_links = write_index()
    print(f"\n  index.html         {n_links} file links, published at")
    print(f"                     {SITE}/upload_packs/")

    if "--verify" in sys.argv:
        print("\nVerifying each pack runs in isolation\n")
        if not verify():
            print("\nAt least one pack failed. Fix the file list in LABS "
                  "before publishing.")
            sys.exit(1)
        print("\nAll packs execute standalone.")

    print(f"\nFolders written to {OUT}")
    print(f"Zips written to    {ZIPS}")
    print("\nCommit both. The zips publish with the site; the folders let a")
    print("student grab a single file without downloading the whole pack.")
