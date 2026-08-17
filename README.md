# MBA 775 — Data Modeling and Analytics

Course materials for MBA 775 at the University of Nevada, Las Vegas.
Instructor: Dr. Skip Crooker.

Lecture notes: **<https://jcrooker.github.io/mba775/>**

---

## Start here (students)

You do not need to install Python. You will run everything through Claude's
code execution, which means three steps:

1. **Upload `COURSE_CONTEXT.md`** to your Claude conversation, once per
   session. It tells Claude how this course expects analytical work to be
   done.
2. **Download the script and the data file** a lecture note asks for, from
   `scripts/` and `data/` below, and upload both.
3. **Paste the prompt** printed in the lecture note.

That is the whole workflow. Every script reads a local CSV — nothing
downloads anything, so nothing depends on a network or an API key.

### Downloading a single file

Open the file on GitHub and click **Download raw file**. Or download the whole
repository at once: green **Code** button → **Download ZIP**.

---

## What is here

| Folder | Contents |
|---|---|
| `scripts/` | Student-facing Python scripts, one or more per lecture note |
| `data/` | CSV data files the scripts read |
| `notes/` | Quarto source for the lecture notes |
| `docs/` | Rendered lecture notes (published as the course website) |
| `tools/` | Instructor tooling — you do not need these |

Scripts are named by the lecture note they belong to: `01a`, `01b`, `01c` are
the three scripts for Chapter 1.

### Chapter 1 — Introduction to Data Analytics and Modeling

| Script | Data file | What it covers |
|---|---|---|
| `01a_inspect_dff.py` | `dff.csv` | Verifying a data set before analyzing it: types, coverage, missing values, provenance |
| `01b_monthly_transformations.py` | `dff.csv` | Selecting an observation vs. calculating a statistic; three monthly measures and how far apart they get |
| `01c_state_cross_section.py` | `state_unemployment.csv` | What makes a cross section a cross section; checking that observations are contemporaneous |

`scripts/_course.py` holds small shared helpers for loading data. Upload it
alongside any script that imports it.

---

## The one rule

**Every number you report must come from code that was actually executed.**

An assistant will write a confident paragraph whether or not the underlying
computation supports it. Confidence in the prose is not evidence about the
data. If you cannot point to the cell that produced a number, the number does
not go in your submission.

---

## Instructor notes

Rendering the lecture notes requires Quarto 1.7+ and a Python environment with
`pandas`, `matplotlib`, `plotly`, `nbformat`, and `jupyter`.

```
quarto render                       # all notes
quarto publish gh-pages             # publish to the course website
```

Refreshing course data is deliberate, not automatic:

```
Rscript tools/seed_fred_cache.R     # download raw FRED CSVs into data/raw/
python tools/make_student_data.py   # build data/dff.csv and data/state_unemployment.csv
```

`tools/seed_fred_cache.R` uses R because FRED's CDN refuses Python's HTTPS
clients on some networks. `tools/fred_tools.py` remains for interactive work
and tries several transports before giving up.

---

## License

Lecture notes and written materials: [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/).
Code (`scripts/`, `tools/`): MIT. See [LICENSE.md](LICENSE.md).

Data are from the Federal Reserve Economic Data (FRED) service of the Federal
Reserve Bank of St. Louis and are redistributed here for coursework. See
[data/README.md](data/README.md) for provenance.
