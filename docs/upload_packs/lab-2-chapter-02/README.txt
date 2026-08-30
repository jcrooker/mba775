LABORATORY 2 - UPLOAD PACK
Chapter 2: displaying data
================================================================

WHAT THIS FOLDER IS

Everything Laboratory 2 needs, in one place. Use this when Claude cannot
reach the course repository on its own - which is normal on a free
Claude account, and is not something you did wrong.

Nothing here needs to be installed, unzipped further, or edited.


WHAT TO DO

1. Download all 5 files listed below.

2. Open a NEW conversation at claude.ai.

3. Drag all 5 files into the message box at once. Wait until each
   one shows as attached.

4. Paste the prompt from the lecture note for whichever exercise you
   are doing - the one that begins "I have uploaded". Send it.

Claude will read COURSE_CONTEXT.md, run the script, and show you the
output.


SCRIPTS YOU CAN RUN FROM THIS PACK

    02_displaying_data.py
        frequency tables, histograms, ogives, Pareto and pie charts, contingency tables, scatter plots

Upload the whole pack once; you can then run any of these in the same
conversation without uploading anything again.


THE FILES

    02_displaying_data.py
    _course.py
    _charts.py
    state_hpi_ur_pop.csv
    COURSE_CONTEXT.md

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


EXCEL WORKBOOKS - DOWNLOAD THESE, DO NOT UPLOAD THEM

    ch02-frequency-BROKEN.xlsx
    ch02-frequency-FIXED.xlsx
    ch02-frequency-STARTER.xlsx

These are not part of the upload above. Open them in Excel (or Google
Sheets, or LibreOffice - all three evaluate the formulas).

    BROKEN   what a one-line prompt returns. Open the Checks sheet
             first. Three rules of Chapter 2 are broken and none of
             them is visible in the chart.

    FIXED    what a specified prompt returns. Open it, change cell
             C10 on the Distribution sheet from 25 to 22, and read
             the Checks sheet again.

    STARTER  the data, the class boundaries, and the six checks. The
             table is yours to build. You are finished when all six
             checks read PASS.

The lecture note walks through all three.
