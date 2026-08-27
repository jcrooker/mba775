LABORATORY 1 - UPLOAD PACK
Chapter 1: inspecting a data series
================================================================

WHAT THIS FOLDER IS

Everything Laboratory 1 needs, in one place. Use this when Claude cannot
reach the course repository on its own - which is normal on a free
Claude account, and is not something you did wrong.

Nothing here needs to be installed, unzipped further, or edited.


WHAT TO DO

1. Download all 7 files listed below.

2. Open a NEW conversation at claude.ai.

3. Drag all 7 files into the message box at once. Wait until each
   one shows as attached.

4. Paste the prompt from the lecture note for whichever exercise you
   are doing - the one that begins "I have uploaded". Send it.

Claude will read COURSE_CONTEXT.md, run the script, and show you the
output.


SCRIPTS YOU CAN RUN FROM THIS PACK

    01a_inspect_dff.py
        verify a data set before analysing it: types, coverage, missing values, provenance

    01b_monthly_transformations.py
        selecting an observation versus calculating a statistic

    01c_state_cross_section.py
        a cross section of the 50 states, and whether the dates actually line up

Upload the whole pack once; you can then run any of these in the same
conversation without uploading anything again.


THE FILES

    01a_inspect_dff.py
    01b_monthly_transformations.py
    01c_state_cross_section.py
    _course.py
    dff.csv
    state_unemployment.csv
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
