MBA 775 - UPLOAD PACKS
================================================================

Each folder here holds every file one laboratory needs, ready to drag
into a Claude conversation.

    lab-1-chapter-01/   Chapter 1 - inspecting a data series
    lab-2-chapter-02/   Chapter 2 - displaying data
    lab-3-chapter-03/   Chapter 3 - calculating descriptive statistics
    lab-4-chapter-04/   Chapter 4 - probability

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
