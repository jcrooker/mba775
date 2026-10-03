MBA 775 - UPLOAD PACKS
================================================================

Each folder here holds every file one laboratory needs, ready to drag
into a Claude conversation.

    lab-1-chapter-01/   Chapter 1 - inspecting a data series
    lab-2-chapter-02/   Chapter 2 - displaying data
    lab-3-chapter-03/   Chapter 3 - calculating descriptive statistics
    lab-4-chapter-04/   Chapter 4 - probability
    lab-5-chapter-05/   Chapter 5 - discrete probability distributions
    lab-6-chapter-06/   Chapter 6 - continuous probability distributions
    lab-7-chapter-07/   Chapter 7 - sampling and sampling distributions
    lab-8-chapters-08-09/   Chapter 8-9 - confidence intervals and hypothesis testing
    lab-10-chapter-10/   Chapter 10 - comparing two populations, with a slice of ANOVA
    lab-16a-forecasting-1/   Chapter 16a - forecasting 1 - naive, moving averages, MAD
    lab-16b-forecasting-2/   Chapter 16b - forecasting 2 - exponential smoothing
    lab-16-data-tools/   Chapter 16 - getting the forecasting-challenge data
    lab-16c-forecasting-3/   Chapter 16c - forecasting 3 - regression and seasonality

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
