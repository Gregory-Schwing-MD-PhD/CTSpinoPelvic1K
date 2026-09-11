"""Build morphometrics/level_references.csv from the published series.

WHY THIS IS A SCRIPT AND NOT A HAND-WRITTEN CSV. There are several hundred numbers here
across seven quantities and thirty-odd series, and a hand-typed CSV of that size is a
transcription-error generator. Written as literals per series, each row inherits its
cohort, modality, dispersion type and citation automatically, so a level cannot silently
acquire another study's error bar.

THREE THINGS THIS FILE IS CAREFUL ABOUT, because the literature is not.

DISPERSION IS NOT INTERCHANGEABLE. Panjabi reports SEM over twelve specimens, which says
how well a mean is pinned down, not how much people differ. Bonczar is a pooled
meta-analytic standard error. Tan states SEM but defines it as SD divided by n rather than
by root n. Three series print a plus-or-minus and never say what it is. Each row carries
its own label and anything not an SD must not be drawn as spread.

PRIMARY AND SECONDARY ARE MARKED. A value read from the paper's own table is primary; one
recovered from a later paper's comparison table is secondary and provisional. Several of
the classics are paywalled with no open copy, so their rows exist only as secondary.

DEFINITIONS DIFFER UNDER ONE NAME. Endplate depth measured as the maximum anteroposterior
extent of the contour runs about 4 mm above the same quantity measured as the midsagittal
chord, within a single cohort. Disc height from a recumbent radiograph at the extreme
margins runs 6 mm above the same disc measured perpendicular to the endplate on CT. Where
a series is known to use a divergent definition it is named in the notes column.
"""
from __future__ import annotations

import csv
from pathlib import Path

LEVELS = ["T12", "L1", "L2", "L3", "L4", "L5"]

# series -> (cohort, n, modality, dispersion, source_kind, citation)
S = {
    "Panjabi 1992": ("cadaveric dry bone", "12", "3-D digitiser", "SEM", "primary",
                     "Panjabi MM et al. Spine 1992;17(3):299-306"),
    "Tan 2004": ("cadaveric dry bone, Chinese Singaporean", "10", "contact digitiser",
                 "SEM (defined as SD/n)", "primary", "Tan SH et al. Eur Spine J 2004;13(2):137-146"),
    "Chen 2011": ("patients, Chinese", "83", "CT MPR 0.625 mm", "SD", "primary",
                  "Chen H et al. Eur Spine J 2011;20(11):1814-1820"),
    "Duman 2026": ("adults, Turkish", "517", "MDCT 0.625 mm", "SD", "primary",
                   "Duman L, Yamansavci Sirzai E. Tomography 2026;12(8):109"),
    "Bonczar 2024": ("meta-analysis of 18 studies", "1481", "mixed", "pooled SE", "primary",
                     "Bonczar M et al. Surg Radiol Anat 2024;47(1):22 (includes Panjabi 1992)"),
    "Gilad 1985": ("healthy working males, Israeli", "157", "lateral radiograph", "SD", "primary",
                   "Gilad I, Nissan M. J Anat 1985;143:115-120"),
    "Busscher 2010": ("fresh-frozen cadaver, male", "6", "CT 1.0 mm", "SD", "primary",
                      "Busscher I et al. Eur Spine J 2010;19(7):1104-1114"),
    "Kunkel 2011": ("cadaveric, German", "30", "lateral radiograph", "SD", "primary",
                    "Kunkel ME et al. J Anat 2011;219(3):375-387"),
    "Front Physiol 2026": ("healthy volunteers, Chinese", "100", "CT 1 mm", "SD", "primary",
                           "Front Physiol 2026;17:1820186"),
    "Yadav 2020": ("adults, Indian", "302", "MDCT", "SD", "primary",
                   "Yadav U et al. Int J Spine Surg 2020;14(2):175-181"),
    "van der Houwen 2009": ("patients, Dutch", "77", "16-MDCT", "median only", "primary",
                            "van der Houwen EB et al. Ann Biomed Eng 2009;38(1):33-40"),
    "Tang 2016": ("adults, USA", "109", "T2 MRI", "SD", "primary",
                  "Tang R et al. Eur Spine J 2016;25(12):4116-4131"),
    "Griffith 2016": ("adults, Hong Kong Chinese", "1080", "CT", "SD", "primary",
                      "Griffith JF et al. Quant Imaging Med Surg 2016;6(6):671-679"),
    "Cizmic 2023": ("symptomatic patients, Bosnian", "200", "MSCT", "SD", "primary",
                    "Cizmic M et al. Acta Inform Med 2023;31(3):200-205"),
    "Abbas 2021": ("control arm, Israeli", "180", "CT", "SD", "primary",
                   "Abbas J et al. BioMed Res Int 2021:7093745"),
    "Masharawi 2021": ("controls, Israeli", "50", "CT", "SD", "primary",
                       "Masharawi Y et al. BMC Musculoskelet Disord 2021;22:1023"),
    "Amonoo-Kuofi 1985": ("cadaveric dry bone, Nigerian", "122", "vernier", "SD", "primary",
                          "Amonoo-Kuofi HS. J Anat 1985;140(1):69-78"),
    "Amonoo-Kuofi 1982": ("adults, Nigerian", "290", "AP radiograph, uncorrected", "SD", "primary",
                          "Amonoo-Kuofi HS. J Anat 1982;135(2):225-233"),
    "Alonge 2021": ("adults, Nigerian", "120", "CT", "SD", "primary",
                    "Alonge OJ et al. Afr J Med Med Sci 2021;50:69-76"),
    "Ahmed 2019": ("symptomatic, Indian", "80", "1.5 T MRI", "SD", "primary",
                   "Ahmed MA et al. J Chalmeda Anand Rao Inst Med Sci 2019;17(1):21-27"),
    "Chaudhary 2015": ("adults, Indian", "300", "AP radiograph, uncorrected", "SD", "primary",
                       "Chaudhary S et al. Int J Adv Res 2015;3(8):33-36"),
    "Italian MRI": ("Caucasian adults, Italian", "604", "1.5 T MRI", "SD", "primary",
                    "Sagittal Normal Limits of Lumbosacral Spine. J Clin Imaging Sci"),
    "Hinck 1966": ("adult controls", "50-59 per level", "radiograph", "SD", "secondary",
                   "Hinck VC et al. 1966, via Ahn NU et al. Acta Orthop Scand 2001;72(1):67-71"),
    "Eisenstein 1977": ("Caucasoid and Zulu", "-", "dry bone", "none", "secondary",
                        "Eisenstein S 1977, via Chaudhary S et al. 2015 (whole mm)"),
    "Shin 2024": ("adults, USA multi-ethnic", "700", "CT", "SD", "primary",
                  "Shin D et al. Brain Spine 2024;5:104162"),
    "Bach 2019": ("adults, USA", "240", "CT", "unlabelled", "primary",
                  "Bach K et al. World Neurosurg 2019;124:e106-e118"),
    "Twomey 1985": ("post-mortem, Australian", "204", "mid-sagittal section", "SD", "primary",
                    "Twomey L, Taylor J. Acta Orthop Scand 1985;56(6):496-499"),
}

# measure -> series -> (sex, [T12, L1, L2, L3, L4, L5] means, [sds or None], notes)
D = {
 "h_post": [
   ("Panjabi 1992", "all", [None, 23.8, 24.3, 23.8, 24.1, 22.9], [None, .92, .95, 1.10, 1.10, .95], ""),
   ("Tan 2004", "all", [21.5, 22.4, 23.1, 22.1, 21.6, 20.0], [.2, .4, .3, .4, .3, .6], ""),
   ("Front Physiol 2026", "all", [None, 27.2, 28.1, 27.3, 26.3, 24.6], [None, 1.3, 2.6, 2.2, 2.1, 2.5], ""),
   ("Duman 2026", "M", [24.83, 25.94, 27.12, None, None, None], [2.07, 1.98, 2.15, None, None, None], ""),
   ("Duman 2026", "F", [22.19, 23.57, 24.68, None, None, None], [1.85, 1.82, 1.92, None, None, None], ""),
   ("Bonczar 2024", "all", [None, 26.57, 27.02, 26.67, 25.76, 24.11], [None, 1.06, 1.06, 1.02, 1.03, .70], "pooled SE, includes Panjabi"),
   ("Gilad 1985", "M", [None, 27.1, 27.0, 27.9, 27.1, 25.7], [None, 2.1, 2.1, 2.1, 2.3, 2.5], ""),
   ("Busscher 2010", "M", [28.5, 28.5, 29.8, 29.8, 28.0, 24.9], [.9, 2.5, 1.2, 1.0, 2.1, 3.8], ""),
   ("Kunkel 2011", "all", [23.12, None, None, None, None, None], [1.94, None, None, None, None, None], ""),
 ],
 "h_ant": [
   ("Tan 2004", "all", [18.7, 20.2, 20.8, 21.4, 21.6, 22.0], [.4, .7, .5, .5, .6, .6], ""),
   ("Front Physiol 2026", "all", [None, 25.1, 26.5, 27.1, 27.6, 27.7], [None, 1.4, 1.3, 1.4, .7, .6], ""),
   ("Duman 2026", "M", [23.61, 24.78, 26.31, None, None, None], [2.14, 2.05, 2.21, None, None, None], ""),
   ("Duman 2026", "F", [21.09, 22.56, 23.85, None, None, None], [1.72, 1.91, 1.95, None, None, None], ""),
   ("Bonczar 2024", "all", [None, 24.54, 25.75, 26.83, 26.86, 27.44], [None, .96, 1.22, 1.16, 1.08, 1.16], "pooled SE"),
   ("Gilad 1985", "M", [None, 25.4, 27.2, 27.9, 27.4, 28.3], [None, 2.2, 2.0, 2.1, 2.2, 2.1], ""),
   ("Busscher 2010", "M", [25.6, 25.5, 27.3, 28.7, 27.8, 29.5], [1.6, 2.5, 2.8, 1.9, 2.4, 1.4], ""),
   ("Kunkel 2011", "all", [20.80, None, None, None, None, None], [1.96, None, None, None, None, None], ""),
 ],
 "endplate": [   # EPWu, superior end-plate transverse width
   ("Panjabi 1992", "all", [None, 41.2, 42.6, 44.1, 46.6, 47.3], [None, 1.03, .74, .88, 1.20, 1.20], ""),
   ("Tan 2004", "all", [34.5, 36.3, 38.2, 39.9, 42.0, 41.6], [.3, .4, .4, .3, .2, .3], ""),
   ("Front Physiol 2026", "all", [None, 41.8, 43.4, 45.2, 47.7, 50.7], [None, 1.0, .7, .6, .8, 2.4], ""),
   ("Bonczar 2024", "all", [None, 41.09, 42.98, 44.82, 47.50, 48.81], [None, 1.60, 1.57, 1.62, 1.87, 1.91], "pooled SE"),
   ("Yadav 2020", "M", [36.99, 40.15, 42.55, 44.34, 46.42, 48.79], [2.89, 3.09, 4.08, 3.17, 3.98, 5.54], ""),
   ("Yadav 2020", "F", [34.62, 36.95, 39.49, 41.54, 43.59, 46.00], [2.82, 3.14, 2.97, 3.47, 3.58, 3.66], ""),
   ("Chen 2011", "M", [38.0, 38.4, 41.4, 42.4, None, None], [3.4, 4.0, 4.0, 3.8, None, None], ""),
   ("Chen 2011", "F", [32.9, 34.4, 36.0, 37.2, None, None], [2.5, 2.6, 2.8, 3.4, None, None], ""),
   ("van der Houwen 2009", "M", [None, 38.7, 38.1, 40.2, 42.4, 43.9], [None]*6, "medians"),
   ("van der Houwen 2009", "F", [None, 32.7, 33.0, 34.7, 37.7, 38.1], [None]*6, "medians"),
   ("Tang 2016", "all", [None, None, None, None, 50.5, 52.1], [None, None, None, None, 3.8, 4.0], ""),
 ],
 "canal_w": [    # SCW, interpedicular
   ("Panjabi 1992", "all", [None, 23.7, 23.8, 24.3, 25.4, 27.1], [None, .92, .71, .64, .49, .88], ""),
   ("Tan 2004", "all", [17.9, 19.4, 19.5, 19.4, 20.2, 23.4], [.3, .2, .2, .4, .5, .6], ""),
   ("Bonczar 2024", "all", [None, 22.04, 22.15, 22.62, 23.19, 26.46], [None, .91, .94, .80, .82, .89], "pooled SE"),
   ("Griffith 2016", "M", [None, 20.87, 21.19, 22.13, 24.55, 29.55], [None, 2.50, 2.62, 2.70, 3.24, 4.25], ""),
   ("Griffith 2016", "F", [None, 19.83, 20.13, 20.96, 22.95, 27.41], [None, 2.16, 2.28, 2.35, 2.90, 3.69], ""),
   ("Cizmic 2023", "all", [None, 24.78, 25.14, 25.86, 27.32, 31.94], [None, 2.29, 2.24, 2.37, 3.06, 4.00], "symptomatic cohort"),
   ("Abbas 2021", "M", [None, 25.5, 26.0, 27.0, 28.6, 34.9], [None, 1.8, 2.1, 2.2, 3.0, 3.9], ""),
   ("Abbas 2021", "F", [None, 24.5, 25.4, 26.5, 28.7, 34.8], [None, 2.4, 2.5, 2.4, 2.7, 4.1], ""),
   ("Masharawi 2021", "M", [None, 22.68, 22.59, 22.47, 22.59, 25.23], [None, 2.67, 2.35, 2.73, 2.66, 3.69], ""),
   ("Masharawi 2021", "F", [None, 20.89, 21.65, 21.97, 22.29, 23.80], [None, 1.61, 2.28, 2.49, 3.52, 3.45], ""),
   ("Amonoo-Kuofi 1982", "M", [None, 22.6, 22.7, 24.5, 26.0, 28.7], [None, 1.9, 1.7, 1.5, 1.7, 2.3], ""),
   ("Amonoo-Kuofi 1982", "F", [None, 21.3, 22.5, 23.7, 25.4, 28.4], [None, 2.0, 2.0, 1.8, 1.6, 2.0], ""),
   ("Alonge 2021", "M", [None, 23.33, 23.62, 25.08, 27.65, 33.50], [None, 1.90, 1.97, 2.16, 3.01, 4.62], ""),
   ("Alonge 2021", "F", [None, 21.75, 22.36, 23.78, 26.47, 31.42], [None, 1.88, 1.86, 2.10, 2.99, 4.00], ""),
   ("Ahmed 2019", "M", [None, 23.32, 23.41, 24.30, 25.23, 27.89], [None, 2.00, 2.16, 1.99, 2.37, 2.76], ""),
   ("Chaudhary 2015", "M", [None, 24.06, 24.73, 26.01, 27.15, 31.34], [None, 2.2, 2.2, 2.4, 3.9, 5.9], ""),
   ("Hinck 1966", "all", [None, 25.0, 25.5, 26.0, 26.9, 29.7], [None, 2.2, 2.3, 2.7, 3.0, 3.7], ""),
   ("Eisenstein 1977", "M", [None, 23, 24, 23, 24, 26], [None]*6, "Caucasoid, whole mm"),
 ],
 "canal_ap": [   # SCD
   ("Panjabi 1992", "all", [None, 19.0, 18.2, 17.5, 18.6, 19.7], [None, .67, .53, .53, .71, .49], ""),
   ("Tan 2004", "all", [12.4, 12.5, 11.7, 11.2, 11.2, 11.4], [.3, .2, .2, .3, .4, .4], "runs 31-42% below Panjabi"),
   ("Bonczar 2024", "all", [None, 16.02, 15.08, 14.69, 15.06, 15.69], [None, 1.07, 1.00, .93, 1.04, 1.37], "pooled SE"),
   ("Griffith 2016", "M", [None, 15.46, 14.58, 14.04, 15.34, 19.67], [None, 2.13, 2.36, 2.45, 2.99, 3.76], ""),
   ("Griffith 2016", "F", [None, 15.58, 14.87, 14.18, 15.03, 18.12], [None, 2.02, 1.97, 2.08, 2.75, 3.28], ""),
   ("Cizmic 2023", "all", [None, 19.06, 18.05, 16.66, 16.79, 17.81], [None, 1.66, 1.93, 1.95, 2.20, 2.71], "symptomatic cohort"),
   ("Abbas 2021", "M", [None, 18.7, 17.8, 16.9, 17.6, 18.8], [None, 1.4, 1.6, 1.7, 1.9, 2.2], ""),
   ("Abbas 2021", "F", [None, 18.7, 18.0, 17.3, 18.0, 18.8], [None, 1.4, 1.6, 1.5, 1.6, 2.2], ""),
   ("Masharawi 2021", "M", [None, 16.42, 14.81, 13.49, 13.62, 14.39], [None, 2.91, 2.44, 1.61, 1.75, 2.22], ""),
   ("Masharawi 2021", "F", [None, 16.33, 15.14, 14.50, 14.21, 14.84], [None, 1.80, 1.64, 1.50, 2.12, 3.09], ""),
   ("Amonoo-Kuofi 1985", "M", [None, 16.6, 15.8, 14.9, 15.6, 16.0], [None, 1.0, 1.0, 1.0, 2.0, 2.4], ""),
   ("Amonoo-Kuofi 1985", "F", [None, 15.8, 15.1, 14.2, 14.1, 14.6], [None, 1.2, 1.1, 1.1, 1.3, 1.2], ""),
   ("Yadav 2020", "M", [16.19, 15.68, 14.57, 13.45, 13.29, 14.73], [1.26, 1.51, 1.59, 1.75, 1.83, 3.09], ""),
   ("Yadav 2020", "F", [16.33, 15.85, 14.74, 13.66, 13.40, 14.67], [1.52, 1.43, 1.38, 1.42, 1.68, 2.72], ""),
   ("Alonge 2021", "M", [None, 16.43, 15.42, 14.92, 15.40, 16.12], [None, 1.32, 1.48, 1.68, 2.07, 2.61], ""),
   ("Ahmed 2019", "M", [None, 14.28, 13.33, 12.63, 12.88, 13.91], [None, 1.16, 1.24, 1.47, 1.44, 2.06], ""),
   ("Italian MRI", "M", [None, 16.1, 14.8, 13.8, 12.9, 13.5], [None, 2.3, 2.2, 2.4, 2.3, 2.3], ""),
   ("Italian MRI", "F", [None, 15.8, 14.6, 13.5, 12.7, 13.9], [None, 2.4, 2.1, 2.3, 2.2, 2.1], ""),
 ],
}

# disc height is indexed by interspace, not vertebral level
DISC_LEVELS = ["T12L1", "L1L2", "L2L3", "L3L4", "L4L5", "L5S1"]
DISC = [
  ("Shin 2024", "all", [None, 6.61, 7.90, 8.83, 9.31, 8.14], [None, 1.39, 1.59, 1.56, 1.75, 1.90], "middle height"),
  ("Bach 2019", "M", [5.6, 6.9, 8.1, 8.7, 9.2, 8.8], [1.1, 1.3, 1.4, 1.5, 1.6, 1.6], "mean of three, dispersion unlabelled"),
  ("Bach 2019", "F", [4.8, 5.8, 6.9, 7.6, 8.5, 8.6], [.8, .9, 1.1, 1.2, 1.6, 1.8], "mean of three, dispersion unlabelled"),
  ("Yadav 2020", "M", [6.45, 7.33, 8.66, 9.76, 10.60, 9.82], [1.27, 1.48, 1.68, 1.80, 1.76, 1.99], ""),
  ("Yadav 2020", "F", [6.35, 7.30, 8.45, 10.17, 10.23, 9.37], [1.27, 1.37, 1.41, 6.87, 1.65, 2.25], ""),
  ("Busscher 2010", "M", [9.3, 10.3, 11.5, 11.8, 12.7, 8.8], [2.1, 3.2, 2.3, 2.1, 2.6, 3.2], "central height"),
  ("Twomey 1985", "M", [None, 9.2, 9.8, 10.3, 11.3, 10.6], [None, 1.0, 1.2, 1.0, 1.3, 1.6], "area divided by AP depth, age 20-35"),
  ("Twomey 1985", "F", [None, 6.1, 8.0, 8.2, 8.5, 8.1], [None, 2.5, 1.3, 1.2, 1.3, 1.2], "area divided by AP depth, age 20-35"),
]

# ---------------------------------------------------------------- Panjabi's thoracic paper
# SAME TWELVE SPINES. The 1991 paper states "A total of 12 C2 to L5 spines were studied",
# so the thoracic and lumbar papers are two reports of one series and their levels join
# into a single curve rather than two references. Read from the tables, not from a figure.
#
# END-PLATE WIDTH IS ABSENT ON PURPOSE. Its thoracic values live in that paper's Table 3,
# on page 892, which is missing from the scan we hold -- the copy runs 891 then 893. The
# figures give end-plate width as a curve (Fig. 4A) and reading values off an axis would
# put graph estimates into a published comparison, so T11 and T12 have no end-plate row.
THORACIC_LEVELS = ["T11", "T12"]
THORACIC = {
    "canal_w":  ([19.4, 22.2], [0.95, 1.12], "Table 4, p. 893 (SCW)"),
    "canal_ap": ([16.0, 18.1], [0.46, 0.62], "Table 4, p. 893 (SCD)"),
    # PDW is the mean of the sides, as the lumbar rows are:
    #   T11  PDWr 8.8 +- 0.43, PDWl 10.7 +- 0.84
    #   T12  PDWr 8.8 +- 0.81, PDWl  8.6 +- 0.68
    "pedicle":  ([9.75, 8.70], [0.635, 0.745], "Table 5, p. 894 (mean of PDWr and PDWl)"),
    # END-PLATE WIDTH IS DIGITISED FROM THE PAPER'S OWN FIGURE, NOT READ FROM A TABLE.
    # Table 3 holds the end-plate dimensions and sits on page 892, which is absent from
    # every scan of this paper we have been able to obtain -- the two copies on hand are
    # byte-identical and both skip it (p. 891 is followed by p. 893). Interlibrary loan is
    # outstanding; when the table arrives these two values should be replaced by it.
    #
    # Until then they come from Figure 4A on p. 896, which plots EPWu, EPWl, EPDu and EPDl
    # against T1-T12 -- the authors' own figure of their own data, not a textbook redraw.
    # Benzel's Fig. 1.1 was considered and rejected: it pools four sources (Berry, both
    # Panjabi papers and White & Panjabi), plots BODY DIAMETER rather than end-plate width,
    # has no T11 tick, and the chapter states its figures may contain extrapolated data.
    #
    # Digitised by locating the open-square marker interiors against the axis ticks
    # (44.0 px per mm, residual-free over 10-50). Four checks agree:
    #   T1 24.54 -> T12 39.20 is a 59.7% increase, inside the 55% (EPWl) to 73% (EPDu)
    #     range the paper states for the four dimensions;
    #   the largest steps are T10->T11 (+4.60) and T11->T12 (+3.96) against ~1.3 for every
    #     earlier level, matching "the increase in width was greatest for the two
    #     distal-most vertebrae (T11 and T12)" on p. 890;
    #   T12 39.2 runs continuously into L1 41.2 from the 1992 lumbar paper;
    #   EPWl(T12) 42.3 exceeds EPWu(L1) 41.2, as the text requires.
    # No SEM is recoverable from the figure, so the dispersion is left empty rather than
    # invented.
    "endplate": ([35.2, 39.2], [None, None],
                 "Figure 4A, p. 896 (EPWu, digitised; Table 3 p. 892 missing from all scans)"),
}
PANJABI_1991 = ("cadaveric dry bone", "12", "3-D digitiser", "SEM", "primary",
                "Panjabi MM et al. Spine 1991;16(8):888-901 (same 12 spines as the 1992 paper)")

OUT = Path(__file__).resolve().parent / "level_references.csv"
FIELDS = ["measure", "level", "series", "sex", "mean", "sd", "dispersion",
          "cohort", "n", "modality", "source_kind", "notes", "citation"]


def main():
    rows = []
    for measure, entries in list(D.items()) + [("disc", DISC)]:
        levels = DISC_LEVELS if measure == "disc" else LEVELS
        for series, sex, means, sds, note in entries:
            cohort, n, modality, disp, kind, cite = S[series]
            for lv, m, sd in zip(levels, means, sds):
                if m is None:
                    continue
                rows.append({"measure": measure, "level": lv, "series": series, "sex": sex,
                             "mean": m, "sd": "" if sd is None else sd, "dispersion": disp,
                             "cohort": cohort, "n": n, "modality": modality,
                             "source_kind": kind, "notes": note, "citation": cite})
    cohort, n, modality, disp, kind, cite = PANJABI_1991
    for measure, (means, sems, note) in THORACIC.items():
        for lv, m, sd in zip(THORACIC_LEVELS, means, sems):
            rows.append({"measure": measure, "level": lv, "series": "Panjabi 1992",
                         "sex": "all", "mean": m, "sd": sd, "dispersion": disp,
                         "cohort": cohort, "n": n, "modality": modality,
                         "source_kind": kind, "notes": note, "citation": cite})

    with OUT.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)
    from collections import Counter
    print(f"wrote {OUT.name}: {len(rows)} rows")
    for k, v in sorted(Counter(r["measure"] for r in rows).items()):
        n_s = len({r["series"] for r in rows if r["measure"] == k})
        print(f"   {k:10s} {v:4d} rows over {n_s} series")


if __name__ == "__main__":
    main()
