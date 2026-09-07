"""What is actually present in the release, per structure class?

Determines which surgically-relevant morphometrics are computable now versus which need
new labels. Sampled rather than exhaustive -- 60 cases is enough to tell "always there"
from "never there".
"""
import collections
import glob
import random

import numpy as np
import nibabel as nib

NAMES = {26: "sacrum", 29: "S1", 30: "hip_L", 31: "hip_R", 32: "femur_L", 33: "femur_R",
         58: "iliolumbar_L", 59: "iliolumbar_R",
         60: "nerve_L4_L", 61: "nerve_L4_R", 62: "nerve_L5_L", 63: "nerve_L5_R",
         64: "nerve_S1_L", 65: "nerve_S1_R", 66: "psoas_L", 67: "psoas_R",
         68: "aorta", 69: "IVC", 70: "iliac_art_L", 71: "iliac_art_R",
         72: "iliac_vein_L", 73: "iliac_vein_R",
         74: "rib_lumbar_L", 75: "rib_lumbar_R", 76: "hardware", 77: "hardware_cage"}
for n in range(1, 13):
    NAMES[7 + n] = f"T{n}"
for n in range(1, 7):
    NAMES[19 + n] = f"L{n}"

files = sorted(glob.glob("data/v5_final/*_label.nii.gz"))
random.seed(0)
sample = random.sample(files, min(60, len(files)))
present = collections.Counter()
for f in sample:
    lab = np.asanyarray(nib.load(f).dataobj)
    for v in np.unique(lab):
        if int(v) in NAMES and (lab == v).sum() > 200:
            present[int(v)] += 1

n = len(sample)
print(f"  sampled {n} cases\n")
print(f"  {'id':>4s}  {'structure':16s} {'present in':>10s}")
for vid in sorted(NAMES):
    c = present.get(vid, 0)
    bar = "#" * int(20 * c / n)
    print(f"  {vid:4d}  {NAMES[vid]:16s} {c:4d}/{n:<4d} {bar}")
