import numpy as np, nibabel as nib, glob, random, collections
fs = sorted(glob.glob("data/v5_final/*_label.nii.gz"))
random.seed(0); fs = random.sample(fs, 60)
n255 = 0; ids = collections.Counter()
for f in fs:
    u = np.unique(np.asanyarray(nib.load(f).dataobj))
    if 255 in u: n255 += 1
    for x in u: ids[int(x)] += 1
print("sampled", len(fs), "cases;", n255, "contain ignore(255)")
print("ids seen:", sorted(ids))
