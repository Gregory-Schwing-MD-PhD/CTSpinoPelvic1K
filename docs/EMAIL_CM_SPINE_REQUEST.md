# Request for the CM Spine Dataset

The Google Drive link in the authors' GitHub README returns HTTP 404 across every
download form of the URL, which means the file was deleted rather than restricted.
There is no mirror on Zenodo, figshare, OSF, Kaggle, Hugging Face, IEEE DataPort or
PhysioNet, no alternative link in any of the repository's 18 commits, and no Wayback
capture, because the Internet Archive blocks Google Drive. Emailing the authors is the
only route.

**To:** ekcheng@iim.ac.cn, songbo@iim.ac.cn
**Cc:** yzeng@iim.ac.cn
**Subject:** CM Spine Dataset access: the Drive link in the Neighbor repository is returning 404

Dear Dr. Cheng and Dr. Song,

I am a surgery resident at the Detroit Medical Center and Wayne State University,
writing about the CM Spine Dataset described in your 2024 Electronics paper,
"Exploring Neighbor Spatial Relationships for Enhanced Lumbar Vertebrae Detection in
X-ray Images" (13(11), 2137).

The Data Availability Statement points to github.com/zengyuyuyu/Neighbor, and the
README there links the dataset on Google Drive. That link now returns a 404 rather
than a permission prompt, so the file appears to have been removed rather than
restricted. I could not find a mirror anywhere else.

Would you be willing to re-upload it, or send it by any route convenient to you? A
Baidu Pan link, WeTransfer, or an institutional server would all work on my end. What
I am after is the published release: the 208 intraoperative G-arm AP images with the
L1 through L5 box annotations and the 148/60 train and test split.

My interest is specific. I work on lumbosacral transitional anatomy, where a vertebra
cannot be numbered reliably by counting because the count itself is ambiguous. Our
group has just released CTSpinoPelvic1K, an 802-record CT dataset that gives explicit
classes to a sixth lumbar vertebra and to a rib borne by a lumbar vertebra, so that a
transitional level can be recorded as what it is instead of forced into a scheme that
has no name for it. Your dataset is, as far as I can tell, the only public
intraoperative spine radiograph collection with per-level annotation, and you note
that seven of the images show sacralization. Intraoperative level identification is
exactly where a numbering error becomes a wrong-level operation, so those cases are
directly relevant.

Two smaller questions if you have a moment. What format are the annotation files in,
and do lateral views exist for any of these patients? The paper reports the AP view
only.

I would cite the Electronics paper in any work that uses the data, and I am happy to
share back anything we derive from it.

Thank you for considering this, and for releasing the work under CC BY in the first
place.

With thanks,

Gregory Schwing, MD, PhD
Department of Surgery
Detroit Medical Center and Wayne State University
gregory.schwing@med.wayne.edu

---

## If they do not reply

Secondary contacts at the same institute: Zhiyong Sun (sunzy@iim.ac.cn), Kun Wang
(kunwang@iim.ac.cn). Clinical collaborator at Anhui Medical University: Changqing
Wang (wangchangqing@ahmu.edu.cn).

## What the dataset is, for planning purposes

208 intraoperative lumbar AP radiographs at 1024x1024, from a mobile dual-mode G-arm
(Geelin500-A), in lumbar disc herniation and vertebral compression fracture surgery.
148 train and 60 test. 537 annotated vertebrae, which is well short of the 1,040 the
image count would allow, because the intraoperative field of view usually shows only
L3 through L5. These are bounding boxes, not landmarks or segmentations, so it serves
detection benchmarking rather than corner regression.
