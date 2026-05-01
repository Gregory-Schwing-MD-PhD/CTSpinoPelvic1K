#!/usr/bin/env bash
#SBATCH --job-name=scan_l6
#SBATCH -q primary
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=logs/scan_l6_%j.out
#SBATCH --error=logs/scan_l6_%j.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
source configs/default.env

singularity exec \
    --bind "${PROJECT_ROOT:-$(pwd)}:/workspace,${DATA_DIR}:/data" \
    --pwd /workspace \
    "${SIF_PATH}" \
    python3 << 'PY'
import json
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import nibabel as nib
import numpy as np

base = Path('/data')  # inside container
placed = json.load(open(base / 'placed/placed_manifest_orientation_fixed.json'))

L6_LABEL = 25

def scan_one(args):
    tok, placed_path = args
    try:
        arr = np.asarray(nib.load(placed_path).dataobj).astype(np.int16)
        return tok, int((arr == L6_LABEL).sum())
    except Exception as e:
        return tok, -1  # failed

work = []
for c in placed['cases']:
    tok = str(c.get('patient_token'))
    sp = c.get('spine') or {}
    p = sp.get('placed')
    if not p:
        continue
    if p.startswith('/data/'):
        p = p[len('/data/'):]
    p = str(base / p)
    if Path(p).exists():
        work.append((tok, p))

print(f'Scanning {len(work)} placed spine masks (16 workers)...', flush=True)
results = {}
with ProcessPoolExecutor(max_workers=16) as ex:
    futs = [ex.submit(scan_one, w) for w in work]
    for i, f in enumerate(as_completed(futs)):
        tok, n_l6 = f.result()
        results[tok] = n_l6
        if (i+1) % 100 == 0:
            print(f'  {i+1}/{len(work)}', flush=True)

l6_tokens = sorted([t for t, n in results.items() if n > 100],
                   key=lambda x: int(x) if x.isdigit() else 99999)
no_l6 = sum(1 for n in results.values() if 0 <= n <= 100)
failed = sum(1 for n in results.values() if n < 0)

print(f'\n=== RESULTS ===')
print(f'scanned: {len(results)}')
print(f'L6 positive (>100 voxels): {len(l6_tokens)}')
print(f'no L6: {no_l6}')
print(f'failed: {failed}')
print(f'\ntokens: {l6_tokens}')

# Distribution of L6 voxel counts for borderline cases
borderline = sorted([(n, t) for t, n in results.items() if 0 < n <= 200])
if borderline:
    print(f'\nborderline (1-200 L6 voxels): {len(borderline)} cases')
    for n, t in borderline[:20]:
        print(f'  token={t}  n_L6={n}')
PY
