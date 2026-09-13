#!/usr/bin/env bash
set -euo pipefail

SESSION="${SESSION:-spino-fresh-fullres}"
GPU="${GPU:-T4}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
COLAB="${COLAB:-colab}"
ADC="${ADC:-${ROOT}/.gcloud-colab/application_default_credentials.json}"
REMOTE_ROOT="/content/spinopelvic_fullres_fresh"
REMOTE_INPUT="${REMOTE_ROOT}/00_uploaded_inputs"
REMOTE_CHUNKS="${REMOTE_ROOT}/upload_chunks"
LOCAL_INPUT="${LOCAL_INPUT:-${ROOT}/Testfiles-fresh}"
RESULT_DIR="${RESULT_DIR:-${ROOT}/ai_spine_feasibility/colab/spinopelvic_fullres_fresh/downloaded_results}"
CHUNK_MB="${CHUNK_MB:-40}"
EXEC_TIMEOUT="${EXEC_TIMEOUT:-21600}"

mkdir -p "${RESULT_DIR}"

echo "Checking Colab CLI auth/session..."
if [[ -f "${ADC}" ]]; then
  export GOOGLE_APPLICATION_CREDENTIALS="${ADC}"
  COLAB_AUTH=(--auth=adc)
else
  COLAB_AUTH=()
fi
"${COLAB}" "${COLAB_AUTH[@]}" sessions >/dev/null

echo "Creating/reusing Colab session: ${SESSION} (${GPU})"
if ! "${COLAB}" "${COLAB_AUTH[@]}" status -s "${SESSION}" >/dev/null 2>&1; then
  "${COLAB}" "${COLAB_AUTH[@]}" new -s "${SESSION}" --gpu "${GPU}"
fi

echo "Preparing remote folders..."
printf 'from pathlib import Path\nPath("%s").mkdir(parents=True, exist_ok=True)\nPath("%s").mkdir(parents=True, exist_ok=True)\n' "${REMOTE_INPUT}" "${REMOTE_CHUNKS}" | "${COLAB}" "${COLAB_AUTH[@]}" exec --timeout "${EXEC_TIMEOUT}" -s "${SESSION}"

echo "Uploading runner..."
"${COLAB}" "${COLAB_AUTH[@]}" upload -s "${SESSION}" "${ROOT}/ai_spine_feasibility/colab/spinopelvic_fullres_fresh/spinopelvic_fullres_nrrd_colab.py" "${REMOTE_ROOT}/spinopelvic_fullres_nrrd_colab.py"

echo "Uploading fresh NRRDs from: ${LOCAL_INPUT}"
find "${LOCAL_INPUT}" -maxdepth 1 -type f \( -iname '*.nrrd' -o -iname '*.nii' -o -iname '*.nii.gz' \) -print0 |
while IFS= read -r -d '' f; do
  base="$(basename "$f")"
  safe="$(printf '%s' "${base}" | tr -cs 'A-Za-z0-9._-' '_' | sed 's/^_//; s/_$//')"
  size="$(stat -c%s "$f")"
  if (( size < CHUNK_MB * 1024 * 1024 )); then
    echo "  -> ${base} as ${safe}"
    "${COLAB}" "${COLAB_AUTH[@]}" upload -s "${SESSION}" "$f" "${REMOTE_INPUT}/${safe}"
  else
    chunk_dir="/tmp/spinopelvic_chunks_${safe}"
    mkdir -p "${chunk_dir}"
    split -b "${CHUNK_MB}M" -d -a 4 "$f" "${chunk_dir}/${safe}.part."
    remote_case_chunks="${REMOTE_CHUNKS}/${safe}"
    printf 'from pathlib import Path\nPath("%s").mkdir(parents=True, exist_ok=True)\n' "${remote_case_chunks}" | "${COLAB}" "${COLAB_AUTH[@]}" exec --timeout "${EXEC_TIMEOUT}" -s "${SESSION}"
    echo "  -> ${base} as ${safe} in chunks"
    for part in "${chunk_dir}/${safe}".part.*; do
      "${COLAB}" "${COLAB_AUTH[@]}" upload -s "${SESSION}" "$part" "${remote_case_chunks}/$(basename "$part")"
    done
    printf 'from pathlib import Path\nout=Path("%s")\nchunk_dir=Path("%s")\nwith out.open("wb") as w:\n    for p in sorted(chunk_dir.glob("*.part.*")):\n        w.write(p.read_bytes())\nprint("reassembled", out, out.stat().st_size)\n' "${REMOTE_INPUT}/${safe}" "${remote_case_chunks}" | "${COLAB}" "${COLAB_AUTH[@]}" exec --timeout "${EXEC_TIMEOUT}" -s "${SESSION}"
  fi
done

echo "Running full-resolution spinopelvic inference. This can take a long time."
if [[ -n "${HF_TOKEN:-}" ]]; then
  printf 'import os, runpy, sys\nos.environ["HF_TOKEN"]=%r\nsys.argv=["/content/spinopelvic_fullres_fresh/spinopelvic_fullres_nrrd_colab.py"]\nrunpy.run_path(sys.argv[0], run_name="__main__")\n' "${HF_TOKEN}" | "${COLAB}" "${COLAB_AUTH[@]}" exec --timeout "${EXEC_TIMEOUT}" -s "${SESSION}"
else
  printf 'import runpy, sys\nsys.argv=["/content/spinopelvic_fullres_fresh/spinopelvic_fullres_nrrd_colab.py"]\nrunpy.run_path(sys.argv[0], run_name="__main__")\n' | "${COLAB}" "${COLAB_AUTH[@]}" exec --timeout "${EXEC_TIMEOUT}" -s "${SESSION}"
fi

echo "Downloading result ZIP..."
"${COLAB}" "${COLAB_AUTH[@]}" download -s "${SESSION}" "/content/spinopelvic_fullres_fresh/09_export/Spinopelvic_Fullres_Fresh_RESULTS.zip" "${RESULT_DIR}/Spinopelvic_Fullres_Fresh_RESULTS.zip"
"${COLAB}" "${COLAB_AUTH[@]}" download -s "${SESSION}" "/content/spinopelvic_fullres_fresh/09_export/ZIP_SHA256.txt" "${RESULT_DIR}/ZIP_SHA256.txt" || true

echo "Done. Results are in: ${RESULT_DIR}"
echo "Stop the VM when finished reviewing:"
echo "  ${COLAB} stop -s ${SESSION}"
