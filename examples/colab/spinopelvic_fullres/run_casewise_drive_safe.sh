#!/usr/bin/env bash
set -euo pipefail

SESSION="${SESSION:-spino-fresh-fullres}"
GPU="${GPU:-T4}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
COLAB="${COLAB:-colab}"
ADC="${ADC:-${ROOT}/.gcloud-colab/application_default_credentials.json}"
FOLDER_ID="${FOLDER_ID:-YOUR_GOOGLE_DRIVE_FOLDER_ID}"
DRIVE_URL="${DRIVE_URL:-https://drive.google.com/drive/folders/${FOLDER_ID}?usp=sharing}"
RESULT_DIR="${RESULT_DIR:-${ROOT}/ai_spine_feasibility/colab/spinopelvic_fullres_fresh/downloaded_results_casewise}"
EXEC_TIMEOUT="${EXEC_TIMEOUT:-21600}"
UPLOAD_TO_DRIVE="${UPLOAD_TO_DRIVE:-0}"

export GOOGLE_APPLICATION_CREDENTIALS="${ADC}"
mkdir -p "${RESULT_DIR}"

"${COLAB}" --auth=adc sessions >/dev/null || true
if ! "${COLAB}" --auth=adc status -s "${SESSION}" >/dev/null 2>&1; then
  if [[ -n "${GPU}" ]]; then
    "${COLAB}" --auth=adc new -s "${SESSION}" --gpu "${GPU}"
  else
    "${COLAB}" --auth=adc new -s "${SESSION}"
  fi
fi

"${COLAB}" --auth=adc upload -s "${SESSION}" \
  "${ROOT}/ai_spine_feasibility/colab/spinopelvic_fullres_fresh/spinopelvic_fullres_nrrd_colab.py" \
  /content/spinopelvic_fullres_fresh/spinopelvic_fullres_nrrd_colab.py

for case_id in CASE_000 CASE_001 CASE_002 CASE_003 CASE_004 CASE_005 CASE_006; do
  echo "=== ${case_id} ==="
  remote_zip="/content/spinopelvic_fullres_fresh/09_export/${case_id}_Spinopelvic_RESULT.zip"
  local_zip="${RESULT_DIR}/${case_id}_Spinopelvic_RESULT.zip"
  if [[ -s "${local_zip}" ]] && unzip -t "${local_zip}" >/dev/null 2>&1; then
    echo "Valid local ZIP exists, skipping ${case_id}: ${local_zip}"
    continue
  fi

  python3 - "${HF_TOKEN:-}" "${DRIVE_URL}" "${case_id}" <<'PY' |
import json
import sys

token, drive_url, case_id = sys.argv[1:4]
runner = "/content/spinopelvic_fullres_fresh/spinopelvic_fullres_nrrd_colab.py"
argv = [
    runner,
    "--drive-folder-url",
    drive_url,
    "--only-case",
    case_id,
    "--force",
]
print("import os, runpy, sys")
print("os.environ['HF_TOKEN']=" + json.dumps(token))
print("sys.argv=" + json.dumps(argv))
print("_ = runpy.run_path(sys.argv[0], run_name='__main__')")
PY
    "${COLAB}" --auth=adc exec --timeout "${EXEC_TIMEOUT}" -s "${SESSION}" || true

  if "${COLAB}" --auth=adc download -s "${SESSION}" "${remote_zip}" "${local_zip}"; then
    if [[ "${UPLOAD_TO_DRIVE}" == "1" ]]; then
      python3 "${ROOT}/ai_spine_feasibility/colab/spinopelvic_fullres_fresh/upload_to_drive.py" \
        "${local_zip}" \
        --folder-id "${FOLDER_ID}" \
        --credentials "${ADC}" \
        --manifest "${RESULT_DIR}/drive_upload_manifest.json"
    fi
  else
    echo "No ZIP for ${case_id}; see remote logs if session survives."
  fi
done

echo "Casewise run complete. Local copies: ${RESULT_DIR}"
