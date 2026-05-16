"""
test_manifest.py — schema / type / JSON-safety tests for the manifest
writers in scripts/export_hf.py.

Scope (deliberately narrow):
  * presence  — every coerced record has EXACTLY the canonical key set
  * types     — each field holds its declared type (or None iff nullable)
  * JSON      — the coerced set round-trips through json.dumps/json.loads
                with one identical key set (the real HF-viewer CastError
                condition, asserted directly)
  * parity    — write_manifest() and write_splits() emit schema-identical
                records

Explicitly NOT tested: DICOM/NIfTI conversion, position propagation, or
the lstv_agreement value-computation logic. These tests never touch disk
for input (records are hand-built dicts) and never run the export.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

# scripts/export_hf.py is a standalone script, not an installed package.
_SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

import export_hf  # noqa: E402
from export_hf import (  # noqa: E402
    _MANIFEST_FIELDS,
    _MANIFEST_SCHEMA,
    _coerce_manifest_record,
    write_manifest,
    write_splits,
)

CANONICAL_KEYS = frozenset(_MANIFEST_FIELDS)
_NULLABLE = {name for name, _, nullable in _MANIFEST_SCHEMA if nullable}
_NON_NULLABLE = {name for name, _, nullable in _MANIFEST_SCHEMA if not nullable}
_PYTYPE = {name: t for name, t, _ in _MANIFEST_SCHEMA}


# --------------------------------------------------------------------------- #
# Synthetic fixture — hand-built dicts, never loaded from disk.
#
# Every record carries ok=True, token and ct_file so the same fixture can be
# fed to write_splits()/write_manifest() (both filter on ok and read raw
# token/ct_file before coercion). The "missing keys" record omits *several*
# schema fields but retains the three structural keys those writers require.
# --------------------------------------------------------------------------- #

def _base(token: str, config: str, ct_file: str) -> dict:
    """A fully-populated, type-correct record skeleton."""
    return dict(
        ok=True,
        error=None,
        token=token,
        position="supine",          # position is a nullable str column
        # DICOM header demographics (schema >=2.7). age is the first
        # nullable INT column; patient_weight/patient_size/slice_thickness/
        # kvp the first nullable FLOAT columns. sex is a nullable enum str.
        age=64,
        sex="male",
        patient_weight=70.5,
        patient_size=1.75,
        convolution_kernel="B30f",
        manufacturer="SIEMENS",
        manufacturer_model="Sensation 64",
        slice_thickness=1.0,
        kvp=120.0,
        config=config,
        match_type=config,
        prov_spine="manual",        # nullable enum: manual|pseudo|
        prov_pelvis="manual",       #   pseudo_corrected|null
        lstv_label="normal",
        lstv_class=0,
        lstv_pelvic="",
        lstv_vertebral="",
        lstv_agreement=None,
        lstv_confusion_zone=False,
        has_l6=False,
        n_lumbar_labels=5,
        alignment_ok=True,
        ct_resampled_to_mask=False,
        postwrite_hip_bone_pct=42.0,
        partial_annotation=False,
        n_voxels_ignore=0,
        n_voxels_fg=10000,
        n_voxels_bg=50000,
        spine_series_uid="1.2.3.spine",
        pelvic_series_uid="1.2.3.pelvic",
        spine_bone_pct=55.5,
        pelvic_bone_pct=60.1,
        ct_file=ct_file,
        label_file=ct_file.replace("ct/", "labels/").replace("_ct", "_label"),
        qc_file=ct_file.replace("ct/", "qc/").replace("_ct.nii.gz", "_qc.png"),
    )


# (a) complete fused record — labels 1..9 present, not partial
_FUSED = _base("0001", "fused", "ct/0001_ct.nii.gz")
_FUSED.update(
    lstv_label="LUMBARIZATION", lstv_class=1, lstv_pelvic="LUMBARIZATION",
    lstv_vertebral="LUMBARIZATION", lstv_agreement=True,
    has_l6=True, n_lumbar_labels=6, partial_annotation=False,
)

# (b) spine_only — lumbar only, sacrum/hips absent, pelvic provenance null
_SPINE_ONLY = _base("0002", "spine_only", "ct/0002_spine_ct.nii.gz")
_SPINE_ONLY.update(
    partial_annotation=True, n_lumbar_labels=5,
    pelvic_series_uid=None, pelvic_bone_pct=None,
    prov_spine="manual", prov_pelvis=None,   # no pelvic source
)

# (c) pelvic_native — sacrum/hips only, no lumbar labels
_PELVIC_NATIVE = _base("0003", "pelvic_native", "ct/0003_pelvic_ct.nii.gz")
_PELVIC_NATIVE.update(
    partial_annotation=True, n_lumbar_labels=0, has_l6=False,
    spine_series_uid=None, spine_bone_pct=None,
    prov_spine=None, prov_pelvis="manual",   # no spine source
)

# (d) separate-pair — two records, SAME token, one spine_only one pelvic
_SEPARATE_SPINE = _base("0004", "spine_only", "ct/0004_spine_ct.nii.gz")
_SEPARATE_SPINE.update(match_type="separate", partial_annotation=True,
                       n_lumbar_labels=5,
                       pelvic_series_uid=None, pelvic_bone_pct=None,
                       prov_spine="manual", prov_pelvis=None)
_SEPARATE_PELVIC = _base("0004", "pelvic_native", "ct/0004_pelvic_ct.nii.gz")
_SEPARATE_PELVIC.update(match_type="separate", partial_annotation=True,
                        n_lumbar_labels=0,
                        spine_series_uid=None, spine_bone_pct=None,
                        prov_spine=None, prov_pelvis="manual")

# (e) record missing several schema keys entirely (keeps ok/token/ct_file)
_MISSING_KEYS = {
    "ok": True,
    "token": "0005",
    "config": "fused",
    "ct_file": "ct/0005_ct.nii.gz",
    # deliberately omitted: position, match_type, lstv_*, has_l6,
    # n_lumbar_labels, alignment_ok, n_voxels_*, *_series_uid, *_bone_pct,
    # label_file, qc_file, ...
}

# (f) lstv_agreement as empty string "" (the original CastError trigger)
_EMPTY_AGREEMENT = _base("0006", "fused", "ct/0006_ct.nii.gz")
_EMPTY_AGREEMENT["lstv_agreement"] = ""

# (g) genuinely disagreeing lstv strings — value logic NOT under test, we
#     only assert the fields stay type-consistent strings (or null).
_DISAGREE = _base("0007", "fused", "ct/0007_ct.nii.gz")
_DISAGREE.update(
    lstv_pelvic="SACRALIZATION", lstv_vertebral="LUMBARIZATION",
    lstv_agreement=False, lstv_label="SACRALIZATION", lstv_class=3,
)

# (h) assorted fields explicitly None
_ASSORTED_NONE = _base("0008", "spine_only", "ct/0008_spine_ct.nii.gz")
_ASSORTED_NONE.update(
    lstv_pelvic=None, lstv_vertebral=None, lstv_agreement=None,
    postwrite_hip_bone_pct=None, spine_series_uid=None,
    pelvic_series_uid=None, spine_bone_pct=None, pelvic_bone_pct=None,
)

# (i) position explicitly None — DICOM Patient Position unavailable upstream
#     (placed_manifest.json schema >=2.5 emits null). Must stay JSON null.
_POS_NONE = _base("0009", "fused", "ct/0009_ct.nii.gz")
_POS_NONE["position"] = None

# (j) position key absent entirely — must coerce to JSON null, never ""
#     and never the legacy "unknown" sentinel.
_POS_MISSING = _base("0010", "fused", "ct/0010_ct.nii.gz")
del _POS_MISSING["position"]

# (k) provenance non-manual enum values — locks the pseudo /
#     pseudo_corrected domain through coercion (str column, unchanged).
_PROV_PSEUDO = _base("0011", "fused", "ct/0011_ct.nii.gz")
_PROV_PSEUDO.update(prov_spine="pseudo", prov_pelvis="pseudo_corrected")

# (l) every DICOM-header demographic explicitly None — no header tag was
#     available upstream. Each must serialize as JSON null, NEVER 0 (age),
#     0.0 (weight/size/slice_thickness/kvp) or "" (sex/kernel/manufacturer).
_DEMO_NONE = _base("0012", "fused", "ct/0012_ct.nii.gz")
_DEMO_NONE.update(
    age=None, sex=None, patient_weight=None, patient_size=None,
    convolution_kernel=None, manufacturer=None, manufacturer_model=None,
    slice_thickness=None, kvp=None,
)

# (m) every demographic key absent entirely — missing key must coerce to
#     JSON null exactly like an explicit None (never a typed zero).
_DEMO_MISSING = _base("0013", "fused", "ct/0013_ct.nii.gz")
for _k in ("age", "sex", "patient_weight", "patient_size",
           "convolution_kernel", "manufacturer", "manufacturer_model",
           "slice_thickness", "kvp"):
    del _DEMO_MISSING[_k]

RAW_RECORDS = [
    _FUSED,
    _SPINE_ONLY,
    _PELVIC_NATIVE,
    _SEPARATE_SPINE,
    _SEPARATE_PELVIC,
    _MISSING_KEYS,
    _EMPTY_AGREEMENT,
    _DISAGREE,
    _ASSORTED_NONE,
    _POS_NONE,
    _POS_MISSING,
    _PROV_PSEUDO,
    _DEMO_NONE,
    _DEMO_MISSING,
]


@pytest.fixture(scope="module")
def raw_records() -> list[dict]:
    """Defensive copies so a test mutating a record can't poison others."""
    return [dict(r) for r in RAW_RECORDS]


@pytest.fixture(scope="module")
def coerced_records(raw_records) -> list[dict]:
    return [_coerce_manifest_record(r) for r in raw_records]


# --------------------------------------------------------------------------- #
# Presence
# --------------------------------------------------------------------------- #

def test_every_coerced_record_has_exactly_the_canonical_key_set(coerced_records):
    key_sets = {frozenset(r.keys()) for r in coerced_records}
    assert key_sets == {CANONICAL_KEYS}, (
        "records produced more than one key set: "
        f"{[sorted(s ^ CANONICAL_KEYS) for s in key_sets if s != CANONICAL_KEYS]}"
    )


def test_ok_and_error_are_stripped(coerced_records):
    for r in coerced_records:
        assert "ok" not in r
        assert "error" not in r


# --------------------------------------------------------------------------- #
# Types
# --------------------------------------------------------------------------- #

def test_lstv_agreement_is_only_true_false_or_none(coerced_records):
    for r in coerced_records:
        v = r["lstv_agreement"]
        assert v is True or v is False or v is None, (
            f"token={r.get('token')!r} lstv_agreement={v!r} "
            f"(type {type(v).__name__}) — must be bool or None, never a string"
        )


def test_each_field_holds_declared_type_or_none_iff_nullable(coerced_records):
    for r in coerced_records:
        for name, py_type, nullable in _MANIFEST_SCHEMA:
            v = r[name]
            if v is None:
                assert nullable, (
                    f"non-nullable field {name!r} is None "
                    f"(token={r.get('token')!r})"
                )
                continue
            # bool is a subclass of int — assert exactly, not isinstance.
            if py_type is bool:
                assert type(v) is bool, f"{name}={v!r} not bool"
            elif py_type is int:
                assert type(v) is int, f"{name}={v!r} not int"
            elif py_type is float:
                assert type(v) is float, f"{name}={v!r} not float"
            else:
                assert type(v) is str, f"{name}={v!r} not str"


def test_non_nullable_fields_are_never_none(coerced_records):
    for r in coerced_records:
        for name in _NON_NULLABLE:
            assert r[name] is not None, (
                f"non-nullable field {name!r} serialized as None "
                f"(token={r.get('token')!r})"
            )


# --------------------------------------------------------------------------- #
# Null-vs-empty-string contract for nullable fields
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("field", sorted(_NULLABLE))
def test_nullable_field_none_or_missing_becomes_null_never_empty_string(field):
    """A nullable field whose raw input is None / "" / absent must coerce to
    Python None (JSON null) — never "" (which is what produced the original
    HF-viewer CastError on lstv_agreement)."""
    base = dict(_FUSED)

    for raw in (None, ""):
        rec = dict(base)
        rec[field] = raw
        out = _coerce_manifest_record(rec)
        assert out[field] is None, (
            f"{field}: raw {raw!r} coerced to {out[field]!r}, expected None"
        )
        assert out[field] != "", f"{field}: serialized as empty string"

    # key entirely absent
    rec = dict(base)
    rec.pop(field, None)
    out = _coerce_manifest_record(rec)
    assert out[field] is None, f"{field}: missing key coerced to {out[field]!r}"


def test_position_is_nullable_none_and_missing_serialize_as_json_null():
    """position is now a nullable str column (placed_manifest schema >=2.5
    emits null when DICOM Patient Position is unavailable). Both an explicit
    None and an absent key must coerce to JSON null — never "" and never the
    legacy "unknown" sentinel. Valid labels pass through unchanged."""
    assert "position" in _NULLABLE and "position" not in _NON_NULLABLE

    for rec in (dict(_POS_NONE), dict(_POS_MISSING)):
        out = _coerce_manifest_record(rec)
        assert out["position"] is None
        assert out["position"] != ""
        assert out["position"] != "unknown"
        reloaded = json.loads(json.dumps(out))
        assert reloaded["position"] is None

    # A valid position label is preserved verbatim.
    keep = _coerce_manifest_record(dict(_FUSED))
    assert keep["position"] == "supine"


# --------------------------------------------------------------------------- #
# DICOM header demographics (schema >=2.7) — the first nullable int / float
# columns. The CastError-class bug here would be a None coercing to 0 / 0.0
# instead of JSON null, silently corrupting an otherwise-nullable column.
# --------------------------------------------------------------------------- #

_DEMO_FLOAT_FIELDS = ("patient_weight", "patient_size",
                      "slice_thickness", "kvp")
_DEMO_STR_FIELDS   = ("sex", "convolution_kernel",
                      "manufacturer", "manufacturer_model")
_SEX_DOMAIN = {"male", "female", "other", None}


def test_age_is_nullable_int_none_and_missing_serialize_as_json_null():
    """age is the first nullable INT column. None / "" / absent must coerce
    to JSON null — NEVER 0 (which the non-nullable int default would give and
    which would read as a real 0-year-old patient)."""
    assert "age" in _NULLABLE and "age" not in _NON_NULLABLE
    assert _PYTYPE["age"] is int

    for rec in (dict(_DEMO_NONE), dict(_DEMO_MISSING)):
        out = _coerce_manifest_record(rec)
        assert out["age"] is None, f"age coerced to {out['age']!r}, expected None"
        assert out["age"] != 0
        assert json.loads(json.dumps(out))["age"] is None

    # explicit "" also collapses to null, not 0
    rec = dict(_FUSED); rec["age"] = ""
    assert _coerce_manifest_record(rec)["age"] is None

    # a real age survives coercion + JSON round-trip as a plain int
    out = _coerce_manifest_record(dict(_FUSED))
    assert out["age"] == 64 and type(out["age"]) is int
    assert json.loads(json.dumps(out))["age"] == 64


@pytest.mark.parametrize("field", _DEMO_FLOAT_FIELDS)
def test_demographic_float_fields_are_nullable_none_never_zero(field):
    """Nullable FLOAT columns: None / "" / absent -> JSON null, NEVER 0.0
    (a real measurement of 0.0 is meaningless; 0.0 here would be a silent
    null-substitution that also pins the column dtype)."""
    assert field in _NULLABLE and field not in _NON_NULLABLE
    assert _PYTYPE[field] is float

    for rec in (dict(_DEMO_NONE), dict(_DEMO_MISSING)):
        out = _coerce_manifest_record(rec)
        assert out[field] is None, f"{field}={out[field]!r}, expected None"
        assert out[field] != 0.0
        assert json.loads(json.dumps(out))[field] is None

    rec = dict(_FUSED); rec[field] = ""
    assert _coerce_manifest_record(rec)[field] is None

    # a real value survives as a plain float
    out = _coerce_manifest_record(dict(_FUSED))
    assert type(out[field]) is float
    assert json.loads(json.dumps(out))[field] == out[field]


def test_sex_is_nullable_enum_in_domain(coerced_records):
    """sex ∈ {male, female, other, null} across every coerced record;
    None / "" / absent -> JSON null."""
    assert "sex" in _NULLABLE and _PYTYPE["sex"] is str
    for r in coerced_records:
        assert r["sex"] in _SEX_DOMAIN, (
            f"token={r.get('token')!r} sex={r['sex']!r} outside "
            f"{sorted(x for x in _SEX_DOMAIN if x)} | None"
        )
    for raw in (None, ""):
        rec = dict(_FUSED); rec["sex"] = raw
        out = _coerce_manifest_record(rec)
        assert out["sex"] is None
        assert json.loads(json.dumps(out))["sex"] is None
    rec = dict(_FUSED); rec.pop("sex", None)
    assert _coerce_manifest_record(rec)["sex"] is None


def test_roundtrip_single_type_per_column_with_nullable_int_and_float(
        coerced_records):
    """The HF-viewer CastError invariant, asserted explicitly for the new
    nullable int (age) and float columns: across the WHOLE record list each
    column is either a single non-null Python type or null — mixed
    int+NoneType or float+NoneType is fine (nullable), int+str is not.
    The fixture set includes records with these fields populated AND fully
    None, so this exercises the actual mixed-presence condition."""
    reloaded = json.loads(json.dumps(coerced_records))
    for name in ("age",) + _DEMO_FLOAT_FIELDS + _DEMO_STR_FIELDS:
        py_type = _PYTYPE[name]
        non_null = {type(r[name]).__name__ for r in reloaded
                    if r[name] is not None}
        assert len(non_null) <= 1, (
            f"column {name!r} has mixed non-null types: {non_null}"
        )
        if non_null:
            expect = {int: "int", float: "float", str: "str"}[py_type]
            assert non_null == {expect}, (
                f"column {name!r} non-null type {non_null} != {{{expect}}}"
            )
        # at least one record exercises the null branch of this column
        assert any(r[name] is None for r in reloaded), (
            f"fixture set never exercises {name!r} as null"
        )


_PROV_DOMAIN = {"manual", "pseudo", "pseudo_corrected", None}


def test_label_provenance_columns_are_nullable_enum_strings(coerced_records):
    """prov_spine / prov_pelvis: flat nullable str columns, values confined
    to manual | pseudo | pseudo_corrected | None. None/missing -> JSON null
    (never "" and never an absent key); valid enum values pass through."""
    for col in ("prov_spine", "prov_pelvis"):
        assert col in _NULLABLE and col not in _NON_NULLABLE
        assert _PYTYPE[col] is str

    for r in coerced_records:
        for col in ("prov_spine", "prov_pelvis"):
            assert r[col] in _PROV_DOMAIN, (
                f"token={r.get('token')!r} {col}={r[col]!r} outside "
                f"{sorted(x for x in _PROV_DOMAIN if x)} | None"
            )

    # None, "", and a missing key all collapse to JSON null.
    for col in ("prov_spine", "prov_pelvis"):
        for raw in (None, ""):
            rec = dict(_FUSED); rec[col] = raw
            out = _coerce_manifest_record(rec)
            assert out[col] is None
            assert json.loads(json.dumps(out))[col] is None
        rec = dict(_FUSED); rec.pop(col, None)
        assert _coerce_manifest_record(rec)[col] is None

    # Non-manual enum values survive coercion + JSON round-trip verbatim.
    out = _coerce_manifest_record(dict(_PROV_PSEUDO))
    assert out["prov_spine"] == "pseudo"
    assert out["prov_pelvis"] == "pseudo_corrected"
    reloaded = json.loads(json.dumps(out))
    assert reloaded["prov_spine"] == "pseudo"
    assert reloaded["prov_pelvis"] == "pseudo_corrected"


def test_nullable_fields_serialize_as_json_null(coerced_records):
    """Every nullable field, when None, must json.dumps to literal null."""
    for field in _NULLABLE:
        rec = dict(_FUSED)
        rec[field] = None
        out = _coerce_manifest_record(rec)
        blob = json.dumps(out)          # must not raise
        reloaded = json.loads(blob)
        assert reloaded[field] is None
        assert f'"{field}": null' in blob or reloaded[field] is None


# --------------------------------------------------------------------------- #
# JSON round-trip — the actual HF-viewer CastError condition
# --------------------------------------------------------------------------- #

def test_full_set_roundtrips_json_with_single_identical_key_set(coerced_records):
    blob = json.dumps(coerced_records, indent=2)   # must not raise
    reloaded = json.loads(blob)

    assert isinstance(reloaded, list)
    assert len(reloaded) == len(coerced_records)

    key_sets = {frozenset(r.keys()) for r in reloaded}
    assert key_sets == {CANONICAL_KEYS}, (
        "post-JSON records do not share one key set — this is exactly the "
        "condition that breaks the HuggingFace dataset viewer"
    )

    # Per-column type uniformity across the whole list (None allowed only
    # for nullable columns) — the other half of the CastError condition.
    for name, py_type, nullable in _MANIFEST_SCHEMA:
        seen = {type(r[name]).__name__ for r in reloaded if r[name] is not None}
        assert len(seen) <= 1, (
            f"column {name!r} has mixed types across records: {seen}"
        )
        if not nullable:
            assert all(r[name] is not None for r in reloaded), (
                f"non-nullable column {name!r} has nulls after round-trip"
            )


# --------------------------------------------------------------------------- #
# write_manifest() vs write_splits() — schema parity
# --------------------------------------------------------------------------- #

def _load_json(p: Path):
    return json.loads(p.read_text())


def test_write_manifest_records_match_canonical_schema(tmp_path, raw_records):
    write_manifest(raw_records, tmp_path)

    recs = _load_json(tmp_path / "manifest.json")
    assert recs, "manifest.json is empty"
    assert {frozenset(r) for r in recs} == {CANONICAL_KEYS}

    for r in recs:
        v = r["lstv_agreement"]
        assert v is True or v is False or v is None


def test_split_manifests_are_schema_identical_to_main_manifest(tmp_path, raw_records):
    main_dir  = tmp_path / "main"
    split_dir = tmp_path / "split"
    main_dir.mkdir()
    split_dir.mkdir()

    write_manifest(raw_records, main_dir)
    write_splits(raw_records, split_dir, seed=42)

    main_recs = _load_json(main_dir / "manifest.json")

    split_recs: list[dict] = []
    for fname in ("manifest_train.json",
                  "manifest_validation.json",
                  "manifest_test.json"):
        split_recs.extend(_load_json(split_dir / fname))

    assert main_recs, "main manifest empty"
    assert split_recs, "all split manifests empty"

    all_recs = main_recs + split_recs

    # 1. one identical key set across BOTH writers' output
    assert {frozenset(r) for r in all_recs} == {CANONICAL_KEYS}, (
        "split manifests and main manifest disagree on the key set"
    )

    # 2. per-field type parity across both writers (None only where nullable)
    for name, _, nullable in _MANIFEST_SCHEMA:
        seen = {type(r[name]).__name__ for r in all_recs if r[name] is not None}
        assert len(seen) <= 1, (
            f"field {name!r} has mixed types across manifest+splits: {seen}"
        )
        if not nullable:
            assert all(r[name] is not None for r in all_recs), name

    # 3. lstv_agreement contract holds in the split outputs too
    for r in split_recs:
        v = r["lstv_agreement"]
        assert v is True or v is False or v is None


def test_separate_pair_shares_one_token_two_configs(coerced_records):
    pair = [r for r in coerced_records if r["token"] == "0004"]
    assert len(pair) == 2
    assert {r["config"] for r in pair} == {"spine_only", "pelvic_native"}
    assert all(r["match_type"] == "separate" for r in pair)
