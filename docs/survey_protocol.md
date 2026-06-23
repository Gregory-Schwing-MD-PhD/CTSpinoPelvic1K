# OpenSpineConsortium PACS demo — survey protocol (CROSS / CHERRIES)

Evaluation of the web-based spinopelvic PACS demo (http://openspineconsortium.com/pacs/)
against respondents' current standard-of-care tools for adult-spinal-deformity (ASD)
CT review and surgical planning.

Reported to **CROSS** (Checklist for Reporting Of Survey Studies) and **CHERRIES**
(Checklist for Reporting Results of Internet E-Surveys). Two instruments: **A — spine
surgeons** (AO Spine / Scoliosis Research Society / Scoliosis Foundation networks) and
**B — neuroradiologists**.

> **Scope correction (important).** The engine is scan- and posture-agnostic; the demo
> ships a single sample CT only to exercise rendering and the automated measurements.
> The survey therefore evaluates the **software** (web deployment, automated
> measurement, classification, surgical-planning + post-op simulation, UI/UX), **not**
> the clinical parameters of the demo scan. No supine-vs-standing questions are asked,
> and that framing is not surfaced to respondents.

---

## 1. Design & standards
- **Type:** cross-sectional, task-based usability + comparative-preference web survey.
- **Reporting:** CROSS (all 40 items) + CHERRIES (web-survey integrity items).
- **Platform:** REDCap or Qualtrics (institutional). Both support unique per-invite
  links, completion-rate tracking, and forced-response where appropriate.
- **Validated instruments embedded:** System Usability Scale (SUS, 10 items) and a
  single NASA-TLX-style cognitive-load item per task.
- **Estimated burden:** 10–12 min (incl. ~4 min hands-on with the demo).

## 2. Methodology (CROSS/CHERRIES items)
1. **Sampling frame & recruitment (CROSS 3–4).** Defined, closed frame via society
   partnership (AO Spine, SRS, Scoliosis Foundation) — society emails its ASD-focused
   members; we report invitations sent → true response rate. NOT an open social-media
   post. (See §5.)
2. **Instrument validation (CROSS 6).** SUS + NASA-TLX (validated) plus domain items
   pilot-tested with 3–5 surgeons / radiologists before launch; revise for clarity.
3. **Data integrity (CHERRIES 2,5).** Unique per-respondent links (one submission each);
   cookie/IP de-duplication; one attention-check item ("select 'Agree' to confirm you
   are reading"); timestamp + completeness logging.
4. **Task-based A/B execution.** Respondents (a) estimate the effort in their CURRENT
   tool, then (b) perform the same tasks in the demo, then (c) rate it — so we compare
   *doing*, not *recalling*. (See task block in each instrument.)
5. **Dropout handling (CHERRIES 7).** Pre-specified: report **view rate** (opened),
   **participation rate** (started), **completion rate** (finished); primary analysis on
   completed responses; report partials separately; no imputation for the pilot.
6. **Ethics.** IRB exemption/approval as a non-interventional usability survey of
   professionals; electronic consent on the landing page; no PHI collected.

## 3. Pre-survey disclaimer (landing page — shown to all)
> This study evaluates a software prototype. The sample CT in the demo is for
> demonstrating rendering, navigation, and automated measurement only — please rate the
> **speed, interface, automation, and tooling**, not the specific anatomy of the demo
> scan. ~10 minutes; your responses are anonymous.

## 4. Instruments

### Common: hands-on task block (both instruments)
Open http://openspineconsortium.com/pacs/ and:
1. **Load** the study (note responsiveness vs your usual PACS pull).
2. **Navigate** — scroll slices, left-drag zoom, right-drag pan (or touch).
3. **Measure** — click **PI, SS, PT, LL**; watch each construction render.
4. **Classify** — read the **SRS-Schwab** modifiers and the **Surgical planning** card.
5. **Simulate** — toggle **Post-op (sim)** and observe the predicted corrected alignment.

After the tasks: SUS (10 items), one cognitive-load item, and the module questions.

---

### Instrument A — Spine surgeons

**A0. Consent + attention check.**

**A1. Demographics / baseline (CROSS).**
- Specialty: Orthopaedic spine / Neurosurgery / Other.
- Years in independent practice: <5 / 5–15 / >15.
- Annual ASD (deformity-correction) case volume: <10 / 10–30 / 30–60 / >60.
- Primary pre-op planning tool: Medtronic UNiD (Surgimap) / Brainlab Elements Spine /
  mediCAD 3D Spine / Sectra / OsiriX–Horos / Visage / standard hospital PACS (manual) /
  other.
- Where you usually plan: IT-managed hospital workstation / personal laptop / both.

**A2. Workflow efficiency (vs current tool).**
- In your current tool, getting PI/PT/SS/LL on a case takes: <1 / 1–3 / 3–5 / >5 min.
- Approx. manual clicks/line-draws to produce those four parameters today: ____.
- "The demo's one-click PI/SS/PT/LL would reduce my measurement time." (1–5)

**A3. Automation & classification.**
- "Automatic, consistent spinopelvic measurement (no manual line-drawing) is valuable
  in my workflow." (1–5)
- "The automated SRS-Schwab modifiers (PI−LL, PT) would be useful at the point of
  review." (1–5)
- "Auto-computed measurements reduce inter-observer variability vs manual measurement."
  (1–5)

**A4. Surgical planning + post-op simulation (vs UNiD-type tools).**
- "A recommended correction (lordosis to restore + a matched approach, from the
  measured parameters) is useful for planning." (1–5)
- "The post-op simulation (predicted corrected alignment) is useful for planning /
  patient communication." (1–5)
- "Compared with my current predictive-planning tool (e.g. UNiD), this demo's
  planning + simulation is: much worse … much better." (1–5)
- Free text: what would the planning/simulation need to be clinically usable?

**A5. Zero-footprint / web deployment.**
- "Loading and manipulating a full 3-D spine study in a browser, with no local install,
  is valuable." (1–5)
- "Web/PACS-embedded availability matters vs a standalone desktop planner." (1–5)

**A6. Performance & usability.**
- "Slice/zoom/pan in the web viewer felt fluid vs my desktop PACS." (1–5)
- **SUS** (10 items, standard wording, anchored to "this demo").
- Cognitive load: "How mentally demanding was getting the four parameters in the demo?"
  (1 very low – 10 very high).

**A7. Adoption & open feedback.**
- "I would use this (or want it embedded in our PACS) if validated." (1–5)
- Top blocker to adoption? (free text)
- Willing to be re-contacted / serve as an advisor? (optional email)

---

### Instrument B — Neuroradiologists

**B0. Consent + attention check.**

**B1. Demographics / baseline.**
- Role: Neuroradiologist / Musculoskeletal radiologist / Trainee / Other.
- Years post-training: <5 / 5–15 / >15.
- Spine-CT studies read per week: <10 / 10–30 / 30–60 / >60.
- Current measurement workflow for spinopelvic/alignment params: manual PACS tools /
  semi-automated / I don't routinely measure these / other.
- PACS/workstation: Sectra / Visage / Philips / GE / OsiriX–Horos / other.

**B2. Reporting workflow & automation.**
- "I currently measure spinopelvic parameters when reading spine CT." (never…always)
- "Automatic PI/SS/PT/LL with the drawn constructions would speed my reporting." (1–5)
- "An auto-generated, structured alignment summary (params + SRS-Schwab) would improve
  my reports." (1–5)
- "Consistent automated measurement would reduce my inter-/intra-observer variability."
  (1–5)

**B3. Trust & verifiability.**
- "Seeing the construction drawn on the image (not just a number) increases my trust in
  the automated value." (1–5)
- "I would want to adjust/override the automated landmarks before signing." (1–5)

**B4. Zero-footprint / web deployment.**
- "Browser-based review with no local install is valuable for my reading setup." (1–5)
- "This would be useful as a PACS plug-in / second-read alignment tool." (1–5)

**B5. Performance & usability.**
- "Slice/zoom/pan felt fluid vs my reading workstation." (1–5)
- **SUS** (10 items, anchored to "this demo").
- Cognitive load: "How mentally demanding was producing the alignment summary?" (1–10).

**B6. Adoption & open feedback.**
- "I would use this for spine-CT alignment reporting if validated." (1–5)
- Most useful feature? Biggest gap? (free text)
- Re-contact / advisor interest? (optional email)

---

## 5. Recruitment & email list
**Preferred (defensible response rate): society partnership.** Email the societies
asking them to distribute the survey to their ASD-relevant members; this gives a defined
denominator (invitations sent) for CROSS-compliant reporting.
- **AO Spine** — Knowledge Forum Deformity; research/community-engagement contact.
- **Scoliosis Research Society (SRS)** — research grants/committee + membership office.
- **Scoliosis Foundation / Setting Scoliosis Straight** — research outreach.
- **Neuroradiology:** ASNR (American Society of Neuroradiology) / ASSR (American Society
  of Spine Radiology) for instrument B.

**Mechanics:** society blasts a unique-link REDCap/Qualtrics URL (or a generic link with
de-dup) to its list; we never see member emails. Report invitations sent, reminders (1–2
spaced ~1 week), and the closing date.

**Fallback if a society won't distribute:** targeted personal invitations to a named list
of ASD surgeons / spine neuroradiologists (still a defined frame, smaller n), plus
snowball via co-authors/advisors (e.g. Dr. Steck, Dr. Khan) — disclose the convenience
element in the limitations.

### Draft outreach to a society (adapt per society)
> Dear [Society / Committee],
>
> I am developing an open, web-based tool that computes spinopelvic parameters
> (PI, SS, PT, LL), SRS-Schwab modifiers, a recommended correction, and a post-operative
> simulation directly from a spine CT in the browser (demo:
> http://openspineconsortium.com/pacs/). I am conducting a brief (~10-minute), IRB-[exempt/
> approved], anonymous usability survey comparing it to the tools your members use today,
> reported to CROSS/CHERRIES standards.
>
> Would [Society] be willing to share the survey link with the relevant members
> (deformity surgeons / spine neuroradiologists)? I would be glad to share results and
> acknowledge [Society]'s support.
>
> Thank you for considering,
> [Name, role, affiliation]

## 6. Analysis (pre-specified)
- Primary: SUS score (mean, 0–100) for the demo; % rating each value-statement ≥4/5.
- Comparative: preference vs current tool (A4), time/clicks delta (A2/B2).
- Subgroups: ortho vs neuro; ASD volume; current-tool cohort.
- Report view/participation/completion rates and all CROSS items in the manuscript.
