"""Builds the organ-segmentation Jupyter notebook (nbformat v4)."""
import nbformat as nbf
from nbformat.v4 import new_notebook, new_markdown_cell, new_code_cell

cells = []
md = lambda s: cells.append(new_markdown_cell(s))
co = lambda s: cells.append(new_code_cell(s))

md("""# Organ segmentation of a lung-cancer CT — heart, aorta, lung, trachea

This notebook segments four organs in the CT scan downloaded from the NCI Imaging Data Commons
(`nsclc_radiomics` collection, patient **LUNG1-133**, 184 axial slices) and lets you scroll
through the volume while toggling each organ overlay on and off.

**Segmentation model:** [TotalSegmentator](https://github.com/wasserth/TotalSegmentator) — a
pretrained 3D nnU-Net that produces 100+ anatomical masks from CT. We restrict it to the four
organs of interest with `roi_subset`.

**Pipeline**
1. Configure paths
2. Load the DICOM series (SimpleITK) → save as NIfTI
3. Run TotalSegmentator → per-organ masks
4. Load masks, merge the 5 lung lobes into one lung, compute organ volumes
5. Interactive viewer: slice slider + window preset + per-organ overlay toggles

> Data license: this series is **CC BY-NC 3.0** (non-commercial use only).""")

md("""## 0. Setup

Run once if these packages are missing. (In the prepared `.idc-venv` they are already installed.)
`torch` requires `numpy<2`.""")
co("""# !pip install SimpleITK nibabel matplotlib "ipywidgets>=8" totalsegmentator pydicom-seg "numpy<2" "pydicom<3"
import numpy as np
import SimpleITK as sitk
import matplotlib.pyplot as plt
from pathlib import Path
print("numpy", np.__version__)""")

md("""## 1. Configure paths

`DICOM_DIR` is auto-detected from the IDC download created earlier
(`data/nsclc_LUNG1-133/.../CT_<SeriesInstanceUID>/`).""")
co("""ROOT = Path.cwd()
# Auto-detect the CT series folder (the one holding the .dcm files)
candidates = sorted(ROOT.glob("data/nsclc_LUNG1-133/**/CT_*"))
assert candidates, "CT folder not found — re-run the IDC download step first."
DICOM_DIR = candidates[0]

WORK     = ROOT / "seg_work"            # intermediate + outputs
CT_NIFTI = WORK / "ct.nii.gz"           # CT converted to NIfTI
SEG_DIR  = WORK / "segmentations"       # TotalSegmentator output
WORK.mkdir(exist_ok=True)

n_dcm = len(list(DICOM_DIR.glob("*.dcm")))
print("DICOM dir :", DICOM_DIR)
print("CT slices :", n_dcm)
print("Work dir  :", WORK)""")

md("""## 2. Load the DICOM series

SimpleITK reads the slices in physical order and assembles a 3-D volume. We keep the geometry
(spacing/orientation) and write a single NIfTI file — TotalSegmentator's masks will come back in
exactly this space, so the CT and masks stay perfectly aligned.

Arrays from `GetArrayFromImage` are indexed `[z, y, x]`, so an axial slice is `volume[z]`.""")
co("""reader = sitk.ImageSeriesReader()
series_ids = reader.GetGDCMSeriesIDs(str(DICOM_DIR))
files = reader.GetGDCMSeriesFileNames(str(DICOM_DIR), series_ids[0])
reader.SetFileNames(files)
ct_img = reader.Execute()

ct = sitk.GetArrayFromImage(ct_img)        # [z, y, x], Hounsfield units
sx, sy, sz = ct_img.GetSpacing()           # mm

print("Volume shape (z,y,x):", ct.shape)
print(f"Voxel spacing       : {sx:.3f} x {sy:.3f} x {sz:.3f} mm")
print(f"HU range            : {ct.min()} .. {ct.max()}")

sitk.WriteImage(ct_img, str(CT_NIFTI))
print("Saved NIfTI ->", CT_NIFTI)""")

md("""### Quick sanity check — one axial slice

Displayed with a soft-tissue window (level 40, width 400).""")
co("""z0 = ct.shape[0] // 2
lo, hi = 40 - 200, 40 + 200
plt.figure(figsize=(5, 5))
plt.imshow(np.clip((ct[z0] - lo) / (hi - lo), 0, 1), cmap="gray")
plt.title(f"axial slice {z0} / {ct.shape[0]-1}")
plt.axis("off"); plt.show()""")

md("""## 3. Run TotalSegmentator

We request only the four organs. The 5 lung lobes are segmented separately and merged in the next
step. `fast=True` uses the 3 mm model — much quicker on CPU and plenty for visualization.

The first run downloads the model weights (a few hundred MB). Output is cached: re-running this
cell skips the model if the masks already exist.""")
co('''ROI = [
    "heart", "aorta", "trachea",
    "lung_upper_lobe_left", "lung_lower_lobe_left",
    "lung_upper_lobe_right", "lung_middle_lobe_right", "lung_lower_lobe_right",
]

expected = [SEG_DIR / f"{r}.nii.gz" for r in ROI]
if all(p.exists() for p in expected):
    print("Segmentations already exist — skipping model run.")
else:
    from totalsegmentator.python_api import totalsegmentator
    totalsegmentator(
        input=str(CT_NIFTI),
        output=str(SEG_DIR),
        roi_subset=ROI,
        fast=True,
        quiet=True,
    )
    print("Done. Masks written to", SEG_DIR)

print(sorted(p.name for p in SEG_DIR.glob("*.nii.gz"))[:12])''')

md("""## 4. Load masks, merge lung lobes, compute volumes

Each mask is a binary volume in the CT's geometry. We OR the five lobes into a single `lung` mask,
then report each organ's volume in millilitres (voxel count x voxel volume / 1000).""")
co('''def load_mask(name):
    img = sitk.ReadImage(str(SEG_DIR / f"{name}.nii.gz"))
    return sitk.GetArrayFromImage(img).astype(bool)   # [z, y, x]

lung_lobes = ["lung_upper_lobe_left", "lung_lower_lobe_left",
              "lung_upper_lobe_right", "lung_middle_lobe_right", "lung_lower_lobe_right"]

masks = {
    "lung":    np.any([load_mask(l) for l in lung_lobes], axis=0),
    "heart":   load_mask("heart"),
    "aorta":   load_mask("aorta"),
    "trachea": load_mask("trachea"),
}

voxel_ml = (sx * sy * sz) / 1000.0
print(f"{'organ':<9}{'voxels':>12}{'volume (mL)':>14}")
for name, m in masks.items():
    n = int(m.sum())
    print(f"{name:<9}{n:>12,}{n * voxel_ml:>14.1f}")''')

md("""## 5. Interactive viewer

Scroll through the volume and toggle each organ overlay.

- **slice** — axial position
- **window** — CT display preset (lung / mediastinum / bone)
- **alpha** — overlay opacity
- **checkboxes** — show/hide each organ

Colors: <span style="color:#1D9E75">lung</span>,
<span style="color:#D85A30">heart</span>,
<span style="color:#534AB7">aorta</span>,
<span style="color:#BA7517">trachea</span>.""")
co('''from ipywidgets import interact, IntSlider, FloatSlider, Dropdown, Checkbox

WINDOWS = {"lung (-600/1500)": (-600, 1500),
           "mediastinum (40/400)": (40, 400),
           "bone (300/1500)": (300, 1500)}

COLORS = {"lung": (0.114, 0.620, 0.459),
          "heart": (0.847, 0.353, 0.188),
          "aorta": (0.325, 0.290, 0.718),
          "trachea": (0.729, 0.459, 0.090)}

def view(slice_idx, window, alpha, lung, heart, aorta, trachea):
    wl, ww = WINDOWS[window]
    lo, hi = wl - ww / 2, wl + ww / 2
    base = np.clip((ct[slice_idx].astype(float) - lo) / (hi - lo), 0, 1)
    rgb = np.stack([base] * 3, axis=-1)
    for name, on in [("lung", lung), ("heart", heart), ("aorta", aorta), ("trachea", trachea)]:
        if not on:
            continue
        m = masks[name][slice_idx]
        if not m.any():
            continue
        col = COLORS[name]
        for c in range(3):
            rgb[..., c][m] = (1 - alpha) * rgb[..., c][m] + alpha * col[c]
    plt.figure(figsize=(7, 7))
    plt.imshow(rgb)
    plt.title(f"axial slice {slice_idx} / {ct.shape[0]-1}   |   {window}")
    plt.axis("off"); plt.show()

interact(
    view,
    slice_idx=IntSlider(min=0, max=ct.shape[0]-1, value=ct.shape[0]//2, description="slice"),
    window=Dropdown(options=list(WINDOWS), value="mediastinum (40/400)", description="window"),
    alpha=FloatSlider(min=0.1, max=0.9, step=0.1, value=0.45, description="alpha"),
    lung=Checkbox(value=True, description="lung"),
    heart=Checkbox(value=True, description="heart"),
    aorta=Checkbox(value=True, description="aorta"),
    trachea=Checkbox(value=True, description="trachea"),
);''')

md("""## 6. Save a static overlay montage (optional)

Picks slices where the organs are present and writes a labelled PNG you can drop into a report.""")
co('''def overlay(slice_idx, alpha=0.45, wl=40, ww=400):
    lo, hi = wl - ww / 2, wl + ww / 2
    base = np.clip((ct[slice_idx].astype(float) - lo) / (hi - lo), 0, 1)
    rgb = np.stack([base] * 3, axis=-1)
    for name, col in COLORS.items():
        m = masks[name][slice_idx]
        for c in range(3):
            rgb[..., c][m] = (1 - alpha) * rgb[..., c][m] + alpha * col[c]
    return rgb

any_organ = np.any([m for m in masks.values()], axis=0).reshape(ct.shape[0], -1).any(axis=1)
zs = np.linspace(np.argmax(any_organ), len(any_organ) - 1 - np.argmax(any_organ[::-1]), 6).astype(int)

fig, axes = plt.subplots(2, 3, figsize=(12, 8))
for ax, z in zip(axes.ravel(), zs):
    ax.imshow(overlay(z)); ax.set_title(f"slice {z}"); ax.axis("off")
handles = [plt.Line2D([0], [0], marker="s", ls="", markersize=12, color=c, label=n)
           for n, c in COLORS.items()]
fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False)
fig.suptitle("LUNG1-133 — organ segmentation (TotalSegmentator)")
fig.tight_layout(rect=[0, 0.04, 1, 1])
out = WORK / "organ_overlay_montage.png"
fig.savefig(out, dpi=110, bbox_inches="tight")
print("Saved ->", out)
plt.show()''')

md("""## 7. Compare with the segmentations that ship with IDC

`nsclc_radiomics` ships expert and AI segmentations for this patient (downloaded to
`data/nsclc_LUNG1-133_segmentations/`):

- **Expert SEG** (manual + semiautomatic): primary tumor (`Neoplasm, Primary` = GTV), lung, spinal cord
- **AI nnU-Net SEG** (we use the `3d_fullres` variant): heart, aorta, trachea, esophagus

We decode these DICOM SEG objects into CT-aligned masks with `pydicom-seg`, then (a) overlay the
**tumor** that TotalSegmentator cannot produce, and (b) score heart/aorta/trachea agreement
(Dice) between our TotalSegmentator run and IDC's reference.

> `pydicom-seg` requires `pydicom<3` — `pip install pydicom-seg "pydicom<3"`.""")
co('''import glob, pydicom, pydicom_seg

SEG_BASE = ROOT / "data/nsclc_LUNG1-133_segmentations"

def seg_to_masks(seg_file):
    """Decode a DICOM SEG into {label: bool array} aligned to the CT grid."""
    dcm = pydicom.dcmread(str(seg_file))
    res = pydicom_seg.SegmentReader().read(dcm)
    out = {}
    for n in res.available_segments:
        label = dcm.SegmentSequence[n - 1].get("SegmentLabel", f"seg{n}")
        seg_img = res.segment_image(n)
        r = sitk.Resample(seg_img, ct_img, sitk.Transform(),
                          sitk.sitkNearestNeighbor, 0, seg_img.GetPixelID())
        m = sitk.GetArrayFromImage(r).astype(bool)
        out[label] = out.get(label, np.zeros_like(m)) | m   # merge same-label segments (e.g. 2x Lung)
    return out

expert_file = None
variant_files = {}   # {"2d": path, "3d_lowres": path, "3d_fullres": path}
for f in glob.glob(str(SEG_BASE / "**/*.dcm"), recursive=True):
    d = pydicom.dcmread(f, stop_before_pixels=True)
    if d.Modality != "SEG":
        continue
    desc = d.get("SeriesDescription", "") or ""
    if desc == "Segmentation":
        expert_file = f
    for v in ("3d_fullres", "3d_lowres", "2d"):
        if desc.startswith(v):
            variant_files[v] = f

expert = seg_to_masks(expert_file)
nnunet_variants = {v: seg_to_masks(variant_files[v]) for v in variant_files}   # all 3 nnU-Net configs
nnunet = nnunet_variants["3d_fullres"]   # reference variant for the viewer

idc_masks = {
    "tumor":       expert["Neoplasm, Primary"],
    "lung":        expert["Lung"],
    "spinal_cord": expert["Spinal cord"],
    "heart":       nnunet["Heart"],
    "aorta":       nnunet["Aorta"],
    "trachea":     nnunet["Trachea"],
    "esophagus":   nnunet["Esophagus"],
}
print("nnU-Net variants loaded:", list(nnunet_variants))
print("\\nIDC-shipped layers, 3d_fullres (mL):")
for k, v in idc_masks.items():
    print(f"  {k:12s} {v.sum() * voxel_ml:8.1f}")''')

md("""### Agreement across all models — Dice

Dice = 2·|A∩B| / (|A|+|B|). 1.0 = identical, 0 = no overlap. ~0.8–0.9 is typical between two good
automated methods. We compare your **TotalSegmentator** against all three IDC **nnU-Net** configs
(2D, 3D-lowres, 3D-fullres) for the three organs all four models share (heart, aorta, trachea).""")
co('''import pandas as pd

def dice(a, b):
    s = a.sum() + b.sum()
    return 2 * np.logical_and(a, b).sum() / s if s else float("nan")

shared = ["heart", "aorta", "trachea"]
variant_order = ["2d", "3d_lowres", "3d_fullres"]

# (1) TotalSegmentator vs each nnU-Net variant
rows = []
for organ in shared:
    row = {"organ": organ, "TotalSeg_mL": round(masks[organ].sum() * voxel_ml, 1)}
    for v in variant_order:
        row[f"Dice_vs_{v}"] = round(dice(masks[organ], nnunet_variants[v][organ.capitalize()]), 3)
    rows.append(row)
ts_vs = pd.DataFrame(rows)
print("TotalSegmentator vs IDC nnU-Net variants:")
print(ts_vs.to_string(index=False))

# Lung: the nnU-Net OAR models do NOT segment lung, so the only reference is the expert SEG.
lung_dice = dice(masks["lung"], idc_masks["lung"])
lung_ts   = masks["lung"].sum() * voxel_ml
lung_exp  = idc_masks["lung"].sum() * voxel_ml
print(f"\\nLung (no nnU-Net variant -> vs expert SEG):  "
      f"TotalSeg={lung_ts:.0f} mL  expert={lung_exp:.0f} mL  "
      f"Dice={lung_dice:.3f}  vol_diff={lung_ts - lung_exp:+.0f} mL")

# (2) Inter-variant agreement among the three nnU-Net configs (model self-consistency)
print("\\nInter-variant Dice (nnU-Net configs vs each other):")
pairs = [("2d", "3d_lowres"), ("2d", "3d_fullres"), ("3d_lowres", "3d_fullres")]
rows2 = []
for organ in shared:
    r = {"organ": organ}
    for a, b in pairs:
        r[f"{a}|{b}"] = round(dice(nnunet_variants[a][organ.capitalize()],
                                   nnunet_variants[b][organ.capitalize()]), 3)
    rows2.append(r)
inter = pd.DataFrame(rows2)
print(inter.to_string(index=False))

# Volume spread per organ across all four models (coefficient of variation)
print("\\nVolume (mL) across models + CV:")
for organ in shared:
    vols = [masks[organ].sum() * voxel_ml] + [nnunet_variants[v][organ.capitalize()].sum() * voxel_ml
                                              for v in variant_order]
    cv = np.std(vols) / np.mean(vols) * 100
    print(f"  {organ:8s} TS={vols[0]:5.0f}  2d={vols[1]:5.0f}  lowres={vols[2]:5.0f}  "
          f"fullres={vols[3]:5.0f}  CV={cv:4.1f}%")''')

md("""### Inter-model variability — interpretation

Two distinct sources of disagreement show up above:

- **Between architectures (TotalSegmentator vs nnU-Net):** the larger gap. These are different networks
  trained on different data, so Dice in the ~0.79–0.91 range (heart < aorta < trachea) reflects genuine
  methodological differences — especially at fuzzy soft-tissue boundaries (heart pericardium, aorta wall)
  where even experts disagree. Compact, high-contrast structures (trachea = air) score highest.
- **Within one architecture (nnU-Net 2D vs 3D-lowres vs 3D-fullres):** smaller, but non-zero. This is the
  *configuration* sensitivity of a single method — 2D works slice-by-slice (weaker through-plane
  continuity), 3D-fullres sees the whole volume at full resolution (usually the most reliable),
  3D-lowres trades resolution for context. The volume coefficient of variation (CV) quantifies the
  total spread.
- **Lung** is a special case: the IDC nnU-Net models here segment only organs-at-risk (heart, aorta,
  trachea, esophagus), so there is **no nnU-Net lung** to compare against — lung is scored against the
  **expert SEG** instead. Agreement is the highest of any structure (**Dice ≈ 0.974**, volume 5723 mL vs
  5868 mL, a −145 mL / ~2.5% difference). That is expected: the lung is a large, high-contrast,
  air-filled organ whose boundary is unambiguous, so even different methods converge. The small volume
  deficit is mostly TotalSegmentator's fast 3 mm model trimming thin peripheral/peri-fissural voxels.

**Takeaway for downstream use (e.g. tumor/organ volumetry):** report which model *and config* produced a
mask, and treat the inter-model CV as a floor on measurement uncertainty. For longitudinal volume change,
hold the model fixed across timepoints so method variability doesn't masquerade as biological change.""")

md("""### Side-by-side viewer — TotalSegmentator vs IDC (with the tumor)

Left = your TotalSegmentator masks. Right = IDC's shipped masks plus the expert **tumor (GTV)** in
magenta. Same slice/window controls drive both panels.""")
co('''from ipywidgets import interact, IntSlider, FloatSlider, Dropdown, Checkbox

CMP_COLORS = {"lung": (0.114, 0.620, 0.459), "heart": (0.847, 0.353, 0.188),
              "aorta": (0.325, 0.290, 0.718), "trachea": (0.729, 0.459, 0.090),
              "tumor": (0.92, 0.13, 0.55)}

def _overlay(ax, z, mask_set, title, window, alpha, organs, show_tumor):
    wl, ww = WINDOWS[window]; lo, hi = wl - ww/2, wl + ww/2
    base = np.clip((ct[z].astype(float) - lo) / (hi - lo), 0, 1)
    rgb = np.stack([base]*3, axis=-1)
    layers = list(organs)
    if show_tumor and "tumor" in mask_set:
        layers = layers + ["tumor"]
    for name in layers:
        if name not in mask_set:
            continue
        m = mask_set[name][z]
        if not m.any():
            continue
        col = CMP_COLORS[name]
        for c in range(3):
            rgb[..., c][m] = (1 - alpha) * rgb[..., c][m] + alpha * col[c]
    ax.imshow(rgb); ax.set_title(title, fontsize=11); ax.axis("off")

def compare(slice_idx, window, alpha, heart, aorta, lung, trachea, tumor):
    organs = [n for n, on in [("heart", heart), ("aorta", aorta),
                              ("lung", lung), ("trachea", trachea)] if on]
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(13, 6.5))
    _overlay(axL, slice_idx, masks,     "TotalSegmentator (yours)", window, alpha, organs, False)
    _overlay(axR, slice_idx, idc_masks, "IDC shipped (+ tumor)",    window, alpha, organs, tumor)
    fig.suptitle(f"axial slice {slice_idx} / {ct.shape[0]-1}", fontsize=12)
    plt.tight_layout(); plt.show()

interact(
    compare,
    slice_idx=IntSlider(min=0, max=ct.shape[0]-1, value=ct.shape[0]//2, description="slice"),
    window=Dropdown(options=list(WINDOWS), value="mediastinum (40/400)", description="window"),
    alpha=FloatSlider(min=0.1, max=0.9, step=0.1, value=0.45, description="alpha"),
    heart=Checkbox(value=True, description="heart"),
    aorta=Checkbox(value=True, description="aorta"),
    lung=Checkbox(value=False, description="lung"),
    trachea=Checkbox(value=True, description="trachea"),
    tumor=Checkbox(value=True, description="tumor (GTV)"),
);''')

md("""## 8. Package this scan as a per-timepoint record

NSCLC-Radiomics is **single-timepoint** (one pre-treatment CT per patient), so tumor *growth rate*
can't be computed from LUNG1-133 alone — that needs ≥2 scans over time. What we *can* do is capture
this scan as one clean, **append-ready timepoint record**: acquisition date/time + GTV size + organ
volumes in a tidy row. Drop additional timepoints into the same CSV later (or see the
`therapyNgrowthtracking/` notebook for a true longitudinal example) and growth metrics fall out.

Acquisition dates are read straight from the DICOM tags. Note: TCIA shifts dates per-patient, so the
absolute calendar date is anonymized but **intervals within a patient are preserved** — exactly what
growth-rate math relies on.""")
co('''import pandas as pd, pydicom, glob, json

ct_file = sorted(glob.glob(str(DICOM_DIR / "*.dcm")))[0]
hdr = pydicom.dcmread(ct_file, stop_before_pixels=True)

record = {
    "collection_id":     "nsclc_radiomics",
    "PatientID":         hdr.get("PatientID", ""),
    "timepoint":         1,                       # increment for future scans
    "StudyDate":         hdr.get("StudyDate", ""),       # YYYYMMDD (anonymized, interval-preserving)
    "SeriesDate":        hdr.get("SeriesDate", ""),
    "AcquisitionDate":   hdr.get("AcquisitionDate", ""),
    "AcquisitionTime":   hdr.get("AcquisitionTime", ""),
    "Modality":          hdr.get("Modality", ""),
    "SeriesInstanceUID": hdr.get("SeriesInstanceUID", ""),
    "n_slices":          int(ct.shape[0]),
    "SliceThickness_mm": float(hdr.get("SliceThickness", "nan")),
    "license":           "CC BY-NC 3.0",
    # measured volumes (mL)
    "GTV_tumor_mL":      round(float(idc_masks["tumor"].sum() * voxel_ml), 3),
    "lung_mL":           round(float(masks["lung"].sum() * voxel_ml), 1),
    "heart_mL":          round(float(masks["heart"].sum() * voxel_ml), 1),
    "aorta_mL":          round(float(masks["aorta"].sum() * voxel_ml), 1),
    "trachea_mL":        round(float(masks["trachea"].sum() * voxel_ml), 1),
    "GTV_source":        "expert SEG (Neoplasm, Primary)",
    "organ_source":      "TotalSegmentator (fast 3mm)",
}

rec_path = WORK / "timepoint_records.csv"
df_rec = pd.DataFrame([record])
if rec_path.exists():
    prev = pd.read_csv(rec_path)
    df_rec = (pd.concat([prev, df_rec], ignore_index=True)
                .drop_duplicates(subset=["PatientID", "SeriesInstanceUID"], keep="last"))
df_rec.to_csv(rec_path, index=False)
print("Per-timepoint record (append-ready):", rec_path)
print(json.dumps(record, indent=2))''')

md("""## 9. Tumor (GTV) segmentation — pre-existing expert annotation

**Is there already a trained tumor-segmentation layer for this dataset?** For the *tumor* specifically,
the segmentation that ships with NSCLC-Radiomics is an **expert-curated GTV**, not an AI model: the
`Neoplasm, Primary` segment in the DICOM SEG (equivalently `GTV-1` in the RTSTRUCT), delineated
manually/semi-automatically for the original "Lung1" radiotherapy study. Two caveats worth stating:

- The **AI** annotations IDC ships for this collection (the nnU-Net variants used above) cover only
  **organs-at-risk** (heart, aorta, trachea, esophagus) — there is no AI *tumor* model here.
- **TotalSegmentator does not segment tumors** either.

So this expert GTV is the authoritative tumor layer. Below we load it and report the GTV and the
GTV-to-total-lung ratio for **every CT image of LUNG1-133** (this patient has a single CT series in
NSCLC-Radiomics, so the table has one row, but the code generalizes to multi-image patients).""")
co('''import pandas as pd, numpy as np, pydicom, glob

def best_slice(mask):
    return int(np.argmax(mask.reshape(mask.shape[0], -1).sum(axis=1)))

tumor      = idc_masks["tumor"]      # expert GTV (Neoplasm, Primary)
lung_total = idc_masks["lung"]       # expert lung = anatomical denominator
gtv_ml  = float(tumor.sum() * voxel_ml)
lung_ml = float(lung_total.sum() * voxel_ml)
ratio   = gtv_ml / lung_ml * 100

suid = pydicom.dcmread(sorted(glob.glob(str(DICOM_DIR / "*.dcm")))[0],
                       stop_before_pixels=True).SeriesInstanceUID
tumor_table = pd.DataFrame([{
    "PatientID":         "LUNG1-133",
    "SeriesInstanceUID": suid,
    "GTV_mL":            round(gtv_ml, 3),
    "total_lung_mL":     round(lung_ml, 1),
    "GTV_to_lung_pct":   round(ratio, 4),
    "tumor_source":      "expert SEG: Neoplasm, Primary (= RTSTRUCT GTV-1)",
}])
tumor_table.to_csv(WORK / "tumor_gtv.csv", index=False)
print(tumor_table.to_string(index=False))
print(f"\\nGTV = {gtv_ml:.2f} mL   total lung = {lung_ml:.0f} mL   GTV/lung = {ratio:.4f} %")

# Visualize the GTV on the slice with the largest tumor cross-section (soft-tissue window)
z = best_slice(tumor)
lo, hi = 40 - 200, 40 + 200
base = np.clip((ct[z].astype(float) - lo) / (hi - lo), 0, 1)
rgb = np.stack([base] * 3, axis=-1)
m = tumor[z]
rgb[..., 0][m] = 0.4 * rgb[..., 0][m] + 0.6 * 0.92
rgb[..., 1][m] = 0.4 * rgb[..., 1][m]
rgb[..., 2][m] = 0.4 * rgb[..., 2][m] + 0.6 * 0.55
plt.figure(figsize=(6, 6))
plt.imshow(rgb); plt.axis("off")
plt.title(f"LUNG1-133 GTV (expert) - slice {z}\\n{gtv_ml:.2f} mL  =  {ratio:.3f}% of lung volume")
plt.show()''')

md("""## 10. Citations and attribution

Generated from `source_DOI` values via `idc-index` plus the canonical method papers. Cite all of these
when publishing results derived from this data. The **tumor GTV** used in Section 9 is part of the
NSCLC-Radiomics dataset (Aerts et al.) — cite that dataset when reporting GTV-derived results.

**Imaging dataset (NSCLC-Radiomics / "Lung1")**
- Aerts, H. J. W. L., Wee, L., Rios Velazquez, E., et al. (2019). *Data From NSCLC-Radiomics* (Version 4)
  [Dataset]. The Cancer Imaging Archive. https://doi.org/10.7937/K9/TCIA.2015.PF0M9REI
- Aerts, H. J. W. L., Velazquez, E. R., Leijenaar, R. T. H., et al. (2014). Decoding tumour phenotype by
  noninvasive imaging using a quantitative radiomics approach. *Nature Communications*, 5, 4006.
  https://doi.org/10.1038/ncomms5006

**The Cancer Imaging Archive (TCIA)**
- Clark, K., Vendt, B., Smith, K., et al. (2013). The Cancer Imaging Archive (TCIA): Maintaining and
  Operating a Public Information Repository. *Journal of Digital Imaging*, 26(6), 1045–1057.
  https://doi.org/10.1007/s10278-013-9622-7

**Imaging Data Commons (IDC)**
- Fedorov, A., Longabaugh, W. J. R., Pot, D., et al. (2023). National Cancer Institute Imaging Data
  Commons: Toward Transparency, Reproducibility, and Scalability in Imaging Artificial Intelligence.
  *RadioGraphics*, 43(12), e230180. https://doi.org/10.1148/rg.230180

**AI segmentation layers used here**
- *TotalSegmentator (this notebook's organ masks):* Wasserthal, J., Breit, H.-C., Meyer, M. T., et al.
  (2023). TotalSegmentator: Robust Segmentation of 104 Anatomical Structures in CT Images.
  *Radiology: Artificial Intelligence*, 5(5), e230024. https://doi.org/10.1148/ryai.230024
- *IDC-shipped nnU-Net annotations (the comparison masks):* Krishnaswamy, D., Bontempi, D., Clunie, D.,
  Aerts, H., & Fedorov, A. (2023). *AI-derived annotations for the NLST and NSCLC-Radiomics CT imaging
  collections* [Dataset]. Zenodo. https://doi.org/10.5281/zenodo.7473970
- *nnU-Net framework:* Isensee, F., Jaeger, P. F., Kohl, S. A. A., Petersen, J., & Maier-Hein, K. H.
  (2021). nnU-Net: a self-configuring method for deep learning-based biomedical image segmentation.
  *Nature Methods*, 18(2), 203–211. https://doi.org/10.1038/s41592-020-01008-z""")

nb = new_notebook(cells=cells, metadata={
    "kernelspec": {"display_name": "Python 3 (.idc-venv)", "language": "python", "name": "idc-venv"},
    "language_info": {"name": "python"},
})
nbf.write(nb, "organ_segmentation.ipynb")
print("Wrote organ_segmentation.ipynb with", len(cells), "cells")
