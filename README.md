# Lung-cancer CT segmentation & tumor growth tracking (NCI Imaging Data Commons)

Reproducible notebooks for organ and tumor segmentation on public lung-cancer CT from the
[NCI Imaging Data Commons (IDC)](https://portal.imaging.datacommons.cancer.gov/), using
[`idc-index`](https://github.com/ImagingDataCommons/idc-index) for data access — no authentication
required. The work compares an off-the-shelf segmentation model against the expert and AI
segmentations that ship with the data, and demonstrates longitudinal tumor-volume tracking across a
therapy window.

> **Data licensing:** the imaging data is **not** redistributed here — the notebooks download it on
> demand via `idc-index`. The NSCLC-Radiomics series used is **CC BY-NC 3.0 (non-commercial only)**.
> The *code* in this repository is MIT-licensed (see [LICENSE](LICENSE)).

## Notebooks

### 1. `seg_work/organ_segmentation.ipynb` — organ & tumor segmentation comparison

Single-timepoint analysis of NSCLC-Radiomics patient **LUNG1-133** (chest CT, 184 axial slices).

- **Loads** the DICOM series with SimpleITK.
- **Segments** heart, aorta, lung, and trachea with [**TotalSegmentator**](https://github.com/wasserth/TotalSegmentator)
  (pretrained 3D nnU-Net), merging the five lung lobes into one lung.
- **Interactive viewer** (ipywidgets): scroll slices, switch CT windows, toggle each organ overlay.
- **Compares** TotalSegmentator against the segmentations IDC ships for this patient (*AI nnU-Net SEG*) in three configs (`2d`, `3d_lowres`, `3d_fullres`): compared performance of 4 models at three organs that all models shared (i.e. heart, aorta, trachea). In case of the *lung*, TotalSegmentator performance was compared to that of the expert-curated segmentation (**expert SEG**).
- **Dice agreement** across all 4 models, plus inter-model variability analysis (cross-architecture
  vs within-architecture config spread).
- **Tumor (GTV)** quantification from the expert SEG: GTV volume and GTV-to-lung ratio (I did this just for fun honestly).
- A static overlay montage tool for figures.

**Headline results (LUNG1-133):**

| Structure | TotalSeg vs reference (Dice) | Notes |
|-----------|------------------------------|-------|
| trachea   | 0.90–0.92 (vs nnU-Net)       | highest match due to air-filled structure (sharp boundary) |
| aorta     | 0.82–0.84 (vs nnU-Net)       | most volume spread across models (CV ~14%) |
| heart     | 0.79 (vs nnU-Net)            | fuzzy soft-tissue boundary |
| lung      | **0.974** (vs expert SEG)    | nnU-Net OAR models don't include lung |

Within one architecture (nnU-Net 2D/3D-lowres/3D-fullres) Dice is ~0.95–0.97 — i.e. **architecture
choice matters more than configuration.** Tumor (expert GTV): **1.40 mL, ~0.024% of lung volume.** I would guess stage 1 Lung cancer diagnosis for this patient, although tumor size is not the sole factor to consider.

### 2. `therapyNgrowthtracking/tumor_growth_tracking.ipynb` — longitudinal growth/shrinkage tracking

Double-timepoint analysis of patient **PD-1-Lung-00001** in **`anti_pd_1_lung`** immunotherapy collection (two CT timepoints 57 days apart, segmentation restricted to *AIMI lung and nodule AI segmentation* as the only shared segmentation model at both timepoints).

- Downloads all CT timepoints (day 0, 57) + tumor segmentations; reads per-scan acquisition dates/times.
- Measures GTV at each timepoint and computes **interval, % change, specific growth rate, and volume
  doubling time (VDT)**.
- Result: GTV falls **1.86 mL → 0.70 mL (−62.5%) over 57 days** (VDT −40 d) — a volumetric treatment
  response approaching (but below) the RECIST-equivalent partial-response threshold.

### 3. `therapyNgrowthtracking/tumor_growth_tracking_multiSEG.ipynb` - refined MTV estimate using PET/CT + AI segmentation vs Human drawn segmentation comparison

Sister notebook to notebook #2 `tumor_growth_tracking.ipynb`. Same dataset plus PET/CT scans with **metabolic** segmentation (FDG-avid volume) for day 0 and two additional human-drawn segmentations for day 57 CT scan.

- Draw unfiltered MTV using FDG-avid AI segmentation.
- Computed **refined MTV** by excluding physiologic uptake and trying two different SUV (PET intensity unit) thresholding methods (day 0).
- Compared the GTV delineated by the AI segmentation to two radiologist-drawn segmentations (day 57).
- Result
  - Refined MTV computed at **1.74 mL** on day 0 using fixed SUV4.0 threshold (**1.39 mL** using patient-derived adaptive threshold).
<p align="center">
    <img src="therapyNgrowthtracking/day0_mtv_method_comparison.png" width="500" alt="CT GTV vs PET MTV">
</p>
  - Across three (1 AI, 2 radiologists) segmentations, GTV estimated at **$0.744 \pm 0.136$** mL on day 57.

## Repository layout

```
.
├── seg_work/                                  
│   ├── organ_segmentation.ipynb               # Notebook no.1 (hand-edited)
│   ├── nsclc_LUNG1-133_ct_metadata.csv         # CT acquisition parameters
│   ├──organ_overlay_montage.png                # Overlay figure: Organ segmentation in axial CT scan 
│   ├──tumor_gtv.csv                            # Saved dataset metrics
│   └──timepoint_records.csv                    # Saved dataset metrics
├── therapyNgrowthtracking/
│   ├── tumor_growth_tracking.ipynb            # Notebook no.2 (hand-edited)
│   ├── gtv_bar_day0_day57.png                  # A bar graph: GTV reduction after immunotherapy
│   ├── tumor_slices_grid.png                   # Overlay figure: all tumor-bearing CT scans
│   ├── growth_metrics.csv                      # Saved computed metrics
│   └── timepoint_records.csv                   # Raw per-timepoint measurements metadata
│   ├── tumor_growth_tracking_multiSEG.ipynb   # Notebook no.3 (hand-edited)
│   ├── day0_mtv_method_comparison.png          # A bar graph: GTV vs refined MTV (fixed, adaptive threshold)
│   ├── day0_refined_mtv.png                    # Overlay figure: refined MTV (adaptive threshold) with anatomic GTV resampled on PET grid
│   └── day0_fixed_suv4_overlay.png             # Overlay figure: refined MTV (fixed threshold) with anatomic GTV resampled on PET grid
├── imaging-data-commons-1.6.4/         # vendored IDC Claude skill (MIT, A. Fedorov)
├── README.md
├── LICENSE
└── .gitignore
```

\* All notebooks have been edited by hand after being written with Claude and is the canonical version; re-running builder py regenerates the original scaffold, not the hand-edited notebook.

Downloaded DICOM (`data/`, `therapyNgrowthtracking/data/`), the venv, and large `.nii.gz` volumes are
gitignored — they are reproducible by re-running the notebooks.

## Setup

Requires **Python ≥ 3.10** (`idc-index` requirement) and `pydicom < 3` (for `pydicom-seg`); `torch`
needs `numpy < 2`. The repo was developed with a Python 3.12 environment created via
[`uv`](https://github.com/astral-sh/uv):

```bash
uv venv --python 3.12 .idc-venv
VIRTUAL_ENV=.idc-venv uv pip install \
  idc-index SimpleITK nibabel matplotlib "ipywidgets>=8" \
  totalsegmentator pydicom-seg "numpy<2" "pydicom<3" \
  jupyter ipykernel nbformat nbconvert

# register the kernel used by the notebooks
.idc-venv/bin/python -m ipykernel install --user --name idc-venv \
  --display-name "Python 3 (.idc-venv)"
```

(Or `pip install` the same packages into any Python ≥ 3.10 environment.)

## Usage

```bash
.idc-venv/bin/jupyter lab organ_segmentation.ipynb
```

Select the **"Python 3 (.idc-venv)"** kernel and *Run All*. The notebooks download the required series
from IDC automatically (first TotalSegmentator run also downloads model weights). The interactive
sliders/overlays only respond in a live kernel — a statically executed notebook shows the rendered
outputs but not the widgets.

## Tools & data

- **Data:** NCI Imaging Data Commons (v24), collections `nsclc_radiomics` and `anti_pd_1_lung`.
- **Access:** `idc-index` (local DuckDB metadata index + public-bucket downloads, no auth).
- **Organ segmentation:** TotalSegmentator (pretrained nnU-Net).
- **Reference segmentations:** expert GTV/OAR + IDC AI nnU-Net annotations shipped with the data.

## Citations

If you use this work, please cite the data and methods (full list with DOIs is in the final cell of
`organ_segmentation.ipynb`):

- **NSCLC-Radiomics dataset:** Aerts HJWL, et al. *Nature Communications* 5:4006 (2014),
  https://doi.org/10.1038/ncomms5006 — and the TCIA dataset, https://doi.org/10.7937/K9/TCIA.2015.PF0M9REI
- **TCIA:** Clark K, et al. *J Digit Imaging* 26(6):1045–1057 (2013), https://doi.org/10.1007/s10278-013-9622-7
- **Imaging Data Commons:** Fedorov A, et al. *RadioGraphics* 43(12):e230180 (2023), https://doi.org/10.1148/rg.230180
- **TotalSegmentator:** Wasserthal J, et al. *Radiology: AI* 5(5):e230024 (2023), https://doi.org/10.1148/ryai.230024
- **IDC AI annotations:** Krishnaswamy D, et al. Zenodo (2023), https://doi.org/10.5281/zenodo.7473970
- **nnU-Net:** Isensee F, et al. *Nature Methods* 18:203–211 (2021), https://doi.org/10.1038/s41592-020-01008-z

## License

Code: [MIT](LICENSE). Imaging data: governed by IDC/TCIA terms (NSCLC-Radiomics is **CC BY-NC 3.0**,
non-commercial). The vendored `imaging-data-commons-1.6.4/` skill is MIT-licensed by its author.
