# 📦 Archive Directory (`archive/`)

This directory houses historical prototypes, legacy artifacts, and standalone bundles preserved for auditability and reference. 

> [!NOTE]
> All active, production-grade applications and engines reside in the repository root. Do not run or import files from this directory for ongoing development.

---

## 🗂️ Archived Inventory & Active Counterparts

| Archived File / Artifact | Description & Archival Rationale | Active Root Counterpart |
| :--- | :--- | :--- |
| **`app.py`** | Original single-rail Streamlit dashboard prototype prior to PaySim dual-rail ledger discrepancy integration. | [`app.py`](file:///c:/Users/HP/Desktop/PDS/app.py) (Root) |
| **`transaction_generator.py`** | Legacy random transaction generator before PCA-fidelity calibration and multivariate Gaussian sampling. | [`data_generator.py`](file:///c:/Users/HP/Desktop/PDS/data_generator.py) (Root) |
| **`README_LEGACY_CAPSTONE.md`** | Earlier iteration of documentation focusing exclusively on the 7 credit card models before microservice deployment. | [`README.md`](file:///c:/Users/HP/Desktop/PDS/README.md) (Root) |
| **`fraud-intelligence-platform.zip`** | Standalone 186 MB archive snapshot of the repository (untracked in Git to adhere to GitHub file size limits). | N/A (Repository source tree) |
| **`models/`** | Legacy pre-trained model checkpoint storage used in early iterations. | [`data/model_weights.npz`](file:///c:/Users/HP/Desktop/PDS/data/model_weights.npz) & [`data/paysim_weights.npz`](file:///c:/Users/HP/Desktop/PDS/data/paysim_weights.npz) |

---

## 🔒 Git Tracking Policy
All zip archives and heavy compressed files inside `archive/` are ignored by [`.gitignore`](file:///c:/Users/HP/Desktop/PDS/.gitignore) to protect GitHub repository quota and prevent repository bloat.
