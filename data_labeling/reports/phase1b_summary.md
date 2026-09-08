# Phase 1B Summary

## Security cleanup
- `external/EgoLoc/auth.env` existed? yes, previously detected
- handled? yes, it was moved out of `external/EgoLoc/`; no secrets were printed
- `.gitignore` includes env/secrets patterns? yes

## Data acquired
- USST EgoPAT3D-DT archive present? True
- USST EgoPAT3D-DT extracted? True
- DeskTIL archive/package present? True
- DeskTIL extracted/sorted? True
- EgoLoc EgoPAT3D sample present? True
- HF EgoPAT3D raw snapshot present? False

## Counts
- EgoPAT3D-DT videos: 11429
- EgoPAT3D-DT trajectory files: 11141
- EgoPAT3D-DT odometry files: 11141
- EgoPAT3D-DT fully paired video+trajectory+odometry items: 11141
- EgoPAT3D-DT videos missing trajectory/odometry: 288
- DeskTIL videos: 130
- DeskTIL annotation files: 1 (`label.xlsx`)
- DeskTIL label workbook rows: 133
- Total videos in inventory: 11990
- Total readable videos in inventory: 11776
- Unreadable videos: 214, all under optional `data/EgoLoc_samples/*/__MACOSX/` metadata folders

## Paths
- EgoPAT3D-DT videos: `data/EgoPAT3D/EgoPAT3D-postproc/video_clips_hand/`
- EgoPAT3D-DT trajectories: `data/EgoPAT3D/EgoPAT3D-postproc/trajectory_repair/`
- EgoPAT3D-DT odometry: `data/EgoPAT3D/EgoPAT3D-postproc/odometry/`
- DeskTIL videos: `data/DeskTIL/videos/`
- DeskTIL annotations: `data/DeskTIL/annotations/label.xlsx`

## Remaining blockers / cautions
- The 288 EgoPAT3D-DT videos without matching trajectory/odometry should be excluded from Phase 2 unless intentionally using video-only examples.
- The 214 unreadable videos are macOS resource-fork files under `__MACOSX`; they are safe to ignore/delete and should not be included in datasets.
- DeskTIL annotations are Excel (`.xlsx`), so downstream scripts should read Excel or convert it to CSV/JSON during Phase 2.

## Phase 2 readiness
READY_FOR_PHASE_2

Reason: EgoPAT3D-DT has 11,141 fully paired video/trajectory/odometry items, and DeskTIL has 130 videos plus `label.xlsx` annotations.

## Cleanup candidates
- `data/EgoLoc_samples/DeskTIL/__MACOSX/`: macOS resource-fork metadata from zip extraction; contains unreadable 176-byte ._*.mp4 placeholder files, not real videos.
- `data/EgoLoc_samples/EgoPAT3D/__MACOSX/`: same macOS zip metadata; not useful for labeling or training.
- `data/raw_downloads/usst/EgoPAT3D-postproc.tar.gz`: original archive; extracted EgoPAT3D-DT folders are present and verified. Delete only if you do not need local re-extraction/provenance backup.
- `data/raw_downloads/desktil/DeskTIL.zip`: original DeskTIL archive; sorted videos and label.xlsx are present. Delete only after you are comfortable losing local re-extraction copy.
- `data/raw_downloads/egoloc/EgoPAT3D.zip`: original EgoLoc EgoPAT3D sample archive; optional sample package is extracted. Not needed for core EgoPAT3D-DT Phase 2.
- `data/EgoLoc_samples/DeskTIL/`: raw extracted copy of DeskTIL package. Once data/DeskTIL/videos and data/DeskTIL/annotations/label.xlsx are verified, Phase 2 can use the sorted data instead. Keep if you want provenance.
- `data/EgoLoc_samples/EgoPAT3D/`: optional EgoLoc sample package, not the USST EgoPAT3D-DT training target. Keep only for dry-runs/comparison.
- `data/EgoPAT3D/raw/`: empty folder; the HF raw snapshot was not downloaded and is not needed for Phase 2 if using EgoPAT3D-DT/DeskTIL.
