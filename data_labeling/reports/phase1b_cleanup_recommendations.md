# Phase 1B Cleanup Recommendations

- `data/EgoLoc_samples/DeskTIL/__MACOSX/`: macOS resource-fork metadata from zip extraction; contains unreadable 176-byte ._*.mp4 placeholder files, not real videos.
- `data/EgoLoc_samples/EgoPAT3D/__MACOSX/`: same macOS zip metadata; not useful for labeling or training.
- `data/raw_downloads/usst/EgoPAT3D-postproc.tar.gz`: original archive; extracted EgoPAT3D-DT folders are present and verified. Delete only if you do not need local re-extraction/provenance backup.
- `data/raw_downloads/desktil/DeskTIL.zip`: original DeskTIL archive; sorted videos and label.xlsx are present. Delete only after you are comfortable losing local re-extraction copy.
- `data/raw_downloads/egoloc/EgoPAT3D.zip`: original EgoLoc EgoPAT3D sample archive; optional sample package is extracted. Not needed for core EgoPAT3D-DT Phase 2.
- `data/EgoLoc_samples/DeskTIL/`: raw extracted copy of DeskTIL package. Once data/DeskTIL/videos and data/DeskTIL/annotations/label.xlsx are verified, Phase 2 can use the sorted data instead. Keep if you want provenance.
- `data/EgoLoc_samples/EgoPAT3D/`: optional EgoLoc sample package, not the USST EgoPAT3D-DT training target. Keep only for dry-runs/comparison.
- `data/EgoPAT3D/raw/`: empty folder; the HF raw snapshot was not downloaded and is not needed for Phase 2 if using EgoPAT3D-DT/DeskTIL.
