# Download Blockers

## USST EgoPAT3D-DT download blocker

Source:
https://github.com/oppo-us-research/USST

Official link found in `external/USST/README.md`:
https://1drv.ms/f/c/0adef1d3209dfb47/Qkf7nSDT8d4ggAoxFQAAAAAAHkE-BxpcdN_G8g

Expected:
EgoPAT3D-postproc.tar.gz

Problem:
Tried a non-interactive HTTP GET probe with Python `urllib.request` and a browser user agent. The request timed out from this server before returning a downloadable archive or redirect target.

Resolution needed from human:
Manually download `EgoPAT3D-postproc.tar.gz` from the USST README OneDrive link and place it at:
`data/raw_downloads/usst/EgoPAT3D-postproc.tar.gz`

Then run:
`tar -zxvf data/raw_downloads/usst/EgoPAT3D-postproc.tar.gz -C data/EgoPAT3D/`

## EgoPAT3D Hugging Face download blocker

Source:
https://huggingface.co/datasets/ai4ce/EgoPAT3Dv1

Expected:
Official EgoPAT3D public dataset snapshot under `data/EgoPAT3D/raw/EgoPAT3Dv1/`

Problem:
Tried to inspect the official dataset non-interactively using `huggingface_hub.HfApi().list_repo_tree(repo_type="dataset", recursive=True)`. The request timed out while connecting to `huggingface.co`, so no reliable file list or size estimate was available from this server. The full snapshot was not downloaded.

Resolution needed from human:
After confirming network access and disk capacity, run:
`FULL_EGOPAT3D_HF_DOWNLOAD=1 bash scripts/download_sources.sh`

## EgoLoc EgoPAT3D sample download blocker

Source:
https://github.com/IRMVLab/EgoLoc

Official link found in `external/EgoLoc/README.md`:
https://pan.sjtu.edu.cn/web/share/a3b27a7d80cd6826aeaeed8cc71b5da8

Expected:
EgoPAT3D sample videos and manual annotations

Problem:
Tried a non-interactive HTTP GET probe. The server returned status `200` with `content-type: text/html` and an SJTU Pan browser application page (`<title>交大云盘</title>`), not a direct archive. This requires browser JavaScript/manual cloud UI interaction.

Resolution needed from human:
Manually download the EgoPAT3D sample package from the EgoLoc README and place the archive at:
`data/raw_downloads/egoloc/`

Then extract into:
`data/EgoLoc_samples/EgoPAT3D/`

## EgoLoc DeskTIL download blocker

Source:
https://github.com/IRMVLab/EgoLoc

Official link found in `external/EgoLoc/README.md`:
https://pan.sjtu.edu.cn/web/share/9c552393f367cfb9290c8ac890ed0c59

Expected:
DeskTIL sample videos and manual annotations

Problem:
Tried a non-interactive HTTP GET probe. The server returned status `200` with `content-type: text/html` and an SJTU Pan browser application page (`<title>交大云盘</title>`), not a direct archive. This requires browser JavaScript/manual cloud UI interaction.

Resolution needed from human:
Manually download the DeskTIL package from the EgoLoc README and place the archive at:
`data/raw_downloads/desktil/`

Then extract into:
`data/EgoLoc_samples/DeskTIL/`

Finally sort videos into:
`data/DeskTIL/videos/`

and annotations into:
`data/DeskTIL/annotations/`
