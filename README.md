# tiny_cp_runner

Minimal Flask webapp for queuing [Cellpose](https://github.com/MouseLand/cellpose) segmentation jobs. Upload images via a browser, get labeled mask TIFFs back.

## Features

- Drag-and-drop image upload
- Sequential background queue (no Celery, no Redis)
- Cellpose model loaded once at startup — no per-image reload
- Live status table with 2s polling
- One-click mask download

## Requirements

- Python environment with Cellpose installed (conda recommended)
- `flask`, `cellpose`, `tifffile`

## Setup

```bash
conda activate <your_cellpose_env>
pip install -r requirements.txt
python app.py
```

Open `http://<host>:5000` in your browser.

## GPU

Change `gpu=False` → `gpu=True` in `app.py` line ~20 on a CUDA-enabled server.

## Output

Results are saved to `results/<stem>_<id8>_masks.tif` as uint32 labeled arrays (0 = background, 1..N = cells). Compatible with Fiji/ImageJ.

## Notes

- No authentication — intended for trusted local networks only
- The `jobs` dict accumulates for the session lifetime and is not persisted across restarts
- In-flight jobs are acceptable loss on Ctrl-C (worker thread is daemonized)
