import queue
import threading
from pathlib import Path

import tifffile
from cellpose import models, io

from config import RESULT_DIR

# ---------------------------------------------------------------------------
# Cellpose model — loaded once at startup
# ---------------------------------------------------------------------------
MODEL = models.CellposeModel(gpu=True, pretrained_model="cpsam")

# ---------------------------------------------------------------------------
# Job state
# ---------------------------------------------------------------------------
jobs: dict[str, dict] = {}
jobs_lock = threading.Lock()
work_q: queue.Queue = queue.Queue()

# ---------------------------------------------------------------------------
# Background worker
# ---------------------------------------------------------------------------

def worker():
    while True:
        job_id = work_q.get()
        with jobs_lock:
            job = jobs[job_id]
            upload_path = Path(job["upload_path"])
            stem = job["stem"]
            settings = job["settings"]
            job["status"] = "processing"

        try:
            zaxis, caxis = None, None
            if settings["do_3d"]:
                img = io.imread_3D(upload_path)
                zaxis = 0
                caxis = 3
            else: 
                img = io.imread_2D(upload_path)
            masks, _, _ = MODEL.eval(
                img,
                diameter=settings["diameter"],
                channels=settings["channels"],
                flow_threshold=settings["flow_threshold"],
                cellprob_threshold=settings["cellprob_threshold"],
                min_size=settings["min_size"],
                do_3D=settings["do_3d"],
                anisotropy=settings["anisotropy"],
                channel_axis=caxis,
                z_axis=zaxis 
            )
            result_name = f"{stem}_{job_id[:8]}_masks.tif"
            result_path = RESULT_DIR / result_name
            tifffile.imwrite(str(result_path), masks.astype("uint32"))
            upload_path.unlink(missing_ok=True)
            with jobs_lock:
                jobs[job_id]["status"] = "done"
                jobs[job_id]["result"] = str(result_path)
        except Exception as exc:
            with jobs_lock:
                jobs[job_id]["status"] = "error"
                jobs[job_id]["error"] = str(exc)
        finally:
            work_q.task_done()


_worker_thread = threading.Thread(target=worker, daemon=True)
_worker_thread.start()
