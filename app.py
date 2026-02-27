import os
import queue
import threading
import uuid
from pathlib import Path

import tifffile
from cellpose import models
from flask import Flask, jsonify, request, send_file

# ---------------------------------------------------------------------------
# Directories
# ---------------------------------------------------------------------------
UPLOAD_DIR = Path("uploads")
RESULT_DIR = Path("results")
UPLOAD_DIR.mkdir(exist_ok=True)
RESULT_DIR.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# Cellpose model — loaded once at startup
# ---------------------------------------------------------------------------
MODEL = models.CellposeModel(gpu=False, pretrained_model="cpsam")

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
            job["status"] = "processing"

        try:
            img = tifffile.imread(upload_path)
            masks, _, _ = MODEL.eval(img, diameter=None, channels=None)
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

# ---------------------------------------------------------------------------
# Flask app
# ---------------------------------------------------------------------------
app = Flask(__name__)

# ---------------------------------------------------------------------------
# Inline HTML UI
# ---------------------------------------------------------------------------
HTML = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Cellpose Queue</title>
<style>
  *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
  body { font-family: system-ui, sans-serif; background: #0f0f11; color: #e2e2e5; min-height: 100vh; padding: 2rem; }
  h1 { font-size: 1.4rem; margin-bottom: 1.5rem; letter-spacing: 0.04em; color: #a78bfa; }
  #drop { border: 2px dashed #444; border-radius: 10px; padding: 2.5rem 1rem; text-align: center;
          cursor: pointer; transition: border-color .2s, background .2s; margin-bottom: 1.5rem; }
  #drop.over { border-color: #a78bfa; background: #1e1b2e; }
  #drop p { color: #888; font-size: .95rem; }
  #drop p span { color: #a78bfa; text-decoration: underline; }
  input[type=file] { display: none; }
  table { width: 100%; border-collapse: collapse; font-size: .9rem; }
  th { text-align: left; padding: .5rem .75rem; color: #888; border-bottom: 1px solid #2a2a2e; font-weight: 500; }
  td { padding: .5rem .75rem; border-bottom: 1px solid #1e1e22; }
  .queued    { color: #facc15; }
  .processing{ color: #60a5fa; }
  .done      { color: #4ade80; }
  .error     { color: #f87171; }
  a.dl { color: #a78bfa; text-decoration: none; }
  a.dl:hover { text-decoration: underline; }
  #msg { height: 1.2rem; font-size: .85rem; color: #888; margin-bottom: .75rem; }
</style>
</head>
<body>
<h1>Cellpose Queue Runner</h1>
<div id="drop">
  <p>Drag &amp; drop image(s) here, or <span>click to browse</span></p>
  <input type="file" id="fileinput" accept="image/*,.tif,.tiff" multiple>
</div>
<div id="msg"></div>
<table>
  <thead><tr><th>File</th><th>Job ID</th><th>Status</th><th>Action</th></tr></thead>
  <tbody id="tbody"></tbody>
</table>
<script>
const drop = document.getElementById('drop');
const fileInput = document.getElementById('fileinput');
const msg = document.getElementById('msg');

drop.addEventListener('click', () => fileInput.click());
drop.addEventListener('dragover', e => { e.preventDefault(); drop.classList.add('over'); });
drop.addEventListener('dragleave', () => drop.classList.remove('over'));
drop.addEventListener('drop', e => { e.preventDefault(); drop.classList.remove('over'); uploadFiles(e.dataTransfer.files); });
fileInput.addEventListener('change', () => uploadFiles(fileInput.files));

async function uploadFiles(files) {
  for (const file of files) {
    const fd = new FormData();
    fd.append('file', file);
    try {
      const r = await fetch('/upload', { method: 'POST', body: fd });
      const j = await r.json();
      if (!r.ok) { msg.textContent = 'Upload error: ' + (j.error || r.status); continue; }
      msg.textContent = 'Queued: ' + file.name;
    } catch (e) { msg.textContent = 'Upload failed: ' + e; }
  }
  fileInput.value = '';
}

async function poll() {
  try {
    const r = await fetch('/status');
    const items = await r.json();
    const tbody = document.getElementById('tbody');
    tbody.innerHTML = '';
    for (const [id, job] of items) {
      const action = job.status === 'done'
        ? `<a class="dl" href="/download/${id}" download>Download masks</a>`
        : job.status === 'error' ? (job.error || 'error') : '—';
      const row = `<tr>
        <td>${esc(job.name)}</td>
        <td style="font-family:monospace;font-size:.8rem">${id.slice(0,8)}</td>
        <td class="${esc(job.status)}">${esc(job.status)}</td>
        <td>${action}</td>
      </tr>`;
      tbody.insertAdjacentHTML('beforeend', row);
    }
  } catch (_) {}
  setTimeout(poll, 2000);
}

function esc(s) {
  return String(s).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
}

poll();
</script>
</body>
</html>
"""


@app.get("/")
def index():
    return HTML, 200, {"Content-Type": "text/html; charset=utf-8"}


@app.post("/upload")
def upload():
    f = request.files.get("file")
    if not f or not f.filename:
        return jsonify({"error": "no file"}), 400

    job_id = str(uuid.uuid4())
    stem = Path(f.filename).stem
    suffix = Path(f.filename).suffix or ".tif"
    save_name = f"{stem}_{job_id[:8]}{suffix}"
    upload_path = UPLOAD_DIR / save_name
    f.save(str(upload_path))

    with jobs_lock:
        jobs[job_id] = {
            "name": f.filename,
            "stem": stem,
            "upload_path": str(upload_path),
            "status": "queued",
            "result": None,
            "error": None,
        }
    work_q.put(job_id)
    return jsonify({"job_id": job_id})


@app.get("/status")
def status():
    with jobs_lock:
        items = list(jobs.items())
    items.reverse()  # newest first
    # Strip internal fields before sending
    return jsonify([
        [jid, {k: v for k, v in job.items() if k not in ("upload_path", "stem")}]
        for jid, job in items
    ])


@app.get("/download/<job_id>")
def download(job_id):
    with jobs_lock:
        job = jobs.get(job_id)
    if not job:
        return jsonify({"error": "unknown job"}), 404
    if job["status"] != "done":
        return jsonify({"error": "not ready"}), 409
    result_path = job["result"]
    if not result_path or not Path(result_path).exists():
        return jsonify({"error": "result file missing"}), 404
    return send_file(result_path, as_attachment=True,
                     download_name=Path(result_path).name)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
