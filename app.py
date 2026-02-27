import uuid
from pathlib import Path

from flask import Flask, jsonify, render_template, request, send_file
from pydantic import ValidationError

from config import UPLOAD_DIR
from schema import SegmentationSettings
from worker import jobs, jobs_lock, work_q

app = Flask(__name__)


@app.get("/")
def index():
    return render_template("index.html")


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

    try:
        s = SegmentationSettings.model_validate(request.form.to_dict())
    except ValidationError as exc:
        return jsonify({"error": exc.errors(include_url=False)}), 400

    settings = {
        "diameter":           s.diameter,
        "channels":           [s.channel_cyto, s.channel_nuc],
        "flow_threshold":     s.flow_threshold,
        "cellprob_threshold": s.cellprob_threshold,
        "min_size":           s.min_size,
    }

    with jobs_lock:
        jobs[job_id] = {
            "name": f.filename,
            "stem": stem,
            "upload_path": str(upload_path),
            "status": "queued",
            "result": None,
            "error": None,
            "settings": settings,
        }
    work_q.put(job_id)
    return jsonify({"job_id": job_id})


@app.get("/status")
def status():
    with jobs_lock:
        items = list(jobs.items())
    items.reverse()  # newest first
    return jsonify([
        [jid, {k: v for k, v in job.items() if k not in ("upload_path", "stem", "settings")}]
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
