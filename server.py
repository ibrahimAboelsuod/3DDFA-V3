"""3DDFA-V3 HTTP API. POST /process with an image, get landmarks + mesh back."""
import io
import uuid
import zipfile
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse

from face_box import face_box
from model.recon import face_model
from util.io import visualize

OUTPUT_ROOT = Path("/tmp/3ddfa-results")
OUTPUT_ROOT.mkdir(exist_ok=True)

class _Args:
    device = "cuda"
    backbone = "resnet50"
    iscrop = True
    detector = "retinaface"
    ldm68 = True
    ldm106 = True
    ldm106_2d = True
    ldm134 = True
    seg = True
    seg_visible = True
    useTex = True
    extractTex = True

print("Loading 3DDFA-V3 (10-15s)...")
_args = _Args()
_facebox = face_box(_args)
_model = face_model(_args)
print("Model ready.")

app = FastAPI(title="3DDFA-V3")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


_LANDMARK_SETS = {
    "68":      ["ldm68"],
    "106+134": ["ldm106", "ldm134"],
    "106_2d":  ["ldm106_2d"],
}

@app.post("/process")
async def process(
    file: UploadFile = File(...),
    landmark_type: str = Form("106+134"),
):
    keys = _LANDMARK_SETS.get(landmark_type)
    if keys is None:
        return JSONResponse(
            {"error": f"Unknown landmark_type '{landmark_type}'. Choose: {list(_LANDMARK_SETS)}"},
            status_code=400,
        )

    contents = await file.read()
    im = Image.open(io.BytesIO(contents)).convert("RGB")

    detections = _facebox.detect_all(im)
    if not detections:
        return JSONResponse({"error": "no face detected"}, status_code=422)

    job_id = str(uuid.uuid4())[:8]
    job_dir = OUTPUT_ROOT / job_id
    job_dir.mkdir(parents=True)

    image_bgr = cv2.cvtColor(np.asarray(im), cv2.COLOR_RGB2BGR)

    faces = []
    for i, (trans_params, im_tensor, rfm_lmks) in enumerate(detections):
        _model.input_img = im_tensor.to(_args.device)
        results = _model.forward()

        face_dir = job_dir / f"face_{i}"
        face_dir.mkdir(parents=True)

        viz = visualize(results, _args)
        viz.visualize_and_output(trans_params, image_bgr, str(face_dir), "result")

        landmarks = {}
        for key in keys:
            arr = results.get(key)
            if arr is not None:
                landmarks[key] = arr[0].astype(float).tolist() if arr.ndim == 3 else arr.astype(float).tolist()

        # retinaface 106 landmarks from the detector (direct regression, not 3DMM)
        detector_ldm106 = rfm_lmks.tolist() if rfm_lmks is not None else None

        verts = results.get("face_shape")
        vertices = verts[0].astype(float).tolist() if verts is not None and verts.ndim == 3 else None

        faces.append({
            "index": i,
            "landmarks": landmarks,
            "detector_ldm106": detector_ldm106,
            "mesh_vertices": vertices,
            "mesh_vertex_count": len(vertices) if vertices else 0,
            "files": {
                "visualization": f"/result/{job_id}/face_{i}/visualization",
            },
        })

    return JSONResponse({
        "job_id": job_id,
        "face_count": len(faces),
        "faces": faces,
    })


@app.get("/result/{job_id}/visualization")
async def get_vis(job_id: str):
    p = OUTPUT_ROOT / job_id / "result.png"
    return FileResponse(p, media_type="image/png") if p.exists() else JSONResponse({"error": "not found"}, 404)


@app.get("/result/{job_id}/face_{face_idx}/visualization")
async def get_face_vis(job_id: str, face_idx: int):
    p = OUTPUT_ROOT / job_id / f"face_{face_idx}" / "result.png"
    return FileResponse(p, media_type="image/png") if p.exists() else JSONResponse({"error": "not found"}, 404)


@app.get("/result/{job_id}/{kind}")
async def get_file(job_id: str, kind: str):
    mapping = {
        "obj_pca": "result_pcaTex.obj",
        "obj_extract": "result_extractTex.obj",
        "npy": "result.npy",
    }
    p = OUTPUT_ROOT / job_id / mapping.get(kind, "")
    return FileResponse(p, media_type="application/octet-stream") if p.exists() else JSONResponse({"error": "not found"}, 404)


@app.get("/result/{job_id}/face_{face_idx}/{kind}")
async def get_face_file(job_id: str, face_idx: int, kind: str):
    mapping = {
        "obj_pca": "result_pcaTex.obj",
        "obj_extract": "result_extractTex.obj",
        "npy": "result.npy",
    }
    p = OUTPUT_ROOT / job_id / f"face_{face_idx}" / mapping.get(kind, "")
    return FileResponse(p, media_type="application/octet-stream") if p.exists() else JSONResponse({"error": "not found"}, 404)


@app.get("/result/{job_id}/zip")
async def get_zip(job_id: str):
    d = OUTPUT_ROOT / job_id
    if not d.exists():
        return JSONResponse({"error": "not found"}, 404)
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for f in d.iterdir():
            zf.write(f, f.name)
    buf.seek(0)
    return StreamingResponse(buf, media_type="application/zip",
                             headers={"Content-Disposition": f"attachment; filename={job_id}.zip"})


@app.get("/health")
async def health():
    return {"status": "ok"}
