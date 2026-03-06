import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.resolve()))

import base64
import time

import cv2
import numpy as np
import torch
from fastapi import FastAPI, File, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from pipeline.config import get_config
from pipeline.circa_engine import CIRCAEngine

app = FastAPI(title="CIRCA Dashboard")

static_path = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=str(static_path)), name="static")

engine = None

@app.on_event("startup")
async def startup_event():
    global engine
    config = get_config()
    engine = CIRCAEngine(config)
    
    dummy_tensor = torch.zeros((1, 3, 256, 256)).to(engine.device)
    with torch.no_grad():
        engine.cnn_a(dummy_tensor)
        engine.cnn_b(dummy_tensor)

@app.get("/")
async def read_root():
    index_path = static_path / "index.html"
    if not index_path.exists():
        return HTMLResponse("<h1>Index HTML not found</h1>", status_code=404)
    with open(index_path, "r", encoding="utf-8") as f:
        return HTMLResponse(content=f.read())

@app.get("/health")
async def health_check():
    return {"status": "ok", "model": "loaded" if engine else "not loaded"}

@app.post("/analyze")
async def analyze_image(file: UploadFile = File(...)):
    start_t = time.perf_counter()
    
    try:
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is None:
            return JSONResponse({"error": "Invalid image format"}, status_code=400)
            
        config = get_config()
        target_res = config.stream.resolution
        resized = cv2.resize(img, target_res)
        rgb_frame = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        
        frame_tensor = torch.from_numpy(rgb_frame).float().permute(2, 0, 1).unsqueeze(0)
        frame_tensor = (frame_tensor / 255.0).to(engine.device)
        
        with torch.no_grad():
            out_a = engine.cnn_a(frame_tensor)
            out_b = engine.cnn_b(frame_tensor)
            
        report = engine._fast_loop(1, rgb_frame, frame_tensor, out_a, out_b)
        
        heatmap_b64 = ""
        if report.heatmap is not None:
            overlay = engine.gradcam.overlay(report.heatmap, rgb_frame)
            overlay_bgr = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)
            _, buffer = cv2.imencode('.jpg', overlay_bgr)
            heatmap_b64 = base64.b64encode(buffer).decode('utf-8')
            
        causes_out = []
        if report.top_causes:
            for c in report.top_causes:
                causes_out.append({
                    "name": c.node_name,
                    "probability": c.probability,
                    "primary": c.is_primary
                })
                
        latency_ms = (time.perf_counter() - start_t) * 1000.0
        
        return {
            "anomaly": report.is_anomaly,
            "confidence": float(report.anomaly_score),
            "causes": causes_out,
            "heatmap_b64": heatmap_b64,
            "dag_age_frames": report.dag_snapshot_age,
            "latency_ms": latency_ms,
            "operator_action": report.operator_action
        }
    except Exception as e:
        import traceback
        traceback.print_exc()
        return JSONResponse({"error": f"Internal Analysis Error: {str(e)}"}, status_code=500)

@app.websocket("/stream")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            await websocket.receive_text()
            await websocket.send_json({"status": "streaming not fully implemented"})
    except WebSocketDisconnect:
        pass

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=False)
