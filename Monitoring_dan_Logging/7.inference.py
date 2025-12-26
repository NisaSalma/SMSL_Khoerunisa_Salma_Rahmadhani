from fastapi import FastAPI
from fastapi.responses import Response
from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST
import time
import psutil
from pydantic import BaseModel

app = FastAPI()

# ===== METRICS =====
REQUEST_COUNT = Counter("request_total", "Total request ke model")
REQUEST_LATENCY = Histogram("request_latency_seconds", "Waktu inferensi")
CPU_USAGE = Gauge("cpu_usage_percent", "CPU usage")
MEMORY_USAGE = Gauge("memory_usage_percent", "Memory usage")
ERROR_COUNT = Counter("error_total", "Total error")

# ===== PREDICT ENDPOINT =====
class PredictRequest(BaseModel):
    x: float

@app.post("/predict")
def predict(req: PredictRequest):
    REQUEST_COUNT.inc()
    start = time.time()
    try:
        y = req.x * 2
        time.sleep(0.3)
        latency = time.time() - start
        REQUEST_LATENCY.observe(latency)
        CPU_USAGE.set(psutil.cpu_percent())
        MEMORY_USAGE.set(psutil.virtual_memory().percent)
        return {"result": y}
    except Exception:
        ERROR_COUNT.inc()
        return {"error": "prediction failed"}

# ===== METRICS ENDPOINT =====
@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
