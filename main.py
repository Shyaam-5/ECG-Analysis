import os
import tempfile
import shutil
import logging
from typing import List, Dict, Any
from pathlib import Path
from uuid import uuid4

from fastapi import FastAPI, HTTPException, UploadFile, File, status, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import wfdb
import torch
import numpy as np

from model import load_model, preprocess_ecg, predict  # keep your existing helpers


# -------------------------
# Logging
# -------------------------
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("ecg-api")


# -------------------------
# App & CORS
# -------------------------
app = FastAPI(
    title="ECG Classification API",
    description="AI-powered ECG signal classification system",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("CORS_ALLOW_ORIGINS", "*").split(","),
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)


# -------------------------
# Config
# -------------------------
class Config:
    # You can override any of these with environment variables in Azure
    MODEL_PATH = Path(os.getenv("MODEL_PATH", "models/best_model_epoch_69.pth"))
    INPUT_CHANNELS = int(os.getenv("INPUT_CHANNELS", "12"))
    EXPECTED_LENGTH = int(os.getenv("EXPECTED_LENGTH", "5000"))
    MAX_FILE_SIZE = int(os.getenv("MAX_FILE_SIZE_BYTES", str(10 * 1024 * 1024)))  # 10 MB default
    ALLOWED_EXTENSIONS = {".dat", ".hea"}

config = Config()


# -------------------------
# Global state
# -------------------------
model = None  # loaded at startup


# -------------------------
# Schemas
# -------------------------
class PredictionResponse(BaseModel):
    success: bool
    predicted_class: str
    predicted_class_index: int
    confidence: float
    ecg_stats: Dict[str, Any]
    message: str


class ErrorResponse(BaseModel):
    success: bool
    error: str
    detail: str


# -------------------------
# Utilities
# -------------------------
def _get_upload_size(upload: UploadFile) -> int:
    """
    Reliable size measurement for UploadFile (works with SpooledTemporaryFile).
    """
    try:
        pos = upload.file.tell()
    except Exception:
        pos = 0
    try:
        upload.file.seek(0, os.SEEK_END)
        size = upload.file.tell()
    finally:
        try:
            upload.file.seek(pos)
        except Exception:
            pass
    return size


def validate_files(files: List[UploadFile]) -> None:
    """
    Ensure we have matching .dat + .hea for the same record base name,
    that sizes are within limit, and extensions are allowed.
    """
    if not files:
        raise HTTPException(status_code=400, detail="No files uploaded.")

    # Check sizes and extensions
    stems_by_ext = {"dat": set(), "hea": set()}
    for f in files:
        size = _get_upload_size(f)
        if size > config.MAX_FILE_SIZE:
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail=f"File {f.filename} is too large. Max size: {config.MAX_FILE_SIZE // (1024*1024)} MB"
            )
        ext = Path(f.filename).suffix.lower()
        if ext not in config.ALLOWED_EXTENSIONS:
            allowed = ", ".join(sorted(config.ALLOWED_EXTENSIONS))
            raise HTTPException(status_code=400, detail=f"Unsupported file type: {ext}. Allowed: {allowed}")

        stem = Path(f.filename).stem
        if ext == ".dat":
            stems_by_ext["dat"].add(stem)
        elif ext == ".hea":
            stems_by_ext["hea"].add(stem)

    # Require at least one matching pair
    common = stems_by_ext["dat"].intersection(stems_by_ext["hea"])
    if not common:
        raise HTTPException(
            status_code=400,
            detail="You must upload matching .dat and .hea files with the same base name (e.g., record.dat + record.hea)."
        )


def calculate_ecg_stats(ecg_data: np.ndarray) -> Dict[str, Any]:
    return {
        "num_leads": int(ecg_data.shape[0]),
        "signal_length": int(ecg_data.shape[1]),
        "mean_amplitude": float(np.mean(ecg_data)),
        "std_amplitude": float(np.std(ecg_data)),
        "max_amplitude": float(np.max(ecg_data)),
        "min_amplitude": float(np.min(ecg_data)),
    }


def _normalize_ecg(ecg_data: np.ndarray) -> np.ndarray:
    # z-score per lead
    return (ecg_data - ecg_data.mean(axis=1, keepdims=True)) / (ecg_data.std(axis=1, keepdims=True) + 1e-8)


# -------------------------
# Lifespan events
# -------------------------
@app.on_event("startup")
def load_model_on_startup():
    global model
    if not config.MODEL_PATH.exists():
        logger.error(f"Model file not found at {config.MODEL_PATH.resolve()}")
        raise RuntimeError(f"Model file not found at {config.MODEL_PATH}")
    try:
        # Ensure CPU-safe model loading on Azure App Service (no GPU)
        logger.info(f"Loading model from {config.MODEL_PATH} on CPU...")
        model = load_model(
            model_path=str(config.MODEL_PATH),
            input_channels=config.INPUT_CHANNELS
        )
        # If your load_model uses torch.load internally, make sure it sets map_location="cpu"
        logger.info("Model loaded successfully.")
    except Exception as e:
        logger.exception("Failed to load model.")
        raise RuntimeError(f"Failed to load model: {e}") from e


# -------------------------
# Routes
# -------------------------
@app.get("/", response_model=Dict[str, str])
def root():
    return {"message": "ECG Classification API is running", "status": "healthy", "version": "1.0.0"}


@app.get("/healthz", response_model=Dict[str, str])
def healthz():
    # simple health endpoint for Azure probes
    return {"status": "ok"}


@app.post("/predict_file", response_model=PredictionResponse)
def predict_ecg_file(files: List[UploadFile] = File(...)):
    """
    Upload ECG files (.dat + .hea) and get classification prediction.
    Multiple pairs are allowed; the first valid pair will be used.
    """
    validate_files(files)

    # Group files by stem so we can find a valid pair
    by_stem: Dict[str, Dict[str, UploadFile]] = {}
    for uf in files:
        ext = Path(uf.filename).suffix.lower()
        stem = Path(uf.filename).stem
        by_stem.setdefault(stem, {})
        by_stem[stem][ext] = uf

    # Find first stem that has both .dat and .hea
    target_stem = None
    for stem, group in by_stem.items():
        if ".dat" in group and ".hea" in group:
            target_stem = stem
            break

    if target_stem is None:
        raise HTTPException(
            status_code=400,
            detail="No matching .dat + .hea pair found in the upload."
        )

    # Work inside a unique temp directory for this request
    with tempfile.TemporaryDirectory(prefix=f"ecg_{uuid4().hex}_") as tmpdir:
        tmpdir_path = Path(tmpdir)
        logger.info(f"Using temp dir: {tmpdir_path}")

        # Save the pair to disk with consistent names: <stem>.dat, <stem>.hea
        dat_dest = tmpdir_path / f"{target_stem}.dat"
        hea_dest = tmpdir_path / f"{target_stem}.hea"

        for src, dest in [(by_stem[target_stem][".dat"], dat_dest),
                          (by_stem[target_stem][".hea"], hea_dest)]:
            with open(dest, "wb") as f:
                shutil.copyfileobj(src.file, f)
            logger.info(f"Saved file: {dest.name} ({dest.stat().st_size} bytes)")

        record_path = str(tmpdir_path / target_stem)

        # Read ECG with wfdb
        try:
            record = wfdb.rdrecord(record_path)
            # wfdb returns signals as (n_samples, n_channels) or (n_channels, n_samples) depending on accessor
            # p_signal is (n_samples, n_channels); we transpose to (channels, length)
            ecg_data = record.p_signal.T.astype(np.float32)  # (12, length) expected
            logger.info(f"ECG data shape read from wfdb: {ecg_data.shape}")
        except Exception as e:
            logger.exception("Failed to read ECG files with wfdb.")
            raise HTTPException(status_code=400, detail=f"Failed to read ECG files: {e}")

        # Optional: length check / pad / truncate (if your model expects fixed length)
        # Here we just log and proceed.
        if ecg_data.shape[0] != config.INPUT_CHANNELS:
            raise HTTPException(
                status_code=400,
                detail=f"Expected {config.INPUT_CHANNELS} leads, got {ecg_data.shape[0]}"
            )

        # Normalize and tensorize
        ecg_normalized = _normalize_ecg(ecg_data)            # (C, L)
        ecg_tensor = torch.tensor(ecg_normalized, dtype=torch.float32).unsqueeze(0)  # (1, C, L)

        # Predict
        try:
            if model is None:
                # Should never happen if startup succeeded, but guard anyway
                raise RuntimeError("Model not loaded.")
            result = predict(model, ecg_tensor)  # your existing helper should return dict
        except Exception as e:
            logger.exception("Model inference failed.")
            raise HTTPException(status_code=500, detail=f"Inference error: {e}")

        stats = calculate_ecg_stats(ecg_data)

        # Prepare response
        response = PredictionResponse(
            success=True,
            predicted_class=str(result.get("predicted_label", "")),
            predicted_class_index=int(result.get("predicted_class_index", -1)),
            confidence=float(result.get("confidence", 0.0)),
            ecg_stats=stats,
            message="ECG classification completed successfully"
        )
        logger.info(f"Prediction done: {response.predicted_class} (conf: {response.confidence:.4f})")
        return response


# -------------------------
# Error handlers
# -------------------------
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    # Preserve HTTP status codes for raised HTTPException
    return JSONResponse(
        status_code=exc.status_code,
        content=ErrorResponse(success=False, error="HTTPException", detail=str(exc.detail)).dict()
    )


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.exception("Unhandled exception")
    return JSONResponse(
        status_code=500,
        content=ErrorResponse(success=False, error="Internal server error", detail="An unexpected error occurred").dict()
    )


# -------------------------
# Local dev (ignored in Azure)
# -------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=int(os.getenv("PORT", "8000")), reload=True, log_level="info")
