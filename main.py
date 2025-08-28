import os
import tempfile
import shutil
import logging
from typing import List, Dict, Any
from pathlib import Path

from fastapi import FastAPI, HTTPException, UploadFile, File, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, validator
import uvicorn
import wfdb
import torch
import numpy as np

from model import load_model, preprocess_ecg, predict

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app with metadata
app = FastAPI(
    title="ECG Classification API",
    description="AI-powered ECG signal classification system",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# Configuration
class Config:
    MODEL_PATH = Path("models/best_model_epoch_69.pth")
    EXPECTED_LENGTH = 5000
    ALLOWED_EXTENSIONS = {".dat", ".hea"}
    MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB

config = Config()

# Load model at startup
try:
    model = load_model(model_path=str(config.MODEL_PATH), input_channels=12)
    logger.info("Model loaded successfully")
except Exception as e:
    logger.error(f"Failed to load model: {e}")
    raise

# Response models
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

# Utility functions
def validate_files(files: List[UploadFile]) -> None:
    """Validate uploaded files"""
    if not files:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No files uploaded"
        )
    
    extensions = {Path(f.filename).suffix.lower() for f in files}
    if not config.ALLOWED_EXTENSIONS.issubset(extensions | {ext for ext in extensions}):
        if not any(ext in config.ALLOWED_EXTENSIONS for ext in extensions):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Please upload files with extensions: {', '.join(config.ALLOWED_EXTENSIONS)}"
            )
    
    for file in files:
        if file.size and file.size > config.MAX_FILE_SIZE:
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail=f"File {file.filename} is too large. Max size: {config.MAX_FILE_SIZE // (1024*1024)}MB"
            )

def calculate_ecg_stats(ecg_data: np.ndarray) -> Dict[str, Any]:
    """Calculate ECG signal statistics"""
    return {
        "num_leads": ecg_data.shape[0],
        "signal_length": ecg_data.shape[1],
        "mean_amplitude": float(np.mean(ecg_data)),
        "std_amplitude": float(np.std(ecg_data)),
        "max_amplitude": float(np.max(ecg_data)),
        "min_amplitude": float(np.min(ecg_data))
    }

# API Routes
@app.get("/", response_model=Dict[str, str])
async def root():
    """Health check endpoint"""
    return {
        "message": "ECG Classification API is running",
        "status": "healthy",
        "version": "1.0.0"
    }

@app.post("/predict_file", response_model=PredictionResponse)
async def predict_ecg_file(files: List[UploadFile] = File(...)):
    """
    Upload ECG files (.dat and .hea) and get classification prediction
    """
    try:
        # Validate files
        validate_files(files)
        
        with tempfile.TemporaryDirectory() as tmpdirname:
            logger.info(f"Processing {len(files)} files")
            
            # Save uploaded files
            filenames = []
            for uploaded_file in files:
                filepath = Path(tmpdirname) / uploaded_file.filename
                with open(filepath, "wb") as f:
                    shutil.copyfileobj(uploaded_file.file, f)
                filenames.append(str(filepath))
                logger.info(f"Saved file: {uploaded_file.filename}")

            # Extract base filename for wfdb
            base_name = Path(filenames[0]).stem
            record_path = str(Path(tmpdirname) / base_name)
            
            # Read ECG data
            try:
                record = wfdb.rdrecord(record_path)
                ecg_data = record.p_signal.T  # Shape: (12, length)
                logger.info(f"ECG data shape: {ecg_data.shape}")
            except Exception as e:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Failed to read ECG files: {str(e)}"
                )

            # Preprocess ECG data
            ecg_normalized = (ecg_data - ecg_data.mean(axis=1, keepdims=True)) / (
                ecg_data.std(axis=1, keepdims=True) + 1e-8
            )
            
            # Create tensor
            ecg_tensor = torch.tensor(ecg_normalized, dtype=torch.float32).unsqueeze(0)
            
            # Make prediction
            result = predict(model, ecg_tensor)
            
            # Calculate statistics
            stats = calculate_ecg_stats(ecg_data)
            
            # Prepare response
            response = PredictionResponse(
                success=True,
                predicted_class=result["predicted_label"],
                predicted_class_index=result["predicted_class_index"],
                confidence=result.get("confidence", 0.0),
                ecg_stats=stats,
                message="ECG classification completed successfully"
            )
            
            logger.info(f"Prediction: {result['predicted_label']} (confidence: {result.get('confidence', 'N/A')})")
            return response

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal server error: {str(e)}"
        )

@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler"""
    logger.error(f"Unhandled exception: {str(exc)}")
    return JSONResponse(
        status_code=500,
        content=ErrorResponse(
            success=False,
            error="Internal server error",
            detail="An unexpected error occurred"
        ).dict()
    )

if __name__ == "__main__":
    uvicorn.run(
        "main:app", 
        host="0.0.0.0", 
        port=8000, 
        reload=True,
        log_level="info"
    )
