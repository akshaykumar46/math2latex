import warnings
warnings.filterwarnings("ignore")

import io
import base64
from typing import Optional, Tuple, List
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
import torch
import cv2
import numpy as np
from PIL import Image

# Import model components
from model.tamer import TAMER
from datamodule.vocab import vocab
from datamodule.transforms import ScaleToLimitRange

# Initialize FastAPI app
app = FastAPI(
    title="Mathematical Expression Converter API",
    description="API for converting mathematical expressions from images to LaTeX strings",
    version="1.0.0"
)

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Global variables for model and device
device = None
model = None
vocab_initialized = False
transform = None

class PredictionRequest(BaseModel):
    beam_size: Optional[int] = 5
    max_len: Optional[int] = 100
    alpha: Optional[float] = 1.0
    early_stopping: Optional[bool] = True
    temperature: Optional[float] = 1.0

class PredictionResponse(BaseModel):
    prediction: str
    confidence: Optional[float] = None
    processing_time: Optional[float] = None

def initialize_model():
    """Initialize the model and load checkpoint"""
    global device, model, vocab_initialized, transform
    
    # Set device
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    
    print(f"Using device: {device}")
    
    # Load checkpoint
    try:
        checkpoint = torch.load("checkpoint.ckpt", map_location='cpu')
        hparams = checkpoint['hyper_parameters']
        
        # Initialize model
        model = TAMER(
            d_model=hparams.get('d_model', 256),
            growth_rate=hparams.get('growth_rate', 24),
            num_layers=hparams.get('num_layers', 16),
            nhead=hparams.get('nhead', 8),
            num_decoder_layers=hparams.get('num_decoder_layers', 3),
            dim_feedforward=hparams.get('dim_feedforward', 1024),
            dropout=hparams.get('dropout', 0.3),
            dc=hparams.get('dc', 32),
            cross_coverage=hparams.get('cross_coverage', True),
            self_coverage=hparams.get('self_coverage', True),
            vocab_size=hparams.get('vocab_size', 248)
        )
        
        # Load state dict
        if 'state_dict' in checkpoint:
            state_dict = {}
            for key, value in checkpoint['state_dict'].items():
                if key.startswith('tamer_model.'):
                    new_key = key.replace('tamer_model.', '')
                    state_dict[new_key] = value
            msg = model.load_state_dict(state_dict, strict=False)
        else:
            msg = model.load_state_dict(checkpoint, strict=False)
        
        print(f"Model loaded with message: {msg}")
        
        # Move model to device
        model.to(device)
        model.eval()
        
        # Initialize vocabulary
        vocab.init("dictionary.txt")
        vocab_initialized = True
        
        # Initialize transform
        transform = ScaleToLimitRange(w_lo=32, w_hi=512, h_lo=32, h_hi=512)
        
        print("Model initialization completed successfully")
        
    except Exception as e:
        print(f"Error initializing model: {e}")
        raise e

def preprocess_image(image_array: np.ndarray) -> Tuple[torch.Tensor, torch.Tensor]:
    """Preprocess image array for model input"""
    # Convert to grayscale if needed
    if len(image_array.shape) == 3:
        img = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
    else:
        img = image_array
    
    # Apply scaling transform
    img = transform(img)
    
    # Convert to tensor and normalize
    img = torch.from_numpy(img).float() / 255.0
    
    # Add batch and channel dimensions: [1, 1, H, W]
    img = img.unsqueeze(0).unsqueeze(0)
    
    # Create mask (assuming no padding needed for inference)
    h, w = img.shape[2:]
    mask = torch.zeros((1, h, w), dtype=torch.long)
    
    return img, mask.to(torch.bool)

def predict_expression(
    image_array: np.ndarray,
    beam_size: int = 5,
    max_len: int = 100,
    alpha: float = 1.0,
    early_stopping: bool = True,
    temperature: float = 1.0
) -> str:
    """Predict mathematical expression from image array"""
    if not vocab_initialized or model is None:
        raise RuntimeError("Model not initialized")
    
    # Preprocess image
    img, mask = preprocess_image(image_array)
    img = img.to(device)
    mask = mask.to(device)
    
    # Run beam search
    with torch.no_grad():
        hyps = model.beam_search(
            img, mask,
            beam_size=beam_size,
            max_len=max_len,
            alpha=alpha,
            early_stopping=early_stopping,
            temperature=temperature
        )
    
    # Convert hypotheses to strings
    predictions = []
    for hyp in hyps:
        # Remove special tokens and convert to string
        tokens = [token for token in hyp.seq if token not in [vocab.SOS_IDX, vocab.EOS_IDX, vocab.PAD_IDX]]
        expression = vocab.indices2label(tokens)
        predictions.append(expression)
    
    return predictions[0] if predictions else ""

# Initialize model on startup for older FastAPI versions
def startup_event():
    """Initialize model on startup"""
    initialize_model()

# Try to use the decorator if available, otherwise call directly
try:
    @app.on_event("startup")
    async def startup_event_decorated():
        """Initialize model on startup"""
        initialize_model()
except AttributeError:
    # For older FastAPI versions, initialize immediately
    startup_event()

@app.get("/", response_class=HTMLResponse)
async def root():
    """Serve the web interface"""
    with open("static/index.html", "r") as f:
        return HTMLResponse(content=f.read())

@app.get("/api/status")
async def api_status():
    """API status endpoint"""
    return {"message": "Mathematical Expression Converter API", "status": "running"}

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "vocab_initialized": vocab_initialized,
        "device": str(device) if device else None
    }

@app.post("/predict", response_model=PredictionResponse)
async def predict_endpoint(
    file: UploadFile = File(...),
    beam_size: int = 5,
    max_len: int = 100,
    alpha: float = 1.0,
    early_stopping: bool = True,
    temperature: float = 1.0
):
    """
    Convert mathematical expression from image to LaTeX string
    
    Args:
        file: Image file (PNG, JPG, JPEG)
        beam_size: Beam search size (default: 5)
        max_len: Maximum sequence length (default: 100)
        alpha: Length penalty parameter (default: 1.0)
        early_stopping: Whether to stop early (default: True)
        temperature: Sampling temperature (default: 1.0)
    
    Returns:
        PredictionResponse with the converted mathematical expression
    """
    import time
    start_time = time.time()
    
    # Validate file type
    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    try:
        # Read image file
        image_data = await file.read()
        image = Image.open(io.BytesIO(image_data))
        
        # Convert to numpy array
        image_array = np.array(image)
        
        # Predict expression
        prediction = predict_expression(
            image_array,
            beam_size=beam_size,
            max_len=max_len,
            alpha=alpha,
            early_stopping=early_stopping,
            temperature=temperature
        )
        
        processing_time = time.time() - start_time
        
        return PredictionResponse(
            prediction=prediction,
            processing_time=processing_time
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")

@app.post("/predict-json", response_model=PredictionResponse)
async def predict_json_endpoint(
    request: PredictionRequest,
    file: UploadFile = File(...)
):
    """
    Convert mathematical expression from image to LaTeX string with JSON parameters
    
    Args:
        request: PredictionRequest with parameters
        file: Image file (PNG, JPG, JPEG)
    
    Returns:
        PredictionResponse with the converted mathematical expression
    """
    return await predict_endpoint(
        file=file,
        beam_size=request.beam_size,
        max_len=request.max_len,
        alpha=request.alpha,
        early_stopping=request.early_stopping,
        temperature=request.temperature
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
