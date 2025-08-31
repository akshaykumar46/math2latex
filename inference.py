import warnings
warnings.filterwarnings("ignore")
from model.tamer import TAMER
import torch
import cv2
from typing import Tuple, List
from datamodule.vocab import vocab
from datamodule.transforms import ScaleToLimitRange 



device = torch.device("mps")



checkpoint = torch.load("checkpoint.ckpt", map_location='cpu')
print(checkpoint.keys())
print(checkpoint.get("hyper_parameters"))

# model = TAMER()
hparams = checkpoint['hyper_parameters']

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
    # Extract model state dict from Lightning checkpoint
    state_dict = {}
    for key, value in checkpoint['state_dict'].items():
        if key.startswith('tamer_model.'):
            new_key = key.replace('tamer_model.', '')
            state_dict[new_key] = value
    
    msg = model.load_state_dict(state_dict, strict=False)
else:
    msg = model.load_state_dict(checkpoint, strict=False)


print(msg)
# print(model)



vocab.init("/Users/akshaykumar/Documents/Projects/mathematicalExpressionConverter/dictionary.txt")
print(len(vocab))

image_path = "/Users/akshaykumar/Documents/Projects/mathematicalExpressionConverter/test_inputs/1002.png"
transform = ScaleToLimitRange(w_lo=32, w_hi=512, h_lo=32, h_hi=512)

model.to(device)

def preprocess_image(image_path: str) -> Tuple[torch.Tensor, torch.Tensor]:

        # Load image
        if isinstance(image_path, str):
            img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        else:
            img = image_path
            
        if img is None:
            raise ValueError(f"Could not load image from {image_path}")
        
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
    
def predict(
    image_path: str, 
    beam_size: int = 5,
    max_len: int = 100,
    alpha: float = 1.0,
    early_stopping: bool = True,
    temperature: float = 1.0
) -> List[str]:

    img, mask = preprocess_image(image_path)
    img = img.to(device)
    mask = mask.to(device)
    
    print(img.shape,mask.shape, type(img))
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
        
    return predictions

def predict_single(
    image_path: str, 
    beam_size: int = 5,
    max_len: int = 100,
    alpha: float = 1.0,
    early_stopping: bool = True,
    temperature: float = 1.0
) -> str:
    predictions = predict(
        image_path, beam_size, max_len, alpha, early_stopping, temperature
    )
    return predictions[0] if predictions else ""

output = predict_single(image_path=image_path)

print(output)
