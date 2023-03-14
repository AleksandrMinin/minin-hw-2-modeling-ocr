import logging
import torch
import onnx
from onnxsim import simplify

from src.config import config
from src.config import IMG_HEIGHT, IMG_WIDTH
from src.constants import TORCH_FILE, ONNX_FILE
from src.crnn import CRNN


DEVICE = 'cpu'

logger = logging.getLogger('model_to_onnx.py')
logging.basicConfig(level=logging.INFO)

state_dict = torch.load(TORCH_FILE)
model = CRNN(**config.model_kwargs)
model.load_state_dict(state_dict)
model.eval()

logger.info("START ONNX EXPORT")
dummy_input = torch.rand(1, 3, IMG_HEIGHT, IMG_WIDTH, device=DEVICE)
torch.onnx.export(
    model,
    dummy_input,
    ONNX_FILE,
    verbose=False,
    input_names=['input'],
    output_names=['output'],
    opset_version=11,
)
logger.info('FINISH ONNX EXPORT')
logger.info('CHECK ONNX')
onnx_model = onnx.load(ONNX_FILE)
onnx.checker.check_model(onnx_model)
logger.info('ONNX OK')
logger.info('SIMPLIFY ONNX MODEL')
_, check = simplify(onnx_model)
if check:
    logger.info('SIMPLIFY SUCCESS')
else:
    logger.error('SIMPLIFY FAILED')