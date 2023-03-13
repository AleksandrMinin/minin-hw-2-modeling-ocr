from os import path as osp


DATA_PATH = "/storage/minin/datasets/barcodes"
DF_PATH = osp.join(DATA_PATH, "num_under_barcodes.tsv")
TRAIN_IMAGES_PATH = osp.join(DATA_PATH, "crop_barcodes")
BACKGROUNDS_DIR = './generator/backgrounds/'
PROJECT_PATH = osp.abspath(osp.join(osp.dirname(osp.realpath(__file__)), "../"))  # noqa: WPS221
BEST_GEN_MODEL = osp.join(PROJECT_PATH, "gen_weights/model.best.pth")
TORCH_FILE = osp.join(PROJECT_PATH, "weights/model.best.pth")
ONNX_FILE = osp.join(PROJECT_PATH, "weights/model.best.onnx")
