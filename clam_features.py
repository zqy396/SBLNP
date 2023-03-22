import slideflow as sf
from slideflow.util import get_slides_from_model_manifest
import matplotlib.pyplot as plt

# P = sf.create_project(
#   root='/mnt/zqy_data/Slideflow/1',
#   slides='/mnt/zqy_data/Slideflow/1/')

P = sf.load_project('/mnt/zqy_data/Slideflow/BLCA_LVI')

P.extract_tiles(tile_px=448,tile_um=112,qc='otsu')

P.generate_features_for_clam(model='resnet50',outdir='/mnt/zqy_data/Slideflow/BLCA_LVI/features',layers=['postconv'])
