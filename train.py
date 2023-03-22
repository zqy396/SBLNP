import slideflow as sf
from slideflow.util import get_slides_from_model_manifest
import matplotlib.pyplot as plt

# P = sf.create_project(
#   root='/mnt/zqy_data/Slideflow/BLCA_LNM',
#   slides='/mnt/zqy_data/LNM/TCGA/tcga_data/')

P = sf.load_project('/mnt/zqy_data/Slideflow/BLCA_LNM')

# dataset = P.dataset(tile_px=512,tile_um='10x',filters={'dataset':['train'],'category':['Her-positive','Her-negative']})
# dataset.extract_tiles(qc='otsu',roi_method = 'auto')
# dataset.num_tiles

P.extract_tiles(tile_px=512,tile_um='10x',qc='otsu')

hp = sf.ModelParams(
    tile_px=512,
    tile_um='10x',
    model='resnet50',
    batch_size=32,
    epochs=[20])

# Train with 5-fold cross-validation
P.train(
    'label',
    params=hp,
    val_k_fold=5,
    filters={'dataset': ['train'], 'label':['LN_positive','LN_negative']})