import slideflow as sf
from slideflow.util import get_slides_from_model_manifest
import matplotlib.pyplot as plt


P = sf.load_project('/mnt/zqy_data/Slideflow/BLCA_LVI')

P.generate_features_for_clam(..., outdir='/eval/clam/path')

P.evaluate_clam(
    exp_name='evaluation',
    pt_files='/eval/clam/path',
    outcomes='category1',
    tile_px=299,
    tile_um=302
)
