import slideflow as sf
from slideflow.util import get_slides_from_model_manifest
import matplotlib.pyplot as plt

# P = sf.create_project(
#   root='/mnt/zqy_data/Slideflow/1',
#   slides='/mnt/zqy_data/Slideflow/1/')

P = sf.load_project('/mnt/zqy_data/Slideflow/BLCA_LVI')

import slideflow.clam

clam_args = sf.clam.get_args(k=3, bag_loss='svm', ...)

P.train_clam(
    exp_name='test_experiment',
    pt_files='/clam/path',
    outcomes='category1',
    dataset=dataset,
    clam_args=clam_args
)
