import slideflow as sf
from slideflow.util import get_slides_from_model_manifest
import matplotlib.pyplot as plt

P = sf.load_project('/mnt/zqy_data/Slideflow/BLCA_LVI')

heatmap = sf.Heatmap(
  slide='/path/to/slide.svs',
  model='/path/to/model'
  batch_size=64,
  num_processes=16
)

from slideflow.slide import qc

# Prepare the slide
wsi = sf.WSI('slide.svs', tile_px=299, tile_um=302, rois='/path')
wsi.qc([qc.Otsu(), qc.Gaussian()])

# Generate a heatmap
heatmap = sf.Heatmap(
  slide=wsi,
  model='/path/to/model'
  batch_size=64,
  num_processes=16
)

heatmap.plot(class_idx=0, cmap='inferno')

heatmap.add_inset(zoom=20, x=(10000, 10500), y=(2500, 3000), loc=1, axes=False)
heatmap.add_inset(zoom=20, x=(12000, 12500), y=(7500, 8000), loc=3, axes=False)
heatmap.plot(class_idx=0, mpp=1)


