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


model = '/mnt/zqy_data/Slideflow/BLCA_LNM/models/00000-label-HP0-kfold1/label-HP0-kfold1_epoch20.zip'

P.evaluate(model=model,outcomes='label',filters={'dataset':['eval']})
P.generate_heatmaps(model=model)
heatmap = sf.Heatmap('/mnt/zqy_data/Slideflow/data/TCGA-2F-A9KT-01Z-00-DX1.ADD6D87C-0CC2-4B1F-A75F-108C9EB3970F.svs',
                     model=model,
                     stride_div=4,
                     num_threads=32)
heatmap.save('/mnt/zqy_data/Slideflow/results')


dts_ftrs = P.generate_features(model,layers='postconv',cache='activations_1.pkl')

slide_map = dts_ftrs.map_activations(
    n_neighbors=50, # UMAP parameter
    min_dist=0.1    # UMAP parameter
)

dataset = P.dataset(tile_px=512,tile_um='10x')
labels, unique_labels = dataset.labels('label', format='name')

# Assign the labels to the slide map, then plot
slide_map.label_by_slide(labels)
slide_map.plot(s=0.1)
plt.savefig('/mnt/zqy_data/Slideflow/umap_preds/slidemap_2.png',dpi=1000)

slide_map.label_by_preds(1)
slide_map.plot(s=0.1)
plt.savefig('/mnt/zqy_data/Slideflow/umap_preds/slidemap_1.png',dpi=1000)




mosaic = P.generate_mosaic(dts_ftrs)
mosaic.save('mosaic.png')

umap = mosaic.slide_map
umap.label_by_preds(1)
umap.save('umap_preds')

# Label by raw preds
umap.label_by_meta('prediction')
umap.save('umap_predictions')