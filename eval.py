import slideflow as sf

P = sf.load_project('/mnt/zqy_data/Slideflow/BLCA_LNM')

model = '/mnt/zqy_data/Slideflow/BLCA_LNM/models/00000-label-HP0-kfold1/label-HP0-kfold1_epoch20.zip'

P.evaluate(model=model,outcomes='label',filters={'dataset':['eval']})