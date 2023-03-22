import os
import pandas as pd
import shutil

data = pd.read_excel('/mnt/zqy_data/LVI/TCGA/BLCA_LVI.xlsx',sheet_name=3)
tcga_wsi = '/mnt/data/Pathology/data/BLCA_train'
target_wsi = '/mnt/zqy_data/LVI/TCGA/tcga_data'
files = os.listdir(tcga_wsi)
list = data['sampleID'].tolist()
for file in files:
    path = os.path.join(tcga_wsi,file)
    a = file[0:12]
    if a in list:
        target_path = os.path.join(target_wsi,file)
        shutil.copyfile(path,target_path)
