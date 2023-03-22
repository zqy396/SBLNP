from histolab.slide import Slide
from histolab.tiler import GridTiler,RandomTiler
import os
from histolab.tiler import ScoreTiler
from histolab.scorer import NucleiScorer
from multiprocessing import Process

def histolab(file):
        path_all = '/mnt/data/Pathology/data/RHWU_level0'
        save_path = '/mnt/data/Pathology/data/RHWU_level0_tiles_1'
        path = os.path.join(path_all,file)
        prefix = file[0:9]
        save_path_simple = os.path.join(save_path,prefix)
        os.mkdir(save_path_simple)
        bladder_slide = Slide(path,processed_path=save_path_simple)
        scored_tiles_extractor = ScoreTiler(
            scorer = NucleiScorer(),
            tile_size=(448,448),
            n_tiles=200,
            level=0,
            check_tissue=True,
            tissue_percent=80.0,
            pixel_overlap=0, # default
            prefix=prefix, # save tiles in the "scored" subdirectory of slide's processed_path
            suffix=".png" # default
        )
        im =  scored_tiles_extractor.locate_tiles(slide=bladder_slide)
        # im.show()
        path1 = '//mnt/data/Pathology/data/1/' + prefix + '.png'
        im.save(path1)
        scored_tiles_extractor.extract(bladder_slide)

if __name__ == '__main__':
    path_all = '/mnt/data/Pathology/data/RHWU_level0'
    files = os.listdir(path_all)
    for file in files:
        p = Process(target=histolab,args=(file,))
        p.start()
