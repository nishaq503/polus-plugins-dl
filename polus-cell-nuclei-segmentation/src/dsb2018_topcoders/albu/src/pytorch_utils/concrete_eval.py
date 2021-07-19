import os

import cv2
cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)
import numpy as np

from .eval import Evaluator
from pathlib import Path


from dataset.neural_dataset import SequentialDataset
from torch.utils.data.dataloader import DataLoader as PytorchDataLoader
import torch
from .eval import predict8tta, read_model
import tqdm
import cupy

class FullImageEvaluator(Evaluator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def process_batch(self, predicted, model, data, prefix=""):
        names = data['image_name']
        for i in range(len(names)):
            self.on_image_constructed(names[i], predicted[i,...], prefix)

    def save(self, name, prediction, prefix=""):
        if self.test:
            path = os.path.join(self.config.dataset_path, name)
        else:
            path = os.path.join(self.config.dataset_path, 'images_all', name)
        # print(Path(path).resolve())
        # print(cv2.imread(path, 0))
        rows, cols = cv2.imread(path, 0).shape[:2]
        prediction = prediction[0:rows, 0:cols,...]
        if prediction.shape[2] < 3:
            zeros = np.zeros((rows, cols), dtype=np.float32)
            prediction = np.dstack((prediction[...,0], prediction[...,1], zeros))
        else:
            prediction = cv2.cvtColor(prediction, cv2.COLOR_RGB2BGR)
        if self.test:
            name = os.path.split(name)[-1]
        cv2.imwrite(os.path.join(self.save_dir, prefix + name), (prediction * 255).astype(np.uint8))
        
def factors(n):
    # Modified from https://stackoverflow.com/a/6909532
    f = [1,1]
    for i in range(1,int(n**0.5)+1):
        if n % i == 0:
            f = [i,n//i]
    return f
        
class TiledImageEvaluator(FullImageEvaluator):

    def predict(self, fold, val_indexes,device=None,num_tiles=None):
        
        if num_tiles == None:
            num_tiles = 1
        
        f = factors(num_tiles)
        X_TILE_SIZE = 512 * f[1]
        Y_TILE_SIZE = 512 * f[0]
        
        prefix = ('fold' + str(fold) + "_") if self.test else ""
        val_dataset = SequentialDataset(self.ds, val_indexes, stage='test', config=self.config, transforms=self.val_transforms)
        val_dl = PytorchDataLoader(val_dataset, batch_size=self.config.predict_batch_size, num_workers=self.num_workers, drop_last=False)
        model = read_model(self.folder, fold)
        pbar = tqdm.tqdm(val_dl, total=len(val_dl))
        
        for data in pbar:
            samples = data['image']
            
            predicted = np.zeros((samples.shape[0],) + samples.shape[-2:] + (samples.shape[1],),dtype=np.float32)
            
            for x in range(0,samples.shape[3],X_TILE_SIZE):
                x_max = min([samples.shape[3],x+X_TILE_SIZE])
                
                for y in range(0,samples.shape[2],Y_TILE_SIZE):
                    y_max = min([samples.shape[2],y+Y_TILE_SIZE])

                    p = predict8tta(model, samples[0:1,:,y:y_max,x:x_max],
                                    self.config.sigmoid)
                    
                    predicted[0,y:y_max,x:x_max,:] = p
                    
            self.process_batch(predicted, model, data, prefix=prefix)
        
        self.post_predict_action(prefix=prefix)
        