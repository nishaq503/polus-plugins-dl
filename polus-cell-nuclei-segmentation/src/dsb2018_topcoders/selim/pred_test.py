
import os
import logging

from params import args

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

from keras.preprocessing.image import img_to_array, load_img

from keras.applications.imagenet_utils import preprocess_input

from models.model_factory import make_model


from os import path, mkdir, listdir

import numpy as np
np.random.seed(1)
import random

random.seed(1)
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

tf.set_random_seed(1)
import timeit
import cv2
from tqdm import tqdm    


test_folder = args.test_folder
test_pred = os.path.join(args.out_root_dir, args.out_masks_folder)

all_ids = []
all_images = []
all_masks = []

OUT_CHANNELS = args.out_channels    
    
def preprocess_inputs(x):
    return preprocess_input(x, mode=args.preprocessing_function)

def factors(n):
    # Modified from https://stackoverflow.com/a/6909532
    f = [1,1]
    for i in range(1,int(n**0.5)+1):
        if n % i == 0:
            f = [i,n//i]
    return f

if __name__ == '__main__':

    
    logging.basicConfig(format='%(asctime)s - %(name)-8s - %(levelname)-8s - %(message)s',
                    datefmt='%d-%b-%y %H:%M:%S')
    logger = logging.getLogger("main")
    logger.setLevel(logging.INFO)
    
    t0 = timeit.default_timer()
    logger.info("Loading model weights")
    weights = [path.join(args.models_dir, m) for m in args.models]
    models = []
    num_tiles = int(args.num_tiles)
    for w in weights:
        model = make_model(args.network, (None, None, 3))
        logger.info("Building model {} from weights {} ".format(args.network, w))
        model.load_weights(w)
        models.append(model)
    os.makedirs(test_pred, exist_ok=True)
    #print('Predicting test')
    for d in tqdm(listdir(test_folder)):
        logger.info("Predicting Image: {}".format(d))
        fid = d
        full_img = cv2.imread(path.join(test_folder, fid), cv2.IMREAD_COLOR)[...,::-1]

        if num_tiles == None:
            num_tiles = 1
        
        f = factors(num_tiles)
        X_TILE_SIZE = 512 * f[1]
        Y_TILE_SIZE = 512 * f[0]

        predicted = np.zeros((full_img.shape[0], full_img.shape[1], 3), dtype='uint8')
        
        for x in range(0,full_img.shape[0],X_TILE_SIZE):
            x_max = min([full_img.shape[0],x+X_TILE_SIZE])
            
            for y in range(0,full_img.shape[1],Y_TILE_SIZE):
                y_max = min([full_img.shape[1],y+Y_TILE_SIZE])
                final_mask = None
                for scale in range(1):

                    img = full_img[x:x_max, y:y_max, :]
 
                    if final_mask is None:
                        final_mask = np.zeros((img.shape[0], img.shape[1], OUT_CHANNELS))
                    if scale == 1:
                        img = cv2.resize(img, None, fx=0.75, fy=0.75, interpolation=cv2.INTER_AREA)
                    elif scale == 2:
                        img = cv2.resize(img, None, fx=1.25, fy=1.25, interpolation=cv2.INTER_CUBIC)

                    x0 = 16
                    y0 = 16
                    x1 = 16
                    y1 = 16
                    if (img.shape[1] % 32) != 0:
                        x0 = int((32 - img.shape[1] % 32) / 2)
                        x1 = (32 - img.shape[1] % 32) - x0
                        x0 += 16
                        x1 += 16
                    if (img.shape[0] % 32) != 0:
                        y0 = int((32 - img.shape[0] % 32) / 2)
                        y1 = (32 - img.shape[0] % 32) - y0
                        y0 += 16
                        y1 += 16
                    img0 = np.pad(img, ((y0, y1), (x0, x1), (0, 0)), 'symmetric')

                    inp0 = []
                    inp1 = []
                    for flip in range(2):
                        for rot in range(4):
                            if flip > 0:
                                img = img0[::-1, ...]
                            else:
                                img = img0
                            if rot % 2 == 0:
                                inp0.append(np.rot90(img, k=rot))
                            else:
                                inp1.append(np.rot90(img, k=rot))

                    inp0 = np.asarray(inp0)
                    inp0 = preprocess_inputs(np.array(inp0, "float32"))
                    inp1 = np.asarray(inp1)
                    inp1 = preprocess_inputs(np.array(inp1, "float32"))

                    mask = np.zeros((img0.shape[0], img0.shape[1], OUT_CHANNELS))

                    for model in models:
                        pred0 = model.predict(inp0, batch_size=1)
                        pred1 = model.predict(inp1, batch_size=1)
                        j = -1
                        for flip in range(2):
                            for rot in range(4):
                                j += 1
                                if rot % 2 == 0:
                                    pr = np.rot90(pred0[int(j / 2)], k=(4 - rot))
                                else:
                                    pr = np.rot90(pred1[int(j / 2)], k=(4 - rot))
                                if flip > 0:
                                    pr = pr[::-1, ...]
                                mask += pr  # [..., :2]

                    mask /= (8 * len(models))
                    mask = mask[y0:mask.shape[0] - y1, x0:mask.shape[1] - x1, ...]
                    if scale > 0:
                        mask = cv2.resize(mask, (final_mask.shape[1], final_mask.shape[0]))
                    final_mask += mask
                final_mask /= 1
                if OUT_CHANNELS == 2:
                    final_mask = np.concatenate([final_mask, np.zeros_like(final_mask)[..., 0:1]], axis=-1)
                final_mask = final_mask * 255
                final_mask = final_mask.astype('uint8')  
                predicted[x:x_max, y:y_max, :] = final_mask

        cv2.imwrite(path.join(test_pred,d), predicted)

    elapsed = timeit.default_timer() - t0
    logger.info('Time Elasped:  {:.3f} min'.format(elapsed / 60))
  
