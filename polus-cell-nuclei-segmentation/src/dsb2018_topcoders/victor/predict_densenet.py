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
from models import get_densenet121_unet_softmax
from tqdm import tqdm
import argparse, logging

test_folder = path.join('..', 'data_test')

models_folder = 'nn_models'
test_pred = path.join('..', 'predictions', 'densenet_test_pred_2')

all_ids = []
all_images = []
all_masks = []

def preprocess_inputs(x):
    x = np.asarray(x, dtype='float32')
    x /= 127.5
    x -= 1.
    return x

def bgr_to_lab(img):
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(17, 17))
    lab = clahe.apply(lab[:, :, 0])
    if lab.mean() > 127:
        lab = 255 - lab
    return lab[..., np.newaxis]

#def load_image(path):
#    bf = bfio.BioReader(path)    
#    image = bf.read_image()
#    image=image[:,:,0,:,0]
#    if image.shape[2] == 3:
#        return image
#    elif image.shape[2]==1:
#        return np.dstack((image[:,:,0], image[:,:,0],image[:,:,0]))

def factors(n):
    # Modified from https://stackoverflow.com/a/6909532
    f = [1,1]
    for i in range(1,int(n**0.5)+1):
        if n % i == 0:
            f = [i,n//i]
    return f

if __name__ == '__main__':
    # intialize logging
    logging.basicConfig(format='%(asctime)s - %(name)-8s - %(levelname)-8s - %(message)s',
                        datefmt='%d-%b-%y %H:%M:%S')
    logger = logging.getLogger("main")
    logger.setLevel(logging.INFO)

    # setup arguement parsing
    parser = argparse.ArgumentParser(prog='predict_densenet')
    parser.add_argument('--num_tiles', dest='num_tiles', type=int, required=True)
    

    # parse the arguments
    args = parser.parse_args()
    num_tiles = args.num_tiles


    t0 = timeit.default_timer()

    if not path.isdir(test_pred):
        mkdir(test_pred)

    models = []
    
    logger.info('Loading  densenet models...')
    
    for it in range(4):
        model = get_densenet121_unet_softmax((None, None), weights=None)
        model.load_weights(path.join(models_folder, 'densenet_weights_{0}.h5'.format(it)))
        models.append(model)
        
    #print('Predicting test')
    for d in tqdm(listdir(test_folder)):
        logger.info('Predicting image: {}'.format(d))
        fid = d
        full_img = cv2.imread(path.join(test_folder, fid), cv2.IMREAD_COLOR)

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
                for scale in range(3):
                    img = full_img[x:x_max, y:y_max, :]
                    #img=load_image(path.join(test_folder, fid))[...,::-1]
                    if final_mask is None:
                        final_mask = np.zeros((img.shape[0], img.shape[1], 3))
                    if scale == 1:
                        img = cv2.resize(img, None, fx=0.75, fy=0.75)
                    elif scale == 2:
                        img = cv2.resize(img, None, fx=1.25, fy=1.25)
                    elif scale == 3:
                        img = cv2.resize(img, None, fx=1.5, fy=1.5)
                        
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
                    img0 = np.pad(img, ((y0,y1), (x0,x1), (0, 0)), 'symmetric')
                    
                    img0 = np.concatenate([img0, bgr_to_lab(img0)], axis=2)
                    
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
                    inp0 = preprocess_inputs(inp0)
                    inp1 = np.asarray(inp1)
                    inp1 = preprocess_inputs(inp1)
                    
                    mask = np.zeros((img0.shape[0], img0.shape[1], 3))
                    
                    for model in models:
                        pred0 = model.predict(inp0, batch_size=1)
                        pred1 = model.predict(inp1, batch_size=1)
                        j = -1
                        for flip in range(2):
                            for rot in range(4):
                                j += 1
                                if rot % 2 == 0:
                                    pr = np.rot90(pred0[int(j / 2)], k=(4-rot))
                                else:
                                    pr = np.rot90(pred1[int(j / 2)], k=(4-rot))
                                if flip > 0:
                                    pr = pr[::-1, ...]
                                mask += pr
                                
                    mask /= (8 * len(models))
                    mask = mask[y0:mask.shape[0]-y1, x0:mask.shape[1]-x1, ...]
                    if scale > 0:
                        mask = cv2.resize(mask, (final_mask.shape[1], final_mask.shape[0]))
                    final_mask += mask
                final_mask /= 3
                final_mask = final_mask * 255
                final_mask = final_mask.astype('uint8')
                predicted[x:x_max, y:y_max, :] = final_mask
        cv2.imwrite(path.join(test_pred, d), predicted)
        
    elapsed = timeit.default_timer() - t0
    print('Time: {:.3f} min'.format(elapsed / 60))
