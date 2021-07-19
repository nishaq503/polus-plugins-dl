import cv2 
import os 
import numpy as np
import argparse
import logging
from bfio import BioReader, BioWriter

def create_binary(image):    
    
    """
    The output of the neural network is a 3 channel image highlighting the nuclei in the image. 
    This function converts the output into a binary image. 
    
    Input: The 3 channel output prediction from the network
    Output: A binary image highlighting the nuclei
    """
    #Channel 1 of the output image highlights the area consisting of the nuclei
    channel1=image[:,:,0]
    
    # Channel 2 of the output image consists of the boundaries between adjoining nuclei
    channel2=image[:,:,1]
    _,channel1=cv2.threshold(channel1, 127,255,cv2.THRESH_BINARY) 
    _,channel2=cv2.threshold(channel2, 127,255,cv2.THRESH_BINARY) 
    
    #Subtracting channel 2 from channel 1 to get the desired output
    img1=channel1-channel2
    
    return img1

def create_and_write_output(predictions_path,output_path,inpDir):
    
    """
    This script uses the bfio utility to write the output.
    Inputs:
        predictions_path: The directory in which the neural networks writes its 3 channel output
        output_path: The directory in which the user wants the final binary output
        inpDir: The input directory consisting of the input collection
    """
    
    filenames= sorted(os.listdir(predictions_path))    
    for filename in filenames:
        
        # read the 3 channel output image from the neural network
        image=cv2.imread(os.path.join(predictions_path,filename))
        
        # create binary image output using the create_binary function
        out_image=create_binary(image)   
        
        # read and store the metadata from the input image
        with BioReader(os.path.join(inpDir,filename)) as br:
            metadata = br.metadata

        # Write the binary output consisting of the metadata using bfio.
        output_image_5channel=np.zeros((out_image.shape[0],out_image.shape[1],1,1,1),dtype=np.uint8)
        output_image_5channel[:,:,0,0,0]=out_image   

        with BioWriter(os.path.join(output_path,filename), metadata=metadata) as bw:
            bw.dtype = output_image_5channel.dtype
            bw.write(output_image_5channel)
 
        

if __name__ == "__main__":
    
    # Set Logging
    logging.basicConfig(format='%(asctime)s - %(name)-8s - %(levelname)-8s - %(message)s',
                        datefmt='%d-%b-%y %H:%M:%S')
    logger = logging.getLogger("main")
    logger.setLevel(logging.INFO) 
    
    # parse the inputs
    parser=argparse.ArgumentParser()
    parser.add_argument('--predPath',dest='predictions_path',type=str,required=True)
    parser.add_argument('--outDir',dest='output_directory',type=str,required=True)
    parser.add_argument('--inpDir',dest='input_directory',type=str,required=True)

    try:
        
        # store the inputs
        args = parser.parse_args()    
        input_dir = args.input_directory
        output_dir = args.output_directory  
        predictions_path=args.predictions_path
        
        logger.info('writing outputs...')    
        create_and_write_output(predictions_path,output_dir,input_dir)
    finally:
        logger.info('Batch complete...') 
        
    
    
    
    