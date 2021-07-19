"""
This script executes the 'unet' neural network. It uses the subprocess module to execute the ./src_unet/segment.py file for 
a batch of 20 images at a time. 
"""

import argparse, subprocess, logging, time
from pathlib import Path
import os

BATCH_SIZE = 20

def execute_unet(input_dir,output_dir):
    
    # Initialize the logger
    logging.basicConfig(format='%(asctime)s - %(name)-8s - %(levelname)-8s - %(message)s',
                        datefmt='%d-%b-%y %H:%M:%S')
    logger = logging.getLogger("main")
    logger.setLevel(logging.INFO)

    
    # Input and output directory    
    logger.info("input_dir: {}".format(input_dir))    
    logger.info("output_dir: {}".format(output_dir))
    
    # Get a list of images
    files = [str(f.absolute()) for f in Path(input_dir).iterdir()]
    main_dir=os.getcwd()
    os.chdir(main_dir+"/src_unet/")
    # Loop over images, 20 at a time
    for ind in range(0,len(files),BATCH_SIZE):
        logger.info('{:.2f}% complete...'.format(100*ind/len(files)))
        batch = ','.join(files[ind:min([ind+BATCH_SIZE,len(files)])])
        
        process = subprocess.Popen("python3 segment.py --batch '{}' --outDir '{}'".format(batch,output_dir),shell=True)
        while process.poll() == None:
            time.sleep(1) # Put the main process to sleep inbetween checks, free up some processing power
    logger.info('100% complete...')


    
