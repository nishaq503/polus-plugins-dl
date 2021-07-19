"""
This script executes the 'topcoders' neural network. It consists of the various helper functions 
which are used in the main() function below. Refer to the main() function to get information about 
the flow of the execution.
"""

import shutil
import os
import subprocess
import logging
from pathlib import Path
import cupy

def execute_NN(main_dir): 
    """
    This function uses the subprocessing module to execute the 'Topcoders' neural network. 
    Is is hardcoded to take the root directory of the plugin as input.
    
    Subprocess 1-3 consist of the arguments from the following shell script in the original codebase : root_dir/dsb2018_topcoders/albu/src/predict_test.sh
    
    Subprocess 4-6 consist of the arguments from the following shell script in the original codebase : root_dir/dsb2018_topcoders/selim/predict_test.sh
    
    Subprocess 6-10 consist of the arguments from the following shell script in the original codebase : root_dir/dsb2018_topcoders/victor/predict_test.sh
    """
   
    os.chdir(main_dir+'/dsb2018_topcoders/albu/src/')
    
    try:   
       mem_free, mem_total = cupy.cuda.Device(0).mem_info
       num_tiles = int(mem_free // (1.5 * 2 ** 30) - 1)
       device = 0
    except:
       device = -1
       num_tiles = 1

    root = "python3 bowl_eval.py"
    
    # Subprocess 1

    process=subprocess.Popen(root + " ./configs/dpn_softmax_s2.json --num_tiles {} --device {}".format(num_tiles,device),shell=True)
    process.wait()      
    
     # Subprocess 2
    process=subprocess.Popen(root + " ./configs/dpn_sigmoid_s2.json --num_tiles {} --device {}".format(num_tiles,device),shell=True)
    process.wait()  
    
     # Subprocess 3
    #num_tiles = int(mem_free // (1 * 2 ** 30) - 1)
    process=subprocess.Popen(root + " ./configs/resnet_softmax_s2.json --num_tiles {} --device {}".format(num_tiles,device),shell=True)
    process.wait() 


    os.chdir(main_dir+'/dsb2018_topcoders/selim/')      
     # Subprocess 4
    process=subprocess.Popen("python3 pred_test.py --gpu 0 --num_tiles {} --preprocessing_function caffe --network resnet101_2 --out_masks_folder pred_resnet101_full_masks --out_channels 2 --models_dir nn_models --models best_resnet101_2_fold0.h5 best_resnet101_2_fold1.h5 best_resnet101_2_fold2.h5 best_resnet101_2_fold3.h5 ".format(num_tiles),shell=True)
    process.wait()
    
     # Subprocess 5
    process=subprocess.Popen("python3 pred_test.py --gpu 0 --num_tiles {} --preprocessing_function torch --network densenet169_softmax --out_masks_folder pred_densenet169_softmax --out_channels 3 --models_dir nn_models --models best_densenet169_softmax_fold0.h5 best_densenet169_softmax_fold1.h5 best_densenet169_softmax_fold2.h5 best_densenet169_softmax_fold3.h5 ".format(num_tiles),shell=True)
    process.wait()
    
     # Subprocess 6
    process=subprocess.Popen("python3 pred_test.py --gpu 0 --num_tiles {} --preprocessing_function caffe --network resnet152_2 --out_masks_folder pred_resnet152 --out_channels 2 --models best_resnet152_2_fold0.h5 best_resnet152_2_fold1.h5 best_resnet152_2_fold2.h5 best_resnet152_2_fold3.h5 ".format(num_tiles),shell=True)
    process.wait() 


    os.chdir(main_dir+'/dsb2018_topcoders/victor/')  
    
     # Subprocess 7 
    process=subprocess.Popen("python3 predict_inception.py --num_tiles {} ".format(num_tiles) ,shell=True)
    process.wait()  
    
     # Subprocess 8
    process=subprocess.Popen("python3 predict_densenet.py --num_tiles {} ".format(num_tiles),shell=True)
    process.wait()  

     # Subprocess 9
    process = subprocess.Popen("python3 merge_test.py",shell=True)
    process.wait() 

     # Subprocess 10
    process = subprocess.Popen("python3 create_submissions.py",shell=True)
    process.wait() 
    
    os.chdir(main_dir)
        
def delete_dir(main_dir):
    """
    This function deletes all the files created during the intermediate steps while predicting the output. 
    Around 20 images are created per single input image. The main purpose of this function is to  prevent
    excessive memory consumption.
    """
    
    # Delete all the supplemntary files
    shutil.rmtree(main_dir+'/dsb2018_topcoders/predictions')
    shutil.rmtree(main_dir+'/dsb2018_topcoders/albu/results_test')
    shutil.rmtree(main_dir+'/dsb2018_topcoders/data_test')
    
    # Create empty directories for the next iteration
    os.makedirs(main_dir+'/dsb2018_topcoders/predictions')
    os.makedirs(main_dir+'/dsb2018_topcoders/albu/results_test')
    os.makedirs(main_dir+'/dsb2018_topcoders/data_test')    

def excecute_topcoders_workflow(input_dir, output_dir):
    """
    This is the main function that executes the neural network named 'topcoders' 
    The steps involved in the exectution are as follows:
        
    1. The neural network is hardcoded to read input images from the following 
       directory : root_dir/dsb2018_topcoders/data_test
    2. The neural network processes images in batches of 4. Four images are copied from the
       input directory to the directory stated in point 1 and the neural network is
       executed.
    3. The model outputs the images(3 channel) to the following 
       directory : root_dir/dsb2018_topcoders/predictions/
    4. The 3 channel output image is converted to a binary image and written
       to the desired output directory stated by the user.
    5. The network creates around 20 intermediate images to create the final 3 channel output.
       These supplementary images are deleted before the next iteration to reduce the
       memory consumption.     
       
    """
    # Set Logging
    logging.basicConfig(format='%(asctime)s - %(name)-8s - %(levelname)-8s - %(message)s',
                        datefmt='%d-%b-%y %H:%M:%S')
    logger = logging.getLogger("main")
    logger.setLevel(logging.INFO)  
    
    # input and output directory    
    logger.info("Welcome")
    logger.info("input_dir: {}".format(input_dir))     
    logger.info("output_dir: {}".format(output_dir))
    
    # store all the images in the input collection
    filenames= sorted(os.listdir(input_dir))
    
    # get the root directory
    main_dir=os.getcwd()
    
    # The network is hardcoded to read input images from the following directory
    test_data_dir=main_dir+'/dsb2018_topcoders/data_test'
    
    # The network writes the output to the following directory. 
    predictions_path=main_dir+'/dsb2018_topcoders/predictions/merged_test/'    
    
    # specify batch_size    
    batch_size=4
    
    # loop over the input in increments of batch_size
    for i in range(0,len(filenames),batch_size):        
           
        # iterate over the minibatch and copy files to the test_data_dir
        for j in range(i,min(i+batch_size,len(filenames))):            
            filename=filenames[j]
            
            # Use symbolic link instead of copy to reduce the amount of data being moved around
            Path(os.path.join(test_data_dir,filename)).symlink_to(os.path.join(input_dir,filename))
            
        logger.info('Executing NN for files in range {:.2f} - {:.2f} ....'.format(i,j))    
        
        # execute the neural network
        execute_NN(main_dir)                 
        
        # create and write the binary output
        logger.info('Writing Outputs.....')       
        write_output = subprocess.Popen("python3 output.py --predPath {} --outDir {} --inpDir {}".format(predictions_path,output_dir, input_dir),shell=True)
        write_output.wait()
        logger.info('Deleting excess files.....')   
        
        # delete the  intermediate images created as discussed in step 5 of the function description above         
        #delete_dir(main_dir) 
    
    logger.info('100% complete...')    
    
