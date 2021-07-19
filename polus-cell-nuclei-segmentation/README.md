# Polus Cell Nuclei Segmentation Plugin

This WIPP plugin consists of pre-trained neural networks to segment cell nuclei. The user can choose which neural network to use from a drop down menu. At present it consists of 2 neural networks listed above. Both the networks are very different to each other in terms of speed and accuracy. Taking an example of an image of size 1024 x 1024, the unet takes under 30 sec per image whereas the topcoders network takes around an hour per image but topcoders network has a far beter output.

For more information on WIPP, visit the [official WIPP page](https://isg.nist.gov/deepzoomweb/software/wipp).
For more information on Bioformats, vist the [official page](https://www.openmicroscopy.org/bio-formats/).

## Reference and credits
The plugin consists of 2 neural networks listed below with the links to their original source code: 

1. Unet
Segments cell nuclei using U-Net in Tensorflow. Neural net architecture and pretrained weights are taken from Data Science Bowl 2018 entry by Muhammad Asim.
Link to original codebase : https://github.com/axium/Data-Science-Bowl-2018

2. topcoders
This neural network  is the first place finisher at the 2018 Data Science Bowl. 
Link to the original codebase: https://github.com/selimsef/dsb2018_topcoders/

# Need 
Identifying cell nuclei is the first step in medical research. Owing to the advancements in laboratory automation, researchers have extensive amounts of image data available to them and its not feasible to manually segment nuclei. Due to this, substantial developements have been made to automate the process of nuclei segmentation and people across the globe have come up with creative models to solve this problem.This plugin enable users to exploit the power of artificial intelligence to segment nuclei. At present, it consists of 2 neural network architectures as mentioned above. 

# Using the plugin

## Inputs

The plugin takes 3 inputs as shown below :\
(i) Path to the input directory 
(ii) Path to the output directory : The output is a binary mask highlighting the nuclei. 
(iii) Model name which the user can choose from the drop down menu. 

| Name       | Description             | I/O    | Type  |
|------------|-------------------------|--------|-------|
| `inpDir`   | Input image collection  | Input  | Path  |
| `outDir`   | Output image collection | Output | Path  |
| `Model`    | Model name              | Model  | String|


## Running the Plugin

Create a local folder to emulate WIPP data folder with the name `<LOCAL_WIPP_FOLDER>`. Folder should have the following structure:
```
.
├── <LOCAL_WIPP_FOLDER>
|   ├── inputs
|   └── outputs
```

Then, run the docker container 
```bash
docker run -v <LOCAL_WIPP_FOLDER>/inputs:/data/inputs -v <LOCAL_WIPP_FOLDER>/outputs:/data/outputs labshare/polus-cell-nuclei-segmentation:2.0.0 \
  --inpDir /data/inputs \
  --outDir /data/outputs\
  --model 'model name'
```

