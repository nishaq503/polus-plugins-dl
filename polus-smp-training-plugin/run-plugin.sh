#!/bin/bash

version=$(<VERSION)
data_path=$(readlink --canonicalize ../../data/smp_training)

# Inputs
pretrainedModel=/data/pretrained_model
modelName="Linknet"
encoderBase="ResNet"
encoderVariant="resnet34"
encoderWeights="imagenet"
optimizerName="Adam"
batchSize=8

imagesDir=/data/images
imagesPattern=".+"
labelsDir=/data/labels
labelsPattern=".+"
trainFraction=0.7

lossName="JaccardLoss"
metricName="IoU"
maxEpochs=10
patience=4
minDelta=1e-4

# Output paths
outputDir=/data/output

# Remove the --gpus all to test on CPU
docker run --mount type=bind,source="${data_path}",target=/data \
            --user "$(id -u)":"$(id -g)" \
            labshare/polus-smp-training-plugin:"${version}" \
            --modelName ${modelName} \
            --encoderBase ${encoderBase} \
            --encoderVariant ${encoderVariant} \
            --encoderWeights ${encoderWeights} \
            --optimizerName ${optimizerName} \
            --batchSize ${optimizerName} \
            --imagesDir ${imagesDir} \
            --imagesPattern ${imagesPattern} \
            --labelsDir ${labelsDir} \
            --labelsPattern ${labelsPattern} \
            --trainFraction ${trainFraction} \
            --lossName ${lossName} \
            --metricName ${metricName} \
            --maxEpochs ${maxEpochs} \
            --patience ${patience} \
            --minDelta ${minDelta} \
            --outputDir ${outputDir}
