#!/bin/bash

PATH_TARGET="red-skirt.png" # This is the image used to generate a recommendation
PATH_EMBEDDINGS="fashion_embeddings_1_sample.hdf5" # All images in the database are placed in a 'k dimension vecotr within this embeddings object'
PATH_MODEL="model_1" # Directory location of the trained model
PATH_IMG_DB="fashion_db_sample" # Directory location of the trained model

python3 fashion_recommender.py -t $PATH_TARGET -l skirts -i $PATH_IMG_DB -e $PATH_EMBEDDINGS -m $PATH_MODEL -k 10