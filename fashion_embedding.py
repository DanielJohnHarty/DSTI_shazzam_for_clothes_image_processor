import keras
#from keras.applications.vgg16 import preprocess_input
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
import argparse
import matplotlib.pyplot as plt
import numpy as np
import h5py
from tqdm import tqdm


from fashion_embedding import *




if __name__ == "__main__":
	ap = argparse.ArgumentParser()
	ap.add_argument('-m', '--model', required=True,
					help = 'model folder name which contains model.json + weights.h5')
	ap.add_argument('-b', '--batch', required=True,
				help = 'batch size for inference')
	ap.add_argument('-l', '--limit', required=True,
				help = 'max number of img to process')
	ap.add_argument('-d', '--dest', required=True,
				help = 'destination folder for hdf5 file with features')
	args = ap.parse_args()


	# load embedding model
	model_choice = args.model.split('/')[-1]
	if model_choice=='model_0':
		model = embeddingModel_0(args.model)
		preprocess_input=lambda x: keras.applications.vgg16.preprocess_input(x)
	if model_choice=='model_1':
		model = embeddingModel_1(args.model)
		preprocess_input=lambda x: keras.applications.vgg16.preprocess_input(x)
	if model_choice=='model_2':
		model = embeddingModel_2(args.model)
		preprocess_input=lambda x: keras.applications.mobilenet_v2.preprocess_input(x)
	if model_choice=='model_3':
		model = embeddingModel_3(args.model)
		preprocess_input=lambda x: keras.applications.mobilenet_v2.preprocess_input(x)

	model.summary()

	# hyper parameters for embedding
	embedding_size = model.get_output_dim()
	input_size = model.get_input_dim()

	batch_size = int(args.batch)
	max_n_img = int(args.limit)


	batch = []
	labels = []
	ids = []
	curr_row_in_hdf5 = 0
	# writing fashion.hdf5 with computed features
	with h5py.File(args.dest, "a") as f:
		
		f_dset = f.create_dataset('fashion_embeddings', (max_n_img,embedding_size), dtype='f', chunks=True)
		ids_dset = f.create_dataset('fashion_ids', (max_n_img,1), dtype='int32', chunks=True)
		l_dset = f.create_dataset('fashion_labels', (max_n_img,1), dtype='i8', chunks=True)

		fo = open('fashion.txt', "r")

		# encode img categories
		categories = {}
		for i, cat in enumerate(fo.readline().split(',')):
			categories[cat.rstrip("\n")] = i

		print('\n\nfeatures extraction from img...\n')
		for i, line in enumerate(tqdm(fo, total=max_n_img)):
			if i >= max_n_img:
				break
			else:
				# load img	
				input_img = load_img(line.rstrip("\n"), target_size=(input_size[0], input_size[1]))

				# preprocess file name: get id + label
				imgName = line.rstrip("\n").split('/')[-1]
				imgId = imgName.split('_')[0]
				imgCat = imgName.split('_')[1][:-4]
				imgEncodedCat = categories[imgCat]


				labels.append([int(imgEncodedCat)])
				ids.append([int(imgId)])
				batch.append(img_to_array(input_img))

				q_patches = i % batch_size
				# do inference on batch using features extractor model and write chunk in hdf5 file 
				if q_patches == 0 and i < max_n_img:
					input_batch = preprocess_input(np.array(batch))
					pred = model.predict(input_batch)
					batchsize = len(pred)
					#chunk = np.hstack((ids,labels,pred))
					f_dset[curr_row_in_hdf5:(curr_row_in_hdf5+batchsize)] = pred
					ids_dset[curr_row_in_hdf5:(curr_row_in_hdf5+batchsize)] = ids
					l_dset[curr_row_in_hdf5:(curr_row_in_hdf5+batchsize)] = labels
					batch = []
					labels = []
					ids = []
					curr_row_in_hdf5 += batchsize
		# write the remaining features inside hdf5 file
		if len(batch)>0:
			input_batch = preprocess_input(np.array(batch))
			pred = model.predict(input_batch)
			batchsize = len(pred)
			#chunk = np.hstack((ids,labels,pred))
			f_dset[curr_row_in_hdf5:(curr_row_in_hdf5+batchsize)] = pred
			ids_dset[curr_row_in_hdf5:(curr_row_in_hdf5+batchsize)] = ids
			l_dset[curr_row_in_hdf5:(curr_row_in_hdf5+batchsize)] = labels
			batch = []
			labels = []
			ids = []
			curr_row_in_hdf5 += batchsize
		fo.close()

