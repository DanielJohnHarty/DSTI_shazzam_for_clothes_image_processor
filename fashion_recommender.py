from keras.applications.vgg16 import preprocess_input
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
import argparse
import numpy as np
import h5py
import os
import matplotlib.pyplot as plt
import pandas as pd


from fashion_embedding import *




if __name__ == "__main__":

	ap = argparse.ArgumentParser()
	ap.add_argument('-t', '--targetImg', required=True,
					help = 'path to one clothes img target from which we want to generate recommendations')
	ap.add_argument('-l', '--targetLabel', required=True,
					help = 'category of clothes for img target')
	ap.add_argument('-i', '--img', required=True,
					help = 'path to clothes img data set')
	ap.add_argument('-e', '--embeddings', required=True,
					help = 'path to hdf5 file containing embeddings, ids and labels')
	ap.add_argument('-m', '--model', required=True,
					help = 'model folder name which contains model.json + weights.h5')
	ap.add_argument('-k', '--K', required=True,
					help = 'number of closest img to recommend')
	args = ap.parse_args()



	# get clothes categories encoder
	fo = open('fashion.txt', "r")
	categories = {}
	for i, cat in enumerate(fo.readline().rstrip("\n").split(',')):
		categories[cat]=i
	fo.close()
	print(categories)
	

	# load features and labels from hdf5 file
	hf = h5py.File(args.embeddings, 'r') 
	data_embeddings = hf['fashion_embeddings']
	data_ids = hf['fashion_ids'][:,0]
	data_labels = hf['fashion_labels'][:,0]
	
	n_features=data_embeddings.shape[1]


	# filter db of embeddings by label
	target_encoded_label = categories[args.targetLabel]
	bool_categories =  data_labels==target_encoded_label

	data_embeddings=data_embeddings[bool_categories,:]
	data_ids=data_ids[bool_categories]
	data_labels=data_labels[bool_categories]

	n_img=data_embeddings.shape[0]
	

	
	# load embedding model
	model_choice = args.model.split('/')[-1]
	if model_choice=='model_0':
		model = embeddingModel_0(args.model)
	if model_choice=='model_1':
		model = embeddingModel_1(args.model)

	embedding_size = model.get_output_dim()
	input_size = model.get_input_dim()

	model.summary()


	# loading img target
	img_target = load_img(args.targetImg, target_size=(input_size[0], input_size[1]))
	img_batch = [img_to_array(img_target)]
	img_batch = preprocess_input(np.array(img_batch))

	# embedding img target
	img_embedding = model.predict(img_batch)
	img_embedding = img_embedding.flatten()
	

	# recommendation by euclidian norm L(2)
	euclid = np.sqrt(np.sum((data_embeddings-img_embedding)**2, axis=1))
	indrank = np.argsort(euclid)
	ind_K = data_ids[indrank[:int(args.K)]]
	closestEuclidFiles=[str(ind)+'_' for ind in ind_K]

	# recommendation by manhattan norm L(1)
	manhat = np.sum(np.abs(data_embeddings-img_embedding), axis=1)
	indrank = np.argsort(manhat)
	ind_K = data_ids[indrank[:int(args.K)]]
	closestManhatFiles=[str(ind)+'_' for ind in ind_K]

	# recommendation by L(2/3) norm
	norm_sk = np.sum(np.abs(data_embeddings-img_embedding)**(2/3), axis=1)**(3/2)
	indrank = np.argsort(norm_sk)
	ind_K = data_ids[indrank[:int(args.K)]]
	closestSkFiles=[str(ind)+'_' for ind in ind_K]

	plt.figure(figsize=(8,5))
	plt.suptitle('recommendation')
	# euclidian reco
	plt.subplot(3,int(args.K)+1,1)
	plt.title('target')
	plt.imshow(img_target)
	plt.axis('off')
	i_imgRec=2
	for file in os.listdir(args.img):
		bool_closestFile = [file.startswith(startName) for startName in closestEuclidFiles]
		if any(bool_closestFile):
			imgRec = plt.imread(os.path.join(args.img, file))
			plt.subplot(3,int(args.K)+1,i_imgRec)
			plt.imshow(imgRec)
			plt.axis('off')
			print(os.path.join(args.img, file))
			i_imgRec+=1
	# manhattan reco
	plt.subplot(3,int(args.K)+1,i_imgRec)
	plt.title('target')
	plt.imshow(img_target)
	plt.axis('off')
	i_imgRec+=1
	for file in os.listdir(args.img):
		bool_closestFile = [file.startswith(startName) for startName in closestManhatFiles]
		if any(bool_closestFile):
			imgRec = plt.imread(os.path.join(args.img, file))
			plt.subplot(3,int(args.K)+1,i_imgRec)
			plt.imshow(imgRec)
			plt.axis('off')
			print(os.path.join(args.img, file))
			i_imgRec+=1
	# norm_sk reco
	plt.subplot(3,int(args.K)+1,i_imgRec)
	plt.title('target')
	plt.imshow(img_target)
	plt.axis('off')
	i_imgRec+=1
	for file in os.listdir(args.img):
		bool_closestFile = [file.startswith(startName) for startName in closestSkFiles]
		if any(bool_closestFile):
			imgRec = plt.imread(os.path.join(args.img, file))
			plt.subplot(3,int(args.K)+1,i_imgRec)
			plt.imshow(imgRec)
			plt.axis('off')
			print(os.path.join(args.img, file))
			i_imgRec+=1
	plt.show()





