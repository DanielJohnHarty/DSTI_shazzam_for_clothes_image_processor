import matplotlib.pyplot as plt
import numpy as np
import h5py
import seaborn as sns
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import argparse

if __name__ == "__main__":
	ap = argparse.ArgumentParser()
	ap.add_argument('-e', '--embeddings', required=True,
					help = 'path to hdf5 file containing embeddings, ids and labels')
	ap.add_argument('-l', '--limit', required=True,
				help = 'max number of samples used to perform visualization (PCA/tSNE)')
	ap.add_argument('-s', '--save', required=True,
				help = 'if "yes" or "YES" it saves plot as png in current folder else does not')
	args = ap.parse_args()



	# get list of clothes categories
	fo = open('fashion.txt', "r")
	categories = []
	for i, cat in enumerate(fo.readline().rstrip("\n").split(',')):
		categories.append(cat)
	print(categories)
	fo.close()

	# load features and labels from hdf5 file
	print(args.embeddings)
	hf = h5py.File(args.embeddings, 'r') 
	print(hf)
	data_embeddings = hf.get('fashion_embeddings')[()]
	data_ids = hf.get('fashion_ids')[()]
	data_labels = hf.get('fashion_labels')[()]
	
	n_features=data_embeddings.shape[1]
	n_img=data_embeddings.shape[0]

	# from np.array to pandas dataframe
	feat_cols=['feat_'+str(i) for i in range(n_features)]
	pd_fashion = pd.DataFrame(data_embeddings,columns=feat_cols)
	pd_fashion['clothes_labels']=data_labels
	pd_fashion['clothes_ids']=data_ids
	del data_labels
	del data_embeddings
	del data_ids
	pd_fashion['clothes_labels']=pd_fashion['clothes_labels'].apply(lambda x: categories[x])

	# display hist plot on clothes labels
	plt.figure(figsize=(16,10))
	sns.set()
	hist_plot=sns.countplot(x="clothes_labels", data=pd_fashion).set_title('clothes distribution')
	if args.save=="yes" or args.save=="YES":
		fig = hist_plot.get_figure()
		fig.savefig("fashion_hist.png")
	plt.show()

	print('shape of pd: ', pd_fashion.shape)

	# take a random sample of dataframe
	rndperm = np.random.permutation(n_img)
	pd_fashion=pd_fashion.iloc[rndperm[:int(args.limit)]]

	# apply PCA on features data
	pca = PCA(n_components=2)
	pca_result = pca.fit_transform(pd_fashion[feat_cols].values)

	pd_fashion['pca-one'] = pca_result[:,0]
	pd_fashion['pca-two'] = pca_result[:,1]

	print('\nExplained variation per principal component: {}'.format(pca.explained_variance_ratio_))

	# display img features 2D represenation using PCA
	plt.figure(figsize=(16,10))
	pca_plot=sns.scatterplot(
		x="pca-one", y="pca-two",
		hue="clothes_labels",
		palette=sns.color_palette("Paired", 11),
		data=pd_fashion,
		legend="full",
		alpha=0.5
	).set_title('clothes features visualization by PCA')
	if args.save=="yes" or args.save=="YES":
		fig = pca_plot.get_figure()
		fig.savefig("fashion_pca.png")
	plt.show()

	# apply t-SNE on features data
	tsne = TSNE(n_components=2, verbose=1, perplexity=10, n_iter=5000)
	tsne_results = tsne.fit_transform(pd_fashion[feat_cols].values)

	pd_fashion['tsne-one'] = tsne_results[:,0]
	pd_fashion['tsne-two'] = tsne_results[:,1]

	# display img features 2D represenation using t-SNE
	plt.figure(figsize=(16,10))
	tsne_plot=sns.scatterplot(
		x="tsne-one", y="tsne-two",
		hue="clothes_labels",
		palette=sns.color_palette("Paired", 11),
		data=pd_fashion,
		legend="full",
		alpha=0.5
	).set_title('clothes features visualization by t-SNE')
	if args.save=="yes" or args.save=="YES":
		fig = tsne_plot.get_figure()
		fig.savefig("fashion_tsne.png")
	plt.show()




