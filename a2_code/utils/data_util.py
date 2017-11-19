from PIL import Image
import os
import matplotlib.pyplot as plt
import matplotlib.image as img
import numpy as np
from sklearn.cluster import KMeans
import scipy.misc
from PIL import Image

def write_array_as_image(data, img_shape, img_path):
	image_array = np.reshape(data, img_shape)
	#im = Image.fromarray(data_reshaped)
	#im.save(img_path)
	#scipy.misc.imsave(img_path, data_as_image)
	scipy.misc.toimage(image_array, cmin=0.0).save(img_path)


def read_image_as_array(data_path):
	image = Image.open(data_path)
	data = np.asarray(image, dtype=np.float)
	row, col, channel = data.shape
	out = np.reshape(data, (row*col, channel))
	return(out, data.shape)


def normalize(data, image_name):

	if(image_name=='zebra.jpg'):
		# for feature wise normalization
		feat_norm = np.linalg.norm(data, axis=0)
		data_norm =  data / feat_norm
		return data_norm

	# for channel-wise normalization
	channel_norm = np.linalg.norm(data, axis=1)
	data_normed = []
	for i in range(data.shape[0]):
		if channel_norm[i]==0:
			data_normed.append(data[i,:])
		else:
			data_normed.append(data[i,:]/channel_norm[i])
	data_normed = np.asarray(data_normed)
	return(data_normed)


def get_initial_params(alpha, gamma, delta, psi, data, num_clusters):
	kmeans = KMeans(n_clusters=num_clusters, init='k-means++', n_init=10, max_iter=1500, tol=0.0001,
		precompute_distances='auto', verbose=0, random_state=None, copy_x=True, n_jobs=-1, algorithm='auto')

	kmeans.fit(data)
	labels = kmeans.labels_
	return(update_models(alpha, gamma, delta, psi, data, labels))


def update_models(alpha_con, gamma, delta, psi, data, labels):
	# split data according to its label. Evaluating model parameters will be easy this way
	data1 = [data[idx,:] for idx, label in enumerate(labels) if label==0]
	data1 = np.asarray(data1)
	data2 = [data[idx,:] for idx, label in enumerate(labels) if label==1]
	data2 = np.asarray(data2)

	# alpha, mu and sigma for gaussian model 1
	model_1={}
	alpha = (1.0*data1.shape[0]) / (data1.shape[0] + data2.shape[0])
	mu = (np.sum(data1, axis=0) + gamma*delta) / (data1.shape[0] + gamma)
	sigma = (((data1 - mu).transpose()).dot(data1 - mu) + gamma * np.outer(mu-delta, mu-delta) + 2*psi) / (data1.shape[0] + 2*alpha_con)

	model_1['alpha'] = alpha
	model_1['mu'] = mu
	model_1['sigma'] = sigma

	# alpha, mu and sigma for gaussian model 2
	model_2={}
	alpha = (1.0*data2.shape[0]) / (data1.shape[0] + data2.shape[0])
	mu = (np.sum(data2, axis=0) + gamma*delta) / (data2.shape[0] + gamma)
	sigma = (((data2 - mu).transpose()).dot(data2 - mu) + gamma * np.outer(mu-delta, mu-delta) + 2*psi) / (data2.shape[0] + 2*alpha_con)

	model_2['alpha'] = alpha
	model_2['mu'] = mu
	model_2['sigma'] = sigma
	return(model_1, model_2)	