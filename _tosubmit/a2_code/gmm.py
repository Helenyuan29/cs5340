import sys, os
sys.path.append('./')
sys.path.append('../')

import argparse
from utils import data_util
import numpy as np 
import math
from tqdm import tqdm


def mahalinobis_distance(x, mu, sigma):
	# return (x-mu) transpose sigma inverse (x-mu)
	sigma_inv = np.linalg.inv(sigma)
	distance = ((x - mu).dot(sigma_inv)).dot(x-mu)
	return(distance)


def conjugate_prior(mu, sigma, estimator):
	if(estimator=='mle'):
		return(1)
	################################################################
	# Uses following global variables initialized in main() 	   #
	# 	[1]. delta : mean of all data points 					   #
	# 	[2]. psi : covariance matrix of all data points 		   #
	# 	[3]. D = 3 												   #
	# 	[4]. alpha_con = 3 : The conjugate alpha 				   #
	#	[5]. gamma = 0.5 										   #
	################################################################	
	sign, log_det = np.linalg.slogdet(sigma)
	log_sigma = sign*log_det
	# log(p(theta)) is given my constant -(alpha+D+2)log(det(sigma)) -0.5[tr(pis * sigma inverse) + gamma * {(mu - delta) transpose sigma inverse (mu - delta)}]
	log_p_theta = -(alpha_con + D + 2) * log_sigma - 0.5 * np.exp( np.trace( psi.dot(np.linalg.inv(sigma)) + gamma * mahalinobis_distance(mu, delta, sigma)))
	return log_p_theta


def get_labels(data, model_1, model_2, estimator):
	sign, log_det = np.linalg.slogdet(model_1['sigma'])
	log_det_sigma1 = sign*log_det
	sign, log_det = np.linalg.slogdet(model_2['sigma'])
	log_det_sigma2 = sign*log_det

	data_labels=[]
	for i in range(data.shape[0]):
		log_p1 = -0.5 * model_1['alpha'] * ( log_det_sigma1 + mahalinobis_distance(x=data[i,:], mu=model_1['mu'], sigma=model_1['sigma']) ) * conjugate_prior(mu=model_1['mu'], sigma=model_1['sigma'], estimator=estimator)
		log_p2 = -0.5 * model_2['alpha'] * ( log_det_sigma2 + mahalinobis_distance(x=data[i,:], mu=model_2['mu'], sigma=model_2['sigma']) ) * conjugate_prior(mu=model_2['mu'], sigma=model_2['sigma'], estimator=estimator)
		# store the label (0 or 1) corresponding to maximum probability
		data_labels.append(np.argmax([log_p1, log_p2]))
	return(data_labels)


def perform_gmm(data, estimator, num_clusters=2, num_iter=500):
	################################################################
	# Run kmeans with kmeans++ initialization 					   #
	# input: data = [n_sample x 3] matrix 						   #
	# returns:  												   #
	# 	model_i_params = [alpha, mu, sigma] 					   #
	# 	alpa = scalar 											   #
	#	mu = 3-dim vector 										   #
	#	sigma = [3x3] matrix 									   #
	################################################################
	model_1, model_2 = data_util.get_initial_params(alpha=alpha_con, gamma=gamma, delta=delta, psi=psi, data=data, num_clusters=2)

	for i in tqdm(range(num_iter)):
		################################################################
		# Assignment Step (Maximization Likelihood Step) 			   #
		################################################################
		data_labels = get_labels(data=data, model_1=model_1, model_2=model_2, estimator=estimator)

		################################################################
		# Parameter Update Step 									   #
		################################################################
		#print "first call to update"		
		model_1, model_2 = data_util.update_models(alpha_con=alpha_con, gamma=gamma, delta=delta, psi=psi, data=data, labels=data_labels)
	return(model_1, model_2)


def get_mask(data, labels, out_path, img_shape):
	# fill model_1 points with ZERO and model_2 points with ONE
	for i in range(data.shape[0]):
		data[i,:] = labels[i]
	data_util.write_array_as_image(data=data, img_shape=img_shape, img_path=out_path)


def get_segments(data, labels, out_path1, out_path2, img_shape):
	segment1=[]
	segment2=[]
	for i in range(data.shape[0]):
		segment1.append(data[i,:]*labels[i])
		segment2.append(data[i,:]*(1-labels[i]))

	segment1 = np.asarray(segment1)
	segment2 = np.asarray(segment2)	
	data_util.write_array_as_image(data=segment1, img_shape=img_shape, img_path=out_path1)	
	data_util.write_array_as_image(data=segment2, img_shape=img_shape, img_path=out_path2)	


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--estimator", type=str, default='mle')
    parser.add_argument("--num_iter", type=int, default=300)    
    parser.add_argument("--image_name", type=str, default='cow.jpg')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
	current_dir = os.getcwd()	
	args = get_args()
	estimator = args.estimator
	image_name = args.image_name
	num_iter = args.num_iter
	in_path = current_dir + '/a2/' + image_name
	out_mask_path = current_dir + '/a2/' + image_name.split('.')[0] + '_mask.jpg'
	out_seg1_path = current_dir + '/a2/' + image_name.split('.')[0] + '_seg1.jpg'
	out_seg2_path = current_dir + '/a2/' + image_name.split('.')[0] + '_seg2.jpg'

	################################################################
	# Read data as [n_samples, 3] matrix 						   #
	# Normalize feature-wise for all three random variables 	   #
	################################################################
	raw_data, original_img_shape = data_util.read_image_as_array(in_path)
	data = data_util.normalize(raw_data, image_name=image_name)

	################################################################
	# Global variables 											   #
	################################################################	
	global alpha_con, gamma, delta, psi, D
	if(estimator=='map'):
		mu = np.mean(data, axis=0)
		delta = mu
		psi = (((data - mu).transpose()).dot(data - mu)) / data.shape[0]
		D=3
		alpha_con = 200
		gamma = 1
	elif(estimator=='mle'):
		mu = 0
		delta = mu
		psi = 0
		D=0
		alpha_con=0
		gamma=0

	################################################################
	# Train GMM and get updated model distribution 				   #
	################################################################
	model_1, model_2 = perform_gmm(data=data, estimator=estimator, num_iter=num_iter)
	labels = get_labels(data, model_1, model_2, estimator)

	################################################################
	# Save mask to a file 										   #
	################################################################
	get_mask(data=data, labels=labels, out_path=out_mask_path, img_shape=original_img_shape)

	################################################################
	# Save foreground and background image to file using mask	   #
	################################################################
	get_segments(data=raw_data, labels=labels, out_path1=out_seg1_path, out_path2=out_seg2_path, img_shape=original_img_shape)

	print "gmm training and evaluation complete"