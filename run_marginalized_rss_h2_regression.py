import sys
sys.path.remove('/n/app/python/3.6.0/lib/python3.6/site-packages')
import os
import pdb
import numpy as np
import pickle
import time
import scipy.sparse
import scipy.optimize
import tensorflow as tf
import tensorflow_recommenders as tfrs


def get_matrix_in_banded_form(A):
	N = np.shape(A)[0]
	non_zero_pos = np.where(A!=0.0)
	D = np.max(non_zero_pos[0] - non_zero_pos[1])
	ab = np.zeros(((2*D+1),N))

	# Upper diag
	row_index = D + 1
	for i in np.arange(1,D+1):
		#ab[row_index,:] = np.concatenate((np.zeros(i,),(np.diag(A,k=i))),axis=None)
		ab[row_index,:] = np.concatenate(((np.diag(A,k=i), np.zeros(i,))),axis=None)
		row_index = row_index + 1

	# Diag
	ab[D,:] = np.diag(A,k=0)
	
	# Lower diag
	row_index = D-1
	for i in np.arange(1,D+1):
		ab[row_index,:] = np.concatenate(((np.zeros(i,), np.diag(A,k=-i))),axis=None)
		row_index = row_index - 1
	if row_index != -1.0:
		print('assumption erroro')
		pdb.set_trace()

	return D, D, ab



def sp_inv(A, identity_mat):
	lower_band, upper_band, ab = get_matrix_in_banded_form(A)
	try:
		y = scipy.linalg.solve_banded((lower_band, upper_band), ab, identity_mat)
	except:
		print('banded inverse failed')
		y = np.linalg.inv(A)
	return y

def rss_neg_log_likelihood_single_chromosome(x, z_scores, chrom_ld_mat_reg_ref, chrom_ld_mat_reg_reg, chrom_anno, sample_size, chrom_num):

	# Compute E[\beta*\beta^T] (This is a diagonal matrix)
	beta_beta_transpose_diag = np.dot(chrom_anno, x)*sample_size

	# Compute single element of covariance
	R_beta2_RT = (chrom_ld_mat_reg_ref.dot(scipy.sparse.diags(beta_beta_transpose_diag))).dot(chrom_ld_mat_reg_ref.transpose())
	

	# Compute full covariance
	cov = R_beta2_RT + chrom_ld_mat_reg_reg
	cov = cov.toarray() 

	# Invert covariance matrix
	precision = sp_inv(cov, np.eye(cov.shape[0]))

	# Compute log determinant
	neg_log_det_info = np.linalg.slogdet(cov)
	neg_log_det = neg_log_det_info[0]*neg_log_det_info[1]

	# Compute log likelihood on this chromosome
	chrom_log_like = (.5)*neg_log_det - (.5)*np.dot(np.dot(z_scores, precision), z_scores)


	return -chrom_log_like


def rss_neg_log_likelihood_on_single_chromosome(x, z_scores, chrom_ld_mat_reg_ref_s, chrom_ld_mat_reg_reg_s, chrom_anno, sample_size):

	# Compute E[\beta*\beta^T] (This is a diagonal matrix)
	beta_beta_transpose_diag = np.dot(chrom_anno, x)*sample_size



	chrom_ld_mat_reg_ref = chrom_ld_mat_reg_ref_s.toarray()
	chrom_ld_mat_reg_reg = chrom_ld_mat_reg_reg_s.toarray()



	# Compute single element of covariance
	#R_beta2_RT_old = (chrom_ld_mat_reg_ref_s.dot(scipy.sparse.diags(beta_beta_transpose_diag))).dot(chrom_ld_mat_reg_ref_s.transpose())
	R_beta2_RT = np.dot(np.multiply(chrom_ld_mat_reg_ref, beta_beta_transpose_diag), np.transpose(chrom_ld_mat_reg_ref))

	# Compute full covariance
	cov = R_beta2_RT + chrom_ld_mat_reg_reg + np.eye(chrom_ld_mat_reg_reg.shape[0])*.8

	# Invert covariance matrix
	#precision = sp_inv(cov, np.eye(cov.shape[0]))
	precision = np.linalg.inv(cov)


	# Compute log determinant
	neg_log_det_info = np.linalg.slogdet(precision)
	neg_log_det = neg_log_det_info[0]*neg_log_det_info[1]

	if neg_log_det_info[0] != 1.0:
		print('assumption eroror')

	# Compute log likelihood on this chromosome
	chrom_log_like = (.5)*neg_log_det - (.5)*np.dot(np.dot(z_scores, precision), z_scores)

	return -chrom_log_like

def rss_w_intercept_neg_log_likelihood_on_single_chromosome(x, z_scores, chrom_ld_mat_reg_ref, chrom_ld_mat_reg_reg, chrom_anno, sample_size):
	#print(x)
	# Seperate intercept from taus
	x_intercept = x[0]
	x_anno = x[1:]

	# Compute E[\beta*\beta^T] (This is a diagonal matrix)
	beta_beta_transpose_diag = np.dot(chrom_anno, x_anno)*sample_size


	# Compute single element of covariance
	R_beta2_RT = np.dot(np.multiply(chrom_ld_mat_reg_ref, beta_beta_transpose_diag), np.transpose(chrom_ld_mat_reg_ref))

	# Compute full covariance
	cov = R_beta2_RT + chrom_ld_mat_reg_reg + np.eye(chrom_ld_mat_reg_reg.shape[0])*x_intercept

	# Invert covariance matrix
	#precision = sp_inv(cov, np.eye(cov.shape[0]))
	precision = np.linalg.inv(cov)

	# Compute log determinant
	neg_log_det_info = np.linalg.slogdet(precision)
	if neg_log_det_info[0] != 1.0:
		print('assumption eroror')
		pdb.set_trace()
	neg_log_det = neg_log_det_info[0]*neg_log_det_info[1]

	# Compute log likelihood on this chromosome
	chrom_log_like = -(.5)*neg_log_det + (.5)*np.dot(np.dot(z_scores, precision), z_scores)

	return chrom_log_like

def rss_w_intercept_softplus_neg_log_likelihood_and_gradient_on_single_chromosome(x, z_scores, chrom_ld_mat_reg_ref, chrom_ld_mat_reg_reg, chrom_anno, sample_size):
	# Seperate intercept from taus
	x_intercept = x[0]
	x_anno = x[1:]

	# Compute E[\beta*\beta^T] (This is a diagonal matrix)
	anno_prod = np.dot(chrom_anno, x_anno)
	anno_prod_exp = np.exp(anno_prod)
	beta_beta_transpose_diag = softplus_np(anno_prod)*sample_size

	#print(beta_beta_transpose_diag/sample_size)


	# Compute single element of covariance
	R_beta2_RT = np.dot(np.multiply(chrom_ld_mat_reg_ref, beta_beta_transpose_diag), np.transpose(chrom_ld_mat_reg_ref))

	# Compute full covariance
	cov = R_beta2_RT + chrom_ld_mat_reg_reg + np.eye(chrom_ld_mat_reg_reg.shape[0])*softplus_np(x_intercept)

	# Invert covariance matrix
	#precision = sp_inv(cov, np.eye(cov.shape[0]))
	precision = np.linalg.inv(cov)

	# Compute log determinant
	neg_log_det_info = np.linalg.slogdet(precision)
	if neg_log_det_info[0] != 1.0:
		print('assumption eroror')
		pdb.set_trace()
	neg_log_det = neg_log_det_info[0]*neg_log_det_info[1]

	# Compute log likelihood on this chromosome
	chrom_log_like = -(.5)*neg_log_det + (.5)*np.dot(np.dot(z_scores, precision), z_scores)


	# Gradient term from inverse term
	inv_temp_term = np.dot(np.transpose(chrom_ld_mat_reg_ref), np.dot(precision,z_scores))
	gradient_inv_term = -sample_size*.5*np.dot(np.transpose(chrom_anno), (anno_prod_exp*inv_temp_term*inv_temp_term)/(anno_prod_exp + 1.0))


	# Gradient term from log-determinant term	
	temp_term1 = np.dot(np.transpose(chrom_ld_mat_reg_ref), precision)
	gradient_log_det_term = np.asarray(sample_size*.5*np.dot(np.transpose(chrom_anno), (anno_prod_exp/(anno_prod_exp+1.0))*np.sum(np.multiply(temp_term1, np.transpose(chrom_ld_mat_reg_ref)),axis=1))).flatten()

	# Full gradient is sum of two gradient terms
	tau_gradient = gradient_inv_term + gradient_log_det_term

	# Gradient of intercept term
	intercept_gradient_inv_term = -.5*np.dot(z_scores, np.dot(precision, np.dot(precision, z_scores)))
	intercept_gradient_log_det_term = .5*np.trace(precision)
	intercept_gradient = (intercept_gradient_log_det_term + intercept_gradient_inv_term)*np.exp(x_intercept)/(np.exp(x_intercept) + 1.0)

	gradient = np.hstack(([intercept_gradient], tau_gradient))

	return chrom_log_like, gradient

def rss_w_intercept_softplus_neg_log_likelihood_on_single_chromosome(x, z_scores, chrom_ld_mat_reg_ref, chrom_ld_mat_reg_reg, chrom_anno, sample_size):
	# Seperate intercept from taus
	x_intercept = x[0]
	x_anno = x[1:]

	# Compute E[\beta*\beta^T] (This is a diagonal matrix)
	anno_prod = np.dot(chrom_anno, x_anno)
	anno_prod_exp = np.exp(anno_prod)
	beta_beta_transpose_diag = softplus_np(anno_prod)*sample_size


	# Compute single element of covariance
	R_beta2_RT = np.dot(np.multiply(chrom_ld_mat_reg_ref, beta_beta_transpose_diag), np.transpose(chrom_ld_mat_reg_ref))

	# Compute full covariance
	cov = R_beta2_RT + chrom_ld_mat_reg_reg + np.eye(chrom_ld_mat_reg_reg.shape[0])*softplus_np(x_intercept)

	# Invert covariance matrix
	#precision = sp_inv(cov, np.eye(cov.shape[0]))
	precision = np.linalg.inv(cov)

	# Compute log determinant
	neg_log_det_info = np.linalg.slogdet(precision)
	if neg_log_det_info[0] != 1.0:
		print('assumption eroror')
		pdb.set_trace()
	neg_log_det = neg_log_det_info[0]*neg_log_det_info[1]

	# Compute log likelihood on this chromosome
	chrom_log_like = -(.5)*neg_log_det + (.5)*np.dot(np.dot(z_scores, precision), z_scores)

	return chrom_log_like






def rss_w_intercept_neg_log_likelihood_and_gradient_on_single_chromosome(x, z_scores, chrom_ld_mat_reg_ref, chrom_ld_mat_reg_reg, chrom_anno, sample_size):
	#print(x)
	# Seperate intercept from taus
	x_intercept = x[0]
	x_anno = x[1:]

	# Compute E[\beta*\beta^T] (This is a diagonal matrix)
	beta_beta_transpose_diag = np.dot(chrom_anno, x_anno)*sample_size


	# Compute single element of covariance
	R_beta2_RT = np.dot(np.multiply(chrom_ld_mat_reg_ref, beta_beta_transpose_diag), np.transpose(chrom_ld_mat_reg_ref))

	# Compute full covariance
	cov = R_beta2_RT + chrom_ld_mat_reg_reg + np.eye(chrom_ld_mat_reg_reg.shape[0])*x_intercept

	# Invert covariance matrix
	#precision = sp_inv(cov, np.eye(cov.shape[0]))
	precision = np.linalg.inv(cov)

	# Compute log determinant
	neg_log_det_info = np.linalg.slogdet(precision)
	if neg_log_det_info[0] != 1.0:
		print('assumption eroror')
		pdb.set_trace()
	neg_log_det = neg_log_det_info[0]*neg_log_det_info[1]

	# Compute log likelihood on this chromosome
	chrom_log_like = -(.5)*neg_log_det + (.5)*np.dot(np.dot(z_scores, precision), z_scores)


	# Gradient term from inverse term
	#inv_temp_term = chrom_ld_mat_reg_ref.transpose().dot(np.dot(precision,z_scores))
	inv_temp_term = np.dot(np.transpose(chrom_ld_mat_reg_ref), np.dot(precision,z_scores))
	gradient_inv_term = -sample_size*.5*np.dot(np.transpose(chrom_anno), inv_temp_term*inv_temp_term)

	####
	# Sanity check based on https://www.matrixcalculus.org/
	#A = chrom_anno*sample_size
	#x = np.copy(x_anno)
	#B = chrom_ld_mat_reg_reg + np.eye(chrom_ld_mat_reg_reg.shape[0])*x_intercept
	#R = (chrom_ld_mat_reg_ref)
	#t_0 = (A).dot(x)
	#e = .5
	#z = z_scores
	# inv term
	#t_1 = (np.linalg.inv((B + (R).dot((t_0[:, np.newaxis] * R.T))))).dot(z)
	#functionValue = (e * (z).dot(t_1))
	#gradient = -(e * (A.T).dot(((R.T).dot(t_1) * (R.T).dot((np.linalg.inv((B.T + ((R * t_0[np.newaxis, :])).dot(R.T)))).dot(z)))))
    # Det term
    #functionValue = -(e * (np.linalg.slogdet(np.linalg.inv((B + (R).dot((np.dot(A,x)[:, np.newaxis] * R.T))))))[1])
    #gradient_pre = (e * np.diag(((R.T).dot(np.linalg.inv((B + ((R * np.dot(A,x)[np.newaxis, :])).dot(R.T))))).dot(R)))
 	#gradient=np.dot(np.transpose(A), gradient_pre)

	# Gradient term from log-determinant term	
	temp_term1 = np.dot(np.transpose(chrom_ld_mat_reg_ref), precision)

	gradient_log_det_term = np.asarray(sample_size*.5*np.dot(np.transpose(chrom_anno), np.sum(np.multiply(temp_term1, np.transpose(chrom_ld_mat_reg_ref)),axis=1))).flatten()

	# Full gradient is sum of two gradient terms
	tau_gradient = gradient_inv_term + gradient_log_det_term

	# Gradient of intercept term
	intercept_gradient_inv_term = -.5*np.dot(z_scores, np.dot(precision, np.dot(precision, z_scores)))
	intercept_gradient_log_det_term = .5*np.trace(precision)
	intercept_gradient = intercept_gradient_log_det_term + intercept_gradient_inv_term

	gradient = np.hstack(([intercept_gradient], tau_gradient))

	return chrom_log_like, gradient



def rss_w_intercept_genotype_intercept_neg_log_likelihood_and_gradient_on_single_chromosome(x, z_scores, chrom_ld_mat_reg_ref, chrom_ld_mat_reg_reg, chrom_anno, sample_size):
	#print(x)
	# Seperate intercept from taus
	x_intercept = x[0]
	x_anno = x[1]

	# Compute E[\beta*\beta^T] (This is a diagonal matrix)
	#beta_beta_transpose_diag = np.dot(chrom_anno, x_anno)*sample_size


	# Compute single element of covariance
	#R_beta2_RT = np.dot(np.multiply(chrom_ld_mat_reg_ref, beta_beta_transpose_diag), np.transpose(chrom_ld_mat_reg_ref))
	R_RT = np.dot(chrom_ld_mat_reg_ref, np.transpose(chrom_ld_mat_reg_ref))
	R_beta2_RT = x_anno*sample_size*R_RT

	# Compute full covariance
	cov = R_beta2_RT + chrom_ld_mat_reg_reg + np.eye(chrom_ld_mat_reg_reg.shape[0])*x_intercept

	# Invert covariance matrix
	#precision = sp_inv(cov, np.eye(cov.shape[0]))
	precision = np.linalg.inv(cov)

	# Compute log determinant
	neg_log_det_info = np.linalg.slogdet(precision)
	if neg_log_det_info[0] != 1.0:
		print('assumption eroror')
		pdb.set_trace()
	neg_log_det = neg_log_det_info[0]*neg_log_det_info[1]

	# Compute log likelihood on this chromosome
	chrom_log_like = -(.5)*neg_log_det + (.5)*np.dot(np.dot(z_scores, precision), z_scores)


	# Gradient term from inverse term
	#inv_temp_term = np.dot(np.transpose(chrom_ld_mat_reg_ref), np.dot(precision,z_scores))
	#gradient_inv_term = -sample_size*.5*np.dot(np.transpose(chrom_anno), inv_temp_term*inv_temp_term)
	gradient_inv_term = -.5*sample_size*np.dot(np.dot(np.dot(np.dot(z_scores, precision), R_RT), precision), z_scores)


	# Gradient term from log-determinant term	
	gradient_log_det_term = sample_size*.5*np.trace(np.dot(R_RT,precision))

	# Full gradient is sum of two gradient terms
	tau_gradient = gradient_inv_term + gradient_log_det_term

	# Gradient of intercept term
	intercept_gradient = -.5*np.dot(z_scores, np.dot(precision, np.dot(precision, z_scores)))

	gradient = np.hstack(([intercept_gradient], [tau_gradient]))

	return chrom_log_like, gradient



def rss_neg_log_likelihood_and_gradient_on_single_chromosome(x, z_scores, chrom_ld_mat_reg_ref, chrom_ld_mat_reg_reg, chrom_anno, sample_size):

	# Compute E[\beta*\beta^T] (This is a diagonal matrix)
	beta_beta_transpose_diag = np.dot(chrom_anno, x)*sample_size

	print(x)



	#chrom_ld_mat_reg_ref = chrom_ld_mat_reg_ref_s.toarray()
	#chrom_ld_mat_reg_reg = chrom_ld_mat_reg_reg_s.toarray()



	# Compute single element of covariance
	#R_beta2_RT_old = (chrom_ld_mat_reg_ref_s.dot(scipy.sparse.diags(beta_beta_transpose_diag))).dot(chrom_ld_mat_reg_ref_s.transpose())
	R_beta2_RT = np.dot(np.multiply(chrom_ld_mat_reg_ref, beta_beta_transpose_diag), np.transpose(chrom_ld_mat_reg_ref))

	# Compute full covariance
	cov = R_beta2_RT + chrom_ld_mat_reg_reg + np.eye(chrom_ld_mat_reg_reg.shape[0])*.2

	# Invert covariance matrix
	#precision = sp_inv(cov, np.eye(cov.shape[0]))
	precision = np.linalg.inv(cov)

	# Compute log determinant
	neg_log_det_info = np.linalg.slogdet(precision)
	if neg_log_det_info[0] != 1.0:
		print('assumption eroror')
		pdb.set_trace()
	neg_log_det = neg_log_det_info[0]*neg_log_det_info[1]

	# Compute log likelihood on this chromosome
	chrom_log_like = (.5)*neg_log_det - (.5)*np.dot(np.dot(z_scores, precision), z_scores)


	# Gradient term from inverse term
	#inv_temp_term = chrom_ld_mat_reg_ref.transpose().dot(np.dot(precision,z_scores))
	inv_temp_term = np.dot(np.transpose(chrom_ld_mat_reg_ref), np.dot(precision,z_scores))
	gradient_inv_term = -sample_size*.5*np.dot(np.transpose(chrom_anno), inv_temp_term*inv_temp_term)

	# Gradient term from log-determinant term	
	temp_term1 = np.dot(np.transpose(chrom_ld_mat_reg_ref), precision)

	gradient_log_det_term = np.asarray(sample_size*.5*np.dot(np.transpose(chrom_anno), np.sum(np.multiply(temp_term1, np.transpose(chrom_ld_mat_reg_ref)),axis=1))).flatten()

	# Full gradient is some of two gradient terms
	gradient = gradient_inv_term + gradient_log_det_term

	#print(gradient)


	return -chrom_log_like, gradient

def rss_neg_log_likelihood_and_gradient_on_single_chromosome_old(x, z_scores, chrom_ld_mat_reg_ref, chrom_ld_mat_reg_reg, chrom_anno, sample_size):

	# Compute E[\beta*\beta^T] (This is a diagonal matrix)
	beta_beta_transpose_diag = np.dot(chrom_anno, x)*sample_size

	print(beta_beta_transpose_diag/sample_size)

	# Compute single element of covariance
	R_beta2_RT = (chrom_ld_mat_reg_ref.dot(scipy.sparse.diags(beta_beta_transpose_diag))).dot(chrom_ld_mat_reg_ref.transpose())

	# Compute full covariance
	cov = R_beta2_RT + chrom_ld_mat_reg_reg
	cov = cov.toarray() 

	# Invert covariance matrix
	#precision = sp_inv(cov, np.eye(cov.shape[0]))
	precision = np.linalg.inv(cov)

	# Compute log determinant
	neg_log_det_info = np.linalg.slogdet(precision)
	neg_log_det = neg_log_det_info[0]*neg_log_det_info[1]

	# Compute log likelihood on this chromosome
	chrom_log_like = (.5)*neg_log_det - (.5)*np.dot(np.dot(z_scores, precision), z_scores)

	# Gradient term from inverse term
	inv_temp_term = chrom_ld_mat_reg_ref.transpose().dot(np.dot(precision,z_scores))
	gradient_inv_term = -sample_size*.5*np.dot(np.transpose(chrom_anno), inv_temp_term*inv_temp_term)	

	# Gradient term from log-determinant term
	temp_term2 = (chrom_ld_mat_reg_ref.transpose())
	temp_term1 = ((chrom_ld_mat_reg_ref.transpose()).dot(scipy.sparse.csr_matrix(precision)))
	gradient_log_det_term = np.asarray(sample_size*.5*np.dot(np.transpose(chrom_anno), np.sum(temp_term1.multiply(temp_term2),axis=1))).flatten()

	# Full gradient is some of two gradient terms
	gradient = gradient_inv_term + gradient_log_det_term



	return -chrom_log_like, gradient


def compute_inv_neg_log_like(x, z_scores, chrom_ld_mat_reg_ref, chrom_ld_mat_reg_reg, chrom_anno, sample_size, chrom_num):
	# Compute E[\beta*\beta^T] (This is a diagonal matrix)
	beta_beta_transpose_diag = np.dot(chrom_anno, x)*sample_size

	# Compute single element of covariance
	R_beta2_RT = (chrom_ld_mat_reg_ref.dot(scipy.sparse.diags(beta_beta_transpose_diag))).dot(chrom_ld_mat_reg_ref.transpose())
	

	# Compute full covariance
	cov = R_beta2_RT + chrom_ld_mat_reg_reg
	cov = cov.toarray() 

	# Invert covariance matrix
	#precision = sp_inv(cov, np.eye(cov.shape[0]))
	precision = np.linalg.inv(cov)

	# Compute log likelihood on this chromosome
	neg_log_like = (.5)*np.dot(np.dot(z_scores, precision), z_scores)
	return neg_log_like

def compute_log_det_neg_log_like(x, z_scores, chrom_ld_mat_reg_ref, chrom_ld_mat_reg_reg, chrom_anno, sample_size, chrom_num):
	# Compute E[\beta*\beta^T] (This is a diagonal matrix)
	beta_beta_transpose_diag = np.dot(chrom_anno, x)*sample_size

	# Compute single element of covariance
	R_beta2_RT = (chrom_ld_mat_reg_ref.dot(scipy.sparse.diags(beta_beta_transpose_diag))).dot(chrom_ld_mat_reg_ref.transpose())
	

	# Compute full covariance
	cov = R_beta2_RT + chrom_ld_mat_reg_reg
	cov = cov.toarray() 

	# Invert covariance matrix
	precision = sp_inv(cov, np.eye(cov.shape[0]))

	# Compute log determinant
	neg_log_det_info = np.linalg.slogdet(cov)
	neg_log_det = neg_log_det_info[0]*neg_log_det_info[1]

	return -(.5)*neg_log_det

def compute_inv_neg_log_like_grad(x, z_scores, chrom_ld_mat_reg_ref, chrom_ld_mat_reg_reg, chrom_anno, sample_size, chrom_num):
	# Compute E[\beta*\beta^T] (This is a diagonal matrix)
	beta_beta_transpose_diag = np.dot(chrom_anno, x)*sample_size

	# Compute single element of covariance
	R_beta2_RT = (chrom_ld_mat_reg_ref.dot(scipy.sparse.diags(beta_beta_transpose_diag))).dot(chrom_ld_mat_reg_ref.transpose())
	

	# Compute full covariance
	cov = R_beta2_RT + chrom_ld_mat_reg_reg
	cov = cov.toarray() 

	# Invert covariance matrix
	#precision = sp_inv(cov, np.eye(cov.shape[0]))
	precision = np.linalg.inv(cov)


	term_a = chrom_ld_mat_reg_ref.transpose().dot(np.dot(precision,z_scores))



	gradient = -sample_size*.5*np.dot(np.transpose(chrom_anno), term_a*term_a)

	return gradient


def compute_log_det_neg_log_like_grad(x, z_scores, chrom_ld_mat_reg_ref, chrom_ld_mat_reg_reg, chrom_anno, sample_size, chrom_num):
	# Compute E[\beta*\beta^T] (This is a diagonal matrix)
	beta_beta_transpose_diag = np.dot(chrom_anno, x)*sample_size

	# Compute single element of covariance
	R_beta2_RT = (chrom_ld_mat_reg_ref.dot(scipy.sparse.diags(beta_beta_transpose_diag))).dot(chrom_ld_mat_reg_ref.transpose())
	

	# Compute full covariance
	cov = R_beta2_RT + chrom_ld_mat_reg_reg
	cov = cov.toarray() 

	# Invert covariance matrix
	precision = sp_inv(cov, np.eye(cov.shape[0]))

	# Terms involved in gradient
	term2 = (chrom_ld_mat_reg_ref.transpose())
	term1 = ((chrom_ld_mat_reg_ref.transpose()).dot(scipy.sparse.csr_matrix(precision)))
	#term2 = (chrom_ld_mat_reg_ref.transpose()).toarray()

	#gradient = -sample_size*.5*np.dot(np.transpose(chrom_anno), np.sum(term1*term2,axis=1))
	gradient = -sample_size*.5*np.dot(np.transpose(chrom_anno), np.sum(term1.multiply(term2),axis=1))

	return np.asarray(gradient).flatten()


def rss_neg_log_likelihood(x, trait_name, shared_input_data_dir, trait_specific_input_data_dir, sample_size):
	log_likelihood = 0.0
	chrom_log_likes = []
	for chrom_num in range(1,23):
		# Extract data on this chromosome
		# Annotation file
		chrom_anno = np.load(shared_input_data_dir + 'reference_annotation.' + str(chrom_num) + '.npy')
		# LD mat reg ref
		ld_mat_reg_ref_file = shared_input_data_dir + 'ld_mat_regression_reference_chr_' + str(chrom_num) + '.npz'
		chrom_ld_mat_reg_ref = scipy.sparse.load_npz(ld_mat_reg_ref_file)
		# LD mat reg reg
		ld_mat_reg_reg_file = shared_input_data_dir + 'ld_mat_regression_regression_chr_' + str(chrom_num) + '.npz'
		chrom_ld_mat_reg_reg = scipy.sparse.load_npz(ld_mat_reg_reg_file)
		# Z scores
		z_score_file = trait_specific_input_data_dir + trait_name + '_z_scores_chr_' + str(chrom_num) + '.txt'
		z_scores = np.loadtxt(z_score_file)
		# Valid regression indices
		valid_regression_indices_file = trait_specific_input_data_dir + trait_name + '_valid_regression_indices_chr_' + str(chrom_num) + '.txt'
		valid_regression_indices = np.loadtxt(valid_regression_indices_file).astype(int)

		# Fillter regression indices to those we have z-scores for
		chrom_ld_mat_reg_ref = chrom_ld_mat_reg_ref[valid_regression_indices,:]
		chrom_ld_mat_reg_reg = chrom_ld_mat_reg_reg[valid_regression_indices,:][:, valid_regression_indices]

		# Compute rss log likelihood on this chromsome
		chrom_log_likelihood = rss_log_likelihood_single_chromosome(x, z_scores, chrom_ld_mat_reg_ref, chrom_ld_mat_reg_reg, chrom_anno, sample_size, chrom_num)
		chrom_log_likes.append(chrom_log_likelihood)

		log_likelihood = log_likelihood + chrom_log_likelihood

	print(np.asarray(chrom_log_likes))

	return -log_likelihood



def run_adam_optimization(x, trait_name, trait_data_summary):
	optimizer = tf.keras.optimizers.Adam()
	#optimizer = tf.keras.optimizers.SGD()
	x_tf = tf.Variable(x, dtype=tf.float64)

	# Loop through windows
	num_windows = trait_data_summary.shape[0]
	for epoch_iter in range(30):
		total_log_like = 0.0
		counter = 0 
		for window_iter in np.random.permutation(range(num_windows)):
			# Extract data on this window
			# Annotation file
			chrom_anno = np.load(trait_data_summary[window_iter, 2])
			chrom_anno = chrom_anno[:,:len(x[1:])]
			#chrom_anno = chrom_anno[:,0].reshape((chrom_anno.shape[0],1))
			# LD mat reg ref
			ld_mat_reg_ref_file = trait_data_summary[window_iter, 3]
			chrom_ld_mat_reg_ref = np.load(ld_mat_reg_ref_file)
			# LD mat reg reg
			ld_mat_reg_reg_file = trait_data_summary[window_iter, 4]
			chrom_ld_mat_reg_reg = np.load(ld_mat_reg_reg_file)
			# Z scores
			z_score_file = trait_data_summary[window_iter, 5]
			z_scores = np.loadtxt(z_score_file)


			# Valid regression indices
			valid_regression_indices_file = trait_data_summary[window_iter, 6]
			valid_regression_indices = np.loadtxt(valid_regression_indices_file).astype(int)

			# Get sample size
			sample_size = float(trait_data_summary[window_iter, 7])

			if chrom_ld_mat_reg_reg.shape[0] == 1:
				continue

			# Fillter regression indices to those we have z-scores for
			chrom_ld_mat_reg_ref = chrom_ld_mat_reg_ref[valid_regression_indices,:]
			chrom_ld_mat_reg_reg = chrom_ld_mat_reg_reg[valid_regression_indices,:][:, valid_regression_indices]

			chrom_neg_likelihood, chrom_gradient = rss_w_intercept_softplus_neg_log_likelihood_and_gradient_on_single_chromosome(x, z_scores, chrom_ld_mat_reg_ref, chrom_ld_mat_reg_reg, chrom_anno, sample_size)

			tf_grad = tf.constant(chrom_gradient)

			optimizer.apply_gradients(zip([tf_grad], [x_tf]))

			x = np.copy(x_tf)
			total_log_like = total_log_like + chrom_neg_likelihood
			print(softplus_np(x))
		print(softplus_np(x))
		print(total_log_like)

def extract_non_colinear_regression_snp_indices(chrom_ld_mat_reg_ref, abs_correlation_thresh):
	corr = np.corrcoef(chrom_ld_mat_reg_ref)
	corr = corr - np.eye(corr.shape[0])
	colinear_pairs = np.where(np.abs(corr) > abs_correlation_thresh)
	n_colinear_pairs = len(colinear_pairs[0])
	discarded = {}
	for colinear_pair_index in range(n_colinear_pairs):
		pair_a = colinear_pairs[0][colinear_pair_index]
		pair_b = colinear_pairs[1][colinear_pair_index]
		if pair_a in discarded:
			continue
		discarded[pair_b] = 1.0
	n_snps = corr.shape[0]
	non_colinear_indices = []
	for snp_iter in range(n_snps):
		if snp_iter in discarded:
			continue
		non_colinear_indices.append(snp_iter)
	non_colinear_indices = np.asarray(non_colinear_indices)
	return non_colinear_indices

def softplus_np(x): return np.log(np.exp(x) + 1)

def softplus_inv_np(x): return np.log(np.exp(x) - 1.0)


def rss_w_intercept_softplus_neg_log_likelihood_and_grad(x, trait_name, trait_data_summary):
	# Initialize likelihood and gradient
	neg_log_likelihood = 0.0
	gradient = np.zeros(len(x))
	print(softplus_np(x))

	# Loop through windows
	num_windows = trait_data_summary.shape[0]
	for window_iter in range(num_windows):
		# Extract data on this window
		# Annotation file
		chrom_anno = np.load(trait_data_summary[window_iter, 2])
		chrom_anno = chrom_anno[:,:len(x[1:])]
		#chrom_anno = chrom_anno[:,0].reshape((chrom_anno.shape[0],1))
		# LD mat reg ref
		ld_mat_reg_ref_file = trait_data_summary[window_iter, 3]
		#chrom_ld_mat_reg_ref = scipy.sparse.load_npz(ld_mat_reg_ref_file)
		chrom_ld_mat_reg_ref = np.load(ld_mat_reg_ref_file)
		# LD mat reg reg
		ld_mat_reg_reg_file = trait_data_summary[window_iter, 4]
		#chrom_ld_mat_reg_reg = scipy.sparse.load_npz(ld_mat_reg_reg_file)
		chrom_ld_mat_reg_reg = np.load(ld_mat_reg_reg_file)
		# Z scores
		z_score_file = trait_data_summary[window_iter, 5]
		z_scores = np.loadtxt(z_score_file)

		# Valid regression indices
		valid_regression_indices_file = trait_data_summary[window_iter, 6]
		valid_regression_indices = np.loadtxt(valid_regression_indices_file).astype(int)

		# Get sample size
		sample_size = float(trait_data_summary[window_iter, 7])

		if chrom_ld_mat_reg_reg.shape[0] == 1:
			continue

		# Fillter regression indices to those we have z-scores for
		chrom_ld_mat_reg_ref = chrom_ld_mat_reg_ref[valid_regression_indices,:]
		chrom_ld_mat_reg_reg = chrom_ld_mat_reg_reg[valid_regression_indices,:][:, valid_regression_indices]



		chrom_neg_likelihood, chrom_gradient = rss_w_intercept_softplus_neg_log_likelihood_and_gradient_on_single_chromosome(x, z_scores, chrom_ld_mat_reg_ref, chrom_ld_mat_reg_reg, chrom_anno, sample_size)
		#eps = np.sqrt(np.finfo(float).eps)
		#chrom_gradient_approx = scipy.optimize.approx_fprime(x, rss_w_intercept_softplus_neg_log_likelihood_on_single_chromosome,eps, z_scores, chrom_ld_mat_reg_ref, chrom_ld_mat_reg_reg, chrom_anno, sample_size)

		# Append to global counter
		neg_log_likelihood = neg_log_likelihood + chrom_neg_likelihood
		gradient = gradient + chrom_gradient



	return neg_log_likelihood, gradient


def rss_w_intercept_neg_log_likelihood_and_grad(x, trait_name, trait_data_summary):
	# Initialize likelihood and gradient
	neg_log_likelihood = 0.0
	gradient = np.zeros(len(x))
	print(x)

	# Loop through windows
	num_windows = trait_data_summary.shape[0]
	for window_iter in range(num_windows):
		# Extract data on this window
		# Annotation file
		chrom_anno = np.load(trait_data_summary[window_iter, 2])
		chrom_anno = chrom_anno[:,:len(x[1:])]
		#chrom_anno = chrom_anno[:,0].reshape((chrom_anno.shape[0],1))
		# LD mat reg ref
		ld_mat_reg_ref_file = trait_data_summary[window_iter, 3]
		#chrom_ld_mat_reg_ref = scipy.sparse.load_npz(ld_mat_reg_ref_file)
		chrom_ld_mat_reg_ref = np.load(ld_mat_reg_ref_file)
		# LD mat reg reg
		ld_mat_reg_reg_file = trait_data_summary[window_iter, 4]
		#chrom_ld_mat_reg_reg = scipy.sparse.load_npz(ld_mat_reg_reg_file)
		chrom_ld_mat_reg_reg = np.load(ld_mat_reg_reg_file)
		# Z scores
		z_score_file = trait_data_summary[window_iter, 5]
		z_scores = np.loadtxt(z_score_file)

		# Valid regression indices
		valid_regression_indices_file = trait_data_summary[window_iter, 6]
		valid_regression_indices = np.loadtxt(valid_regression_indices_file).astype(int)

		# Get sample size
		sample_size = float(trait_data_summary[window_iter, 7])

		if chrom_ld_mat_reg_reg.shape[0] == 1:
			continue

		# Fillter regression indices to those we have z-scores for
		chrom_ld_mat_reg_ref = chrom_ld_mat_reg_ref[valid_regression_indices,:]
		chrom_ld_mat_reg_reg = chrom_ld_mat_reg_reg[valid_regression_indices,:][:, valid_regression_indices]


		# Compute rss log likelihood on this window
		chrom_neg_likelihood, chrom_gradient = rss_w_intercept_neg_log_likelihood_and_gradient_on_single_chromosome(x, z_scores, chrom_ld_mat_reg_ref, chrom_ld_mat_reg_reg, chrom_anno, sample_size)
		#chrom_gradient_approx = scipy.optimize.approx_fprime(x, rss_w_intercept_neg_log_likelihood_on_single_chromosome,eps, z_scores, chrom_ld_mat_reg_ref, chrom_ld_mat_reg_reg, chrom_anno, sample_size)

		# Append to global counter
		neg_log_likelihood = neg_log_likelihood + chrom_neg_likelihood
		gradient = gradient + chrom_gradient



	return neg_log_likelihood, gradient

def rss_neg_log_likelihood_and_grad(x, trait_name, trait_data_summary):
	# Initialize likelihood and gradient
	neg_log_likelihood = 0.0
	gradient = np.zeros(len(x))

	# Loop through windows
	num_windows = trait_data_summary.shape[0]
	for window_iter in range(num_windows):
		# Extract data on this window
		# Annotation file
		chrom_anno = np.load(trait_data_summary[window_iter, 2])
		chrom_anno = chrom_anno[:,:len(x)]
		#chrom_anno = chrom_anno[:,0].reshape((chrom_anno.shape[0],1))
		# LD mat reg ref
		ld_mat_reg_ref_file = trait_data_summary[window_iter, 3]
		#chrom_ld_mat_reg_ref = scipy.sparse.load_npz(ld_mat_reg_ref_file)
		chrom_ld_mat_reg_ref = np.load(ld_mat_reg_ref_file)
		# LD mat reg reg
		ld_mat_reg_reg_file = trait_data_summary[window_iter, 4]
		#chrom_ld_mat_reg_reg = scipy.sparse.load_npz(ld_mat_reg_reg_file)
		chrom_ld_mat_reg_reg = np.load(ld_mat_reg_reg_file)
		# Z scores
		z_score_file = trait_data_summary[window_iter, 5]
		z_scores = np.loadtxt(z_score_file)

		# Valid regression indices
		valid_regression_indices_file = trait_data_summary[window_iter, 6]
		valid_regression_indices = np.loadtxt(valid_regression_indices_file).astype(int)

		# Get sample size
		sample_size = float(trait_data_summary[window_iter, 7])

		if chrom_ld_mat_reg_reg.shape[0] == 1:
			continue

		# Fillter regression indices to those we have z-scores for
		chrom_ld_mat_reg_ref = chrom_ld_mat_reg_ref[valid_regression_indices,:]
		chrom_ld_mat_reg_reg = chrom_ld_mat_reg_reg[valid_regression_indices,:][:, valid_regression_indices]

		# Extract non-colinear regression snps
		non_colinear_regression_snp_indices = extract_non_colinear_regression_snp_indices(chrom_ld_mat_reg_ref, .97)
		pdb.set_trace()
		chrom_ld_mat_reg_ref = chrom_ld_mat_reg_ref[non_colinear_regression_snp_indices, :]
		chrom_ld_mat_reg_reg = chrom_ld_mat_reg_reg[non_colinear_regression_snp_indices,:][:, non_colinear_regression_snp_indices]
		z_scores = z_scores[non_colinear_regression_snp_indices]

		# Compute rss log likelihood on this window
		chrom_neg_likelihood, chrom_gradient = rss_neg_log_likelihood_and_gradient_on_single_chromosome(x, z_scores, chrom_ld_mat_reg_ref, chrom_ld_mat_reg_reg, chrom_anno, sample_size)
		#chrom_neg_likelihood, chrom_gradient = rss_neg_log_likelihood_and_gradient_on_single_chromosome_old(x, z_scores, chrom_ld_mat_reg_ref, chrom_ld_mat_reg_reg, chrom_anno, sample_size)

		# Append to global counter
		neg_log_likelihood = neg_log_likelihood + chrom_neg_likelihood
		gradient = gradient + chrom_gradient


	return neg_log_likelihood, gradient

def run_sldsc_for_debugging_purposes(trait_data_summary):
	chi_sq_stats_arr = []
	ld_scores_arr = []
	weights_arr = []
	num_anno=1
	# Loop through windows
	num_windows = trait_data_summary.shape[0]
	for window_iter in range(num_windows):
		# Extract data on this window
		# Annotation file
		chrom_anno = np.load(trait_data_summary[window_iter, 2])
		chrom_anno = chrom_anno[:,:num_anno]
		#chrom_anno = chrom_anno[:,0].reshape((chrom_anno.shape[0],1))
		# LD mat reg ref
		ld_mat_reg_ref_file = trait_data_summary[window_iter, 3]
		aa = time.time()
		chrom_ld_mat_reg_ref = scipy.sparse.load_npz(ld_mat_reg_ref_file)
		bb = time.time()
		# LD mat reg reg
		ld_mat_reg_reg_file = trait_data_summary[window_iter, 4]
		cc = time.time()
		chrom_ld_mat_reg_reg = scipy.sparse.load_npz(ld_mat_reg_reg_file)
		dd = time.time()

		# Z scores
		z_score_file = trait_data_summary[window_iter, 5]
		z_scores = np.loadtxt(z_score_file)

		# Valid regression indices
		valid_regression_indices_file = trait_data_summary[window_iter, 6]
		valid_regression_indices = np.loadtxt(valid_regression_indices_file).astype(int)


		# Get sample size
		sample_size = float(trait_data_summary[window_iter, 7])

		# Fillter regression indices to those we have z-scores for
		chrom_ld_mat_reg_ref = chrom_ld_mat_reg_ref[valid_regression_indices,:]
		chrom_ld_mat_reg_reg = chrom_ld_mat_reg_reg[valid_regression_indices,:][:, valid_regression_indices]	

		chi_sq_stats = np.square(z_scores)
		ld_scores = np.sum(np.square(chrom_ld_mat_reg_ref.toarray()),axis=1)

		



def debugging(x, trait_name, shared_input_data_dir, trait_specific_input_data_dir, sample_size):
	chrom_num = 21
	# Extract data on this chromosome
	# Annotation file
	chrom_anno = np.load(shared_input_data_dir + 'reference_annotation.' + str(chrom_num) + '.npy')
	# LD mat reg ref
	ld_mat_reg_ref_file = shared_input_data_dir + 'ld_mat_regression_reference_chr_' + str(chrom_num) + '.npz'
	chrom_ld_mat_reg_ref = scipy.sparse.load_npz(ld_mat_reg_ref_file)
	# LD mat reg reg
	ld_mat_reg_reg_file = shared_input_data_dir + 'ld_mat_regression_regression_chr_' + str(chrom_num) + '.npz'
	chrom_ld_mat_reg_reg = scipy.sparse.load_npz(ld_mat_reg_reg_file)
	# Z scores
	z_score_file = trait_specific_input_data_dir + trait_name + '_z_scores_chr_' + str(chrom_num) + '.txt'
	z_scores = np.loadtxt(z_score_file)
	# Valid regression indices
	valid_regression_indices_file = trait_specific_input_data_dir + trait_name + '_valid_regression_indices_chr_' + str(chrom_num) + '.txt'
	valid_regression_indices = np.loadtxt(valid_regression_indices_file).astype(int)

	# Fillter regression indices to those we have z-scores for
	chrom_ld_mat_reg_ref = chrom_ld_mat_reg_ref[valid_regression_indices,:]
	chrom_ld_mat_reg_reg = chrom_ld_mat_reg_reg[valid_regression_indices,:][:, valid_regression_indices]

	# log determinant neg log like
	#log_det_neg_like = compute_log_det_neg_log_like(x, z_scores, chrom_ld_mat_reg_ref, chrom_ld_mat_reg_reg, chrom_anno, sample_size, chrom_num)
	#log_det_neg_like_grad = compute_log_det_neg_log_like_grad(x, z_scores, chrom_ld_mat_reg_ref, chrom_ld_mat_reg_reg, chrom_anno, sample_size, chrom_num)
	#grad2 = scipy.optimize.approx_fprime(x, compute_log_det_neg_log_like,1.4901161193847656e-08, z_scores, chrom_ld_mat_reg_ref, chrom_ld_mat_reg_reg, chrom_anno, sample_size, chrom_num)

	# Inverse term
	inv_neg_log_like_grad = compute_inv_neg_log_like_grad(x, z_scores, chrom_ld_mat_reg_ref, chrom_ld_mat_reg_reg, chrom_anno, sample_size, chrom_num)
	inv_neg_log_like = compute_inv_neg_log_like(x, z_scores, chrom_ld_mat_reg_ref, chrom_ld_mat_reg_reg, chrom_anno, sample_size, chrom_num)
	grad = scipy.optimize.approx_fprime(x, compute_inv_neg_log_like,1.4901161193847656e-08, z_scores, chrom_ld_mat_reg_ref, chrom_ld_mat_reg_reg, chrom_anno, sample_size, chrom_num)
	pdb.set_trace()

	# Full likelihood
	#full_neg_log_like = rss_neg_log_likelihood_single_chromosome(x, z_scores, chrom_ld_mat_reg_ref, chrom_ld_mat_reg_reg, chrom_anno, sample_size, chrom_num)
	#full_neg_log_like2, full_neg_log_like_grad = rss_neg_log_likelihood_and_gradient_on_single_chromosome(x, z_scores, chrom_ld_mat_reg_ref, chrom_ld_mat_reg_reg, chrom_anno, sample_size, chrom_num)
	#grad = scipy.optimize.approx_fprime(x, rss_neg_log_likelihood_single_chromosome,1.4901161193847656e-08, z_scores, chrom_ld_mat_reg_ref, chrom_ld_mat_reg_reg, chrom_anno, sample_size, chrom_num)



trait_name = sys.argv[1]
trait_data_summary_file = sys.argv[2]
marginalized_rss_h2_results_dir = sys.argv[3]


# Load in trait data summary file
trait_data_summary = np.loadtxt(trait_data_summary_file, dtype=str,delimiter='\t')[1:,:]
#trait_data_summary =trait_data_summary[trait_data_summary[:,1].astype(float) == 21,:]



################################################
# ADAM With ONLY INTERCEPT WITH SOFTPLUS LINK
################################################
# Initialize tau vector
num_anno = np.load(trait_data_summary[0,2]).shape[1] +1
num_anno = 2
tau_0 = np.zeros(num_anno) 
tau_0[1] = softplus_inv_np(1e-5)
tau_0[0] = softplus_inv_np(.05)

run_adam_optimization(tau_0, trait_name, trait_data_summary)


################################################
# LBFGS USING FULL BASELINELD WITH SOFTPLUS LINK
################################################
# Initialize tau vector
'''
num_anno = np.load(trait_data_summary[0,2]).shape[1] +1
tau_0 = np.zeros(num_anno) 
tau_0[1] = softplus_inv_np(1e-5)
tau_0[0] = softplus_inv_np(.05)


opti=scipy.optimize.fmin_l_bfgs_b(rss_w_intercept_softplus_neg_log_likelihood_and_grad, tau_0, args=(trait_name, trait_data_summary), approx_grad=False, iprint=101)
print(softplus_np(opti[0]))
print(opti[2]['warnflag'])
print(opti)
pdb.set_trace()
'''


'''
################################################
# LBFGS USING ONLY INTERCEPT WITH_o SOFTPLUS LINK
################################################
# Initialize tau vector
num_anno = np.load(trait_data_summary[0,2]).shape[1] +1
num_anno = 2
tau_0 = np.zeros(num_anno) 
tau_0[1] = 1e-5
tau_0[0] = .05

opti=scipy.optimize.fmin_l_bfgs_b(rss_w_intercept_neg_log_likelihood_and_grad, tau_0, args=(trait_name, trait_data_summary),bounds=[(1e-5,2.0), (1e-20,2.0)], approx_grad=False, iprint=101)
print(opti[0])
print(opti[2]['warnflag'])
print(opti)
pdb.set_trace()
'''










