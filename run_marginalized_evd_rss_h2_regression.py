import sys
sys.path.remove('/n/app/python/3.7.4-ext/lib/python3.7/site-packages')
import os
import pdb
import numpy as np
from pandas_plink import read_plink1_bin
import pickle
import time
import scipy.sparse
import scipy.optimize


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


def rss_neg_log_likelihood_and_gradient_on_single_chromosome(x, z_scores, chrom_ld_mat_reg_ref, chrom_anno, sample_size):

	# Compute E[\beta*\beta^T] (This is a diagonal matrix)
	beta_beta_transpose_diag = np.dot(chrom_anno, x)*sample_size

	print(beta_beta_transpose_diag/sample_size)


	# Compute single element of covariance
	R_beta2_RT = np.dot(np.multiply(chrom_ld_mat_reg_ref, beta_beta_transpose_diag), np.transpose(chrom_ld_mat_reg_ref))

	# Compute full covariance
	cov = R_beta2_RT + np.eye(R_beta2_RT.shape[0])

	# Invert covariance matrix
	precision = np.linalg.inv(cov)
	#precision = np.linalg.pinv(cov)

	# Compute log determinant
	neg_log_det_info = np.linalg.slogdet(precision)
	neg_log_det = neg_log_det_info[0]*neg_log_det_info[1]

	# Compute log likelihood on this chromosome
	chrom_log_like = (.5)*neg_log_det - (.5)*np.dot(np.dot(z_scores, precision), z_scores)

	# Gradient term from inverse term
	inv_temp_term = np.dot(np.transpose(chrom_ld_mat_reg_ref), np.dot(precision,z_scores))
	gradient_inv_term = -sample_size*.5*np.dot(np.transpose(chrom_anno), inv_temp_term*inv_temp_term)

	# Gradient term from log-determinant term
	temp_term2 = np.transpose(chrom_ld_mat_reg_ref)
	temp_term1 = np.dot(temp_term2, precision)
	gradient_log_det_term = np.asarray(sample_size*.5*np.dot(np.transpose(chrom_anno), np.sum(np.multiply(temp_term1, temp_term2),axis=1))).flatten()

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


def old_rss_neg_log_likelihood_and_grad_old(x, trait_name, shared_input_data_dir, trait_specific_input_data_dir, sample_size):
	neg_log_likelihood = 0.0
	gradient = np.zeros(len(x))
	chrom_log_likes = []
	#for chrom_num in range(1,23):
	for chrom_num in range(21,22):
		# Extract data on this chromosome
		# Annotation file
		chrom_anno = np.load(shared_input_data_dir + 'reference_annotation.' + str(chrom_num) + '.npy')
		chrom_anno = chrom_anno[:,0].reshape((chrom_anno.shape[0],1))
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
		chrom_neg_likelihood, chrom_gradient = rss_neg_log_likelihood_and_gradient_on_single_chromosome(x, z_scores, chrom_ld_mat_reg_ref, chrom_ld_mat_reg_reg, chrom_anno, sample_size, chrom_num)
		chrom_log_likes.append(chrom_neg_likelihood)

		neg_log_likelihood = neg_log_likelihood + chrom_neg_likelihood
		gradient = gradient + chrom_gradient

	print(np.asarray(chrom_log_likes))

	return neg_log_likelihood, gradient

def rss_neg_log_likelihood_and_grad_with_means(x, trait_name, trait_data_summary, anno_means, anno_sdevs):
	# Initialize likelihood and gradient
	neg_log_likelihood = 0.0
	gradient = np.zeros(len(x))

	# Loop through windows
	num_windows = trait_data_summary.shape[0]
	for window_iter in range(num_windows):
		# Extract data on this window
		# Annotation file
		chrom_anno = np.load(trait_data_summary[window_iter, 2])
		for itera in range(len(anno_means)):
			chrom_anno[:, itera] = (chrom_anno[:, itera] - anno_means[itera])/anno_sdevs[itera]

		#chrom_anno = chrom_anno[:,0].reshape((chrom_anno.shape[0],1))
		# LD mat reg ref
		ld_mat_reg_ref_file = trait_data_summary[window_iter, 3]
		chrom_ld_mat_reg_ref = np.load(ld_mat_reg_ref_file)
		# Z scores
		z_score_file = trait_data_summary[window_iter, 4]
		z_scores = np.loadtxt(z_score_file)

		# Get sample size
		sample_size = float(trait_data_summary[window_iter, 5])


		# Compute rss log likelihood on this window
		chrom_neg_likelihood, chrom_gradient = rss_neg_log_likelihood_and_gradient_on_single_chromosome(x, z_scores, chrom_ld_mat_reg_ref, chrom_anno, sample_size)

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
		#chrom_anno = chrom_anno[:,0].reshape((chrom_anno.shape[0],1))
		# LD mat reg ref
		ld_mat_reg_ref_file = trait_data_summary[window_iter, 3]
		chrom_ld_mat_reg_ref = np.load(ld_mat_reg_ref_file)
		# Z scores
		z_score_file = trait_data_summary[window_iter, 4]
		z_scores = np.loadtxt(z_score_file)

		# Get sample size
		sample_size = float(trait_data_summary[window_iter, 5])


		# Compute rss log likelihood on this window
		chrom_neg_likelihood, chrom_gradient = rss_neg_log_likelihood_and_gradient_on_single_chromosome(x, z_scores, chrom_ld_mat_reg_ref, chrom_anno, sample_size)

		# Append to global counter
		neg_log_likelihood = neg_log_likelihood + chrom_neg_likelihood
		gradient = gradient + chrom_gradient


	return neg_log_likelihood, gradient

def get_anno_mean_and_sdevs(trait_data_summary):
	annos = []
	# Loop through windows
	num_windows = trait_data_summary.shape[0]
	for window_iter in range(num_windows):
		# Extract data on this window
		# Annotation file
		chrom_anno = np.load(trait_data_summary[window_iter, 2])
		annos.append(chrom_anno)
	annos = np.vstack(annos)
	means = []
	sdevs = []
	means.append(0.0)
	sdevs.append(1.0)
	for anno_iter in range(1, chrom_anno.shape[1]):
		means.append(np.mean(annos[:,anno_iter]))
		sdevs.append(np.std(annos[:, anno_iter]))
	means = np.asarray(means)
	sdevs = np.asarray(sdevs)
	return means, sdevs

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
trait_data_summary = trait_data_summary[trait_data_summary[:,1].astype(float) > 20,:]
#trait_data_summary = trait_data_summary[trait_data_summary[:,1] == '21',:]

# Initialize tau vector
num_anno = np.load(trait_data_summary[0,2]).shape[1]
#num_anno = 1
tau_0 = np.zeros(num_anno) 
tau_0[0] = 1e-8


nll, grad = rss_neg_log_likelihood_and_grad(tau_0, trait_name, trait_data_summary)

#anno_means, anno_sdevs = get_anno_mean_and_sdevs(trait_data_summary)

#opti=scipy.optimize.fmin_l_bfgs_b(rss_neg_log_likelihood_and_grad, tau_0, args=(trait_name, trait_data_summary), approx_grad=False, iprint=101)
#opti=scipy.optimize.fmin_l_bfgs_b(rss_neg_log_likelihood_and_grad_with_means, tau_0, args=(trait_name, trait_data_summary, anno_means, anno_sdevs), approx_grad=False, iprint=101)

print(opti[0])
print(opti[2]['warnflag'])
print(opti)


#neg_log_likelihood, gradient = rss_neg_log_likelihood_and_grad(tau_0, trait_name, shared_input_data_dir, trait_specific_input_data_dir, sample_size)
#pdb.set_trace()

#debugging(tau_0, trait_name, shared_input_data_dir, trait_specific_input_data_dir, sample_size)