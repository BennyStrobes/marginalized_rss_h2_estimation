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

def sp_inv(A, x):

    #A = A.toarray()
    N = np.shape(A)[0]

    non_zero_pos = np.where(A!=0.0)
    D = np.max(non_zero_pos[0] - non_zero_pos[1]) + 1
    #D = np.count_nonzero(A[0,:])
    ab = np.zeros((D,N))
    for i in np.arange(1,D):
        ab[i,:] = np.concatenate((np.diag(A,k=i),np.zeros(i,)),axis=None)
    ab[0,:] = np.diag(A,k=0)
    y = scipy.linalg.solveh_banded(ab,x,lower=True)
    return y


def rss_log_likelihood_single_chromosome(x, z_scores, chrom_ld_mat_reg_ref, chrom_ld_mat_reg_reg, chrom_anno, sample_size):
	# Initialize chromosome log likelihood
	chrom_log_like = 0.0

	# Compute E[\beta*\beta^T] (This is a diagonal matrix)
	beta_beta_transpose_diag = np.dot(chrom_anno, x)*sample_size

	# Compute single element of covariance
	R_beta2_RT = (chrom_ld_mat_reg_ref.dot(scipy.sparse.diags(beta_beta_transpose_diag))).dot(chrom_ld_mat_reg_ref.transpose())
	
	# Compute full covariance
	cov = R_beta2_RT + chrom_ld_mat_reg_reg
	cov = cov.toarray()

	# Invert covariance matrix
	print('start')
	pdb.set_trace()
	precision = sp_inv(cov, np.eye(cov.shape[0]))
	pdb.set_trace()


	return chrom_log_like


def rss_log_likelihood(x, trait_name, shared_input_data_dir, trait_specific_input_data_dir, sample_size):
	log_likelihood = 0.0
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
		chrom_log_likelihood = rss_log_likelihood_single_chromosome(x, z_scores, chrom_ld_mat_reg_ref, chrom_ld_mat_reg_reg, chrom_anno, sample_size)


	return log_likelihood







trait_name = sys.argv[1]
shared_input_data_dir = sys.argv[2]
trait_specific_input_data_dir = sys.argv[3]
marginalized_rss_h2_results_dir = sys.argv[4]


# Initialize tau vector
num_anno = 5
num_anno = np.load(shared_input_data_dir + 'reference_annotation.21.npy').shape[1]
tau_0 = np.zeros(num_anno)

# Get GWAS sample size
sample_size = np.loadtxt(trait_specific_input_data_dir + trait_name + '_sample_size.txt')*1.0

log_likelihood = rss_log_likelihood(tau_0, trait_name, shared_input_data_dir, trait_specific_input_data_dir, sample_size)


