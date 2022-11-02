import sys
sys.path.remove('/n/app/python/3.7.4-ext/lib/python3.7/site-packages')
import os
import pdb
import numpy as np
from pandas_plink import read_plink1_bin
import pickle
import time
import scipy.sparse
import gzip



def load_in_regression_snps(regression_snp_file):
	f = open(regression_snp_file)
	dicti = {}
	for line in f:
		line = line.rstrip()
		data = line.split('\t')
		rsid = data[1]
		if rsid in dicti:
			print('assumption eroror')
			pdb.set_trace()
		dicti[rsid] = 1

	f.close()

	return dicti

def load_in_regression_snps_random_subset(regression_snp_file):
	f = open(regression_snp_file)
	dicti = {}
	counter = 0
	for line in f:
		line = line.rstrip()
		data = line.split('\t')
		rsid = data[1]
		if rsid in dicti:
			print('assumption eroror')
			pdb.set_trace()
		if np.mod(counter, 30.0) == 0.0:
			dicti[rsid] = 1
		counter = counter + 1

	f.close()

	return dicti


def get_indices_corresponding_to_regression_snps(all_snps, regression_snps):
	indices = []
	
	for snp_index, snp_name in enumerate(all_snps):
		if snp_name in regression_snps:
			indices.append(snp_index)

	indices = np.asarray(indices)

	if len(indices) != len(regression_snps):
		print('ASSUMPTION ERRORO"')
		pdb.set_trace()

	return indices


def corr2_coeff(A, B):
    # Rowwise mean of input arrays & subtract from input arrays themeselves
    A_mA = A - A.mean(1)[:, None]
    B_mB = B - B.mean(1)[:, None]

    # Sum of squares across rows
    ssA = (A_mA**2).sum(1)
    ssB = (B_mB**2).sum(1)

    # Finally get corr coeff
    return np.dot(A_mA, B_mB.T) / np.sqrt(np.dot(ssA[:, None],ssB[None]))

def sp_inv(A, x):

    A = A.toarray()
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


def create_ld_matrix(G_obj, row_snp_indices, column_snp_indices):
	snp_pos = np.asarray(G_obj.pos)
	G_t = np.transpose(G_obj.values)
	G =G_obj.values

	ld_mat = scipy.sparse.lil_matrix((len(row_snp_indices), len(column_snp_indices)))

	for row_iter, row_snp_index in enumerate(row_snp_indices):
		valid_columns = np.abs(snp_pos[row_snp_index] - snp_pos[column_snp_indices]) <= 1000000.0
		corrz = corr2_coeff(G_t[row_snp_index,None], G_t[column_snp_indices[valid_columns],:])[0,:]
		ld_mat[row_iter, valid_columns] = corrz


	ld_mat2 = ld_mat.tocsr(copy=False)

	'''
	# Experimenting with sparse matrix ops
	cov = ld_mat2.dot(ld_mat2.transpose())
	time1 = time.time()
	invy = np.linalg.inv(cov.toarray())
	time2 = time.time()
	time3 = time.time()
	invy2 = sp_inv(cov, np.eye(cov.shape[0]))
	time4 = time.time()
	det_info = np.linalg.slogdet(cov.toarray())
	pdb.set_trace()
	'''


	return ld_mat2

def print_bim_for_set_of_snps(G_obj, snp_indices, output_file):
	chrom_array = np.asarray(G_obj.chrom)
	rs_id_array = np.asarray(G_obj.snp)
	cm_array = np.asarray(G_obj.cm)
	pos_array = np.asarray(G_obj.pos)
	a0_array = np.asarray(G_obj.a0)
	a1_array = np.asarray(G_obj.a1)
	t = open(output_file,'w')
	for snp_inex in snp_indices:
		t.write(chrom_array[snp_inex] + '\t' + rs_id_array[snp_inex] + '\t' + str(cm_array[snp_inex]) + '\t' + str(pos_array[snp_inex]) + '\t' + str(a0_array[snp_inex]) + '\t' + str(a1_array[snp_inex]) + '\n')
	t.close()
	return

def print_reference_annotations(snp_ids, ldsc_input_annotation_file, reference_annotation_output):
	f = gzip.open(ldsc_input_annotation_file)
	head_count = 0
	snp_counter = 0
	annos = []
	for line in f:
		if head_count == 0:
			head_count = head_count + 1
			continue
		line = line.rstrip().decode('utf-8')
		data = line.split('\t')
		rsid = data[2]
		if rsid != snp_ids[snp_counter]:
			print('assumption eroror')
			pdb.set_trace()
		snp_counter = snp_counter + 1
		anno = np.asarray(data[4:]).astype(float)
		annos.append(anno)
	f.close()
	# Put into df
	annos = np.asarray(annos)
	np.save(reference_annotation_output, annos)





chrom_num = sys.argv[1]
ldsc_annotation_dir = sys.argv[2]
ldsc_genotype_dir = sys.argv[3]
shared_input_data_dir = sys.argv[4]


# Load in regression snps (ie hapmap3 snps)
regression_snp_file = ldsc_genotype_dir + '1000G.EUR.QC.hm3_noMHC.' + chrom_num + '.bim'
regression_snps = load_in_regression_snps_random_subset(regression_snp_file)

# Load in PLINK DATA
geno_stem = ldsc_genotype_dir + '1000G.EUR.QC.' + chrom_num + '.'
G_obj = read_plink1_bin(geno_stem + 'bed', geno_stem + 'bim', geno_stem + 'fam', verbose=False)

# get snp indices corresponding to regression snps
regression_snp_indices = get_indices_corresponding_to_regression_snps(np.asarray(G_obj.snp), regression_snps)


# Print reference snp indices bim
reference_bim_output = shared_input_data_dir + 'reference_snp.' + chrom_num + '.bim'
print_bim_for_set_of_snps(G_obj, np.arange(len(G_obj.a1)), reference_bim_output)

# Print regression snp indices bim
regression_bim_output = shared_input_data_dir + 'regression_snp.' + chrom_num + '.bim'
print_bim_for_set_of_snps(G_obj, regression_snp_indices, regression_bim_output)



# Print reference snp annotations
reference_annotation_output = shared_input_data_dir + 'reference_annotation.' + chrom_num + '.npy'
input_annotation_file = ldsc_annotation_dir + 'baselineLD.' + chrom_num + '.annot.gz'
print_reference_annotations(G_obj.snp, input_annotation_file, reference_annotation_output)


# Extract sparse LD matrix of dimension regression snps by reference snps
ld_mat_reg_ref = create_ld_matrix(G_obj, regression_snp_indices, np.arange(len(G_obj.a1)))
# Save to output
output_ld_mat_reg_ref = shared_input_data_dir + 'ld_mat_regression_reference_chr_' + chrom_num + '.npz'
scipy.sparse.save_npz(output_ld_mat_reg_ref, ld_mat_reg_ref, compressed=True)


# Extract sparse LD matrix of dimension regression snps by regression snps
ld_mat_reg_reg = create_ld_matrix(G_obj, regression_snp_indices, regression_snp_indices)
# Save to output
output_ld_mat_reg_reg = shared_input_data_dir + 'ld_mat_regression_regression_chr_' + chrom_num + '.npz'
scipy.sparse.save_npz(output_ld_mat_reg_reg, ld_mat_reg_reg, compressed=True)


