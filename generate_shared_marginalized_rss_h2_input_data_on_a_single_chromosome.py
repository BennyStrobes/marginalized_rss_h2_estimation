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
    A_mA = A - np.mean(A,axis=1)[:,None]
    B_mB = B - np.mean(B,axis=1)[:,None]
   
    # Sum of squares across rows
    ssA = np.sum(np.square(A_mA), axis=1)
    ssB = np.sum(np.square(B_mB), axis=1)


    corr_coef = np.dot(A_mA, np.transpose(B_mB)) / np.sqrt(np.dot(ssA[:, None],ssB[None]))

    #A_mA = A - A.mean(1)[:, None]
    #B_mB = B - B.mean(1)[:, None]

    # Sum of squares across rows
    #ssA = (A_mA**2).sum(1)
    #ssB = (B_mB**2).sum(1)

    # Finally get corr coeff
    #return np.dot(A_mA, B_mB.T) / np.sqrt(np.dot(ssA[:, None],ssB[None]))

    return corr_coef

def create_ld_matrix_no_distance_filter(G_obj, row_snp_indices, column_snp_indices):
	snp_pos = np.asarray(G_obj.pos)
	G_t = np.float64(np.transpose(G_obj.values))
	#G =G_obj.values

	ld_mat = scipy.sparse.lil_matrix((len(row_snp_indices), len(column_snp_indices)))

	for row_iter, row_snp_index in enumerate(row_snp_indices):
		valid_columns = np.abs(snp_pos[row_snp_index] - snp_pos[column_snp_indices]) > -2.0 # Silly hack to get all columns
		corrz = corr2_coeff(G_t[row_snp_index,None], G_t[column_snp_indices[valid_columns],:])[0,:]
		corrz = np.around(corrz, decimals=8)
		#corrz2 = corr2_coeff(G_t[column_snp_indices[valid_columns],:], G_t[row_snp_index,None])[:,0]
		#corrz2 = np.around(corrz2, decimals=5)
		ld_mat[row_iter, valid_columns] = corrz

	ld_mat2 = ld_mat.tocsr(copy=False)


	return ld_mat2

def create_ld_matrix(G_obj, row_snp_indices, column_snp_indices):
	snp_pos = np.asarray(G_obj.pos)
	G_t = np.float64(np.transpose(G_obj.values))
	#G =G_obj.values

	ld_mat = scipy.sparse.lil_matrix((len(row_snp_indices), len(column_snp_indices)))

	for row_iter, row_snp_index in enumerate(row_snp_indices):
		valid_columns = np.abs(snp_pos[row_snp_index] - snp_pos[column_snp_indices]) <= 1000000.0
		corrz = corr2_coeff(G_t[row_snp_index,None], G_t[column_snp_indices[valid_columns],:])[0,:]
		corrz = np.around(corrz, decimals=8)
		#corrz2 = corr2_coeff(G_t[column_snp_indices[valid_columns],:], G_t[row_snp_index,None])[:,0]
		#corrz2 = np.around(corrz2, decimals=5)
		ld_mat[row_iter, valid_columns] = corrz

	ld_mat2 = ld_mat.tocsr(copy=False)


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

def print_reference_annotations(window_annos, reference_annotation_output):
	# Put into df
	window_annos = np.asarray(window_annos)
	np.save(reference_annotation_output, window_annos)


def load_in_genomic_annotations(ldsc_input_annotation_file):
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
		snp_counter = snp_counter + 1
		anno = np.asarray(data[4:]).astype(float)
		annos.append(anno)
	f.close()
	# Put into df
	annos = np.asarray(annos)
	return annos



def window_overlaps_specific_region(chrom_num, window_flank_left_start, window_flank_right_end, region_chrom, region_start, region_end):
	overlap_bool = False
	if chrom_num == str(region_chrom):
		if window_flank_left_start >= region_start and window_flank_left_start <= region_end:
			overlap_bool = True
		if window_flank_right_end >= region_start and window_flank_right_end <= region_end:
			overlap_bool = True	
			
		if region_start >= window_flank_left_start and region_start <= window_flank_right_end:
			overlap_bool = True	
		if region_end >= window_flank_left_start and region_end <= window_flank_right_end:
			overlap_bool = True	

	return overlap_bool


def get_and_print_genomic_windows(G_obj, window_size, window_file, regression_snp_indices, chrom_num):
	# load in snp positions on this chromosome
	snp_positions = np.asarray(np.asarray(G_obj.pos))
	chrom_start = np.min(snp_positions)
	chrom_end = np.max(snp_positions)

	# Initialize output window file
	t = open(window_file,'w')
	t.write('window_name\tchr\twindow_flank_left_start\twindow_middle_start\twindow_flank_right_start\twindow_flank_right_end\tnum_regression_snps\tnum_reference_snps\n')

	# Get current window position for while loop
	window_start = chrom_start

	# Term that is 1MB
	one_mb = 1000000.0

	# Loop through to create windows
	while window_start < chrom_end:
		window_flank_left_start = window_start
		window_middle_start = window_start + one_mb
		window_flank_right_start = window_start + (window_size-1)*one_mb
		window_flank_right_end = window_start + (window_size)*one_mb

		# If last window (make sure it has full right window flank)
		if window_flank_right_end > chrom_end:
			window_flank_right_end = chrom_end + 1
			window_flank_right_start = window_flank_right_end - one_mb

		# Update for next iteration
		window_start = window_start + (window_size)*one_mb

		# Checks to skip windows
		#Remove window if middle (b/w window_middle_start and window_flank_right_start) is non-existent (can only occur in last window)
		if window_flank_right_start <= window_middle_start:
			continue
		# Remove window if no regression snps
		num_regression_snps = sum((snp_positions[regression_snp_indices] >= window_middle_start) & (snp_positions[regression_snp_indices] < window_flank_right_start))
		num_reference_snps = sum((snp_positions >= window_flank_left_start) & (snp_positions < window_flank_right_end))
		if num_reference_snps <= 0:
			print('skip because no reference snps ' + str(chrom_num) + '\t' + str(window_flank_left_start))
			continue
		if num_regression_snps <= 0:
			print('skip because no regression snps ' + str(chrom_num) + '\t' + str(window_flank_left_start))
			continue
		if window_overlaps_specific_region(chrom_num, window_flank_left_start, window_flank_right_end, 6, 25500000, 33500000):
			print('skip because overlaps specified region ' + str(chrom_num) + '\t' + str(window_flank_left_start))
			continue
		if window_overlaps_specific_region(chrom_num, window_flank_left_start, window_flank_right_end, 8, 8000000, 12000000):
			print('skip because overlaps specified region ' + str(chrom_num) + '\t' + str(window_flank_left_start))
			continue
		if window_overlaps_specific_region(chrom_num, window_flank_left_start, window_flank_right_end, 11, 46000000, 57000000):
			print('skip because overlaps specified region ' + str(chrom_num) + '\t' + str(window_flank_left_start))
			continue

		# Name of window
		window_name = 'window_' + str(chrom_num) + '_' + str(int(window_flank_left_start))
		# Print to output
		t.write(window_name + '\t' + str(chrom_num) + '\t' + str(window_flank_left_start) + '\t' + str(window_middle_start) + '\t' + str(window_flank_right_start) + '\t' + str(window_flank_right_end) + '\t' + str(num_regression_snps) + '\t' + str(num_reference_snps) + '\n')

	t.close()


chrom_num = sys.argv[1]
ldsc_annotation_dir = sys.argv[2]
ldsc_genotype_dir = sys.argv[3]
shared_input_data_dir = sys.argv[4]
window_size = float(sys.argv[5])


# Load in regression snps (ie hapmap3 snps)
regression_snp_file = ldsc_genotype_dir + '1000G.EUR.QC.hm3_noMHC.' + chrom_num + '.bim'
#regression_snps = load_in_regression_snps_random_subset(regression_snp_file)
regression_snps = load_in_regression_snps(regression_snp_file)

# Load in PLINK DATA
geno_stem = ldsc_genotype_dir + '1000G.EUR.QC.' + chrom_num + '.'
G_obj = read_plink1_bin(geno_stem + 'bed', geno_stem + 'bim', geno_stem + 'fam', verbose=False)

# Load in genomic annotations
input_annotation_file = ldsc_annotation_dir + 'baselineLD.' + chrom_num + '.annot.gz'
geno_anno = load_in_genomic_annotations(input_annotation_file)

# Quick error check
if len(G_obj.pos) != geno_anno.shape[0]:
	print('assumption eroror')
	pdb.set_trace()

# get snp indices corresponding to regression snps
global_regression_snp_indices = get_indices_corresponding_to_regression_snps(np.asarray(G_obj.snp), regression_snps)

# Get and print windows on this chromosome
window_file = shared_input_data_dir + 'genomic_' + str(int(window_size)) + '_mb_windows_chrom_' + chrom_num + '.txt'
get_and_print_genomic_windows(G_obj, window_size, window_file, global_regression_snp_indices, chrom_num)

# Snp positions
snp_positions = np.asarray(G_obj.pos)

# reference snp indices
reference_snp_indices = np.arange(len(snp_positions))

# Now loop through windows from this chromosome to produce output files
head_count = 0
f = open(window_file)
for line in f:
	line = line.rstrip()
	data = line.split('\t')
	# Skip header
	if head_count == 0:
		head_count = head_count + 1
		continue

	# Extract relevent info for this window
	window_name = data[0]
	window_chrom = data[1]
	window_flank_left_start = float(data[2])
	window_middle_start = float(data[3])
	window_flank_right_start = float(data[4])
	window_flank_right_end = float(data[5])
	num_regression_snps = int(data[6])
	num_reference_snps = int(data[7])
	print(window_name)

	# Need to get window regression snp indices and window reference snp indices
	window_reference_snp_indices = reference_snp_indices[(snp_positions >= window_flank_left_start) & (snp_positions < window_flank_right_end)]
	window_regression_snp_indices = global_regression_snp_indices[(snp_positions[global_regression_snp_indices] >= window_middle_start) & (snp_positions[global_regression_snp_indices] < window_flank_right_start)]

	# Print reference snp indices bim
	reference_bim_output = shared_input_data_dir + 'reference_snp.' + str(int(window_size)) + '_mb_windows_' + window_name + '.bim'
	print_bim_for_set_of_snps(G_obj, window_reference_snp_indices, reference_bim_output)

	# Print regression snp indices bim
	regression_bim_output = shared_input_data_dir + 'regression_snp.'+ str(int(window_size)) + '_mb_windows_' +  window_name + '.bim'
	print_bim_for_set_of_snps(G_obj, window_regression_snp_indices, regression_bim_output)

	# Print reference snp annotations
	reference_annotation_output = shared_input_data_dir + 'reference_annotation.'+ str(int(window_size)) + '_mb_windows_' + window_name + '.npy'
	print_reference_annotations(geno_anno[window_reference_snp_indices,:], reference_annotation_output)

	# Extract sparse LD matrix of dimension regression snps by regression snps
	ld_mat_reg_reg = create_ld_matrix(G_obj, window_regression_snp_indices, window_regression_snp_indices)
	# Save to output
	output_ld_mat_reg_reg = shared_input_data_dir + 'ld_mat_regression_regression_chr_' + str(int(window_size)) + '_mb_windows_' + window_name + '.npz'
	scipy.sparse.save_npz(output_ld_mat_reg_reg, ld_mat_reg_reg, compressed=True)

	# Extract sparse LD matrix of dimension regression snps by reference snps
	ld_mat_reg_ref = create_ld_matrix(G_obj, window_regression_snp_indices, window_reference_snp_indices)
	# Save to output
	output_ld_mat_reg_ref = shared_input_data_dir + 'ld_mat_regression_reference_chr_' + str(int(window_size)) + '_mb_windows_' + window_name + '.npz'
	scipy.sparse.save_npz(output_ld_mat_reg_ref, ld_mat_reg_ref, compressed=True)

	# Extract sparse LD matrix of dimension regression snps by regression snps
	ld_mat_reg_reg_no_distance_filter = create_ld_matrix_no_distance_filter(G_obj, window_regression_snp_indices, window_regression_snp_indices)
	# Save to output
	output_ld_mat_reg_reg_no_dist = shared_input_data_dir + 'ld_mat_regression_regression_chr_' + str(int(window_size)) + '_mb_windows_' + window_name + 'no_distance_filter.npy'
	np.save(output_ld_mat_reg_reg_no_dist, ld_mat_reg_reg_no_distance_filter.toarray())

	# Extract sparse LD matrix of dimension regression snps by reference snps
	ld_mat_reg_ref_no_distance_filter = create_ld_matrix_no_distance_filter(G_obj, window_regression_snp_indices, window_reference_snp_indices)
	# Save to output
	output_ld_mat_reg_ref_no_dist = shared_input_data_dir + 'ld_mat_regression_reference_chr_' + str(int(window_size)) + '_mb_windows_' + window_name + '_no_distance_filter.npy'
	np.save(output_ld_mat_reg_ref_no_dist, ld_mat_reg_ref_no_distance_filter.toarray())


f.close()


