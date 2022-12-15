import numpy as np 
import os
import sys
import pdb
import scipy.sparse


def extract_sumstat_data(sumstat_file):
	f = open(sumstat_file)
	z_scores = []
	variant_ids = []
	variant_alleles = []
	sumstat_sample_sizes = []
	head_count = 0
	rsids = {}
	for line in f:
		line = line.rstrip()
		data = line.split('\t')
		if head_count == 0:
			head_count = head_count + 1
			continue

		rsid = data[0]
		if rsid in rsids:
			print('assumption error')
			pdb.set_trace()
		rsids[rsid] = (float(data[5]), data[1])
		sumstat_sample_sizes.append(float(data[3]))

	f.close()
	unique_sample_sizes = np.unique(sumstat_sample_sizes)
	if len(unique_sample_sizes) != 1:
		print('assumptino eroor')
		pdb.set_trace()
	sample_size = unique_sample_sizes[0]
	return rsids, sample_size

def get_valid_snp_indices_and_z_scores(geno_bim, rsid_to_info):
	valid_snp_indices = []
	z_scores = []
	f = open(geno_bim)
	total_count = 0
	counter = 0
	for line in f:
		total_count = total_count + 1
		line = line.rstrip()
		data = line.split('\t')
		if data[1] in rsid_to_info:
			valid_snp_indices.append(counter)
			if rsid_to_info[data[1]][1] == data[4]:
				z_scores.append(rsid_to_info[data[1]][0])
			elif rsid_to_info[data[1]][1] == data[5]:
				z_scores.append(-rsid_to_info[data[1]][0])
			else:
				print('assumption eroror!')

		counter = counter + 1


	f.close()

	return np.asarray(valid_snp_indices), np.asarray(z_scores)


trait_name = sys.argv[1]
window_size = sys.argv[2]
shared_input_data_dir = sys.argv[3]
ldsc_results_dir = sys.argv[4]
sumstat_dir = sys.argv[5]



# Extract trait-specific summary stat data
sumstat_file = sumstat_dir + trait_name + '.sumstats'
rsid_to_info, gwas_sample_size = extract_sumstat_data(sumstat_file)

chi_sq = []
ld_scores = []
window_names = []

# Loop through chromosomes
for chrom_num in range(1,23):
	# Get chromosome window file
	chrom_window_file = shared_input_data_dir + 'genomic_' + window_size + '_mb_windows_chrom_' + str(chrom_num) + '.txt'

	# loop through windows on this chromosome
	f = open(chrom_window_file)
	head_count = 0
	for line in f:
		line = line.rstrip()
		data = line.split('\t')
		if head_count == 0:
			head_count = head_count + 1
			continue
		# Get window info
		window_name = data[0]
		chrom_num = data[1]
		print(window_name)


		# Extract indices corresponding to regression snps
		regression_snp_indices, z_scores = get_valid_snp_indices_and_z_scores(shared_input_data_dir + 'regression_snp.' + window_size + '_mb_windows_' + window_name + '.bim', rsid_to_info)


		# Load in evd weights amtrix
		evd_weights = np.load(shared_input_data_dir + 'ld_mat_evd_regression_snp_weights_chr_' + window_size + '_mb_windows_' + window_name + '.npy')
		# Filter to weights from observed regression snps
		evd_weights = evd_weights[:, regression_snp_indices]

		# Get evd z scores
		evd_z_scores = np.dot(evd_weights, z_scores)


		# Premade output file paths
		ld_mat_reg_ref_file = shared_input_data_dir + 'ld_mat_evd_regression_reference_chr_' + window_size + '_mb_windows_' + window_name + '.npy'
		ld_mat_reg_ref = np.load(ld_mat_reg_ref_file)


		# Premade output file paths
		annotation_file = shared_input_data_dir + 'reference_annotation.' + window_size + '_mb_windows_' + window_name + '.npy'
		anno = np.load(annotation_file)

		# Create annotation weighted ld scores
		squared_ld_reg_ref = np.square(ld_mat_reg_ref)
		# Adjusted r-squared
		#squared_ld_reg_ref[squared_ld_reg_ref!=0.0] = squared_ld_reg_ref[squared_ld_reg_ref!=0.0] - ((1.0 - squared_ld_reg_ref[squared_ld_reg_ref!=0.0])/(487.0))
		# Create annotation weighted ld scores
		anno_weighted_ld_scores = np.dot(squared_ld_reg_ref, anno)

		# Get chi-squared statistics
		chi_squared_stats = np.square(evd_z_scores)

		chi_sq.append(chi_squared_stats)
		ld_scores.append(anno_weighted_ld_scores)
		window_names.append(np.asarray([window_name]*len(chi_squared_stats)))

	f.close()


# Put all into matrix format
chi_sq = np.hstack(chi_sq)
ld_scores = np.vstack(ld_scores)
window_names = np.hstack(window_names)


# Remove outlier points
max_chi_sq_val = np.max((80.0, .001*gwas_sample_size))
valid_indices = chi_sq <= max_chi_sq_val
chi_sq = chi_sq[valid_indices]
ld_scores = ld_scores[valid_indices, :]
window_names = window_names[valid_indices]

# Heteroskedasticity weights
mean_chi_sq = np.mean(chi_sq)
sum_ld_scores = np.sum(ld_scores,axis=1)
mean_sum_ld_scores = np.mean(sum_ld_scores)
tau_hat = (mean_chi_sq - 1.0)/(gwas_sample_size*mean_sum_ld_scores)
het_var = np.square(1.0 + (gwas_sample_size*tau_hat*sum_ld_scores))

# Regression weights
regression_weights = 1.0/(het_var)

# Save to output
chi_sq_output = ldsc_results_dir + trait_name + '_' + window_size + '_mb_windows_chi_sq_stats.txt'
np.savetxt(chi_sq_output, chi_sq, fmt="%s", delimiter='\t')

regression_weights_output = ldsc_results_dir + trait_name + '_' + window_size + '_mb_windows_regression_weights.txt'
np.savetxt(regression_weights_output, regression_weights, fmt="%s", delimiter='\t')

ld_scores_output = ldsc_results_dir + trait_name + '_' + window_size + '_mb_windows_ld_scores.txt'
np.savetxt(ld_scores_output, ld_scores, fmt="%s", delimiter='\t')

window_output = ldsc_results_dir + trait_name + '_' + window_size + '_mb_windows_window_names.txt'
np.savetxt(window_output, window_names, fmt="%s", delimiter='\t')

samp_size_output = ldsc_results_dir + trait_name + '_' + window_size + '_mb_windows_samp_size.txt'
np.savetxt(samp_size_output, [gwas_sample_size], fmt="%s", delimiter='\t')




