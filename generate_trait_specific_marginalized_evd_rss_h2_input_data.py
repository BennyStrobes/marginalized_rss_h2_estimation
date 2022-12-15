import numpy as np 
import os
import sys
import pdb


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
shared_input_data_dir = sys.argv[2]
trait_specific_input_data_dir = sys.argv[3]
marginalized_rss_h2_results_dir = sys.argv[4]
sumstat_dir = sys.argv[5]
window_size = sys.argv[6]


# Output file to summary trait data (one line per window)
trait_data_summary_file = trait_specific_input_data_dir + trait_name + '_' + str(window_size) + '_mb_window_summary.txt'
t = open(trait_data_summary_file,'w')
t.write('window_name\tchrom_num\tannotation_file\tld_mat_reg_ref\tz_score_file\tsample_size\n')


# Extract trait-specific summary stat data
sumstat_file = sumstat_dir + trait_name + '.sumstats'
rsid_to_info, gwas_sample_size = extract_sumstat_data(sumstat_file)

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


		# Extract indices corresponding to regression snps
		regression_snp_indices, z_scores = get_valid_snp_indices_and_z_scores(shared_input_data_dir + 'regression_snp.' + window_size + '_mb_windows_' + window_name + '.bim', rsid_to_info)

		# Load in evd weights amtrix
		evd_weights = np.load(shared_input_data_dir + 'ld_mat_evd_regression_snp_weights_chr_' + window_size + '_mb_windows_' + window_name + '.npy')
		# Filter to weights from observed regression snps
		evd_weights = evd_weights[:, regression_snp_indices]

		# Get evd z scores
		evd_z_scores = np.dot(evd_weights, z_scores)

		# Save to output
		snp_index_output_file = trait_specific_input_data_dir + trait_name + '_valid_regression_indices_' + window_size + '_mb_windows_' + window_name + '.txt'
		np.savetxt(snp_index_output_file, regression_snp_indices, fmt="%s", delimiter='\n')
		z_output_file = trait_specific_input_data_dir + trait_name + '_z_scores_' + window_size + '_mb_windows_' + window_name + '.txt'
		np.savetxt(z_output_file, evd_z_scores, fmt="%s", delimiter='\n')

		# Premade output file paths
		annotation_file = shared_input_data_dir + 'reference_annotation.' + window_size + '_mb_windows_' + window_name + '.npy'
		ld_mat_reg_ref_file = shared_input_data_dir + 'ld_mat_evd_regression_reference_chr_' + window_size + '_mb_windows_' + window_name + '.npy'
		# Print to output
		t.write(window_name + '\t' + chrom_num + '\t' + annotation_file + '\t' + ld_mat_reg_ref_file + '\t' + z_output_file + '\t' + str(gwas_sample_size) + '\n')

	f.close()
t.close()
