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

sumstat_file = sumstat_dir + trait_name + '.sumstats'

rsid_to_info, gwas_sample_size = extract_sumstat_data(sumstat_file)

for chrom_num in range(1,23):
	regression_snp_indices, z_scores = get_valid_snp_indices_and_z_scores(shared_input_data_dir + 'regression_snp.' + str(chrom_num) + '.bim', rsid_to_info)
	# Save to output
	snp_index_output_file = trait_specific_input_data_dir + trait_name + '_valid_regression_indices_chr_' + str(chrom_num) + '.txt'
	np.savetxt(snp_index_output_file, regression_snp_indices, fmt="%s", delimiter='\n')
	z_output_file = trait_specific_input_data_dir + trait_name + '_z_scores_chr_' + str(chrom_num) + '.txt'
	np.savetxt(z_output_file, z_scores, fmt="%s", delimiter='\n')

np.savetxt(trait_specific_input_data_dir + trait_name + '_sample_size.txt', [gwas_sample_size], fmt="%s", delimiter='\n')




