#!/bin/bash
#SBATCH -c 1                               # Request one core
#SBATCH -t 0-04:00                         # Runtime in D-HH:MM format
#SBATCH -p short                           # Partition to run in
#SBATCH --mem=20G                         # Memory total in MiB (for all cores)





chrom_num="$1"
ldsc_baseline_ld_hg19_annotation_dir="$2"
ldsc_genotype_dir="$3"
shared_input_data_dir="$4"

source ~/.bash_profile


python3 generate_shared_marginalized_rss_h2_input_data_on_a_single_chromosome.py $chrom_num $ldsc_baseline_ld_hg19_annotation_dir $ldsc_genotype_dir $shared_input_data_dir