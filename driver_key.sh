##################################
# Input data
##################################
# LDSC baselineLD annotations (hg19)
ldsc_baseline_ld_hg19_annotation_dir="/n/groups/price/ldsc/reference_files/1000G_EUR_Phase3/baselineLD_v2.2/"

# LDSC 1KG genotype files (hg19)
ldsc_genotype_dir="/n/groups/price/ldsc/reference_files/1000G_EUR_Phase3/plink_files/"

# Summary statistic directory
sumstat_dir="/n/groups/price/ldsc/sumstats_formatted_2021/"




##################################
# Output data
##################################
# Output root
output_root="/n/scratch3/users/b/bes710/marginalized_rss_h2_estimation/"

# Directory containing shared marginalized_rss_h2_input data
shared_input_data_dir=$output_root"cross_trait_shared_input_data/"

# Directory containing trait specific input data
trait_specific_input_data_dir=$output_root"trait_specific_input_data/"

# Directory containing heritability results
marginalized_rss_h2_results_dir=$output_root"marginalized_rss_heritabilities/"




# Preprocess genotype data
if false; then
for chrom_num in {1..22}
do
	sbatch generate_shared_marginalized_rss_h2_input_data_on_a_single_chromosome.sh $chrom_num $ldsc_baseline_ld_hg19_annotation_dir $ldsc_genotype_dir $shared_input_data_dir
done
fi


trait_name="UKB_460K.blood_WHITE_COUNT"
if false; then
sh run_marginalized_rss_h2_regression.sh $trait_name $shared_input_data_dir $trait_specific_input_data_dir $marginalized_rss_h2_results_dir $sumstat_dir
fi