##################################
# Input data
##################################
# LDSC baselineLD annotations (hg19)
ldsc_baseline_ld_hg19_annotation_dir="/n/groups/price/ldsc/reference_files/1000G_EUR_Phase3/baselineLD_v2.2/"

# LDSC 1KG genotype files (hg19)
ldsc_genotype_dir="/n/groups/price/ldsc/reference_files/1000G_EUR_Phase3/plink_files/"

# Summary statistic directory
sumstat_dir="/n/groups/price/ldsc/sumstats_formatted_2021/"

# Ldscore regression code
ldsc_code_dir="/n/groups/price/ldsc/ldsc/"

# Ldsc weights
ldsc_weights_dir="/n/groups/price/ldsc/reference_files/1000G_EUR_Phase3/weights/"



##################################
# Output data
##################################
# Output root
output_root="/n/scratch3/users/b/bes710/marginalized_rss_h2_estimation/"

# Directory containing shared marginalized_rss_h2_input data
shared_input_data_dir=$output_root"cross_trait_shared_input_data/"

# Directory containing shared marginalized_rss_h2_input data
shared_evd_input_data_dir=$output_root"cross_trait_shared_evd_input_data/"

# Directory containing trait specific input data
trait_specific_input_data_dir=$output_root"trait_specific_input_data/"

# Directory containing trait specific input data
trait_specific_evd_input_data_dir=$output_root"trait_specific_evd_input_data/"

# Directory containing heritability results
marginalized_rss_h2_results_dir=$output_root"marginalized_rss_heritabilities/"

# Directory containing heritability results
marginalized_evd_rss_h2_results_dir=$output_root"marginalized_evd_rss_heritabilities/"

# LDSC results dir
ldsc_results=$output_root"ldsc_results/"

# LDSC results dir
ldsc_evd_results=$output_root"ldsc_evd_results/"

# Standard sldsc resutls
standard_sldsc_results=$output_root"standard_sldsc_results/"

# h2 visualization
h2_viz_dir=$output_root"h2_viz/"


# Size of windows (In MB)
window_size="5"

# Preprocess genotype data
if false; then
for chrom_num in {1..20}
do
	sbatch generate_shared_marginalized_rss_h2_input_data_on_a_single_chromosome.sh $chrom_num $ldsc_baseline_ld_hg19_annotation_dir $ldsc_genotype_dir $shared_input_data_dir $window_size
done
fi

trait_name="UKB_460K.blood_WHITE_COUNT"
window_size="5"
if false; then
sh run_marginalized_rss_h2_regression.sh $trait_name $shared_input_data_dir $trait_specific_input_data_dir $marginalized_rss_h2_results_dir $sumstat_dir $window_size
fi







# Preprocess genotype data for evd rss
if false; then
for chrom_num in {1..22}
do
	sbatch generate_shared_marginalized_evd_rss_h2_input_data_on_a_single_chromosome.sh $chrom_num $ldsc_baseline_ld_hg19_annotation_dir $ldsc_genotype_dir $shared_evd_input_data_dir $window_size
done
fi

trait_name="UKB_460K.blood_WHITE_COUNT"
window_size="10"
if false; then
sh run_marginalized_evd_rss_h2_regression.sh $trait_name $shared_evd_input_data_dir $trait_specific_evd_input_data_dir $marginalized_evd_rss_h2_results_dir $sumstat_dir $window_size
fi


trait_name="UKB_460K.blood_WHITE_COUNT"
window_size="5"
if false; then
sbatch run_block_ld_score_regression.sh $trait_name $window_size $shared_input_data_dir $shared_evd_input_data_dir $ldsc_results $ldsc_evd_results $sumstat_dir
fi



if false; then
sh run_ldsc.sh $trait_name $sumstat_dir $ldsc_code_dir $ldsc_baseline_ld_hg19_annotation_dir $ldsc_weights_dir $ldsc_genotype_dir $standard_sldsc_results
fi

if false; then
source ~/.bash_profile
module load R/3.5.1
fi
if false; then
Rscript visualize_partitioned_h2_estimates.R $ldsc_results $ldsc_evd_results $standard_sldsc_results $h2_viz_dir
fi


