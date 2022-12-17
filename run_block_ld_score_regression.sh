#!/bin/bash
#SBATCH -c 1                               # Request one core
#SBATCH -t 0-10:00                         # Runtime in D-HH:MM format
#SBATCH -p short                           # Partition to run in
#SBATCH --mem=50G                         # Memory total in MiB (for all cores)



trait_name="$1"
window_size="$2"
shared_input_data_dir="$3"
shared_evd_input_data_dir="$4"
ldsc_results="$5"
ldsc_evd_results="$6"
sumstat_dir="$7"

if false; then
python3 prepare_ldsc_input_data_for_a_trait.py $trait_name $window_size $shared_input_data_dir $ldsc_results $sumstat_dir
fi

chi_sq_file=$ldsc_results$trait_name"_"$window_size"_mb_windows_chi_sq_stats.txt"
ld_score_file=$ldsc_results$trait_name"_"$window_size"_mb_windows_ld_scores.txt"
regression_weights_file=$ldsc_results$trait_name"_"$window_size"_mb_windows_regression_weights.txt"
window_names_file=$ldsc_results$trait_name"_"$window_size"_mb_windows_window_names.txt"
sample_size_file=$ldsc_results$trait_name"_"$window_size"_mb_windows_samp_size.txt"
output_root=$ldsc_results$trait_name"_"$window_size"_mb_windows_ldsc_results"
if false; then
python3 run_block_ld_score_regression.py $trait_name $chi_sq_file $ld_score_file $regression_weights_file $window_names_file $sample_size_file $output_root
fi


if false; then
python3 prepare_evd_ldsc_input_data_for_a_trait.py $trait_name $window_size $shared_evd_input_data_dir $ldsc_evd_results $sumstat_dir
fi


chi_sq_file=$ldsc_evd_results$trait_name"_"$window_size"_mb_windows_chi_sq_stats.txt"
ld_score_file=$ldsc_evd_results$trait_name"_"$window_size"_mb_windows_ld_scores.txt"
regression_weights_file=$ldsc_evd_results$trait_name"_"$window_size"_mb_windows_regression_weights.txt"
window_names_file=$ldsc_evd_results$trait_name"_"$window_size"_mb_windows_window_names.txt"
sample_size_file=$ldsc_evd_results$trait_name"_"$window_size"_mb_windows_samp_size.txt"
output_root=$ldsc_evd_results$trait_name"_"$window_size"_mb_windows_ldsc_results"
if false; then
python3 run_block_ld_score_regression.py $trait_name $chi_sq_file $ld_score_file $regression_weights_file $window_names_file $sample_size_file $output_root
fi
