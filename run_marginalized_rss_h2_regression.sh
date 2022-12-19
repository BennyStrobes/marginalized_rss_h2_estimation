#!/bin/bash
#SBATCH -c 1                               # Request one core
#SBATCH -t 0-10:00                         # Runtime in D-HH:MM format
#SBATCH -p short                           # Partition to run in
#SBATCH --mem=50G                         # Memory total in MiB (for all cores)






trait_name="$1"
shared_input_data_dir="$2"
trait_specific_input_data_dir="$3"
marginalized_rss_h2_results_dir="$4"
sumstat_dir="$5"


if false; then
source ~/.bash_profile
# Preprocess data a little bit (quick)
python3 generate_trait_specific_marginalized_rss_h2_input_data.py $trait_name $shared_input_data_dir $trait_specific_input_data_dir $marginalized_rss_h2_results_dir $sumstat_dir
fi

module load gcc/6.2.0
module load python/3.6.0
source /n/groups/price/ben/environments/tensor_flow_cpu/bin/activate
# Run regression
window_size="3"
trait_data_summary_file=$trait_specific_input_data_dir$trait_name"_"$window_size"_mb_window_summary.txt"
trait_data_summary_file=$trait_specific_input_data_dir$trait_name"_quasi_independent_ld_blocks_window_summary.txt"

python3 run_marginalized_rss_h2_regression.py $trait_name $trait_data_summary_file $marginalized_rss_h2_results_dir
