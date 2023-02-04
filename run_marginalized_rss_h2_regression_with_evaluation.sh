#!/bin/bash
#SBATCH -c 1                               # Request one core
#SBATCH -t 0-20:00                         # Runtime in D-HH:MM format
#SBATCH -p medium                           # Partition to run in
#SBATCH --mem=50G                         # Memory total in MiB (for all cores)





trait_name="$1"
trait_data_summary_file="$2"
model_type="$3"
held_out_chromosome="$4"
learn_intercept="$5"
marginalized_rss_h2_results_dir="$6"
batch_size="$7"


echo $model_type
echo $batch_size

module load gcc/6.2.0
module load python/3.6.0
source /n/groups/price/ben/environments/tensor_flow_cpu/bin/activate



python3 run_marginalized_rss_h2_regression_with_evaluation.py $trait_name $trait_data_summary_file $model_type $held_out_chromosome $learn_intercept $marginalized_rss_h2_results_dir $batch_size
