import numpy as np 
import os
import sys
import pdb
from sklearn.linear_model import LinearRegression



def run_ld_score_regression(chi_sq, ld_scores, regression_weights, sample_size):
	X = ld_scores*sample_size
	reg = LinearRegression().fit(X, chi_sq, sample_weight=regression_weights)
	return reg.coef_, reg.intercept_ - 1.0

def get_ordered_unique_window_names(window_names):
	ordered_window_names = []
	used = {}
	for window_name in window_names:
		if window_name in used:
			continue
		used[window_name] = 1
		ordered_window_names.append(window_name)
	return np.asarray(ordered_window_names)

def jacknife_estimates(bs_vec):
	num_jacknife_samples = len(bs_vec)
	bs_mean = np.mean(bs_vec)
	diff_squared = np.square(bs_vec - bs_mean)
	jacknife_var = np.sum(diff_squared)*(num_jacknife_samples-1.0)/num_jacknife_samples
	jacknife_se = np.sqrt(jacknife_var)
	return jacknife_se, bs_mean

def calculate_bootstrap_std_errors(output_root, num_bootstrap_windows):
	taus = []
	intercepts = []
	for bootstrap_iter in range(num_bootstrap_windows):
		bs_tau_file = output_root + '_tau_estimates_bs_' + str(bootstrap_iter) + '.txt'
		bs_intercept_file = output_root + '_intercept_estimates_bs_' + str(bootstrap_iter) + '.txt'
		taus.append(np.loadtxt(bs_tau_file))
		intercepts.append(np.loadtxt(bs_intercept_file))
	bs_taus = np.vstack(taus)
	bs_intercepts = np.hstack(intercepts)

	bs_intercept_se, bs_intercept_mean = jacknife_estimates(bs_intercepts)

	tau_means = np.loadtxt(output_root + '_tau_estimates.txt')

	bs_tau_means = []
	bs_tau_ses = []
	for anno_num in range(bs_taus.shape[1]):
		bs_tau_se, bs_tau_mean = jacknife_estimates(bs_taus[:, anno_num])
		bs_tau_means.append(bs_tau_mean)
		bs_tau_ses.append(bs_tau_se)
	bs_tau_means = np.asarray(bs_tau_means)
	bs_tau_ses = np.asarray(bs_tau_ses)

	bs_tau_output_file = output_root + '_tau_bootstrapped_ci.txt'
	t = open(bs_tau_output_file,'w')
	t.write('estimate\tstandard_error\n')
	for anno_num in range(bs_taus.shape[1]):
		t.write(str(bs_tau_means[anno_num]) + '\t' + str(bs_tau_ses[anno_num]) + '\n')
	t.close()


trait_name = sys.argv[1]
chi_sq_file = sys.argv[2]
ld_score_file = sys.argv[3]
regression_weights_file = sys.argv[4]
window_names_file = sys.argv[5]
sample_size_file = sys.argv[6]
output_root = sys.argv[7]


'''
print('loading in data')
# Load in data
chi_sq = np.loadtxt(chi_sq_file)
ld_scores = np.loadtxt(ld_score_file)
regression_weights = np.loadtxt(regression_weights_file)
'''
window_names = np.loadtxt(window_names_file,dtype=str)
'''
sample_size = np.loadtxt(sample_size_file)*1.0


print('running regression')
taus, intercept = run_ld_score_regression(chi_sq, ld_scores, regression_weights, sample_size)

np.savetxt(output_root + '_tau_estimates.txt', taus, fmt="%s")
np.savetxt(output_root + '_intercept_estimates.txt', [intercept], fmt="%s")

'''
ordered_unique_window_names = get_ordered_unique_window_names(window_names)
'''
for window_iter, window_name in enumerate(ordered_unique_window_names):
	print(window_iter)
	bootstrap_indices = window_names != window_name
	bs_taus, bs_intercept = run_ld_score_regression(chi_sq[bootstrap_indices], ld_scores[bootstrap_indices,:], regression_weights[bootstrap_indices], sample_size)

	np.savetxt(output_root + '_tau_estimates_bs_' + str(window_iter) + '.txt', bs_taus, fmt="%s")
	np.savetxt(output_root + '_intercept_estimates_bs_' + str(window_iter) + '.txt', [bs_intercept], fmt="%s")

'''


calculate_bootstrap_std_errors (output_root, len(ordered_unique_window_names))
