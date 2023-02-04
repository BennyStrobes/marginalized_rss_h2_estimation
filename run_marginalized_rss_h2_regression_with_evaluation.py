import sys
sys.path.remove('/n/app/python/3.6.0/lib/python3.6/site-packages')
import os
import pdb
import numpy as np
import pickle
import time
import scipy.sparse
import scipy.optimize
import tensorflow as tf
import tensorflow_recommenders as tfrs

def init_non_linear_no_drops_mapping_from_genomic_annotations_to_gamma(annotation_data_dimension, nn_dimension, scale_boolean):
	# Initialize Neural network model
	model = tf.keras.models.Sequential()
	if scale_boolean:
		model.add(tf.keras.layers.experimental.preprocessing.Normalization())
	model.add(tf.keras.layers.Dense(units=nn_dimension, activation='relu', input_dim=annotation_data_dimension, dtype=tf.float64))
	model.add(tf.keras.layers.Dense(units=nn_dimension, activation='relu', dtype=tf.float64))
	model.add(tf.keras.layers.Dense(units=1, activation='softplus', dtype=tf.float64))

	return model

def init_linear_mapping_from_genomic_annotations_to_gamma(annotation_data_dimension):
	# Initialize Neural network model
	model = tf.keras.models.Sequential()
	model.add(tf.keras.layers.Dense(units=1, activation='softplus', input_dim=annotation_data_dimension, dtype=tf.float64))

	return model


def init_non_linear_batch_norm_mapping_from_genomic_annotations_to_gamma(annotation_data_dimension, nn_dimension, scale_boolean):
	# Initialize Neural network model
	model = tf.keras.models.Sequential()
	if scale_boolean:
		model.add(tf.keras.layers.experimental.preprocessing.Normalization())
	model.add(tf.keras.layers.Dense(units=nn_dimension, activation='relu', input_dim=annotation_data_dimension, dtype=tf.float64))
	model.add(tf.keras.layers.BatchNormalization(dtype=tf.float64))
	model.add(tf.keras.layers.Dense(units=nn_dimension, activation='relu', dtype=tf.float64))
	model.add(tf.keras.layers.BatchNormalization(dtype=tf.float64))
	model.add(tf.keras.layers.Dense(units=1, activation='softplus', dtype=tf.float64))

	return model


def init_non_linear_layer_norm_mapping_from_genomic_annotations_to_gamma(annotation_data_dimension, nn_dimension, scale_boolean):
	# Initialize Neural network model
	model = tf.keras.models.Sequential()
	if scale_boolean:
		model.add(tf.keras.layers.experimental.preprocessing.Normalization())
	model.add(tf.keras.layers.Dense(units=nn_dimension, activation='relu', input_dim=annotation_data_dimension, dtype=tf.float64))
	model.add(tf.keras.layers.LayerNormalization(dtype=tf.float64))
	model.add(tf.keras.layers.Dense(units=nn_dimension, activation='relu', dtype=tf.float64))
	model.add(tf.keras.layers.LayerNormalization(dtype=tf.float64))
	model.add(tf.keras.layers.Dense(units=1, activation='softplus', dtype=tf.float64))

	return model

def init_non_linear_layer_norm_experiment_mapping_from_genomic_annotations_to_gamma(annotation_data_dimension, nn_dimension, scale_boolean):
	# Initialize Neural network model
	model = tf.keras.models.Sequential()
	if scale_boolean:
		model.add(tf.keras.layers.experimental.preprocessing.Normalization())
	model.add(tf.keras.layers.Dense(units=nn_dimension, activation='relu', input_dim=annotation_data_dimension))
	model.add(tf.keras.layers.LayerNormalization())
	model.add(tf.keras.layers.Dense(units=nn_dimension, activation='relu'))
	model.add(tf.keras.layers.LayerNormalization())
	model.add(tf.keras.layers.Dense(units=nn_dimension, activation='relu'))
	model.add(tf.keras.layers.LayerNormalization())
	model.add(tf.keras.layers.Dense(units=nn_dimension, activation='relu'))
	model.add(tf.keras.layers.LayerNormalization())
	model.add(tf.keras.layers.Dense(units=1, activation='softplus'))

	return model


def initialize_genomic_anno_model(model_type, annotation_data_dimension):
	if model_type == 'neural_network_no_drops':
		genomic_anno_to_gamma_model = init_non_linear_no_drops_mapping_from_genomic_annotations_to_gamma(annotation_data_dimension, 64, False)
	elif model_type == 'linear_model':
		genomic_anno_to_gamma_model = init_linear_mapping_from_genomic_annotations_to_gamma(annotation_data_dimension)
	elif model_type == 'intercept_model':
		genomic_anno_to_gamma_model = init_linear_mapping_from_genomic_annotations_to_gamma(1)
	elif model_type == 'neural_network_no_drops_scale':
		genomic_anno_to_gamma_model = init_non_linear_no_drops_mapping_from_genomic_annotations_to_gamma(annotation_data_dimension, 64, True)
	elif model_type == 'neural_network_batch_norm':
		genomic_anno_to_gamma_model = init_non_linear_batch_norm_mapping_from_genomic_annotations_to_gamma(annotation_data_dimension, 64, True)
	elif model_type == 'neural_network_layer_norm':
		genomic_anno_to_gamma_model = init_non_linear_layer_norm_mapping_from_genomic_annotations_to_gamma(annotation_data_dimension, 64, True)
	elif model_type == 'neural_network_layer_norm_experiment':
		genomic_anno_to_gamma_model = init_non_linear_layer_norm_experiment_mapping_from_genomic_annotations_to_gamma(annotation_data_dimension, 64, True)
	return genomic_anno_to_gamma_model


def softplus_np(x): return np.log(np.exp(x) + 1)

def softplus_tf(x): return tf.math.log(tf.math.exp(x) + 1)

def softplus_inv_np(x): return np.log(np.exp(x) - 1.0)

def marginalized_rss_loss_fxn_tf(window_pred_tau, intercept_variable, z_scores, sample_size, chrom_ld_mat_reg_ref, chrom_ld_mat_reg_reg):
	# Compute E[\beta*\beta^T] (This is a diagonal matrix)
	beta_beta_transpose_diag = window_pred_tau*sample_size

	# Compute single element of covariance
	R_beta2_RT = tf.linalg.matmul(tf.multiply(chrom_ld_mat_reg_ref, beta_beta_transpose_diag[:,0]), tf.transpose(chrom_ld_mat_reg_ref))

	# Compute full covariance
	cov = R_beta2_RT + chrom_ld_mat_reg_reg + tf.eye(chrom_ld_mat_reg_reg.shape[0], dtype=tf.float64)*softplus_tf(intercept_variable)

	# Invert covariance matrix
	#precision = sp_inv(cov, np.eye(cov.shape[0]))
	precision = tf.linalg.inv(cov)

	# Compute log determinant
	neg_log_det_info = tf.linalg.slogdet(precision)
	if np.copy(neg_log_det_info[0]) != 1.0:
		print('assumption eroror')
		pdb.set_trace()
	neg_log_det = neg_log_det_info[0]*neg_log_det_info[1]

	# Compute log likelihood on this chromosome
	chrom_neg_log_like = -(.5)*neg_log_det + (.5)*tf.linalg.matmul(tf.linalg.matmul(tf.transpose(z_scores), precision), z_scores)

	return chrom_neg_log_like


def marginalized_rss_batch_loss_fxn_tf(batch_pred_tau, batch_indices, intercept_variable, batch_z_scores, batch_sample_sizes, batch_chrom_ld_mat_reg_refs, batch_chrom_ld_mat_reg_regs):
	n_windows = len(batch_indices)
	for window_iter in range(n_windows):
		loss_value = marginalized_rss_loss_fxn_tf(tf.gather(batch_pred_tau, batch_indices[window_iter]), intercept_variable, batch_z_scores[window_iter], batch_sample_sizes[window_iter], batch_chrom_ld_mat_reg_refs[window_iter], batch_chrom_ld_mat_reg_regs[window_iter])
		if window_iter == 0:
			agg_loss = loss_value
		else:
			agg_loss = agg_loss + loss_value
	return agg_loss

def rss_w_intercept_softplus_neg_log_likelihood_and_gradient_on_single_chromosome(tau_pred, x_intercept, z_scores, chrom_ld_mat_reg_ref, chrom_ld_mat_reg_reg, sample_size):
	# Compute E[\beta*\beta^T] (This is a diagonal matrix)
	beta_beta_transpose_diag = tau_pred*sample_size


	# Compute single element of covariance
	R_beta2_RT = np.dot(np.multiply(chrom_ld_mat_reg_ref, beta_beta_transpose_diag), np.transpose(chrom_ld_mat_reg_ref))

	# Compute full covariance
	cov = R_beta2_RT + chrom_ld_mat_reg_reg + np.eye(chrom_ld_mat_reg_reg.shape[0])*softplus_np(x_intercept)

	# Invert covariance matrix
	#precision = sp_inv(cov, np.eye(cov.shape[0]))
	precision = np.linalg.inv(cov)

	# Compute log determinant
	neg_log_det_info = np.linalg.slogdet(precision)
	if neg_log_det_info[0] != 1.0:
		print('assumption eroror')
		pdb.set_trace()
	neg_log_det = neg_log_det_info[0]*neg_log_det_info[1]

	# Compute log likelihood on this chromosome
	chrom_log_like = -(.5)*neg_log_det + (.5)*np.dot(np.dot(z_scores, precision), z_scores)


	return chrom_log_like

def marginalized_rss_h2_evaluation_w_tf(genomic_anno_to_gamma_model, intercept_value, data_obj, model_type):
	total_neg_log_like = 0.0
	# A window is a region of dna space
	# This is number of windows we split dna space
	num_windows = data_obj.shape[0]

	for window_iter in range(num_windows):
		# Load in data for this window
		# Annotation file
		chrom_anno = np.load(data_obj[window_iter, 2])
		# LD mat reg ref
		ld_mat_reg_ref_file = data_obj[window_iter, 3]
		chrom_ld_mat_reg_ref = np.load(ld_mat_reg_ref_file)
		# LD mat reg reg
		ld_mat_reg_reg_file = data_obj[window_iter, 4]
		chrom_ld_mat_reg_reg = np.load(ld_mat_reg_reg_file)
		# Z scores
		z_score_file = data_obj[window_iter, 5]
		z_scores = np.loadtxt(z_score_file)
		# Valid regression indices
		valid_regression_indices_file = data_obj[window_iter, 6]
		valid_regression_indices = np.loadtxt(valid_regression_indices_file).astype(int)
		# Get sample size
		sample_size = float(trait_data_summary[window_iter, 7])
		# Error checks
		if chrom_ld_mat_reg_reg.shape[0] == 1:
			continue
		if valid_regression_indices.size == 1:
			continue
		if len(valid_regression_indices) == 0:
			continue
		# Fillter regression indices to those we have z-scores for
		chrom_ld_mat_reg_ref = chrom_ld_mat_reg_ref[valid_regression_indices,:]
		chrom_ld_mat_reg_reg = chrom_ld_mat_reg_reg[valid_regression_indices,:][:, valid_regression_indices]

		# If using intercept, alter genomic annotations
		if model_type == 'intercept_model':
			chrom_anno = np.ones((chrom_ld_mat_reg_ref.shape[1], 1))

		pred_tau = genomic_anno_to_gamma_model(tf.convert_to_tensor(chrom_anno,dtype=tf.float64), training=False)
		window_neg_log_like = rss_w_intercept_softplus_neg_log_likelihood_and_gradient_on_single_chromosome(np.copy(pred_tau[:,0]), intercept_value, z_scores, chrom_ld_mat_reg_ref, chrom_ld_mat_reg_reg, sample_size)
		total_neg_log_like = total_neg_log_like + window_neg_log_like
	return total_neg_log_like


def marginalized_rss_h2_batch_evaluation_w_tf(genomic_anno_to_gamma_model, intercept_value, data_obj, model_type, batch_size):
	total_neg_log_like = 0.0
	# A window is a region of dna space
	# This is number of windows we split dna space
	num_windows = data_obj.shape[0]

	for window_iter in range(num_windows):
		# Load in data for this window
		# Annotation file
		chrom_anno = np.load(data_obj[window_iter, 2])
		# LD mat reg ref
		ld_mat_reg_ref_file = data_obj[window_iter, 3]
		chrom_ld_mat_reg_ref = np.load(ld_mat_reg_ref_file)
		# LD mat reg reg
		ld_mat_reg_reg_file = data_obj[window_iter, 4]
		chrom_ld_mat_reg_reg = np.load(ld_mat_reg_reg_file)
		# Z scores
		z_score_file = data_obj[window_iter, 5]
		z_scores = np.loadtxt(z_score_file)
		# Valid regression indices
		valid_regression_indices_file = data_obj[window_iter, 6]
		valid_regression_indices = np.loadtxt(valid_regression_indices_file).astype(int)
		# Get sample size
		sample_size = float(trait_data_summary[window_iter, 7])
		# Error checks
		if chrom_ld_mat_reg_reg.shape[0] == 1:
			continue
		if valid_regression_indices.size == 1:
			continue
		if len(valid_regression_indices) == 0:
			continue
		# Fillter regression indices to those we have z-scores for
		chrom_ld_mat_reg_ref = chrom_ld_mat_reg_ref[valid_regression_indices,:]
		chrom_ld_mat_reg_reg = chrom_ld_mat_reg_reg[valid_regression_indices,:][:, valid_regression_indices]

		# If using intercept, alter genomic annotations
		if model_type == 'intercept_model':
			chrom_anno = np.ones((chrom_ld_mat_reg_ref.shape[1], 1))

		pdb.set_trace()
		pred_tau = genomic_anno_to_gamma_model(tf.convert_to_tensor(chrom_anno,dtype=tf.float64), training=False)
		window_neg_log_like = rss_w_intercept_softplus_neg_log_likelihood_and_gradient_on_single_chromosome(np.copy(pred_tau[:,0]), intercept_value, z_scores, chrom_ld_mat_reg_ref, chrom_ld_mat_reg_reg, sample_size)
		total_neg_log_like = total_neg_log_like + window_neg_log_like
	return total_neg_log_like


def marginalized_rss_h2_regression_with_evaluation_w_batched_tf(training_data, evaluation_data, model_type, learn_intercept, evaluation_output_file, intercept_output_file, batch_size=1, max_epochs=200, learning_rate=0.001):
	# Get number of annotations
	annotation_data_dimension = np.load(training_data[0,2]).shape[1]

	# Initialize mapping from annotations to per snp heritability
	genomic_anno_to_gamma_model = initialize_genomic_anno_model(model_type, annotation_data_dimension)

	# Initialize optimizer
	optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

	# Open ouutput file handle
	t = open(evaluation_output_file, 'w')
	t.write('iteration\ttraining_data_marginalized_rss_neg_loglikelihood\tevaluation_data_marginalized_rss_neg_loglikelihood\n')

	# Whether or not to learn intercept in LDSC
	# Initial value is 
	null_intercept = softplus_inv_np(1.0)
	if learn_intercept == 'learn_intercept':
		intercept_variable = tf.Variable(initial_value=null_intercept,trainable=True, name='intercept', dtype=tf.float64)
	elif learn_intercept == 'fixed_intercept':
		intercept_variable = tf.Variable(initial_value=null_intercept,trainable=False, name='intercept', dtype=tf.float64)
	else:
		print('assumption error: intercept model called ' + learn_intercept + ' not currently implemented')
		pdb.set_trace()

	# Initialize vectors to keep track of training and evaluation loss
	training_loss = []
	evaluation_loss = []

	# A window is a region of dna space
	# This is number of windows we split dna space
	num_windows = training_data.shape[0]

	# Lopp through windows
	for epoch_iter in range(max_epochs):
		# Keep track of training log likelihoods and weights of each regression snp
		#epoch_training_log_likelihoods = []
		#epoch_training_weights = []

		# Loop through windows
		print('###################################')
		print('epoch iter ' + str(epoch_iter))
		print('###################################')
		start_time = time.time()
		perm_windows = np.random.permutation(range(num_windows))
		for batch_counter, batch_arr in enumerate(np.array_split(perm_windows,len(perm_windows)/batch_size)):
			# Load in data for this batch
			batch_annos = []
			batch_z_scores = []
			batch_sample_sizes = []
			batch_chrom_ld_mat_reg_refs = []
			batch_chrom_ld_mat_reg_regs = []
			batch_indices = []
			start_index = 0
			for window_iter in batch_arr:
				# Annotation file
				chrom_anno = np.load(training_data[window_iter, 2])
				# LD mat reg ref
				ld_mat_reg_ref_file = training_data[window_iter, 3]
				chrom_ld_mat_reg_ref = np.load(ld_mat_reg_ref_file)
				# LD mat reg reg
				ld_mat_reg_reg_file = training_data[window_iter, 4]
				chrom_ld_mat_reg_reg = np.load(ld_mat_reg_reg_file)
				# Z scores
				z_score_file = training_data[window_iter, 5]
				z_scores = np.loadtxt(z_score_file)
				# Valid regression indices
				valid_regression_indices_file = training_data[window_iter, 6]
				valid_regression_indices = np.loadtxt(valid_regression_indices_file).astype(int)
				# Get sample size
				sample_size = float(trait_data_summary[window_iter, 7])
				# Error checks
				if chrom_ld_mat_reg_reg.shape[0] == 1:
					continue
				if valid_regression_indices.size == 1:
					continue
				if len(valid_regression_indices) == 0:
					continue
				# Fillter regression indices to those we have z-scores for
				chrom_ld_mat_reg_ref = chrom_ld_mat_reg_ref[valid_regression_indices,:]
				chrom_ld_mat_reg_reg = chrom_ld_mat_reg_reg[valid_regression_indices,:][:, valid_regression_indices]

				# Conver to tensors
				chrom_ld_mat_reg_ref = tf.convert_to_tensor(chrom_ld_mat_reg_ref, dtype=tf.float64)
				chrom_ld_mat_reg_reg = tf.convert_to_tensor(chrom_ld_mat_reg_reg, dtype=tf.float64)
				z_scores = tf.convert_to_tensor(z_scores.reshape(len(z_scores),1), dtype=tf.float64)

				# If using intercept, alter genomic annotations
				if model_type == 'intercept_model':
					chrom_anno = np.ones((chrom_ld_mat_reg_ref.shape[1], 1))

				# Append data to batch array
				batch_annos.append(chrom_anno)
				batch_z_scores.append(z_scores)
				batch_sample_sizes.append(sample_size)
				batch_chrom_ld_mat_reg_refs.append(chrom_ld_mat_reg_ref)
				batch_chrom_ld_mat_reg_regs.append(chrom_ld_mat_reg_reg)
				batch_indices.append(np.arange(start_index, start_index+chrom_anno.shape[0]))
				start_index = start_index + chrom_anno.shape[0]

			chrom_anno = np.vstack(batch_annos)

			# Use tf.gradient tape to compute gradients
			with tf.GradientTape() as tape:
				batch_pred_tau = genomic_anno_to_gamma_model(tf.convert_to_tensor(chrom_anno,dtype=tf.float64), training=True)
				#loss_value = marginalized_rss_loss_fxn_tf(window_pred_tau, intercept_variable, z_scores, sample_size, chrom_ld_mat_reg_ref, chrom_ld_mat_reg_reg)
				loss_value = marginalized_rss_batch_loss_fxn_tf(batch_pred_tau, batch_indices, intercept_variable, batch_z_scores, batch_sample_sizes, batch_chrom_ld_mat_reg_refs, batch_chrom_ld_mat_reg_regs)

			# Define trainable variables
			trainable_variables = genomic_anno_to_gamma_model.trainable_weights
			if learn_intercept == 'learn_intercept':
				trainable_variables.append(intercept_variable)
			# Compute and apply gradients
			grads = tape.gradient(loss_value, trainable_variables)
			optimizer.apply_gradients(zip(grads, trainable_variables))

		evaluation_loss = marginalized_rss_h2_evaluation_w_tf(genomic_anno_to_gamma_model, np.copy(intercept_variable)*1.0, evaluation_data, model_type)
		training_loss = marginalized_rss_h2_evaluation_w_tf(genomic_anno_to_gamma_model, np.copy(intercept_variable)*1.0, training_data, model_type)
		t.write(str(epoch_iter) + '\t' + str(training_loss) + '\t' + str(evaluation_loss) + '\n')
		t.flush()
		print('Training loss: ' + str(training_loss))
		print('Evaluation loss: ' + str(evaluation_loss))
		print(softplus_np(np.copy(intercept_variable)*1.0))
		np.savetxt(intercept_output_file, [softplus_np(np.copy(intercept_variable)*1.0)], fmt="%s", delimiter='\t')

	t.close()

def marginalized_rss_h2_regression_with_evaluation_w_tf(training_data, evaluation_data, model_type, learn_intercept, evaluation_output_file, intercept_output_file, max_epochs=200, learning_rate=0.001):
	# Get number of annotations
	annotation_data_dimension = np.load(training_data[0,2]).shape[1]

	# Initialize mapping from annotations to per snp heritability
	genomic_anno_to_gamma_model = initialize_genomic_anno_model(model_type, annotation_data_dimension)

	# Initialize optimizer
	optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

	# Open ouutput file handle
	t = open(evaluation_output_file, 'w')
	t.write('iteration\ttraining_data_marginalized_rss_neg_loglikelihood\tevaluation_data_marginalized_rss_neg_loglikelihood\n')

	# Whether or not to learn intercept in LDSC
	# Initial value is 
	null_intercept = softplus_inv_np(1.0)
	if learn_intercept == 'learn_intercept':
		intercept_variable = tf.Variable(initial_value=null_intercept,trainable=True, name='intercept', dtype=tf.float64)
	elif learn_intercept == 'fixed_intercept':
		intercept_variable = tf.Variable(initial_value=null_intercept,trainable=False, name='intercept', dtype=tf.float64)
	else:
		print('assumption error: intercept model called ' + learn_intercept + ' not currently implemented')
		pdb.set_trace()

	# Initialize vectors to keep track of training and evaluation loss
	training_loss = []
	evaluation_loss = []

	# A window is a region of dna space
	# This is number of windows we split dna space
	num_windows = training_data.shape[0]

	# Lopp through windows
	for epoch_iter in range(max_epochs):
		# Keep track of training log likelihoods and weights of each regression snp
		#epoch_training_log_likelihoods = []
		#epoch_training_weights = []

		# Loop through windows
		print('###################################')
		print('epoch iter ' + str(epoch_iter))
		print('###################################')
		start_time = time.time()
		for window_counter, window_iter in enumerate(np.random.permutation(range(num_windows))):
			#print(softplus_np(np.copy(intercept_variable)*1.0))
			#print(window_iter)

			#print(window_counter)

			# Load in data for this window
			# Annotation file
			chrom_anno = np.load(training_data[window_iter, 2])
			# LD mat reg ref
			ld_mat_reg_ref_file = training_data[window_iter, 3]
			chrom_ld_mat_reg_ref = np.load(ld_mat_reg_ref_file)
			# LD mat reg reg
			ld_mat_reg_reg_file = training_data[window_iter, 4]
			chrom_ld_mat_reg_reg = np.load(ld_mat_reg_reg_file)
			# Z scores
			z_score_file = training_data[window_iter, 5]
			z_scores = np.loadtxt(z_score_file)
			# Valid regression indices
			valid_regression_indices_file = training_data[window_iter, 6]
			valid_regression_indices = np.loadtxt(valid_regression_indices_file).astype(int)
			# Get sample size
			sample_size = float(trait_data_summary[window_iter, 7])
			# Error checks
			if chrom_ld_mat_reg_reg.shape[0] == 1:
				continue
			if valid_regression_indices.size == 1:
				continue
			if len(valid_regression_indices) == 0:
				continue
			# Fillter regression indices to those we have z-scores for
			chrom_ld_mat_reg_ref = chrom_ld_mat_reg_ref[valid_regression_indices,:]
			chrom_ld_mat_reg_reg = chrom_ld_mat_reg_reg[valid_regression_indices,:][:, valid_regression_indices]

			#pred_tau = genomic_anno_to_gamma_model(tf.convert_to_tensor(chrom_anno,dtype=tf.float64), training=False)
			#print(np.sort(np.copy(pred_tau[:,0])))
			#log_like = rss_w_intercept_softplus_neg_log_likelihood_and_gradient_on_single_chromosome(np.copy(pred_tau[:,0]), np.copy(intercept_variable)*1.0, z_scores, chrom_ld_mat_reg_ref, chrom_ld_mat_reg_reg, chrom_anno, sample_size)

			# Conver to tensors
			chrom_ld_mat_reg_ref = tf.convert_to_tensor(chrom_ld_mat_reg_ref, dtype=tf.float64)
			chrom_ld_mat_reg_reg = tf.convert_to_tensor(chrom_ld_mat_reg_reg, dtype=tf.float64)
			z_scores = tf.convert_to_tensor(z_scores.reshape(len(z_scores),1), dtype=tf.float64)

			# If using intercept, alter genomic annotations
			if model_type == 'intercept_model':
				chrom_anno = np.ones((chrom_ld_mat_reg_ref.shape[1], 1))
			# Use tf.gradient tape to compute gradients
			with tf.GradientTape() as tape:
				window_pred_tau = genomic_anno_to_gamma_model(tf.convert_to_tensor(chrom_anno,dtype=tf.float64), training=True)
				loss_value = marginalized_rss_loss_fxn_tf(window_pred_tau, intercept_variable, z_scores, sample_size, chrom_ld_mat_reg_ref, chrom_ld_mat_reg_reg)

			# Define trainable variables
			trainable_variables = genomic_anno_to_gamma_model.trainable_weights
			if learn_intercept == 'learn_intercept':
				trainable_variables.append(intercept_variable)
			# Compute and apply gradients
			grads = tape.gradient(loss_value, trainable_variables)
			optimizer.apply_gradients(zip(grads, trainable_variables))

		evaluation_loss = marginalized_rss_h2_evaluation_w_tf(genomic_anno_to_gamma_model, np.copy(intercept_variable)*1.0, evaluation_data, model_type)
		training_loss = marginalized_rss_h2_evaluation_w_tf(genomic_anno_to_gamma_model, np.copy(intercept_variable)*1.0, training_data, model_type)
		t.write(str(epoch_iter) + '\t' + str(training_loss) + '\t' + str(evaluation_loss) + '\n')
		t.flush()
		print('Training loss: ' + str(training_loss))
		print('Evaluation loss: ' + str(evaluation_loss))
		print(softplus_np(np.copy(intercept_variable)*1.0))
		np.savetxt(intercept_output_file, [softplus_np(np.copy(intercept_variable)*1.0)], fmt="%s", delimiter='\t')

	t.close()

trait_name = sys.argv[1]
trait_data_summary_file = sys.argv[2]
model_type = sys.argv[3]
held_out_chromosome = float(sys.argv[4])
learn_intercept = sys.argv[5]
marginalized_rss_h2_results_dir = sys.argv[6]
batch_size = int(sys.argv[7])


# Load in trait data summary file
trait_data_summary = np.loadtxt(trait_data_summary_file, dtype=str,delimiter='\t')[1:,:]
# Split into training and test data
training_data =trait_data_summary[trait_data_summary[:,1].astype(float) != held_out_chromosome,:]
#training_data =trait_data_summary[trait_data_summary[:,1].astype(float) ==21,:]
evaluation_data =trait_data_summary[trait_data_summary[:,1].astype(float) == held_out_chromosome,:]


# Output stem
output_stem = marginalized_rss_h2_results_dir + 'evaluate_marginalized_rss_w_tf_' + trait_name + '_' + model_type + '_' + str(held_out_chromosome) + '_' + learn_intercept + '_' + str(batch_size)
# Evaluation outpuut file
evaluation_output_file = output_stem + '_evaluation_res3.txt'
intercept_output_file = output_stem + '_intercept_res3.txt'

marginalized_rss_h2_regression_with_evaluation_w_batched_tf(training_data, evaluation_data, model_type, learn_intercept, evaluation_output_file, intercept_output_file, batch_size)
#marginalized_rss_h2_regression_with_evaluation_w_tf(training_data, evaluation_data, model_type, learn_intercept, evaluation_output_file, intercept_output_file)
