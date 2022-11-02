import sys
sys.path.remove('/n/app/python/3.7.4-ext/lib/python3.7/site-packages')
import os
import pdb
import numpy as np
from pandas_plink import read_plink1_bin
import pickle
import time
import scipy.sparse
import scipy.optimize
















trait_name = sys.argv[1]
shared_input_data_dir = sys.argv[2]
trait_specific_input_data_dir = sys.argv[3]
marginalized_rss_h2_results_dir = sys.argv[4]



rss_log_likelihood()