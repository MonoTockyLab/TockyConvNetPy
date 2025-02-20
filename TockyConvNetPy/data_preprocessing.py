# data_preprocessing.py

import numpy as np

def log_scale_age(age_array, age_max):
    age_array_log_scaled = np.log1p(age_array) / np.log1p(age_max)
    return age_array_log_scaled
    
