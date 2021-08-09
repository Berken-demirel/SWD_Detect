from swd_utils import *

seed(162)
tf.random.set_seed(162)
config = {'window_width': 10, 'sensitivity': 0.1, 'overlap': 0}

### following functions creates Absence Seizure dataset for given configurations
### since it is already done, these functions are commented out,
### but one can retry to generate data with this or maybe other configurations with these functions from TUSZ dataset
# prepare_database('./', 'absz_patients', config=config, only_absz=True)
# adjust_absz_patients('./absz_patients', 'absz_all',config=config)

### following functions are training the model for the time and psd input
### they are commented out since it takes time; the trained models are uploaded to the proper folder in the github repo.
# multitaperpsd_leave_n_out_cross_validation(1, './absz_patients/absz_all.npz', 'psd_model/', sample_dim=(1, 2, 1251))
# time_leave_n_out_cross_validation(1, './absz_patients/absz_all.npz', 'time_model/', sample_dim=(1, 2, 2500))

apply_assumptions(models_folder='./psd_model', dataset_path='./absz_patients/absz_all.npz', sample_dim=(1, 2, 1251), is_time=False)
apply_assumptions(models_folder='./time_model', dataset_path='./absz_patients/absz_all.npz', sample_dim=(1, 2, 2500), is_time=True)




