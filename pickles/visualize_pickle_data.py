from helpers.functions import  unpickle_me_this

"""

Relative imports of read_data and save_figures brake :( 

Copy EVERYTHING from 'cosmological_tomography.py' or 'synthetic_catalogue.py' 
and REPLACE the call to 'optimize_kl()' with something like 

    >>> posterior_realizations = unpickle_me_this("synthetic_posterior_realizations.pickle")
    >>> mean, var = posterior_realizations.sample_stat(signal)
    
And delete these lines:
    >>> posterior_realizations,final_posterior_position = samples
    >>> pickle_me_this("synthetic_posterior_realizations"+fluctuations_info, posterior_realizations)      # save data in 'pickles' subfolder as backup
    >>> pickle_me_this("synthetic_final_posterior_position"+fluctuations_info, final_posterior_position)  # save data in 'pickles' subfolder as backup
    >>> print("pickled inference run.")

Do NOT forget to change the parameters of the corr. field model to be the same as in 'cosmological_tomography.py'.

Continue then here with the visualization of the posterior in case something went wrong,
or you want to reanalyze the data (e.g. adjust the scale), implement your own functions etc.

You will need to set the 'from_pickles_folder' keyword to True for each plot function.

"""
