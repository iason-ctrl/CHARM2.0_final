import numpy as np
import nifty8 as ift
from helpers.functions import radial_los, domain_unity, pickle_me_this, planck_cosmology
from helpers.plotters import  synthetic_plot, compare_in_signal_space, visualize_and_analyze_posterior_power_spectrum, compare_in_data_space

# -------  Basic parameters and constants  ------- #

switch1 = False         # save in 'figures' folder: Ground truth in signal and pow spec + data realization
switch2 = True         # show via plt.show(): Ground truth in signal and pow spec + data realization

switch3 = False         # save in 'figures' folder: Comparison of reconstruction and ground truth in signal space
switch4 = True         # show via plt.show(): Comparison of reconstruction and ground truth in signal space

switch5 = False         # save in 'figures' folder: Visualization of posterior power spectrum
switch6 = True         # show via plt.show(): Visualization of posterior power spectrum

switch7 = False         # save in 'figures' folder: Comparison of reconstruction and ground truth in data space
switch8 = True         # show via plt.show(): Comparison of reconstruction and ground truth in data space

d_h = 3e8/(68.6e3)*1e6  # Hubble-length in parsecs
noise = 0.1             # Noise level

n_pix = 8000
x_length = 6.7          # 6.7 ~ natural redshift of CMB ~ 'beginning' of the universe.
n_datapoints = 200


# -------  Define domains and basic fields. Generate random redshifts.  ------- #


signal_space = ift.RGSpace(n_pix,distances=x_length/n_pix)
data_space = ift.UnstructuredDomain((n_datapoints,))

natural_redshift_starts, natural_redshift_ends = radial_los(n_datapoints)           # randomly generated natural redshifts x
redshifts = np.exp(natural_redshift_ends[0]) - 1                                    # corresponding 'canonical' redshifts z

print(f"Number of datapoints: {n_datapoints}. Resolution: {n_pix}")


# -------  Build a correlated field model  ------- #


'''
We set the slope of the Power Spectrum to be -4. We set the fluctuations (additive offset to the Power Spectrum) to 
be 1 within some small margin for error. Asperity and Flexibility are set to None since no deviation from power law behaviour 
is expected.
'''



args = {
    "offset_mean" :     0,
    "offset_std" :      None,
    "fluctuations":     (1.1,1e-16),
    "loglogavgslope":   (-4,1e-16),
    "asperity":         None,
    "flexibility":      None
}
signal = ift.SimpleCorrelatedField(signal_space, **args)
fluctuations_info = f"_fluct_{args['fluctuations']}"

# ------- Build necessary fields and operators for signal response ------- #

signal_coordinate_field = domain_unity(signal_space)        # a 1D array containing the argument values of the correlated field i.e. x in s = s(x)

weights_data_space = np.ones(n_datapoints) + redshifts      # corresponds to e^x = 1 + z. There n_datapoints many 1+z values.
weights_signal_space = np.exp(signal_coordinate_field.val)  # corresponds to e^x. There n_pix many e^x values.

WEIGHT_signal_space = ift.DiagonalOperator(diagonal=ift.Field.from_raw(signal_space,weights_signal_space))  # Weight operator in signal space
WEIGHT_data_space = ift.DiagonalOperator(diagonal=ift.Field.from_raw(data_space,weights_data_space))        # Weight operator in data space

SUBTRACT_FIVES = ift.Adder(a=-5,domain=ift.DomainTuple.make(data_space,))

N = ift.ScalingOperator(data_space, noise, np.float64)
R = ift.LOSResponse(signal_space, starts=natural_redshift_starts, ends=natural_redshift_ends)


# ------- Build the signal response as chain of operators  ------- #


# Build the response operator as a chain of operators step by step to detect any errors
control_step_one = WEIGHT_signal_space(ift.exp(-1/2*signal))
control_step_two = R(control_step_one)
control_step_three = WEIGHT_data_space(control_step_two)
control_step_four = SUBTRACT_FIVES(5*np.log10(d_h*control_step_three))

# Build the response operator chain in one line
def response_operator(signal_field):
    return SUBTRACT_FIVES(5 * np.log10(d_h * (WEIGHT_data_space(R(WEIGHT_signal_space(ift.exp(-1 / 2 * signal_field)))))))
signal_response = response_operator(signal)

# ------- Generate synthetic data  ------- #


mock_position = ift.from_random(signal.domain, 'normal', mean=10, std=1 )
data_realization = signal_response(mock_position) + N.draw_sample()
data_field = ift.Field(domain=ift.DomainTuple.make(data_space,),val=data_realization.val)

synthetic_plot(save=switch1,show=switch2,x=signal_coordinate_field.val, y=signal(mock_position).val,
               p=signal.power_spectrum.force(mock_position).val, z_data=redshifts,
               d=data_realization.val, cf_args={"fluctuations: " : args["fluctuations"],"loglogavgslope: " : args["loglogavgslope"]})


# ------- Minimization and sampling  controllers ------- #


ic_sampling_lin = ift.AbsDeltaEnergyController(name="Sampling (linear)", deltaE=0.05, iteration_limit=100)
ic_sampling_nl = ift.AbsDeltaEnergyController(name="Sampling (nonlinear)", deltaE=0.01, iteration_limit=60, convergence_level=2)
ic_newton_minimization = ift.AbsDeltaEnergyController(name="Newton Minimization. Searching for energy descent direction", deltaE=0.5, iteration_limit=35, convergence_level=2)

minimizer_geoVI_MGVI = ift.NewtonCG(ic_newton_minimization)
nonlinear_geoVI_minimizer = ift.NewtonCG(ic_sampling_nl)

likelihood_energy = ift.GaussianEnergy(data_field, inverse_covariance=N.inverse) @ signal_response

global_iterations = 15
n_samples = lambda iiter: 10 if iiter < 5 else 50 # increase sample rate for higher initial indices (get first a ball park and then converge with force)
samples = ift.optimize_kl(likelihood_energy=likelihood_energy,
                          total_iterations=global_iterations,
                          n_samples=n_samples,
                          kl_minimizer=minimizer_geoVI_MGVI,
                          sampling_iteration_controller=ic_sampling_lin,
                          nonlinear_sampling_minimizer= nonlinear_geoVI_minimizer,
                          output_directory=None,
                          return_final_position=True)

print("Kullback Leibler Divergence has been minimized, Posterior Samples found.")


# ------- Visualize Posterior ------- #


posterior_realizations,final_posterior_position = samples
pickle_me_this("synthetic_posterior_realizations"+fluctuations_info, posterior_realizations)      # save data in 'pickles' subfolder as backup
pickle_me_this("synthetic_final_posterior_position"+fluctuations_info, final_posterior_position)  # save data in 'pickles' subfolder as backup
print("pickled inference run.")

mean, var = posterior_realizations.sample_stat(signal)

power_spectra = posterior_realizations.iterator(signal.power_spectrum)
average_power_spectrum = posterior_realizations.average(signal.power_spectrum)
ground_truth_powspec = signal.power_spectrum.force(mock_position)

compare_in_signal_space(save=switch3, show=switch4, synthetic=True, sig_c_field=signal_coordinate_field.val,
                        x_max_data=max(natural_redshift_ends[0]),reconstructed_mean=mean.val,
                        reconstructed_var=var.val,comparison_field=signal(mock_position).val,
                        y_scale=(0,24))


loglogavgslope, fluctuations = visualize_and_analyze_posterior_power_spectrum(save=switch5,show=switch6,power_spectra=power_spectra,
                                   power_spec_average=average_power_spectrum.val,
                                   ground_truth=ground_truth_powspec.val,place_params_at=(23.7,0.0004),
                                   y_scale=[2.12e-5,0.0005])


k_mean, k_std = loglogavgslope
flucts_mean, flucts_std = fluctuations

data_produced_by_posterior = posterior_realizations.average(signal_response)
data_produced_by_ground_truth = signal_response(mock_position)


compare_in_data_space(save=switch7, show=switch8, synthetic=True, data_produced_by_posterior=data_produced_by_posterior.val,
                      data_produced_by_comparison_field=data_produced_by_ground_truth.val,
                      actual_data=data_realization.val,x_data=natural_redshift_ends[0])


