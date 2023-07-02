import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import AutoMinorLocator
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from .functions import get_power_spec_params, conversion_from_offset_to_fluctuations, read_data

__all__ = ['synthetic_plot', 'real_data_plot',
           'compare_in_data_space', 'compare_in_signal_space',
           'visualize_and_analyze_posterior_power_spectrum', 'compare_data']

plt.rcParams["font.family"] = "Hiragino Maru Gothic Pro"
plt.rcParams["font.size"] = "12"
plt.rc('axes', labelsize=15)
plt.rc('figure', titlesize=20)

fig, ax = plt.subplots()

ax.tick_params(axis='both',direction='in',width=1.5)
ax.tick_params(which='major',direction='in', length=7, )
ax.tick_params(which='minor',direction='in', length=4, )
ax.xaxis.set_minor_locator(AutoMinorLocator())
ax.yaxis.set_minor_locator(AutoMinorLocator())

palette = sns.color_palette('muted')
blue, orange, green, red, mauve, brown, pink, gray, yellow, lightblue = palette

path_extra = "../"



def synthetic_plot(save:'bool', show:'bool', x:'np.ndarray', y:'np.ndarray', p:'np.ndarray', z_data:'np.ndarray', d:'np.ndarray', cf_args=None, from_pickle_directory=False):
    """
        Returns a figure containing the ground truth power spectrum and signal field as well as the used data realization
        and the used parameters for the correlated field model.

            Parameters
            ----------
            save        :type       :`~bool`        If True the figure is saved in the 'figures' folder.
            show        :type       :`~bool`        If True shows the figure via plt.show()
            x           :type       :`~np.ndarray`  The ground truth DOMAIN values ('continuous' natural redshifts x)
            y           :type       :`~np.ndarray`  The ground truth SIGNAL FIELD values
            p           :type       :`~np.ndarray`  The ground truth POWER SPECTRUM values
            d           :type       :`~np.ndarray`  The data realization values
            z_data      :type       :`~np.ndarray`  The domain values of the data realizations ('discrete' 'canonical' redshifts z)
            cf_args     :type       :`~dict`        Dictionary containing names and values of correlated field parameters. Optional

            Returns
            ----------
            mpl.fig
    """
    plt.subplot(2, 2, 1)

    plt.title("Ground Truth: Signal (Synth.)",fontsize=20)
    plt.xlabel("scale factor magnitude $x$", fontsize=15)
    plt.ylabel("Signal field $s(x)$",fontsize=15)
    plt.plot(x,y,ls="--",lw=2, color=red)

    plt.subplot(2, 2, 2)
    plt.title("Ground Truth: Pow Spec (Synth.)",fontsize=20)
    plt.xlabel("Fourier modes $k$ (log-scale)",fontsize=15)
    plt.ylabel("$P_s(k)$ (log-scale)",fontsize=15)
    plt.yscale('log')
    plt.xscale('log')
    plt.plot(p, ls="--", lw=2, color=red)

    plt.subplot(2, 2, 3)
    plt.title("Data Realizations (Synth.)", fontsize=20)
    plt.xlabel("Redshifts $z$",fontsize=15)
    plt.ylabel("Distance moduli $\mu$",fontsize=15)
    plt.plot(z_data, d, markersize=8, marker="o", color="black", markerfacecolor='white',linewidth=0, label=f"{len(z_data)} datapoints ")
    plt.legend()

    if cf_args is not None:

        plt.subplot(2, 2, 4)

        curr_ax = plt.gca()
        curr_ax.axes.xaxis.set_ticklabels([])
        curr_ax.axes.yaxis.set_ticklabels([])

        clock=0-1
        for i in cf_args:
            clock+=1
            y_pos = 0.75-0.1*clock
            plt.text(0.1,y_pos,f"{i} :  {cf_args[i]}")

    plt.subplots_adjust(hspace=0.4)

    if show:
        plt.show()
        plt.clf()
    if save:
        fig.set_size_inches(16, 9)
        groundpath = "figures/"
        if from_pickle_directory:
            path = path_extra + groundpath
        else:
            path = groundpath
        name = "synthetic_ground_truth_&_data"
        plt.savefig(path + name + ".png", dpi=300, bbox_inches='tight')
        print("Saved as " + name + ".png")
        plt.clf()
    if not show and not save:
        plt.clf()


def compare_in_data_space(save:'bool', show:'bool',synthetic:'bool',
                          data_produced_by_posterior:'np.ndarray',
                          data_produced_by_comparison_field:'np.ndarray',
                          actual_data:'np.ndarray',x_data:'np.ndarray', from_pickle_directory=False):
    """
                Compares the reconstructed data from the mean reconstructed signal field to another Field.
                (e.g. ground truth or planck cosmology) and computes residuals.

                    Parameters
                    ----------
                    save                                :type       :`~bool`        If True the figure is saved in the 'figures' folder.
                    show                                :type       :`~bool`        If True shows the figure via plt.show()
                    sig_c_field                         :type       :`~np.ndarray`  The signal coordinate field ('continuous' natural redshifts x)
                    synthetic                           :type       :`~bool`        Sets certain texts and titles to "Synthetic Data" if True, "Real Data" else.
                    data_produced_by_posterior          :type       :`~np.ndarray`  The mean of the reconstructed posterior samples.
                    data_produced_by_comparison_field   :type       :`~np.ndarray`  The square of the posterior means standard deviation.
                    actual_data                         :type       :`~np.ndarray`  The field to compare the posterior mean to.
                    x_data                              :type       :`~np.ndarray`  The x-coordinate of the DATA ('natural redshift').

                    Returns
                    ----------
                    mpl.fig
            """

    if synthetic:
        kw1 = "synthetic"
        kw2 = "ground truth"
        kw3 = "ground truth + noise"
        kw4 = "ground truth"
    else:
        kw1 = "real"
        kw2 = "planck cosmology"
        kw3 = "actual data"
        kw4 = "actual data"

    plt.subplot(2, 1, 1)

    plt.xlabel("scale factor magnitude $x$", fontsize=15)
    plt.ylabel("Distance Modulus $\mu$", fontsize=15)
    plt.title(f"Comparison of {kw1} reconstruction and {kw2} in data space", fontsize=20)
    plt.plot(x_data, data_produced_by_comparison_field, label=f"{kw2}", markersize=8, marker="o", color="black", markerfacecolor=red,linewidth=0,alpha=0.4)
    plt.plot(x_data, data_produced_by_posterior, label="reconstructed posterior",markersize=8, marker="o", color="black", markerfacecolor=blue, linewidth=0,alpha=0.4)
    plt.plot(x_data, actual_data, markersize=8, marker="o", color="black", markerfacecolor="white",label=kw3, lw=0, alpha=0.4)



    plt.legend()

    plt.subplot(2, 1, 2)

    plt.title(f"Residuals of reconstruction to {kw4}")
    plt.xlabel("scale factor magnitude $x$", fontsize=15)
    plt.ylabel("$\Delta \mu$", fontsize=15)

    plt.plot(x_data, np.zeros(len(x_data)), ls="--", lw=2, color="black", label="null line")

    if synthetic:
        difference = data_produced_by_posterior - data_produced_by_comparison_field # reconstructed data - ground truth
        plt.plot(x_data, difference, "x", color=blue, alpha=0.4, label="Difference of reconstructed moduli to ground truth moduli")
    else:
        difference1 = data_produced_by_posterior - actual_data                       # reconstructed data - actual data
        difference2 = data_produced_by_comparison_field - actual_data                # reconstructed data from planck cosmology - ground truth
        plt.plot(x_data, difference1, "x", color=blue, alpha=0.8,label="Difference of reconstructed moduli to actual moduli")
        plt.plot(x_data, difference2, "x", color=red, alpha=0.8,label="Difference of moduli reconstructed from planck cosmology to actual moduli")


    plt.legend()

    """if x_scale is not None:
        plt.xlim(x_scale[0], x_scale[1])
    else:
        plt.xlim(0, x_max_data)

    if y_scale_deviation is not None:
        plt.ylim(y_scale_deviation[0], y_scale_deviation[1])
    else:
        plt.ylim(-1, 1)"""

    plt.subplots_adjust(hspace=0.4)

    if show:
        plt.show()
        plt.clf()
    if save:
        fig.set_size_inches(16, 9)
        groundpath = "figures/"
        if from_pickle_directory:
            path = path_extra + groundpath
        else:
            path = groundpath
        name = f"Comparison_of_{kw1}_reconstruction_and_{kw2}_dataspace"
        plt.savefig(path + name + ".png", dpi=300, bbox_inches='tight')
        print("Saved as " + name + ".png")
        plt.clf()
    if not show and not save:
        plt.clf()
    pass

def compare_in_signal_space(save:'bool', show:'bool',synthetic:'bool',sig_c_field:'np.ndarray',x_max_data:'float',
                            reconstructed_mean:'np.ndarray',reconstructed_var:'np.ndarray',
                            comparison_field:'np.ndarray',x_scale=None,y_scale=None,y_scale_deviation=None, from_pickle_directory=False):
    """
            Compares the mean signal value of the reconstruction to another Field (e.g. ground truth
            or planck cosmology) and computes deviation.

                Parameters
                ----------
                save                :type       :`~bool`        If True the figure is saved in the 'figures' folder.
                show                :type       :`~bool`        If True shows the figure via plt.show()
                sig_c_field         :type       :`~np.ndarray`  The signal coordinate field ('continuous' natural redshifts x)
                synthetic           :type       :`~bool`        Sets certain texts and titles to "Synthetic Data" if True, "Real Data" else.
                reconstructed_mean  :type       :`~np.ndarray`  The mean of the reconstructed posterior samples.
                reconstructed_var   :type       :`~np.ndarray`  The square of the posterior means standard deviation.
                comparison_field    :type       :`~np.ndarray`  The field to compare the posterior mean to.
                x_max_data          :type       :`~float`       The maximum x-coordinate of the DATA ('natural redshift'). Sets xlim on the plot
                y_scale:            :type       :`~tuple`       A 2D tuple containing the min and max visible y value. Sets ylim on the plot. If None (default value) set by mpl.
                y_scale_deviation   :type       :`~tuple`       Does the same as 'y_scale' for the deviation plot. Optional, default is (-1, 1)
                x_scale:            :type       :`~tuple`       Does the same as 'y_scale' for the x-axis. Optional, default x_scale is (0, x_max)

                Returns
                ----------
                mpl.fig
        """

    if synthetic:
        kw1 = "synthetic"
        kw2 = "ground truth"
    else:
        kw1 = "real"
        kw2 = "planck cosmology"

    plt.subplot(2,1,1)

    plt.xlabel("scale factor magnitude $x$",fontsize=15)
    plt.ylabel("Signal field $s(x)$",fontsize=15)
    plt.title(f"Comparison of {kw1} reconstruction and {kw2}",fontsize=20)
    plt.errorbar(sig_c_field,reconstructed_mean,yerr=np.sqrt(reconstructed_var),label="reconstruction (lightblue: std)", color=blue,ecolor=lightblue,elinewidth=1.5, linewidth=1.5)
    plt.plot(sig_c_field, comparison_field, label=f"{kw2}", color=red, ls="--", lw=2)

    if x_scale is not None:
        plt.xlim(x_scale[0], x_scale[1])
    else:
        plt.xlim(0, x_max_data)
    if y_scale is not None:
        plt.ylim(y_scale[0], y_scale[1])


    plt.legend()

    plt.subplot(2, 1, 2)

    plt.xlabel("scale factor magnitude $x$",fontsize=15)
    difference = reconstructed_mean-comparison_field
    plt.errorbar(sig_c_field,difference,yerr=np.sqrt(reconstructed_var),label="deviation (lightblue: std)", color=mauve,ecolor=lightblue,elinewidth=1.5, linewidth=1.5)
    plt.plot(sig_c_field,np.zeros(len(sig_c_field)), ls="--", lw=2, color="black", label="null line")
    plt.ylabel("Deviation $\Delta s(x)$", fontsize=15)
    plt.legend()


    if x_scale is not None:
        plt.xlim(x_scale[0], x_scale[1])
    else:
        plt.xlim(0, x_max_data)

    if y_scale_deviation is not None:
        plt.ylim(y_scale_deviation[0],y_scale_deviation[1])
    else:
        plt.ylim(-1, 1)

    plt.subplots_adjust(hspace=0.4)

    if show:
        plt.show()
        plt.clf()
    if save:
        fig.set_size_inches(16, 9)
        groundpath = "figures/"
        if from_pickle_directory:
            path = path_extra + groundpath
        else:
            path = groundpath
        name = f"Comparison_of_{kw1}_reconstruction_and_{kw2}_signalspace"
        plt.savefig(path + name + ".png", dpi=300, bbox_inches='tight')
        print("Saved as " + name + ".png")
        plt.clf()
    if not show and not save:
        plt.clf()

def visualize_and_analyze_posterior_power_spectrum(save,show,power_spectra:'np.ndarray', power_spec_average:'np.ndarray',
                                       place_params_at:'tuple',ground_truth=None,x_scale=None,y_scale=None,plot_debug:'bool'=False, from_pickle_directory=False):


    exponents = []
    y_offsets = []

    clock=0
    for pow_spec_sample in power_spectra:

        # calculate exponent and y-offset
        params = get_power_spec_params(pow_spec_sample.val, plot=plot_debug)

        k_exp, y_offs = params
        exponents.append(k_exp)
        y_offsets.append(y_offs)

        clock+=1
        if not plot_debug:
            if clock==1:
                plt.plot(pow_spec_sample.val, color=orange, lw=2, label="power spectrum samples", alpha=1)
            else:
                plt.plot(pow_spec_sample.val, color=orange, lw=2, alpha=1/clock)

    if plot_debug:
        plt.show()
        plt.clf()

    exponents = np.array(exponents)
    y_offsets = np.array(y_offsets)
    flucts = conversion_from_offset_to_fluctuations(y_offsets)

    k_mean = np.round(np.mean(exponents),3)
    k_std = np.round(np.std(exponents),3)

    flucts_mean = np.round(np.mean(flucts),3)
    flucts_std = np.round(np.std(flucts),3)


    plt.plot(power_spec_average, color=blue, lw=2, label="power spectrum average")
    if ground_truth is not None:
        plt.plot(ground_truth, color=red, lw=2,ls="--", label="power spectrum ground truth")

    plt.title("Power Spectra samples and average (Synth.)", fontsize=20)
    plt.xlabel("Fourier modes $k$ (log-scale)", fontsize=15)
    plt.ylabel("$P_s(k)$ (log-scale)", fontsize=15)
    plt.yscale('log')
    plt.xscale('log')

    plt.legend()

    plt.text(place_params_at[0],place_params_at[1],f"loglogavgslope: ${k_mean} \pm {k_std}$ \nfluctuations: ${flucts_mean} \pm {flucts_std} $", horizontalalignment='left',)

    if x_scale is not None:
        plt.xlim(x_scale[0],x_scale[1])
    else:
        plt.xlim(place_params_at[0] - 5, place_params_at[0] + 10)
    if y_scale is not None:
        plt.ylim(y_scale[0],y_scale[1])
    else:
        pass


    if show:
        plt.show()
        plt.clf()
    if save:
        fig.set_size_inches(16, 9)
        groundpath = "figures/"
        if from_pickle_directory:
            path = path_extra + groundpath
        else:
            path = groundpath
        name = f"Posterior_Power_Spectrum"
        plt.savefig(path + name + ".png", dpi=300, bbox_inches='tight')
        print("Saved as " + name + ".png")
        plt.clf()
    if not show and not save:
        plt.clf()

    return (k_mean,k_std),(flucts_mean,flucts_std)



def real_data_plot(save:'bool', show:'bool',z_data:'np.ndarray',mu:'np.ndarray',keyword="Pantheon+", from_pickle_directory=False):
    """
            Returns a figure containing showing the redshift - moduli distribution of a given catalogue.

                Parameters
                ----------
                save        :type       :`~bool`        If True the figure is saved in the 'figures' folder.
                show        :type       :`~bool`        If True shows the figure via plt.show()
                mu          :type       :`~np.ndarray`  The moduli
                z_data      :type       :`~np.ndarray`  The domain values of the real data ('discrete' 'canonical' redshifts z)

                Returns
                ----------
                mpl.fig
        """
    plt.title(f"Modulus-Redshift distribution of {keyword} compilation",fontsize=20)
    plt.xlabel("'Canonical' redshift $z$", fontsize=15)
    plt.ylabel("Distance modulus $\mu$", fontsize=15)

    plt.plot(z_data, mu, markersize=8, marker="o", color="black", markerfacecolor='white', linewidth=0,
             label=f"{len(z_data)} datapoints ")
    plt.legend()

    if show:
        plt.show()
        plt.clf()
    if save:
        fig.set_size_inches(16, 9)
        groundpath = "figures/"
        if from_pickle_directory:
            path = path_extra + groundpath
        else:
            path = groundpath
        name = f"{keyword}_modulus_redshift_distribution"
        plt.savefig(path  + name + ".png", dpi=300, bbox_inches='tight')
        print("Saved as " + name + ".png")
        plt.clf()

    plt.clf()


def compare_data(save:'bool', show:'bool'):
    """
        Returns two figures: Moduli vs Redshift distribution side by side of Union2.1 and Pantheon+ Compilation,
        as well as a histogramm comparing the two.
        # Todo write the code nicer and complete documentation!

            Parameters
            ----------
            save        :type       :`~bool`        If True the figure is saved in the 'figures' folder.
            show        :type       :`~bool`        If True shows the figure via plt.show()

            Returns
            ----------
            mpl.figs
    """

    fig, ax = plt.subplots()

    redshifts1, moduli1, noise_data1, name1 = read_data(keyword="Pantheon+")
    redshifts2, moduli2, noise_data2, name2 = read_data(keyword="Union2.1")

    n_datapoints1 = len(redshifts1)
    n_datapoints2 = len(redshifts2)

    plt.subplot(2, 1, 1)

    plt.xlim(-0.25, max(redshifts1))
    plt.ylim(min(moduli1), max(moduli1))

    plt.plot(redshifts1, moduli1, markersize=8, marker="o",
                 color="darkred", markerfacecolor=red, linewidth=0,
                 label=f"datapoints ({n_datapoints1}) {name1}", alpha=0.5)
    plt.ylabel(r"Distance modulus $\mu$", fontsize=15)
    plt.legend()
    plt.subplot(2, 1, 2)
    plt.plot(redshifts2, moduli2, markersize=8, marker="o", color="darkblue", markerfacecolor=blue, linewidth=0,
             label=f"datapoints ({n_datapoints2}) {name2}", alpha=0.5)

    plt.xlabel("Redshift $z$", fontsize=15)
    plt.ylabel(r"Distance modulus $\mu$", fontsize=15)
    plt.xlim(-0.25, max(redshifts1))
    plt.ylim(min(moduli1), max(moduli1))
    plt.suptitle(fr"{name1} and {name2}: Distance moduli $\mu$ against redshifts $z$", fontsize=20)
    plt.legend()


    if save:
        fig.set_size_inches(16, 9)
        name = "Pantheon_Union_data_comparison"
        plt.savefig("figures/"+name + ".png", dpi=300, bbox_inches='tight')
        print("Saved as " + name + ".png")
    if show:
        plt.show()

    plt.clf()

    ###### Histogram Part

    fig, (ax1, ax2) = plt.subplots(1, 2)

    redshift_resol = 12
    maximum = max(redshifts1)
    bin_width = maximum / redshift_resol
    number_bins = redshift_resol

    union_density = [0] * number_bins
    pantheon_density = [0] * number_bins

    print("redshift resolution: ", redshift_resol, " ==> number of bins: ", number_bins, " each of width ", bin_width)
    print("max redshift : ", maximum)

    for j in range(1, number_bins + 1):
        upper_cutoff = bin_width * j
        bin_index = j - 1
        if bin_index == 0:
            for z in redshifts2:
                if z <= upper_cutoff:
                    union_density[bin_index] += 1
        else:
            prior_cutoff_level = bin_width * (j - 1)
            for z in redshifts2:
                if prior_cutoff_level < z <= upper_cutoff:
                    union_density[bin_index] += 1

    for j in range(1, number_bins + 1):
        upper_cutoff = bin_width * j
        bin_index = j - 1
        if bin_index == 0:
            for z in redshifts1:
                if z <= upper_cutoff:
                    pantheon_density[bin_index] += 1
        else:
            prior_cutoff_level = bin_width * (j - 1)
            for z in redshifts1:
                if prior_cutoff_level < z <= upper_cutoff:
                    pantheon_density[bin_index] += 1

    print("sanity checks: Sum of all density elements in array: ", sum(union_density), sum(pantheon_density), " should be 580 and 1701.")



    ax1.set_ylim(0, 960)
    ax1.set_xlim(-0.1, maximum + 0.1)
    clock = 0
    for frequency in pantheon_density:
        clock += 1
        central_position_of_bar = clock * bin_width - bin_width / 2
        if clock == 1:
            ax1.bar(x=central_position_of_bar, height=frequency, width=bin_width, color=red, ec=red, alpha=0.5,
                    label="Pantheon+")
        else:
            ax1.bar(x=central_position_of_bar, height=frequency, width=bin_width, color=red, ec=red, alpha=0.5)

    ax1.legend(loc=4)

    ax1.set_xlabel(f"{redshift_resol} redshift bins each of width {round(bin_width, 2)} z", fontsize=15)
    ax1.set_ylabel("Frequency", fontsize=15)

    ax2.set_ylim(0, 960)
    ax2.set_xlim(-0.1, maximum + 0.1)
    clock = 0
    for frequency in union_density:
        clock += 1
        central_position_of_bar = clock * bin_width - bin_width / 2
        if clock == 1:
            ax2.bar(x=central_position_of_bar, height=frequency, width=bin_width, color=blue, ec=blue, alpha=0.5,
                    label="Union2.1")
        else:
            ax2.bar(x=central_position_of_bar, height=frequency, width=bin_width, color=blue, ec=blue, alpha=0.5, )

    ax2.legend(loc=4)
    ax2.set_xlabel(f"{redshift_resol} redshift bins each of width {round(bin_width, 2)} z", fontsize=15)

    save = True

    plt.suptitle("Histogram: Frequency of data sorted in redshift bins", fontsize=20)

    axins1 = inset_axes(ax1, width="45%", height="50%", loc=1, borderpad=2)
    axins2 = inset_axes(ax2, width="45%", height="50%", loc=1, borderpad=2)

    axins1.set_ylim(0, 40)
    axins1.set_xlim(0.3, maximum + 0.1)

    axins2.set_ylim(0, 40)
    axins2.set_xlim(0.3, maximum + 0.1)

    clock = 0
    for frequency in union_density:
        clock += 1
        central_position_of_bar = clock * bin_width - bin_width / 2
        if clock >= 5:
            axins2.bar(x=central_position_of_bar, height=frequency, width=bin_width, color=blue, ec=blue, alpha=0.5)

    clock = 0
    for frequency in pantheon_density:
        clock += 1
        central_position_of_bar = clock * bin_width - bin_width / 2
        if clock >= 5:
            axins1.bar(x=central_position_of_bar, height=frequency, width=bin_width, color=red, ec=red, alpha=0.5)

    if save:
        fig.set_size_inches(16, 9)
        name = "Density_Distribution_of_data_Union_and_Pantheon"
        plt.savefig("figures/"+name + ".png", dpi=300, bbox_inches='tight')
        print("Saved as " + name + ".png")
    if show:
        plt.show()
    plt.clf()