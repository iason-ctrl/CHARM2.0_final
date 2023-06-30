import numpy as np
from scipy.optimize import curve_fit
import pandas as pd
import nifty8 as ift
import pickle
import matplotlib.pyplot as plt

__all__ = ['read_data', 'build_cov_pantheon', 'pickle_me_this',
           'unpickle_me_this', 'radial_los',
           'domain_unity', 'get_power_spec_params', 'planck_cosmology',
           'conversion_from_fluctuations_to_offset', 'conversion_from_offset_to_fluctuations',
           'MatrixApplyInverseNoise']

# -------  Data Readers  ------- #

def read_data(keyword="Pantheon+", from_pickle_directory=False):
    if from_pickle_directory:
        path = "../"
    else:
        path=""
    if keyword=="Pantheon+":
        required_fields = ["zHEL", "MU_SH0ES"]
        df = pd.read_csv(path+"raw_data/Pantheon+SH0ES.csv", sep=" ", skipinitialspace=True, usecols=required_fields)
        redshifts_heliocentric = np.array(df.zHEL)
        moduli_real = np.array(df.MU_SH0ES)
        name=keyword
        return redshifts_heliocentric, moduli_real, [], name
    elif keyword=="Union2.1":
        z = np.loadtxt(path+"raw_data/Union_2_1_data.txt", usecols=[1], dtype=float) # read data and sort by redshift
        index = np.argsort(z)
        redshift = z[index]
        mu = np.loadtxt(path+"raw_data/Union_2_1_data.txt", usecols=[2], dtype=float)[index]
        Noise = np.loadtxt(path+"raw_data/Union2_1_Cov-syst.txt", dtype=float)
        name = keyword
        return np.array(redshift), np.array(mu), np.array(Noise), name

def build_cov_pantheon():
    path = "raw_data/Pantheon+SH0ES_STAT+SYS.cov"
    df = pd.read_table(path)
    arr = np.array(df["1701"]) # first line corresponds to '1701' = length of data array
    cov_matrix = arr.reshape(1701,1701)
    return cov_matrix


# -------  Data De- and Encoders  ------- #


def pickle_me_this(filename, data_to_pickle):
    path = "pickles/" + filename +".pickle"
    file = open(path, 'wb')
    pickle.dump(data_to_pickle, file)
    file.close()

def unpickle_me_this(filename):
    file = open(filename, 'rb')
    data = pickle.load(file)
    file.close()
    return data


# -------  Random radial line of sight generator. LOS are distributed normally  ------- #

ift.random.push_sseq_from_seed(19)

def radial_los(n_los):
    starts = [np.zeros(n_los)]
    ends = [np.array(abs(ift.random.current_rng().normal(0.05,0.4,n_los)))]
    return starts, ends


# -------  Unit operator acting on a domain and returning the domains values.   ------- #

def domain_unity(signal_domain, return_domain=None):
    """
    Returns a field with the domain values of the input signal correlated field domain.
    Parameters. If return_domain = None, signal_domain is used.

        Parameters
        ----------
        signal_domain :class :`~nifty8.domain.Domain` the domain on which the correlated field is defined
        return_domain :class :`~nifty8.domain.Domain` the domain on which the returned field is defined.

        Returns
        ----------
        domain_coordinate_field :class :`~nifty8.Field` a Field containing the equidistantly spaced domain values

    """
    size = signal_domain.distances[0]
    values = []
    for iteration in range(1, signal_domain.size + 1):
        values.append(iteration * size)
    try:
        if not return_domain:
            dom = ift.DomainTuple.make(signal_domain,)
        else:
            dom = ift.DomainTuple.make(signal_domain,)
    except:
        print("Can't make DomainTuple out of return_domain. Assuming unstructured domain.")
        dom = return_domain
    domain_coordinate_field = ift.Field(dom, val=np.array(values))
    return domain_coordinate_field


# -------  Posterior Analysis  ------- #

def get_power_spec_params(power_spectrum:'np.ndarray',plot=False):
    """
        Takes a power spectrum of the posterior and curve_fits() to exponent *x + offset on the log-log scale. Here, 'exponent'
        is the power law component of the power spectrum and 'offset' is intersection of the power spectrum with the y-axis
        on the log-log scale. The y-offset parameter can be translated into the 'fluctuations' parameter.
        This y-offset is the intersection of the graph with the y-axis, if the base 10 log of the power spectrum is drawn
        against the base 10 log of its index set N

            Parameters
            ----------
            power_spectra   :type :`~np.ndarray`    A power spectrum sample, containing its values.
            plot            :type :`~bool`          If True, plot the base 10 log of the power spectrum against the base 10 log of its index set N

            Returns
            ----------
            params          :type :`~tuple`         Tuple containing power law exponent and log-log-scale offset of the spectrum.
    """


    def linear(x,a,b):
        return a*x+b

    pow_log = np.log10(power_spectrum)[1:len(power_spectrum)]                   # cut out and IGNORE the first element (-inf)
    k_modes_log = np.log10(range(len(power_spectrum)))[1:len(power_spectrum)]   # cut out and IGNORE the first element (-inf)

    if plot:
        # This will only work for the first power spectrum sample, then it breaks
        plt.plot(k_modes_log,pow_log,"b.")
        plt.show()
        plt.clf()

    popt = curve_fit(linear,k_modes_log,pow_log)
    params = popt[0]
    covariance_matrix = popt[1]


    return params


def planck_cosmology(signal_space, signal_coordinate_field):
    # Returns the signal field for a LambdaCDM model constraint by the cosmological parameters
    # of the planck mission
    def planck_curve(x):
        return np.log(0.314*np.exp(3*x)+0.686)
    values = planck_curve(signal_coordinate_field)
    return ift.Field(domain=ift.DomainTuple.make(signal_space,), val=values)

def conversion_from_fluctuations_to_offset(fluct):
    y_offset =  2*np.log10(fluct) + 1.32 # y-offset on log-log scale
    return y_offset

def conversion_from_offset_to_fluctuations(y_offset):
    fluct = 10**(0.5* (y_offset - 1.32))
    return fluct

class MatrixApplyInverseNoise(ift.MatrixProductOperator):
    # TODO write nicer documentation

    def __init__(self, domain, matrix, spaces=None, flatten=False, sampling_dtype=None):
        """
        `matrix` is the non-inverted Noise Covariance matrix. It will be inverted here.
        """
        super(MatrixApplyInverseNoise, self).__init__(domain, np.linalg.inv(matrix), spaces=spaces, flatten=flatten)
        self._dtype = sampling_dtype


    def diagonalize(self):
        """ Construct diagonal matrix and its transformation matrices with numpy according to
            A = U A_diag U^{-1}     where we assume real entries throughout
        """
        A = self._mat
        if not np.allclose(A, A.T, rtol=1e-2,atol=1e-8):
            raise ValueError("Matrix is not symmetric, which is assumed in the diagonalization process (use of np.linalg.eigh instead of np.linalg.eig) ")
        rows, columns = A.shape
        if rows!= columns:
            raise ValueError("Matrix is not square.")

        eigenvalues, eigenvectors = np.linalg.eigh(A) # eigh is specifically good for symmetric matrices

        A_diag = np.diag(eigenvalues)
        U = eigenvectors
        U_inv = np.linalg.inv(U)

        A_recomposed = U @ A_diag @ U_inv
        sanity_check = np.allclose(A, A_recomposed,rtol=1e-2,atol=1e-8)

        if not sanity_check:
            raise ValueError("Matrix is not element-wise almost equal to its decomposition in diagonalized and transformation matrices.")

        return U, A_diag, U_inv


    def _get_fct(self, from_inverse):
        if not isinstance(self._mat, np.ndarray):
            raise ValueError(
                "For application of a full noise covariance, the matrix values have to be of type numpy ndarray."
                "Not of type" + type(self._mat))
        U, A_diag, U_inv = self.diagonalize()
        eigenvalues = A_diag.flatten()
        if any(fct.imag != 0. or fct.real < 0. or
               (fct.real == 0. and from_inverse) for fct in eigenvalues):
            raise ValueError("diagonal operator not positive definite. Bad behaved Noise matrix.")
        mtrx1 = 1. / np.sqrt(A_diag)
        mtrx2 = np.sqrt(A_diag)
        return U @ mtrx1 @ U_inv if from_inverse else U @ mtrx2 @ U_inv

    def get_sqrt(self):
        mtrx_sqrt = self._get_fct(False)

        flattened_mtrx = mtrx_sqrt.flatten()
        if any(np.iscomplexobj(fct) for fct in flattened_mtrx):
            raise ValueError("get_sqrt() works only for positive definite operators. Detected complex entry.")
        return MatrixApplyInverseNoise(self._domain, mtrx_sqrt)