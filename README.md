CHARM2.0: Cosmic History Agnostic Reconstruction Method
=================
Reconstructing cosmic expansion history (cosmic energy density as a function of redshift) 
without assuming a specific type of matter distribution in the Universe. Data are distance moduli and
redshifts from Supernovae Ia from the 
[Union2.1](https://supernova.lbl.gov/Union/) compilation and [Pantheon+](https://pantheonplussh0es.github.io) Analysis.

Powered by [NIFTy](https://ift.pages.mpcdf.de/nifty/user/installation.html) (v8). 

Original [CHARM](https://gitlab.mpcdf.mpg.de/natalia/charm) by [Natalia Porqueres et al. 2017](https://arxiv.org/abs/1608.04007) + geoVI instead of an iterative MAP approach.


Requirements
=================
*   NIFTy can be installed using pip:

        pip install nifty8
* If necessary, after NIFTy installation: Numpy, Scipy, Pandas, pickle, matplotlib, seaborn 

> **Warning**

>When visualizing the power spectrum of the posterior by using the function
> `visualize_and_analyze_posterior_power_spectrum()`, the mean and uncertainty of 
> `fluctuations` and `loglogavgslope` is returned. This **only holds** if the parameters of the correlated field model
> have no standard deviation, i.e. the second parameter in the parameter tuple is e.g. $10^{-16}$. This parameters are found 
> by fitting the posterior power spectrum in a log-log scale and use a preliminary numerical relation that will be replaced in the 
> future, so interpret these values with caution. This numerical relation also only holds for fixed length of the signal 
> domain, which has been chosen here to be $x_{max}=6.7$.




Usage and Workflow
=================
*   Download a local copy of this project per the <span style="color:green"> green code button above </span> (e.g. as zip). Unpack, move into directory and run 

        python synthetic_catalogue.py
* There will be four matplotlib figures showing up, each need to be closed for the program to continue running
* Inference run with synthetic data: `synthetic_catalogue.py`. Inference run with real data: `cosmological_tomography.py`

.
├── ...
├── test                    # Test files (alternatively `spec` or `tests`)
│   ├── benchmarks          # Load and stress tests
│   ├── integration         # End-to-end, integration tests (alternatively `e2e`)
│   └── unit                # Unit tests
└── ...

Pickles WILL be overwritten if their name is not changed manually (if you want e.g. to play 
around with cf parameters.)
Explain how to deploy this project. Maybe minimum computer specifications or browser requirements are listed here as well.

