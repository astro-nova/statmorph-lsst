statmorph-lsst
=========

An adaptation of the `statmorph <https://github.com/vrodgom/statmorph>`_ Python code for calculating non-parametric morphological diagnostics of
galaxy images. The changes are described in the publication *"statmorph-lsst: quantifying and correcting morphological biases in galaxy surveys"* (Sazonova et al., in prep.).


Documentation
-------------

You can install the package by cloning/downloading this repository, navigating there, and running `pip install .` The package will be uploaded to pip soon to make the installation simpler.

Full documentation describing the changes to the parent package is in preparation.
Major changes as of now:

* `isophote_asymmetry`: similar to shape asymmetry of `Pawlik et al. 2016 <https://ui.adsabs.harvard.edu/abs/2016MNRAS.456.3032P/abstract>`_, returns asymmetry of different flux isophotes given by `asymmetry_isophotes` argument. If None, isophotal asymmetry is not calculated. Ideally, isophotes should be defined by converting desired surface brightness limits to flux units.
* `substructure`: similar to Smoothness of `Conselice et al. 2003 <https://ui.adsabs.harvard.edu/abs/2003ApJS..147....1C/abstract>`_, with an additional step of detecting contiguous clumps on the smoothed residual. This is the same procedure as what is commonly used to find high-redshift clumps (e.g., `Shibuya 2016 <https://ui.adsabs.harvard.edu/abs/2016ApJ...821...72S/abstract>`_)

Tutorial / How to use
---------------------

Please see the
`statmorph tutorial <https://statmorph.readthedocs.io/en/latest/notebooks/tutorial.html>`_.

Bias diagnostics and corrections
---------------------

You can see the dependence of each parameter in this suite in the `diagnostic_plots <https://github.com/astro-nova/statmorph-lsst/tree/master/diagnostic_plots>`_ folder. Bias corrections are
derived where possible with SymbolicRegression and are available in the paper (for now).

Citing
------

If you use this code for a scientific publication, please cite the following
article:

- `Rodriguez-Gomez et al. (2019) <https://ui.adsabs.harvard.edu/abs/2019MNRAS.483.4140R>`_
- Sazonova et al. (in prep.)
