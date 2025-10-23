statmorph-lsst
=========

An adaptation of the `statmorph <https://github.com/vrodgom/statmorph>`_ Python code for calculating non-parametric morphological diagnostics of
galaxy images. The changes are described in the publication *"statmorph-lsst: quantifying and correcting morphological biases in galaxy surveys"* (Sazonova et al., in prep.).


Documentation
-------------

The installation instructions can be found on
`ReadTheDocs <http://statmorph.readthedocs.io/en/latest/>`_.

Full documentation describing the changes to the parent package is in preparation.

Tutorial / How to use
---------------------

Please see the
`statmorph tutorial <https://statmorph.readthedocs.io/en/latest/notebooks/tutorial.html>`_.

Bias diagnostics and corrections
---------------------

You can see the dependence of each parameter in this suite in the `diagnostic_plots` folder. Bias corrections are
derived where possible with SymbolicRegression and are available in the paper (for now).

Citing
------

If you use this code for a scientific publication, please cite the following
article:

- `Rodriguez-Gomez et al. (2019) <https://ui.adsabs.harvard.edu/abs/2019MNRAS.483.4140R>`_
- `Sazonova et al. (in prep.)