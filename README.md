# PRISMO

[![codecov](https://codecov.io/gh/bioFAM/prismo/graph/badge.svg?token=IJP1IA4JEU)](https://codecov.io/gh/bioFAM/prismo)

PRISMO is a versatile factor analysis framework designed to streamline the construction and training of complex matrix factorisation models for omics data. PRISMO is a probabilistic programming-based Bayesian factor analysis framework that integrates concepts from multiple existing methods while remaining modular and extensible. It generalises widely used matrix factorisation tools by incorporating flexible prior options (including structured sparsity priors for multi-omics data and covariate-informed priors for spatio-temporal data), non-negativity constraints, and diverse data likelihoods - allowing users to mix and match components to suit their specific needs. Additionally, PRISMO introduces a novel module for integrating prior biological knowledge in the form of gene sets or, more generally, variable sets, enabling the inference of interpretable latent factors linked to specific molecular programs.

## Getting started

Please refer to the [documentation][link-docs]. In particular, the

- [API documentation][link-api].

## Installation

You need to have Python 3.10 or newer installed on your system. If you don't have
Python installed, we recommend installing [Micromamba](https://mamba.readthedocs.io/en/latest/installation/micromamba-installation.html).

There are several alternative options to install prismo:

<!--
1) Install the latest release of `prismo` from [PyPI][link-pypi]:

```bash
pip install prismo
```
-->

1. Install the latest development version:

```bash
pip install git+https://github.com/bioFAM/prismo.git@main
```

## Release notes

See the [changelog][changelog].

## Contact

For questions and help requests, you can reach out in the [scverse discourse][scverse-discourse].
If you found a bug, please use the [issue tracker][issue-tracker].

## Citation

> t.b.a

[issue-tracker]: https://github.com/bioFAM/prismo/issues
[changelog]: https://prismo.readthedocs.io/latest/changelog.html
[link-docs]: https://prismo.readthedocs.io
[link-api]: https://prismo.readthedocs.io/latest/api.html
[link-pypi]: https://pypi.org/project/prismo
