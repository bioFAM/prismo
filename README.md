# MOFA-FLEX

[![Tests][badge-tests]][tests]
[![codecov][badge-codecov]][codecov]
[![Documentation][badge-docs]][documentation]

[badge-tests]: https://github.com/bioFAM/mofaflex/actions/workflows/test.yaml/badge.svg
[badge-codecov]: https://codecov.io/gh/bioFAM/mofaflex/graph/badge.svg?token=IJP1IA4JEU
[badge-docs]: https://img.shields.io/readthedocs/mofaflex


![graphical abstract](https://raw.githubusercontent.com/bioFAM/mofaflex/main/docs/_static/img/mofaflex_schematic.svg)

MOFA-FLEX is a versatile factor analysis framework designed to streamline the construction and training of complex matrix factorisation models for omics data. MOFA-FLEX is a probabilistic programming-based Bayesian factor analysis framework that integrates concepts from multiple existing methods while remaining modular and extensible. It generalises widely used matrix factorisation tools by incorporating flexible prior options (including structured sparsity priors for multi-omics data and covariate-informed priors for spatio-temporal data), non-negativity constraints, and diverse data likelihoods - allowing users to mix and match components to suit their specific needs. Additionally, MOFA-FLEX introduces a novel module for integrating prior biological knowledge in the form of gene sets or, more generally, variable sets, enabling the inference of interpretable latent factors linked to specific molecular programs.

## Getting started

Please refer to the [documentation][]. In particular, the

- [API documentation][].

## Installation

You need to have Python 3.11 or newer installed on your system. If you don't have
Python installed, we recommend installing [Micromamba](https://mamba.readthedocs.io/en/latest/installation/micromamba-installation.html).

There are several alternative options to install MOFA-FLEX:

<!--
1) Install the latest release of MOFA-FLEX from [PyPI][]:

```bash
pip install mofaflex
```
-->

1. Install the latest development version:

```bash
pip install git+https://github.com/bioFAM/mofaflex.git@main
```

## Release notes

See the [changelog][].

## Contact

For questions and help requests, you can reach out in the [discussions][].
If you found a bug, please use the [issue tracker][].

## Citation

> t.b.a

[issue tracker]: https://github.com/bioFAM/mofaflex/issues
[tests]: https://github.com/bioFAM/mofaflex/actions/workflows/test.yaml
[codecov]: https://codecov.io/gh/bioFAM/mofaflex
[documentation]: https://mofaflex.readthedocs.io
[discussions]: https://github.com/bioFAM/mofaflex/discussions
[changelog]: https://mofaflex.readthedocs.io/latest/changelog.html
[api documentation]: https://mofaflex.readthedocs.io/en/latest/api.htm
[pypi]: https://pypi.org/project/mofaflex
