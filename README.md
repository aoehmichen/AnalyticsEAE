# AnalyticsEAE

The analytics eAE project is part of the [eTRIKS Analytical Environement (eAE)](https://eae.doc.ic.ac.uk/) project.

This project contains all the publicly available analytics which are part of the 1.0 release of the project.

At the moment, it includes three analytics:

1. The Cross-Validation analysis pipeline aims at providing an unbiased approach for automated model generation and validation.

2. The General Testing statistical analysis pipeline aims at providing statistical insights about the datasets for further research, without any prior statistical knowledge, by performing multiple statistical tests on a given data set.

3. The Pathway Enrichment analysis pipeline is a method to identify classes of genes  that are over-represented in a large set of pathways. It relies on [Fisher's exact test](https://en.wikipedia.org/wiki/Fisher%27s_exact_test) and [KEGG](http://www.genome.jp/kegg/).

All those analytics are available in the [tranSMART platform](https://github.com/transmart/transmartApp) through the [eAE plugin](https://github.com/aoehmichen/eAE) developed for the platform.
