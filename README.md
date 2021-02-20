# constant-function-sims

Simple Python functions to easily simulate any constant function style AMMs.

## Description

This provides a modest number of succint functions that can be helful to easily prototype the design of new constant function style AMMs. The methods and formalism described in [Angeris and Chitra (2020)](https://arxiv.org/pdf/2003.10001.pdf) are used.

``invariants.py`` contains a set of classic invariants currently implemented by AMMs such as Uniswap, Balancer or Curve. 

``solver.py`` contains the functions needed to calculate quantities of interest such as the spot price for a given pair, the slippage, etc.

## Pre-requisites

- Python 3
- ``pip install numpy``
- ``pip install scipy``
