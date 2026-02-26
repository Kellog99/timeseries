# Lazy financial market: unified library for financial analysis

The idea of this library is to analyse all the stock options available on `Yahoo Finance` and create an analysis on
them.
It started as a library on generic time series and then switched this kind of analysis.

## Background

What is a multivariate timeseries?

**Definition**   
A multivariate timeseries is a finite part of a realization from a stochastic process $\{X_t,t\in T\}$
in $\mathbb{R}^k$, i.e. $\{X_t,t\in T_0\}\subseteq\mathbb{R}^k$ with $T_0\subset T$

In this case $\mathbb{R}^k$ is the result of a concatenation of 2 space:

1. $\mathcal{C} \subset \mathbf{N}^c$ the set of categorical variables.
2. $\mathcal{W} \subset \mathbf{R}^{k-c}$ the set of numerical variables

For this reason it is possible to write $\forall t\in T_0,
X_t=c_tw_t$ with $c_t\in\mathcal{C}, w_t\in \mathcal{W}$
where $c_tw_t$ means concatenation.

***   
**Important**  
We assume that the categorical variables are always known, i.e. $\mathcal{C}$ is known $\forall t$.

***   

## Structure

This folder is composed of

1. **timeseries**: all the models that will be used for forecasting the timeseries
2. **notebook**: for all the simple testing
3. **tests**: this folder contains all the test for executing the timeseries library
4. **services**: all the services that has to

## Timeseries

This folder contains all the models that have been implemented.

## Tests

This contains all the main test for implementing a new model in the regime.

## Services

This folder contains all the services that have to be used for sending all the information to the FE (front end).

