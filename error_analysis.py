#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" This module provides all experiments involving the analysis and
interpretation of GBRT. Each experiment is implemented as a self contained
`exp_X` method (only input argument is the entire global and gris data set)
that then delegates everything to a configurable `plot_X` method. All plot and
dump files are handled via `util.save_cur_fig`, `util.pickle_dump`,
`util.pickle_load`.

All `plot_X` methods accept two convenience keyword arguments:

- `dumpfile`: the path of data dump file for this experiment. This file is
  written to if `replot` (below) is falsey and is read from if `replot` is
  truthy. The default is None which means the results of the experiment are not
  stored on disk (None is unacceptable for `replot=True`).
- `replot`: whether or not to replot from an existing data `dumpfile` (must be
  provided if truthy) or to re-run the experiment.

Note that `dumpfile=foo, replot=False` overwrites any existing data in `foo`.

If this module is run as main all experiments that were presented in the
accompanying paper are run from scratch. Other experiments can be run by
uncommenting their invocation.
"""

import sys
from random import randint
from itertools import compress
from itertools import combinations
from sklearn.ensemble import partial_dependence
from sklearn.feature_selection import RFE
from util import (
    plt,
    np,
    mean_squared_error,
    load_global_data,
    save_cur_fig,
    pickle_dump,
    pickle_load,
    split_with_circle,
    split_by_distance,
    tune_params,
    train_gbrt,
    get_gbrt,
    train_linear,
    error_summary,
    random_prediction_ctr,
    greenland_train_test_sets,
    CATEGORICAL_FEATURES,
    GREENLAND_RADIUS,
    FEATURE_NAMES,
    PROXIMITY_FEATURES,
)


def compare_models(data, roi_density, radius, center, **gbrt_params):
    """ For a fixed sample density, ROI center, and ROI radius, splits the data
        set into a training and validation set and returns the measures of
        error (normalized rmse and r2) of GBRT, linear regression, and constant
        predictor.

        Args:
            data (pandas.DataFrame): entire data set to use.
            roi_density (float): required sample density in ROI.
            radius (float): ROI radius in km.
            center (tuple): longitutde-latitude coordinates of ROI center.

        Return:
            dict: keys are 'gbrt', 'linear', and 'constant', values are r2 and
                  rmse pairs as produced by `error_summary`.
        """
    X_train, y_train, X_test, y_test = \
        split_with_circle(data, center, roi_density=roi_density, radius=radius)
    assert not X_test.empty

    X_train = X_train.drop(['Latitude_1', 'Longitude_1'], axis=1)
    X_test = X_test.drop(['Latitude_1', 'Longitude_1'], axis=1)

    # consider 3 predictors: GBRT, linear regression, and a constant predictor
    gbrt = train_gbrt(X_train, y_train, **gbrt_params)
    y_gbrt = gbrt.predict(X_test)

    lin_reg = train_linear(X_train, y_train)
    y_lin = lin_reg.predict(X_test)

    y_const = y_train.mean() + np.zeros(len(y_test))
    # error_summary returns (r2, rmse) pairs
    return {'gbrt': error_summary(y_test, y_gbrt),
            'linear':  error_summary(y_test, y_lin),
            'constant': error_summary(y_test, y_const)}

def plot_error_by_density(data, roi_densities, radius, ncenters, region='NA-WE',
                          replot=False, dumpfile=None, **gbrt_params):
    """ ncenters random centers are picked and over all given ROI densities.
        Cross-validation errors (normalized RMSE and r2) are averaged over
        ncenters. One standard deviation mark is shown by a shaded region.
    """
    sys.stderr.write('=> Experiment: Error by Density (region: %s, no. centers: %d, no. densities: %d)\n' %
                     (region, ncenters, len(roi_densities)))
    fig = plt.figure(figsize=(11,5))
    ax_rmse, ax_r2 = fig.add_subplot(1, 2, 1), fig.add_subplot(1, 2, 2)

    if replot:
        results = pickle_load(dumpfile)
    else:
        centers = [
            random_prediction_ctr(data, radius, region=region, min_density=max(roi_densities))
            for _ in range(ncenters)
        ]
        shape = (ncenters, len(roi_densities))
        # blank error matrix (keyed by center number and roi density index),
        # used to initialize multiple components of the results dictionary.
        blank = np.zeros(shape)

        results = {
            'ncenters': ncenters,
            'roi_densities': roi_densities,
            'errors': {
                'gbrt': {'rmse': blank.copy(), 'r2': blank.copy()},
                'linear': {'rmse': blank.copy(), 'r2': blank.copy()},
                'constant': {'rmse': blank.copy(), 'r2': blank.copy()},
            },
        }
        for idx_density, roi_density in enumerate(roi_densities):
            for idx_ctr, center in enumerate(centers):
                sys.stderr.write('# density = %.2f, center %d/%d ' % (roi_density, idx_ctr + 1, ncenters))
                comp = compare_models(data, roi_density, radius, center, **gbrt_params)
                for k in results['errors'].keys():
                    # k is one of gbrt, linear, or constant
                    results['errors'][k]['r2'][idx_ctr][idx_density] = comp[k][0]
                    results['errors'][k]['rmse'][idx_ctr][idx_density] = comp[k][1]
        if dumpfile:
            pickle_dump(dumpfile, results, comment='GBRT performance results')

    errors = results['errors']
    roi_densities = results['roi_densities']
    ncenters = results['ncenters']
    num_sigma = 1

    # Plot GBRT results
    kw = {'alpha': .9, 'lw': 1, 'marker': 'o', 'markersize': 4, 'color': 'b'}
    mean_rmse = errors['gbrt']['rmse'].mean(axis=0)
    sd_rmse = np.sqrt(errors['gbrt']['rmse'].var(axis=0))
    lower_rmse = mean_rmse - num_sigma * sd_rmse
    higher_rmse = mean_rmse + num_sigma * sd_rmse
    ax_rmse.plot(roi_densities, mean_rmse, label='GBRT', **kw)
    ax_rmse.fill_between(roi_densities, lower_rmse, higher_rmse, facecolor='b', edgecolor='b', alpha=.3)

    mean_r2 = errors['gbrt']['r2'].mean(axis=0)
    sd_r2 = np.sqrt(errors['gbrt']['r2'].var(axis=0))
    lower_r2 = mean_r2 - num_sigma * sd_r2
    higher_r2 = mean_r2 + num_sigma * sd_r2
    ax_r2.plot(roi_densities, errors['gbrt']['r2'].mean(axis=0), **kw)
    ax_r2.fill_between(roi_densities, lower_r2, higher_r2, facecolor='b', edgecolor='b', alpha=.2)

    # Plot Linear Regression results
    kw = {'alpha': .7, 'lw': 1, 'marker': 'o', 'markersize': 4, 'markeredgecolor': 'r', 'color': 'r'}
    mean_rmse = errors['linear']['rmse'].mean(axis=0)
    sd_rmse = np.sqrt(errors['linear']['rmse'].var(axis=0))
    lower_rmse = mean_rmse - num_sigma * sd_rmse
    higher_rmse = mean_rmse + num_sigma * sd_rmse
    ax_rmse.plot(roi_densities, mean_rmse, label='linear regression', **kw)
    ax_rmse.fill_between(roi_densities, lower_rmse, higher_rmse, facecolor='r', edgecolor='r', alpha=.3)

    mean_r2 = errors['linear']['r2'].mean(axis=0)
    sd_r2 = np.sqrt(errors['linear']['r2'].var(axis=0))
    lower_r2 = mean_r2 - num_sigma * sd_r2
    higher_r2 = mean_r2 + num_sigma * sd_r2
    ax_r2.plot(roi_densities, errors['linear']['r2'].mean(axis=0), **kw)
    ax_r2.fill_between(roi_densities, lower_r2, higher_r2, facecolor='r', edgecolor='r', alpha=.2)

    # Plot constant predictor results
    kw = {'alpha': .7, 'lw': 1, 'ls': '--', 'marker': 'o', 'markersize': 4, 'color': 'k', 'markeredgecolor': 'k'}
    ax_rmse.plot(roi_densities, errors['constant']['rmse'].mean(axis=0), label='constant predictor', **kw)
    ax_r2.plot(roi_densities, errors['constant']['r2'].mean(axis=0), **kw)

    # Style plot
    ax_rmse.set_ylabel('Normalized RMSE', fontsize=14)
    ax_r2.set_ylabel('$r^2$', fontsize=16)
    ax_r2.set_ylim(-.05, 1)
    ax_r2.set_xlim(min(roi_densities) - 5, max(roi_densities) + 5)
    ax_r2.set_yticks(np.arange(0, 1.01, .1))
    ax_rmse.set_ylim(0, .5)
    ax_rmse.set_yticks(np.arange(0, .51, .05))
    ax_rmse.set_xlim(*ax_r2.get_xlim())
    for ax in [ax_rmse, ax_r2]:
        # FIXME force xlims to be the same
        ax.set_xlabel('density of training points in ROI ($10^{-6}$ km $^{-2}$)',
                      fontsize=14)
        ax.grid(True)
    ax_rmse.legend(prop={'size':15}, numpoints=1)
    fig.tight_layout()

def plot_error_by_radius(data, roi_density, radii, ncenters, region='NA-WE',
                         replot=False, dumpfile=None, **gbrt_params):
    """ ncenters random centers are picked and over all given radii.
        Cross-validation errors (normalized RMSE and r2) are averaged over
        ncenters. One standard deviation mark is shown by a shaded region.
    """
    fig = plt.figure(figsize=(11,5))
    ax_rmse, ax_r2 = fig.add_subplot(1, 2, 1), fig.add_subplot(1, 2, 2)

    if replot:
        results = pickle_load(dumpfile)
    else:
        centers = [
            # HACK there's no easy way to check if for a given center the
            # demanded density is attainable for circles of all desired radii.
            # Ask for twice the density we need on the largest radius and hope
            # for the best!
            random_prediction_ctr(data, max(radii), region=region, min_density=2*roi_density)
            for _ in range(ncenters)
        ]
        shape = (ncenters, len(radii))
        # blank error matrix (keyed by center number and roi density index),
        # used to initialize multiple components of the results dictionary.
        blank = np.zeros(shape)

        results = {
            'ncenters': ncenters,
            'radii': radii,
            'errors': {
                'gbrt': {'rmse': blank.copy(), 'r2': blank.copy()},
                'linear': {'rmse': blank.copy(), 'r2': blank.copy()},
                'constant': {'rmse': blank.copy(), 'r2': blank.copy()},
            },
        }
        for idx_radius, radius in enumerate(radii):
            for idx_ctr, center in enumerate(centers):
                sys.stderr.write('# radius = %.0f, center %d/%d ' % (radius, idx_ctr + 1, ncenters))
                comp = compare_models(data, roi_density, radius, center, **gbrt_params)
                for k in results['errors'].keys():
                    # k is one of gbrt, linear, or constant
                    results['errors'][k]['r2'][idx_ctr][idx_radius] = comp[k][0]
                    results['errors'][k]['rmse'][idx_ctr][idx_radius] = comp[k][1]
        if dumpfile:
            pickle_dump(dumpfile, results, comment='GBRT performance results')

    errors = results['errors']
    radii = results['radii']
    ncenters = results['ncenters']

    num_sigma = 1

    # Plot GBRT results
    kw = {'alpha': .9, 'lw': 1, 'marker': 'o', 'markersize': 4, 'color': 'b'}
    mean_rmse = errors['gbrt']['rmse'].mean(axis=0)
    sd_rmse = np.sqrt(errors['gbrt']['rmse'].var(axis=0))
    lower_rmse = mean_rmse - num_sigma * sd_rmse
    higher_rmse = mean_rmse + num_sigma * sd_rmse
    ax_rmse.plot(radii, mean_rmse, label='GBRT', **kw)
    ax_rmse.fill_between(radii, lower_rmse, higher_rmse, facecolor='b', edgecolor='b', alpha=.3)

    mean_r2 = errors['gbrt']['r2'].mean(axis=0)
    sd_r2 = np.sqrt(errors['gbrt']['r2'].var(axis=0))
    lower_r2 = mean_r2 - num_sigma * sd_r2
    higher_r2 = mean_r2 + num_sigma * sd_r2
    ax_r2.plot(radii, errors['gbrt']['r2'].mean(axis=0), **kw)
    ax_r2.fill_between(radii, lower_r2, higher_r2, facecolor='b', edgecolor='b', alpha=.2)

    # Plot Linear Regression results
    kw = {'alpha': .7, 'lw': 1, 'marker': 'o', 'markersize': 4, 'markeredgecolor': 'r', 'color': 'r'}
    mean_rmse = errors['linear']['rmse'].mean(axis=0)
    sd_rmse = np.sqrt(errors['linear']['rmse'].var(axis=0))
    lower_rmse = mean_rmse - num_sigma * sd_rmse
    higher_rmse = mean_rmse + num_sigma * sd_rmse
    ax_rmse.plot(radii, mean_rmse, label='linear regression', **kw)
    ax_rmse.fill_between(radii, lower_rmse, higher_rmse, facecolor='r', edgecolor='r', alpha=.3)

    mean_r2 = errors['linear']['r2'].mean(axis=0)
    sd_r2 = np.sqrt(errors['linear']['r2'].var(axis=0))
    lower_r2 = mean_r2 - num_sigma * sd_r2
    higher_r2 = mean_r2 + num_sigma * sd_r2
    ax_r2.plot(radii, errors['linear']['r2'].mean(axis=0), **kw)
    ax_r2.fill_between(radii, lower_r2, higher_r2, facecolor='r', edgecolor='r', alpha=.2)

    # Plot constant predictor results
    kw = {'alpha': .7, 'lw': 1, 'ls': '--', 'marker': 'o', 'markersize': 4, 'color': 'k', 'markeredgecolor': 'k'}
    ax_rmse.plot(radii, errors['constant']['rmse'].mean(axis=0), label='constant predictor', **kw)
    ax_r2.plot(radii, errors['constant']['r2'].mean(axis=0), **kw)

    # Style plot
    ax_rmse.set_ylabel('Normalized RMSE', fontsize=14)
    ax_r2.set_ylabel('$r^2$', fontsize=16)
    ax_r2.set_ylim(-.05, 1)
    ax_r2.set_xlim(min(radii) - 100, max(radii) + 100)
    ax_r2.set_yticks(np.arange(0, 1.01, .1))
    ax_rmse.set_ylim(0, .5)
    ax_rmse.set_yticks(np.arange(0, .51, .05))
    ax_rmse.set_xlim(*ax_r2.get_xlim())
    for ax in [ax_rmse, ax_r2]:
        # FIXME force xlims to be the same
        ax.set_xlabel('radius of ROI (km)',
                      fontsize=14)
        ax.grid(True)
    ax_rmse.legend(prop={'size':15}, numpoints=1)
    fig.tight_layout()

def plot_sensitivity_analysis(data, roi_density, radius, noise_amps, ncenters,
                              replot=False, dumpfile=None):
    """ For each given noise amplitude, performs cross-validation on ncenters
        with given radius and density, the average over ncenters of
        normalized rmse between noise-free predictions and predictions based on
        noisy GHF is calculated. This perturbation in predictions is plotted
        against the expected absolute value of applied noise (amplitude).

        Both GBRT and linear regression are considered.
        One standard deviation is indicated by a shaded region.
        The case of Greenland is considered separately and overlayed.
    """
    fig = plt.figure(figsize=(10, 5))
    ax_gbrt = fig.add_subplot(1, 2, 1)
    ax_lin = fig.add_subplot(1, 2, 2)

    def _predict(X_train, y_train, X_test, noise_amp):
        # If noise ~ N(0, s^2), then mean(|noise|) = s * sqrt(2/pi),
        # cf. https://en.wikipedia.org/wiki/Half-normal_distribution
        # To get noise with mean(|noise|) / mean(y) = noise_ampl, we need to
        # have noise ~ N(0, s*^2) with s* = mean(y) * noise_ampl * sqrt(pi/2).
        noise = np.mean(y_train) * noise_amp * np.sqrt(np.pi/ 2) * np.random.randn(len(y_train))
        gbrt = train_gbrt(X_train.drop(['Latitude_1', 'Longitude_1'], axis=1),
                          y_train + noise)
        lin_reg = train_linear(X_train.drop(['Latitude_1', 'Longitude_1'], axis=1),
                               y_train + noise)
        gbrt_pred = gbrt.predict(X_test.drop(['Latitude_1', 'Longitude_1'], axis=1))
        lin_pred = lin_reg.predict(X_test.drop(['Latitude_1', 'Longitude_1'], axis=1))
        return gbrt_pred, lin_pred

    if replot:
        res = pickle_load(dumpfile)
        rmses_gbrt, rmses_lin = res['rmses_gbrt'], res['rmses_lin']
        noise_amps = res['noise_amps']
    else:
        centers = [random_prediction_ctr(data, radius, min_density=roi_density)
                   for _ in range(ncenters)]
        y0 = []
        centers = [None] + centers # one extra "center" (Greenland)
        rmses_gbrt = np.zeros((len(centers), len(noise_amps)))
        rmses_lin = np.zeros((len(centers), len(noise_amps)))
        for idx_ctr, center in enumerate(centers):
            if center is None:
                # Greenland case
                X_train, y_train, X_test = greenland_train_test_sets()
            else:
                X_train, y_train, X_test, _ = \
                    split_with_circle(data, center, roi_density=roi_density, radius=radius)
            sys.stderr.write('(ctr %d) noise_amp = 0.00 ' % (idx_ctr + 1))
            y0_gbrt, y0_lin = _predict(X_train, y_train, X_test, 0)
            for idx_noise, noise_amp in enumerate(noise_amps):
                sys.stderr.write('(ctr %d) noise_amp = %.2f ' % (idx_ctr + 1, noise_amp))
                y_gbrt, y_lin = _predict(X_train, y_train, X_test, noise_amp)
                rmse_gbrt = sqrt(mean_squared_error(y0_gbrt, y_gbrt)) / np.mean(y0_gbrt)
                rmse_lin = sqrt(mean_squared_error(y0_lin, y_lin)) / np.mean(y0_lin)
                rmses_gbrt[idx_ctr][idx_noise] = rmse_gbrt
                rmses_lin[idx_ctr][idx_noise] = rmse_lin

        if dumpfile:
            res = {'rmses_lin': rmses_lin, 'rmses_gbrt': rmses_gbrt, 'noise_amps': noise_amps}
            pickle_dump(dumpfile, res, 'sensitivity analysis')

    noise_amps = np.append([0], noise_amps)
    for idx in range(ncenters+1):
        if idx == 0:
            # Greenland case
            kw = dict(color='g', alpha=.5, lw=2.5, marker='o',
                      markeredgewidth=0.0, label='Greenland')
            ax_lin.plot(noise_amps, np.append([0], rmses_lin[0]), **kw)
            ax_gbrt.plot(noise_amps, np.append([0], rmses_gbrt[0]), **kw)
        else:
            kw = dict(color='k', alpha=.2, lw=1)
            ax_lin.plot(noise_amps, np.append([0], rmses_lin[idx]), **kw)
            ax_gbrt.plot(noise_amps, np.append([0], rmses_gbrt[idx]), **kw)

    kw = dict(alpha=.9, lw=1.5, marker='o', color='k', label='global average')

    num_sigma = 1
    mean_rmse = np.append([0], rmses_lin[1:].mean(axis=0))
    sd_rmse = np.sqrt(np.append([0], rmses_lin[1:]).var(axis=0))
    lower_rmse = mean_rmse - num_sigma * sd_rmse
    higher_rmse = mean_rmse + num_sigma * sd_rmse
    ax_lin.plot(noise_amps, mean_rmse, **kw)
    ax_lin.fill_between(noise_amps, lower_rmse, higher_rmse, facecolor='k', edgecolor='k', alpha=.2)

    mean_rmse = np.append([0], rmses_gbrt[1:].mean(axis=0))
    sd_rmse = np.sqrt(np.append([0], rmses_gbrt[1:]).var(axis=0))
    lower_rmse = mean_rmse - num_sigma * sd_rmse
    higher_rmse = mean_rmse + num_sigma * sd_rmse
    ax_gbrt.plot(noise_amps, mean_rmse, **kw)
    ax_gbrt.fill_between(noise_amps, lower_rmse, higher_rmse, facecolor='k', edgecolor='k', alpha=.2)

    for ax in [ax_gbrt, ax_lin]:
        ax.set_xlabel('Relative magnitude of noise in training GHF', fontsize=12)
        ax.set_xlim(0, max(noise_amps) * 1.1)
        ax.set_aspect('equal')
        ax.grid(True)
        ax.set_xticks(np.arange(0, .35, .05))
        ax.set_yticks(np.arange(0, .35, .05))
        ax.set_xlim(-.025, .325)
        ax.set_ylim(-.025, .325)
        ax.legend(loc=1, fontsize=12)
    ax_gbrt.set_ylabel(r'Normalized RMSE difference in $\widehat{GHF}_{\mathrm{GBRT}}$', fontsize=12)
    ax_lin.set_ylabel(r'Normalized RMSE difference in $\widehat{GHF}_{\mathrm{lin}}$', fontsize=12)

    fig.tight_layout()


def plot_generalization_analysis(data, roi_density, radius, ncenters,
                                 ns_estimators, replot=False, dumpfile=None):
    """ For all given values for n_estimators (number of trees) for GBRT,
        perform cross-validation over ncenters ROIs with given radius and
        sample density. The average training and validation error for each
        number of trees is plotted. This is the standard plot to detect
        overfitting defined as the turning point beyond which validation error
        starts increasing while training error is driven down to zero. As
        expected, GBRT does not overfit (validation error plateaus).

        One standard deviation is indicated by a shaded region.
    """
    fig, ax = plt.subplots()

    if replot:
        res = pickle_load(dumpfile)
        for v in ['roi_density', 'radius', 'ns_estimators', 'train_rmses', 'test_rmses']:
            exec('%s = res["%s"]' % (v, v))
        assert len(train_rmses) == len(test_rmses), \
               'array length (# of centers) should be the same for training and test'
    else:
        sys.stderr.write('=> Experiment: Generalization ' + \
                         '(roi_density: %.2f, radius: %.2f,' % (roi_density, radius) +
                         ' no. centers: %d, no. of n_estimators: %d)\n' % (ncenters, len(ns_estimators)))
        centers = [random_prediction_ctr(data, radius, min_density=roi_density)
                   for _ in range(ncenters)]

        train_rmses = np.zeros([ncenters, len(ns_estimators)])
        test_rmses = np.zeros([ncenters, len(ns_estimators)])
        for center_idx, center in enumerate(centers):
            sys.stderr.write('# center %d/%d\n' % (center_idx + 1, ncenters))
            X_train, y_train, X_test, y_test = \
                split_with_circle(data, center, roi_density=roi_density, radius=radius)
            X_train = X_train.drop(['Latitude_1', 'Longitude_1'], axis=1)
            X_test = X_test.drop(['Latitude_1', 'Longitude_1'], axis=1)
            assert not X_test.empty

            for n_idx, n in enumerate(ns_estimators):
                sys.stderr.write('  # n_estimators: %d ' % n)
                gbrt = train_gbrt(X_train, y_train, n_estimators=n)
                _, train_rmse = error_summary(y_train, gbrt.predict(X_train))
                _, test_rmse  = error_summary(y_test, gbrt.predict(X_test))
                train_rmses[center_idx][n_idx] = train_rmse
                test_rmses[center_idx][n_idx] = test_rmse

        if dumpfile:
            res = {'roi_density': roi_density,
                   'radius': radius,
                   'ns_estimators': ns_estimators,
                   'train_rmses': train_rmses,
                   'test_rmses': test_rmses}
            pickle_dump(dumpfile, res, comment='generalization errors')

    num_sigma = 1

    mean_rmse = test_rmses.mean(axis=0)
    sd_rmse = np.sqrt(test_rmses.var(axis=0))
    lower_rmse = mean_rmse - num_sigma * sd_rmse
    higher_rmse = mean_rmse + num_sigma * sd_rmse
    ax.plot(ns_estimators, mean_rmse, 'r', marker='o', markersize=3, alpha=.9, label='validation')
    ax.fill_between(ns_estimators, lower_rmse, higher_rmse, facecolor='r', edgecolor='r', alpha=.3)

    mean_rmse = train_rmses.mean(axis=0)
    sd_rmse = np.sqrt(train_rmses.var(axis=0))
    lower_rmse = mean_rmse - num_sigma * sd_rmse
    higher_rmse = mean_rmse + num_sigma * sd_rmse
    ax.plot(ns_estimators, mean_rmse, 'g', marker='o', markersize=3, alpha=.9, label='training')
    ax.fill_between(ns_estimators, lower_rmse, higher_rmse, facecolor='g', edgecolor='g', alpha=.3)

    ax.grid(True)
    ax.set_xlim(ns_estimators[0] - 100, ns_estimators[-1] + 100)
    ax.set_ylim(0, .3)
    ax.set_yticks(np.arange(0, .31, .05))
    ax.set_xlabel('Number of trees', fontsize=16)
    ax.set_ylabel('Normalized RMSE', fontsize=16)
    ax.legend(prop={'size':12.5})
    fig.tight_layout()

# FIXME update names in dumpfile

def plot_feature_importance_analysis(data, roi_density, radius, ncenters,
                                     dumpfile=None, replot=False, **gbrt_params):
    """ Plots feature importance results (cf. Friedman 2001 or ESL) averaged
        over ncenters rounds of cross validation for given ROI training density
        and radius.
    """
    raw_features = list(data)
    for f in ['Latitude_1', 'Longitude_1', 'GHF']:
        raw_features.pop(raw_features.index(f))

    # a map to collapse categorical dummies for feature importances. The dict
    # has keys in `raw_features` indices, and values in `features` indices.
    decat_by_raw_idx = {}
    features = []
    for idx, f in enumerate(raw_features):
        match = [c for c in CATEGORICAL_FEATURES if c == f[:len(c)]]
        if match:
            assert len(match) == 1
            try:
                i = features.index(match[0])
            except ValueError:
                features.append(match[0])
                i = len(features) - 1
            decat_by_raw_idx[idx] = i
            continue
        features.append(f)
        decat_by_raw_idx[idx] = len(features) - 1

    if replot:
        res = pickle_load(dumpfile)
        gbrt_importances = res['gbrt_importances']
    else:
        # at this point features contains original feature names and raw_features
        # contains categorical dummies, in each round we map
        # feature_importances_, which has the same size as raw_features, to feature
        # importances for original features by adding the importances of each
        # categorical dummy.

        centers = [random_prediction_ctr(data, radius, min_density=roi_density) for _ in range(ncenters)]
        gbrt_importances = np.zeros([ncenters, len(features)])
        for center_idx, center in enumerate(centers):
            sys.stderr.write('%d / %d ' % (center_idx + 1, ncenters))
            X_train, y_train, X_test, y_test = \
                split_with_circle(data, center, roi_density=roi_density, radius=radius)
            X_train = X_train.drop(['Latitude_1', 'Longitude_1'], axis=1)
            X_test = X_test.drop(['Latitude_1', 'Longitude_1'], axis=1)
            assert not X_test.empty

            gbrt = train_gbrt(X_train, y_train, **gbrt_params)
            raw_importances = gbrt.feature_importances_
            for idx, value in enumerate(raw_importances):
                gbrt_importances[center_idx][decat_by_raw_idx[idx]] += value

        if dumpfile:
            res = {'gbrt_importances': gbrt_importances, 'features': features}
            pickle_dump(dumpfile, res, 'feature importances')

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)

    means = gbrt_importances.mean(axis=0)
    sds = np.sqrt(gbrt_importances.var(axis=0))
    sort_order = list(np.argsort(means))

    feature_names = [FEATURE_NAMES[features[i]] for i in sort_order]

    means, sds = [means[i] for i in sort_order], [sds[i] for i in sort_order]
    _yrange = [i-0.4 for i in range(len(features))] # labels in the middle of bars
    ax.barh(_yrange, means, color='k', ecolor='k', alpha=.3, xerr=sds[::-1])
    ax.set_ylim(-1, len(features))
    ax.grid(True)
    ax.set_yticks(range(len(features)))
    ax.set_yticklabels(feature_names, rotation=0, fontsize=10)
    ax.set_title('GBRT feature importances')
    fig.subplots_adjust(left=0.3) # for vertical xtick labels


def plot_space_leakage(data, num_samples, normalize=False, features=None,
                       dumpfile=None, replot=False):
    """ Scatter plots spatial distance vs euclidean distance in feature space
        for specified features. If features is None all features excluding
        latitude/longitude are included. Since the total number of pairs of
        points is typically large pairs are picked by sampling the data set
        randomly.
    """
    raw_features = list(data)
    if replot:
        res = pickle_load(dumpfile)
        distances = res['distances']
    else:
        distance_features = ['Latitude_1', 'Longitude_1']
        if normalize:
            # normalize all features to [0, 1]
            for f in list(data):
                if f in distance_features:
                    continue
                data[f] = (data[f] - data[f].min()) / (data[f].max() - data[f].min())

        if features is None:
            non_features = distance_features + ['GHF']
            features = [x for x in list(data) if x not in non_features]

        distances = []
        sys.stderr.write('Sampling %d pairs of points: \n' % num_samples)
        for i in range(num_samples):
            if (i+1) % 100 == 0:
                sys.stderr.write('%d...\n' % (i+1))
            p1, p2 = np.random.randint(0, len(data), 2)
            p1, p2 = data.iloc[p1], data.iloc[p2]
            feature_d = np.linalg.norm(p1[features] - p2[features])
            spatial_d = np.linalg.norm([p1['Latitude_1'] - p2['Latitude_1'],
                                        p1['Longitude_1'] - p2['Longitude_1']])
            distances.append((spatial_d, feature_d))
        if dumpfile:
            res = {'distances': distances}
            pickle_dump(dumpfile, res, 'space leakage')

    fig = plt.figure(figsize=(8, 10))
    ax = fig.add_subplot(1, 1, 1)
    ax.scatter([x[0] for x in distances], [x[1] for x in distances],
               edgecolor=None, facecolor='k', alpha=.5)
    ax.set_xlabel('Distance in latitude-longitude')
    ax.set_ylabel('Distance in feature space')
    ax.grid(True)
    ax.set_title('Opacity of selected features with respect to spatial coordinates')

    fig.tight_layout()


def plot_partial_dependence(X_train, y_train, include_features=None, n_ways=1):
    """ Plots one-way or two-way partial dependencies (cf. Friedman 2001 or
        ESL). If include_features is given, only those features will be
        considered, otherwise all non-categorical features will be included.
    """
    raw_features = list(X_train)
    features, feature_names = [], []
    for i in range(len(raw_features)):
        if raw_features[i] in FEATURE_NAMES: # everything but categoricals
            # feature_name indexes match those of full training data column no.
            feature_names.append(FEATURE_NAMES[raw_features[i]])
            if include_features is None or raw_features[i] in include_features:
                features.append(i)
        else:
            # will never be used because categoricals are excluded but we
            # should keep track of indices nevertheless
            feature_names.append('Some categorical')
    assert len(feature_names) == len(raw_features)
    sys.stderr.write('Plotting %d-way partial depdnence for %d features\n' %
                     (n_ways, len(features)))

    if n_ways == 1:
        target_features = features # one-way pdp
    elif n_ways == 2:
        target_features = list(combinations(features, 2)) # two-way pdp
    else:
        raise Exception('only one-way and two-way partial dependence plots allowed, %d given' % int(n_ways))

    reg = train_gbrt(X_train, y_train)
    fig, axs = partial_dependence.plot_partial_dependence(
        reg, X_train, target_features, figsize=(22, 12),
        feature_names=feature_names, n_jobs=3, grid_resolution=50
    )
    for ax in axs:
        ax.yaxis.label.set_size(8)
        ax.grid(True)
        for tick in ax.xaxis.get_major_ticks():
            tick.label.set_fontsize(8)
    fig.tight_layout()


def run_reverse_feature_elimination(X_train, y_train, n_features_to_select, step=1):
    """ Performs reverse feature eleminiation on a single training set. No
        plots are produced; only the top n_features_to_select features are
        reported to standard output."""
    sys.stderr.write('Reverse feature elimination down to %d features ...\n' % n_features_to_select)
    gbrt = get_gbrt()
    rfe = RFE(gbrt, n_features_to_select=n_features_to_select, verbose=3, step=step)
    selector = rfe.fit(X_train, y_train)
    support = list(selector.support_)
    features = list(X_train)
    sys.stdout.write('--> Selected %d features:\n   * ' % n_features_to_select)
    sys.stdout.write('\n   * '.join(
        FEATURE_NAMES.get(f, f) for f in compress(features, support)
    ) + '\n')
    sys.stdout.flush()


def exp_error_by_density(data):
    """ Evaluates prediction error (normalized rmse and r2) for GBRT, linear
        regression and constant predictor by using increasingly large sample
        densities in ROIs, constrained to the specified region, with radius
        equal to that of Greenland. Plot is saved to <OUT_DIR>/error_by_density[<region>].png.
    """
    densities = np.append(np.array([1]), np.arange(5, 51, 5))
    radius = GREENLAND_RADIUS
    ncenters = 50
    # region constraints: 'NA-WE', 'NA', 'WE', or None (i.e all)
    region = 'NA-WE'
    dumpfile = 'error_by_density[%s].txt' % region
    plotfile = 'error_by_density[%s].png' % region
    plot_error_by_density(data, densities, radius, ncenters, region=region, dumpfile=dumpfile, replot=False)
    save_cur_fig(plotfile)


def exp_error_by_radius(data):
    """ Evaluates prediction error (normalized rmse and r2) for GBRT, linear
        regression and constant predictor by using increasingly large radii for
        ROIs, constrained to the specified region, with sample density equal to
        that of Greenland. Plot is saved to <OUT_DIR>/error_by_radius[<region>].png.
    """
    radius = GREENLAND_RADIUS
    roi_density = 11.3 # Greenland
    ncenters = 50
    radii = np.arange(500, 4001, 500)
    region = 'NA-WE'
    dumpfile = 'error_by_radius[%s].txt' % region
    plotfile = 'error_by_radius[%s].png' % region

    sys.stderr.write('=> Experiment: Error by Radius (region: %s, no. centers: %d, no. radii: %d)\n' % (region, ncenters, len(radii)))
    plot_error_by_radius(data, roi_density, radii, ncenters, region=region, dumpfile=dumpfile, replot=False)
    save_cur_fig(plotfile)


def exp_sensitivity(data):
    """ Evaluates sensitivity of GBRT and linear regression to perturbations in
        training GHF. Plot is saved to <OUT_DIR>/sensitivity.png.
    """
    radius = GREENLAND_RADIUS
    roi_density = 11.3 # Greenland
    noise_amps = np.arange(0.025, .31, .025)
    ncenters = 50
    dumpfile = 'sensitivity.txt'
    plotfile = 'sensitivity.png'
    plot_sensitivity_analysis(data, roi_density, radius, noise_amps, ncenters, dumpfile=dumpfile, replot=False)
    save_cur_fig(plotfile, title='GBRT prediction sensitivity to noise in training GHF', set_title_for=None)


def exp_generalization(data):
    """ Evaluates the generalization power of GBRT with increasing complexity
        (number of regression tress). This is used to verify that GBRT is
        robust against overfitting and to pick an appropriate number of trees
        for reported results and used in all other experiments (cf. `util.GBRT_params`).
    """
    radius = GREENLAND_RADIUS
    ncenters = 50
    roi_density = 11.3 # Greenland
    ns_estimators = range(50, 750, 100) + range(750, 3001, 750)
    dumpfile = 'generalization.txt'
    plotfile = 'generalization.png'
    plot_generalization_analysis(data, roi_density, radius, ncenters, ns_estimators, dumpfile=dumpfile, replot=False)
    save_cur_fig(plotfile, title='GBRT generalization power', set_title_for=None)


def exp_feature_importance(data):
    """ Plots feature importances for averaged over 50 ROIs with the same
        radius and sample density as the GrIS train/validation split. The max
        depth of trees is increased to 8 to avoid a few influential features
        taking over all trees."""
    radius = GREENLAND_RADIUS
    ncenters = 50
    roi_density = 11.3 # Greenland
    max_depth = 8
    dumpfile = 'feature_importances.txt'
    plot_feature_importance_analysis(data, roi_density, radius, ncenters, dumpfile=dumpfile, max_depth=max_depth, replot=False)
    save_cur_fig('feature-importance.png', title='Relative feature importances in GBRT', set_title_for=None)


def exp_tune_params(data):
    """ Performs a parameter tuning experiment over a range of values of
        interest for each GBRT parameter (cf. `util.tune_params`).
    """
    param_grid = {
        'n_estimators': [200],
        'criterion': ['friedman_mse', 'mse'],
        'learning_rate': [0.01, 0.05, 0.2, 0.5],
        'subsample': [1, .9, .5, .1], # < 1 implies stochastic boosting
        'min_samples_leaf': [1, 3, 10, 20],
        'max_depth': [4, 10, 20],
        'min_impurity_decrease': [1e-07, 1e-3, 1e-1],
        'max_features': [.1, .3, .7]
    }
    tune_params(data, param_grid, cv_fold=10)


def exp_space_leakage(data):
    """ Evaluates whether a set of features (default PROXIMITY_FEATURES) "leak"
        spatial information to GBRT by producing a scatter plot of spatial
        distance vs Euclidean distance in the subspace of specified features
        for a collection of random pairs of data points (default n = 20k).
        Plot is saved to <OUT_DIR>/space-leakage.png.
    """
    dumpfile = 'space_leakage.txt'
    num_samples = 20000
    plot_space_leakage(data, num_samples, features=PROXIMITY_FEATURES, dumpfile=dumpfile, replot=False)
    save_cur_fig('space-leakage.png', title='Spatial information leakage through proximity features', set_title_for=None)


def exp_partial_dependence():
    """ Produces partial dependence plots for GBRT. The one-way PPD is produced
        for all non-categorical features. The two-way PPD is produced for all
        combinations of a fixed set of top 6 features."""
    X_train, y_train, _ = greenland_train_test_sets()
    X_train = X_train.drop(['Latitude_1', 'Longitude_1'], axis=1)

    plot_partial_dependence(X_train, y_train, n_ways=1, include_features=None)
    save_cur_fig('partial-dependence-one-way.png', title='One way partial dependences', set_title_for=None)

    top_features = ['age', 'G_d_2yng_r', 'd_2trench', 'litho_asth', 'ETOPO_1deg', 'moho_GReD']
    plot_partial_dependence(X_train, y_train, n_ways=2, include_features=top_features)
    save_cur_fig('partial-dependence-two-way.png', title='Two way partial dependences', set_title_for=None)


def exp_reverse_feature_elimination():
    """ Performs reverse feature elimination over the greenland training set
        (i.e all global data and GrIS ice core padded points). The top 5
        features and top 10 features are printed to standard output. """
    X_train, y_train, _ = greenland_train_test_sets()
    X_train = X_train.drop(['Latitude_1', 'Longitude_1'], axis=1)
    run_reverse_feature_elimination(X_train, y_train, 5)
    run_reverse_feature_elimination(X_train, y_train, 10)


if __name__ == '__main__':
    data = load_global_data()

    exp_error_by_density(data)
    exp_error_by_radius(data)
    exp_sensitivity(data)
    exp_generalization(data)
    exp_feature_importance(data)
    # The following experiments were not used in the paper:
    #exp_space_leakage(data)
    #exp_tune_params(data)
    #exp_partial_dependence()
    #exp_reverse_feature_elimination()
