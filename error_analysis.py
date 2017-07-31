import sys
from random import randint
from math import sqrt, pi
from ghf_prediction import (
    plt, np, mean_squared_error,
    load_global_gris_data, save_cur_fig, pickle_dump, pickle_load,
    split_with_circle, split_by_distance, tune_params,
    train_gbrt, train_linear, error_summary, random_prediction_ctr,
    CATEGORICAL_FEATURES, GREENLAND_RADIUS
)
from ghf_greenland import greenland_train_test_sets

# for a fixed center, t, and radius, returns r2 and normalized rmse
def compare_models(data, roi_density, radius, center, **gdr_params):
    X_train, y_train, X_test, y_test = \
        split_with_circle(data, center, roi_density=roi_density, radius=radius)
    assert not X_test.empty

    X_train = X_train.drop(['Latitude_1', 'Longitude_1'], axis=1)
    X_test = X_test.drop(['Latitude_1', 'Longitude_1'], axis=1)

    # consider 3 predictors: GBRT, linear regression, and a constant predictor
    gbrt = train_gbrt(X_train, y_train, **gdr_params)
    y_gbrt = gbrt.predict(X_test)

    lin_reg = train_linear(X_train, y_train)
    y_lin = lin_reg.predict(X_test)

    y_const = y_train.mean() + np.zeros(len(y_test))
    # error_summary returns (r2, rmse) pairs
    return {'gbrt': error_summary(y_test, y_gbrt),
            'linear':  error_summary(y_test, y_lin),
            'constant': error_summary(y_test, y_const)}

# ncenters random centers are picked and over all given ROI densities
# cross-validation error (normalized RMSE and r2) are averaged
def plot_error_by_density(data, roi_densities, radius, ncenters, replot=False,
                          dumpfile=None, **gdr_params):
    fig = plt.figure(figsize=(11,5))
    ax_rmse, ax_r2 = fig.add_subplot(1, 2, 1), fig.add_subplot(1, 2, 2)

    if replot:
        results = pickle_load(dumpfile)
    else:
        centers = [
            random_prediction_ctr(data, radius, min_density=max(roi_densities))
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
                comp = compare_models(data, roi_density, radius, center, **gdr_params)
                for k in results['errors'].keys():
                    # k is one of gbrt, linear, or constant
                    results['errors'][k]['r2'][idx_ctr][idx_density] = comp[k][0]
                    results['errors'][k]['rmse'][idx_ctr][idx_density] = comp[k][1]
        if dumpfile:
            pickle_dump(dumpfile, results, comment='GBRT performance results')

    errors = results['errors']
    roi_densities = results['roi_densities']
    ncenters = results['ncenters']

    # Plot GBRT results
    kw = {'alpha': .9, 'lw': 2, 'marker': 'o', 'markersize': 5, 'color': 'k'}
    ax_rmse.plot(roi_densities, errors['gbrt']['rmse'].mean(axis=0), label='GBRT', **kw)
    ax_r2.plot(roi_densities, errors['gbrt']['r2'].mean(axis=0), **kw)

    # Plot GBRT individual results (thin lines)
    #kw = {'alpha': .2, 'lw': .5, 'color': 'k'}
    #for idx_ctr in range(ncenters):
        #ax_rmse.plot(roi_densities, errors['gbrt']['rmse'][idx_ctr], **kw)
        #ax_r2.plot(roi_densities, errors['gbrt']['r2'][idx_ctr], **kw)

    # Plot Linear Regression results
    kw = {'alpha': .7, 'lw': 1, 'ls': '--', 'marker': 'o', 'markersize': 4, 'markeredgecolor': 'b', 'color': 'b'}
    ax_rmse.plot(roi_densities, errors['linear']['rmse'].mean(axis=0), label='linear regression', **kw)
    ax_r2.plot(roi_densities, errors['linear']['r2'].mean(axis=0), **kw)

    # Plot constant predictor results
    kw = {'alpha': .7, 'lw': 1, 'ls': '--', 'marker': 'o', 'markersize': 4, 'color': 'r', 'markeredgecolor': 'r'}
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
    ax_rmse.legend(prop={'size':12}, numpoints=1)
    fig.tight_layout()

# ncenters random centers are picked and over all given radii
# cross-validation error (normalized RMSE and r2) are averaged
def plot_error_by_radius(data, roi_density, radii, ncenters, replot=False,
                         dumpfile=None, **gdr_params):
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
            random_prediction_ctr(data, max(radii), min_density=2*roi_density)
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
                comp = compare_models(data, roi_density, radius, center, **gdr_params)
                for k in results['errors'].keys():
                    # k is one of gbrt, linear, or constant
                    results['errors'][k]['r2'][idx_ctr][idx_radius] = comp[k][0]
                    results['errors'][k]['rmse'][idx_ctr][idx_radius] = comp[k][1]
        if dumpfile:
            pickle_dump(dumpfile, results, comment='GBRT performance results')

    errors = results['errors']
    radii = results['radii']
    ncenters = results['ncenters']

    # Plot GBRT results
    kw = {'alpha': .9, 'lw': 2, 'marker': 'o', 'markersize': 5, 'color': 'k'}
    ax_rmse.plot(radii, errors['gbrt']['rmse'].mean(axis=0), label='GBRT', **kw)
    ax_r2.plot(radii, errors['gbrt']['r2'].mean(axis=0), **kw)

    # Plot GBRT individual results (thin lines)
    #kw = {'alpha': .2, 'lw': .5, 'color': 'k'}
    #for idx_ctr in range(ncenters):
        #ax_rmse.plot(radii, errors['gbrt']['rmse'][idx_ctr], **kw)
        #ax_r2.plot(radii, errors['gbrt']['r2'][idx_ctr], **kw)

    # Plot Linear Regression results
    kw = {'alpha': .7, 'lw': 1, 'ls': '--', 'marker': 'o', 'markersize': 4, 'markeredgecolor': 'b', 'color': 'b'}
    ax_rmse.plot(radii, errors['linear']['rmse'].mean(axis=0), label='linear regression', **kw)
    ax_r2.plot(radii, errors['linear']['r2'].mean(axis=0), **kw)

    # Plot constant predictor results
    kw = {'alpha': .7, 'lw': 1, 'ls': '--', 'marker': 'o', 'markersize': 4, 'color': 'r', 'markeredgecolor': 'r'}
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
    ax_rmse.legend(prop={'size':12}, numpoints=1)
    fig.tight_layout()

# For each given noise amplitude, performs cross-validation on ncenters with
# given radius and test ratio and the average normalized rmse is reported as
# the perturbation in prediction caused by noise.
def plot_sensitivity_analysis(data, roi_density, radius, noise_amps, ncenters,
                              replot=False, dumpfile=None):
    fig = plt.figure(figsize=(10, 5))
    ax_gbrt = fig.add_subplot(1, 2, 1)
    ax_lin = fig.add_subplot(1, 2, 2)

    def _predict(X_train, y_train, X_test, noise_amp):
        # If noise ~ N(0, s^2), then mean(|noise|) = s * sqrt(2/pi),
        # cf. https://en.wikipedia.org/wiki/Half-normal_distribution
        # So to get noise with mean(|noise|) / mean(y) = noise_ampl, we need to
        # have noise ~ N(0, s*^2) with s* = mean(y) * noise_ampl * sqrt(pi/2).
        noise = np.mean(y_train) * noise_amp * sqrt(pi / 2) * np.random.randn(len(y_train))
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

    kw = dict(alpha=.9, lw=2.5, marker='o', color='k', label='global average')
    ax_lin.plot(noise_amps, np.append([0], rmses_lin[1:].mean(axis=0)), **kw)
    ax_gbrt.plot(noise_amps, np.append([0], rmses_gbrt[1:].mean(axis=0)), **kw)

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


# For all given values for n_estimators (number of trees) for GBRT, perform
# cross-validation over ncenters circles with given radius and test ratio. The
# average training and validation error for each number of trees is plotted.
# This is the standard plot to detect overfitting defined as the turning point
# beyond which validation error starts increasing while training error is
# driven down to zero. As expected, GBRT does not overfit (test error
# plateaus).
def plot_generalization_analysis(data, roi_density, radius, ncenters,
                                 ns_estimators, replot=False, dumpfile=None):
    fig, ax = plt.subplots()

    if replot:
        res = pickle_load(dumpfile)
        for v in ['roi_density', 'radius', 'ns_estimators', 'train_rmses', 'test_rmses']:
            exec('%s = res["%s"]' % (v, v))
        assert len(train_rmses) == len(test_rmses), \
               'array length (# of centers) should be the same for training and test'
    else:
        # FIXME min_density
        centers = [random_prediction_ctr(data, radius) for _ in range(ncenters)]

        train_rmses = np.zeros([ncenters, len(ns_estimators)])
        test_rmses = np.zeros([ncenters, len(ns_estimators)])
        for center_idx, center in enumerate(centers):
            sys.stderr.write('%d / %d\n' % (center_idx + 1, ncenters))
            X_train, y_train, X_test, y_test = \
                split_with_circle(data, center, roi_density=roi_density, radius=radius)
            X_train = X_train.drop(['Latitude_1', 'Longitude_1'], axis=1)
            X_test = X_test.drop(['Latitude_1', 'Longitude_1'], axis=1)
            assert not X_test.empty

            for n_idx, n in enumerate(ns_estimators):
                sys.stderr.write('# estimators: %d ' % n)
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

    for center_idx in range(len(train_rmses)):
        ax.plot(ns_estimators, train_rmses[center_idx], 'g', alpha=.2, lw=1)
        ax.plot(ns_estimators, test_rmses[center_idx], 'r', alpha=.2, lw=1)

    ax.plot(ns_estimators, train_rmses.mean(axis=0), 'g', marker='o', alpha=.9, lw=1.5, label='training')
    ax.plot(ns_estimators, test_rmses.mean(axis=0), 'r', marker='o', alpha=.9, lw=1.5, label='validation')
    ax.grid(True)
    ax.set_xlim(ns_estimators[0] - 100, ns_estimators[-1] + 100)
    ax.set_ylim(0, .3)
    ax.set_yticks(np.arange(0, .31, .05))
    ax.set_xlabel('Number of trees', fontsize=16)
    ax.set_ylabel('Normalized RMSE', fontsize=16)
    ax.legend(prop={'size':12.5})
    fig.tight_layout()

# Plot feature importance results for ncenters rounds of cross validation for
# given ROI training density and radius.
def plot_feature_importance_analysis(data, roi_density, radius, ncenters, **gdr_params):
    raw_features = list(data)
    for f in ['Latitude_1', 'Longitude_1', 'GHF']:
        raw_features.pop(raw_features.index(f))

    for f in raw_features + ['GHF']:
        data[f] = (data[f] - data[f].min()) / (data[f].max() - data[f].min())
        #print f, round(data[f].min(), 2), round(data[f].max(), 2), round(data[f].mean(), 2)

    # collapse categorical dummies for feature importances
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

    # at this point features contains original feature names and raw_features
    # contains categorical dummies, in each round we map
    # feature_importances_, which has the same size as raw_features, to feature
    # importances for original features by adding the importances of each
    # categorical dummy.

    centers = [random_prediction_ctr(data, radius, min_density=roi_density) for _ in range(ncenters)]
    fig = plt.figure(figsize=(8, 10))
    ax_gbrt = fig.add_subplot(2, 1, 1)
    ax_lin = fig.add_subplot(2, 1, 2)

    gbrt_importances = np.zeros([ncenters, len(features)])
    lin_coefficients = np.zeros([ncenters, len(features)])
    lin_intercepts = np.zeros(ncenters)
    for center_idx, center in enumerate(centers):
        sys.stderr.write('%d / %d ' % (center_idx + 1, ncenters))
        X_train, y_train, X_test, y_test = \
            split_with_circle(data, center, roi_density=roi_density, radius=radius)
        X_train = X_train.drop(['Latitude_1', 'Longitude_1'], axis=1)
        X_test = X_test.drop(['Latitude_1', 'Longitude_1'], axis=1)
        assert not X_test.empty

        gbrt = train_gbrt(X_train, y_train, **gdr_params)
        raw_importances = gbrt.feature_importances_
        for idx, value in enumerate(raw_importances):
            gbrt_importances[center_idx][decat_by_raw_idx[idx]] += value

        lin_reg = train_linear(X_train, y_train)
        raw_coefficients = lin_reg.coef_
        lin_intercepts[center_idx] = lin_reg.intercept_

        for idx, value in enumerate(raw_coefficients):
            lin_coefficients[center_idx][decat_by_raw_idx[idx]] += value

        ax_gbrt.plot(range(len(features)), gbrt_importances[center_idx], 'k', alpha=.2, lw=1)
        ax_lin.plot(range(len(features)), np.log(np.abs(lin_coefficients[center_idx])), 'b', alpha=.2, lw=1)

    ax_gbrt.plot(range(len(features)), gbrt_importances.mean(axis=0), 'k', alpha=.4, lw=3)
    ax_lin.plot(range(len(features)), np.log(np.abs(lin_coefficients.mean(axis=0))), 'b', alpha=.4, lw=3)
    c = lin_coefficients.mean(axis=0)
    for ax in [ax_lin, ax_gbrt]:
        ax.set_xlim(-1, len(features) + 1)
        ax.grid(True)
        ax.set_xticks(range(len(features)))
    ax_gbrt.set_xticklabels([])
    ax_lin.set_xticklabels(features, rotation=90, fontsize=8)
    ax_gbrt.set_title('GBRT feature importance')
    ax_lin.set_title(r'Normalized linear regression $\log(|w_i|)$')
    #fig.subplots_adjust(bottom=0.2) # for vertical xtick labels

# For each given value of n_estimators, peforms ncenters rounds of cross
# validation with given radius and ROI density. Plots the average bias and
# variance in predictions for _any prediction point_ against the increasing
# number of estimators.
def plot_bias_variance_analysis(data, roi_density, radius, ncenters, ns_estimators):
    fig = plt.figure()
    ax_bias = fig.add_subplot(2, 1, 1)
    ax_var = fig.add_subplot(2, 1, 2)

    results = {}
    for n_idx, n in enumerate(ns_estimators):
        for _ in range(ncenters):
            sys.stdout.write('--------- center %d / %d, %d estimators' % (_+1, ncenters, n))
            center = random_prediction_ctr(data, radius)
            X_train, y_train, X_test, y_test = \
                split_with_circle(data, center, roi_density=roi_density, radius=radius)

            points = [tuple(i) for i in
                      X_test[['Latitude_1', 'Longitude_1']].values]
            X_train = X_train.drop(['Latitude_1', 'Longitude_1'], axis=1)
            X_test = X_test.drop(['Latitude_1', 'Longitude_1'], axis=1)
            assert not X_test.empty

            gbrt = train_gbrt(X_train, y_train, n_estimators=n)
            sys.stdout.flush()
            y_pred = gbrt.predict(X_test)
            for y_idx, y in enumerate(y_pred):
                point = points[y_idx]
                if point not in results:
                    results[point] = {'value': y_test.iloc[y_idx],
                                      'preds':[[] for _ in ns_estimators]}
                results[point]['preds'][n_idx].append(y)

    ns_estimators = np.array(ns_estimators)
    bias = np.zeros([len(results), len(ns_estimators)])
    var = np.zeros([len(results), len(ns_estimators)])
    for idx, point in enumerate(results):
        for n_idx in range(len(ns_estimators)):
            #print point
            #print results[point]
            if results[point]['preds'][n_idx]:
                bias[idx][n_idx] = np.mean(results[point]['preds'][n_idx]) \
                                 - results[point]['value']
                var[idx][n_idx] = np.var(results[point]['preds'][n_idx])
            else:
                bias[idx][n_idx] = np.NaN
                var[idx][n_idx] = np.NaN

        mask = np.isfinite(bias[idx])
        ax_bias.plot(ns_estimators[mask], bias[idx][mask], 'k', alpha=.2, lw=1)
        ax_var.plot(ns_estimators[mask], var[idx][mask], 'k', alpha=.2, lw=1)

    ax_bias.plot(ns_estimators, np.nanmean(bias, axis=0), 'b', alpha=.9, lw=2.5)
    ax_var.plot(ns_estimators, np.nanmean(var, axis=0), 'b', alpha=.9, lw=2.5)

    ax_bias.grid(True)
    ax_var.grid(True)

    ax_bias.set_xlabel('Number of trees')
    ax_var.set_xlabel('Number of trees')

    ax_bias.set_ylabel('Bias')
    ax_var.set_ylabel('Variance')

    ax_bias.set_xlim(ns_estimators[0] - 100, ns_estimators[-1] + 100)
    ax_var.set_xlim(ns_estimators[0] - 100, ns_estimators[-1] + 100)

    ax_var.set_ylim(-.1 * np.nanmax(var), None)


# Plots the average performance of GBRT, as measured by normalized RMSE, over
# ncenters cross validation sets of given radius and ROI density, as a function
# of the number of features used for prediction. The features list is assumed
# to be in decreasing order of importance.
def plot_feature_selection_analysis(data, t, radius, ncenters, features,
                                    **gdr_params):
    data = data.copy()
    non_features = ['Latitude_1', 'Longitude_1', 'GHF']
    noise_cols = [f + '_noise' for f in list(data) if f not in non_features]
    for f in list(noise_cols):
        data[f] = np.random.randn(len(data))

    rmses = np.zeros([ncenters, len(features)])
    junk_rmses = np.zeros([ncenters, len(features)])
    const_rmses = np.zeros([ncenters, len(features)])

    fig, ax = plt.subplots()
    # FIXME min_density
    centers = [random_prediction_ctr(data, radius) for _ in range(ncenters)]
    #centers = [0 for _ in range(ncenters)]
    ns_features = range(1, len(features) + 1)
    for idx_ctr, center in enumerate(centers):
        sys.stderr.write('center: %d / %d\n' % (1 + idx_ctr, ncenters))
        # all three versions use the same split of data; note that X_train and
        # X_test now have both noise columns and ordinary columns
        X_train, y_train, X_test, y_test = \
            split_with_circle(data, center, roi_density=roi_density, radius=radius)
        assert not X_test.empty
        for idx_n, n_features in enumerate(ns_features):
            cols = non_features[:] # copy it; we'll be modifying it
            cols_noise = non_features[:]
            for idx_f, feature in enumerate(features):
                if idx_f == n_features:
                    break
                if feature in CATEGORICAL_FEATURES:
                    for col in list(data):
                        if col[:len(feature)] == feature:
                            if col[:-len('_noise')] == '_noise':
                                cols_noise.append(col)
                            else:
                                cols.append(col)
                else:
                    cols.append(feature)
                    cols_noise.append(feature + '_noise')

            X_train_ = X_train.loc[:, cols]
            X_test_ = X_test.loc[:, cols]
            gbrt = train_gbrt(X_train_.drop(non_features, axis=1),
                                  y_train, **gdr_params)
            y_pred = gbrt.predict(X_test_.drop(non_features, axis=1))
            rmses[idx_ctr][idx_n] = sqrt(mean_squared_error(y_pred, y_test)) / y_test.mean()
            #print 'error', rmses[idx_ctr][idx_n]

            # GBRT with junk feature values (signal-to-noise ratio = 0)
            X_train_ = X_train.loc[:, cols_noise]
            X_test_ = X_test.loc[:, cols_noise]
            gbrt = train_gbrt(X_train_.drop(non_features, axis=1),
                                  y_train, **gdr_params)
            y_pred = gbrt.predict(X_test_.drop(non_features, axis=1))
            junk_rmses[idx_ctr][idx_n] = sqrt(mean_squared_error(y_pred, y_test)) / y_test.mean()
            #print 'on noise', junk_rmses[idx_ctr][idx_n]

            # the simplest (constant) predictor: avg(GHF) over training
            y_pred = y_train.mean() + np.zeros(len(y_test))
            const_rmses[idx_ctr][idx_n] = sqrt(mean_squared_error(y_pred , y_test)) / y_test.mean()
            #print 'const. predictor', const_rmses[idx_ctr][idx_n]

        ax.plot(ns_features, rmses[idx_ctr], 'g', alpha=.2, lw=1)
        #ax.plot(ns_features, junk_rmses[idx_ctr], 'k', alpha=.5, lw=1)

    ax.plot(ns_features, rmses.mean(axis=0), 'g', alpha=.7, lw=3, marker='.', label='GBRT')
    ax.plot(ns_features, junk_rmses.mean(axis=0), 'k', alpha=.7, lw=3, marker='.', label='GBRT trained on noise')
    ax.plot(ns_features, const_rmses.mean(axis=0), 'r', alpha=.9, lw=1 , marker='.', label='Constant predictor')

    ax.set_xlabel('number of included features')
    ax.set_ylabel('normalized RMSE')
    ax.set_ylim(0, .6)
    ax.set_xlim(0, len(features) + 1)
    ax.grid(True)
    ax.legend(fontsize=10)
    ax.set_xticks(ns_features)
    xtick_labels = ['%s - %d' % (feature, idx + 1) for idx, feature in enumerate(features)]
    ax.set_xticklabels(xtick_labels, rotation=90, fontsize=8)

    fig.subplots_adjust(bottom=0.25)

# ================================================= #
# Experiment functions
# --------------------
# each "experiment" is self contained (no parameters; only input is the data
# set), calls a plot_X function and writes a figure to disk.
# ================================================= #
def exp_error_by_density(data):
    densities = np.append(np.array([1]), np.arange(5, 51, 5))
    radius = GREENLAND_RADIUS
    ncenters = 50
    dumpfile = 'error_by_density.txt'
    plot_error_by_density(data, densities, radius, ncenters, dumpfile=dumpfile)#, replot=True)
    save_cur_fig('GB_error_by_density.png')

def exp_error_by_radius(data):
    radius = GREENLAND_RADIUS
    roi_density = 60. / (np.pi * (radius / 1000.) ** 2)
    ncenters = 50
    radii = np.arange(500, 4001, 500)
    dumpfile = 'error_by_radius.txt'
    plot_error_by_radius(data, roi_density, radii, ncenters, dumpfile=dumpfile)#, replot=True)
    save_cur_fig('GB_error_by_radius.png')

def exp_sensitivity(data):
    radius = GREENLAND_RADIUS
    roi_density = 60. / (np.pi * (radius / 1000.) ** 2)
    noise_amps = np.arange(0.025, .31, .025)
    ncenters = 50
    dumpfile = 'sensitivity.txt'
    plot_sensitivity_analysis(data, roi_density, radius, noise_amps, ncenters, dumpfile=dumpfile)#, replot=True)
    save_cur_fig('GB_sensitivity.png')

def exp_generalization(data):
    radius = GREENLAND_RADIUS
    ncenters = 50
    roi_density = 60. / (np.pi * (radius / 1000.) ** 2)
    ns_estimators = range(50, 750, 100) + range(750, 3001, 750)
    dumpfile = 'generalization.txt'
    plot_generalization_analysis(data, roi_density, radius, ncenters, ns_estimators, dumpfile=dumpfile)#, replot=True)
    save_cur_fig('generalization.png')

def exp_bias_variance(data):
    radius = GREENLAND_RADIUS
    ncenters = 200
    roi_density = 11.3 # Greenland
    ns_estimators = range(200, 1100, 200)
    plot_bias_variance_analysis(data, roi_density, radius, ncenters, ns_estimators)
    save_cur_fig('bias-variance.png', title='GBRT bias/variance for different number of trees')

def exp_feature_importance(data):
    radius = GREENLAND_RADIUS
    ncenters = 50
    roi_density = 11.3 # Greenland
    n_estimators = 1000
    plot_feature_importance_analysis(data, roi_density, radius, ncenters, n_estimators=n_estimators)
    save_cur_fig('feature-importance.png')

def exp_feature_selection(data):
    # plot performance by varying number of features
    # features in decreasing order of importance:
    features = [
        'd_2trench',
        'd_2hotspot',
        'd_2volcano',
        'd_2ridge',
        'G_d_2yng_r',
        'age',
        'G_heat_pro',
        'WGM2012_Bo',
        'd2_transfo',
        'ETOPO_1deg',
        'upman_den_',
        'moho_GReD',
        'crusthk_cr',
        'litho_asth',
        'thk_up_cru',
        'G_ther_age', # categorical
        'magnetic_M',
        'G_u_m_vel_', # categorical
        'thk_mid_cr',
        'lthlgy_mod', # categorical
    ]
    ncenters = 100
    roi_density = 11.3 # Greenland

    radius = GREENLAND_RADIUS
    _gdr_params = {
        #'learning_rate': 0.1, # shrinkage
        'n_estimators': 200, # no of weak learners
        'subsample': 0.5, # stochastic GBRT
        'max_depth': 10, # max depth of individual (weak) learners
    }
    plot_feature_selection_analysis(data, roi_density, radius, ncenters, features, **_gdr_params)
    save_cur_fig('feature-selection-t=1-stochastic.png', 'Stochastic GBRT performance, whole circles as test set')

def exp_tune_params():
    param_grid = {
        'n_estimators': [200],
        'criterion': ['friedman_mse', 'mse'],
        'learning_rate': [0.01, 0.05, 0.2, 0.5],
        'subsample': [1, .9, .5, .1], # < 1 implies stochastic boosting
        'min_samples_leaf': [1, 3, 10, 20],
        'max_depth': [4, 10, 20],
        'min_impurity_split': [1e-07, 1e-3, 1e-1],
        'max_features': [.1, .3, .7]
    }
    tune_params(data, param_grid, cv_fold=10)

if __name__ == '__main__':
    data = load_global_gris_data()
    # FIXME artificially forced to 135.0 in source
    data.loc[data.GHF == 135.0, 'GHF'] = 0
    data.loc[data.GHF == 0, 'GHF'] = np.nan
    data.dropna(inplace=True)

    exp_error_by_density(data)
    #exp_error_by_radius(data)
    #exp_sensitivity(data)
    #exp_generalization(data)
    #exp_bias_variance(data)
    #exp_feature_importance(data)
    #exp_feature_selection(data)
    #exp_tune_params()
