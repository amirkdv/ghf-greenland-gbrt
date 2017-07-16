import sys
from random import randint
from math import sqrt, pi
from ghf_prediction import (
    plt, np, mean_squared_error,
    load_global_gris_data, save_cur_fig, pickle_dump, pickle_load,
    split_with_circle, split_by_distance, tune_params,
    train_regressor, error_summary, random_prediction_ctr,
    CATEGORICAL_FEATURES, GREENLAND_RADIUS
)
from ghf_greenland import greenland_train_test_sets

# for a fixed center, t, and radius, returns r2 and normalized rmse
def _eval_prediction(data, roi_density, radius, center, **gdr_params):
    X_train, y_train, X_test, y_test = \
        split_with_circle(data, center, roi_density=roi_density, radius=radius)
    assert not X_test.empty

    reg = train_regressor(X_train.drop(['Latitude_1', 'Longitude_1'], axis=1),
                          y_train, **gdr_params)
    y_pred = reg.predict(X_test.drop(['Latitude_1', 'Longitude_1'], axis=1))
    y_avg = y_train.mean() + np.zeros(len(y_test))
    return error_summary(y_test, y_pred), error_summary(y_test, y_avg)

# ncenters random centers are picked and over all given ROI densities
# cross-validation error (normalized RMSE and r2) are averaged
def plot_error_by_density(data, roi_densities, radius, ncenters, load_from=None,
                          dump_to='error_by_radius.txt', **gdr_params):
    fig = plt.figure(figsize=(12,8))
    ax_rmse, ax_r2 = fig.add_subplot(1, 2, 1), fig.add_subplot(1, 2, 2)

    if load_from is not None:
        res = pickle_load(load_from)
        for v in ['roi_densities', 'ncenters', 'rmses', 'r2s', 'rmses_baseline', 'r2s_baseline']:
            exec('%s = res["%s"]' % (v, v))
        assert len(rmses_baseline) == len(r2s_baseline) == len(rmses) == len(r2s), \
               'array length (# of centers) should be the same for baseline and train vs test'
    else:
        required_density = max(roi_densities)
        centers = [random_prediction_ctr(data, radius, min_density=required_density)
                   for _ in range(ncenters)]
        shape = (ncenters, len(roi_densities))
        r2s, rmses = np.zeros(shape), np.zeros(shape)
        r2s_baseline, rmses_baseline = np.zeros(shape), np.zeros(shape)
        for idx_density, roi_density in enumerate(roi_densities):
            for idx_ctr, center in enumerate(centers):
                sys.stderr.write('# density = %.2f ' % roi_density)
                (r2, rmse), (r2_baseline, rmse_baseline) = \
                    _eval_prediction(data, roi_density, radius, center, **gdr_params)
                rmses[idx_ctr][idx_density] = rmse
                r2s[idx_ctr][idx_density] = r2

                rmses_baseline[idx_ctr][idx_density] = rmse_baseline
                r2s_baseline[idx_ctr][idx_density] = r2_baseline
        if dump_to:
            res = {'roi_densities': roi_densities, 'ncenters': ncenters,
                   'rmses': rmses, 'r2s': r2s,
                   'rmses_baseline': rmses_baseline, 'r2s_baseline': r2s_baseline}
            pickle_dump(dump_to, res, comment='GBRT performance results')

    kw = {'alpha': .2, 'lw': 1, 'color': 'k'}
    for idx_ctr in range(ncenters):
        ax_rmse.plot(roi_densities, rmses[idx_ctr], **kw)
        ax_r2.plot(roi_densities, r2s[idx_ctr], **kw)

    kw = {'alpha': .9, 'lw': 2.5, 'marker': 'o', 'markersize': 6, 'color': 'k'}
    ax_rmse.plot(roi_densities, rmses.mean(axis=0), **kw)
    ax_r2.plot(roi_densities, r2s.mean(axis=0), **kw)

    kw = {'alpha': .9, 'lw': 1, 'marker': 'o', 'markersize': 4, 'ls': '--', 'color': 'k'}
    ax_rmse.plot(roi_densities, rmses_baseline.mean(axis=0), label='baseline predictor', **kw)
    ax_r2.plot(roi_densities, r2s_baseline.mean(axis=0), **kw)

    ax_rmse.set_ylabel('Normalized RMSE', fontsize=16)
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
                      fontsize=16)
        ax.grid(True)
    ax_rmse.legend(prop={'size':14}, numpoints=1)
    fig.tight_layout()
    #fig.suptitle('GBRT performance for different densities of training points in ROI',
                 #fontsize=16)

# ncenters random centers are picked and over all given radii
# cross-validation error (normalized RMSE and r2) are averaged
def plot_error_by_radius(data, roi_density, radii, ncenters, load_from=None,
                         dump_to='error_by_radius.txt', **gdr_params):
    fig = plt.figure(figsize=(12,8))
    ax_rmse, ax_r2 = fig.add_subplot(1, 2, 1), fig.add_subplot(1, 2, 2)

    if load_from is not None:
        res = pickle_load(load_from)
        for v in ['radii', 'ncenters', 'rmses', 'r2s', 'rmses_baseline', 'r2s_baseline']:
            exec('%s = res["%s"]' % (v, v))
        assert len(rmses_baseline) == len(r2s_baseline) == len(rmses) == len(r2s), \
               'array length (# of centers) should be the same for baseline and train vs test'
    else:
        centers = [random_prediction_ctr(data, min(radii), min_density=roi_density)
                   for _ in range(ncenters)]
        shape = (ncenters, len(radii))
        r2s, rmses = np.zeros(shape), np.zeros(shape)
        r2s_baseline, rmses_baseline = np.zeros(shape), np.zeros(shape)
        for idx_radius, radius in enumerate(radii):
            for idx_ctr, center in enumerate(centers):
                sys.stderr.write('# radius = %.0f ' % radius)
                (r2, rmse), (r2_baseline, rmse_baseline) = \
                    _eval_prediction(data, roi_density, radius, center, **gdr_params)
                rmses[idx_ctr][idx_radius] = rmse
                r2s[idx_ctr][idx_radius] = r2

                rmses_baseline[idx_ctr][idx_radius] = rmse_baseline
                r2s_baseline[idx_ctr][idx_radius] = r2_baseline
        if dump_to:
            res = {'radii': radii, 'roi_density': roi_density,
                   'ncenters': ncenters, 'rmses': rmses, 'r2s': r2s,
                   'rmses_baseline': rmses_baseline, 'r2s_baseline': r2s_baseline}
            pickle_dump(dump_to, res, comment='GBRT performance results')

    kw = {'alpha': .2, 'lw': 1, 'color': 'k'}
    for idx_ctr in range(ncenters):
        ax_rmse.plot(radii, rmses[idx_ctr], **kw)
        ax_r2.plot(radii, r2s[idx_ctr], **kw)
    kw = {'alpha': .9, 'lw': 2.5, 'marker': 'o', 'color': 'k'}
    ax_rmse.plot(radii, rmses.mean(axis=0), **kw)
    ax_rmse.plot(radii, rmses_baseline.mean(axis=0), ls='--', **kw)
    ax_r2.plot(radii, r2s.mean(axis=0), **kw)
    ax_r2.plot(radii, r2s_baseline.mean(axis=0), ls='--', **kw)

    ax_rmse.set_ylabel('Normalized RMSE')
    ax_r2.set_ylabel('$r^2$')
    ax_r2.set_ylim(-.05, 1)
    ax_r2.set_xlim(min(radii) - 100, max(radii) + 100)
    ax_r2.set_yticks(np.arange(0, 1.01, .1))
    ax_rmse.set_ylim(0, .5)
    ax_rmse.set_yticks(np.arange(0, .51, .05))
    ax_rmse.set_xlim(*ax_r2.get_xlim())
    for ax in [ax_rmse, ax_r2]:
        # FIXME force xlims to be the same
        ax.set_xlabel('radius of ROI (km)')
        ax.grid(True)
    ax_r2.legend(loc=6, prop={'size':12.5})

# For each given noise amplitude, performs cross-validation on ncenters with
# given radius and test ratio and the average normalized rmse is reported as
# the perturbation in prediction caused by noise.
def plot_sensitivity_analysis(data, roi_density, radius, noise_amps, ncenters,
                              load_from=None, dump_to='sensitivity.txt'):
    fig, ax = plt.subplots()
    #fig.suptitle('sensitivity of GBRT predictions to noise in training GHF')
    ax.set_xlabel('Relative magnitude of noise in training GHF', fontsize=14)
    ax.set_ylabel('Normalized RMSE difference in $\widehat{\\mathrm{GHF}}$', fontsize=14)
    ax.set_xlim(0, max(noise_amps) * 1.1)
    ax.set_aspect('equal')
    ax.grid(True)

    def _predict(X_train, y_train, X_test, noise_amp):
        # If noise ~ N(0, s^2), then mean(|noise|) = s * sqrt(2/pi),
        # cf. https://en.wikipedia.org/wiki/Half-normal_distribution
        # So to get noise with mean(|noise|) / mean(y) = noise_ampl, we need to
        # have noise ~ N(0, s*^2) with s* = mean(y) * noise_ampl * sqrt(pi/2).
        noise = np.mean(y_train) * noise_amp * sqrt(pi / 2) * np.random.randn(len(y_train))
        reg = train_regressor(X_train.drop(['Latitude_1', 'Longitude_1'], axis=1),
                              y_train + noise)
        return reg.predict(X_test.drop(['Latitude_1', 'Longitude_1'], axis=1))

    if load_from is not None:
        res = pickle_load(load_from)
        rmses = res['rmses']
        noise_amps = res['noise_amps']
    else:
        centers = [random_prediction_ctr(data, radius, min_density=roi_density)
                   for _ in range(ncenters)]
        y0 = []
        centers = [None] + centers # the first one is for the Greenland case
        rmses = np.zeros((len(centers), len(noise_amps)))
        for idx_ctr, center in enumerate(centers):
            if center is None:
                # Greenland case
                X_train, y_train, X_test = greenland_train_test_sets()
            else:
                X_train, y_train, X_test, _ = \
                    split_with_circle(data, center, roi_density=roi_density, radius=radius)
            sys.stderr.write('(ctr %d) noise_amp = 0.00 ' % (idx_ctr + 1))
            y0 = _predict(X_train, y_train, X_test, 0)
            for idx_noise, noise_amp in enumerate(noise_amps):
                sys.stderr.write('(ctr %d) noise_amp = %.2f ' % (idx_ctr + 1, noise_amp))
                y_pred = _predict(X_train, y_train, X_test, noise_amp)
                rmse = sqrt(mean_squared_error(y0, y_pred)) / np.mean(y0)
                rmses[idx_ctr][idx_noise] = rmse

        res = {'rmses': rmses, 'noise_amps': noise_amps}
        pickle_dump('sensitivity.txt', res, 'sensitivity analysis')

    for idx in range(ncenters+1):
        if idx == 0:
            # Greenland case
            ax.plot(noise_amps, rmses[0], color='b', alpha=.5, lw=2.5, marker='o', markeredgewidth=0.0)
        else:
            ax.plot(noise_amps, rmses[idx], color='k', alpha=.2, lw=1)

    ax.plot(noise_amps, rmses[1:].mean(axis=0), alpha=.9, lw=2.5, marker='o', color='k')
    ax.set_ylim(0, .3)
    fig.tight_layout()


# For all given values for n_estimators (number of trees) for GBRT, perform
# cross-validation over ncenters circles with given radius and test ratio. The
# average training and validation error for each number of trees is plotted.
# This is the standard plot to detect overfitting defined as the turning point
# beyond which validation error starts increasing while training error is
# driven down to zero. As expected, GBRT does not overfit (test error
# plateaus).
def plot_generalization_analysis(data, roi_density, radius, ncenters,
                                 ns_estimators, load_from=None):
    fig, ax = plt.subplots()

    if load_from is not None:
        res = pickle_load(load_from)
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
                reg = train_regressor(X_train, y_train, n_estimators=n)
                _, train_rmse = error_summary(y_train, reg.predict(X_train))
                _, test_rmse  = error_summary(y_test, reg.predict(X_test))
                train_rmses[center_idx][n_idx] = train_rmse
                test_rmses[center_idx][n_idx] = test_rmse

        res = {'roi_density': roi_density,
               'radius': radius,
               'ns_estimators': ns_estimators,
               'train_rmses': train_rmses,
               'test_rmses': test_rmses}
        pickle_dump('generalization.txt', res, comment='generalization errors')

    for center_idx in range(len(train_rmses)):
        ax.plot(ns_estimators, train_rmses[center_idx], 'g', alpha=.2, lw=1)
        ax.plot(ns_estimators, test_rmses[center_idx], 'r', alpha=.2, lw=1)

    ax.plot(ns_estimators, train_rmses.mean(axis=0), 'g', marker='o', alpha=.9, lw=1.5, label='training')
    ax.plot(ns_estimators, test_rmses.mean(axis=0), 'r', marker='o', alpha=.9, lw=1.5, label='validation')
    ax.grid(True)
    ax.set_xlim(ns_estimators[0] - 100, ns_estimators[-1] + 100)
    ax.set_ylim(0, .3)
    ax.set_yticks(np.arange(0, .31, .05))
    ax.set_xlabel('Number of trees')
    ax.set_ylabel('Normalized RMSE')
    ax.legend()

# Plot feature importance results for ncenters rounds of cross validation for
# given test ratio and radius.
def plot_feature_importance_analysis(data, roi_density, radius, ncenters, **gdr_params):
    raw_features = list(data)
    for f in ['Latitude_1', 'Longitude_1', 'GHF']:
        raw_features.pop(raw_features.index(f))

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
    fig, ax = plt.subplots()

    importances = np.zeros([ncenters, len(features)])
    for center_idx, center in enumerate(centers):
        X_train, y_train, X_test, y_test = \
            split_with_circle(data, center, roi_density=roi_density, radius=radius)
        X_train = X_train.drop(['Latitude_1', 'Longitude_1'], axis=1)
        X_test = X_test.drop(['Latitude_1', 'Longitude_1'], axis=1)
        assert not X_test.empty

        reg = train_regressor(X_train, y_train, **gdr_params)
        raw_importances = reg.feature_importances_
        for idx, value in enumerate(raw_importances):
            importances[center_idx][decat_by_raw_idx[idx]] += value

        ax.plot(range(len(features)), importances[center_idx], 'k', alpha=.2, lw=1)

    ax.plot(range(len(features)), importances.mean(axis=0), 'b', alpha=.9, lw=2.5)
    ax.set_xticks(range(len(features)))
    ax.set_xticklabels(features, rotation=90, fontsize=8)
    ax.set_xlim(-1, len(features) + 1)
    ax.grid(True)
    fig.subplots_adjust(bottom=0.2)

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

            reg = train_regressor(X_train, y_train, n_estimators=n)
            sys.stdout.flush()
            y_pred = reg.predict(X_test)
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
            reg = train_regressor(X_train_.drop(non_features, axis=1),
                                  y_train, **gdr_params)
            y_pred = reg.predict(X_test_.drop(non_features, axis=1))
            rmses[idx_ctr][idx_n] = sqrt(mean_squared_error(y_pred, y_test)) / y_test.mean()
            #print 'error', rmses[idx_ctr][idx_n]

            # GBRT with junk feature values (signal-to-noise ratio = 0)
            X_train_ = X_train.loc[:, cols_noise]
            X_test_ = X_test.loc[:, cols_noise]
            reg = train_regressor(X_train_.drop(non_features, axis=1),
                                  y_train, **gdr_params)
            y_pred = reg.predict(X_test_.drop(non_features, axis=1))
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
    plot_error_by_density(data, densities, radius, ncenters)#, load_from='error_by_density.txt')
    save_cur_fig('GB_performance_by_density.png')

def exp_error_by_radius(data):
    radius = GREENLAND_RADIUS
    roi_density = 60. / (np.pi * (radius / 1000.) ** 2)
    ncenters = 50
    radii = np.arange(500, 4001, 500)
    plot_error_by_radius(data, roi_density, radii, ncenters)#, load_from='error_by_radius.txt')
    save_cur_fig('GB_performance_by_radius.png', title='GBRT performance for different radii of ROI')

def exp_sensitivity(data):
    radius = GREENLAND_RADIUS
    roi_density = 60. / (np.pi * (radius / 1000.) ** 2)
    noise_amps = np.arange(0.025, .31, .025)
    ncenters = 50
    plot_sensitivity_analysis(data, roi_density, radius, noise_amps, ncenters)#, load_from='sensitivity.txt')
    save_cur_fig('GB_sensitivity.png')

def exp_generalization(data):
    radius = GREENLAND_RADIUS
    ncenters = 50
    roi_density = 60. / (np.pi * (radius / 1000.) ** 2)
    ns_estimators = range(50, 750, 100) + range(750, 3001, 750)
    plot_generalization_analysis(data, roi_density, radius, ncenters, ns_estimators)#, load_from='generalization.txt')
    save_cur_fig('generalization.png', title='GBRT generalization power for different number of trees')

def exp_bias_variance(data):
    radius = GREENLAND_RADIUS
    ncenters = 200
    roi_density = 11.3 # Greenland
    ns_estimators = range(200, 1100, 200)
    plot_bias_variance_analysis(data, roi_density, radius, ncenters, ns_estimators)
    save_cur_fig('bias-variance.png', title='GBRT bias/variance for different number of trees')

def exp_feature_importance(data):
    radius = GREENLAND_RADIUS
    ncenters = 200
    roi_density = 11.3 # Greenland
    n_estimators = 200
    plot_feature_importance_analysis(data, roi_density, radius, ncenters, n_estimators=n_estimators)
    save_cur_fig('feature-importance.png', title='GBRT feature importances')

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
