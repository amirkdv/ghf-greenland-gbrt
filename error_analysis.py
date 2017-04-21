import sys
from random import randint
from math import sqrt, pi
from ghf_prediction import (
    plt, np, mean_squared_error,
    load_global_gris_data, save_cur_fig, save_np_object,
    split, split_by_distance, train_regressor, error_summary,
    CATEGORICAL_FEATURES,
)
from ghf_greenland import fill_in_greenland_GHF


# Returns a random longitude-latitude pair that serves as the center of
# validation circle.
def random_prediction_ctr(data, radius, min_points=100):
    cands = data.loc[(data.Latitude_1 > 45) & (data.Longitude_1 > -100) & (data.Longitude_1 < 50)]
    while True:
        center = cands.sample(n=1)
        center = center.Longitude_1, center.Latitude_1
        test, train = split_by_distance(data, center, radius)
        if len(test) >= min_points:
            return round(center[0], 2), round(center[1], 2)

# For all combinations of given radii and test_ratios, ncenters random centers
# are picked for cross-validation and their average normalized RMSE and r2 are
# plotted.
def plot_performance_analysis(data, test_ratios, radii, colors, ncenters,
                              plot_r2=True, **gdr_params):
    # for a fixed center, t, and radius, returns r2 and normalized rmse
    def _eval_prediction(data, t, radius, center):
        X_train, y_train, X_test, y_test = \
            split(data, center, test_size=t, max_dist=radius)
        assert not X_test.empty

        reg = train_regressor(X_train.drop(['Latitude_1', 'Longitude_1'], axis=1),
                              y_train, **gdr_params)
        y_pred = reg.predict(X_test.drop(['Latitude_1', 'Longitude_1'], axis=1))
        return error_summary(y_test, y_pred)

    centers = [random_prediction_ctr(data, min(radii)) for _ in range(ncenters)]
    fig, ax1 = plt.subplots()
    ax1.set_ylabel('Normalized RMSE (solid lines)')
    ax1.set_xlim(0, 100)
    # when comparing different setups it's useful to fix the ylim
    #ax1.set_ylim(0, .5)
    ax1.set_xlabel('$t$ (percentage of points in circle to predict)')
    ax1.grid(True)

    if plot_r2:
        ax2 = ax1.twinx()
        ax2.set_ylabel('$r^2$ (dashed lines)')
        ax2.set_ylim(0.3, 1)
        ax2.set_xlim(0, 100)

    assert len(radii) == len(colors)
    radii_errors = np.zeros([1,3])
    for radius, color in zip(radii, colors):
        shape = (ncenters, len(test_ratios))
        r2s, rmses = np.zeros(shape), np.zeros(shape)
        for idx_t, t in enumerate(test_ratios):
            for idx_ctr, center in enumerate(centers):
                sys.stderr.write('** t = %.2f, r = %d, center = %s:\n' % (t, radius, repr(center)))
                r2, rmse = _eval_prediction(data, t, radius, center)
                rmses[idx_ctr][idx_t] = rmse
                r2s[idx_ctr][idx_t] = r2

            lngth = len(test_ratios)
            radius_error = np.hstack([test_ratios.reshape(lngth,1),r2s.mean(axis=0).reshape(lngth,1),rmses.mean(axis=0).reshape(lngth,1)])

        radii_errors = np.vstack([radii_errors,radius_error])

        #for idx in range(ncenters):
            #ax1.plot(test_ratios * 100, r2s[idx], color=color, alpha=.2, lw=1)
            #ax2.plot(test_ratios * 100, rmses[idx], color=color, alpha=.2, lw=1, ls='--')

        kw = {'alpha': .9, 'lw': 2.5, 'marker': 'o', 'color': color}
        ax1.plot(test_ratios * 100, rmses.mean(axis=0), label='%d km' % radius, **kw)
        if plot_r2:
            ax2.plot(test_ratios * 100, r2s.mean(axis=0), label='%d km' % radius, ls='--', **kw)

        save_np_object('error_details.txt', 't, r2, and rmse details', radii_errors[1:,:], delimiter=', ',
                       header='t, r2, rmse', fmt='%10.5f')

    ax1.legend(loc=6, prop={'size':12.5})

# For each given noise amplitude, performs cross-validation on ncenters with
# given radius and test ratio and the average normalized rmse is reported as
# the perturbation in prediction caused by noise.
def plot_sensitivity_analysis(data, t, radius, noise_amps, ncenters):
    centers = [random_prediction_ctr(data, radius) for _ in range(ncenters)]

    fig, ax = plt.subplots()
    ax.set_xlabel('Relative noise magnitude')
    ax.set_ylabel('RMSE in predicted GHF')
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

    y0 = []
    rmses = np.zeros((ncenters, len(noise_amps)))
    for idx_ctr, center in enumerate(centers):
        X_train, y_train, X_test, y_test = \
            split(data, center, test_size=t, max_dist=radius)
        sys.stderr.write('** noise_amp = 0, center = %s:\n' % repr(center))
        y0 = _predict(X_train, y_train, X_test, 0)
        for idx_noise, noise_amp in enumerate(noise_amps):
            sys.stderr.write('** noise_amp = %.2f, center = %s:\n' % \
                (noise_amp, repr(center)))
            y_pred = _predict(X_train, y_train, X_test, noise_amp)
            rmse = sqrt(mean_squared_error(y0, y_pred)) / np.mean(y0)
            sys.stderr.write('-> RMSE=%.2f\n' % rmse)
            rmses[idx_ctr][idx_noise] = rmse

    for idx in range(ncenters):
        ax.plot(noise_amps, rmses[idx], color='k', alpha=.2, lw=1)

    ax.plot(noise_amps, rmses.mean(axis=0), alpha=.9, lw=2.5, marker='o', color='k')


## same as plot_sensitivity_analysis
## applied only for Greenland. ncenter
## is removed and hard-coded as 1
## y_test does not exist
def plot_sensitivity_analysis_greenland(X_train, y_train, X_test, noise_amps):

    fig, ax = plt.subplots()
    ax.set_xlabel('Relative noise magnitude')
    ax.set_ylabel('RMSE in predicted GHF')
    ax.set_xlim(0, max(noise_amps) * 1.1)
    ax.set_aspect('equal')
    ax.grid(True)

    def _predict_greenland(X_train, y_train, X_test, noise_amp):
        # If noise ~ N(0, s^2), then mean(|noise|) = s * sqrt(2/pi),
        # cf. https://en.wikipedia.org/wiki/Half-normal_distribution
        # So to get noise with mean(|noise|) / mean(y) = noise_ampl, we need to
        # have noise ~ N(0, s*^2) with s* = mean(y) * noise_ampl * sqrt(pi/2).
        noise = np.mean(y_train) * noise_amp * sqrt(pi / 2) * np.random.randn(len(y_train))
        reg = train_regressor(X_train.drop(['Latitude_1', 'Longitude_1'], axis=1),
                              y_train + noise)
        return reg.predict(X_test.drop(['Latitude_1', 'Longitude_1'], axis=1))

    rmses = np.zeros((1, len(noise_amps)))
    y0 = _predict_greenland(X_train, y_train, X_test, 0)
    for idx_noise, noise_amp in enumerate(noise_amps):
        sys.stderr.write('** noise_amp = %.2f:' %noise_amp)
        y_pred = _predict_greenland(X_train, y_train, X_test, noise_amp)
        rmse = sqrt(mean_squared_error(y0, y_pred)) / np.mean(y0)
        sys.stderr.write('-> RMSE=%.2f\n' % rmse)
        rmses[0][idx_noise] = rmse

    ax.plot(noise_amps, rmses[0], color='b', alpha=.9, lw=2.5, marker='o')


# For all given values for n_estimators (number of trees) for GBRT, perform
# cross-validation over ncenters circles with given radius and test ratio. The
# average training and validation error for each number of trees is plotted.
# This is the standard plot to detect overfitting defined as the turning point
# beyond which validation error starts increasing while training error is
# driven down to zero. As expected, GBRT does not overfit (test error
# plateaus).
def plot_generalization_analysis(data, t, radius, ncenters, ns_estimators):
    centers = [random_prediction_ctr(data, radius) for _ in range(ncenters)]

    fig, ax = plt.subplots()

    train_rmses = np.zeros([ncenters, len(ns_estimators)])
    test_rmses = np.zeros([ncenters, len(ns_estimators)])
    for center_idx, center in enumerate(centers):
        sys.stderr.write('%d / %d\n' % (center_idx + 1, ncenters))
        X_train, y_train, X_test, y_test = \
            split(data, center, test_size=t, max_dist=radius)
        X_train = X_train.drop(['Latitude_1', 'Longitude_1'], axis=1)
        X_test = X_test.drop(['Latitude_1', 'Longitude_1'], axis=1)
        assert not X_test.empty

        for n_idx, n in enumerate(ns_estimators):
            reg = train_regressor(X_train, y_train, n_estimators=n)
            _, train_rmse = error_summary(y_train, reg.predict(X_train))
            _, test_rmse  = error_summary(y_test, reg.predict(X_test))
            train_rmses[center_idx][n_idx] = train_rmse
            test_rmses[center_idx][n_idx] = test_rmse

        ax.plot(ns_estimators, train_rmses[center_idx], 'g', alpha=.2, lw=1)
        ax.plot(ns_estimators, test_rmses[center_idx], 'r', alpha=.2, lw=1)

    ax.plot(ns_estimators, train_rmses.mean(axis=0), 'g', alpha=.9, lw=2.5)
    ax.plot(ns_estimators, test_rmses.mean(axis=0), 'r', alpha=.9, lw=2.5)
    ax.grid(True)
    ax.set_xlim(ns_estimators[0] - 100, ns_estimators[-1] + 100)
    ax.set_ylim(0, .5)
    ax.set_xlabel('Number of trees')
    ax.set_ylabel('Normalized RMSE')
    ax.legend()

# Plot feature importance results for ncenters rounds of cross validation for
# given test ratio and radius.
def plot_feature_importance_analysis(data, t, radius, ncenters, **gdr_params):
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

    centers = [random_prediction_ctr(data, radius) for _ in range(ncenters)]
    fig, ax = plt.subplots()

    importances = np.zeros([ncenters, len(features)])
    for center_idx, center in enumerate(centers):
        X_train, y_train, X_test, y_test = \
            split(data, center, test_size=t, max_dist=radius)
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
# validation with given radius and test ratio. Plots the average bias and
# variance in predictions for _any prediction point_ against the increasing
# number of estimators.
def plot_bias_variance_analysis(data, t, radius, ncenters, ns_estimators):
    fig = plt.figure()
    ax_bias = fig.add_subplot(2, 1, 1)
    ax_var = fig.add_subplot(2, 1, 2)

    results = {}
    for n_idx, n in enumerate(ns_estimators):
        for _ in range(ncenters):
            sys.stdout.write('--------- center %d / %d, %d estimators' % (_+1, ncenters, n))
            center = random_prediction_ctr(data, radius)
            X_train, y_train, X_test, y_test = \
                split(data, center, test_size=t, max_dist=radius)

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
# ncenters cross validation sets of given radius and ratio, as a function of
# the number of features used for prediction. The features list is assumed to
# be in decreasing order of importance.
def plot_feature_selection_analysis(data, t, radius, ncenters, features, **gdr_params):
    data = data.copy()
    non_features = ['Latitude_1', 'Longitude_1', 'GHF']
    noise_cols = [f + '_noise' for f in list(data) if f not in non_features]
    for f in list(noise_cols):
        data[f] = np.random.randn(len(data))

    rmses = np.zeros([ncenters, len(features)])
    junk_rmses = np.zeros([ncenters, len(features)])
    const_rmses = np.zeros([ncenters, len(features)])

    fig, ax = plt.subplots()
    centers = [random_prediction_ctr(data, radius) for _ in range(ncenters)]
    ns_features = range(1, len(features) + 1)
    for idx_ctr, center in enumerate(centers):
        print 'center: %d / %d' % (1 + idx_ctr, ncenters)
        # all three versions use the same split of data; note that X_train and
        # X_test now have both noise columns and ordinary columns
        X_train, y_train, X_test, y_test = \
            split(data, center, test_size=t, max_dist=radius)
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

            # GBRT with junk feature values (signal-to-noise ratio = 0)
            X_train_ = X_train.loc[:, cols_noise]
            X_test_ = X_test.loc[:, cols_noise]
            reg = train_regressor(X_train_.drop(non_features, axis=1),
                                  y_train, **gdr_params)
            y_pred = reg.predict(X_test_.drop(non_features, axis=1))
            junk_rmses[idx_ctr][idx_n] = sqrt(mean_squared_error(y_pred, y_test)) / y_test.mean()

            # the simplest (constant) predictor: avg(GHF) over training
            y_pred = y_train.mean() + np.zeros(len(y_test))
            const_rmses[idx_ctr][idx_n] = sqrt(mean_squared_error(y_pred , y_test)) / y_test.mean()

        ax.plot(ns_features, rmses[idx_ctr], 'g', alpha=.4, lw=1)
        ax.plot(ns_features, junk_rmses[idx_ctr], 'k', alpha=.4, lw=1)

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

data = load_global_gris_data()
# FIXME artificially forced to 135.0 in source
data.loc[data.GHF == 135.0, 'GHF'] = 0
data.loc[data.GHF == 0, 'GHF'] = np.nan
data.dropna(inplace=True)

# plot model performance
ts = np.arange(.1, 1, .05)
radii = np.arange(1000, 2501, 500)
colors = 'rgkb'
ncenters = 50
plot_performance_analysis(data, ts, radii, colors, ncenters)
save_cur_fig('GB_performance.png', title='GBRT performance for different radii')

# plot model sensitivity excluding Greenland
noise_amps = np.arange(0.025, .31, .025)
radius = 1500
ncenters = 10
t = .9
plot_sensitivity_analysis(data, t, radius, noise_amps, ncenters)
save_cur_fig('GB_sensitivity.png', title='GBRT sensitivity for different noise levels')

# plot generalization analysis
radius = 1700
ncenters = 10
t = .9
ns_estimators = range(200, 3500, 500)
plot_generalization_analysis(data, t, radius, ncenters, ns_estimators)
save_cur_fig('generalization.png', title='GBRT generalization power for different number of trees')

# plot bias/variance analysis
radius = 1700
ncenters = 200
t = .9
ns_estimators = range(200, 1100, 200)
plot_bias_variance_analysis(data, t, radius, ncenters, ns_estimators)
save_cur_fig('bias-variance.png', title='GBRT bias/variance for different number of trees')

# plot feature importance analysis
radius = 1700
ncenters = 200
t = .9
n_estimators = 200
plot_feature_importance_analysis(data, t, radius, ncenters, n_estimators=n_estimators)
save_cur_fig('feature-importance.png', title='GBRT feature importances')

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
ncenters = 50
t = .9
radius = 1700
plot_feature_selection_analysis(data, t, radius, ncenters, features)
save_cur_fig('feature-selection.png', 'GBRT performance for different number of features')

# plot model sensitivity for Greenland
data_ = load_global_gris_data()
gris_known, gris_unknown = fill_in_greenland_GHF(data_)
X_train = gris_known.drop(['GHF'], axis=1)
y_train = gris_known.GHF
X_test = gris_unknown.drop(['GHF'], axis=1)
noise_amps = np.arange(0.025, .31, .025)
plot_sensitivity_analysis_greenland(X_train, y_train, X_test, noise_amps)
save_cur_fig('GB_sensitivity_greenland.png', title='GBRT sensitivity for different noise levels for Greenland')
