import sys
from random import randint
from math import sqrt, pi
from ghf_prediction import (
    plt, np, mean_squared_error,
    load_global_gris_data, save_cur_fig, save_np_object,
    split, split_by_distance, train_regressor, error_summary
)
from ghf_greenland import fill_in_greenland_GHF


def eval_prediction(data, t, radius, center, **gdr_params):
    X_train, y_train, X_test, y_test = \
        split(data, center, test_size=t, max_dist=radius)
    assert not X_test.empty

    reg = train_regressor(X_train.drop(['Latitude_1', 'Longitude_1'], axis=1),
                          y_train, **gdr_params)
    y_pred = reg.predict(X_test.drop(['Latitude_1', 'Longitude_1'], axis=1))
    return error_summary(y_test, y_pred)

def random_prediction_ctr(data, radius, min_points=100):
    cands = data.loc[(data.Latitude_1 > 45) & (data.Longitude_1 > -100) & (data.Longitude_1 < 50)]
    while True:
        center = cands.sample(n=1)
        center = center.Longitude_1, center.Latitude_1
        test, train = split_by_distance(data, center, radius)
        if len(test) >= min_points:
            return round(center[0], 2), round(center[1], 2)

def plot_performance_analysis(data, test_ratios, radii, colors, ncenters):
    centers = [random_prediction_ctr(data, min(radii)) for _ in range(ncenters)]
    fig, ax1 = plt.subplots()
    ax1.set_ylabel('Normalized RMSE (solid lines)')
    ax1.set_xlim(0, 100)
    ax1.set_xlabel('$t$ (percentage of points in circle to predict)')
    ax1.set_title('GBRT performance for different radii')
    ax1.grid(True)

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
                r2, rmse = eval_prediction(data, t, radius, center)
                sys.stderr.write('-> r2 = %.2f, RMSE=%.2f\n' % (r2, rmse))
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
        ax2.plot(test_ratios * 100, r2s.mean(axis=0), label='%d km' % radius, ls='--', **kw)

        save_np_object('error_details.txt', 't, r2, and rmse details', radii_errors[1:,:], delimiter=', ',
                       header='t, r2, rmse', fmt='%10.5f')

    ax1.legend(loc=6, prop={'size':12.5})

def plot_sensitivity_analysis(data, t, radius, noise_amps, ncenters):
    centers = [random_prediction_ctr(data, radius) for _ in range(ncenters)]

    fig, ax = plt.subplots()
    ax.set_xlabel('Relative noise magnitude')
    ax.set_ylabel('RMSE in predicted GHF')
    ax.set_xlim(0, max(noise_amps) * 1.1)
    ax.set_title('GBRT sensitivity to noise in GHF measurements')
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
    ax.set_title('GBRT sensitivity to noise in GHF measurements')
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


def plot_generalization_analysis(data, t, radius, ncenters, ns_estimators):
    centers = [random_prediction_ctr(data, radius) for _ in range(ncenters)]

    fig, ax = plt.subplots()

    train_rmses = np.zeros([ncenters, len(ns_estimators)])
    test_rmses = np.zeros([ncenters, len(ns_estimators)])
    for center_idx, center in enumerate(centers):
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

data = load_global_gris_data()
# FIXME artificially forced to 135.0 in source
data.loc[data.GHF == 135.0, 'GHF'] = 0
data.loc[data.GHF == 0, 'GHF'] = np.nan
data.dropna(inplace=True)

# plot model performance
ts = np.arange(.1, 1, .05)
radii = np.arange(1000, 2501, 500)
colors = 'rgkb'
ncenters = 10
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
#ns_estimators = range(20, 100, 20)
plot_bias_variance_analysis(data, t, radius, ncenters, ns_estimators)
save_cur_fig('bias-variance.png', title='GBRT bias/variance for different number of trees')

# plot model sensitivity for Greenland
data_ = load_global_gris_data()
gris_known, gris_unknown = fill_in_greenland_GHF(data_)
X_train = gris_known.drop(['GHF'], axis=1)
y_train = gris_known.GHF
X_test = gris_unknown.drop(['GHF'], axis=1)
noise_amps = np.arange(0.025, .31, .025)
plot_sensitivity_analysis_greenland(X_train, y_train, X_test, noise_amps)
save_cur_fig('GB_sensitivity_greenland.png', title='GBRT sensitivity for different noise levels for Greenland')
