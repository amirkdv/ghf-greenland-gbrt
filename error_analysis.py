import sys
from random import randint
from ghf_prediction import (
    plt, np,
    load_global_gris_data, save_cur_fig, save_np_object,
    split, split_by_distance, train_regressor, error_summary
)

def eval_prediction_multiple(data, tasks):
    return {task: eval_prediction(data, *task) for task in tasks}

def eval_prediction(data, t, radius, center):
    X_train, y_train, X_test, y_test = \
        split(data, center, test_size=t, max_dist=radius)
    assert not X_test.empty

    reg = train_regressor(X_train.drop(['Latitude_1', 'Longitude_1'], axis=1),
                          y_train)
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
    ax1.set_xlabel('$t$ (percentage of points in circle to predict)')
    ax1.set_ylabel('$r^2$ (solid lines)')
    ax1.set_title('GBRT performance for different radii')
    ax1.set_xlim(0, 100)
    ax1.set_ylim(0.3, 1)
    ax1.grid(True)

    ax2 = ax1.twinx()
    ax2.set_ylabel('Normalized RMSE (dashed lines)')
    ax2.set_xlim(0, 100)

    assert len(radii) == len(colors)
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

        #for idx in range(ncenters):
            #ax1.plot(test_ratios * 100, r2s[idx], color=color, alpha=.2, lw=1)
            #ax2.plot(test_ratios * 100, rmses[idx], color=color, alpha=.2, lw=1, ls='--')

        kw = {'alpha': .9, 'lw': 2.5, 'marker': 'o', 'color': color}
        ax1.plot(test_ratios * 100, r2s.mean(axis=0), label='%d km' % radius, **kw)
        ax2.plot(test_ratios * 100, rmses.mean(axis=0), label='%d km' % radius, ls='--', **kw)

    ax1.legend(loc=6, prop={'size':12.5})

data = load_global_gris_data()
# FIXME artificially forced to 135.0 in source
data.loc[data.GHF == 135.0, 'GHF'] = 0
data.loc[data.GHF == 0, 'GHF'] = np.nan
data.dropna(inplace=True)

ts = np.arange(.1, 1, .05)
radii = np.arange(1200, 2701, 500)
colors = 'rgkb'
ncenters = 10
plot_performance_analysis(data, ts, radii, colors, ncenters)
save_cur_fig('GB_performance.png', title='GBRT performance for different radii')
