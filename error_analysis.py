import sys
from random import randint
from ghf_prediction import (
    plt, np,
    load_global_gris_data, save_cur_fig, save_np_object,
    split, train_regressor, error_summary
)

def eval_prediction_multiple(data, tasks):
    return {task: eval_prediction(data, *task) for task in tasks}

def eval_prediction(data, t, r, center):
    X_train, y_train, X_test, y_test = \
        split(data, center, test_size=t, max_dist=r)
    assert not X_test.empty

    reg = train_regressor(X_train.drop(['Latitude_1', 'Longitude_1'], axis=1),
                          y_train)
    y_pred = reg.predict(X_test.drop(['Latitude_1', 'Longitude_1'], axis=1))
    return error_summary(y_test, y_pred)

def random_prediction_ctr():
    return randint(-100, 50), randint(45, 90)

def pick_centers(ncenters, t, r, min_points=100):
    centers = []
    repeat = ncenters
    sys.stderr.write('Picking %d centers with at least %d ' % (ncenters, min_points) +
                     'data points for t = %.2f, r = %d\n' % (t, r))
    while repeat:
        center = random_prediction_ctr()
        _, _, X_test, _ = split(data, center, t, r)
        if len(X_test) > min_points:
            centers.append(center)
            repeat -= 1
        else:
            sys.stderr.write('-> skip %s with %d points\n' % \
                             (repr(center), len(X_test)))

    return centers

data = load_global_gris_data()
# FIXME artificially forced to 135.0 in source
data.loc[data.GHF == 135.0, 'GHF'] = 0
data.loc[data.GHF == 0, 'GHF'] = np.nan
data.dropna(inplace=True)

ts = np.arange(.1, 1, .05)
rs = np.arange(1200, 2701, 500)
cs = 'rgkb'
ncenters = 50

centers = pick_centers(ncenters, max(ts), min(rs))

fig, ax1 = plt.subplots()
ax1.set_xlabel('$t$ (percentage of points in circle to predict)')
ax1.set_ylabel('$r^2$ (solid lines)')
ax1.set_title('GBRT performance for different radii')
ax1.set_xlim(0, 100)
ax1.set_ylim(0.3, 1)
ax1.grid(True)

ax2 = ax1.twinx()
ax2.set_ylabel('RMSE (dashed lines)')
ax2.set_xlim(0, 100)

assert len(rs) == len(cs)
for r, color in zip(rs, cs):
    shape = (ncenters, len(ts))
    r2s, rmses = np.zeros(shape), np.zeros(shape)
    for idx_t, t in enumerate(ts):
        for idx_ctr, center in enumerate(centers):
            sys.stderr.write('** t = %.2f, r = %d, center = %s:\n' % (t, r, repr(center)))
            r2, rmse = eval_prediction(data, t, r, center)
            sys.stderr.write('-> r2 = %.2f, RMSE=%.2f\n' % (r2, rmse))
            rmses[idx_ctr][idx_t] = rmse
            r2s[idx_ctr][idx_t] = r2

    #for idx in range(ncenters):
        #ax1.plot(ts * 100, r2s[idx], color=color, alpha=.2, lw=1)
        #ax2.plot(ts * 100, rmses[idx], color=color, alpha=.2, lw=1, ls='--')

    ax1.plot(ts * 100, r2s.mean(axis=0), color=color, alpha=.9, lw=2.5, label='%d km' % r, marker='o')
    ax2.plot(ts * 100, rmses.mean(axis=0), color=color, alpha=.9, lw=2.5, ls='--', label='%d km' % r, marker='o')

ax1.legend(loc=6, prop={'size':12.5})
fig.savefig('GB_performance.png',dpi=400)
