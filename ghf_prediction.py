import os
import sys
import scipy
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
from time import time
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import GradientBoostingRegressor
from math import radians, sin, cos, asin, sqrt, floor

pd.set_option('display.max_columns', 80)
plt.ticklabel_format(useOffset=False)
plt.rc('font', family='TeX Gyre Schola')

MAX_GHF  = 150   # max limit of ghf considered
GREENLAND_RADIUS = 1300

OUT_DIR = os.getenv('OUT_DIR', 'plots/')
if not os.path.isabs(OUT_DIR):
    OUT_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)), OUT_DIR)

GLOBAL_CSV = '1deg_all_resampled_w_missing_from_goutorbe.csv'
GRIS_CSV = '1deg_greenland_GHF_added2.csv' # FIXME remove old csv file from repo?
IGNORED_COLS = [
    'OBJECTID_1', 'continent', 'lthlgy_all', 'num_in_cel', 'num_in_con',
     'WGM2012_Ai', 'depthmoho', 'moho_Pasya', 'lithk_cona',
    #'G_heat_pro', 'd_2hotspots', 'd_2volcano', 'crusthk_cr', 'litho_asth',
    #'d_2trench', 'G_d_2yng_r', 'd_2ridge', 'd2_transfo'
    ]
CATEGORICAL_FEATURES = ['G_u_m_vel_', 'lthlgy_mod', 'G_ther_age']
GDR_PARAMS = {
    'loss': 'ls',
    'learning_rate': 0.05,
    'n_estimators': 1000,
    'subsample': 1., # less than 1 values would lead to stochastic GBRT
    'criterion': 'friedman_mse',
    'min_samples_split': 2,
    'min_samples_leaf': 9,
    'min_weight_fraction_leaf': 0.0,
    'max_depth': 4,
    'min_impurity_split': 1e-07,
    'init': None,
    'random_state': 0,
    'max_features': 0.3,
    'alpha': 0.9,
    'verbose': 0,
    #'verbose': 10,
    'max_leaf_nodes': None,
    'warm_start': False,
    'presort': 'auto',
}

# Reads csv source and applies general filters.
def read_csv(path):
    data = pd.read_csv(path, index_col=0, na_values=-999999)

    data[data['lthlgy_mod'] == 0] = np.nan
    data[data['lthlgy_mod'] == -9999] = np.nan
    data.dropna(inplace=True)

    # drop rows with out of range GHF
    data = data.drop(IGNORED_COLS, axis=1)

    return data

# loads GLOBAL_CSV and GRIS_CSV, performs GRIS specific filters, and returns a
# single DataFrame.
def load_global_gris_data(plot_projections_to=None):
    data_global = read_csv(GLOBAL_CSV)
    data_gris = read_csv(GRIS_CSV)
    data_gris = process_greenland_data(data_gris)

    data = pd.concat([data_global, data_gris])

    if plot_projections_to:
        fig = plt.figure(figsize=(16, 20))
        center = (28.67, 45.5)
        data_roi, data_nonroi = split_by_distance(data, center, GREENLAND_RADIUS)
        for idx, f in enumerate(list(data)):
            if f == 'GHF':
                continue
            ax = fig.add_subplot(6, 4, idx + 1)
            ax.scatter(data_nonroi[f], data_nonroi['GHF'], c='g', lw=0, s=2, alpha=.4, label='outside ROI')
            ax.scatter(data_roi[f], data_roi['GHF'], c='r', lw=0, s=2, alpha=.7, label='inside ROI')
            ax.set_title(f)
            ax.tick_params(labelsize=6)
            ax.grid(True)
            ax.legend(fontsize=12)

        save_cur_fig(plot_projections_to)

    data = pd.get_dummies(data, columns=CATEGORICAL_FEATURES)

    return data

# GRIS specific filters
def process_greenland_data(data):
    # mapping from old to new values of lthlgy_mod
    # Legend: volcanic=1, metamorphic=2, sedimentary=3
    lthlgy_mod_rewrites = {
        1: 2, 2: 3, 3: 3, 4: 3, 5: 1, 6: 2, 7: 1, 8: 3, 9: 2, 10: 2
    }
    data['lthlgy_mod'] = data.apply(
        lambda row: lthlgy_mod_rewrites[row['lthlgy_mod']],
        axis=1
    )
    return data

# Returns a random longitude-latitude pair that serves as the center of
# validation circle.
def random_prediction_ctr(data, radius, min_density=0):
    cands = data.loc[(data.Latitude_1 > 45) & (data.Longitude_1 > -100) & (data.Longitude_1 < 50)]
    while True:
        center = cands.sample(n=1)
        center = center.Longitude_1, center.Latitude_1
        test, train = split_by_distance(data, center, radius)
        area = np.pi * (radius / 1000.) ** 2
        # FIXME this can loop infinitely if a large enough min_density is given
        if len(test) / area >= min_density:
            return round(center[0], 2), round(center[1], 2)

# returns a pair of DataFrames: one containing rows in data that are closer
# than radius to center, and those that are not.
def split_by_distance(data, center, radius):
    # store distances in a temporary column '_distance'
    data['_distance'] = haversine_distance(data, center)
    within = data[data._distance < radius].drop('_distance', axis=1)
    beyond = data[data._distance > radius].drop('_distance', axis=1)
    data.drop('_distance', axis=1, inplace=True)

    return within, beyond

# returns a column containing distances of each row of data from center
def haversine_distance(data, center):
    def _haversine(row):
        lon1, lat1 = center
        lon2, lat2 = row['Longitude_1'], row['Latitude_1']
        lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
        dlon = lon2 - lon1
        dlat = lat2 - lat1
        a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
        c = 2 * asin(sqrt(a))
        km = 6367 * c
        return km

    return data.apply(_haversine, axis=1)

# density in sample per 1e6 km^2, radius in km
def roi_density_to_test_size(density, radius, num_samples):
    area = np.pi * (radius / 1000.) ** 2
    test_size = 1 - (area * density) / num_samples
    max_density = num_samples / area
    assert max_density >= density, \
        'demanded density (%.2f) larger than max density in ROI (%.2f)' % (density, max_density)
    return test_size

# splits rows in data into a training and test set according to the following
# rule: consider a circle C with given center and radius. The training set is
# those rows outside C and a randomly chosen subset of those rows within C
# (proportion of points in C kept for test is given by test_size; float between
# 0 and 1).
def split_with_circle(data, center, roi_density=None, radius=3500):
    data_test, data_train = split_by_distance(data, center, radius)
    test_size = roi_density_to_test_size(roi_density, radius, len(data_test))
    assert test_size > 0
    if test_size < 1:
        additional_train, reduced_data_test = train_test_split(
            data_test, random_state=0, test_size=test_size
        )
        #print 'ROI area:', round(area, 2), \
              #'ROI test size:', round(len(reduced_data_test) * 1. / len(data_test), 2), 'demanded test size:', round(test_size, 2), \
              #'density:', round(len(additional_train) / area, 2), 'demanded density:', round(roi_density, 2)
        data_train = pd.concat([data_train, additional_train])
        data_test = reduced_data_test

    X_train, y_train = data_train.drop('GHF', axis=1), data_train['GHF']
    X_test,  y_test  = data_test.drop('GHF', axis=1),  data_test['GHF']

    return X_train, y_train, X_test, y_test

def tune_params(data, param_grid, cv_fold=10):
    def _score(reg, X_test, y_test):
        y_pred = reg.predict(X_test)
        return sqrt(mean_squared_error(y_test, y_pred)) / np.mean(y_test)

    gbm = GradientBoostingRegressor(**GDR_PARAMS)
    search = GridSearchCV(gbm, param_grid, scoring=_score, cv=cv_fold, n_jobs=1, verbose=10)
    search.fit(data.drop(['Latitude_1', 'Longitude_1', 'GHF'], axis=1), data['GHF'])
    print search.best_params_

# plots a series of GHF values at given latitude and longitude positions
def plot_GHF_on_map(m, lons, lats, values,
                    parallel_step=20., meridian_step=60.,
                    clim=(20., 150.), clim_step=10,
                    colorbar_args={}, scatter_args={}):
    m.drawparallels(
        np.arange(-80., 81., parallel_step), labels=[1, 0, 0, 0], fontsize=10
    )
    m.drawmeridians(
        np.arange(-180., 181., meridian_step), labels=[0, 0, 0, 1], fontsize=10
    )
    m.drawmapboundary(fill_color='white')
    m.drawcoastlines(linewidth=0.5)

    x, y = m(lons, lats)

    cs = m.scatter(x, y, c=values, **scatter_args)

    cbar = m.colorbar(cs, **colorbar_args)
    cbar.set_label('mW m$^{-2}$')
    labels = range(int(clim[0]), int(clim[1]) + 1, clim_step)
    cbar.set_ticks(labels)
    cbar.set_ticklabels(labels)
    plt.clim(*clim)

# plots a series of GHF values at given latitude and longitude positions in
# ascii format
def plot_GHF_on_map_pcolormesh(m, lons, lats, values,
                    parallel_step=20., meridian_step=60.,
                    clim=(20., 150.), clim_step=10,
                    colorbar_args={}, pcolor_args={}):
    m.drawparallels(
        np.arange(-80., 81., parallel_step), labels=[1, 0, 0, 0], fontsize=10
    )
    m.drawmeridians(
        np.arange(-180., 181., meridian_step), labels=[0, 0, 0, 1], fontsize=10
    )
    m.drawmapboundary(fill_color='white')
    m.drawcoastlines(linewidth=0.5)

    lons_min, lons_max = np.min(lons), np.max(lons)
    lats_min, lats_max = np.min(lats), np.max(lats)

    lons_array = np.linspace(lons_min,lons_max,lons_max-lons_min+1)
    lats_array = np.linspace(lats_min,lats_max,lats_max-lats_min+1)

    lon, lat = np.meshgrid(lons_array, lats_array)

    x, y = m(lon, lat)

    ascii = np.zeros([len(lats_array),len(lons_array)])
    ascii = np.where(ascii==0,np.nan,0)

    for item in np.vstack([lons,lats,values]).T:
        j = np.floor(lons_array).tolist().index(floor(item[0]))
        i = np.floor(lats_array).tolist().index(floor(item[1]))
        ascii[i][j] = item[2]

    ascii = np.ma.masked_where(np.isnan(ascii),ascii)

    cs = m.pcolormesh(x,y,ascii,**pcolor_args)

    cbar = m.colorbar(cs, **colorbar_args)
    cbar.set_label('mW m$^{-2}$')
    labels = range(int(clim[0]), int(clim[1]) + 1, clim_step)
    cbar.set_ticks(labels)
    cbar.set_ticklabels(labels)
    plt.clim(*clim)

# plots a series of GHF values at given latitude and longitude positions in
# ascii format interpolated using basemap transform_scalar functions
def plot_GHF_on_map_pcolormesh_interp(m, lons, lats, values,
                    parallel_step=20., meridian_step=60.,
                    clim=(20., 150.), clim_step=10,
                    colorbar_args={}, pcolor_args={}):
    m.drawparallels(
        np.arange(-80., 81., parallel_step), labels=[1, 0, 0, 0], fontsize=10
    )
    m.drawmeridians(
        np.arange(-180., 181., meridian_step), labels=[0, 0, 0, 1], fontsize=10
    )
    m.drawmapboundary(fill_color='white')
    m.drawcoastlines(linewidth=0.5)

    lons_min, lons_max = np.min(lons), np.max(lons)
    lats_min, lats_max = np.min(lats), np.max(lats)

    lons_array = np.linspace(lons_min,lons_max,lons_max-lons_min+1)
    lats_array = np.linspace(lats_min,lats_max,lats_max-lats_min+1)

    lon, lat = np.meshgrid(lons_array, lats_array)

    x, y = m(lon, lat)

    ascii = np.zeros([len(lats_array),len(lons_array)])
    ascii = np.where(ascii==0,np.nan,0)

    for item in np.vstack([lons,lats,values]).T:
        j = np.floor(lons_array).tolist().index(floor(item[0]))
        i = np.floor(lats_array).tolist().index(floor(item[1]))
        ascii[i][j] = item[2]

    ascii_interp, x, y = m.transform_scalar(
        ascii, lons_array, lats_array,
        5*len(lons_array), 5*len(lats_array), returnxy=True
    )

    ascii_interp = np.ma.masked_where(np.isnan(ascii_interp),ascii_interp)

    cs = m.pcolormesh(x,y,ascii_interp,**pcolor_args)

    cbar = m.colorbar(cs, **colorbar_args)
    cbar.set_label('mW m$^{-2}$')
    labels = range(int(clim[0]), int(clim[1]) + 1, clim_step)
    cbar.set_ticks(labels)
    cbar.set_ticklabels(labels)
    plt.clim(*clim)

# saves current matplotlib plot to given filename in OUT_DIR
def save_cur_fig(filename, title=None):
    if title:
        ax = plt.gcf().add_subplot(111)
        ax.set_title(title)
#        plt.gcf().suptitle(title)
    plt.savefig(os.path.join(OUT_DIR, filename), dpi=400)
    sys.stderr.write('Saved %s to %s.\n' % (repr(title), filename))
    plt.clf()

def pickle_dump(path, obj, comment=None):
    with open(os.path.join(OUT_DIR, path), 'w') as f:
        pickle.dump(obj, f)
    sys.stderr.write('dumped %s to %s.\n' % (comment if comment else 'object', path))

def pickle_load(path):
    with open(os.path.join(OUT_DIR, path), 'rb') as f:
        return pickle.load(f)


def train_linear(X_train, y_train):
    reg = LinearRegression()
    reg.fit(X_train, y_train)
    return reg

# Trains and returns a GradientBoostingRegressor over the given training
# feature and value vectors. Feature importance values are stored in
# OUTDIR/logfile
def train_gbrt(X_train, y_train, logfile=None, **gdr_params):
    sys.stderr.write('-> Training ...')
    start = time()
    # allow keyword arguments to override default GDR parameters
    _gdr_params = GDR_PARAMS.copy()
    _gdr_params.update(gdr_params)
    reg = GradientBoostingRegressor(**_gdr_params)
    reg.fit(X_train, y_train)
    sys.stderr.write(' (%.2f secs)\n' % (time() - start))

    importance = reg.feature_importances_
    hdrs = list(X_train.columns.values)
    logs = np.asarray(sorted(zip(hdrs, importance), key=lambda x: x[1]))
#    if logfile:
#        save_np_object(logfile, 'feature importances', logs, fmt="%s")

    return reg

# Returns r^2 of y, y^ linear regression and RMSE of y, y^ normalized to the
# average of y; the latter is a unitless and *scale-invariant* measure of
# performance.
# - cf. http://stats.stackexchange.com/a/190948
#   RMSE normalized to mean of y is scale-invariant.
# - cf. http://www.stat.columbia.edu/~gelman/research/published/standardizing7.pdf
#   r^2 is not scale-invariant.
def error_summary(y_test, y_pred):
    _, _, r_value, _, _= scipy.stats.linregress(y_test, y_pred)
    rmse = sqrt(mean_squared_error(y_test, y_pred)) / np.mean(y_test)
    return r_value ** 2, rmse

# plots the linear regression of two GHF value series (known test values and
# predicted values) and saves the plot to OUT_DIR/filename.
def plot_test_pred_linregress(y_test, y_pred, label=None, color='blue'):
    # first x=y line, then dumb predictor (average), then the
    # correlation between y_test and y_pred
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ghf_range = np.linspace(0,MAX_GHF,50)
    ax.plot(ghf_range, ghf_range, 'k', alpha=0.5, lw=2, label='ideal predictor')

    data = load_global_gris_data()
    data.loc[data.GHF == 135.0, 'GHF'] = 0
    data.loc[data.GHF == 0, 'GHF'] = np.nan
    data.dropna(inplace=True)

    ax.plot(ghf_range, np.ones([50,1]) * data.GHF.mean(),'k--', lw=1, alpha=0.7,
            label='constant predictor')
    r2, rmse = error_summary(y_test, y_pred)

    scatter_kw = {'color': color, 'edgecolor': 'white', 's': 30, 'alpha': .9}
    ax.scatter(y_test, y_pred, label=label, **scatter_kw)
    ax.grid(linestyle='dotted')
    ax.set_aspect('equal')
    ax.set_xlabel('$GHF$ (mW m$^{-2}$)')
    ax.set_ylabel('$\widehat{GHF}$ (mW m$^{-2}$)')
    ax.set_xlim([0, MAX_GHF])
    ax.set_ylim([0, MAX_GHF])

    ax.set_title('$r^2=%.2f, RMSE=%.2f$' % (r2, rmse))
    ax.legend(loc=2)

# plots the histogram of given GHF values
def plot_GHF_histogram(values):
    hist, _, _ = plt.hist(values, np.linspace(0, MAX_GHF, 61), normed=True,
                          lw=1, alpha=.7, color='k', edgecolor='k')
    plt.xlabel('GHF (mW m$^{-2}$)')
    plt.ylabel('Normalized Frequency')
    plt.grid(linestyle='dotted')
    plt.xlim([0, MAX_GHF])
    plt.ylim([0, max(hist) * 1.1])
