#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""This module provides common utilities used in main scripts. Its main
responsibilities are loading and filtering data sets and handling plots and
dump files.

# Data Files

Two data sets must be provided: a global data set (cf `GLOBAL_CSV` below)
containing all features of interest and GHF measurements, and a data set
corresponding to a region of interest with scarce data (Greenland, cf.
`GREENLAND_CSV`) containing only feature measurements and no GHF values.

## Accessing data

Global and GrIS data should be loaded using `load_global_data` and
`load_gris_data`. GrIS ice core data is automatically loaded into global
variable `GREENLAND`.

## Producing data

Methods whose names start with `plot_` are plotting methods that do _not_ use
the object oriented matplotlib API. After calling any such method the resulting
figure can be saved via `save_cur_fig`.

For writing and reading data dumps `pickle_load, pickle_load` are provided.

## Data paths

All paths below can be either relative (taken with respect to repo root) or
absolute and can be overriden by an environment variable with the same
name.

- GLOBAL_CSV: global data set csv, default: `global.csv`.
- GRIS_CSV: Greenland data set csv, default: `gris_features.csv`.
- GRIS_ICE_CORE_CSV: Greenland ice core data set cv, default: `gris_ice_cores.csv`.
- OUT_DIR: directory in which plots and dump files are saved, default: `plots/`.

# Features

Each feature has a human readable name and a column name used in csv data
files (i.e GLOBAL_CSV and GRIS_CSV).

- FEATURE_NAMES: a dict mapping column names to human readable names.
- CATEGORICAL_FEATURES: a dict column names that correspond to categorical
                        features to their list of allowed values.
- PROXIMITY_FEATURES: columns that correspond to proximity features

# Constants

The following numbers are used as constants in this module:

- MAX_GHF: maximum limit of GHF for plotting purposes only (150)
- GREENLAND_RADIUS: radius (km) of an ROI that contains all of GrIS (1300)
- MAX_ICE_CORE_DIST: the farthest reach (km) of any ice core to affect GHF
    estimates to pad GrIS training data (150)
"""
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

MAX_GHF  = 150   # max limit of ghf considered for plotting only
GREENLAND_RADIUS = 1300
MAX_ICE_CORE_DIST = 150. # radius of gaussian paddings in Greenland

def _make_absolute(path):
    if not os.path.isabs(path):
        path = os.path.join(os.path.dirname(os.path.realpath(__file__)), path)
    return path

OUT_DIR = _make_absolute(os.getenv('OUT_DIR', 'plots/'))
GLOBAL_CSV = _make_absolute(os.getenv('GLOBAL_CSV', 'global.csv'))
GRIS_CSV = _make_absolute(os.getenv('GRIS_CSV', 'gris_features.csv'))
GRIS_ICE_CORE_CSV = _make_absolute(os.getenv('GRIS_ICE_CORE_CSV', 'gris_ice_cores.csv'))

# dict from categorical feature name to its anticipated values
CATEGORICAL_FEATURES = {
    'G_u_m_vel_': range(1, 13),
    'lthlgy_mod': [1, 2, 3],
    'G_ther_age': range(1, 7),
}
PROXIMITY_FEATURES = ['G_d_2yng_r', 'd2_transfo', 'd_2hotspot', 'd_2ridge', 'd_2trench', 'd_2volcano']
# features included in this list are dropped as soon as csv source is loaded
IGNORED_COLS = []

GBRT_PARAMS = {
    'loss': 'ls',               # 'ls' refers to sum of squares loss (i.e least squares regression)
    'learning_rate': 0.05,      # ν, shrinkage factor
    'n_estimators': 1000,       # M, total no. of regression trees
    'subsample': 1.,            # values less than 1 lead to stochastic GBRT
    'max_depth': 4,             # J, individual tree depth
    'max_features': 0.3,        # proportion of all features used in each tree
    'verbose': 0,               # verbosity of reporting
}
FEATURE_NAMES = {
    'age'       : 'age',
    'crusthk_cr': 'crustal thickness',
    'd2_transfo': 'dist. to transform ridge',
    'd_2hotspot': 'dist. to hotspots',
    'd_2ridge'  : 'dist. to ridge',
    'd_2trench' : 'dist. to trench',
    'd_2volcano': 'dist. to volcano',
    'ETOPO_1deg': 'topography',
    'G_d_2yng_r': 'dist. to young rift',
    'G_heat_pro': 'heat production provinces',
    'G_ther_age': 'last thermo-tectonic event',
    'G_u_m_vel_': 'upper mantle vel. structure',
    'litho_asth': 'lithos.-asthenos. boundary',
    'lthlgy_mod': 'rock type',
    'magnetic_M': 'magnetic anom.',
    'moho_GReD' : 'depth to Moho',
    'thk_mid_cr': 'thickness of middle crust',
    'thk_up_cru': 'thickness of upper crust',
    'upman_den_': 'upper mantle density anom.',
    'WGM2012_Bo': 'Bougeur gravity anom.',
}

GREENLAND = pd.read_csv(GRIS_ICE_CORE_CSV)

SPECTRAL_CMAP = plt.get_cmap('spectral', 13)
SPECTRAL_CMAP.set_under('black')
SPECTRAL_CMAP.set_over('grey')


def _load_data_set(path):
    data = pd.read_csv(path, index_col=0)
    data = data.drop(IGNORED_COLS, axis=1)
    for categorical, categories in CATEGORICAL_FEATURES.items():
        unknown_categories = set(data[categorical]).difference(categories)
        assert not unknown_categories, \
               'categorical feature %s in %s has unexpected value(s):\n%s' % \
               (categorical, path, ', '.join(str(x) for x in unknown_categories))
        # mark all categorical features as dtype=categorical in pandas data
        # frame (cf. https://pandas.pydata.org/pandas-docs/stable/categorical.html).
        # We need to do this because otherwise when get_dummies() is invoked
        # on separate data sets the resulting columns (and hence the shape of
        # data frame) depends on the values observed in that specific data set.
        # The categories argument forces the choice of dummy variables
        # produced below.
        data[categorical] = data[categorical].astype('category', categories=categories)
    data = pd.get_dummies(data, columns=CATEGORICAL_FEATURES.keys())

    # sort columns alphabetically
    data = data[sorted(list(data), key=lambda s: s.lower())]

    return data


def load_global_data():
    return _load_data_set(GLOBAL_CSV)


def load_gris_data():
    return _load_data_set(GRIS_CSV)


# Approximates GHF values at rows with unknown GHF according to a Gaussian
# decay formula based on known GHF values in GREENLAND.
def fill_in_greenland_GHF(data):
    dist_cols = []
    ghf_cols = []
    for _, point in GREENLAND.iterrows():
        # distance from each existing data point used to find gaussian
        # estimates for GHF: data['distance_X'] is the distance of each row to
        # data point X.
        dist_col = 'distance_' + point.core
        dist_cols.append(dist_col)
        data[dist_col] = haversine_distances(data, (point.lon, point.lat))
        # GHF estimates from gaussians centered at each existing data
        # point: data['GHF_radial_X'] is the GHF estimate corresponding to
        # data point point X.
        ghf_col = 'GHF_radial_' + point.core
        ghf_cols.append(ghf_col)
        data[ghf_col] = data.apply(
            lambda row: point.ghf * np.exp(- row[dist_col] ** 2. / point.rad ** 2),
            axis=1
        )
        data.loc[data[dist_col] > MAX_ICE_CORE_DIST, ghf_col] = np.nan

    data['GHF'] = data[ghf_cols].mean(skipna=True, axis=1)
    data = data.drop(dist_cols + ghf_cols, axis=1)

    # sort columns alphabetically
    data = data[sorted(list(data), key=lambda s: s.lower())]

    return data[data.GHF.notnull()], data[data.GHF.isnull()].drop('GHF', axis=1)

# Put together X_train, y_train, X_test (y_test is unknown) for Greenland GHF
# prediction using global dataset.
def greenland_train_test_sets():
    data_global, data_gris = load_global_data(), load_gris_data()
    gris_known, X_test = fill_in_greenland_GHF(data_gris)
    gris_known = pd.concat([data_global, gris_known])
    return gris_known.drop(['GHF'], axis=1), gris_known['GHF'], X_test


# Returns a random longitude-latitude pair that serves as the center of
# validation circle. The region argument specifies a spatial constraint on
# the latitude and longitude of the center:
#   1. None: no constraint
#   2. 'NA' (North America): 45 < lat, -100 < lon < -45
#   3. 'WE' (Western Europe): 45 < lat, -45 < lon < 50
#   4. 'NA-WE' (both of the above): 45 < lat, -100 < lon < 50
# FIXME document units of density (samples per million km^2), distance (km)
def random_prediction_ctr(data, radius, min_density=0, region='NA-WE'):
    if region is None:
        cands = data
    elif region == 'NA':
        cands = data.loc[(45 < data.Latitude_1) & (-100 < data.Longitude_1) & (data.Longitude_1 < -45)]
    elif region == 'WE':
        cands = data.loc[(45 < data.Latitude_1) & ( -45 < data.Longitude_1) & (data.Longitude_1 <  50)]
    elif region == 'NA-WE':
        cands = data.loc[(45 < data.Latitude_1) & (-100 < data.Longitude_1) & (data.Longitude_1 <  50)]
    else:
        raise Exception('Invalid value for region given ("%s")' % str(region))

    while True:
        center = cands.sample(n=1)
        center = center.Longitude_1, center.Latitude_1
        roi, non_roi = split_by_distance(data, center, radius)
        area = np.pi * (radius / 1000.) ** 2
        # FIXME this can loop infinitely if a large enough min_density is given
        if len(roi) / area >= min_density:
            return round(center[0], 2), round(center[1], 2)


# returns a pair of DataFrames: one containing rows in data that are closer
# than radius to center, and those that are not.
def split_by_distance(data, center, radius):
    # store distances in a temporary column '_distance'
    data['_distance'] = haversine_distances(data, center)
    within = data[data._distance < radius].drop('_distance', axis=1)
    beyond = data[data._distance > radius].drop('_distance', axis=1)
    data.drop('_distance', axis=1, inplace=True)

    return within, beyond


# calculates the haversine distance (in km) between two longitutde-latitude
# pairs. Each argument is an iterable with two entries: the longitude and the
# latitude of the corresponding point
def haversine_distance(p1, p2):
    lon1, lat1 = p1
    lon2, lat2 = p2
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a))
    km = 6367 * c
    return km


# returns a column containing distances of each row of data from center
def haversine_distances(data, center):
    def _haversine(row):
        p = row[['Longitude_1', 'Latitude_1']].as_matrix()
        return haversine_distance(p, center)

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
        data_train = pd.concat([data_train, additional_train])
        data_test = reduced_data_test

    X_train, y_train = data_train.drop('GHF', axis=1), data_train['GHF']
    X_test,  y_test  = data_test.drop('GHF', axis=1),  data_test['GHF']

    return X_train, y_train, X_test, y_test


def tune_params(data, param_grid, cv_fold=10):
    def _score(reg, X_test, y_test):
        y_pred = reg.predict(X_test)
        return sqrt(mean_squared_error(y_test, y_pred)) / np.mean(y_test)

    gbm = GradientBoostingRegressor(**GBRT_PARAMS)
    search = GridSearchCV(gbm, param_grid, scoring=_score, cv=cv_fold, n_jobs=1, verbose=10)
    search.fit(data.drop(['Latitude_1', 'Longitude_1', 'GHF'], axis=1), data['GHF'])
    print search.best_params_


# plots a series of values at given latitude and longitude positions
def plot_values_on_map(m, lons, lats, values,
                    parallel_step=20., meridian_step=60.,
                    clim=(20., 150.), clim_step=10,
                    colorbar_args={}, scatter_args={}, cbar_label='mW m$^{-2}$'):
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
    cbar.set_label(cbar_label)
    labels = range(int(clim[0]), int(clim[1]) + 1, clim_step)
    cbar.set_ticks(labels)
    cbar.set_ticklabels(labels)
    plt.clim(*clim)


# plots a series of GHF values at given latitude and longitude positions in
# ascii format
def plot_values_on_map_pcolormesh(m, lons, lats, values,
                    parallel_step=20., meridian_step=60.,
                    clim=(20., 150.), clim_step=10,
                    colorbar_args={}, pcolor_args={}, cbar_label='mW m$^{-2}$'):
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
    cbar.set_label(cbar_label)
    labels = range(int(clim[0]), int(clim[1]) + 1, clim_step)
    cbar.set_ticks(labels)
    cbar.set_ticklabels(labels)
    plt.clim(*clim)


# plots a series of GHF values at given latitude and longitude positions in
# ascii format interpolated using basemap transform_scalar functions
def plot_values_on_map_pcolormesh_interp(m, lons, lats, values,
                    parallel_step=20., meridian_step=60.,
                    clim=(20., 150.), clim_step=10,
                    colorbar_args={}, pcolor_args={}, cbar_label='mW m$^{-2}$'):
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
    cbar.set_label(cbar_label)
    labels = range(int(clim[0]), int(clim[1]) + 1, clim_step)
    cbar.set_ticks(labels)
    cbar.set_ticklabels(labels)
    plt.clim(*clim)


# Plots scatter plots between GHF and all features in data set
def plot_GHF_feature_projections(data):
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


# saves current matplotlib plot to given filename in OUT_DIR
def save_cur_fig(filename, title=None, set_title_for='ax'):
    if title:
        fig = plt.gcf()
        if set_title_for == 'fig':
            fig.suptitle(title)
        elif set_title_for == 'ax':
            fig.get_axes()[0].set_title(title)
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


def get_gbrt(**gbrt_params):
    _gbrt_params = GBRT_PARAMS.copy()
    _gbrt_params.update(gbrt_params)
    return GradientBoostingRegressor(**_gbrt_params)


# Trains and returns a GradientBoostingRegressor over the given training
# feature and value vectors.
def train_gbrt(X_train, y_train, **gbrt_params):
    sys.stderr.write('-> Training ...')
    start = time()
    # allow keyword arguments to override default GDR parameters
    reg = get_gbrt(**gbrt_params)
    reg.fit(X_train, y_train)
    sys.stderr.write(' (%.2f secs)\n' % (time() - start))

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

    data = load_global_data() # FIXME pass in mean_ghf as argument
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
def plot_values_histogram(values):
    hist, _, _ = plt.hist(values, np.linspace(0, MAX_GHF, 61), normed=True,
                          lw=1, alpha=.7, color='k', edgecolor='k')
    plt.xlabel('GHF (mW m$^{-2}$)')
    plt.ylabel('Normalized Frequency')
    plt.grid(linestyle='dotted')
    plt.xlim([0, MAX_GHF])
    plt.ylim([0, max(hist) * 1.1])
