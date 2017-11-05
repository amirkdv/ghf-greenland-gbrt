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
import math

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
    'upper_mantle_vel_structure': range(1, 13),
    'rock_type': [1, 2, 3],
    'thermo_tecto_age': range(1, 7),
}
PROXIMITY_FEATURES = [
    'd_2_hotspot',
    'd_2_ridge',
    'd_2_trans_ridge',
    'd_2_trench',
    'd_2_volcano',
    'd_2_young_rift',
]
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
    'age':                          'age',
    'bougeur_gravity_anomaly':      'Bougeur gravity anom.',
    'd_2_hotspot':                  'dist. to hotspots',
    'd_2_ridge':                    'dist. to ridge',
    'd_2_trans_ridge':              'dist. to transform ridge',
    'd_2_trench':                   'dist. to trench',
    'd_2_volcano':                  'dist. to volcano',
    'd_2_young_rift':               'dist. to young rift',
    'depth_to_moho':                'depth to Moho',
    'heat_prod_provinces':          'heat production provinces',
    'lithos_asthenos_bdry':         'lithos.-asthenos. boundary',
    'magnetic_anomaly':             'magnetic anom.',
    'rock_type':                    'rock type',
    'thermo_tecto_age':             'last thermo-tectonic event',
    'thickness_crust':              'crustal thickness',
    'thickness_middle_crust':       'thickness of middle crust',
    'thickness_upper_crust':        'thickness of upper crust',
    'topography':                   'topography',
    'upper_mantle_density_anomaly': 'upper mantle density anom.',
    'upper_mantle_vel_structure':   'upper mantle vel. structure',
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
        dtype = pd.api.types.CategoricalDtype(categories=categories)
        data[categorical] = data[categorical].astype(dtype)
    data = pd.get_dummies(data, columns=CATEGORICAL_FEATURES.keys())

    # sort columns alphabetically
    data = data[sorted(list(data), key=lambda s: s.lower())]

    return data


def load_global_data():
    return _load_data_set(GLOBAL_CSV)


def load_gris_data():
    return _load_data_set(GRIS_CSV)


def fill_in_greenland_GHF(data):
    """ Approximates GHF values at rows with unknown GHF according to a
        Gaussian decay formula based on known GHF values in GREENLAND. For each
        record, any ice core within MAX_ICE_CORE_DIST gives a GHF estimate
        through a Gaussian decay centered at the ice core. The final GHF value
        of each record is the average estimates provided by all nearby ice
        cores.

        Args:
            data (pandas.DataFrame):
                the data set to be updated, typically all GrIS records most of
                which do not have a GHF value. Existing GHF values are
                overwritten.
    """
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

def greenland_train_test_sets():
    """ Put together X_train, y_train, X_test (y_test is unknown) for Greenland
        GHF prediction using global dataset.
    """
    data_global, data_gris = load_global_data(), load_gris_data()
    gris_known, X_test = fill_in_greenland_GHF(data_gris)
    gris_known = pd.concat([data_global, gris_known])
    return gris_known.drop(['GHF'], axis=1), gris_known['GHF'], X_test


def random_prediction_ctr(data, radius, min_density=0, region='NA-WE'):
    """ Returns a random longitude-latitude pair that serves as the center of
        validation circle.

        Args:
            data (pandas.DataFrame):
                the entire data set used to pick random centers that satisfy
                the required ROI density minimum. Only relevant columns are lat
                and lon.
            region (string):
                specifies a spatial constraint on the latitude and longitude of
                the center. None is no constraint, 'NA' is North America, 'WE'
                is Western Europe, and 'NA-WE' is both the latter two.
            min_density (float):
                the minimum required ROI density; otherwise a randomly chosen
                ROI is dismissed.
    """
    if region is None:
        cands = data
    elif region == 'NA':
        cands = data.loc[(45 < data.lat) & (-100 < data.lon) & (data.lon < -45)]
    elif region == 'WE':
        cands = data.loc[(45 < data.lat) & ( -45 < data.lon) & (data.lon <  50)]
    elif region == 'NA-WE':
        cands = data.loc[(45 < data.lat) & (-100 < data.lon) & (data.lon <  50)]
    else:
        raise Exception('Invalid value for region given ("%s")' % str(region))

    while True:
        center = cands.sample(n=1)
        center = center.lon, center.lat
        roi, non_roi = split_by_distance(data, center, radius)
        area = np.pi * (radius / 1000.) ** 2
        # FIXME this can loop infinitely if a large enough min_density is given
        if len(roi) / area >= min_density:
            return round(center[0], 2), round(center[1], 2)


def split_by_distance(data, center, radius):
    """ Returns a pair of data frames, one containing rows in data that are
        closer than radius to center, and the other containing those that are
        not.

        Args:
            data (pandas.DataFrame): the entire data set.

        Return:
            (within, beyond):
                two data frames resulting from partitioning the data set to
                within ROI and beyond ROI.
    """
    # store distances in a temporary column '_distance'
    data['_distance'] = haversine_distances(data, center)
    within = data[data._distance < radius].drop('_distance', axis=1)
    beyond = data[data._distance > radius].drop('_distance', axis=1)
    data.drop('_distance', axis=1, inplace=True)

    return within, beyond


def haversine_distance(p1, p2):
    """ Calculates the haversine distance (in km) between two
        longitutde-latitude pairs. Each argument is an iterable with two
        entries: the longitude and the latitude of the corresponding point
    """
    lon1, lat1 = p1
    lon2, lat2 = p2
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * math.asin(np.sqrt(a))
    km = 6367 * c
    return km

def haversine_distances(data, center):
    """ Returns a column containing distances of each row of data from center.
    """
    def _haversine(row):
        p = row[['lon', 'lat']].as_matrix()
        return haversine_distance(p, center)

    return data.apply(_haversine, axis=1)

def roi_density_to_test_size(density, radius, num_samples):
    """ Translates ROI density (sample per 1e6 km^2) for a given radius (km) to
        a test_size parameter (between 0 and 1) acceptable by `train_test_split`
        to partition within-ROI samples to a validation set and a supplementary
        within-ROI training data.
    """
    area = np.pi * (radius / 1000.) ** 2
    test_size = 1 - (area * density) / num_samples
    max_density = num_samples / area
    assert max_density >= density, \
        'demanded density (%.2f) larger than max density in ROI (%.2f)' % (density, max_density)
    return test_size


def split_with_circle(data, center, roi_density=None, radius=3500):
    """ Splits rows in data into a training and test set according to the
        following rule: consider an ROI with given center and radius. The
        training set is those rows outside ROI and a randomly chosen subset of
        those rows within ROI (proportion of points in ROI kept for test is
        calculated based on demanded ensity).

        Args:
            data (pandas.DataFrame): the entire data set.
            center (tuple): lon-lat coordinates of ROI center.
            roi_density (float): required density, cf. roi_density_to_test_size

        Return:
            (X_train, y_train, X_test, y_test):
            A tuple of 4 data frames of dimensions nxp, nx1, (N-n)xp, (N-n)x1
            where N is the total number of records, n is the number of training
            records and p is the number of features.
    """
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
    """ Wraps `sklearn.model_selection.GridSearchCV` to use normalized rmse as
        a measure of performance. We cannot control the cross-validation scheme
        and must rely on the random selection done by sklearn.

        Args:
            data (pandas.DataFrame): entire data set.
            param_grid (dict): a dictionary from parameter names to lists of
            values to be considered for each parameter.

        Return:
            GridSearchCV.best_params_
    """
    def _score(reg, X_test, y_test):
        y_pred = reg.predict(X_test)
        return sqrt(mean_squared_error(y_test, y_pred)) / np.mean(y_test)

    gbm = GradientBoostingRegressor(**GBRT_PARAMS)
    search = GridSearchCV(gbm, param_grid, scoring=_score, cv=cv_fold, n_jobs=1, verbose=10)
    search.fit(data.drop(['lat', 'lon', 'GHF'], axis=1), data['GHF'])
    print search.best_params_

def plot_values_on_map(m, lons, lats, values,
                    parallel_step=20., meridian_step=60.,
                    clim=(20., 150.), clim_step=10,
                    colorbar_args={}, scatter_args={}, cbar_label='mW m$^{-2}$'):
    """ Plots a series of values at given latitude and longitude positions on a
        given basemap object as points.

        Args:
            m (basemap.Basemap): map object to draw data on.
            lons: one-dimensional list (native, numpy, or pandas) of longitudes.
            lons: similar, latitudes.
            ghfs: similar, GHF values.
    """
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


def plot_values_on_map_pcolormesh(m, lons, lats, values,
                    parallel_step=20., meridian_step=60.,
                    clim=(20., 150.), clim_step=10,
                    colorbar_args={}, pcolor_args={}, cbar_label='mW m$^{-2}$'):
    """ Plots a series of values at given latitude and longitude positions as a
        pseudocolor map.

        Args:
            m (basemap.Basemap): map object to draw data on.
            lons: one-dimensional list (native, numpy, or pandas) of longitudes.
            lons: similar, latitudes.
            ghfs: similar, GHF values.
    """
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
        j = np.floor(lons_array).tolist().index(np.floor(item[0]))
        i = np.floor(lats_array).tolist().index(np.floor(item[1]))
        ascii[i][j] = item[2]

    ascii = np.ma.masked_where(np.isnan(ascii),ascii)

    cs = m.pcolormesh(x,y,ascii,**pcolor_args)

    cbar = m.colorbar(cs, **colorbar_args)
    cbar.set_label(cbar_label)
    labels = range(int(clim[0]), int(clim[1]) + 1, clim_step)
    cbar.set_ticks(labels)
    cbar.set_ticklabels(labels)
    plt.clim(*clim)


def plot_values_on_map_pcolormesh_interp(m, lons, lats, values,
                    parallel_step=20., meridian_step=60.,
                    clim=(20., 150.), clim_step=10,
                    colorbar_args={}, pcolor_args={}, cbar_label='mW m$^{-2}$'):
    """ Plots a series of values at given latitude and longitude positions as a
        pseudocolor map interpolated using basemap.Basemap.transform_scalar.

        Args:
            m (basemap.Basemap): map object to draw data on.
            lons: one-dimensional list (native, numpy, or pandas) of longitudes.
            lons: similar, latitudes.
            ghfs: similar, GHF values.
    """
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
        j = np.floor(lons_array).tolist().index(np.floor(item[0]))
        i = np.floor(lats_array).tolist().index(np.floor(item[1]))
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

def plot_GHF_feature_projections(data):
    """ Plots scatter plots between GHF and all features in data set.
    """
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

def save_cur_fig(filename, title=None, set_title_for='ax'):
    """ Saves current matplotlib plot to given filename in OUT_DIR.

        Args:
            filename (string): path relative to OUT_DIR.
            title (string): title used in logging and (maybe) in plot.
            set_title_for (string):
                if 'ax' only sets the given title for the first axis object in
                figure, if 'fig' sets the given title for the whole figure,
                otherwise title is only used for logging.
    """
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
    """ Dumps an object to a pickle file in OUT_DIR.
    """
    with open(os.path.join(OUT_DIR, path), 'w') as f:
        pickle.dump(obj, f)
    sys.stderr.write('dumped %s to %s.\n' % (comment if comment else 'object', path))


def pickle_load(path):
    """ Loads an object from a pickle file in OUT_DIR.
    """
    with open(os.path.join(OUT_DIR, path), 'rb') as f:
        return pickle.load(f)


def train_linear(X_train, y_train):
    """ Trains a linear regression model using the given training features and
        GHFs.
    """
    reg = LinearRegression()
    reg.fit(X_train, y_train)
    return reg


def get_gbrt(**gbrt_params):
    """ Builds a GBRT object (not trained) with the specified parameters. The
        default values of all unspecified parameters are as in `GBRT_PARAMS`.
    """
    _gbrt_params = GBRT_PARAMS.copy()
    _gbrt_params.update(gbrt_params)
    return GradientBoostingRegressor(**_gbrt_params)


def train_gbrt(X_train, y_train, **gbrt_params):
    """ Trains a GBRT model using the given training features and GHFs.
    """
    sys.stderr.write('-> Training ...')
    start = time()
    # allow keyword arguments to override default GDR parameters
    reg = get_gbrt(**gbrt_params)
    reg.fit(X_train, y_train)
    sys.stderr.write(' (%.2f secs)\n' % (time() - start))

    return reg


def error_summary(y_test, y_pred):
    """ Returns r^2 of linear correlation between y, y^ (predictions) and the
        RMSE between y, y^ normalized to the
        average of y; the latter is a unitless and *scale-invariant* measure of
        performance.
        - cf. http://stats.stackexchange.com/a/190948
          RMSE normalized to mean of y is scale-invariant.
        - cf. http://www.stat.columbia.edu/~gelman/research/published/standardizing7.pdf
          r^2 is not scale-invariant.

        Args:
            y_test: list or vector of known values for validation set.
            y_pred: list or vector of predicted values for validation set.

        Return:
            (r2, rmse)
  """
    _, _, r_value, _, _= scipy.stats.linregress(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred)) / np.mean(y_test)
    return r_value ** 2, rmse


def plot_test_pred_linregress(y_test, y_pred, label=None, color='blue'):
    """ Plots the linear correlation of two GHF value series (known test values
        and predicted values) and saves the plot to OUT_DIR/filename.
    """
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

def plot_values_histogram(values):
    """ Plots the histogram of given GHF values over 60 bins of GHF from 0 to
        `MAX_GHF`.
    """
    hist, _, _ = plt.hist(values, np.linspace(0, MAX_GHF, 61), normed=True,
                          lw=1, alpha=.7, color='k', edgecolor='k')
    plt.xlabel('GHF (mW m$^{-2}$)')
    plt.ylabel('Normalized Frequency')
    plt.grid(linestyle='dotted')
    plt.xlim([0, MAX_GHF])
    plt.ylim([0, max(hist) * 1.1])
