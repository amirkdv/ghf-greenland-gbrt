import sys
import scipy
import random
import os.path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from random import randint
from mpl_toolkits.basemap import Basemap
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from math import radians, sin, cos, asin, sqrt, floor

pd.set_option('display.max_columns', 80)
plt.ticklabel_format(useOffset=False)

MAX_GHF  = 150   # max limit of ghf considered
N_ESTIMATORS = 300 # number of estimators for gradient boosting regressor

OUT_DIR = 'global_learning_plots_gb_circles_gaussian/'
OUT_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)), OUT_DIR)

GLOBAL_CSV = '1deg_all_resampled_w_missing_from_goutorbe.csv'
GRIS_CSV = '1deg_greenland_GHF_added2.csv' # srb
IGNORED_COLS = [
    'OBJECTID_1', 'continent', 'lthlgy_all', 'num_in_cel', 'num_in_con',
     'WGM2012_Ai', 'depthmoho', 'moho_Pasya', 'lithk_cona',
    #'G_heat_pro', 'd_2hotspots', 'd_2volcano', 'crusthk_cr', 'litho_asth',
    #'d_2trench', 'G_d_2yng_r', 'd_2ridge', 'd2_transfo'
    ]

# The only existing data points for Greenland are at the following ice
# cores: data_points[X] contains info for data point at ice core X. 'rad'
# is the radius used for Gaussian estimates from each point.
GREENLAND = pd.DataFrame({
    'lat':  [ 72.58,  72.60,   65.18,  75.10,   77.18,   61.40,   60.98,   60.73,      66.50],
    'lon':  [-37.64, -38.50,  -43.82, -42.32,  -61.13,  -48.18,  -45.98,  -45.75,     -50.33],
    'ghf':  [ 51.3,   60.,     20.,    135.,    50.,     43.,     32.,     51.,        31.05],  # NOTE: GHF at NGRIP is questionable
    'rad':  [ 1000.,  1000.,   1000.,  140.,    1000.,   1000.,   1000.,   1000.,      1000.],
    'core': ['GRIP', 'GISP2', 'DYE3', 'NGRIP', 'CC',    'SASS1', 'SASS2', 'LANGSETH', 'GAP'],
})
GREENLAND.set_index('core')

# Reads csv source and applies general filters.
def read_csv(path):
    data = pd.read_csv(path, index_col=0, na_values=-999999)

    data[data['lthlgy_mod'] == 0] = np.nan
    data[data['lthlgy_mod'] == -9999] = np.nan
    data.dropna(inplace=True)

    # drop rows with out of range GHF
    #data = data[data['GHF'] < MAX_GHF] # srb
    data = data.drop(IGNORED_COLS, axis=1)

    return data

# loads GLOBAL_CSV and GRIS_CSV, performs GRIS specific filters, and returns a
# single DataFrame.
def load_global_gris_data():
    data_global = read_csv(GLOBAL_CSV)
    data_gris = read_csv(GRIS_CSV)
    data_gris = process_greenland_data(data_gris)

    # raw data has rounded integer values for GHF, add a little dispersion
    # HACK turn off for checksum purposes
    #data_global['GHF'] = data_global['GHF'] + abs(np.random.normal(0, 0.25)) # srb

    data = pd.concat([data_global, data_gris])
    data = pd.get_dummies(data, columns=['G_u_m_vel_', 'lthlgy_mod', 'G_ther_age'])

    return data

# GRIS specific filters
def process_greenland_data(data):
    # mapping from old to new values of lthlgy_mod
    # Legend: volcanic=1, metamorphic=2, sedimentary=3
    lthlgy_mod_rewrites = {1: 2, 2: 3, 3: 3, 4: 3, 5: 1, 6: 2, 7: 1, 8: 3, 9: 2, 10: 2}
    data['lthlgy_mod'] = data.apply(
        lambda row: lthlgy_mod_rewrites[row['lthlgy_mod']],
        axis=1
    )
    return data

# Approximates GHF values at rows with unknown GHF according to a Gaussian
# decay formula based on known GHF values in GREENLAND.
def fill_in_greenland_GHF(data):
    def gauss(amp, dist, rad):
        return amp/np.exp(dist**2./rad**2.)

    # distance beyond which ice core GHF effect will be ignored
    max_ice_core_dist = 150.

    dist_cols = []
    ghf_cols = []
    for _, point in GREENLAND.iterrows():
        # distance from each existing data point used to find gaussian
        # estimates for GHF: data['distance_X'] is the distance of each row to
        # data point X.
        dist_col = 'distance_' + point.core
        dist_cols.append(dist_col)
        data[dist_col] = haversine_distance(data, (point.lon, point.lat))
        # GHF estimates from gaussians centered at each existing data
        # point: data['GHF_radial_X'] is the GHF estimate corresponding to
        # data point point X.
        ghf_col = 'GHF_radial_' + point.core
        ghf_cols.append(ghf_col)
        data[ghf_col] = data.apply(
            lambda row: gauss(point.ghf, row[dist_col], point.rad),
            axis=1
        )
        data.loc[data[dist_col] > max_ice_core_dist, ghf_col] = np.nan

    data['GHF'] = data[ghf_cols + ['GHF']].mean(skipna=True, axis=1)
    #data['GHF'] = data[ghf_cols].mean(skipna=True, axis=1) # srb
    data = data.drop(dist_cols + ghf_cols, axis=1)

    data.loc[data.GHF == 135.0, 'GHF'] = 0 # FIXME artificially forced to 135.0 in source
    # The gris data set has many rows with feature values but no GHF
    # measurements. We want to predict GHF for these.
    #gris_unknown = data[data.GHF == 0] # srb
    gris_unknown = data[data.GHF.isnull()] #srb
    data.loc[data.GHF == 0, 'GHF'] = np.nan
    data.dropna(inplace=True)
    return data, gris_unknown

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

# splits rows in data into a training and test set according to the following
# rule: consider a circle C centered at center and with radius max_dist. The
# training set is those rows outside C and a randomly chosen subset of those
# rows within C (proportion of points in C kept for test is given by
# test_size; float between 0 and 1).
def split(data, center, test_size=.15, max_dist=3500):
    data_test, data_train = split_by_distance(data, center, max_dist)
    additional_train, data_test = train_test_split(data_test, random_state=0, test_size=test_size)
    data_train = pd.concat([data_train, additional_train])

    X_train, y_train = data_train.drop('GHF', axis=1), data_train['GHF']
    X_test,  y_test  = data_test.drop('GHF', axis=1),  data_test['GHF']

    return X_train, y_train, X_test, y_test

# plots a series of GHF values at given latitude and longitude positions
def plot_GHF_on_map(m, lons, lats, values,
                    parallel_step=20., meridian_step=60.,
                    clim=(20., 150.), clim_step=10,
                    colorbar_args={}, scatter_args={}):
    m.drawparallels(np.arange(-80., 81., parallel_step), labels=[1, 0, 0, 0], fontsize=10)
    m.drawmeridians(np.arange(-180., 181., meridian_step), labels=[0, 0, 0, 1], fontsize=10)
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

# plots a series of GHF values at given latitude and longitude positions in ascii format
def plot_GHF_on_map_pcolormesh(m, lons, lats, values,
                    parallel_step=20., meridian_step=60.,
                    clim=(20., 150.), clim_step=10,
                    colorbar_args={}, pcolor_args={}):
    m.drawparallels(np.arange(-80., 81., parallel_step), labels=[1, 0, 0, 0], fontsize=10)
    m.drawmeridians(np.arange(-180., 181., meridian_step), labels=[0, 0, 0, 1], fontsize=10)
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

# plots a series of GHF values at given latitude and longitude positions in ascii format
# interpolated using basemap transform_scalar functions
def plot_GHF_on_map_pcolormesh_interp(m, lons, lats, values,
                    parallel_step=20., meridian_step=60.,
                    clim=(20., 150.), clim_step=10,
                    colorbar_args={}, pcolor_args={}):
    m.drawparallels(np.arange(-80., 81., parallel_step), labels=[1, 0, 0, 0], fontsize=10)
    m.drawmeridians(np.arange(-180., 181., meridian_step), labels=[0, 0, 0, 1], fontsize=10)
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

    ascii_interp, x, y = m.transform_scalar(ascii, lons_array, lats_array, 5*len(lons_array), 5*len(lats_array), returnxy=True)

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
    plt.title(title)
    plt.savefig(os.path.join(OUT_DIR, filename), dpi=400)
    sys.stderr.write('Saved %s to %s.\n' % (repr(title), filename))
    plt.clf()

GDR_PARAMS = {
    'loss': 'ls',
    'learning_rate': 0.05,
    'n_estimators': N_ESTIMATORS,
    'subsample': 1.0,
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
    #'verbose': 0,
    'verbose': 10,
    'max_leaf_nodes': None,
    'warm_start': False,
    'presort': 'auto',
}

# Trains and returns a GradientBoostingRegressor over the given training
# feature and value vectors. Feature importance values are stored in
# OUTDIR/logfile
def train_regressor(X_train, y_train, logfile):
    reg = GradientBoostingRegressor(**GDR_PARAMS)
    sys.stderr.write('Training Gradient Boosting Regressor ...')
    reg.fit(X_train, y_train)
    sys.stderr.write('\n')

    importance = reg.feature_importances_
    hdrs = list(X_train.columns.values)
    logs = np.asarray(sorted(zip(hdrs, importance), key=lambda x: x[1]))
    np.savetxt(os.path.join(OUT_DIR, logfile), logs, fmt="%s")
    sys.stderr.write('Saved feature importances to %s.\n' % (logfile))

    return reg

# plots the linear regression of two GHF value series (known test values and
# predicted values) and saves the plot to OUT_DIR/filename.
def plot_test_pred_linregress(y_test, y_pred, filename, title=None):
    plt.clf()
    slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(y_test, y_pred)

    plt.scatter(y_test, y_pred, label='tests, r2=%f'%r_value**2)
    plt.grid(True)
    plt.axes().set_aspect('equal')
    plt.xlabel('$GHF$')
    plt.ylabel('$\widehat{GHF}$')
    plt.xlim([0, MAX_GHF])
    plt.ylim([0, MAX_GHF])

    title = title + '\n$r^2=%.3f, p=%.1e$' % (r_value**2, p_value)
    save_cur_fig(filename, title=title)

# plots the histogram of given GHF values
def plot_GHF_histogram(values, max_density=.05):
    plt.hist(values, bins=np.linspace(0, 150, 16), linewidth=2, color='blue', normed=True)
    plt.xlabel('GHF (mW m$^{-2}$)')
    plt.ylabel('Normalized Frequency')
    plt.grid(True)
    plt.xlim([0, MAX_GHF])
    plt.ylim([0, max_density])

# saves GRIS known and predicted GHF and latitude/longitude and values as a
# numpy array in OUT_DIR/filename.
def save_gris_prediction_data(gris_unknown, gris_known, filename):
    LATS = np.hstack([gris_unknown.Longitude_1.as_matrix(), gris_known.Longitude_1.as_matrix()])
    LONS = np.hstack([gris_unknown.Latitude_1.as_matrix(),  gris_known.Latitude_1.as_matrix()])
    GHFS = np.hstack([y_gris, gris_known.GHF.as_matrix()])

    final = np.zeros([len(LATS), 3])
    final[:, 0] = LATS
    final[:, 1] = LONS
    final[:, 2] = GHFS

    path = os.path.join(OUT_DIR, filename)
    np.savetxt(path, final, delimiter=', ', header='lon, lat, ghf', fmt='%10.5f')
    sys.stderr.write('Saved gris data to %s\n' % filename)

# Returns the average RMSE over ncenters number of circles with radius
# max_dist keeping test_size (float between 0 and 1) of the points for testing.
def average_rmse(data, test_size, max_dist, ncenters):
    rmses = []
    for _ in range(ncenters):
        center = (randint(-180, 180), randint(-90, 90))
        print 'splitting data set with test_size=%.2f, max_dist=%d, center=%s' % \
              (test_size, max_dist, repr(center))
        X_train, y_train, X_test, y_test = split(data, center, test_size=test_size, max_dist=max_dist)
        if len(X_test) == 0:
            print 'no test points left; skipping'
            continue
        reg = train_regressor(X_train.drop(['Latitude_1', 'Longitude_1'], axis=1),
                              y_train, 'GHF_1deg_averaged_logfile.txt')
        y_pred = reg.predict(X_test.drop(['Latitude_1', 'Longitude_1'], axis=1))
        rmses.append(sqrt(mean_squared_error(y_test, y_pred)))
    return sum(rmses) / len(rmses)

def plot_average_rmse_fixed_test_size(data, test_size, max_dists, ncenters):
    avg_rmses = []
    for max_dist in max_dists:
        avg_rmses.append(average_rmse(data, test_size, max_dist, ncenters))
    plt.scatter(max_dists, avg_rmses)
    plt.grid(True)
    plt.xlabel('Testing set radius')
    plt.ylabel('$\mathbb{E}\left[\sqrt{MSE}\\right]$ over random centers')

def plot_average_rmse_fixed_max_dist(data, test_sizes, max_dist, ncenters):
    avg_rmses = []
    for test_size in test_sizes:
        avg_rmses.append(average_rmse(data, test_size, max_dist, ncenters))
    plt.scatter(test_sizes, avg_rmses)
    plt.grid(True)
    plt.xlabel('portion of points within circle kept for testing')
    plt.ylabel('$\mathbb{E}\left[\sqrt{MSE}\\right]$ over random centers')

def plot_average_rmse(data):
    max_dist = 3500
    plot_average_rmse_fixed_max_dist(data, [.05 * i for i in range(1, 20)], max_dist, 3)
    save_cur_fig('avg_rmse_ratio.png',
                 'Mean RMSEs for different test/training ratios (radius = %d)' % max_dist)

    test_size = .9
    plot_average_rmse_fixed_test_size(data, test_size, range(1000, 5000, 200), 3)
    save_cur_fig('avg_rmse_dist.png',
                 'Mean RMSEs for different test set radius (ratio = %.2f)' % test_size)

# Evaluate Gradient Boosting Regressor
# ====================================

# Create average RMSE plots for different test_size and max_dist values
# ---------------------------------------------------------------------
data = load_global_gris_data()
#plot_average_rmse(data)

# Prepare training and test sets for Greenland
# --------------------------------------------
gris_known, gris_unknown = fill_in_greenland_GHF(data)
center = GREENLAND.loc[GREENLAND['core'] == 'GRIP']
center = (float(center.lon), float(center.lat))
X_train, y_train, X_test, y_test = split(gris_known, center)

# Plot known GHF values for training and test sets
# ------------------------------------------------
m = Basemap(projection='robin',lon_0=0,resolution='c')
spectral_cmap = plt.get_cmap('spectral', 13)
spectral_cmap.set_under('black')
spectral_cmap.set_over('grey')
colorbar_args = {'location': 'bottom', 'pad': '10%'}
scatter_args = {'marker': 'o', 's': 15, 'lw': 0, 'cmap': spectral_cmap}

plot_GHF_on_map(m,
                X_train.Longitude_1.as_matrix(), X_train.Latitude_1.as_matrix(),
                y_train,
                colorbar_args=colorbar_args,
                scatter_args=scatter_args)
save_cur_fig('GHF_1deg_averaged_map_train.png', title='GHF at train set')

plot_GHF_on_map(m,
                X_test.Longitude_1.as_matrix(), X_test.Latitude_1.as_matrix(),
                y_test,
                colorbar_args=colorbar_args,
                scatter_args=scatter_args)
save_cur_fig('GHF_1deg_averaged_map_test.png', title='GHF at test set')

# Predict GHF over test set
# -----------------------------
reg = train_regressor(X_train.drop(['Latitude_1', 'Longitude_1'], axis=1),
                      y_train, 'GHF_1deg_averaged_logfile.txt')
y_pred = reg.predict(X_test.drop(['Latitude_1', 'Longitude_1'], axis=1))

m = Basemap(width=1600000, height=2650000, resolution='l',
            projection='stere', lat_ts=71, lon_0=-41.5, lat_0=72)
colorbar_args = {'location': 'right', 'pad': '5%', 'extend': 'both'}
scatter_args = {'marker': 'o', 's': 25, 'lw': 0, 'cmap': spectral_cmap}

plot_GHF_on_map(m,
                X_test.Longitude_1.as_matrix(), X_test.Latitude_1.as_matrix(),
                y_pred,
                parallel_step=5., meridian_step=10.,
                colorbar_args=colorbar_args,
                scatter_args=scatter_args)
save_cur_fig('Greenland_GHF_predicted_1deg.png',
             title='GHF predicted for Greenland (mW m$^{-2}$) \n globally trained')

# NOTE dropped 'Greenland_GHF_1deg.png': predicted and known GHF are overlayed.

# Plot GHF difference between predictions and known values
# --------------------------------------------------------
m = Basemap(projection='robin',lon_0=0,resolution='c')
seismic_cmap = plt.get_cmap('seismic', 20)
scatter_args = {'marker': 'o', 's': 15, 'lw': 0, 'cmap': seismic_cmap}
colorbar_args = {'location': 'bottom', 'pad': '10%'}

plot_GHF_on_map(m,
                X_test.Longitude_1.as_matrix(), X_test.Latitude_1.as_matrix(),
                y_test - y_pred,
                clim=(-10, 10), clim_step=2,
                colorbar_args=colorbar_args,
                scatter_args=scatter_args)
save_cur_fig('GHF_1deg_diff_map.png',
             title='GHF error on test set (true - predicted)')

m = Basemap(width=1600000, height=2650000, resolution='l',
            projection='stere', lat_ts=71, lon_0=-41.5, lat_0=72)
seismic_cmap = plt.get_cmap('seismic', 20)
scatter_args = {'marker': 'o', 's': 15, 'lw': 0, 'cmap': seismic_cmap}
colorbar_args = {'location': 'right', 'pad': '5%'}

plot_GHF_on_map(m,
                X_test.Longitude_1.as_matrix(), X_test.Latitude_1.as_matrix(),
                y_test - y_pred,
                clim=(-10, 10), clim_step=2,
                parallel_step=5., meridian_step=10.,
                colorbar_args=colorbar_args,
                scatter_args=scatter_args)
save_cur_fig('GHF_1deg_diff_map_Greenland.png',
             title='GHF error on test set (true - predicted)')

# Linear Regression between known and predicted values in test set
# ----------------------------------------------------------------
plot_test_pred_linregress(y_test, y_pred, 'GHF_1deg_averaged_plot.png',
                          title='Linear regression between predicted vs true GHF')

# Predictions for Greenland
# =========================
X_gris = gris_unknown.drop(['GHF'], axis=1)
y_gris = reg.predict(X_gris.drop(['Latitude_1', 'Longitude_1'], axis=1))

m = Basemap(width=1600000, height=2650000, resolution='l',
            projection='stere', lat_ts=71, lon_0=-41.5, lat_0=72)
seismic_cmap = plt.get_cmap('seismic', 20)
scatter_args = {'marker': 'o', 's': 20, 'lw': 0, 'cmap': spectral_cmap}
colorbar_args = {'location': 'right', 'pad': '5%'}
plot_GHF_on_map(m,
                X_gris.Longitude_1.as_matrix(), X_gris.Latitude_1.as_matrix(),
                y_gris,
                parallel_step=5., meridian_step=10.,
                colorbar_args=colorbar_args,
                scatter_args=scatter_args)
save_cur_fig('predicted_Greenland_GHF_1deg.png',
             title='GHF predicted for Greenland (mW m$^{-2}$)')

m = Basemap(width=1600000, height=2650000, resolution='l',
            projection='stere', lat_ts=71, lon_0=-41.5, lat_0=72)
scatter_args = {'marker': 'o', 's': 20, 'lw': 0, 'cmap': spectral_cmap}
colorbar_args = {'location': 'right', 'pad': '5%'}
plot_GHF_on_map(m,
                X_gris.Longitude_1.as_matrix(), X_gris.Latitude_1.as_matrix(),
                y_gris,
                parallel_step=5., meridian_step=10.,
                colorbar_args=colorbar_args,
                scatter_args=scatter_args)
scatter_args = {'marker': 'd', 's': 30, 'lw': 0, 'cmap': spectral_cmap}
plot_GHF_on_map(m,
                gris_known.Longitude_1.as_matrix(), gris_known.Latitude_1.as_matrix(),
                gris_known.GHF,
                parallel_step=5., meridian_step=10.,
                colorbar_args=colorbar_args,
                scatter_args=scatter_args)
save_cur_fig('TEST.png',
             title='GHF predicted for Greenland (mW m$^{-2}$)')

m = Basemap(projection='robin',lon_0=0,resolution='c')
spectral_cmap = plt.get_cmap('spectral', 13)
spectral_cmap.set_under('black')
spectral_cmap.set_over('grey')
colorbar_args = {'location': 'bottom', 'pad': '10%'}
pcolor_args = {'cmap': spectral_cmap}
plot_GHF_on_map_pcolormesh(m,
                X_train.Longitude_1.as_matrix(), X_train.Latitude_1.as_matrix(),
                y_train,
                colorbar_args=colorbar_args,
                pcolor_args=pcolor_args)
save_cur_fig('pcolormesh.png', title='GHF at train set')

m = Basemap(width=1600000, height=2650000, resolution='l',
            projection='stere', lat_ts=71, lon_0=-41.5, lat_0=72)
spectral_cmap = plt.get_cmap('spectral', 13)
spectral_cmap.set_under('black')
spectral_cmap.set_over('grey')
pcolor_args = {'cmap': spectral_cmap}
colorbar_args = {'location': 'right', 'pad': '5%'}
plot_GHF_on_map_pcolormesh(m,
                X_gris.Longitude_1.as_matrix(), X_gris.Latitude_1.as_matrix(),
                y_gris,
                parallel_step=5., meridian_step=10.,
                colorbar_args=colorbar_args,
                pcolor_args=pcolor_args)
pcolor_args = {'cmap': spectral_cmap}
plot_GHF_on_map_pcolormesh(m,
                gris_known.Longitude_1.as_matrix(), gris_known.Latitude_1.as_matrix(),
                gris_known.GHF,
                parallel_step=5., meridian_step=10.,
                colorbar_args=colorbar_args,
                pcolor_args=pcolor_args)
save_cur_fig('TEST_pcolormesh.png',
             title='GHF predicted for Greenland (mW m$^{-2}$)')


frames = [X_gris,gris_known]
greenland = pd.concat(frames)

m = Basemap(width=1600000, height=2650000, resolution='l',
            projection='stere', lat_ts=71, lon_0=-41.5, lat_0=72)
spectral_cmap = plt.get_cmap('spectral', 13)
spectral_cmap.set_under('black')
spectral_cmap.set_over('grey')
pcolor_args = {'cmap': spectral_cmap}
colorbar_args = {'location': 'right', 'pad': '5%'}
plot_GHF_on_map_pcolormesh_interp(m,
                greenland.Longitude_1.as_matrix(), greenland.Latitude_1.as_matrix(),
                np.hstack([y_gris,gris_known.GHF.as_matrix()]),
                parallel_step=5., meridian_step=10.,
                colorbar_args=colorbar_args,
                pcolor_args=pcolor_args)
save_cur_fig('TEST_pcolormesh_interpolated.png',
             title='GHF predicted for Greenland (mW m$^{-2}$)')


# Histograms: Greenland (predicted) and global (known)
# ----------------------------------------------------
plot_GHF_histogram(y_gris)
save_cur_fig('hist_greenland.png', title='GHF predicted in Greenland')

#plot_GHF_histogram(y_train)
#save_cur_fig('hist_global.png', title='GHF global measurement')

# Store greenland predictions and known values for ARC GIS
# --------------------------------------------------------
save_gris_prediction_data(X_gris, gris_known, 'lat_lon_ghf.txt')



