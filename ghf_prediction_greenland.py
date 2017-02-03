from mpl_toolkits.basemap import Basemap
from circles import equi
from ghf_prediction import (
    plt, pd, np,
    load_global_gris_data, save_cur_fig, save_np_object,
    haversine_distance,
    split, train_regressor,
    plot_GHF_on_map,
    plot_GHF_on_map_pcolormesh,
    plot_GHF_on_map_pcolormesh_interp,
    plot_test_pred_linregress,
    plot_GHF_histogram,
)

MAX_ICE_CORE_DIST = 150.

# The only existing data points for Greenland are at the following ice
# cores: data_points[X] contains info for data point at ice core X. 'rad'
# is the radius used for Gaussian estimates from each point.
GREENLAND = pd.DataFrame({
    'lat':  [ 72.58,  72.60,   65.18,  75.10,   77.18,   61.40,   60.98,   60.73,      66.50],
    'lon':  [-37.64, -38.50,  -43.82, -42.32,  -61.13,  -48.18,  -45.98,  -45.75,     -50.33],
    'ghf':  [ 51.3,   60.,     20.,    135.,    50.,     43.,     32.,     51.,        31.05],  # NOTE: GHF at NGRIP is questionable
    'rad':  [ 1000.,  1000.,   1000.,  160.,    1000.,   1000.,   1000.,   1000.,      1000.],
    'core': ['GRIP', 'GISP2', 'DYE3', 'NGRIP', 'CC',    'SASS1', 'SASS2', 'LANGSETH', 'GAP'],
})
GREENLAND.set_index('core')

# Approximates GHF values at rows with unknown GHF according to a Gaussian
# decay formula based on known GHF values in GREENLAND.
def fill_in_greenland_GHF(data):
    def gauss(amp, dist, rad):
        return amp * np.exp(- dist ** 2. / rad ** 2)

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
        data.loc[data[dist_col] > MAX_ICE_CORE_DIST, ghf_col] = np.nan

    data['GHF'] = data[ghf_cols + ['GHF']].mean(skipna=True, axis=1)
    data = data.drop(dist_cols + ghf_cols, axis=1)

    # FIXME artificially forced to 135.0 in source
    data.loc[data.GHF == 135.0, 'GHF'] = 0
    # The gris data set has many rows with feature values but no GHF
    # measurements. We want to predict GHF for these.
    gris_unknown = data[data.GHF.isnull()]
    data.loc[data.GHF == 0, 'GHF'] = np.nan
    data.dropna(inplace=True)
    return data, gris_unknown

# saves GRIS known and predicted GHF and latitude/longitude and values as a
# numpy array in OUT_DIR/filename.
def save_gris_prediction_data(gris_unknown, gris_known, y_gris, filename):
    lats = np.hstack([gris_unknown.Longitude_1.as_matrix(),
                      gris_known.Longitude_1.as_matrix()])
    lons = np.hstack([gris_unknown.Latitude_1.as_matrix(),
                      gris_known.Latitude_1.as_matrix()])
    ghfs = np.hstack([y_gris, gris_known.GHF.as_matrix()])

    final = np.zeros([len(lats), 3])
    final[:, 0] = lats
    final[:, 1] = lons
    final[:, 2] = ghfs

    save_np_object(filename, 'gris data', final, delimiter=', ',
                   header='lon, lat, ghf', fmt='%10.5f')


data = load_global_gris_data()

# Prepare training and test sets for Greenland
# Note that there is no longer a y_test for
# Greenland predicitons. It is only X_test
# --------------------------------------------
gris_known, gris_unknown = fill_in_greenland_GHF(data)
center = GREENLAND.loc[GREENLAND['core'] == 'GRIP']
center = (float(center.lon), float(center.lat))
#X_train, y_train, X_test, y_test = split(gris_known, center)
X_train = gris_known.drop(['GHF'], axis=1)
y_train = gris_known.GHF

X_test = gris_unknown.drop(['GHF'], axis=1)

# Plot known GHF values for training and test sets
# ------------------------------------------------
#m = Basemap(projection='robin',lon_0=0,resolution='c')
m = Basemap(projection='aeqd',
      lon_0 = -37.64,
      lat_0 = 72.58,
      width = 7500000,
      height = 7500000)

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

for core in GREENLAND.core:
    centerlon = GREENLAND[GREENLAND['core'] == core].lon.as_matrix()
    centerlat = GREENLAND[GREENLAND['core'] == core].lat.as_matrix()
    equi(m, centerlon, centerlat, MAX_ICE_CORE_DIST,
         lw=2, linestyle='-', color='brown', alpha=.8)

save_cur_fig('GHF_1deg_averaged_map_train.png', title='GHF at train set')

#plot_GHF_on_map(m,
#                X_test.Longitude_1.as_matrix(), X_test.Latitude_1.as_matrix(),
#                y_test,
#                colorbar_args=colorbar_args,
#                scatter_args=scatter_args)
#save_cur_fig('GHF_1deg_averaged_map_test.png', title='GHF at test set')

# Predict GHF over test set
# -----------------------------
reg = train_regressor(X_train.drop(['Latitude_1', 'Longitude_1'], axis=1),
                      y_train, logfile='GHF_1deg_averaged_logfile.txt')
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
#m = Basemap(projection='robin',lon_0=0,resolution='c')
#seismic_cmap = plt.get_cmap('seismic', 20)
#scatter_args = {'marker': 'o', 's': 15, 'lw': 0, 'cmap': seismic_cmap}
#colorbar_args = {'location': 'bottom', 'pad': '10%'}

#plot_GHF_on_map(m,
#                X_test.Longitude_1.as_matrix(), X_test.Latitude_1.as_matrix(),
#                y_test - y_pred,
#                clim=(-10, 10), clim_step=2,
#                colorbar_args=colorbar_args,
#                scatter_args=scatter_args)
#save_cur_fig('GHF_1deg_diff_map.png',
#             title='GHF error on test set (true - predicted)')

#m = Basemap(width=1600000, height=2650000, resolution='l',
#            projection='stere', lat_ts=71, lon_0=-41.5, lat_0=72)
#seismic_cmap = plt.get_cmap('seismic', 20)
#scatter_args = {'marker': 'o', 's': 15, 'lw': 0, 'cmap': seismic_cmap}
#colorbar_args = {'location': 'right', 'pad': '5%'}

#plot_GHF_on_map(m,
#                X_test.Longitude_1.as_matrix(), X_test.Latitude_1.as_matrix(),
#                y_test - y_pred,
#                clim=(-10, 10), clim_step=2,
#                parallel_step=5., meridian_step=10.,
#                colorbar_args=colorbar_args,
#                scatter_args=scatter_args)
#save_cur_fig('GHF_1deg_diff_map_Greenland.png',
#             title='GHF error on test set (true - predicted)')

## Linear Regression between known and predicted values in test set
## ----------------------------------------------------------------
#plot_test_pred_linregress(y_test, y_pred, 'GHF_1deg_averaged_plot.png',
#                          title='Linear regression between predicted vs true GHF')

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

m = Basemap(width=1600000, height=2800000, resolution='l',
            projection='stere', lat_ts=71, lon_0=-41.5, lat_0=71.50)
colorbar_args = {'location': 'right', 'pad': '5%'}
#scatter_args = {'marker': 'o', 's': 20, 'lw': 0, 'cmap': spectral_cmap}
#plot_GHF_on_map(m,
#                X_gris.Longitude_1.as_matrix(), X_gris.Latitude_1.as_matrix(),
#                y_gris,
#                parallel_step=5., meridian_step=10.,
#                colorbar_args=colorbar_args,
#                scatter_args=scatter_args)
scatter_args = {'marker': 'o', 's': 20, 'lw': 0, 'cmap': spectral_cmap}
plot_GHF_on_map(m,
                gris_known.Longitude_1.as_matrix(), gris_known.Latitude_1.as_matrix(),
                gris_known.GHF,
                parallel_step=5., meridian_step=10.,
                colorbar_args=colorbar_args,
                scatter_args=scatter_args)

for core in GREENLAND.core:
    centerlon = GREENLAND[GREENLAND['core'] == core].lon.as_matrix()
    centerlat = GREENLAND[GREENLAND['core'] == core].lat.as_matrix()
    equi(m, centerlon, centerlat, MAX_ICE_CORE_DIST,
         lw=2, linestyle='-', color='brown', alpha=.8)

scatter_args = {'marker': 's', 's': 35, 'lw': 1, 'cmap': spectral_cmap, 'edgecolor':'black'}
plot_GHF_on_map(m,
                GREENLAND.lon.as_matrix(), GREENLAND.lat.as_matrix(),
                GREENLAND.ghf.as_matrix(),
                parallel_step=5., meridian_step=10.,
                colorbar_args=colorbar_args,
                scatter_args=scatter_args)
save_cur_fig('greenland_prescribed_GHF.png',
             title='Points with prescribed GHF \n around GHF measurements (mW m$^{-2}$)')

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
                greenland.Longitude_1.as_matrix(),
                greenland.Latitude_1.as_matrix(),
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

plot_GHF_histogram(y_train)
save_cur_fig('hist_global.png', title='GHF global measurement')

# Store greenland predictions and known values for ARC GIS
# --------------------------------------------------------
save_gris_prediction_data(X_gris, gris_known, y_gris, 'lat_lon_ghf.txt')
