from mpl_toolkits.basemap import Basemap
from circles import equi
from ghf_prediction import (
    plt, pd, np,
    load_global_gris_data, save_cur_fig,
    train_gbrt, train_linear,
    plot_GHF_on_map,
    plot_GHF_on_map_pcolormesh,
    plot_GHF_on_map_pcolormesh_interp,
    plot_test_pred_linregress,
    plot_GHF_histogram,
)
from ghf_greenland import (
    GREENLAND,
    fill_in_greenland_GHF,
    MAX_ICE_CORE_DIST,
)

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

#    save_np_object(filename, 'gris data', final, delimiter=', ',
#                   header='lon, lat, ghf', fmt='%10.5f')


data = load_global_gris_data()

# Prepare training and test sets for Greenland
# Note that there is no longer a y_test for
# Greenland predicitons. It is only X_test
# --------------------------------------------
gris_known, gris_unknown = fill_in_greenland_GHF(data)
#center = GREENLAND.loc[GREENLAND['core'] == 'GRIP']
#center = (float(center.lon), float(center.lat))
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
      width = 6500000,
      height = 6500000)

spectral_cmap = plt.get_cmap('spectral', 13)
spectral_cmap.set_under('black')
spectral_cmap.set_over('grey')
colorbar_args = {'location': 'bottom', 'pad': '10%'}
scatter_args = {'marker': 'o', 's': 15, 'lw': 0, 'cmap': spectral_cmap}

plot_GHF_on_map(m,
                X_train.Longitude_1.as_matrix(), X_train.Latitude_1.as_matrix(),
                y_train,
                parallel_step=5., meridian_step=15.,
                colorbar_args=colorbar_args,
                scatter_args=scatter_args)

for core in GREENLAND.core:
    centerlon = GREENLAND[GREENLAND['core'] == core].lon.as_matrix()
    centerlat = GREENLAND[GREENLAND['core'] == core].lat.as_matrix()
    equi(m, centerlon, centerlat, MAX_ICE_CORE_DIST,
         lw=1, linestyle='-', color='black', alpha=.3)

save_cur_fig('greenland_training_w_gris.png', title='GHF at training set')

#plot_GHF_on_map(m,
#                X_test.Longitude_1.as_matrix(), X_test.Latitude_1.as_matrix(),
#                y_test,
#                colorbar_args=colorbar_args,
#                scatter_args=scatter_args)
#save_cur_fig('GHF_1deg_averaged_map_test.png', title='GHF at test set')

# Predict GHF over test set
# -----------------------------
reg = train_gbrt(X_train.drop(['Latitude_1', 'Longitude_1'], axis=1),
                      y_train, logfile='GHF_1deg_averaged_logfile.txt')
y_pred = reg.predict(X_test.drop(['Latitude_1', 'Longitude_1'], axis=1))

#m = Basemap(width=1600000, height=2650000, resolution='l',
#            projection='stere', lat_ts=71, lon_0=-41.5, lat_0=72)
#colorbar_args = {'location': 'right', 'pad': '5%', 'extend': 'both'}
#scatter_args = {'marker': 'o', 's': 25, 'lw': 0, 'cmap': spectral_cmap}

#plot_GHF_on_map(m,
#                X_test.Longitude_1.as_matrix(), X_test.Latitude_1.as_matrix(),
#                y_pred,
#                parallel_step=5., meridian_step=10.,
#                colorbar_args=colorbar_args,
#                scatter_args=scatter_args)
#save_cur_fig('Greenland_GHF_predicted_1deg.png',
#             title='GHF predicted for Greenland (mW m$^{-2}$) \n globally trained')

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
save_cur_fig('greenland_predicted_points.png',
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
scatter_args = {'marker': 'o', 's': 18, 'lw': 0, 'cmap': spectral_cmap}
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
         lw=2, linestyle='-', color='black', alpha=.3)

scatter_args = {'marker': 's', 's': 45, 'lw': 1, 'cmap': spectral_cmap, 'edgecolor':'white'}
plot_GHF_on_map(m,
                GREENLAND.lon.as_matrix(), GREENLAND.lat.as_matrix(),
                GREENLAND.ghf.as_matrix(),
                parallel_step=5., meridian_step=10.,
                colorbar_args=colorbar_args,
                scatter_args=scatter_args)
save_cur_fig('greenland_prescribed_GHF.png',
             title='Points with prescribed GHF \n around GHF measurements (mW m$^{-2}$)')

#m = Basemap(projection='robin',lon_0=0,resolution='c')
#spectral_cmap = plt.get_cmap('spectral', 13)
#spectral_cmap.set_under('black')
#spectral_cmap.set_over('grey')
#colorbar_args = {'location': 'bottom', 'pad': '10%'}
#pcolor_args = {'cmap': spectral_cmap}
#plot_GHF_on_map_pcolormesh(m,
#                X_train.Longitude_1.as_matrix(), X_train.Latitude_1.as_matrix(),
#                y_train,
#                colorbar_args=colorbar_args,
#                pcolor_args=pcolor_args)
#save_cur_fig('pcolormesh_global.png', title='GHF at training set')

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

scatter_args = {'marker': 's', 's': 45, 'lw': 1, 'cmap': spectral_cmap, 'edgecolor':'white'}
plot_GHF_on_map(m,
                GREENLAND.lon.as_matrix(), GREENLAND.lat.as_matrix(),
                GREENLAND.ghf.as_matrix(),
                parallel_step=5., meridian_step=10.,
                colorbar_args=colorbar_args,
                scatter_args=scatter_args)
save_cur_fig('greenland_predicted.png',
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

scatter_args = {'marker': 's', 's': 45, 'lw': 1, 'cmap': spectral_cmap, 'edgecolor':'white'}
plot_GHF_on_map(m,
                GREENLAND.lon.as_matrix(), GREENLAND.lat.as_matrix(),
                GREENLAND.ghf.as_matrix(),
                parallel_step=5., meridian_step=10.,
                colorbar_args=colorbar_args,
                scatter_args=scatter_args)

save_cur_fig('greenland_predicted_interpolated.png',
             title='GHF predicted for Greenland (mW m$^{-2}$)')


# Histograms: Greenland (predicted) and global (known)
# FIXME: plot the histogram for all Greenland not just
# predicted values
# ----------------------------------------------------
plot_GHF_histogram(y_gris)
save_cur_fig('hist_greenland.png', title='GHF predicted in Greenland')

plot_GHF_histogram(y_train)
save_cur_fig('hist_global.png', title='GHF global measurement')

# Store greenland predictions and known values for ARC GIS
# FIXME: out of use for now - save_np_object removed 
# --------------------------------------------------------
#save_gris_prediction_data(X_gris, gris_known, y_gris, 'lat_lon_ghf.txt')
