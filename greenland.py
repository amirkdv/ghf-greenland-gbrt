from mpl_toolkits.basemap import Basemap
from circles import equi
from util import (
    plt, pd, np,
    load_global_gris_data, save_cur_fig,
    train_gbrt, train_linear,
    plot_GHF_on_map,
    plot_GHF_on_map_pcolormesh,
    plot_GHF_on_map_pcolormesh_interp,
    plot_test_pred_linregress,
    plot_GHF_histogram,
    fill_in_greenland_GHF,
    MAX_ICE_CORE_DIST,
    greenland_train_test_sets,
    GREENLAND,
)

SPECTRAL_CMAP = plt.get_cmap('spectral', 13)
SPECTRAL_CMAP.set_under('black')
SPECTRAL_CMAP.set_over('grey')


def plot_known_GHF(X_train, y_train, X_test):
    m = Basemap(projection='aeqd',
          lon_0 = -37.64,
          lat_0 = 72.58,
          width = 6500000,
          height = 6500000)

    colorbar_args = {'location': 'bottom', 'pad': '10%'}
    scatter_args = {'marker': 'o', 's': 15, 'lw': 0, 'cmap': SPECTRAL_CMAP}

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


def plot_prescribed_GHF(X_train, X_test, y_train):
    m = Basemap(width=1600000, height=2800000, resolution='l',
                projection='stere', lat_ts=71, lon_0=-41.5, lat_0=71.50)
    colorbar_args = {'location': 'right', 'pad': '5%'}
    scatter_args = {'marker': 'o', 's': 18, 'lw': 0, 'cmap': SPECTRAL_CMAP}
    plot_GHF_on_map(m,
                    X_train.Longitude_1.as_matrix(), X_train.Latitude_1.as_matrix(),
                    y_train,
                    parallel_step=5., meridian_step=10.,
                    colorbar_args=colorbar_args,
                    scatter_args=scatter_args)

    for core in GREENLAND.core:
        centerlon = GREENLAND[GREENLAND['core'] == core].lon.as_matrix()
        centerlat = GREENLAND[GREENLAND['core'] == core].lat.as_matrix()
        equi(m, centerlon, centerlat, MAX_ICE_CORE_DIST,
             lw=2, linestyle='-', color='black', alpha=.3)

    scatter_args = {'marker': 's', 's': 45, 'lw': 1, 'cmap': SPECTRAL_CMAP, 'edgecolor':'white'}
    plot_GHF_on_map(m,
                    GREENLAND.lon.as_matrix(), GREENLAND.lat.as_matrix(),
                    GREENLAND.ghf.as_matrix(),
                    parallel_step=5., meridian_step=10.,
                    colorbar_args=colorbar_args,
                    scatter_args=scatter_args)


def plot_predicted_points(X_train, X_test, y_train, y_pred):
    m = Basemap(width=1600000, height=2650000, resolution='l',
                projection='stere', lat_ts=71, lon_0=-41.5, lat_0=72)
    seismic_cmap = plt.get_cmap('seismic', 20)
    scatter_args = {'marker': 'o', 's': 20, 'lw': 0, 'cmap': SPECTRAL_CMAP}
    colorbar_args = {'location': 'right', 'pad': '5%'}
    plot_GHF_on_map(m,
                    X_test.Longitude_1.as_matrix(), X_test.Latitude_1.as_matrix(),
                    y_pred,
                    parallel_step=5., meridian_step=10.,
                    colorbar_args=colorbar_args,
                    scatter_args=scatter_args)


def plot_predictions(X_train, X_test, y_train, y_pred):
    m = Basemap(width=1600000, height=2650000, resolution='l',
                projection='stere', lat_ts=71, lon_0=-41.5, lat_0=72)
    pcolor_args = {'cmap': SPECTRAL_CMAP}
    colorbar_args = {'location': 'right', 'pad': '5%'}
    plot_GHF_on_map_pcolormesh(m,
                    X_test.Longitude_1.as_matrix(), X_test.Latitude_1.as_matrix(),
                    y_pred,
                    parallel_step=5., meridian_step=10.,
                    colorbar_args=colorbar_args,
                    pcolor_args=pcolor_args)
    pcolor_args = {'cmap': SPECTRAL_CMAP}
    plot_GHF_on_map_pcolormesh(m,
                    X_train.Longitude_1.as_matrix(), X_train.Latitude_1.as_matrix(),
                    y_train,
                    parallel_step=5., meridian_step=10.,
                    colorbar_args=colorbar_args,
                    pcolor_args=pcolor_args)

    scatter_args = {'marker': 's', 's': 45, 'lw': 1, 'cmap': SPECTRAL_CMAP, 'edgecolor':'white'}
    plot_GHF_on_map(m,
                    GREENLAND.lon.as_matrix(), GREENLAND.lat.as_matrix(),
                    GREENLAND.ghf.as_matrix(),
                    parallel_step=5., meridian_step=10.,
                    colorbar_args=colorbar_args,
                    scatter_args=scatter_args)


def plot_interpolated_predictions(X_train, X_test, y_train, y_pred):
    greenland = pd.concat([X_test, X_train])

    m = Basemap(width=1600000, height=2650000, resolution='l',
                projection='stere', lat_ts=71, lon_0=-41.5, lat_0=72)
    pcolor_args = {'cmap': SPECTRAL_CMAP}
    colorbar_args = {'location': 'right', 'pad': '5%'}
    plot_GHF_on_map_pcolormesh_interp(m,
                    greenland.Longitude_1.as_matrix(),
                    greenland.Latitude_1.as_matrix(),
                    np.hstack([y_pred, y_train.as_matrix()]),
                    parallel_step=5., meridian_step=10.,
                    colorbar_args=colorbar_args,
                    pcolor_args=pcolor_args)

    scatter_args = {'marker': 's', 's': 45, 'lw': 1, 'cmap': SPECTRAL_CMAP, 'edgecolor':'white'}
    plot_GHF_on_map(m,
                    GREENLAND.lon.as_matrix(), GREENLAND.lat.as_matrix(),
                    GREENLAND.ghf.as_matrix(),
                    parallel_step=5., meridian_step=10.,
                    colorbar_args=colorbar_args,
                    scatter_args=scatter_args)
    m.drawparallels(np.arange(-80., 81., 5.), labels=[1, 0, 0, 0], fontsize=10, color='#c6c6c6')
    m.drawmeridians(np.arange(-180., 181., 10.), labels=[0, 0, 0, 1], fontsize=10, color='#c6c6c6')

    m.drawcoastlines(color='grey', linewidth=0.5)
    m.drawmapboundary(color='grey')


if __name__ == '__main__':
    X_TRAIN, Y_TRAIN, X_TEST = greenland_train_test_sets()
    reg = train_gbrt(X_TRAIN.drop(['Latitude_1', 'Longitude_1'], axis=1),
                          Y_TRAIN, logfile='GHF_1deg_averaged_logfile.txt')
    Y_PRED = reg.predict(X_TEST.drop(['Latitude_1', 'Longitude_1'], axis=1))

    # -------------------- Plot training data  -------------------------
    plot_known_GHF(X_TRAIN, Y_TRAIN, X_TEST)
    save_cur_fig('greenland_training_w_gris.png', title='GHF at training set')

    plot_prescribed_GHF(X_TRAIN, X_TEST, Y_TRAIN)
    save_cur_fig('greenland_prescribed_GHF.png',
                 title='Points with prescribed GHF \n around GHF measurements (mW m$^{-2}$)')

    # -------------------- Plot predicted results ----------------------
    plot_predicted_points(X_TRAIN, X_TEST, Y_TRAIN, Y_PRED)
    save_cur_fig('greenland_predicted_points.png',
                 title='GHF predicted for Greenland (mW m$^{-2}$)')

    plot_predictions(X_TRAIN, X_TEST, Y_TRAIN, Y_PRED)
    save_cur_fig('greenland_predicted.png',
                 title='GHF predicted for Greenland (mW m$^{-2}$)')

    plot_interpolated_predictions(X_TRAIN, X_TEST, Y_TRAIN, Y_PRED)
    save_cur_fig('greenland_predicted_interpolated.png',
                 title='GHF predicted for Greenland (mW m$^{-2}$)')


    # --------------------- Plot GHF histograms --------------------------
    # FIXME plot the histogram for all Greenland not just predicted values
    plot_GHF_histogram(Y_PRED)
    save_cur_fig('hist_greenland.png', title='GHF predicted in Greenland')

    plot_GHF_histogram(Y_TRAIN)
    save_cur_fig('hist_global.png', title='GHF global measurement')
