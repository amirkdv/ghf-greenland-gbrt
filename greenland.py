from mpl_toolkits.basemap import Basemap
from circles import equi
from util import (
    plt,
    np,
    save_cur_fig,
    train_gbrt,
    train_linear,
    plot_GHF_on_map,
    plot_GHF_on_map_pcolormesh,
    plot_GHF_on_map_pcolormesh_interp,
    plot_test_pred_linregress,
    plot_GHF_histogram,
    MAX_ICE_CORE_DIST,
    greenland_train_test_sets,
    GREENLAND,
)

SPECTRAL_CMAP = plt.get_cmap('spectral', 13)
SPECTRAL_CMAP.set_under('black')
SPECTRAL_CMAP.set_over('grey')

GREENLAND_BASEMAP_ARGS = {
    'lon_0': -41.5,
    'lat_0': 72,
    'lat_ts': 71,
    'width': 1600000,
    'height': 2650000,
    'resolution': 'l',
    'projection': 'stere'
}

def _mark_ice_cores(m, lons, lats, ghfs):
    colorbar_args = {'location': 'right', 'pad': '5%'}
    scatter_args = {'marker': 's', 's': 45, 'lw': 1, 'cmap': SPECTRAL_CMAP, 'edgecolor':'white'}
    plot_GHF_on_map(m, lons, lats, ghfs,
                    parallel_step=5., meridian_step=10.,
                    colorbar_args=colorbar_args,
                    scatter_args=scatter_args)

def _mark_ice_core_gaussians(m, cores):
    for _, core in cores.iterrows():
        equi(m, core['lon'], core['lat'], MAX_ICE_CORE_DIST,
             lw=2, linestyle='-', color='black', alpha=.3)


def plot_training_GHF_mark_greenland(train_lons, train_lats, train_ghfs):
    m = Basemap(width=6500000, height=6500000, projection='aeqd', lon_0=-37.64, lat_0=72.58)
    _mark_ice_core_gaussians(m, GREENLAND)

    # plot all known GHF values
    colorbar_args = {'location': 'bottom', 'pad': '10%'}
    scatter_args = {'marker': 'o', 's': 15, 'lw': 0, 'cmap': SPECTRAL_CMAP}
    plot_GHF_on_map(m, train_lons, train_lats, train_ghfs,
                    parallel_step=5., meridian_step=15.,
                    colorbar_args=colorbar_args,
                    scatter_args=scatter_args)


def plot_greenland_gaussian_prescribed_GHF(lons, lats, ghfs):
    args = GREENLAND_BASEMAP_ARGS.copy()
    args['height'] = 2800000
    args['lat_0'] = 71.5

    m = Basemap(**args)
    _mark_ice_cores(m, GREENLAND.lon.as_matrix(), GREENLAND.lat.as_matrix(),
                    GREENLAND.ghf.as_matrix())
    _mark_ice_core_gaussians(m, GREENLAND)

    # plot all known GHFs, only prescribed Greenland values being visible in
    # the frame defined by GREENLAND_BASEMAP_ARGS.
    colorbar_args = {'location': 'right', 'pad': '5%'}
    scatter_args = {'marker': 'o', 's': 18, 'lw': 0, 'cmap': SPECTRAL_CMAP}
    plot_GHF_on_map(m, lons, lats, ghfs, parallel_step=5., meridian_step=10.,
                    colorbar_args=colorbar_args, scatter_args=scatter_args)


def plot_greenland_prediction_points(lons, lats, ghfs):
    m = Basemap(**GREENLAND_BASEMAP_ARGS)
    seismic_cmap = plt.get_cmap('seismic', 20)
    scatter_args = {'marker': 'o', 's': 20, 'lw': 0, 'cmap': SPECTRAL_CMAP}
    colorbar_args = {'location': 'right', 'pad': '5%'}
    plot_GHF_on_map(m, lons, lats, ghfs, parallel_step=5., meridian_step=10.,
                    colorbar_args=colorbar_args, scatter_args=scatter_args)


def plot_greenland_prediction(lons, lats, ghfs):
    m = Basemap(**GREENLAND_BASEMAP_ARGS)
    pcolor_args = {'cmap': SPECTRAL_CMAP}
    colorbar_args = {'location': 'right', 'pad': '5%'}
    plot_GHF_on_map_pcolormesh(m, lons, lats, ghfs,
                               parallel_step=5., meridian_step=10.,
                               colorbar_args=colorbar_args,
                               pcolor_args=pcolor_args)


def plot_greenland_prediction_interpolated(lons, lats, ghfs):
    m = Basemap(**GREENLAND_BASEMAP_ARGS)
    _mark_ice_cores(m, GREENLAND.lon.as_matrix(), GREENLAND.lat.as_matrix(),
                    GREENLAND.ghf.as_matrix())
    pcolor_args = {'cmap': SPECTRAL_CMAP}
    colorbar_args = {'location': 'right', 'pad': '5%'}
    plot_GHF_on_map_pcolormesh_interp(m, lons, lats, ghfs,
                    parallel_step=5., meridian_step=10.,
                    colorbar_args=colorbar_args, pcolor_args=pcolor_args)


    m.drawparallels(np.arange(-80., 81., 5.), labels=[1, 0, 0, 0], fontsize=10, color='#c6c6c6')
    m.drawmeridians(np.arange(-180., 181., 10.), labels=[0, 0, 0, 1], fontsize=10, color='#c6c6c6')

    m.drawcoastlines(color='grey', linewidth=0.5)
    m.drawmapboundary(color='grey')


if __name__ == '__main__':
    X_train, y_train, X_test = greenland_train_test_sets()

    train_lons = X_train.Longitude_1.as_matrix()
    train_lats = X_train.Latitude_1.as_matrix()
    X_train = X_train.drop(['Latitude_1', 'Longitude_1'], axis=1)

    test_lons = X_test.Longitude_1.as_matrix()
    test_lats = X_test.Latitude_1.as_matrix()
    X_test = X_test.drop(['Latitude_1', 'Longitude_1'], axis=1)

    # -------------------- Plot training data  -------------------------
    plot_training_GHF_mark_greenland(train_lons, train_lats, y_train)
    save_cur_fig('greenland_training_GHF.png', title='GHF at training set')

    plot_greenland_gaussian_prescribed_GHF(train_lons, train_lats, y_train)
    save_cur_fig('greenland_prescribed_GHF.png',
                 title='Points with prescribed GHF \n around GHF measurements (mW m$^{-2}$)')

    # -------------------- Plot predicted results ----------------------
    reg = train_gbrt(X_train, y_train, logfile='GHF_1deg_averaged_logfile.txt')
    y_pred = reg.predict(X_test)

    plot_greenland_prediction_points(test_lons, test_lats, y_pred)
    save_cur_fig('greenland_prediction_points.png',
                 title='GHF predicted for Greenland (mW m$^{-2}$)')

    plot_greenland_prediction(test_lons, test_lats, y_pred)
    save_cur_fig('greenland_prediction.png',
                 title='GHF predicted for Greenland (mW m$^{-2}$)')

    lons = np.hstack([train_lons, test_lons])
    lats = np.hstack([train_lats, test_lats])
    ghfs = np.hstack([y_train, y_pred]),
    plot_greenland_prediction_interpolated(lons, lats, ghfs)
    save_cur_fig('greenland_prediction_interpolated.png',
                 title='GHF predicted for Greenland (mW m$^{-2}$)')


    # --------------------- Plot GHF histograms --------------------------
    plot_GHF_histogram(y_pred)
    save_cur_fig('hist_greenland.png', title='GHF predicted in Greenland')

    plot_GHF_histogram(y_train)
    save_cur_fig('hist_global.png', title='GHF global measurement')
