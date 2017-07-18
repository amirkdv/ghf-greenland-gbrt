import sys
from random import randint
from math import sqrt, pi
from ghf_prediction import (
    plot_GHF_on_map,
    plot_GHF_on_map_pcolormesh,
    plot_GHF_on_map_pcolormesh_interp,
    plt, np, mean_squared_error,
    load_global_gris_data, save_cur_fig, pickle_dump, pickle_load,
    split_with_circle, split_by_distance, tune_params,
    train_regressor, error_summary, random_prediction_ctr,
    CATEGORICAL_FEATURES, GREENLAND_RADIUS,
    plot_test_pred_linregress, train_test_split
)
from ghf_greenland  import greenland_train_test_sets
from error_analysis import _eval_prediction
from mpl_toolkits.basemap import Basemap
from circles import equi

data = load_global_gris_data()
# FIXME artificially forced to 135.0 in source
data.loc[data.GHF == 135.0, 'GHF'] = 0
data.loc[data.GHF == 0, 'GHF'] = np.nan
data.dropna(inplace=True)

# Supplementary Figure 4
# if the random_prediction_ctr is chosen, occasionally errors
# may occur if that random center does not have the density of
# 50. If so, simply re-run
roi_densities = [50, 0, 20, 10, 5]
#center = random_prediction_ctr(data, GREENLAND_RADIUS)
center = (28.67, 45.5)

for roi_density in roi_densities:
    X_train, y_train, X_test, y_test = split_with_circle(data, center, 
                                       roi_density=roi_density, radius=GREENLAND_RADIUS)
    reg = train_regressor(X_train.drop(['Latitude_1', 'Longitude_1'], axis=1),
                          y_train, logfile='%i_rho_logfile.txt'%roi_density)
    y_pred = reg.predict(X_test.drop(['Latitude_1', 'Longitude_1'], axis=1))

    m = Basemap(projection='merc',lat_0=center[0], lon_0=center[1],
                resolution = 'l', area_thresh = 1000.0,
                llcrnrlon=0, llcrnrlat=25,
                urcrnrlon=60, urcrnrlat=61)

    seismic_cmap = plt.get_cmap('seismic', 20)
    scatter_args = {'marker': 'o', 's': 25, 'lw': 0.25, 'cmap': seismic_cmap,'edgecolor': 'k'}
    colorbar_args = {'location': 'bottom', 'pad': '10%'}
    plot_GHF_on_map(m,
                    X_test.Longitude_1.as_matrix(), X_test.Latitude_1.as_matrix(),
                    y_test - y_pred,
                    clim=(-10, 10), clim_step=2,
                    parallel_step=10., meridian_step=10.,
                    colorbar_args=colorbar_args,
                    scatter_args=scatter_args)
    equi(m, center[0], center[1], GREENLAND_RADIUS,
         lw=2, linestyle='-', color='black', alpha=.5)
    save_cur_fig('%i-diff-map.png'%roi_density,
         title='GHF error on test set (true - predicted) with '+r'$\rho_{ROI}$ = %i'%roi_density)
        
    plot_test_pred_linregress(y_test, y_pred, '%i_linear_correlation.png'%roi_density,
                              title=r'$\rho_{ROI}$ = %i'%roi_density)

    print center


# Main Text Figure 2
X = data.drop(['GHF'],axis=1)
y = data.GHF
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=None, test_size=0.15)

reg = train_regressor(X_train.drop(['Latitude_1', 'Longitude_1'], axis=1),
                      y_train, logfile='random_logfile.txt')
y_pred = reg.predict(X_test.drop(['Latitude_1', 'Longitude_1'], axis=1))

m = Basemap(projection='robin',lon_0=0,resolution='c')
seismic_cmap = plt.get_cmap('seismic', 20)
scatter_args = {'marker': 'o', 's': 15, 'lw': 0, 'cmap': seismic_cmap}
colorbar_args = {'location': 'bottom', 'pad': '10%'}
plot_GHF_on_map(m,
                X_test.Longitude_1.as_matrix(), X_test.Latitude_1.as_matrix(),
                y_test - y_pred,
                clim=(-10, 10), clim_step=2,
                parallel_step=20., meridian_step=60.,
                colorbar_args=colorbar_args,
                scatter_args=scatter_args)
save_cur_fig('random_difference.png',
     title='GHF error on test set (true - predicted)')
    
spectral_cmap = plt.get_cmap('spectral', 13)
spectral_cmap.set_under('black')
spectral_cmap.set_over('grey')
colorbar_args = {'location': 'bottom', 'pad': '10%'}
scatter_args = {'marker': 'o', 's': 15, 'lw': 0, 'cmap': spectral_cmap}
plot_GHF_on_map(m,
                X_test.Longitude_1.as_matrix(), X_test.Latitude_1.as_matrix(),
                y_test,
                clim=(20, 150), clim_step=10,
                parallel_step=20., meridian_step=60.,
                colorbar_args=colorbar_args,
                scatter_args=scatter_args)
save_cur_fig('random_test.png',
     title='measured GHF at test set')

spectral_cmap = plt.get_cmap('spectral', 13)
spectral_cmap.set_under('black')
spectral_cmap.set_over('grey')
colorbar_args = {'location': 'bottom', 'pad': '10%'}
scatter_args = {'marker': 'o', 's': 15, 'lw': 0, 'cmap': spectral_cmap}
plot_GHF_on_map(m,
                X_train.Longitude_1.as_matrix(), X_train.Latitude_1.as_matrix(),
                y_train,
                clim=(20, 150), clim_step=10,
                parallel_step=20., meridian_step=60.,
                colorbar_args=colorbar_args,
                scatter_args=scatter_args)
save_cur_fig('random_train.png',
     title='GHF at training set')

plot_test_pred_linregress(y_test, y_pred, 'random_linear_correlation.png',
                          title='   ')

