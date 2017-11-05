import sys
from random import randint
from math import sqrt, pi
from mpl_toolkits.basemap import Basemap
from circles import equi
from util import (
    plt,
    np,
    load_global_data,
    save_cur_fig,
    split_with_circle,
    train_gbrt,
    train_linear,
    error_summary,
    random_prediction_ctr,
    plot_test_pred_linregress,
    plot_values_on_map,
    plot_values_histogram,
    train_test_split,
    greenland_train_test_sets,
    GREENLAND_RADIUS,
    SPECTRAL_CMAP,
)
COLORBAR_ARGS = {'location': 'bottom', 'pad': '10%'}

plt.rc('font', size=15)

data = load_global_data()

# Supplementary Figure 4
# if the random_prediction_ctr is chosen, occasionally errors
# may occur if that random center does not have the density of
# 50. If so, simply re-run
roi_densities = [50, 0, 20, 10, 5]
#center = random_prediction_ctr(data, GREENLAND_RADIUS)
center = (28.67, 45.5)

plt.clf()

for roi_density in roi_densities:
    print 'center: ', center
    X_train, y_train, X_test, y_test = split_with_circle(data, center,
                                       roi_density=roi_density, radius=GREENLAND_RADIUS)
    reg = train_gbrt(X_train.drop(['lat', 'lon'], axis=1), y_train)
    y_pred = reg.predict(X_test.drop(['lat', 'lon'], axis=1))

    r2, rmse = error_summary(y_test, y_pred)

    m = Basemap(projection='merc',lat_0=center[0], lon_0=center[1],
                resolution = 'l', area_thresh = 1000.0,
                llcrnrlon=0, llcrnrlat=25,
                urcrnrlon=60, urcrnrlat=61)

    m.drawlsmask(land_color = "#ffffff",
                   ocean_color="#e8f4f8",
                   resolution = 'l')

    x,y = m(X_train.lon.as_matrix(), X_train.lat.as_matrix())
    m.scatter(x,y,marker='o', s=5, color='#7a7a7a')

    diff_cmap = plt.get_cmap('PiYG', 20)
    scatter_args = {'marker': 'o', 's': 35, 'lw': 0.25, 'cmap': diff_cmap,'edgecolor': 'k'}
    colorbar_args = {'location': 'bottom', 'pad': '5%'}
    plot_values_on_map(m,
                       X_test.lon.as_matrix(), X_test.lat.as_matrix(),
                       y_test - y_pred,
                       clim=(-10, 10), clim_step=2,
                       parallel_step=10., meridian_step=10.,
                       colorbar_args=COLORBAR_ARGS,
                       scatter_args=scatter_args)

    equi(m, center[0], center[1], GREENLAND_RADIUS,
         lw=2, linestyle='-', color='black', alpha=.5)
    title = r'$GHF - \widehat{GHF}$ on validation set with ' + \
            r'$\rho_{ROI}$ = %d'%roi_density
    save_cur_fig('%d-diff-map.png' % roi_density, title=title)

    plot_test_pred_linregress(y_test, y_pred, label='GBRT', color='b')
    save_cur_fig('%d_linear_correlation.png' % roi_density,
                 title=r'$\rho_{ROI}$ = %i, $r^2=%.2f, RMSE=%.2f$' % (roi_density,r2, rmse))

### Main Text Figure 1
## gbrt regression
X = data.drop(['GHF'],axis=1)
y = data.GHF

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=11, test_size=0.10)

reg_gbrt = train_gbrt(X_train.drop(['lat', 'lon'], axis=1), y_train)
y_pred_gbrt = reg_gbrt.predict(X_test.drop(['lat', 'lon'], axis=1))

plt.clf()
m = Basemap(projection='robin',lon_0=0,resolution='c')
m.drawlsmask(land_color = "#ffffff",
               ocean_color="#e8f4f8",
               resolution = 'l')

diff_cmap = plt.get_cmap('PiYG', 20)
scatter_args = {'marker': 'o', 's': 15, 'lw': 0.25, 'edgecolor':'black','cmap': diff_cmap}
plot_values_on_map(m,
                   X_test.lon.as_matrix(), X_test.lat.as_matrix(),
                   y_test - y_pred_gbrt,
                   clim=(-10, 10), clim_step=2,
                   parallel_step=20., meridian_step=60.,
                   colorbar_args=COLORBAR_ARGS,
                   scatter_args=scatter_args)
save_cur_fig('gbrt_random_difference.png',
     title='$GHF - \widehat{GHF}_{\mathrm{GBRT}}$ on validation set')

plt.clf()
scatter_args = {'marker': 'o', 's': 15, 'lw': 0, 'cmap': SPECTRAL_CMAP}
plot_values_on_map(m,
                   X_test.lon.as_matrix(), X_test.lat.as_matrix(),
                   y_test,
                   clim=(20, 150), clim_step=10,
                   parallel_step=20., meridian_step=60.,
                   colorbar_args=COLORBAR_ARGS,
                   scatter_args=scatter_args)
save_cur_fig('gbrt_random_test.png',
     title='measured GHF at validation set')

plt.clf()
scatter_args = {'marker': 'o', 's': 15, 'lw': 0, 'cmap': SPECTRAL_CMAP}
plot_values_on_map(m,
                   X_train.lon.as_matrix(), X_train.lat.as_matrix(),
                   y_train,
                   clim=(20, 150), clim_step=10,
                   parallel_step=20., meridian_step=60.,
                   colorbar_args=COLORBAR_ARGS,
                   scatter_args=scatter_args)
save_cur_fig('gbrt_random_train.png', title='GHF at training set')

plt.clf()
plot_test_pred_linregress(y_test, y_pred_gbrt, label='GBRT', color='b')
save_cur_fig('gbrt_random_linear_correlation.png')

## linear regression
reg_linear = train_linear(X_train.drop(['lat', 'lon'], axis=1),
                      y_train)
y_pred_linear = reg_linear.predict(X_test.drop(['lat', 'lon'], axis=1))

m = Basemap(projection='robin',lon_0=0,resolution='c')
m.drawlsmask(land_color = "#ffffff",
               ocean_color="#e8f4f8",
               resolution = 'l')

diff_cmap = plt.get_cmap('PiYG', 20)
scatter_args = {'marker': 'o', 's': 15, 'lw': 0.25, 'edgecolor':'black','cmap': diff_cmap}
plot_values_on_map(m,
                   X_test.lon.as_matrix(), X_test.lat.as_matrix(),
                   y_test - y_pred_linear,
                   clim=(-10, 10), clim_step=2,
                   parallel_step=20., meridian_step=60.,
                   colorbar_args=COLORBAR_ARGS,
                   scatter_args=scatter_args)
save_cur_fig('linear_random_difference.png',
     title=r'$GHF - \widehat{GHF}_{\mathrm{linear}}$ on validation set')

plt.clf()
scatter_args = {'marker': 'o', 's': 15, 'lw': 0, 'cmap': SPECTRAL_CMAP}
plot_values_on_map(m,
                   X_test.lon.as_matrix(), X_test.lat.as_matrix(),
                   y_test,
                   clim=(20, 150), clim_step=10,
                   parallel_step=20., meridian_step=60.,
                   colorbar_args=COLORBAR_ARGS,
                scatter_args=scatter_args)
save_cur_fig('linear_random_test.png',
     title='measured GHF at validation set')

plt.clf()
scatter_args = {'marker': 'o', 's': 15, 'lw': 0, 'cmap': SPECTRAL_CMAP}
plot_values_on_map(m,
                   X_train.lon.as_matrix(), X_train.lat.as_matrix(),
                   y_train,
                   clim=(20, 150), clim_step=10,
                   parallel_step=20., meridian_step=60.,
                   colorbar_args=COLORBAR_ARGS,
                   scatter_args=scatter_args)
save_cur_fig('linear_random_train.png',
     title='GHF at training set')

plt.clf()
plot_test_pred_linregress(y_test, y_pred_linear, label='linear predictor', color='r')
save_cur_fig('linear_random_linear_correlation.png')

## global GHF and its histogram
plt.clf()
m = Basemap(projection='robin',lon_0=0,resolution='c')
m.drawlsmask(land_color = "#ffffff",
               ocean_color="#e8f4f8",
               resolution = 'l')

scatter_args = {'marker': 'o', 's': 15, 'lw': 0, 'cmap': SPECTRAL_CMAP}
plot_values_on_map(m,
                   X.lon.as_matrix(), X.lat.as_matrix(),
                   y,
                   clim=(20, 150), clim_step=10,
                   parallel_step=20., meridian_step=60.,
                   colorbar_args=COLORBAR_ARGS,
                   scatter_args=scatter_args)
save_cur_fig('global_ghf.png', title='Global GHF measurements')

plt.clf()
plot_values_histogram(y)
save_cur_fig('global_ghf_histogram.png')

