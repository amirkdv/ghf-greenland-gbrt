import numpy as np
import pandas as pd
from ghf_prediction import haversine_distance

MAX_ICE_CORE_DIST = 150.

# The only existing data points for Greenland are at the following ice
# cores: data_points[X] contains info for data point at ice core X. 'rad'
# is the radius used for Gaussian estimates from each point.
GREENLAND = pd.DataFrame({
    'lat':  [ 72.58,  72.60,   65.18,  75.10,   77.18,   61.40,   60.98,   60.73,      66.50],
    'lon':  [-37.64, -38.50,  -43.82, -42.32,  -61.13,  -48.18,  -45.98,  -45.75,     -50.33],
    'ghf':  [ 51.3,   60.,     20.,    135.,    50.,     43.,     32.,     51.,        31.05],  # NOTE: GHF at NGRIP is questionable
    'rad':  [ 1000.,  1000.,   1000.,  200.,    1000.,   1000.,   1000.,   1000.,      1000.],  # NOTE: rad at NGRIP is subject to change
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

