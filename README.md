This repo contains the data and code to reproduce the results published in:

_Rezvanbehbahani S., L.A. Stearns, A. Kadivar, J.D. Walker, and C.J. vanderVeen (2017), Predicting the Geothermal Heat Flux in Greenland: A Machine Learning Approach, Geophys. Res. Lett., 44, doi:[10.1002/2017GL075661](http://dx.doi.org/10.1002/2017GL075661)._

The data for GHF map of Greenland (Figure 4 of the article) are provided in [`greenland_predictions`](greenland_predictions) folder. 

To cite either of the [data sets](#data-sets) or the code, use the above citation.

Requirements
------------
The following are requirements to reproduce the results:

* _System wide_: Python 2.7 with developer tools, Tk with developer tools,
  build toolchain.
* _Python packages_: see [`requirements.txt`](requirements.txt)
* _Basemap_ (tested with veresions `1.0.7` and `1.1.0`), built from source. For
  automated procedures for Linux see [`Makefile`](Makefile) which automatically
  detects the virtual environment from the `VIRTUAL_ENV` environment variable.

On Debian/Ubuntu the following installs all requirements:
```
$ apt-get install python-dev build-essential tk tk-dev python-pip virtualenv
$ ... # git clone this repo; cd to repo
$ virtualenv env
$ . env/bin/activate
(env) $ pip install -r requirements.txt
(env) $ make basemap-install
```

Usage
-----
In later versions of basemap, if `geos` is installed in a virtual environment
its lib directory must be added to the `LD_LIBRARY_PATH` environment variable.
To produce all figures in the paper:
```
(env) $ export LD_LIBRARY_PATH=env/lib:$LD_LIBRARY_PATH
(env) $ python density_plots.py   # Figures 1 and 2
(env) $ python greenland.py       # Figures 4, S5, S6
(env) $ python error_analysis.py  # Figures 3, 5, S2, S3, S4
```

Features
--------
Each [data set](#data-sets) below contains the following continuous features:

* `age`,
* `bougeur_gravity_anomaly`,
* `depth_to_moho`           (depth to Mohorovičić discontinuity),
* `d_2_hotspot`             (distance to hotspots),
* `d_2_ridge`               (distance to ridge),
* `d_2_trans_ridge`         (distance to transform ridge),
* `d_2_trench`              (distance to trench),
* `d_2_volcano`             (distance to volcano),
* `d_2_young_rift`          (distance to young rift),
* `heat_prod_provinces`     (heat production provinces),
* `lithos_asthenos_bdry`    (lithosphere-asthenosphere boundary),
* `magnetic_anomaly`,
* `thickness_crust`,
* `thickness_middle_crust`,
* `thickness_upper_crust`,
* `topography`,
* `upper_mantle_density_anomaly`.

and three categorical features:

* `thermo_tecto_age` (age of last thermo-tectonic event): allowed values are 1-12.
* `upper_mantle_vel_structure` (upper mantle velocity structure): allowed values are 1-6.
* `rock_type` (rock type): as per *Hartmann and Moosdorf (2012)*, allowed values are:
  **1** (volcanic), **2** (metamorphic), **3** (sedimentary).

Data Sets
---------

- *Global data (features and GHF)*: The global data set can be found at
  [`global.csv`](global.csv) which contains feature values and GHF measurements
  from points predominantly outside of GrIS.
- *GrIS data (features)*: All GrIS data set can be found at
  [`gris_features.csv`](gris_features.csv) which contains feature values (no
  GHF) from points on GrIS.
- *GrIS ice cores (GHF)*: Latitude, longitude, and GHF measurements from ice
   cores in Greenland.
- *"Original" data*: For both data sets above a more complete version is also
  included for posterity. The global data set differs from its corresponding
  original data set `global_original.csv` and the GrIS data set differs from
  `gris_features_original.csv` by:
    * For both global and GrIS data sets the following columns in original data
      sets are excluded:
        - `_lithk_cona`,
        - `_num_in_cell`,
        - `_num_in_continent`,
        - `_lithology_HM_unmodified`,
        - `_depth_to_moho_pasyanos`,
        - `_depth_to_moho_?`,
        - `_airy_gravity_anomaly`,
        - `_continent`,
    * The global data sets excludes two original global records with unknown rock
      type (`rock_type == -9999`).
    * Rock types (`rock_type` column) in original GrIS data set take values
      between 1-10, as per *Dawes (2009)*, which are mapped in GrIS data set to
      the three values described above, as per *Hartmann and Moosdorf (2012)*.
    * Rock types (`rock_type` column) in original global data set are taken by
      mapping the 16 values of `_lithology_HM_unmodified`, as per
      *Hartmann and Moosdorf (2012)*, to the three values described above.
