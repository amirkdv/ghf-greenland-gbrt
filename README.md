Geothermal Heat Flux Prediction in Greenland with Gradient Boosted Regression Trees
===================================================================================

Requirements
------------
The following are requirements to reproduce the results:

* _System wide_: Python 2.7 with developer tools, Tk with developer tools,
  build toolchain.
* _Python packages_: see [`requirements.txt`](requirements.txt)
* _Basemap_ (tested with veresions `1.0.7` and `1.1.0`), built from source. For
  automated procedures for Linux see [`Makefile`](Makefile) which automatically
  detects the virtual environment from the `VIRTUAL_ENV` environment variable.

On Debian/Linux the following installs all requirements:
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
(env) $ python density_plots # Figures 1 and 2
(env) $ python greenland.py # Figures 4, S5, S6
(env) $ python error_analysis # Figures 3, 5, S2, S3, S4
```

Features
--------
Each [data set](#data-files) contains the following continuous features:

* `age`        (age),
* `crusthk_cr` (crustal thickness),
* `d2_transfo` (distance to transform ridge),
* `d_2hotspot` (distance to hotspots),
* `d_2ridge`   (distance to ridge),
* `d_2trench`  (distance to trench),
* `d_2volcano` (distance to volcano),
* `ETOPO_1deg` (topography),
* `G_d_2yng_r` (distance to young rift),
* `G_heat_pro` (heat production provinces),
* `litho_asth` (lithosphere-asthenosphere boundary),
* `magnetic_M` (magnetic anomaly),
* `moho_GReD`  (depth to Mohorovičić discontinuity),
* `thk_mid_cr` (thickness of middle crust),
* `thk_up_cru` (thickness of upper crust),
* `upman_den_` (upper mantle density anom.),
* `WGM2012_Bo` (Bougeur gravity anomaly),

and three categorical features:

* `G_ther_age` (age of last thermo-tectonic event): allowed values are [TODO SRB]
* `G_u_m_vel_` (upper mantle velocity structure): allowed values are [TODO
  SRB]
* `lthlgy_mod` (rock type): allowed values are (as per *Hartmann and Moosdorf (2012)*):
  **1** (volcanic), **2** (metamorphic), **3** (sedimentary).

Date Files
----------

- *Global data (features and GHF)*: All global data can be found at
  `global_original.csv` which contains ...  [TODO SRB]. The subset of this
  file that is used in code can be found at `global.csv` which excludes the
  following:
    * two rows with unknown rock type (i.e `lthlgy_mod == -9999`),
    * dropped columns [TODO SRB]
- *GrIS data (features)*: All feature values for GrIS can be found at
  `gris_features_original.csv` which contains ... [TODO SRB]. The
  corresponding subset of this file that is used in code can be found at
  `gris_features.csv` which contains the following modifications:
    * translate rock types (`lthlgy_mod`) from the 10 categories of *Dowe
      (2009)* to the three categories of *Hartmann and Moosdorf (2012)* used
      in the global data set.
    * dropped columns similar to global data.
- *GrIS ice cores (GHF)*: Latitude, longitude, and GHF measurements from ice
   cores in Greenland.

*Note*: Only `global.csv`, `gris_features.csv`, and `gris_ice_cores.csv` are used in
in this repo (see `util.py`). The "original" data files (`global_original.csv`
and `gris_original.csv`) are only included for completeness.
