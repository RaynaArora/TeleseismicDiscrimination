# Dataset for Teleseismic Discrimination
Dataset used for Deep Learning Approaches for Teleseismic Discrimination

This repository contains the training, validation, and test datasets in Pandas dataframes that have been pickled. The training data is split into 8 files: trainaa, trainab, ..., trainag.
After you download the data, combine the training files into a single pickle file:

    cat traina* > train_info.pic

Below is the python code to unpickle the file.
```python

import pickle

with open("train_info.pic", "rb") as fp:
    df_train = pickle.load(fp)
with open("val_info.pic", "rb") as fp:
    df_val = pickle.load(fp)
with open("test_info.pic", "rb") as fp:
    df_test = pickle.load(fp)

```

Here are descriptions of the columns in the Pandas dataframes in order:

- ISC EventID
- Event Date
- Event Time
- Arrival Date
- Arrival Time
- Latitude of event
- Longitude of event
- Type of magnitude
- Magnitude
- Station which recorded this waveform
- Channel of station which recorded this waveform
- Seismic Phase Name
- Distance event was detected at in degrees (1 degree is about 111 km)
- Is explosion: True if this event was an explosion, false if it was an earthquake. Explosions in this dataset include nuclear explosions, chemical explosions, and mining explosions. In the rockburst dataframes(rock_train_info, rock_val_info, rock_test_info), this column is always true.
- SampRate: Please ignore this. All waveforms have been downsampled to 20 Hz.
- Samples: Length 3600 array of amplitudes in waveform from 60 seconds before to 120 seconds after arrival at station. Sampled at 20 Hz. Filtered with highpass of 1 Hz. In data passed to model, this sample array has been clipped to 10 seconds before to 80 seconds after the arrival and the amplitudes have been normalized.


# Citing

If you use this dataset, be sure to include the following citations:

- Arora, R., Arora, N., and Le Bras, R.: Analyzing Deep Learning Performance for Seismic Waveform Discrimination at Global Distances, EGU General Assembly 2021, online, 19–30 Apr 2021, EGU21-1831, https://doi.org/10.5194/egusphere-egu21-1831, 2021.
- Storchak, D.A., Harris, J., Brown, L., Lieser, K., Shumba, B., Verney, R., Di Giacomo, D., Korger, E. I. M. (2017). Rebuild of the Bulletin of the International Seismological Centre (ISC), part 1: 1964–1979. Geosci. Lett. (2017) 4: 32, https://doi.org/10.1186/s40562-017-0098-z
- Storchak, D.A., Harris, J., Brown, L., Lieser, K., Shumba, B., Di Giacomo, D. (2020) Rebuild of the Bulletin of the International Seismological Centre (ISC)—part 2: 1980–2010. Geosci. Lett. 7: 18, https://doi.org/10.1186/s40562-020-00164-6

Include the following in your acknowledgements:
- The facilities of IRIS Data Services, and specifically the IRIS Data Management Center, were used for access to waveforms, related metadata, and/or derived products used in this study. IRIS Data Services are funded through the Seismological Facilities for the Advancement of Geoscience (SAGE) Award of the National Science Foundation under Cooperative Support Agreement EAR-1851048.
- All seismic data were downloaded through the IRIS Wilber 3 system (https://ds.iris.edu/wilber3/) or IRIS Web Services (https://service.iris.edu/), including the following seismic networks: (1) the AZ (ANZA; UC San Diego, 1982); (2) the TA (Transportable Array; IRIS, 2003); (3) the US (USNSN, Albuquerque, 1990); (4) the IU (GSN; Albuquerque, 1988)."

