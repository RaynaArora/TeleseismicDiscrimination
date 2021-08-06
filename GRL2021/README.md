# Dataset for Teleseismic Discrimination
Dataset used for Teleseismic Discrimination Using Deep Learning

This repository contains the training, validation, and test datasets in Pandas dataframes that have been pickled. Some large files have been split into files with the endings .aa, .ab, etc.
After you download the data, combine the split files:

    cat df_train.pic.a* > df_train.pic
    cat x_train.pic.a* > x_train.pic
    cat df_rock.pic.a* > df_rock.pic
    cat x_rock.pic.a* > x_rock.pic

You can also run the "Collect and Format Data" notebook to newly obtain all the data from the internet and store it in the same pickle files.

Once the data is combined or you've collected it yourself, run the "Model and Figures" notebook to train and test the proposed model on the data.  The notebook also includes code to generate figures from the results.

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

- Arora, R., Le Bras, R.: TBD
- Storchak, D.A., Harris, J., Brown, L., Lieser, K., Shumba, B., Verney, R., Di Giacomo, D., Korger, E. I. M. (2017). Rebuild of the Bulletin of the International Seismological Centre (ISC), part 1: 1964–1979. Geosci. Lett. (2017) 4: 32, https://doi.org/10.1186/s40562-017-0098-z
- Storchak, D.A., Harris, J., Brown, L., Lieser, K., Shumba, B., Di Giacomo, D. (2020) Rebuild of the Bulletin of the International Seismological Centre (ISC)—part 2: 1980–2010. Geosci. Lett. 7: 18, https://doi.org/10.1186/s40562-020-00164-6

Include the following in your acknowledgements:
- The facilities of IRIS Data Services, and specifically the IRIS Data Management Center, were used for access to waveforms, related metadata, and/or derived products used in this study. IRIS Data Services are funded through the Seismological Facilities for the Advancement of Geoscience (SAGE) Award of the National Science Foundation under Cooperative Support Agreement EAR-1851048.
- All seismic data were downloaded through the IRIS Wilber 3 system (https://ds.iris.edu/wilber3/) or IRIS Web Services (https://service.iris.edu/), including the following seismic networks: (1) the AZ (ANZA; UC San Diego, 1982); (2) the TA (Transportable Array; IRIS, 2003); (3) the US (USNSN, Albuquerque, 1990); (4) the IU (GSN; Albuquerque, 1988)."

