# ijcai21_data
Dataset used for Deep Learning Approaches for Teleseismic Discrimination and its Societal Implications

This repository contains the training, validation, and test datasets in Pandas dataframes that have been pickled. The training data is split into 8 files: trainaa, trainab, ... trainag.
After you download the data please combine the training files into a single pickle file:
    cat traina* > train.pic

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
- Is explosion: True if this event was an explosion, false if it was an earthquake. Explosions in this dataset include nuclear explosions, chemical explosions, and mining explosions.
- SampRate: Please ignore this. All waveforms have been downsampled to 20 Hz.
- Samples: Length 1800 array of amplitudes in waveform from 10 seconds before to 80 seconds after arrival at station. Sampled at 20 Hz. Filtered with highpass of 1 Hz.
