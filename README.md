# Caugation

This repository provides implementation of CGAD (Causal Graph for Multivariate Time Series Anomaly Detection)

# Requirements

The code is based on Python 3.9

```
pip install -r requirements.txt
```

# Data preparation

The corresponding raw datasets should be placed in `./data`. the datasets can be preprocessed using this command:

```
python preprocess_data.py SWAT WADI MSL SMAP SMD
```

You can download the processed data for the SMD dataset from [here](https://drive.google.com/file/d/15qAW47HIzJ3UWO8euAMxCvrqX2NgGuih/view?usp=sharing). After downloading, unzip the file and move all .npy files to the `./preprocessed_data/SMD` directory.

# Run

Run the code using this command:

```
python main.py --dataset <dataset name> --subset <subset name> --retrain
```
