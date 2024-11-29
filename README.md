# CGAD

This repository provides implementation of CGAD (Causal Graph for Multivariate Time Series Anomaly Detection)

# Requirements

This code is based on **Python 3.9**.

1. Install PyTorch with CUDA 12.1 support:

   ```bash
   pip install torch==2.2.0 --index-url https://download.pytorch.org/whl/cu121
   ```

2. Install PyTorch Geometric

   ```bash
   pip install torch-geometric==2.5.0 torch-scatter torch-sparse torch-cluster torch-spline-conv -f https://data.pyg.org/whl/torch-2.2.0+cu121.html
   ```

3. Install additional dependencies from the `requirements.txt` file:
   ```bash
   pip install -r requirements.txt
   ```

# Data preparation

1.  **Process raw datasets**. The corresponding raw datasets should be placed in `./data`. the datasets can be preprocessed using this command:

    ```
    python preprocess_data.py SWAT WADI MSL SMAP SMD
    ```

2.  **Use preprocessed data**. You can download our processed data for the SMD dataset from [here](https://drive.google.com/file/d/15qAW47HIzJ3UWO8euAMxCvrqX2NgGuih/view?usp=sharing). After downloading, unzip the file and move all .npy files to the `./preprocessed_data/SMD` directory.

# Run

Run the code using this command:

```
python main.py --dataset <dataset name> --subset <subset name> --retrain
```

For example:

```
python main.py --dataset SMD --subset machine-1-1 --retrain
```
