## Non-official Code Repository for Paper 'A Novel Closed-Loop Clustering Algorithm for Hierarchical Load Forecasting'

### REQUIREMENT
- Python
- Pandas
- Numpy

### INTRODUCTION

- `Temperature_1.ipynb`: Download temperature data from the [open-source data website]("https://archive-api.open-meteo.com/v1/era5")
- `Generation_1.ipynb`: Generate users' load data (simulated_load.npy) based temperature data, and save in `Simulated_data/`
- `Benchmark_1 / 2 / 3.ipynb`: Benchmark methods for Experiment 1 (simulated data) / 2 (30min) / 3 (60min)
- `CLC_1 / 2 / 3.ipynb`: Proposed Closed-Loop Clustering Algorithm (CLC) for Experiment 1 (simulated data) / 2 (30min) / 3 (60min)
- `Radiation_Gen.ipynb`: Generate irradiation data based on distributed PV data
- `PV_data.npz`: Distributed photovoltaic data
- `user_load.npz`: Irish consumer load data
- `utils.py`: General Functions for Experiment 1 (simulated data)
- `utils2.py`: General Functions for Experiment 2 (30min) / 3 (60min)
- `Metrics_1 / 2 / 3.ipynb`: Calculate the results
- `\\Process_1 / 2 / 3`: Experiments results are saved in these folders

### The Display of Trim & Merge Mechanism

![Trim & Merge](Trim%20&%20Merge.gif)


### if you have any question, please feel free to contace [ME](linrd@connect.hku.hk)
