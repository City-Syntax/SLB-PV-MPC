# SLB-PV-MPC: Degradation-Aware Optimization for Second-Life Batteries
This repository contains a first-of-its-kind simulation framework for integrating Second-Life Batteries (SLBs) into photovoltaic-integrated buildings. The project optimizes building energy systems by balancing electricity price arbitrage with long-term battery health. The core objective is to minimize building electricity costs under day-ahead pricing while explicitly managing the accelerated degradation of retired electric vehicle (EV) batteries. 

## Project Overview
As the volume of retired EV batteries surges, SLBs offer a lower-cost alternative to new lithium-ion batteries for stationary storage. However, their performance is characterized by higher uncertainty and reduced state-of-health (SOH).This project implements a Rolling Day-Ahead Model Predictive Control (R-DA-MPC) strategy that treats battery degradation as a finite "cycling budget". By balancing electricity price arbitrage with an equivalent full-cycle (EFC) allocation mechanism, the framework ensures the battery achieves its 5-year intended service life while maximizing economic returns.

## Key Features
1. Physics-Based SLB Modeling
- Equivalent Circuit Model (ECM): Uses an R–2RC structure parameterized with experimental data from retired 18650 cells.
- Degradation-Aware: Captures both SOC and SOH-dependent behavior, specifically modeling how power capability declines as internal resistance grows.
- Dynamic Power Limits: Derives physically realistic charge/discharge constraints to prevent overcharge and excessive stress

2. Rolling Day-Ahead MPC (R-DA-MPC)
- Optimization Logic: Solves a daily optimization problem to decide when to charge (low-price/PV surplus) and discharge (high-price/peak demand).
- EFC Budgeting: Features a bisection search for an "ageing weight" ($\lambda$) that ensures daily battery usage aligns with its remaining lifetime.
- Receding Horizon: Continuously adapts to updated forecasts and actual battery states to close the feedback loop.

3. Energy Flow Forecasting
- Transformer Models: Utilizes two independent Transformer-based neural networks to predict building demand and PV generation.
- Rolling Training: Employs a sliding-window strategy (using the most recent 3 weeks of data) to maintain forecast accuracy across seasonal shifts.

## Case Study Performance
The framework was validated using a 5-year dataset from a German industrial facility.
- Economic Viability: Achieved 10.1% lifecycle cost savings, outperforming rule-based strategies which stayed below 2%.
- Payback Potential: Under favorable conditions (SLBs 80% cheaper than new batteries), investment recovery is feasible within a 5-year lifecycle even with 20% market friction.
rbitrage Drivers: Results indicate that electricity price spread is the dominant factor for performance, whereas PV surplus has a more limited impact on buildings with high existing self-consumption.

## Research Significance
This framework bridges the gap between theoretical optimization and practical operation. It demonstrates that with degradation-aware control, retired EV batteries can be transformed into a viable, cost-effective energy storage solution for modern sustainable buildings.

## Repository Structure
The repository is organized so that each stage of the simulation pipeline is encapsulated by a single notebook, supported by reusable Python modules and folders that hold inputs, intermediate artifacts, and final results.

- Core notebooks (pipeline entry points, run in this order):
  - [data_cleaning.ipynb](data_cleaning.ipynb)
  - [prediction_generation.ipynb](prediction_generation.ipynb)
  - [Cell_Model_Generation.ipynb](Cell_Model_Generation.ipynb)
  - [rule_based_control_run_final.ipynb](rule_based_control_run_final.ipynb)
  - [theoretical_limit_final.ipynb](theoretical_limit_final.ipynb)
  - [R-DA-MPC_final_v2.ipynb](R-DA-MPC_final_v2.ipynb)
  - [visualization_and_analysis.ipynb](visualization_and_analysis.ipynb)
- Supporting Python modules (called from the notebooks): [cell_slb.py](cell_slb.py), [pack_slb.py](pack_slb.py), [battery_system_slb.py](battery_system_slb.py), [control_rule_based_slb.py](control_rule_based_slb.py), [control_rule_based_cost_slb.py](control_rule_based_cost_slb.py), [prediction_main.py](prediction_main.py), [theo_dp_core_v2.py](theo_dp_core_v2.py), [theoratical_limit_calculation.py](theoratical_limit_calculation.py), [run_search_then_apply_rolling_v2.py](run_search_then_apply_rolling_v2.py), [plot_config.py](plot_config.py).
- Data and parameters:
  - `DATASET/` - raw inputs (building electricity load, weather, multi-year day-ahead electricity prices).
  - `Processed_Dataset/` - cleaned and resampled data produced by `data_cleaning.ipynb`.
  - `Prediction_Result/` - day-ahead demand and PV forecasts produced by `prediction_generation.ipynb`.
  - `battery_model_params_20260214/` - precomputed SLB cell-model lookup tables produced by `Cell_Model_Generation.ipynb`.
  - [SLB_para.xlsx](SLB_para.xlsx) - experimentally identified parameters for the retired 18650 cell.
- Simulation outputs:
  - `Operation_Comparison_Result/` - hourly operation logs for RBC, RBC/A, theoretical limit, and R-DA-MPC, plus the per-day calibration log under `Operation_Comparison_Result/R-DA-MPC_log/`.
  - `Past_Weeks_Result/`, `Recalibration_Day_Gap_Result/`, `Prediction_Accuracy_Result/` - sensitivity analysis outputs from `R-DA-MPC_final_v2.ipynb`.
  - `Plots/` - final paper figures (PDF and SVG).

## Installation
The code is developed and tested with Python 3.10+. A CUDA-capable GPU is recommended for the Transformer-based forecasting in `prediction_generation.ipynb`, but the rest of the pipeline runs on CPU.

```bash
git clone <this-repository-url>
cd SLB-PV-MPC
python -m venv .venv
# Windows PowerShell
.venv\Scripts\Activate.ps1
# macOS / Linux
# source .venv/bin/activate
pip install -r requirements.txt
```

Notes on PyTorch:
- [requirements.txt](requirements.txt) pins `torch==2.7.1+cu118`, `torchvision==0.22.1+cu118`, and `torchaudio==2.7.1+cu118`, which are CUDA 11.8 wheels.
- If you do not have CUDA 11.8 (for example, you are CPU-only or use a different CUDA version), install a matching PyTorch build from the official PyTorch index first, then run `pip install -r requirements.txt` (the matching torch package will already be satisfied).

Once dependencies are installed, launch Jupyter from the repository root:

```bash
jupyter lab
# or
jupyter notebook
```

## Usage Workflow
The simulation pipeline is split into seven stages. Each stage corresponds to a single notebook; running them in order reproduces every result reported in the paper. The inputs and outputs of every stage are written into the folders listed under "Repository Structure", so each stage can find what the previous stage produced.

**Important note on exporting results**: throughout this repository, the lines that save CSV files and export figures (PDF/SVG) are commented out by default. If you want to write outputs to disk, you must uncomment the corresponding save/export lines in the notebooks you run.

1. Data cleaning - [data_cleaning.ipynb](data_cleaning.ipynb)
   - Purpose: clean and resample the case-study load and weather data, and consolidate the multi-year day-ahead electricity prices into a single time series.
   - Inputs: `DATASET/electricity_P.csv`, `DATASET/weather.csv`, `DATASET/Day_Ahead_Price2020.csv` ... `DATASET/Day_Ahead_Price2025.csv`.
   - Outputs: `Processed_Dataset/3.5year_elec_df_1h_2.csv`, `Processed_Dataset/3.5year_weather_df_1h_2.csv`, `Processed_Dataset/price_df.csv`.

2. Forecast generation - [prediction_generation.ipynb](prediction_generation.ipynb)
   - Purpose: train the two Transformer models (building demand and PV generation) with a 3-week rolling window and emit the day-ahead forecasts that feed the R-DA-MPC controller.
   - Inputs: cleaned data in `Processed_Dataset/`. Uses [prediction_main.py](prediction_main.py) for model definition and training utilities.
   - Outputs: `Prediction_Result/pred_result_df_2.csv`, `Prediction_Result/pred_performance_demand_df_2.csv`, `Prediction_Result/pred_performance_pv_df_2.csv`.

3. SLB cell-model precomputation - [Cell_Model_Generation.ipynb](Cell_Model_Generation.ipynb)
   - Purpose: precompute the SLB cell ECM and degradation lookup tables for several SOC operating windows so that every operation-time simulation can evaluate cell behavior with table lookups instead of solving the ECM online.
   - Inputs: [SLB_para.xlsx](SLB_para.xlsx). Uses [cell_slb.py](cell_slb.py), [pack_slb.py](pack_slb.py), and [battery_system_slb.py](battery_system_slb.py).
   - Outputs: per-window subfolders under `battery_model_params_20260214/` (e.g., `1_hour_0.0_1.0/`, `1_hour_0.1_0.9/`, ...).

4. Lower benchmarks - [rule_based_control_run_final.ipynb](rule_based_control_run_final.ipynb)
   - Purpose: simulate the rule-based controller (RBC) and the cost-aware variant (RBC/A) used as lower benchmarks.
   - Inputs: `Processed_Dataset/`, `battery_model_params_20260214/`. Uses [control_rule_based_slb.py](control_rule_based_slb.py) and [control_rule_based_cost_slb.py](control_rule_based_cost_slb.py).
   - Outputs: `Operation_Comparison_Result/RBC#*.csv` and `Operation_Comparison_Result/RBC-A#*.csv`.

5. Upper benchmark - [theoretical_limit_final.ipynb](theoretical_limit_final.ipynb)
   - Purpose: compute the theoretical upper bound on lifecycle savings via dynamic programming with perfect foresight.
   - Inputs: `Processed_Dataset/`, `battery_model_params_20260214/`. Uses [theo_dp_core_v2.py](theo_dp_core_v2.py) and [theoratical_limit_calculation.py](theoratical_limit_calculation.py).
   - Outputs: `Operation_Comparison_Result/TL#*.csv` (e.g., `TL#1@46%.csv`, `TL#1@80%.csv`).

6. Proposed strategy and sensitivity analysis - [R-DA-MPC_final_v2.ipynb](R-DA-MPC_final_v2.ipynb)
   - Purpose: simulate the proposed Rolling Day-Ahead MPC strategy and run all sensitivity studies (past-weeks window, recalibration day gap, prediction accuracy, market friction).
   - Inputs: `Processed_Dataset/`, `Prediction_Result/`, `battery_model_params_20260214/`. Uses [run_search_then_apply_rolling_v2.py](run_search_then_apply_rolling_v2.py) for the bisection search of the ageing weight and the receding-horizon application.
   - Outputs:
     - Main run: `Operation_Comparison_Result/R-DA-MPC#*.csv` plus per-day calibration logs in `Operation_Comparison_Result/R-DA-MPC_log/`.
     - Sensitivity studies: `Past_Weeks_Result/`, `Recalibration_Day_Gap_Result/`, `Prediction_Accuracy_Result/`.

7. Analysis and visualization - [visualization_and_analysis.ipynb](visualization_and_analysis.ipynb)
   - Purpose: aggregate every CSV produced in steps 1-6, compute the headline metrics (lifecycle cost savings, EFC usage, payback), and render every data-driven figure in the paper.
   - Inputs: `Processed_Dataset/`, `Prediction_Result/`, `Operation_Comparison_Result/`, `Past_Weeks_Result/`, `Recalibration_Day_Gap_Result/`, `Prediction_Accuracy_Result/`. Uses [plot_config.py](plot_config.py) for shared styling.
   - Outputs: paper figures under `Plots/` as PDF and SVG (`FIG2-...` through `FIG15-...`).

## Reproducing Paper Figures
After installing the dependencies, run the notebooks in the order listed under "Usage Workflow" (steps 1 through 6) to populate `Processed_Dataset/`, `Prediction_Result/`, `battery_model_params_20260214/`, `Operation_Comparison_Result/`, and the three sensitivity-analysis folders. Then execute [visualization_and_analysis.ipynb](visualization_and_analysis.ipynb) end to end. Every data-driven figure in the paper, named `FIG2-...` through `FIG15-...` in both PDF and SVG, will be regenerated under `Plots/`. Schematic figures distributed as `.pptx` (FIG1, FIG5, FIG6) are not produced by code and are kept in `Plots/` as-is. 

## License
This project is released under the terms of the [LICENSE](LICENSE) file included in the repository.
