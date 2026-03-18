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
