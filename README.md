\# Real-Time Financial Fraud Detection Pipeline



An end-to-end, production-grade fraud detection machine learning system built on transactional data. The project emphasizes leak-free feature engineering, temporal boundary validation, operational alert-capacity thresholding, and sub-millisecond inference parity.



\---



\## 📌 Architecture \& Design Principles



Traditional fraud detection benchmarks often introduce subtle data leakage by aggregating future transaction patterns or relying on post-transaction target balances. This pipeline enforces strict operational realism:



1\. \*\*Leak-Free Historical Aggregations (V3.1):\*\* Features rely strictly on historical account statistics up to the exact transaction timestamp ($t\_{tx} - \\epsilon$), preventing future event leakage.

2\. \*\*Current vs. Historical Ablation:\*\* Explicit ablation testing isolates the exact marginal uplift delivered by historical customer behavioral profiles over raw point-in-time transaction features.

3\. \*\*Operational Alert Capacity:\*\* Threshold selection is evaluated against realistic SOC / fraud investigation bandwidth rather than default 0.5 classification thresholds.

4\. \*\*Standalone Inference Parity:\*\* Inference runs through an independent, self-contained pipeline (`src/inference\_pipeline.py`) verified against training outputs and boundary conditions.



\---



\## 🗂️ Project Structure



```text

├── models/

│   └── experiments/

│       ├── lightgbm\_historical\_v3\_1.txt              # Core validated LightGBM model

│       ├── lightgbm\_current\_only.txt                 # Baseline comparison model

│       └── ablation/                                 # Feature ablation models

├── src/

│   ├── inference\_pipeline.py                         # Production inference entrypoint

│   ├── prepare\_historical\_features\_v3\_1.py           # Validated historical feature generator

│   ├── train\_historical\_ablation.py                  # Ablation experiments \& training

│   ├── train\_historical\_comparison.py                # Comparative model benchmarking

│   ├── evaluate\_alert\_capacity.py                    # Alert budget \& threshold evaluation

│   ├── validate\_data.py                              # Schema \& data sanity validation

│   ├── test\_inference\_predictions.py                 # Numerical parity testing

│   ├── test\_inference\_validation.py                  # Schema input validation suite

│   ├── verify\_full\_inference.py                      # End-to-end inference verification

│   ├── verify\_history\_boundaries.py                  # Temporal leakage audit

│   ├── verify\_inference\_pipeline.py                  # Pipeline lifecycle audit

│   ├── verify\_prediction\_parity.py                   # Model consistency validation

│   ├── perform\_eda.py                                # Exploratory data analysis

│   ├── inspect\_data.py                               # Raw data inspection

│   ├── inspect\_v3\_extremes.py                        # Feature extreme/outlier inspection

│   ├── audit\_features.py                             # Feature distribution checks

│   ├── audit\_historical\_features.py                  # Historical feature integrity

│   ├── audit\_historical\_features\_v3.py               # V3 feature integrity

│   └── audit\_account\_reuse.py                        # Account re-use behavior audit

├── requirements.txt                                  # Environment dependencies

└── .gitignore                                        # Excludes datasets, caches \& archives

