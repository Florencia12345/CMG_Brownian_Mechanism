This repo contains code for simulating and analyzing CMG helicase DNA unwinding dynamics.

**Main Simulation**
cmg_brownian_ratchet_improved.py
  Simulates trajectories using different transition matrices and analyzes the resulting dwell time data.
  It generates:
    n trajectories, performs individual dwell time analysis for each trajectory, performs collective analysis across all trajectories of the same transition matrix. 
  It computes:
    Survival probability
    First-passage time density
  And it fits:
    Linear fit to log(survival) for sanity check
    Basic and robust fits to first-passage density (e.g. biased ratchet motion models)
  Outputs include trajectories, dwell time stats, survival curves, and fitting results.

**Helper Functions**
  visualize_2.py
    Contains helper functions used by cmg_brownian_ratchet_improved.py.
    Includes utilities for:
      Data processing
      Plotting
      Survival / first-passage calculations
      
**Dwell Time PDF Fitting**
  fit_dwelltime.py
  It fits a more complex probability density function (PDF) to first-passage time data using Maximum Likelihood Estimation (MLE).
  The fitted parameters can be interpreted in terms of physical meaning (e.g., kinetic rates, mechanistic properties).

**HMM / HSMM Analysis(still under development and it may not work eventually)**
  cmg_hmm_hsmm_analysis.py
  hmm_analysis_outputs/
Performs:
  Hidden Markov Model (HMM) fitting
  HSMM-related analysis (if applicable)
