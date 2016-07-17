# Analysis of Recurrent Neural Networks forProbabilistic Modeling of Driver Behavior #

**Note:** The data used for this project is not included in this repository.  The reconstructed NGSIM data can be obtained by following the instructions [here](http://multitude-project.eu/enhanced-ngsim.html).

### List of Files ###

* *RNNspeed-predict.lua*: used for training LSTM networks
* *analyze.lua*: propagates simulated trajectories using trained LSTM networks
* *FFspeed-predict.lua*: used for training feedforward networks
* *FF_analyze.lua*: propagates simulated trajectories using trained feedforward networks
* The *util* folder contains various scripts called by the main scripts
* The *model* folder contains files that define the structure of the LSTM networks
* The *analysis* folder contains files that were used to analyze propagated trajectories.  These include:
    * *IDM_fit.ipynb*: used to learn the parameters for the IDM model
    * *compare_traj.jl*: used to find RWSE, KL Divergence, and fraction of negative state values