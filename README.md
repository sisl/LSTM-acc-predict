# CS 221 Final Project #

Jeremy Morton -- 12/8/15

**Note:** I am not allowed to distribute the data used for this project, so unfortunately this code will not run solely using the files in this repository.

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