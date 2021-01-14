# O-RAAC
Offline Risk-Averse Actor-Critic (O-RAAC). A model-free RL algorithm for risk-averse RL in a fully offline setting.

Code to reproduce the results in: [Risk-Averse Offline Reinforcement Learning]("https://openreview.net/forum?id=TBIzh9b5eaz")

## Installation

O-RAAC can be installed by cloning the repository as follows:
```
git clone git@github.com:nuria95/O-RAAC.git
cd O-RAAC
pip install -e .
```

In order to run O-RAAC you need to install D4RL (follow the instructions in https://github.com/rail-berkeley/d4rl).

# Running:
To train the ORAAC model you need to:
Activate the environment:
`source venv/bin/activate`
Run the code:
`python3 experiments/oraac.py --config_name 'name_json_file'`

where 'name_json_file' is a .json file stored in json_params/ORAAC
We provide the json files with the default parameters we used for each 
environment to get the results in the paper.
To optimize for different risk distortions modify the field in the json_file
or provide it as an argument accordingly.
--RISK_DISTORTION 'risk_distortion'
where 'risk_distortion' can be 'cvar' or 'cpw' or 'wang'.
 

The best models (according to 0.1 CVaR and Mean metrics)
during evaluation episodes in training are saved in 
data_ICLR/O_RAAC/environment_name/train/models/...


We also implemented the BEAR algorithm for the baselines.
To run it:
`python3 experiments/run.py --agent_name BEAR --config_name 'name_json_file'`




