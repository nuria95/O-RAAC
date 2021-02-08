# O-RAAC
Offline Risk-Averse Actor-Critic (O-RAAC). A model-free RL algorithm for risk-averse RL in a fully offline setting.

Code to reproduce the results in: [Risk-Averse Offline Reinforcement Learning]("https://openreview.net/forum?id=TBIzh9b5eaz")

## Installation

O-RAAC can be installed by cloning the repository as follows. I recommend creating a separate virtual environment for it.

```
git clone git@github.com:nuria95/O-RAAC.git
cd O-RAAC
virtualenv venv
pip install -e .
```

In order to run O-RAAC you need to install D4RL (follow the instructions in https://github.com/rail-berkeley/d4rl) and  you need MuJoCo as a dependency. You may need to obtain a license and follow the setup instructions for mujoco_py. This mostly involves copying the key to your MuJoCo installation folder.

# Running:
To train the ORAAC model you need to:
Activate the environment:
`source venv/bin/activate`
Run the code:
`python3 experiments/oraac.py --config_name 'name_json_file'`

where 'name_json_file' is a .json file stored in json_params/ORAAC.
We provide the json files with the default parameters we used for each 
environment to get the results in the paper.
To optimize for different risk distortions modify the field in the json_file
or provide it as an argument accordingly.
--RISK_DISTORTION 'risk_distortion'
where 'risk_distortion' can be 'cvar' or 'cpw' or 'wang'.
 
The best trained models (according to 0.1 CVaR and Mean metrics) for each environment are saved in the `model-zoo` folder.
To evaluate the policies using such models you can do:
`python3 experiments/oraac.py --model_path 'name of environment'`
where 'name of environment' can be:
* halfcheetah-medium
* halfcheetah-expert
* walker2d-medium
* walker2d-expert
* hopper-medium
* hopper-expert

We also implemented the BEAR algorithm for the baselines.
To run it:
`python3 experiments/run.py --agent_name BEAR --config_name 'name_json_file'`




