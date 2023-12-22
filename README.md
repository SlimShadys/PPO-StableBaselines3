# PPO-StableBaselines3
This repository contains a re-implementation of the Proximal Policy Optimization (PPO) algorithm, originally sourced from Stable-Baselines3.

The purpose of this re-implementation is to provide insight into the inner workings of the PPO algorithm in these environments:
- LunarLander-v2
- CartPole-v1

## Requirements
1. Install Python version 3.9.x
2. Install Visual C++ 14.0 or greater from https://visualstudio.microsoft.com/visual-cpp-build-tools/
3. Run `pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118`
4. Run `pip install stable-baselines3[extra]==2.2.1`
5. Run `pip install swig`
6. Run `pip install gymnasium`
7. Run `pip install gymnasium[box2d]`

## Run the script
1. Change the game in `main.py` as you wish (`LunarLander-v2` / `CartPole-v1`)
2. Simply run `python main.py`

## Test your model
1. Simply run `python test.py` (as of now, running the test script will load my best model for both LunarLander-v2 and CartPole-v1)


## To-do
- [x] Rollout Buffer
- [x] Model
- [x] Training phase
- [x] Testing phase
- [ ] Run game from Terminal (Example: `python main.py --game 'LunarLander-v2'`)
- [ ] Load model from Terminal (Example: `python main.py --game 'LunarLander-v2' --model 'model.pt'`)
- [ ] Support `CarRacing-v2` environment

### Disclaimer
This repository includes parts of code that has been adapted from the Stable Baselines library (https://github.com/DLR-RM/stable-baselines3) for educational purposes only.
The original code is the property of its respective owners and is subject to their licensing terms.

I do not claim any ownership, copyright, or proprietary rights over the code obtained from Stable Baselines. The use of this code in this repository is solely for educational and learning purposes, and any commercial use or distribution is subject to the original licensing terms provided by Stable Baselines.

The original Stable Baselines code is licensed under the MIT License, and any use of their code in this repository is also subject to the terms of the MIT License.