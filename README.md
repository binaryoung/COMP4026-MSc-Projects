# COMP4026 MSc Projects <br/>Solving Sokoban puzzle using multiple reinforcement learning methods


## Introduction

This repository contains code accompanying the dissertation"Solving Sokoban puzzle using multiple reinforcement learning methods". It includes the code of all the models experimented in the dissertation, also including the code of Boxoban environment, Boxoban level collection and Boxoban level generator.

## Dependencies
- numpy
- pytorch
- pandas
- gym-sokoban

## Folder Structure

    ├── boxoban-environment                                    # Boxoban environment  
    ├── boxoban-level-collection                               # Boxoban level collection  
    ├── boxoban-level-generator                                # Boxoban level generator  
    │   ├── levels                                             # One million training levels 
    ├── curriculum-learning
    │   ├── ppo.py                                             # PPO
    │   ├── ppo_lstm.py                                        # PPO LSTM
    │   ├── ppo_lstm_stacked.py                                # PPO LSTM Stacked  
    │   ├── ppo_lstm_truncated.py                              # PPO LSTM BPTT  
    │   ├── ppo_lstm_combine_observation.py                    # PPO LSTM + Observation  
    │   ├── ppo_convlstm.py                                    # PPO ConvLSTM
    │   ├── ppo_resnet.py                                      # PPO CNN + ResNet
    │   ├── ppo_resnet_pro.py                                  # PPO ResNet 
    │   ├── ppo_curriculum.py                                  # PPO Curriculum 
    │   ├── ppo_lstm_curriculum.py                             # PPO LSTM Curriculum 
    │   ├── ppo_lstm_resnet.py                                 # PPO LSTM ResNet
    ├── hierarchical
    │   ├── a2c.py                                             # A2C
    │   ├── a2c_lstm.py                                        # A2C LSTM
    │   ├── feudal_network.py                                  # FeUdal Networks  
    ├── imitation
    │   ├── gail.py                                            # GAIL
    │   ├── gail_resnet.py                                     # GAIL ResNet
    └── meta
    │   ├── maml.py                                            # MAML
    │   ├── meta_rl.py                                         # Meta-RL
    │   ├── meta_rl_resnet.py                                  # Meta-RL ResNet  

## Result
| Model                                   | Solved at 2e7 steps |
| --------------------------------------- | ------------------- |
| Our method - GAIL ResNet(strength=0.05) | 83.46%              |
| DRC(3, 3)                               | 80.00%              |
| I2A(unroll=15)                         | 21.00%              |
| VIN                                     | 12.00%              |
| ATreeC                                  | 1.00%               |
