# HRL-grid


The project is designed to simulate hierarchical reinforcement learning algorithms. 
There is two various environments: [grid_maze_env README](https://github.com/cog-isa/HRL-grid/blob/master/environments/grid_maze_env/GRID_MAZE_README.md) and [arm_env README](https://github.com/cog-isa/HRL-grid/blob/master/environments/arm_env/ARM_README.md). One should check the environments' READMEs for more information. 


### Hierarchies of Abstract Machines  
You can run experiments with handcrafted machines hierarchies in module [ham_experiments](https://github.com/cog-isa/HRL-grid/tree/master/HAM_new/HAM_experiments). 
And also examine [HAM's readme file](https://github.com/cog-isa/HRL-grid/blob/master/HAM_new/HAM_README.md).

### Prerequisites

Things you need to install the software:

```
sudo apt-get install python3-tk
sudo apt-get install python3-dev
```

For drawing graphs with pygraphviz one should install:

```
sudo apt-get install graphviz
sudo apt-get install graphviz-dev
sudo pip3.5 install pygraphviz --install-option="--include-path=/usr/include/graphviz" --install-option="--library-path=/usr/lib/graphviz/"
```
## Installing

To run `random_policy.py` and test the environment, you must install the following libraries:
```
gym
scipy
pandas
matplotlib
numpy
pygraphviz
```

## Getting Started


Run the file [`q-policy.py`](https://github.com/cog-isa/HRL-grid/blob/master/environments/q-policy.py), which will show an example of interaction on both environments with q-learning and random policy. 

## Authors

* **Alexander Panov** - *Project management* - [grafft](https://github.com/grafft)
* **Alexey Skrynnik** - *Environments.* *Hierarchical RL on HAMs* - [Tviskaron](https://github.com/tviskaron)
* **Vadim Kuzmin** - *Hierarchical RL on Options* - [vtkuzmin](https://github.com/vtkuzmin)


## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](https://github.com/cog-isa/HRL-grid/blob/master/LICENSE) file for details

