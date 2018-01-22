The environment `maze_world_env.py` is a labyrinth of repeating patterns.

```
 O  O  O  O  O  O  O  O  O  O  O  O  O  O  O 
 O  .  .  .  O  O  .  .  .  O  O  .  S  .  O 
 O  .  O  .  .  .  .  O  .  .  .  .  O  .  O 
 O  .  .  .  O  O  .  .  .  O  O  .  .  .  O 
 O  O  .  O  O  O  O  .  O  O  O  O  .  O  O 
 O  O  .  O  O  O  O  .  O  O  O  O  .  O  O 
 O  .  .  .  O  O  .  .  .  O  O  .  .  .  O 
 O  .  O  .  .  .  .  O  .  x  .  .  O  .  O 
 O  .  .  .  O  O  .  .  .  O  O  .  .  .  O 
 O  O  .  O  O  O  O  .  O  O  O  O  .  O  O 
 O  O  .  O  O  O  O  .  O  O  O  O  .  O  O 
 O  .  .  .  O  O  .  .  .  O  O  .  .  .  O 
 O  .  O  .  .  .  .  O  .  .  .  .  O  .  O 
 O  .  .  .  O  O  .  .  .  O  O  .  .  .  O 
 O  O  .  O  O  O  O  .  O  O  O  O  .  O  O 
 O  O  .  O  O  O  O  .  O  O  O  O  .  O  O 
 O  .  .  .  O  O  .  .  .  O  O  .  .  .  O 
 O  .  O  .  .  .  .  O  .  .  .  .  O  .  O 
 O  .  F  O  O  O  .  .  .  O  O  .  .  .  O 
 O  O  O  O  O  O  O  O  O  O  O  O  O  O  O 
```





>  `O` -- is an obstacle 
 `S` -- start 
 `F` -- finish 
 `-` -- free cell 
 `x` -- agent position



The agent can perform four actions to move: up, down, left and right.
If the agent encounters a wall as a result of the action, the reward is `-5`.
If, after performing an action, the agent has moved to an empty cell - the reward is `-1`.
If the agent is in the terminal state `F` - the reward is `+100`.

In the file `grid_maze_generator.py` there is methods for automatic generation of labyrinths. Each labyrinth consists of patterns:
```
 O  O  .  O  O   
 O  .  .  .  O  
 .  .  O  .  .
 O  .  .  .  O 
 O  O  .  O  O  
```	  

In which instead of empty points ```.``` additional walls ```O``` can be specified.. 
In the file `random_policy.py` there is an example of interaction with the environment. And after the end of the work schedules are drawn. It is possible to display the state of the environment at the current time.

## Getting Started

Run the file `random_policy.py`, which will show an example of interaction with the environment. To add your own strategy proposes to understand the structure of this file.
