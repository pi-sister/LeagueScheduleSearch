# Charmander's Schedulue

### To Run
to run the file, you can run 
```
__init__.py
```
You can enter what's needed in the terminal input

Or

You can either manually run the code by editing the file:
__init__.py runs on text files, the path to which needs to be hard coded on line 12 of __init__.py
as the first variable of the Environment initializer. The second variable of the Environment is a list 
containing all the pentalies. 

### Tree
Our project is a setbased search that gets valid answers from an or-tree and through the use of a genetic
algorithm gets other valid ansewrs with different fitnesses. the genetic algorithm has a heap that contains all 
the valid schedules and their fitness.

to control the size of this heap, you can change the first variable of processor.processSchedules() in line 20 
of  __init__.py . 

To control the number of iterations the genetic algorithm runs, you can control the second variable of processor.processSchedules()
in line 20 of __init__.py .

### Optimizing
To optimize the code for large inputs, we first schedule in the games/practices with the most constraints. This is determined by the the score() function in the orTree.py file.
Additionally, we also returned only the valid possible slot options to branch into. This helped prune the tree by not expanding to branches that fail hard constraints.
We also implemented guided crossover to try and generate the optimal schedule. This is the paper this concept was found on: https://ijisae.org/index.php/IJISAE/article/view/790/pdf
