to run the file, you can run __init__.py .

__init__.py runs on text files, the path to which needs to be hard coded on line 12 of __init__.py
as the first variable of the Environment initializer. the second variable of the Environment is a list 
containing all the pentalies. 

our project is a setbased search that gets valid answers from an or-tree and through the use of a genetic
algorithm gets other valid ansewrs with different fitnesses. the genetic algorithm has a heap that contains all 
the valid schedules and their fitness.

to control the size of this heap, you can change the first variable of processor.processSchedules() in line 20 
of  __init__.py . 

To control the number of iterations the genetic algorithm runs, you can control the second variable of processor.processSchedules()
in line 20 of __init__.py .

