from environment import Environment
from schedule import Schedule
from orTree import OrTreeScheduler
from constr import Constr
import time
import timeit

if __name__ == "__main__":
    env = Environment('ShortExample.txt', [1,0,1,0,10,10,10,10])
    sched = Schedule(env)
    
    constraints = Constr(env)
    scheduler = OrTreeScheduler(constraints, env)
    
    starting = sched.get_Starting()
    
    # sched.assign(scheduler.generate_schedule(starting))
    
    # sched.assign(["MO8:00","MO8:00","MO9:00","MO8:00","MO8:00","MO8:00","MO8:00","MO8:00"], True)
    # sched.assign([])
    assign2_time = timeit.timeit(lambda: sched.assign2(["MO8:00","MO8:00","MO9:00","MO8:00","MO8:00","MO8:00","MO8:00","MO8:00"], True), number=1)

    assign_time = timeit.timeit(lambda: sched.assign(["MO8:00","MO8:00","MO9:00","MO8:00","MO8:00","MO8:00","MO8:00","MO8:00"], True), number=1)
   
    print(f"Time taken for assign: {assign_time} seconds")
    print(f"Time taken for assign2: {assign2_time} seconds")
