from environment import Environment
from schedule import Schedule
from orTree import OrTreeScheduler
from constr import Constr
import search_process
import timeit

if __name__ == "__main__":
    env = Environment('Jamie.txt', [1,0,1,0,10,10,10,10], verbose = 1)
    sched = Schedule(env)
    
    constraints = Constr(env)
    # scheduler = OrTreeScheduler(constraints, env)
    # processor = search_process.ScheduleProcessor(scheduler)
    
    # schedule = processor.processSchedules(100,10)
    
    # print(schedule)
    
    
    
    
    
    # Testing the assign function
    # sched.assign(scheduler.generate_schedule(starting))
    
    # sched.assign(["MO8:00","MO8:00","MO9:00","MO8:00","MO8:00","MO8:00","MO8:00","MO8:00"], True)
    # sched.assign([])
    assign_time = timeit.timeit(lambda: sched.assign(["MO8:00","MO8:00","MO9:00","MO8:00","MO8:00","MO8:00","MO8:00","MO8:00","MO9:00","MO9:00"], True), number=1)
    assign2_time = timeit.timeit(lambda: sched.assign2(["MO8:00","MO8:00","MO9:00","MO8:00","MO8:00","MO8:00","MO8:00","MO8:00","MO9:00","MO9:00"], True), number=1)

    
    print(f"Time taken for assign: {assign_time} seconds")
    print(f"Time taken for assign2: {assign2_time} seconds")


