from environment import Environment
from schedule import Schedule
from orTree import OrTreeScheduler
from constr import Constr
import search_process
import timeit

if __name__ == "__main__":
    # wminfilled wpref wpair wsecdiff pengamemin penpracticemin pennotpaired pensection 
    # env = Environment('softConstraintsWOComments.txt', [1,1,1,1,7,8,5,10], verbose = 1)
    env = Environment('CPSC433F24-LargeInput1.txt', [1,1,1,1,7,8,5,10], verbose = 1)

    sched = Schedule(env)
    
    constraints = Constr(env)
    scheduler = OrTreeScheduler(constraints, env)
    processor = search_process.ScheduleProcessor(scheduler)

    schedule = processor.processSchedules(100,10)
    
    print(schedule)
    
    
    
    
    
    # Testing the assign function
    # sched.assign(scheduler.generate_schedule(starting))
    
    # sched.assign(["MO8:00","MO8:00","MO9:00","MO8:00","MO8:00","MO8:00","MO8:00","MO8:00"], True)
    # sched.assign([])
    # assign_time = timeit.timeit(lambda: sched.assign_and_eval(["MO8:00","MO8:00","TU10:00","TU10:00","MO9:00","FR9:00","FR8:00","TU12:30", "FR10:00", "TU9:00", "TU12:30", "TU12:30", "TU12:30"], verbose = 1), number=1)
    assign_time = timeit.timeit(lambda: sched.assign_and_eval([], verbose = 1), number=1)
    # assign2_time = timeit.timeit(lambda: sched.assign2_and_eval(["MO8:00","MO8:00","MO9:00","MO9:00","MO8:00","MO8:00","MO8:00","MO8:00"], True), number=1)

    
    print(f"Time taken for assign: {assign_time} seconds")
    # print(f"Time taken for assign2: {assign2_time} seconds")


