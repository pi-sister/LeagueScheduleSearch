from environment import Environment
from schedule import Schedule
from orTree import OrTreeScheduler
from constr import Constr
import search_process
import timeit

if __name__ == "__main__":
    env = Environment('hc7.txt', [1,1,1,1,10,10,10,10], verbose = 1)
    sched = Schedule(env)
    
    constraints = Constr(env)
    scheduler = OrTreeScheduler(constraints, env)
    processor = search_process.ScheduleProcessor(scheduler)

    schedule = processor.processSchedules(100,10)
    
    print(schedule)
    
    timeslots = schedule.get_scheduled()
    
    # Sort the DataFrame alphabetically based on League, Tier, Div, Practice_Type, and Num
    timeslots = timeslots.sort_values(by=['League', 'Tier', 'Div', 'Practice_Type', 'Num'])

    # Maximum width for the left part before ':' to ensure consistent alignment
    max_width = max(timeslots.apply(lambda row: f"{row['League']} {row['Tier']} DIV {row['Div']} {row['Practice_Type']} {row['Num']}".strip(), axis=1).apply(len))

    print(f"Eval-value: {schedule.eval}")
    for _, row in timeslots.iterrows():
        base = f"{row['League']} {row['Tier']} DIV {row['Div']}"
        if row['Practice_Type']:
            base += f" {row['Practice_Type']} {row['Num']}"
        day, time = row['Assigned'][:2], row['Assigned'][2:]
        print(f"{base:<{max_width}} : {day}, {time}")


    # Testing the assign function
    # sched.assign(scheduler.generate_schedule(starting))
    
    # sched.assign(["MO8:00","MO8:00","MO9:00","MO8:00","MO8:00","MO8:00","MO8:00","MO8:00"], True)
    # sched.assign([])

    #assign_time = timeit.timeit(lambda: sched.assign_and_eval(["MO8:00","MO8:00","MO9:00","MO9:00","MO8:00","MO8:00","MO8:00","MO8:00"], verbose = 1), number=1)

    # assign_time = timeit.timeit(lambda: sched.assign_and_eval(["MO8:00","MO8:00","TU10:00","TU10:00","MO9:00","FR9:00","FR8:00","TU12:30", "FR10:00", "TU9:00", "TU12:30", "TU12:30", "TU12:30"], verbose = 1), number=1)

    # assign2_time = timeit.timeit(lambda: sched.assign2_and_eval(["MO8:00","MO8:00","MO9:00","MO9:00","MO8:00","MO8:00","MO8:00","MO8:00"], True), number=1)

    
    #print(f"Time taken for assign: {assign_time} seconds")
    # print(f"Time taken for assign2: {assign2_time} seconds")


