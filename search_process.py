



# imports for heap
from heapq import heapify, heappush, heappop, nlargest
# Import the f_select function
from f_select import f_select
from main import set_Eval
from or_tree import OrTreeScheduler




# class
class ScheduleProcessor:
    """
    Class to manage and process a collection of schedules using a minimum heap data structure.
    
    This class supports three main operations:
    1. Adding a random schedule if the number of schedules is below a threshold. The random schedule will be generated using an Or-Tree function written by Emily
    2. Deleting the worst schedule if the number of schedules exceeds a predefined limit.
    3. Selecting the best schedule using a custom selection process called fselect. Fselect will be genearted by Kourosh

    """

    def __init__(self, gameSlots, practiceSlots, env):
        """
        Initializes the ScheduleProcessor with an empty heap.
        """
        # Create an empty heap for schedules
        self.heap = []
        heapify(self.heap)  # Convert the list into a valid heap structure
        scheduler = OrTreeScheduler(gameSlots, practiceSlots, env) 

    def assignNum(self, limitOfSchedules):
        """
        Determines which operation to perform based on the number of schedules in the heap.

        Parameters:
            limitOfSchedules (int): The maximum allowable number of schedules in the heap.

        Returns:
            int: Operation code:
                0 -> Add a new random schedule.
                1 -> Remove the worst schedule.
                2 -> Select the next best schedule.
        """
        if len(self.heap) <= 4:
            num = 0  # Random operation
        elif len(self.heap) > limitOfSchedules:
            num = 1  # Deletion operation
        else:
            num = 2  # fselect
        return num

    def fwert(self, num):
        """
        Executes one of the three operations based on the input operation code.

        Parameters:
            num (int): Operation code determined by the assignNum method:
                0 -> Add a random schedule.
                1 -> Remove the worst schedule.
                2 -> Perform a selection operation.
        """
        match num:
            case 0:  
                # generate a random new schedule
                newSchedule = scheduler.generate_schedule()
                # Get the value of it from set_Eval
                fitness = newSchedule.set_Eval()
                # Add the new schedule to the heap.
                # this is a max heap for effeciency so invert the value
                heappush(self.heap, (-fitness, newSchedule))

                
                
            case 1:  
                # Remove the worst schedule
                # Remove the largest value in the heap (the worst schedule in a min heap).
                worst = heappop(self.heap)  # Remove by value


            case 2:  
                # Select the best schedule
                # Call the selection function.
                schedule = f_select(self.heap)
                

            case _:  # Default case: also perform selection
                # Select the best schedule
                # Call the selection function.
                schedule = f_select(self.heap)
        return schedule


def f_select(current_state):
    """
    f_select function to select a transition from the given state based on the probabilities.
    
    Args:
        current_state (list): List of all (fitness, schedules) in the current state.
        fitness (function): A function to calculate the fitness value of a given schedule.
    
    Returns:
        a new schedule by calling either mutate or crossover?
    """
    

    # have to add the '-' because this is a max heap with inverted values
    # Calculate the total fitness of all schedules in the current state
    total_fitness = sum(-fitness for fitness, schedule in current_state)
    
    # Calculate the normalized probability for each schedule
    # have to add the '-' because this is a max heap with inverted values
    probabilities = [1 - (-fitness / total_fitness) for fitness, schedule in current_state]
    
    # Normalize the probabilities so they sum up to 1
    total_prob = sum(probabilities)
    normalized_probabilities = [prob / total_prob for prob in probabilities]
    
    # Choose between Mutation and Crossover with equal probability
    transition = random.choices(['Mutation', 'Crossover'], weights=[0.5, 0.5])[0]
    

    if transition == 'Mutation':
        # Select one schedule for mutation based on the calculated probabilities
        selected_schedule = random.choices(current_state, weights=normalized_probabilities, k=1)[0]
        # get the second part of the tuple which is the schedule itself
        schedule = scheduler.generate_schedule(selected_schedule[1])
    
    elif transition == 'Crossover':
        # Select two schedules for crossover based on the calculated probabilities
        selected_schedules = random.choices(current_state, weights=normalized_probabilities, k=2)
        # get the second part of the tuple which is the schedule itself
        schedule = scheduler.generate_schedule(selected_schedules[0][1], selected_schedule[1][1])

    return schedule

# we needa keep in mind that we can get repeat schedules and we don't want that!
    def processSchedules(self, limitOfSchedules):
        """
        Main function to manage schedules. Determines the operation to perform and executes it.

        Steps:
        1. Define the maximum allowable number of schedules.
        2. Determine the appropriate operation using assignNum.
        3. Execute the operation using fwert.
        """
        
        while(1):
            # Determine the operation to perform
            num = self.assignNum(limitOfSchedules)
            # Executes the determined operation ONCE
            self.fwert(num)
        




# Example usage
if __name__ == "__main__":
    # Create a ScheduleProcessor instance
    processor = ScheduleProcessor()
    
    # Exampel Maximum number of schedules allowed in the heap
    limitOfSchedules = 30
    
    # Process schedules
    processor.processSchedules(limitOfSchedules)

