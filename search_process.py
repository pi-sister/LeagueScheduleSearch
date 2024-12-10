
# imports for heap
from heapq import heapify, heappush, heappop, nlargest
import schedule
from orTree import OrTreeScheduler
import random
from tqdm import tqdm

#note: the OrtreeScheduler needs to give me a schedule of type Schedule so 
#that i can call set_Eval on it.



class ScheduleProcessor:
    """
    Class to manage and process a collection of schedules using a minimum heap data structure.
    
    This class supports three main operations:
    1. Adding a random schedule if the number of schedules is below a threshold. The random schedule will be generated using an Or-Tree function written by Emily
    2. Deleting the worst schedule if the number of schedules exceeds a predefined limit.
    3. Selecting the best schedule using a custom selection process called fselect. Fselect will be genearted by Kourosh

    """

    def __init__(self, scheduler):
        """
        Initializes the ScheduleProcessor with an empty heap.
        """
        # Create an empty heap for schedules
        self.heap = []
        heapify(self.heap)  # Convert the list into a valid heap structure
        self.scheduler = scheduler

    def chooseAction(self, limitOfSchedules):
        """
        Determines which operation to perform based on the number of schedules in the heap.

        Parameters:
            limitOfSchedules (int): The maximum allowable number of schedules in the heap.

        """
        populate_for = 50
        while populate_for > limitOfSchedules:
            populate_for -= 1
            print("too many schedules to be added")

        if len(self.heap) <= populate_for:
            self.fwert(0)  # add new random schedule
            print(self.heap)

        elif len(self.heap) > limitOfSchedules:
            self.fwert(1)  # Deletion operation
            print(self.heap)

        else:
            # self.fwert(0)  # add new random schedule

            newTuple = self.f_select(self.heap)
            # new schedule from mutation or crossover   
            if newTuple != 0:
                heappush(self.heap, newTuple)
            # if the generated tuple is new add it to the heap
            print(self.heap)




    def fwert(self, num):
        """
        Executes one of the three operations based on the input operation code.

        Parameters:
            num (int): Operation code determined by the assignNum method:
                0 -> Add a random schedule.
                1 -> Remove the worst schedule.
        """
        match num:
            case 0:  
                # generate a random new schedule
                newSchedule = self.scheduler.generate_schedule()
                # Get the value of it from set_Eval
                # if newSchedule == []:
                #     return "failed hard constraints"
                fitness = newSchedule.set_Eval()
                if newSchedule.eval is None:  # Adjust this to the actual attribute name
                    newSchedule.eval = fitness

                # Add the new schedule to the heap.
                # this is a max heap for effeciency so invert the value
                if all(newSchedule != existing_schedule for _, existing_schedule in self.heap):
                    heappush(self.heap, (-fitness, newSchedule))
              
                
            case 1:  
                # Remove the worst schedule
                # Remove the largest value in the heap (the worst schedule in a min heap).
                # Remove the largest value in the heap (the worst schedule in a min heap).
                heappop(self.heap)
                # Remove the worst by E_value

        


    def f_select(self, current_state):
        """
        f_select function to select a transition from the given state based on the probabilities.
    
        Args:
            current_state (list): List of all (fitness, schedules) in the current state.
            fitness (function): A function to calculate the fitness value of a given schedule.
    
        Returns:
            a new schedule tuple, or 0 for the same schedule
         """

        # getting the fitnesses and the schedules from the tuples
        # have to add the '-' because this is a max heap with inverted values
        fitnesses = [-fitness for fitness, schedule in current_state]
        schedules = [schedule for fitness, schedule in current_state]

        # Sort schedules and fitnesses together based on fitness (ascending order)
        sorted_state = sorted(current_state, key=lambda x: x[0])  # Sort by fitness (first element of the tuple)
        # Extract schedules after sorting
        sorted_schedules = [schedule for _, schedule in sorted_state]

        # Calculate the total fitness of all schedules in the current state
        total_fitness = sum(fitness for fitness in fitnesses)
        if total_fitness == 0:
            total_fitness = 1        
        
        # Calculate the normalized probability for each schedule
        probabilities = [1 - (fitness / total_fitness) for fitness in fitnesses]
        
    
        # Normalize the probabilities so they sum up to 1
        total_prob = sum(probabilities)
        normalized_probabilities = [prob / total_prob for prob in probabilities]
    
        # Choose between Mutation and Crossover with equal probability
        transition = random.choices(['Mutation', 'Crossover'], weights=[0.5, 0.5])[0]


        if transition == 'Mutation':
            # Select one schedule for mutation based on the calculated probabilities
            selected_schedule = random.choices(schedules, weights=normalized_probabilities, k=1)[0]
            schedule = self.scheduler.generate_schedule(selected_schedule)
    
        elif transition == 'Crossover':
            # select two schedules for crossover based on the calculated probabilities
            # selected_schedules = random.choices(schedules, weights=normalized_probabilities, k=2)
            selected_schedules = random.choices(schedules, weights=normalized_probabilities, k=1)
            best_schedule = sorted_schedules[0]
            schedule = self.scheduler.generate_schedule(best_schedule, selected_schedules[0])
            
        if not schedule:
            return 0
        # we dont want the same schedules in the heap
        if (schedule not in schedules):
            return (-schedule.set_Eval(), schedule)
        else:
            return 0


    def processSchedules(self, limitOfSchedules, iterNum):
        """
        Args: 
            limitOfSchedules: maximum allowable number of schedules.
            iterNum : number of iterations we want the loop to have
            as this is a suboptimal approach

        returns:
            a good enough answer, and the best one we have
        """
        
        for i in tqdm(range(iterNum)):
            self.chooseAction(limitOfSchedules)
        return max(self.heap)[1]
            
        




# # Example usage
# if __name__ == "__main__":
#     # Create a ScheduleProcessor instance
#     processor = ScheduleProcessor()
    
#     # Exampel Maximum number of schedules allowed in the heap
#     limitOfSchedules = 30
    
#     # Process schedules
#     processor.processSchedules(limitOfSchedules)

