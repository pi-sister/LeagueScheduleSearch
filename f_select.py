import random
import OrTreeScheduler
#import fitness or define it
def f_select(current_state):
    """
    f_select function to select a transition from the given state based on the probabilities.
    
    Args:
        current_state (list): List of schedules in the current state.
        fitness (function): A function to calculate the fitness value of a given schedule.
    
    Returns:
        a new schedule by calling either mutate or crossover?
    """
    # Calculate the total fitness of all schedules in the current state
    total_fitness = sum(fitness(schedule) for schedule in current_state)
    
    # Calculate the normalized probability for each schedule
    probabilities = [1 - (fitness(schedule) / total_fitness) for schedule in current_state]
    
    # Normalize the probabilities so they sum up to 1
    total_prob = sum(probabilities)
    normalized_probabilities = [prob / total_prob for prob in probabilities]
    
    # Choose between Mutation and Crossover with equal probability
    transition = random.choices(['Mutation', 'Crossover'], weights=[0.5, 0.5])[0]
    
    if transition == 'Mutation':
        # Select one schedule for mutation based on the calculated probabilities
        selected_schedule = random.choices(current_state, weights=normalized_probabilities, k=1)[0]
        schedule = OrTreeScheduler.generate_schedule([selected_schedule])
    
    elif transition == 'Crossover':
        # Select two schedules for crossover based on the calculated probabilities
        selected_schedules = random.choices(current_state, weights=normalized_probabilities, k=2)
        schedule = OrTreeScheduler.generate_schedule(selected_schedules)
        #these last two line for crossover and mutation will require some change
        # question to ask: why are we choosing the least fit schedules for mutation and crossover?
    return schedule
