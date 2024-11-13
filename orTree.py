"""
Or-Tree Class for use in Genetic Algorithm/Or-Tree Search Algorithm
Class written by Emily Kiddle
Contributors:
Group members****
"""
class OrTreeScheduler:
    def __init__(self, game_slots, practice_slots, env):
        """
        Initializes the orTreeScheduler with the abstracted game slots and any necessary information found in the env
        variable.
        """
        self.game_slots = game_slots
        self.practice_slots = practice_slots
        self.env = env
        self.fringe = []

    def altern(self, pr):
        alternatives = []
        for i, slot in enumerate(pr):
            # go through until you get to an unscheduled slot
            if slot == '*': 
                # Then, we check if we should schedule this into a game slot or practice slot
                if i < self.game_slots:
                    # create an altern for every game_slot to be scheduled here
                    for slot in self.game_slots:
                        newPr = pr[:i] + slot + pr[i+1:]
                        alternatives.append(newPr)
                else:
                    for slot in self.practice_slots:
                        newPr = pr[:i] + slot + pr[i+1:]
                        alternatives.append(newPr)
            self.fringe.append(alternatives)
            return True
        return False # there were no empty schedules
                   

    def ftrans(self, state):
        pr = state[0]
        # if constraints are violated
        if (not self.constr(state[0])):
            return((pr, 'no'))
        # if schedule is complete
        if ('*' not in pr):
            return((pr, 'yes'))
        # else, we generate new branches from the current pr using the altern function
        self.altern(pr)
        return (pr, '?')
    
    def fleaf_random(self, state):
        return 0
    
    def fleaf_mutation(self, state, index):
        return 0
    
    def fleaf_crossover(self, state):
        return 0

    def constr(self, schedule):
        """
        Function that evaluates the constraints. Returns True if no constraints are violated, False otherwise
        """
        return True

    def search(self, pr0):
        """
        Main search algorithm of the orTree. Takes the current state (you would like to start a search on) and returns
        a completed schedule, or an empty schedule if there are no solutions
        """
        state = (pr0, '?') # tuple of pr,sol_entry
        # Check if the goal state has been reached (pr,yes) or all (pr,no)
        while state[1] != 'yes':
            if state[1] == 'no': # our node is a no. go. We can assume all nodes are no too then, if not fleaf would've chosen another
                return []
            else:
                # (do fleaf at end because we initially have the start state (no leafs to choose from))
                # so, we do ftrans (and take a transition)
                state = self.ftrans(state)
                # check to see if our ftrans changed our state to yes
                if state[1] == 'yes':
                    break

                # Now, we will select a new leaf

        # return the completed schedule (pr') or empty list if failed.
        return state[0]


    def generate_schedule(self, tempA = [], tempB=[]):
        """
        creates a schedule from the orTree search algorithm with 0,1, or 2 input templates. This is included in the env variable
        and defaults to an empty list, meaning no templates. 
        0 templates = random orTree, 1 template = mutation orTree, 2 templates = crossover orTree
        """
        self.tempA = tempA
        self.tempB = tempB
        self.fringe = []

        # create start_state, and set preassignments right away.
        pr0 = ['*'] * (len(self.game_slots)+len(self.practice_slots))
        schedule = self.search(pr0)

        # return the found schedule
        return schedule
    