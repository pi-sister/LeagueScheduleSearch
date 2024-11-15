
# import list
import random

class OrTreeScheduler:
    """
    Or-Tree Class for Genetic Algorithm / Or-Tree Search Algorithm

    This class implements the Or-Tree model, used in conjunction with a Genetic Algorithm (GA) 
    to ensure that each solution in the GA contains a valid and complete schedule that 
    satisfies the hard constraints of a given scheduling problem.

    Author:
        Emily Kiddle

    Contributors:
        [List your group members here, e.g., Name1, Name2, Name3, etc.]
    """

    def __init__(self, game_slots, practice_slots, env):
        """
        Initializes the orTreeScheduler with the abstracted game slots and any necessary information found in the env
        variable. Put the contraint information in the env variable.

            Parameters:
                game_slots (list): list of abstracted game slots
                practice_slots (list): list of abstracted practice slots
                env (dictionary): contains all the necessary information to complete orTree (particularly
                for the constr function)
        """
        # populate local variables
        self.game_slots = game_slots
        self.practice_slots = practice_slots
        self.env = env
        self.length = (len(self.game_slots)+len(self.practice_slots))
        self.fringe = []


    def altern(self, pr):
        """
        Altern function generates new prs from the current pr passed in. And it puts all of the prs
        into the fringe, which is maintained in a min heap (number of '*'s is minimal.)

            Parameters:
                pr (list): the schedule to be extended. (We can assume sol_entry for the respective
                schedule in this list is ?)

            Returns:
                Boolean: True if altern function successfully applied, False otherwise
        """
        for i, slot in enumerate(pr):
            # Find the first unscheduled slot
            if slot == '*': 
                self.pushFringe(i, pr)
                return True
        # Return False if no unscheduled slots were found
        return False


    def pushFringe(self, index, pr, mut = False):
        """
        Function to push to the fringe all possible state combinations. Used in part of altern function

            Parameters:
                index (Int): the index in the schedule to schedule a slot into
                pr (list): the current schedule being modified
        """
        # Check if it's in the range for game slots
        if index < len(self.game_slots):
            for game_slot in self.game_slots:
                # Create a new schedule with this game slot
                new_pr = pr[:index] + [game_slot] + pr[index+1:]
                # Push into heap with the '*' count as priority
                if not mut:
                    self.fringe.append((-index, (new_pr,'?')))
                else:
                    self.fringe.append((1, (new_pr,'?')))
        # Otherwise, it must be a practice slot
        else:
            for practice_slot in self.practice_slots:
                # Create a new schedule with this practice slot
                new_pr = pr[:index] + [practice_slot] + pr[index+1:]
                # Push into heap with the '*' count as priority
                if not mut:
                    self.fringe.append((-index, (new_pr,'?')))
                else:
                    self.fringe.append((1, (new_pr,'?')))


    def ftrans(self, state):
        """
        transition function that selects an action to complete. This either changes sol_entry to yes, no, or
        calls the altern function that produces new branches/leaves of the tree. Returns the changed state (if it was
        changed. If not, juts returns how it was before '?')

            Parameters:
                state (schedule, sol_entry): The chosen state from fleaf that will go through a transition function.
        """
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
    

    def fleaf_base(self):
        """
        This is the base fleaf function because it has similarities between each of the fleaf variations. Here, I just
        find the highest index value of next schedule to be scheduled (#1 in min heap), and filter to choose between the
        deepest leaves in the more selective functions of fleaf_random, fleaf_mutation, or fleaf_crossover.

            Returns:
                tuple: (schedule, sol_entry)
        """
        # Start looping until a leaf is found (there are some situations where a leaf won't be found and it will have to start
        # looping.)
        newState = None
        while newState == None and self.fringe:
            self.fringe.sort(key=lambda x: x[0])
            # Step 2: Find the maximum of these indices across all schedules in the fringe (since it is heap, should be at front)
            max_scheduled_index = -self.fringe[0][0]

            # Step 3: Filter to get all nodes that have this max scheduled index
            max_scheduled_nodes = []
            for leaf in self.fringe:
                num, state = leaf
                if num == -max_scheduled_index:
                    max_scheduled_nodes.append(leaf)
            # Check how many template we have. 2 templates = crossover. 1 template = mutation. 0 templates = random
            if self.tempA or self.tempB:
                newState= self.fleaf_template(max_scheduled_nodes, max_scheduled_index)
            else:
                newState = self.fleaf_random(max_scheduled_nodes)
        return newState


    def fleaf_random(self, leaves):
        """
        fleaf - random variation. This function selects a random leaf out of the deepest leaves.

            Parameters:
                leaves (list): List of leaves, where each leaf is a tuple (num*, (schedule, sol_entry))

            Returns:
                tuple: (schedule, sol_entry)

        """
        # Randomly select one node from the inputed list
        selected = random.choice(leaves)
    
        # Remove the selected node from self.fringe (since it should still be in there)
        self.fringe.remove(selected)

        # Return the selected state
        return selected[1]


    def fleaf_template(self, leaves, index):
        """
        Follows 1 or 2 templates in the schedule, and removes all other leaves that are not the same as the template(s).

        Parameters:
            leaves (list): each leaf in the form (num*, (schedule, sol_entry))
            index: the index of the schedule to compare and choose to extend.

        Returns:  
            tuple: (schedule, sol_entry) of the state that is chosen by fleaf.
        """
        selected = None
        choices = []
        
        # Go through each leaf of the leaves that could possibly extend and single out the leaves that match the template.
        for leaf in leaves:
            num, state = leaf
            schedule, sol_entry = state
            if self.tempA and (index < len(self.tempA)) and (schedule[index] == self.tempA[index]):
                choices.append(leaf)
            elif self.tempB and (index < len(self.tempB)) and (schedule[index] == self.tempB[index]):
                choices.append(leaf)
            else:
                self.fringe.remove(leaf)
        
        # Check if choices is empty, and handle the fallback case
        if choices:
            selected = random.choice(choices)
            self.fringe.remove(selected)
        else:
            # Fallback: pick a random leaf from `leaves` if no matches were found
            print("Warning: No matching choices found at index", index)
            selected = random.choice(leaves)  # Fallback choice from all leaves

        # Remove the selected leaf from the fringe
        return selected[1]



    def constr(self, schedule):
        """
        Function that evaluates the constraints. Returns True if no constraints are violated, False otherwise

            Parameters:
                schedule (list): partially or fully defined schedule
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
                # Check if there are no more leaf nodes, indicating that there are no solutions. So, we should return an empty list.
                if not self.fringe:
                    return []

                # Now, we will select a new leaf
                state = self.fleaf_base()

        # return the completed schedule (pr') or empty list if failed.
        return state[0]


    def mutate(self, pr0):
        """
        Base mutation function. That starts by randomly mutating 1 schedule and starting a search on it.
        
            Parameters:
                pr0 (list): initial schedule to start mutation on. (usually empty schedule)

            Returns:
                schedule (list): empty list if not successful, or valid schedule generated

        """
        schedule = []
        # run until we get a completed schedule.
        while not schedule and self.randomNumbers:
            # Select the index that we will randomly mutate. remove it from the random list (we will come
            # back to here and select another random number if it doesn't produce any valid solutions.)
            rand = random.choice(self.randomNumbers)
            self.randomNumbers.remove(rand)
            # populate the fringe with our initial nodes of all the possible combos our mutation can be.
            self.pushFringe(rand, pr0, True)

            prMut = None
            while not prMut:
                # select our starting node so that we can start our search. (this is pretty much an initial fleaf selection)
                selected = random.choice(self.fringe)
                self.fringe.remove(selected)
                num, state = selected
                pr, sol_entry = state
                # we make sure that our mutated schedule is not going to end up being the same as our initial schedule.
                if pr[rand] != self.tempA[rand]:
                    prMut = pr
            schedule = self.search(prMut)
        return schedule


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
        pr0 = ['*'] * self.length

        if (tempA or tempB) and not (tempA and tempB):
            self.randomNumbers = list(range(self.length))
            schedule = self.mutate(pr0)      
        else:
            schedule = self.search(pr0)

        # return the found schedule
        return schedule


if __name__ == "__main__":
    game_slots = ['s1', 's2', 's3', 's4']
    practice_slots = ['s5', 's6', 's7', 's8']
    env = None

    scheduler = OrTreeScheduler(game_slots, practice_slots, env)

    schedule1 = scheduler.generate_schedule()
    print("schedule 1", schedule1)

    schedule3 = scheduler.generate_schedule()
    print("schedule 3", schedule3)

    schedule4 = scheduler.generate_schedule(schedule1, schedule3)
    print("schedule 4", schedule4)