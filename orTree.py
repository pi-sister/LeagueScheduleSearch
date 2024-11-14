
# import list
import heapq
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
        """
        self.game_slots = game_slots
        self.practice_slots = practice_slots
        self.env = env
        self.fringe = []


    def altern(self, pr):
        """
        Altern function generates new prs from the current pr passed in. And it puts all of the prs
        into the fringe, which is maintained in a min heap (number of '*'s is minimal.)
        """
        for i, slot in enumerate(pr):
            # Find the first unscheduled slot
            if slot == '*': 
                return self.pushFringe(i, pr)
        # Return False if no unscheduled slots were found
        return False

    def pushFringe(self, index, pr):
        # Check if it's in the range for game slots
        if index < len(self.game_slots):
            for game_slot in self.game_slots:
                # Create a new schedule with this game slot
                new_pr = pr[:index] + [game_slot] + pr[index+1:]
                # Push into heap with the '*' count as priority
                heapq.heappush(self.fringe, (new_pr.count('*'), (new_pr,'?')))
        # Otherwise, it must be a practice slot
        else:
            for practice_slot in self.practice_slots:
                # Create a new schedule with this practice slot
                new_pr = pr[:index] + [practice_slot] + pr[index+1:]
                # Push into heap with the '*' count as priority
                heapq.heappush(self.fringe, (new_pr.count('*'), (new_pr,'?')))
        return True
                   
    def ftrans(self, state):
        """
        transition function that selects an action to complete. This either changes sol_entry to yes, no, or
        calls the altern function that produces new branches/leaves of the tree. Returns the changed state (if it was
        changed. If not, juts returns how it was before '?')
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
        """
        # Start looping until a leaf is found (there are some situations where a leaf won't be found and it will have to start
        # looping.)
        newState = None
        while newState == None and self.fringe:
            # Step 1: Find the highest index of the last scheduled slot in each schedule (just before the first '*')
            def find_max_scheduled_index(schedule):
                try:
                    return schedule.index('*') - 1  # The index just before the first '*'
                except ValueError:
                    return len(schedule) - 1  # If no '*', return the last index

            # Step 2: Find the maximum of these indices across all schedules in the fringe
            max_scheduled_index = max(self.fringe, key=lambda x: find_max_scheduled_index(x[1]))[1].index('*') - 1

            # Step 3: Filter to get all nodes that have this max scheduled index
            max_scheduled_nodes = [item for item in self.fringe if find_max_scheduled_index(item[1]) == max_scheduled_index]

            # Check how many template we have. 2 templates = crossover. 1 template = mutation. 0 templates = random
            if self.tempA and self.tempB:
                newState= self.fleaf_crossover(max_scheduled_nodes)
            elif self.tempA or self.tempB:
                newState = self.fleaf_mutation(max_scheduled_nodes)
            else:
                newState = self.fleaf_random(max_scheduled_nodes)

        return newState


    def fleaf_random(self, leaves):
        """
        fleaf - random variation. This function selects a random leaf out of the deepest leaves.
        """
        # Randomly select one node from the inputed list
        selected = random.choice(leaves)
    
        # Remove the selected node from self.fringe (since it should still be in there)
        self.fringe.remove(selected)
    
        # Return the selected state
        return selected[1]


    def fleaf_mutation(self, leaves, index):
        """
        Selects a leaf from a list of leaves where the slot at `index` matches `match_value`.

        Parameters:
        leaves (list): List of leaves, where each leaf is a tuple (schedule, '?')
        index (int): Index position to check in the schedule

        Returns:
        tuple: The first leaf that matches the condition, or a leaf that doesn't match if at the random index point
        """
        selected = None
        for leaf in leaves:
            num, pr = leaf
            schedule, status = pr
            if schedule[index] == self.tempA[index]:
                if index == self.mutate:
                    leaves.remove(leaf)
                    selected = random.choice(leaves)
                else:
                    selected = leaf
                self.fringe.remove(selected)
                break
        return selected[1]


    def fleaf_crossover(self, leaves, index):
        """
        fleaf - crossover variation. This function randomly selects a leaf so long as it matches 1 of the 2
        templates.
        """
        choices = []
        selected = None
        for leaf in leaves:
            num, pr = leaf
            schedule, status = pr
            if schedule[index] == self.tempA[index] or schedule[index] == self.tempB[index]:
                choices.append(leaf)

        selected = random.choice(choices)
        self.fringe.remove(selected)
        return selected[1]


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
                # Check if there are no more leaf nodes, indicating that there are no solutions. So, we should return an empty list.
                if not self.fringe:
                    return []

                # Now, we will select a new leaf
                state = self.fleaf_base()

        # return the completed schedule (pr') or empty list if failed.
        return state[0]

    def mutate(self, pr0):
        """
        Base mutation function.
        """
        rand = random.choice(self.randomNumbers)
        self.randomNumbers.remove(rand)
        self.pushFringe(rand, pr0)
        prMut = pr0
        while self.fringe:
            selected = random.choice(self.fringe)
            self.fringe.remove(selected)
            num, state = selected
            pr, sol_entry = state
            if pr[rand] != self.tempA[rand]:
                prMut = pr
                break
        return self.search(prMut)


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
        length = (len(self.game_slots)+len(self.practice_slots))
        pr0 = ['*'] * length

        if (tempA or tempB) and not (tempA and tempB):
            self.randomNumbers = list(range(length))
            schedule = self.mutate(pr0)      
        else:
            schedule = self.search(pr0)

        # return the found schedule
        return schedule
    