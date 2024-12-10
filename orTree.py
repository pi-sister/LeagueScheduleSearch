
# import list
import random
import pandas as pd

from constr import Constr
from environment import Environment as env
import schedule
import time

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
        
    Attributes:
        game_slots (list): List of abstracted game slots.
        practice_slots (list): List of abstracted practice slots.
        games (int): Number of games.
        constraints (Constr): Constraints object to evaluate the schedule.
        env (Environment): Environment object containing necessary information.
        length (int): Length of the event.
        fringe (list): List to maintain the fringe in a min heap.
        starter_slots (list): List of preassigned slots. 
        tempA (list): Template A for the schedule.
        tempB (list): Template B for the schedule.
        randomNumbers (list): List of random numbers for mutation.
        
    Methods:
        __init__(constraints, env):
            Initializes the OrTreeScheduler with the abstracted game slots and any necessary information found in the env variable.
        altern(pr):
            Generates new prs from the current pr passed in and puts all of the prs into the fringe, which is maintained in a min heap.
        pushFringe(index, pr, mut=False):
            Pushes to the fringe all possible state combinations. Used in part of altern function.
        ftrans(state):
            Transition function that selects an action to complete. Returns the changed state if it was changed, otherwise returns the state as it was before.
        fleaf_base():
            Base fleaf function that finds the highest index value of the next schedule to be scheduled and filters to choose between the deepest leaves.
        fleaf_random(leaves):
            Selects a random leaf out of the deepest leaves.
        fleaf_template(leaves, index):
            Follows 1 or 2 templates in the schedule and removes all other leaves that are not the same as the template(s).
        constr(schedule):
            Evaluates the constraints. Returns True if no constraints are violated, False otherwise.
        search(pr0):
            Main search algorithm of the OrTree. Takes the current state and returns a completed schedule, or an empty schedule if there are no solutions.
        mutate(pr0):
            Base mutation function that starts by randomly mutating 1 schedule and starting a search on it.
        generate_schedule(tempA=[], tempB=[]):
            Creates a schedule from the OrTree search algorithm with 0, 1, or 2 input templates. Defaults to an empty list, meaning no templates.
    """
    def __init__(self, constraints, env):
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
        self.game_slots = env.game_slots

        self.practice_slots = env.practice_slots

        self.events = env.events

        # Filter the events DataFrame to include only games
        self.games = env.events[env.events['Type']=='G']
        self.constraints = constraints
        self.env = env
        self.length = env.event_length()
        self.bad_guys = None
        self.fringe = []
        self.starter_slots = env.preassigned_slots
        self.fitness = 10000000
        
        # Populate a new database with scores for each event to set the order in which games/practices will be populated
        self.df_with_scores = self.score(self.events)
        self.df_with_scores = self.df_with_scores.sort_values(by='Score', ascending=True)
    
    def starterSlot(self, row):
        """
        Determine the starter slot value based on the division and type of the row.
        Args:
            row (dict): A dictionary containing 'Div' and 'Type' keys.
        Returns:
            int: The starter slot value based on the following conditions:
                - If 'Div' starts with "9" and 'Type' is "G", return 4.
                - If 'Div' starts with "9" and 'Type' is "P", return 6.
                - If 'Div' does not start with "9" and 'Type' is "G", return 20.
                - If 'Div' does not start with "9" and 'Type' is "P", return 31.
        """
        starterValue = 0

        if row['Div'].startswith("9") and row['Type'] == "G":
            starterValue = 4 

        elif row['Div'].startswith("9") and row['Type'] == "P":
            starterValue = 6 

        elif not row['Div'].startswith("9") and row['Type'] == "G":
            starterValue = 20 
        elif not row['Div'].startswith("9") and row['Type'] == "P":
            starterValue = 31

        return starterValue
    

    def disallowedSlots(self, row):
        """
        Calculate the number of disallowed time slots for a given game.
        Args:
            row (dict): A dictionary representing a game, which contains an 'Unwanted' key.
                        The 'Unwanted' key holds a list of time slots that are not allowed for this game.
        Returns:
            int: The number of disallowed time slots for the given game.
        """
        # so on every row (every specifc game), we get that games name and we also have a list of invalid assignments. if a game can go into a specifc invalid time slot, we add a penalty
        value = 0
        for _ in row['Unwanted']:
            value += 1
        return value
    

    def tierBusy(self, row, sameTeam):
        """
        Calculate the penalty value for a team's busy schedule in specific tiers.
        This method checks if the given games are in the tiers U15, U16, U17, or U19.
        If a game is in one of these tiers, it increments the penalty value.
        Args:
            row (pandas.Series): A row from the DataFrame, representing a game.
            sameTeam (pandas.DataFrame): DataFrame containing games for the same team.
        Returns:
            int: The penalty value based on the number of games in the specified tiers.
        """
        # we will be given a game, only games, and have to see if it's in the tiers U15,16,17,19 and if it is, then we add a penalty
        games_df = sameTeam[sameTeam['Type'] == 'G']
        tierBusyValue = 0  

        if not games_df.empty:
            for _, row in games_df.iterrows():
                if row['Tier'].startswith(('U15', 'U16', 'U17', 'U19')):
                    tierBusyValue += 1

        return tierBusyValue

    def timeConflicts(self, row):
        """
        Calculate the time conflict penalty for a given row.
        This method checks if the game/practice has any incompatible values and 
        adds a penalty for each incompatible value found.
        Args:
            row (dict): A dictionary representing a row with a key 'Incompatible' 
                        that contains a list of incompatible values.
        Returns:
            int: The total time conflict penalty based on the number of incompatible values.
        """
        # we need to see if the game/practice has any non-comptiable values, if they do, add a penalty
        otherDivison = row['Incompatible']
        timeConflictValue = 0
        if otherDivison != []:
            for value in row['Incompatible']:
                timeConflictValue += 1
        return timeConflictValue


    def soft(self, row):
        softCol = row['Preference']
        softValue = 0
        if softCol != []:
            for index, value in enumerate(row['Preference']):
                softValue = 1/(index+1)
        return softValue
    def score(self, givenDataset):
        """
        Calculate the score of a game or practice to determine its priority in scheduling

        Parameters:
            A data frame containing games and practices to be scored.

        Returns:
            A score column appended to the dataframe
        """
        # add a new column with the index name just so it's easier to compare something later
        df_reset = givenDataset.reset_index().rename(columns={'index': 'Label'})

        scores = []  # Initialize an empty list to store scores

        for _, row in df_reset.iterrows():
            starterVal = self.starterSlot(row)
            disallowedSlotsVal = self.disallowedSlots(row)
            tierBusyVal = self.tierBusy(row, df_reset)
            timeConflictsVal = self.timeConflicts(row) #math
            softVal = self.soft(row)
            scoreVal = starterVal - disallowedSlotsVal - tierBusyVal - timeConflictsVal - softVal
            scores.append(scoreVal)  # Append the score to the list
        
        # Add the scores as a new column to the DataFrame
        givenDataset['Score'] = scores
        return givenDataset


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
        # regenerate and sort scores to always pick the lowest priority first
        for index in range(len(self.df_with_scores)):
            if self.pushFringe(index, pr):
                return True
        return False


    def pushFringe(self, index, pr, mut = False):
        """
        Function to push to the fringe all possible state combinations. Used in part of altern function

            Parameters:
                index (Int): the index in the schedule to schedule a slot into
                pr (list): the current schedule being modified
        """
        if not mut:
            # we get the score value that we would like to iterate over using the index decided in altern
            min_row_label = self.df_with_scores['Score'].index[index]  # we get the corresponding lowest score's label 
            idx = self.events.index.get_loc(min_row_label) # here it gets the index of the lowest score
            assigned_indices = {i for i, slot in enumerate(pr) if slot != '*'} # make sure we haven't already assigned this game

            if idx in assigned_indices: # Go back to altern and try again.
                return False
        else:
            idx = index

        # Grab the star count that will be stored for sorting purposes of the leaves. (Lowest star count chosen first)    
        star_count = pr.count('*')

        # Grab the possible slots that this particular schedule can be scheduled to
        slots = self.constr(pr, self.events.iloc[idx])
        for slot in slots:
            new_pr = pr[:idx] + [slot] + pr[idx+1:]
            #calcuates the currecnt scheudels eval
            new_sched = schedule.Schedule.list_to_schedule(new_pr, self.env)
            new_fitness = new_sched.set_Eval()

            #now we need a way to store the best eval
            if new_fitness < self.fitness:
                self.fitness = new_fitness
                # add it to the fringe
            else:
                continue
            # Push into heap with the '*' count as priority
            if not mut:
                push = False
                if self.tempA and (self.bad_guys is not None) and not self.bad_guys.empty:
                    # ok so we have a df with our worst columns, we just gotta see if the label we're working with
                    #ok so we have our label of what we're working with - min_row_lavel
                    # now we gotta check if min_row_label is in the worst eval df
                    # if it is, change it to the rando
                    # if it isn't don't change it
                    if min_row_label in self.bad_guys['Label'].values:
                        if self.tempB and self.tempB[idx] == slot:
                            push = True
                        else:
                            push = False
                    else:
                        if self.tempA and self.tempA[idx] == slot:
                            push = True
                        else:
                            push = False
                else:
                    push = True
                if push:
                    self.fringe.append((star_count, (new_pr,'?')))
                
            else:
                if self.tempA[idx] == slot:
                    continue
                self.fringe.append((star_count, (new_pr,'?')))

        return True


    def ftrans(self, state):
        """
        transition function that selects an action to complete. This either changes sol_entry to yes, no, or
        calls the altern function that produces new branches/leaves of the tree. Returns the changed state (if it was
        changed. If not, juts returns how it was before '?')

            Parameters:
                state (schedule, sol_entry): The chosen state from fleaf that will go through a transition function.
        """
        pr = state[0]
        # We no longer check for constraints here since altern will not push invalid schedules
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
            if self.tempA and (index < len(self.tempA[index])) and (schedule[index] == self.tempA[index]):
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


    def constr(self, sched_list: list, curr_row):
        """
        Function that evaluates the constraints. Returns True if no constraints are violated, False otherwise

            Parameters:
                schedule (list): partially or fully defined schedule
                game or practice row:   the row in the dataframe corresponding to the game that wants to be scheduled
            
            Returns:
                A list of slots that the specified game or practice can be scheduled into.

        """
        tempSched = schedule.Schedule.list_to_schedule(sched_list, self.env)

        # return all not maxed out games / practices
        available_slots = tempSched.return_not_maxed(curr_row['Type']).index.to_list()

        if curr_row['Tier'].startswith(('U13T1S', 'U12T1S')):
            if 'TU18:00' in available_slots:
                return ['TU18:00']
            else:
                #print("Special Practice Max")
                return []
            
        # check if we need to worry about incompatible
        bad_slots = []
        if curr_row["Incompatible"]:
            #print("Has Incompatible")
            bad_slots.extend(self.constraints.another_incompatible(tempSched.get_scheduled(), curr_row['Incompatible'], curr_row['Type']))
            #print(f'Bad slots after incompatible: {bad_slots}')
        
        # check for unwanted
        if curr_row["Unwanted"]:
            #print("Has Unwanted")
            bad_slots.extend(curr_row['Unwanted'])
            #print(f'Bad slots after unwanted: {bad_slots}')

        # check if we need to worry about u15+
        if ((curr_row['Tier'].startswith(('U15', 'U16', 'U17','U19'))) and (curr_row['Type'] == 'G')):
            #print("Is U15+")
            bad_slots.extend(self.constraints.avoid_u15_plus(tempSched.get_scheduled()))
            #print(f'Bad slots after u15161719: {bad_slots}')
            
        # check for special practice
        if curr_row['Tier'].startswith(('U13T1', 'U12T1')):
            #print("Is Div 13/12")
            bad_slots.append('TU18:00')
            #print(f'Bad slots after u1312: {bad_slots}')
        
        # check for game/practice overlaps
        bad_slots.extend(self.constraints.check_game_practice_pair(tempSched.get_scheduled(), curr_row, curr_row['Type']))
        #print(f'Bad slots after game_practice check: {bad_slots}')

        # check for evening divs
        if curr_row['Div'].startswith('9'):
            #print("Is Evening Div")
            bad_slots.extend(self.constraints.another_check_evening_div(curr_row['Type']))
            #print(f"Bad slots after evening:\n {bad_slots}")
        
        available_slots = [slot for slot in available_slots if slot not in bad_slots]

        # print(f"available slots: {available_slots}\n")

        return available_slots


    def search(self, pr0):
        """
        Main search algorithm of the orTree. Takes the current state (you would like to start a search on) and returns
        a completed schedule, or an empty schedule if there are no solutions
        """
        state = (pr0, '?') # tuple of pr,sol_entry
        start_time = time.time()

        # Check if the goal state has been reached (pr,yes) or all (pr,no)
        while state[1] != 'yes':
            if state[1] == 'no': # our node is a no. go. We can assume all nodes are no too then, if not fleaf would've chosen another
                return []
            elif ((time.time() - start_time) > 3):
                state = (pr0, '?') # tuple of pr,sol_entry
                start_time = time.time()
                self.df_with_scores = self.score(self.events)
                self.df_with_scores = self.df_with_scores.sort_values(by='Score', ascending=True)

                self.tempA = []
                self.tempB = []
                self.fringe = []
                start_time = time.time()

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
            # so we get the index value of the game in the df, but we actually need the index of that game in the priority list
            rand_row_label = self.events.index[rand]  # we get the corresponding lowest score's label 
            idx = self.df_with_scores.index.get_loc(rand_row_label) # here it gets the index of the lowest score

            # populate the fringe with our initial nodes of all the possible combos our mutation can be.
            self.pushFringe(idx, pr0, True)

            prMut = None
            while not prMut:
                # select our starting node so that we can start our search. (this is pretty much an initial fleaf selection)
                if not self.fringe:
                    break
                selected = random.choice(self.fringe)
                self.fringe.remove(selected)
                num, state = selected
                pr, sol_entry = state
                # we make sure that our mutated schedule is not going to end up being the same as our initial schedule.
                if pr[rand] != self.tempA[rand]:
                    prMut = pr
            if prMut:
                schedule = self.search(prMut)
        return schedule


    def generate_schedule(self, tempA = [], tempB=[]):
        """
        creates a schedule from the orTree search algorithm with 0,1, or 2 input templates. This is included in the env variable
        and defaults to an empty list, meaning no templates. 
        0 templates = random orTree, 1 template = mutation orTree, 2 templates = crossover orTree
        
        Args:
            tempA (list, optional): Template A for the schedule. Defaults to an empty list.
            tempB (list, optional): Template B for the schedule. Defaults to an empty list.

        Returns:
            list: A valid schedule generated by the OrTree search algorithm.
        """
        # self.df_with_scores_changing = self.df_with_scores
        self.df_with_scores = self.score(self.events)
        self.df_with_scores = self.df_with_scores.sort_values(by='Score', ascending=True)
        # print(f"SCORES: {self.df_with_scores['Score']}")

        self.tempA = tempA
        self.tempB = tempB
        self.fringe = []

        # create start_state, and set preassignments right away.
        pr0 = self.starter_slots

        if (tempA or tempB) and not (tempA and tempB):

            self.randomNumbers = list(range(self.length))
            sched_list = self.mutate(pr0) 
            
        else:
            if (tempA and tempB):
                # tempSched = schedule.Schedule.list_to_schedule(tempA, self.env)
                self.bad_guys = tempA.get_top_offenders(0.1)
            sched_list = self.search(pr0)
        

        # return the found schedule
        print(f'Final Sched: {sched_list}')
        if tempA and not sched_list:
            print("from mutation or crossover")
            return None
        return schedule.Schedule.list_to_schedule(sched_list, self.env)


if __name__ == "__main__":
# #     # Load CSV with the first column as the index
    env = env('tests/CPSC433F24-LargeInput2.txt', [1,0,1,0,10,10,10,10], verbose = 1)

    constraints = Constr(env)

    scheduler = OrTreeScheduler(constraints, env)

    schedule1 = scheduler.generate_schedule().assigned

    schedule2 = scheduler.generate_schedule(schedule1).assigned

    print("\n\nschedule 1\n", schedule1)
    print("\n\nschedule 2\n", schedule2)

    # schedule3 = scheduler.generate_schedule().assigned
    # schedule3 = scheduler.generate_schedule()
    # print("schedule 3", schedule3)

    # schedule4 = scheduler.generate_schedule(schedule1, schedule3).assigned
    # schedule4 = scheduler.generate_schedule(schedule1, schedule3)
    # print("schedule 4", schedule4)