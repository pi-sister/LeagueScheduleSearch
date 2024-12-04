
# import list
import random
import pandas as pd

from constr import Constr
from environment import Environment as env
import schedule

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
        # print(f"ENV: {env}")
        # print(f"TYPE: {env.game_slots  }")
        # print(f'help: {dir(env)}')

        # populate local variables
        self.game_slots = env.game_slots
        self.practice_slots = env.practice_slots
        self.events = env.events
        # print(f"eventsss: {self.events}")

        # Filter the events DataFrame to include only games
        self.games = env.events[env.events['Type']=='G']
        # print("\nEvents\n", env.events)
        # print("\ngames\n ", self.games)
        self.constraints = constraints
        self.env = env
        self.length = env.event_length()
        self.fringe = []
        self.starter_slots = env.preassigned_slots
        
        
        # ok so right now it's seeing if we used up all the games, and starts assigning them one by one
        # what we want to do instead is assign a value based on the score of a game fulfilling certain hardconstraints
        # if it fulfills more hard constraints (meaning that it has more hard constraints), it will have a higher score
        # we can call a score function that does all the math
        self.df_with_scores = self.score(self.events)
        print(f"SCORES: {self.df_with_scores['Score']}")
        
    # TODO: @Emily 

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
                self.df_with_scores = self.score(self.df_with_scores)
                print(f"SCORE!!!S: {self.df_with_scores['Score']}")

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
       # ok so instead of doing what's down below, we want to first get the game/practice with the highest score
        if not self.df_with_scores_changing.empty:
            lowest_score = self.df_with_scores_changing['Score'].min() # we get the highest score here
            min_row_label = self.df_with_scores_changing['Score'].idxmin()  # we get the corresponding highest score's label 
            min_row = self.df_with_scores_changing.loc[[self.df_with_scores_changing['Score'].idxmin()]] # here it gets the whole row of that highest scroe
            idx = self.events.index.get_loc(min_row_label) # here it gets the index of the highest score
            print(f'max score : {lowest_score}')
            print(f'max score label : {min_row_label}')
            print(f'max row  : {min_row}')
            print(f'idx  : {idx}')
            # ok, now we have the max label and score, we need to say that we're first adding that to our schedule
            if (min_row['Type'].iloc[0] == "G"):
                for game_slot in self.game_slots.index:
                    new_pr = pr[:idx] + [game_slot] + pr[idx+1:]
                    print(f'new pr  : {new_pr}')
                    # # Push into heap with the '*' count as priority
                    if not mut:
                        self.fringe.append((-index, (new_pr,'?')))
                    else:
                        self.fringe.append((1, (new_pr,'?')))
                    # we also have to update the old dataset so we can do the calculation again
                    self.df_with_scores.loc[min_row_label, 'Part_assign'] = game_slot
                # now we gotta remove the highest index game we just did so it doesn't slot it again
                self.df_with_scores_changing = self.df_with_scores_changing.drop(min_row_label)


            else:
                for practice_slot in self.practice_slots.index:
                    new_pr = pr[:idx] + [practice_slot] + pr[idx+1:]
                    print(f'new pr  : {new_pr}')
                    # # Push into heap with the '*' count as priority
                    if not mut:
                        self.fringe.append((-index, (new_pr,'?')))
                    else:
                        self.fringe.append((1, (new_pr,'?')))
                    # we also have to update the old dataset so we can do the calculation again
                    self.df_with_scores.loc[min_row_label, 'Part_assign'] = practice_slot
                # now we gotta remove the highest index game we just did so it doesn't slot it again
                self.df_with_scores_changing = self.df_with_scores_changing.drop(min_row_label)
        # else:
            # return pr
        # Create a new schedule with this slot
        # new_pr = pr[:idx] + [max_row] + pr[idx+1:]
        # print(f'new pr  : {new_pr}')
        # # # Push into heap with the '*' count as priority
        # if not mut:
        #     self.fringe.append((-index, (new_pr,'?')))
        # else:
        #     self.fringe.append((1, (new_pr,'?')))
        
        #emily's code
        # # Check if it's in the range for game slots
        # if index < len(self.games):
        #     for game_slot in self.game_slots.index:
        #         # Create a new schedule with this game slot
        #         new_pr = pr[:index] + [game_slot] + pr[index+1:]
        #         print(f'new pr  : {new_pr}')

        #         # Push into heap with the '*' count as priority
        #         if not mut:
        #             self.fringe.append((-index, (new_pr,'?')))
        #         else:
        #             self.fringe.append((1, (new_pr,'?')))
        # # # Otherwise, it must be a practice slot
        # else:
        #     for practice_slot in self.practice_slots.index:
        #         # Create a new schedule with this practice slot
        #         new_pr = pr[:index] + [practice_slot] + pr[index+1:]
        #         # Push into heap with the '*' count as priority
        #         if not mut:
        #             self.fringe.append((-index, (new_pr,'?')))
        #         else:
        #             self.fringe.append((1, (new_pr,'?')))

    def starterSlot(self, row):
        if row['Div'] == "09" and row['Type'] == "G":
            # using reverse here to say that the smallest number of slots available should hold the greatest weight, adding 1 just to make the resulting value bigger
            starterValue = 4 
        elif row['Div'] == "09" and row['Type'] == "P":
            starterValue = 6 
        elif row['Div'] != "09" and row['Type'] == "G":
            starterValue = 20 
        elif row['Div'] != "09" and row['Type'] == "P":
            starterValue = 31 
        else:
            starterValue = 0
        return starterValue
    def disallowedSlots(self, row):
        # so on every row (every specifc game), we get that games name and we also have a list of invalid assignments. if a game can go into a specifc invalid time slot, we add a penalty
        row_label = row['Label']
        if row_label in row['Unwanted']:
            value = 1
        else:
            value = 0
        return  value
    def teamBusy(self, row, df): 
        sameTeam = df[
            (df['League'] == row['League']) &
            (df['Div'] == row['Div']) &
            (df['Tier'] == row['Tier'])]
        # so we need to add 1 if our practice is schdeuled on moday anytime or on tuesday (not including 9am, 15:00,and 18:00)
        excluded_times = ['TU9:00', 'TU15:00', 'TU18:00']
        teamBusyValue = 0  

        # we need to add 2 if the game is assigned on monday or tuesday at 9, 15:00, 18:00,, or there's a friday practice, or tuesday game
        # we first need to get the team, it can either be w/o prc or with
        # CMSA U10T3 DIV 01 PRC 01
        # CMSA U10T3 DIV 01
        # so we need compare that all three r the same
        # ok, we get a df with the same team
        # we need to check if the team has a practice on monday
        # Split into game DataFrame and practice DataFrame
        games_df = sameTeam[sameTeam['Type'] == 'G']
        practices_df = sameTeam[sameTeam['Type'] == 'P']
        if not practices_df.empty:
            for _, row in practices_df.iterrows():
                if row['Part_assign'] == "MO9:00":
                    teamBusyValue += 1
                if row['Part_assign'].startswith('TU') and row['Part_assign'] not in excluded_times:
                    teamBusyValue += 1
                if row['Part_assign'] in excluded_times:
                    teamBusyValue += 2
                if row['Part_assign'].startswith("FR"):
                    teamBusyValue += 2
        if not games_df.empty:
            for _, row in games_df.iterrows():
                if row['Part_assign'].startswith("MO") or row['Part_assign'].startswith("MO"):
                    teamBusyValue += 2
        return teamBusyValue

    def tierBusy(self, row, sameTeam):
        # we will be given a game, only games, and have to see if it's in the tiers U15,16,17,19 and if it is and it is assinged, then we add a penalty
        games_df = sameTeam[sameTeam['Type'] == 'G']
        tierBusyValue = 0  

        if not games_df.empty:
            for _, row in games_df.iterrows():
                if row['Tier'].startswith("U15") or row['Tier'].startswith("U16") or row['Tier'].startswith("U17") or row['Tier'].startswith("U19"):
                    tierBusyValue += 1
        return tierBusyValue

    def timeConflicts(self, row):
        # ok, so we get the not compatible column and see if there are any times our current row (div) shares with any of the divs in the column
        # to do that we first get our row and say look at the non-compatible column!
        otherDivison = row['Incompatible']
        timeConflictValue = 0

        # then we need to go back to our df and check each value in that column to see if it has the same time slots for either game or practice
        # got to that label in the df
        # otherDivTiming = totalSlots[otherDivison, 'Part_assign']
        if otherDivison != []:
            timeConflictValue += 1
            
        return timeConflictValue


    
    def score(self, givenDataset):
        # Reset the index, moving index labels to a new column
        df_reset = givenDataset.reset_index()
        # Rename the new column for clarity (optional)
        df_reset = df_reset.rename(columns={'index': 'Label'})
        
        scores = []  # Initialize an empty list to store scores

        for _, row in df_reset.iterrows():
            starterVal = self.starterSlot(row)
            disallowedSlotsVal = self.disallowedSlots(row)
            teamBusyVal = self.teamBusy(row, df_reset)
            tierBusyVal = self.tierBusy(row, df_reset)
            timeConflictsVal = self.timeConflicts(row)
            scoreVal = starterVal - disallowedSlotsVal - teamBusyVal - tierBusyVal - timeConflictsVal
            scores.append(scoreVal)  # Append the score to the list
        
        # Add the scores as a new column to the DataFrame
        givenDataset['Score'] = scores
        return givenDataset


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



    def constr(self, sched_list: list):
        """
        Function that evaluates the constraints. Returns True if no constraints are violated, False otherwise

            Parameters:
                schedule (list): partially or fully defined schedule

        """
        tempSched = schedule.Schedule.list_to_schedule(sched_list, self.env)

  
        self.constraints.reset_slots()

        for event_id, event_details in tempSched:
            if event_details["Assigned"] == "*":
                continue

            if not tempSched.max_exceeded(event_details["Assigned"], event_details["Type"]):
                print("Failed Max")
                return False

            if (event_details["Assigned"] != event_details["Part_assign"]) and (event_details["Part_assign"] != "*"):
                print("Failed Part_assign")
                return False
            
            if event_details["Assigned"] in event_details['Unwanted']:
                print("Failed Unwanted")
                return False

            if not self.constraints.incompatible(tempSched.get_Assignments(), event_details["Incompatible"], event_details["Type"], event_details["Assigned"], event_id):
                print("Failed Incompatible")
                return False
            
            if not self.constraints.check_evening_div(event_details["Assigned"][2:], event_details["Div"]):
                print("Failed Evening Div")
                return False

            if not self.constraints.check_assign(tempSched.get_Assignments(), event_details["Tier"], event_details["Assigned"], event_details["Corresp_game"],"regcheck"):
                print("Failed U15-U19 Check")
                return False

            if self.constraints.special_events:
                if not self.constraints.check_assign(tempSched.get_Assignments(), event_details["Tier"], event_details["Assigned"], event_details["Corresp_game"],"specialcheck"):
                    print("Failed Special Check")
                    return False   
                     
            if event_details["Type"] == "P" and ((event_details["Tier"] != 'U13T1S') or (event_details["Tier"] != 'U12T1S')):
                if not self.constraints.check_assign(tempSched.get_Assignments(), event_details["Tier"], event_details["Assigned"], event_details["Corresp_game"],"pcheck"):
                    print("Failed Practice Check")
                    return False

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
        self.df_with_scores_changing = self.df_with_scores
        self.tempA = tempA
        self.tempB = tempB
        self.fringe = []

        # create start_state, and set preassignments right away.
        pr0 = self.starter_slots

        if (tempA or tempB) and not (tempA and tempB):

            self.randomNumbers = list(range(self.length))
            sched_list = self.mutate(pr0) 
        else:
            sched_list = self.search(pr0)

        # return the found schedule
        print(f'Final Sched: {sched_list}')
        if tempA and not sched_list:
            print("from mutation or crossover")
            return None
        return schedule.Schedule.list_to_schedule(sched_list, self.env)


# if __name__ == "__main__":
#     # Load CSV with the first column as the index
#     # env = env('Jamie copy.txt', [1,0,1,0,10,10,10,10], verbose = 1)
#     env = env('example.txt', [1,0,1,0,10,10,10,10], verbose = 1)


#     # print(f'Preassignments: {env.preassigned_slots}')
#     print(f'Events:\n {env.events.columns}')
#     print(f'Practiceslots:\n {env.practice_slots}')
#     print(f'Gameslots:\n {env.game_slots}')

#     constraints = Constr(env)

#     scheduler = OrTreeScheduler(constraints, env)

#     # schedule1 = scheduler.generate_schedule().assigned
#     schedule1 = scheduler.generate_schedule()
#     print("schedule 1", schedule1)

    #schedule2 = scheduler.generate_schedule(schedule1).assigned
    #print("schedule 2", schedule2)

    # schedule3 = scheduler.generate_schedule().assigned
    # schedule3 = scheduler.generate_schedule()
    # print("schedule 3", schedule3)

    # schedule4 = scheduler.generate_schedule(schedule1, schedule3).assigned
    # schedule4 = scheduler.generate_schedule(schedule1, schedule3)
    # print("schedule 4", schedule4)