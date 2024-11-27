import pandas as pd
from datetime import datetime

class Constr:
    """
    Constraint function

    This class implements the Constraints utilized in the Or-Tree Class 
    to ensure that any schedule or partially generated schedule in the Or-Tree algorithm 
    satisfies the hard constraints of a given scheduling problem.

    Author:
        Joanne Millard

    Contributors:
        [List your group members here, e.g., Name1, Name2, Name3, etc.]
    """

    def __init__(self, env):
        """
        Initializes the Constr with the abstracted constraints from the environment and
        each game/practice and their slots variable.

            Parameters:
                game_slots (dataframe): dataframe of each game slot and their attributes
                practice_slots (dataframe): dataframe of each practice slot and their attributes
                events (dataframe): dataframe of each game and practice plus their attributes
                env (dictionary): contains all the necessary information to complete orTree (particularly
                for the constr function) <- will eventually add as I go through each function
        """
        # populate local variables
        self.gslots_and_info = env.game_slots
        self.pslots_and_info = env.practice_slots
        self.events_and_info = env.events
        self.game_slot_num = env.game_slot_num()
        self.practice_slot_num = env.practice_slot_num()

        # evening
        self.evening = datetime.strptime("18:00", "%H:%M").time()

        # Variables for checking if max exceeded
        self.game_counter = []
        self.practice_counter = []

        self.game_slot_lookup = dict(zip(list(self.gslots_and_info.index), list(range(0,self.game_slot_num))))
        self.practice_slot_lookup = dict(zip(list(self.pslots_and_info.index), list(range(0,self.practice_slot_num))))
    
    # Change this so we accept one slot and create a counter that will retunr T or F
    # Find a way to reset
    def max_exceeded_reset(self):
        """
        resets the counter for number of games or practices in a slot to prepare
        for checking the constraints in a new schedule
        """
        # Lists to count the # of occurrences of each slot in the schedule
        self.game_counter = [0] * self.game_slot_num
        self.practice_counter = [0] * self.practice_slot_num


    def max_exceeded(self, slot, slot_type):
        """
        accepts a slot and the slot type (game or practice) as input and add it
        to the counter. If the slots counter exceeds the limit return False
        """
        if slot_type == 'G':
            self.game_counter[self.game_slot_lookup[slot]] += 1
            if self.game_counter[self.game_slot_lookup[slot]] > self.gslots_and_info.loc[slot,'Max']:
                return False
            
        if slot_type == 'P':
            self.practice_counter[self.practice_slot_lookup[slot]] += 1
            if self.practice_counter[self.practice_slot_lookup[slot]] > self.pslots_and_info.loc[slot,'Max']:
                return False
            
        return True
    
    def incompatible(self, event_index, schedule):
        # wrong : (
        if self.events_and_info.iloc[event_index]['Incompatible']:
            return True
          
        row_numbers = [self.events_and_info.index.get_loc(label) for label in self.events_and_info.iloc[event_index]['Incompatible']]

        slot_set = set()
        for i in row_numbers:
            if schedule[i] == "*":
                break
            
            if (schedule[i] in slot_set):
                return False
            
            slot_set.add(schedule[i])

        return True
    
    def check_assign(self, practice_index, schedule):
        practice = self.events_and_info.index[practice_index]

        if 'PRC' in practice:
            corresponding_game = practice.partition('PRC')[0]
        else:
            corresponding_game = practice.partition('OPN')[0]

        related_games = self.events_and_info[self.events_and_info.index.str.startswith(corresponding_game) & (self.events_and_info['Type'] == 'G')].index.tolist()

        game_indices = [self.events_and_info.index.get_loc(label) for label in related_games]

        for game_index in game_indices:
            if (schedule[practice_index] == schedule[game_index]):
                return False
        
        return True
    
    def check_unwanted(self, event_index, time_slot):
        return time_slot not in self.events_and_info.iloc[event_index]['Unwanted']
    
    def check_partassign(self, event_index, time_slot):
        return time_slot != self.events_and_info.iloc[event_index]['Part_assign']
    
    def check_evening_div(self, event_index, time_slot, event_type):
        if self.events_and_info.iloc[event_index]['Div'] != '09':
            return True
        
        event_time = None
        if event_type == 'G':
            event_time = self.gslots_and_info.loc[time_slot,'Start']
        else:
            event_time = self.pslots_and_info.loc[time_slot,'Start']

        event_time = datetime.strptime(event_time, "%H:%M").time()

        return self.evening >= event_time

