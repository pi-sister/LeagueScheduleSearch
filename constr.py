import pandas as pd
from datetime import datetime, timedelta

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
                env (dictionary): contains all the necessary information to complete orTree (particularly
                for the constr function) <- will eventually add as I go through each function
                    env.game_slots (dataframe): dataframe of each game slot and their attributes
                    env.practice_slots (dataframe): dataframe of each practice slot and their attributes
                    env.events (dataframe): dataframe of each game and practice plus their attributes
        """
        # populate local variables
        self.env = env
        # evening
        self.evening = datetime.strptime("18:00", "%H:%M").time()

        self.__u13t1s = env.events["Tier"].isin(["U13T1"]).any()
        self.__u12t1s = env.events["Tier"].isin(["U12T1"]).any()


        # Variables for checking if max exceeded
        self.game_counter = []
        self.practice_counter = []

        self.game_slot_lookup = dict(zip(list(env.game_slots.index), list(range(0,self.env.game_slot_num()))))
        self.practice_slot_lookup = dict(zip(list(env.practice_slots.index), list(range(0,self.env.practice_slot_num()))))
    
    # Change this so we accept one slot and create a counter that will retunr T or F
    # Find a way to reset
    @property
    def special_events(self):
        return (self.__u13t1s or self.__u12t1s)
    
    def max_exceeded_reset(self):
        """
        resets the counter for number of games or practices in a slot to prepare
        for checking the constraints in a new schedule
        """
        # Lists to count the # of occurrences of each slot in the schedule
        self.game_counter = [0] * self.env.game_slot_num()
        self.practice_counter = [0] * self.env.practice_slot_num()


    def max_exceeded(self, slot: str, slot_type: str):
        """
        accepts a slot and the slot type (game or practice) as input and add it
        to the counter. If the slots counter exceeds the limit return False
        """

        if slot_type == 'G':
            self.game_counter[self.game_slot_lookup[slot]] += 1
            # print("game_counter", self.game_counter, "\nlookupslot: ", self.game_slot_lookup[slot], "\nmaxSlots type: ", type(self.env.game_slots.loc[slot, 'Max']))
            if self.game_counter[self.game_slot_lookup[slot]] > int(self.env.game_slots.loc[slot,'Max']):
                print("Broke Gamemax")
                return False
            
        if slot_type == 'P':
            self.practice_counter[self.practice_slot_lookup[slot]] += 1
            if self.practice_counter[self.practice_slot_lookup[slot]] > int(self.env.practice_slots.loc[slot,'Max']):
                print("Broke Practicemax")
                return False
            
        return True
    
    def incompatible(self, event_index, schedule):
        # wrong : (
        if not self.env.events.iloc[event_index]['Incompatible']:
            return True
          
        row_numbers = [self.env.events.index.get_loc(label) for label in self.env.events.iloc[event_index]['Incompatible']]

        slot_set = set()
        for i in row_numbers:
            if schedule[i] == "*":
                break
            
            if (schedule[i] in slot_set):
                print("Broke Incompatible")
                return False
            
            slot_set.add(schedule[i])

        return True
    
    def incompatible2(self, event_assignments, incompatible_list, event):
        if not incompatible_list:
            return True
        
        incompatible_checker = set()
        relevant_events = event_assignments.loc[incompatible_list]

        for event_id, slot in relevant_events.items():
            if event_id == event:
                continue

            if slot in incompatible_checker:
                return False
            
            incompatible_checker.add(slot)

        return True
    
    def check_assign(self, current_index, schedule, mode):
        related_events = []

        if mode == "regcheck":
            if not self.env.events.iloc[current_index]['Tier'].startswith(('U15', 'U16', 'U17', 'U18', 'U19')):
                return True
            
            related_events = self.env.events[
                (self.env.events['Tier'].str.startswith(('U15', 'U16', 'U17', 'U18', 'U19'))) &
                (self.env.events['Type'] == 'G')
            ].index.tolist()

            event_indices = [self.env.events.index.get_loc(label) for label in related_events]

            for event_index in event_indices:
                if (schedule[event_index] == "*") or (event_index == current_index):
                    continue
                if (schedule[current_index] == schedule[event_index]):
                    return False
            
            return True
        
        if mode == "specialcheck":
            if (self.__u13t1s and self.env.events.iloc[current_index]['Tier'].startswith('U13T1S')):
                related_events = self.env.events[
                    (self.env.events['Tier'].str.startswith('U13T1'))
                ].index.tolist()

            if (self.__u12t1s and self.env.events.iloc[current_index]['Tier'].startswith('U12T1S')):
                related_events = self.env.events[
                    (self.env.events['Tier'].str.startswith('U12T1'))
                ].index.tolist()

            if not related_events:
                return True
            
            event_indices = [self.env.events.index.get_loc(label) for label in related_events]

            return self.pcheck(schedule, current_index, event_indices)


        compared_event = self.env.events.index[current_index]

        if mode == "pcheck":
            if 'PRC' in compared_event:
                corresponding_game = compared_event.partition('PRC')[0]
            else:
                corresponding_game = compared_event.partition('OPN')[0]

            related_events = self.env.events[
                (self.env.events.index.str.startswith(corresponding_game))
            ].index.tolist()

            event_indices = [self.env.events.index.get_loc(label) for label in related_events]

            return self.pcheck(schedule, current_index, event_indices)
    
    def pcheck(self, schedule, practice_index, search_events):
        practice_day = schedule[practice_index][:2]
        practice_time = datetime.strptime(schedule[practice_index][2:], "%H:%M").time()
        practice_datetime = datetime.combine(datetime.min, practice_time)

        if (practice_day == 'MO') or (practice_day == 'TU'):
            practice_duration = timedelta(hours=1)
        else:
            practice_duration = timedelta(hours=2)
        
        practice_end_datetime = practice_datetime + practice_duration

        practice_end = practice_end_datetime.time()

        result = True
        for event_index in search_events:
            if (schedule[event_index] == "*") or (event_index == practice_index):
                continue

            if event_index < self.env.game_slot_num():
                if ((practice_day == 'FR') and (schedule[event_index][:2] == 'MO')):
                    game_start = datetime.strptime(schedule[event_index][2:], "%H:%M").time()

                    result = not((practice_time <= game_start) and (game_start < practice_end))
                    
                elif ((practice_day == 'TU') and (schedule[event_index][:2] == 'TU')):
                    game_start = datetime.strptime(schedule[event_index][2:], "%H:%M").time()

                    # Convert time to datetime (use a dummy date, e.g., '1900-01-01')
                    game_start_datetime = datetime.combine(datetime.min, game_start)

                    # Add 1 hour and 30 minutes
                    game_duration = timedelta(hours=1, minutes=30)
  
                    game_end_datetime = game_start_datetime + game_duration

                    # Convert back to time object
                    game_end = game_end_datetime.time()

                    result = not(
                        ((game_start <= practice_time) and (practice_time < game_end)) and 
                        ((game_start <= practice_end) and (practice_time < practice_end))
                    )
                else: 
                    result = not(schedule[practice_index] == schedule[event_index])
            else: 
                result = not(schedule[practice_index] == schedule[event_index])

            if not result:
                return False
        
        return True
    
    def check_unwanted(self, event_index, time_slot):
        return time_slot not in self.env.events.iloc[event_index]['Unwanted']
    
    def check_partassign(self, event_index, time_slot):
        if self.env.events.iloc[event_index]['Part_assign'] == "*":
            return True
        
        return time_slot == self.env.events.iloc[event_index]['Part_assign']
    
    def check_evening_div(self, event_index, time_slot, event_type):
        if self.env.events.iloc[event_index]['Div'] != '09':
            return True
        
        event_time = None
        if event_type == 'G':
            event_time = self.env.game_slots.loc[time_slot,'Start']
        else:
            event_time = self.env.practice_slots.loc[time_slot,'Start']

        event_time = datetime.strptime(event_time, "%H:%M").time()

        return self.evening >= event_time
    
    def check_evening_div2(self, time_string, division):
        if division != '09':
            return True
        
        event_time = datetime.strptime(time_string, "%H:%M").time()

        return self.evening >= event_time


