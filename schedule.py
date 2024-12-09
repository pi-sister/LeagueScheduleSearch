# %%
# # Installations for your venv
# %pip install numpy
# %pip install pandas


# %% [markdown]
# # Imports and Variables

# %%
import os
import sys
import re
import pandas as pd
import numpy as np
import environment
from math import comb

# %%
# Class of schedule
debug_pair_with = True
test_stuff_debug = True

class Schedule:  
    """
    The Schedule class is responsible for managing and evaluating a schedule of events within a given environment. 
    It provides methods to calculate various penalties based on the schedule's adherence to constraints and preferences.
        env (environment.Environment): The environment object containing game slots, practice slots, events, and penalty weights.
        eval (int): The evaluation score for the current schedule.
        event_list (list): A list of event indices from the environment.
        assignments (pandas.Series): A series of assigned parts from the events.
        assigned (list): A list of assigned slots.
    Methods:
        __init__(env: environment.Environment):
            Initializes the Schedule object with the given environment.
        get_Starting():
            Returns the list of assignments as a list.
        set_Eval():
            Calculates and sets the evaluation score for the current schedule based on various penalties.
        min_filled(df: pandas.DataFrame, slots: pandas.DataFrame, penalty: int) -> int:
            Calculates the minimum penalty for unfilled slots in a schedule.
        pref_penalty(df: pandas.DataFrame) -> int:
            Calculates the preference penalty for each row in the DataFrame and returns the total penalty sum.
        pair_penalty(df: pandas.DataFrame, penalty: int) -> int:
            Calculates the penalty for unpaired assignments in a DataFrame.
        slot_diff_penalty(df: pandas.DataFrame, penalty: int) -> int:
            Calculates the penalty based on the number of pairs of games assigned to the same slot.
        alt_set_Eval():
            An alternative method to calculate and set the evaluation score for the current schedule.
        assign(slots: list, verbose: bool = False) -> int:
            Assigns the given slots to the schedule and calculates the evaluation score.
        assign2(slots: list, verbose: bool = False) -> int:
            Assigns the given slots to the schedule using the alternative evaluation method and calculates the evaluation score.
    """
    def __init__(self, env: environment.Environment):
        self.events = env.events.copy()
        self.env = env
        self.eval = None
        self.event_list = self.events.index.tolist()
        self.assignments = self.events['Part_assign']
        self.assigned = []

        self.gslots = env.game_slots.copy()
        self.gslots['Max'] = pd.to_numeric(self.gslots['Max'])

        self.pslots = env.practice_slots.copy()
        self.pslots['Max'] = pd.to_numeric(self.pslots['Max'])

    def __lt__(self, other):
        #  Comparing based on the `eval` attribute
        if isinstance(other, Schedule):
            return self.eval < other.eval
        return NotImplemented
    
    def __len__(self):
        """
        Method obtain the number of events
        """
        return len(self.events.index)

    def __iter__(self):
        """
        Method to iterate through schedule events
        """
        return self.events.iterrows()
    
    def __getitem__(self, index):
        """
        Method to obtain assigned slot with an index
        """
        return self.events.iloc[index]['Assigned']
    
    def __str__(self):
        """
        Method changed schedule to String format (for printing purposes.)
        """
        string = "(" + str(self.assigned) + ", " + str(self.eval) + ")"
        return string

    def get_Assignments(self):
        """
        Method to obtain all current assignments
        """
        return self.events[['Assigned', 'League', 'Tier', 'Div','Type', 'Corresp_game']]

    def get_Starting(self):
        """
        Method to obtain the preassignments
        """
        return self.assignments.to_list()
        
    def set_Eval(self, verbose = 0):
        """
        Calculate and set the evaluation score for the current schedule.
        This method computes the total penalty score for the current schedule
        based on various criteria including minimum filled slots, preference penalties,
        pairing penalties, and section differences. The computed score is then assigned
        to the `self.eval` attribute.
        
        The penalties are calculated as follows:
        - Minimum filled slots penalty: Ensures that the minimum number of game and practice slots are filled.
        - Preference penalty: Penalizes based on how well the assigned slots match the preferred slots.
        - Pairing penalty: Penalizes based on how well the assigned slots are paired.
        - Section difference penalty: Penalizes based on the difference in slot sections.
        The penalties are weighted by their respective weights defined in the environment.
        """
        
        env = self.env

        total_df = self.events

        # Add 'Assigned' column to total_df and initialize it
        total_df['Assigned'] = self.assigned
        g_df = total_df[total_df['Type'] == 'G']
        p_df = total_df[total_df['Type'] == 'P']
        g_df = pd.merge(g_df, self.gslots, how = 'left', left_on = 'Assigned', right_index = True)
        p_df = pd.merge(p_df, self.pslots, how = 'left', left_on = 'Assigned', right_index = True)    
        g_df['Type'] = 'G' # Gslots contain merged data for games
        p_df['Type'] = 'P' # Pslots contain merged data for practices
        
        df = pd.concat([g_df, p_df])
        
        min = self.min_filled(self.gslots, env.pen_notpaired, verbose)
        min += self.min_filled(self.pslots, env.pen_notpaired, verbose)
        min *= env.w_minfilled
        
        pref = self.pref_penalty(df, verbose)
        pref *= env.w_pref
        
        pair = self.pair_penalty(df, env.pen_notpaired, verbose)
        pair *= env.w_pair
        
        sdiff = self.slot_diff_penalty(df, env.pen_section, verbose)
        sdiff *= env.w_secdiff
        
        self.eval = min + pref + pair + sdiff
        return self.eval
        
    def min_filled(self, slots, penalty: int, verbose = 0) -> int:
        """
        Calculate the minimum penalty for unfilled slots in a schedule.
        
        Parameters:
            slots (pandas.DataFrame): DataFrame containing the slot data with a 'Min' column.
            penalty (int): Penalty value to be applied for each unfilled slot.
            verbose (int): Verbosity level for printing debug information.
        
        Returns:
            int: The total minimum penalty for unfilled slots.
        """
        self.update_counters() # Update the counters for the slots
        if slots.empty:
            return 0
        
        def min_penalty(row):
            difference = int(row['Min']) - int(row['count'])
            return max(0,  difference)
        
        slots['min_penalty'] = slots.apply(min_penalty, axis = 1)
        
        if verbose:
            print(f'\nMinimum Penalty: {slots["min_penalty"].sum()}\n')
        
        return slots['min_penalty'].sum()
    
    def max_exceeded(self, slot, slot_type) -> bool:
        """
        Accepts a slot and the slot type (game or practice) as input and adds it
        to the counter. If the slots counter exceeds the limit, return False.
        
        Args:
            slot (str): The slot to be checked.
            slot_type (str): The type of slot ('G' for game, 'P' for practice).
        
        Returns:
            bool: True if the slot limit is not exceeded, False otherwise.
        """
 
        if slot_type == 'G':
            return self.gslots.at[slot, 'count'] <= self.gslots.at[slot, 'Max']
        elif slot_type == 'P':
            if slot in self.pslots.index:
                return self.pslots.at[slot, 'count'] <= self.pslots.at[slot, 'Max']
            # else:
            #     print("Invalid Schedule")
            #     exit()
            #raise ValueError(f"Slot {slot} is not in practice slots")
            
        else:
            raise ValueError("Invalid slot type. Must be 'G' or 'P'.")

    def return_not_maxed(self, slot_type):
        if slot_type =='G':
            return self.gslots[self.gslots['count'] < self.gslots['Max']]
        else:
            return self.pslots[self.pslots['count'] < self.pslots['Max']]

    def update_counters(self):
        """
        Update the game or practice slot counter based on the given slot and slot type.
        
        Args:
            slot (str): The slot to be updated.
            slot_type (str): The type of slot ('G' for game, 'P' for practice).
        """
        self.update_typed_counters(self.events, 'G')
        self.update_typed_counters(self.events, 'P')
        
    def update_typed_counters(self, df, slot_type):
        """
        Update the counters for the specified slot type based on the given DataFrame.
        This method filters the DataFrame based on the slot type ('G' or 'P') and updates
        the corresponding slot counters with the count of assigned values.
        Args:
            df (pandas.DataFrame): The DataFrame containing the schedule data.
            slot_type (str): The type of slot to update ('G' for gslots or 'P' for pslots).
        Returns:
            None
        """
        
        if slot_type == 'G':
            df = df[df['Type'] == 'G']
            slots = self.gslots
        elif slot_type == 'P':
            df = df[df['Type'] == 'P']
            slots = self.pslots
        
        assigned_counts = df['Assigned'].value_counts()
        slots['count'] = assigned_counts
        slots['count'] = slots['count'].fillna(0)
        
        
    def pref_penalty(self, df, verbose = 0): ## DUPLICATE FUNCTION - Will be replaced with @Khadeeja's function
        """
        Calculate the preference penalty for each row in the DataFrame and return the total penalty sum.
        
        Args:
            df (pandas.DataFrame): The DataFrame containing the schedule data. It must have columns 'Preference' and 'Assigned'.
            
        Returns:
            int: The total preference penalty sum for the DataFrame.
            
        The function works as follows:
        - For each row in the DataFrame, it checks the 'Preference' column.
        - If 'Preference' is an empty list, the penalty for that row is 0.
        - Otherwise, it iterates through the preferences and their associated values.
        - For each preference, it checks if the 'Assigned' value of the preference does not match the 'Assigned' value of the current row.
        - If they do not match, it adds the associated value to the penalty.
        - The penalty for each row is stored in a new column 'pref_penalty'.
        - Finally, the function returns the sum of all penalties in the 'pref_penalty' column.
        """    
        def pref_calc(row):
            if row['Preference'] == []:
                return 0
            else:
                penalty = 0
                # If list isn't empty, iterate through each preference
                if isinstance(row['Preference'], list): 
                    # For each tuple in this row's preference list, check if the assigned slot is not the preferred slot
                    for item in row['Preference']:
                        # If the assigned slot is not the preferred slot, add the penalty value to the total penalty for this row
                        if item[0] != row['Assigned']:
                            penalty += int(item[1])
                return penalty
        df['pref_penalty'] = df.apply(pref_calc, axis = 1)
        # Return the sum of all preference penalties
        all_pref_penalty = df['pref_penalty'].sum()
        if verbose:
            print(f'\nPreference Penalty: {all_pref_penalty}\n')
        return all_pref_penalty

    def pair_penalty(self, df, penalty: int, verbose = 0):
        """
        Calculate the penalty for unpaired assignments in a DataFrame.
        This function iterates over each row in the DataFrame and counts the number of 
        pairs that are not assigned together. It then calculates the total penalty 
        based on the number of unpaired assignments.
        
        Parameters:
            df (pandas.DataFrame): The DataFrame containing the schedule data. 
                               It must have columns 'Pair_with' and 'Assigned'.
            penalty (int): The penalty value to be multiplied by the total number 
                                of unpaired assignments.
                                
        Returns:
            int: The total penalty for unpaired assignments.
        """
        def count_unpaired(row):
            if row['Pair_with'] == []:
                return 0
            else:
                count = 0
                # If list isn't empty, iterate through each pair
                if isinstance(row['Pair_with'], list):
                    for pair in row['Pair_with']:
                        if df.loc[pair]['Assigned'] != row['Assigned']:
                            count += 1
                return count/2
        # Apply the count_unpaired function to each row in the DataFrame
        df['not_paired'] = df.apply(count_unpaired, axis=1)
        count = df['not_paired'].sum()
        total_penalty = count * penalty
        
        if verbose:
            print(f'\nPair Penalty: {count} number of unpaired assignments x {penalty} = {total_penalty}\n')
        
        # Return the count of unpaired assignments multiplied by the penalty value
        return total_penalty
        
    def slot_diff_penalty(self, df, penalty: int, verbose = 0) -> int:
        """
        Calculate the penalty based on the number of pairs of games assigned to the same slot.
        This function iterates through each unique league and tier in the DataFrame, counts the number of games assigned to each slot,
        and calculates the number of pairs of games assigned to the same slot. The total penalty is then computed by multiplying the 
        number of pairs by the given penalty value.
        
        Args:
            df (pandas.DataFrame): DataFrame containing the schedule data with columns 'Type', 'League', 'Tier', and 'Assigned'.
            penalty (int): The penalty value to be applied for each pair of games assigned to the same slot.
            
        Returns:
            int: The total penalty calculated based on the number of pairs of games assigned to the same slot.
        """        
        games = df[df['Type'] == 'G']
        
        pairs = 0
        # Iterate over each unique league and tier in the DataFrame
        for league in games['League'].unique():
            for tier in games['Tier'].unique():
                # Filter games based on league and tier
                same_tier = games[(games['League'] == league) & (games['Tier'] == tier)]
                # Counts the number of games assigned to each slot
                counts = same_tier['Assigned'].value_counts()
                for _, count in counts.items():
                    if count > 1:
                        # Combination of games assigned to the same slot
                        pairs += comb(count, 2)
        total_penalty = pairs * penalty
        if verbose:
            print(f'\nSlot Difference Penalty: {pairs} number of pairs x {penalty} = {total_penalty}\n')
            
        return total_penalty

    
    def assign(self, slots):
        """
        Assigns the given slots to the instance and updates the events and counters.
        Parameters:
            slots (list): A list of slots to be assigned.
        Returns:
            None
        """
        self.assigned = slots
        if self.assigned:
            self.events['Assigned'] = self.assigned
            self.update_counters()
        else:
            print("Invalid Schedule")
            sys.exit() 

    def get_scheduled(self) -> pd.DataFrame:
        """
        Retrieve scheduled events.
        This method filters the events DataFrame to return only the rows where the 'Assigned' column
        does not contain the '*' character, indicating that the event is scheduled.
        Returns:
            pandas.DataFrame: A DataFrame containing only the scheduled events.
        """
        scheduled = self.events[self.events['Assigned'] != '*']
        return scheduled
    
    @staticmethod
    def list_to_schedule(lst: list, env: environment):
        """
        Converts a list of strings representing event assignments to a Schedule object.
        
        Args:
            lst (list): A list of strings representing event assignments.
            env (environment.Environment): The environment object containing game slots, practice slots, events, and penalty weights.
            
        Returns:
            Schedule: A Schedule object with the given assignments.
        """
        sched = Schedule(env)
        sched.assign(lst)
        return sched

        
