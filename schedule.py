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
        self.pslots = env.practice_slots.copy()
        
    def get_Starting(self):
        return self.assignments.to_list()
        
    def set_Eval(self):
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

        total_df['Assigned'] = self.assigned
        g_df = total_df[total_df['Type'] == 'G']
        p_df = total_df[total_df['Type'] == 'P']
        g_df = pd.merge(g_df, self.gslots, how = 'left', left_on = 'Assigned', right_index = True)
        p_df = pd.merge(p_df, self.pslots, how = 'left', left_on = 'Assigned', right_index = True)    
        g_df['Type'] = 'G'
        p_df['Type'] = 'P'
        
        df = pd.concat([g_df, p_df])
        
        print(df['Assigned'])
        [print(df.columns)]
        
        min = self.min_filled(g_df, self.gslots, env.pen_notpaired)
        min += self.min_filled(p_df, self.pslots, env.pen_notpaired)
        min *= env.w_minfilled
        
        pref = self.pref_penalty(df)
        pref *= env.w_pref
        
        pair = self.pair_penalty(df, env.pen_notpaired)
        pair *= env.w_pair
        
        sdiff = self.slot_diff_penalty(df, env.pen_section)
        sdiff *= env.w_secdiff
        
        self.eval = min + pref + pair + sdiff
        
    def min_filled(self, df, slots, penalty: int):
        """
        Calculate the minimum penalty for unfilled slots in a schedule.
        
        Parameters:
            df (pandas.DataFrame): DataFrame containing the schedule data with an 'Assigned' column.
            slots (pandas.DataFrame): DataFrame containing the slot data with a 'Min' column.
            penalty (int): Penalty value to be applied for each unfilled slot.
        
        Returns:
            int: The total minimum penalty for unfilled slots.
        """
        self.update_counters()
        def min_penalty(row):
            return penalty * max(0, int(row['Min']) - int(row['count']))
        
        slots['min_penalty'] = slots.apply(min_penalty, axis = 1)
        
        return slots['min_penalty'].sum()
    
    def max_exceeded(self, slot, slot_type):
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
            return self.pslots.at[slot, 'count'] <= self.pslots.at[slot, 'Max']
        else:
            raise ValueError("Invalid slot type. Must be 'G' or 'P'.")
        
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
        if slot_type == 'G':
            df = df[df['Type'] == 'G']
            slots = self.gslots
        elif slot_type == 'P':
            df = df[df['Type'] == 'P']
            slots = self.pslots
        
        assigned_counts = df['Assigned'].value_counts()
        slots = pd.merge(slots, assigned_counts, how='left', left_index=True, right_index=True)
        slots['count'] = slots['count'].fillna(0)
        
        
    def pref_penalty(self, df):
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
                for pref,value in row['Preference']:
                    if pref != row['Assigned']:
                        penalty += int(value)
                return penalty
        df['pref_penalty'] = df.apply(pref_calc, axis = 1)
        return df['pref_penalty'].sum()

    def pair_penalty(self, df, penalty: int):
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
                for pair in row['Pair_with']:
                    if df.loc[pair]['Assigned'] != row['Assigned']:
                        count += 1
                return count
        df['not_paired'] = df.apply(count_unpaired, axis=1)
        return df['not_paired'].sum() * penalty
        
    def slot_diff_penalty(self, df, penalty: int):
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
        for league in games['League'].unique():
            for tier in games['Tier'].unique():
                same_tier = games[(games['League'] == league) & (games['Tier'] == tier)]
                counts = same_tier['Assigned'].value_counts()
                for _, count in counts.items():
                    if count > 1:
                        pairs += comb(count, 2)
    
        return pairs * penalty


    def alt_set_Eval(self):
        # Initialize environment variables
        env = self.env
        events = env.events
        game_slots = env.game_slots
        practice_slots = env.practice_slots
        total_penalty = 0
        total_df = events.copy()

        # Add 'Assigned' column to total_df and initialize it
        total_df['Assigned'] = self.assigned
        total_df['Assigned'] = total_df['Assigned'].apply(lambda x: '' if isinstance(x, list) and not x else x)

        # Merge total_df with game_slots and practice_slots on 'Assigned' column
        df = pd.merge(total_df, game_slots, how='left', left_on='Assigned', right_index=True)
        df = pd.merge(df, practice_slots, how='left', left_on='Assigned', right_index=True)
        # This needs to be fixed - separate merges for game and practice slots


        # Filter rows where 'Assigned' is not empty
        filterRows = df[df['Assigned'] != ""]

        # Initialize dictionaries to track the number of games and practices in each slot
        game_slot_dic = dict()
        prac_slot_dic = dict()

        # Initialize game_slot_dic with game slot indices
        for slot in env.game_slots.index:
            game_slot_dic[slot] = 0

        # Initialize prac_slot_dic with practice slot indices
        for slot in env.practice_slots.index:
            prac_slot_dic[slot] = 0

        # Iterate over filtered rows
        for row in filterRows.index:
            # Increment the count of games in the assigned slot
            game_slot_dic[df['Assigned'][row]] += 1

            # Check for paired games and apply penalties if they are not in the same slot
            if debug_pair_with:
                events_non_empty = events[events['Pair_with'].apply(lambda x: not isinstance(x, list) or len(x) > 0)]
                pair_with_dict = events_non_empty['Pair_with'].to_dict()

            # Check if the current row has any paired games
            if pair_with_dict.get(row) is not None:
                # Get the list of games paired with the current row
                games_to_check_for_row = pair_with_dict.get(row)
                
                # Debugging: print the list of paired games
                if test_stuff_debug:
                    print(f"{games_to_check_for_row = }")

                # Iterate over each paired game
                for game in games_to_check_for_row:
                    # Check if the paired game is in the filtered rows
                    if game in filterRows.index:
                        # Get the assigned slot for the current row
                        var = df['Assigned'][row]
                        # Get the assigned slot for the paired game
                        v2 = df['Assigned'][game]

                        # If the assigned slots are different, apply the pairing penalty
                        if var != v2:
                            cal = env.pen_notpaired * env.w_pair
                            total_penalty += cal / 2
                    else:
                        # Debugging: print a message if the paired game is not in the index
                        print("game not in index")

            # Check for preferred slots and apply penalties if they are not in the preferred slot
            events_pref_non_empty = events[events['Preference'].apply(lambda x: not isinstance(x, list) or len(x) > 0)]
            pref_with_dict = events_pref_non_empty['Preference'].to_dict()

            if pref_with_dict.get(row) is not None:
                slots_to_check = pref_with_dict.get(row)

                for slot in slots_to_check:
                    if slot in filterRows.index:
                        var = df['Assigned'][row]
                        v2 = df['Assigned'][slot]

                        if var != v2:
                            pen = env.w_pref * df['Pref_value'][slot]
                            total_penalty += pen

        # Define function to calculate game minimum penalty
        def game_min_penalty():
            penalty_game_min_total = 0
            current_game_penalty = 0
            gslots_min = game_slots['Min'].to_dict()
            for slot in game_slot_dic:
                current_game_penalty = env.pen_gamemin * env.w_minfilled * max(0, int(gslots_min[slot]) - game_slot_dic[slot])
                penalty_game_min_total += current_game_penalty
            print(f"{penalty_game_min_total = }")
            return penalty_game_min_total

        # Add game minimum penalty to total penalty
        total_penalty = game_min_penalty() + total_penalty

        # Define function to calculate practice minimum penalty
        def prac_min_penalty():
            penalty_prac_min_total = 0
            current_prac_penalty = 0
            pslots_min = practice_slots['Min'].to_dict()
            for slot in prac_slot_dic:
                current_prac_penalty = env.pen_gamemin * env.w_minfilled * max(0, int(pslots_min[slot]) - prac_slot_dic[slot])
                penalty_prac_min_total += current_prac_penalty
            print(f"{penalty_prac_min_total = }")
            return penalty_prac_min_total

        # Add practice minimum penalty to total penalty
        total_penalty = prac_min_penalty() + total_penalty
                
            
            
            # if(test_stuff_debug):
            #     print(f"{total_penalty = }")

        self.eval = total_penalty
    
    def assign(self, slots, verbose = False):
        self.assigned = slots
        self.events['Assigned'] = self.assigned
        self.update_counters()
        self.set_Eval()
        if verbose:
            print(f"Assigned: {self.assigned}")
            print(f"Evaluation: {self.eval}")
        return self.eval
        
    def assign2(self, slots, verbose = False):
        self.assigned = slots
        self.alt_set_Eval()
        if verbose:
            print(f"Assigned: {self.assigned}")
            print(f"Evaluation: {self.eval}")
        return self.eval
    
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
        
