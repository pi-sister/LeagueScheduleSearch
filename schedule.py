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
        return self.events['Assigned']
    
    def get_Tiers(self):
        """
        Method to obtain all current assignments
        """
        return self.events[['Assigned', 'Tier', 'Type']]

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
        
        def min_penalty(row):
            return penalty * max(0, int(row['Min']) - int(row['count']))
        
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
                return count
            
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


    def set_Eval1(self):
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
        gdf = total_df[total_df['Type'] == 'G']
        pdf = total_df[total_df['Type'] == 'P']
        gdf = pd.merge(gdf, game_slots, how='left', left_on='Assigned', right_index=True)
        pdf = pd.merge(pdf, practice_slots, how='left', left_on='Assigned', right_index=True)
        # This needs to be fixed - separate merges for game and practice slots
        df = pd.concat([gdf, pdf])

        print('***Testing df****')
        print(df.columns)

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
            # end game pair stuff
            
            # now we gotta do the fit pref
            # we need to get the preferred values for the games and practices
            # if the games and practices are not in their preferred slot, we add a penalty
            events_pref_non_empty = events[events['Preference'].apply(lambda x: not isinstance(x, list) or len(x) > 0)]
            pref_with_dict = events_pref_non_empty['Preference'].to_dict()

            if pref_with_dict.get(row) is not None:
                slots_to_check = pref_with_dict[row]  # List of tuples with preferred slots and penalties
 
                # Loop through each preferred time slot and its associated penalty value
                for preferred_slot, penalty_value in slots_to_check:
                    if preferred_slot in filterRows.index:  # Check if this slot exists in filterRows
                        # Get the assigned slots for the current event and the preferred event
                        assigned_event = df.loc[row, 'Assigned']
                        assigned_slot = df.loc[preferred_slot, 'Assigned']
                        
                        # If assigned times don't match, apply a penalty
                        if assigned_event != assigned_slot:
                            penalty = env.w_pref * penalty_value
                            total_penalty += penalty
            
            
            # Different divisional games within a single age/tier group should be scheduled at different times. For each pair of divisions that is scheduled into the same slot, we add a penalty to the Eval-value of an assignment.
            # okay to do that we needa check a game and see its division. if there are any other games with the same division as it, we add a penatly (half of one to account for the loop also adding a penalty to the divison)
            
            section_with_dict = events['Div'].to_dict()
            sections_to_check = section_with_dict.get(row)

            if sections_to_check: # Ensure division is not None
                for other_row in filterRows.index:
                    if other_row != row:  # Avoid comparing the same game
                        other_division = df.loc[other_row, 'Div']
                        assigned_slot = df.loc[row, 'Assigned']
                        other_assigned_slot = df.loc[other_row, 'Assigned']
                        
                        # Apply penalty if two games with the same division are in the same slot
                        if sections_to_check == other_division and assigned_slot == other_assigned_slot:
                            total_penalty += env.pen_section  # Adding half to account for double-counting in both loops
                            
                    # if section in filterRows.index:
                    #     assigned_section = df.loc[row, 'Assigned']
                    #     assigned_slot = df.loc[section, 'Assigned']
                        
                    #     # If there are 2 or more games with the same divison in the same time slot, then apply a penalty
                    #     if

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
    
    def assign_and_eval(self, slots, verbose = 0):
        """
        Assigns slots to events, updates counters, evaluates the assignment, and optionally prints the details.
        Args:
            slots (list): A list of slots to be assigned to events.
            verbose (int, optional): If set to a non-zero value, prints the assigned slots and evaluation. Defaults to 0.
        Returns:
            float: The evaluation score after assigning the slots.
        """
        self.assigned = slots
        self.events['Assigned'] = self.assigned
        self.update_counters()
        self.set_Eval(verbose=verbose)
        if verbose:
            print(f"Assigned: {self.assigned}")
            print(f"Evaluation: {self.eval}")
        return self.eval
        
    def assign2_and_eval(self, slots, verbose = False):
        """
        Assigns the provided slots to the instance, evaluates the assignment, and optionally prints the details.
        Args:
            slots (list): The slots to be assigned.
            verbose (bool, optional): If True, prints the assigned slots and evaluation. Defaults to False.
        Returns:
            float: The evaluation result after assigning the slots.
        """
        self.assigned = slots
        self.set_Eval1()
        if verbose:
            print(f"Assigned: {self.assigned}")
            print(f"Evaluation: {self.eval}")
        return self.eval
    
    def assign(self, slots):
        """
        Assigns the given slots to the instance and updates the events and counters.
        Parameters:
            slots (list): A list of slots to be assigned.
        Returns:
            None
        """
        self.assigned = slots
        self.events['Assigned'] = self.assigned
        self.update_counters()
        
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

        
