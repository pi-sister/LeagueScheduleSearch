# %% [markdown]
# # Imports and Variables

# %%
import os
import sys
import re
import pandas as pd
import numpy as np
import argparse

# %% [markdown]
# Parse command line

# %% [markdown]
# ## Parse file - Using DataFrames

# %%
class Environment:
    def __init__(self, file_name: str = None, integers: list = None, verbose = 1):
        """
        Initialize the environment with the given file and integer parameters.
        Parameters:
        file_name (str, optional): The name of the file to read data from. If not provided, the user will be prompted to input a file name.
        integers (list, optional): A list of 8 positive integers representing weights and penalties. If not provided, the user will be prompted to input these values.
        verbose (int, optional): Verbosity level for processing. Default is 1.
        Raises:
        ValueError: If any of the integers provided are negative.
        FileNotFoundError: If the specified file does not exist.
        Notes:
        - The file is expected to contain specific sections identified by keywords such as 'Name:', 'Game slots:', 'Practice slots:', etc.
        - The integers list should contain exactly 8 positive integers.
        - The method processes game and practice slots, games, and practices, and combines them into a single DataFrame with additional properties.
        """

        if file_name is None: # If no file is given, ask for it
            file_name = input('File')
            integers = input('Weights and Penalties:').split(' ')
            integers = list(map(int, integers))
            print(f'File chosen = {file_name}')
            print(f'Integer inputs = {integers}\n')

        # check if file exists
        if os.path.isfile(file_name):
            with open(file_name, "r") as inputfile:   # opens file     
                data = inputfile.read()          # starts reading file
                
                # splits the file based on the key words
                split_data = re.split(r"Name:|Game slots:|Practice slots:|Games:|Practices:|Not compatible:|Unwanted:|Preferences:|Pair:|Partial assignments:", data, flags=re.IGNORECASE)
                # print(split_data)
        else: 
            print("Unable to open file. Please try again.")
            
        # check if integers are positive
        if any(i < 0 for i in integers):
            print("Please enter 8 positive integers.")
        else:
            self.__w_minfilled, self.__w_pref, self.__w_pair, self.__w_secdiff, self.__pen_gamemin, self.__pen_pracmin, self.__pen_notpaired, self.__pen_section = integers
            
        # Process game slots
        self.__game_slots = _PrivateParser.process_slots(split_data[2], 'G', verbose)
            
        # Process practice slots
        self.__practice_slots = _PrivateParser.process_slots(split_data[3], 'P', verbose)
        
        # Process games into a lookup table
        games = _PrivateParser.process_games_practices(split_data[4], 'G')

        # Process practices into a lookup table
        practices = _PrivateParser.process_games_practices(split_data[5], 'P')

        # Combine Practices and Games
        events = pd.concat([games, practices], axis=0)
        events = events.drop_duplicates()
        
        # Add properties to the events DataFrame
        self.__events = _PrivateParser.add_properties(events, split_data, verbose)

    @property
    def w_minfilled(self):
        return self.__w_minfilled

    @property
    def w_pref(self):
        return self.__w_pref

    @property
    def w_pair(self):
        return self.__w_pair

    @property
    def w_secdiff(self):
        return self.__w_secdiff

    @property
    def pen_gamemin(self):
        return self.__pen_gamemin

    @property
    def pen_pracmin(self):
        return self.__pen_pracmin

    @property
    def pen_notpaired(self):
        return self.__pen_notpaired

    @property
    def pen_section(self):
        return self.__pen_section
    
    @property
    def game_slots(self):
        return self.__game_slots
    
    @property
    def practice_slots(self):
        return self.__practice_slots
    
    @property
    def events(self):
        return self.__events
    
    def __str__(self):
        return f'Environment: \n {self.__game_slots} \n {self.__practice_slots} \n {self.__events}'
    
    def event_length(self):
        return len(self.__events)
    
    def game_slot_num(self):
        return len(self.__game_slots)
    
    def practice_slot_num(self):
        return len(self.__practice_slots)

class _PrivateParser:
    @staticmethod
    def process_slots(data: str, event_type: str, verbose = 1):
        """
        Processes a string of slot data and converts it into a pandas DataFrame.
        Args:
            data (str): A string containing slot data, with each slot separated by a newline.
            columns (list): A list of column names for the DataFrame.
            event_type (str): A character indicating either games 'G' or practices 'P'
        Returns:
            pd.DataFrame: A DataFrame containing the processed slot data.
        """
        columns = ['Day', 'Start', 'Max', 'Min', 'Invalid_Assign']
        slots = []
        indices = []
        for slot in data.split('\n'):
            slot = slot.replace(' ','')
            if len(slot) > 0:
                # print(slot)
                day, time, max, min = slot.strip().split(',')
                # if not properties == [''] :
                slots.append({'Day': day,
                            'Start': time,
                            'Max': max,
                            'Min': min})
                indices.append(day + time)
        df = pd.DataFrame(slots, columns=columns, index=indices)
        df['Invalid_Assign'] = np.empty((len(df), 0)).tolist()
        df['Type'] = event_type
        if verbose:
            print(f'Processed {len(df)} {event_type} slots: \n {df}\n')
        return df
    
    @staticmethod
    def process_games_practices(data, event_type, verbose = 1):
        """
        Processes game and practice data and converts it into a pandas DataFrame.
        Args:
            data (str): The raw data as a string, where each line represents a game or practice entry.
            columns (list): A list of column names for the resulting DataFrame.
            verbose (int, optional): Verbosity level. If set to 1, prints the processed DataFrame. Defaults to 1.
        Returns:
            pd.DataFrame: A DataFrame containing the processed game and practice data.
        """
        
        items = []
        indices = []
        strings = data.split('\n')
        for item in strings:
            index = _PrivateParser.get_index(item)
            # print(index)
            if len(index) > 0: # If not empty
                list_attributes = re.split(r'\s+', item.strip())
                if event_type == 'P' and len(list_attributes) == 6: # Normal practice
                    # print(item)
                    league, tier, _, divnum, ptype, pnum = list_attributes
                elif event_type == 'P' and len(list_attributes) == 4: # Practice used by all Divs
                    # print(item)
                    league, tier, ptype, pnum = list_attributes
                    divt = 'DIV'
                    divnum = ''
                elif event_type == 'G' and len(list_attributes) == 4: # Game
                    # print(item)
                    league, tier, _, divnum = list_attributes
                    ptype = ''
                    pnum = ''
                else: # Something is wrong
                    raise ValueError(f"Invalid data format for event type {event_type}: {item}")
                # Dictionary of this item
                items.append({'League': league,
                            'Tier': tier,
                            'Div': divnum,
                            'Practice_Type': ptype,
                            'Num': pnum})
                indices.append(index)
        # DataFrame with items
        df = pd.DataFrame(items, index = indices)
        df['Type'] = event_type
        if verbose:
            print(f'Processed {len(df)} {event_type} items: \n {df}\n')
        return df

        
    @staticmethod
    def add_properties(events: pd.DataFrame, split_data: list, verbose = 1):
        
        special_practices = []
        def special_detection(event):
            if event['League'] == 'CMSA' and event['Tier'] == 'U12T1':
                U12_practice = ['CMSA','U12T1S', '', '', '', 'P']
                special_practices.append(U12_practice)
            elif event['League'] == 'CMSA' and event['Tier'] == 'U13T1':
                U13_practice = ['CMSA','U13T1S', '', '', '', 'P']
                special_practices.append(U13_practice)

        events.apply(special_detection, axis=1)
        special_df = pd.DataFrame(special_practices, columns=['League', 'Tier', 'Div', 'Practice_Type', 'Num', 'Type'])
        special_df['LeagueTier'] = special_df['League'] + special_df['Tier']
        special_df = special_df.drop_duplicates(subset='LeagueTier', keep='first')
        special_df.set_index('LeagueTier', inplace=True)
        
        events = pd.concat([events, special_df], axis=0)

        # Prepare columns of empty lists
        events['Unwanted'] = np.empty((len(events), 0)).tolist()
        events['Incompatible'] = np.empty((len(events), 0)).tolist()
        events['Pair_with'] = np.empty((len(events), 0)).tolist()
        events['Preference'] = np.empty((len(events), 0)).tolist()
        events['Part_assign'] = '*'
        
        # Add all unwanted slots to the list in 'Unwanted' column for each event mentioned
        for unwanted in split_data[7].split('\n'):
            unwanted = unwanted.replace(' ','')
            if len(unwanted)>0:
                event, day, time = unwanted.strip().split(',')
                if not event in events.index:
                    print(f'Unwanted entry error: {event} is not in table')
                else:
                    event = _PrivateParser.get_index(event)
                    events.at[event, 'Unwanted'].append(day + time)
                    
        # Add all slot preferences to the preference and preference value
        # columns for each event mentioned
        for pref in split_data[8].split('\n'):
            pref = pref.replace(' ','')
            if len(pref) > 0:
                # print(pref)
                day, time, index, value = pref.strip().split(',')
                if not index in events.index:
                    print(f'Preference entry error: {index} is not in table')
                else:
                    events.at[index, 'Preference'].append((day + time, value))

        for pair in split_data[9].split('\n'):
            string = pair.replace(' ', '')
            if len(string) > 0: # If not empty
                # print(string)
                event1, event2 = string.strip().split(',')
                if event1 in events.index and event2 in events.index:
                    events.at[event1, 'Pair_with'].append(event2)
                    events.at[event2, 'Pair_with'].append(event1)
                else:
                    print(f'Pairs entry error: {event1} or {event2} is not in table')    

        # Partial assignments used when f_trans selects branchs, before random choices
        partial_assignments = []
        for assign in split_data[10].split('\n'):
            string = assign.replace(' ', '')
            if len(string) > 0: # If not empty
                team, day, time = string.strip().split(',')
                partial_assignments.append((team, day+time))
                if team in events.index:
                    events.at[team, 'Part_assign'] = day+time
                else:
                    print(f'Part Assign entry error: {team} is not in table')
        events['Part_assign'] = events['Part_assign'].astype(str)

        special_practices = ['CMSAU12T1S', 'CMSAU13T1S']
        df = events[(events['Part_assign'] != "TU1800") & (events['Part_assign'] != "*")]
        print(df)
        for special in special_practices:
            if special in events.index:
                if special in df.index:
                    print(f'Special Practice error: {special} is assigned to slot other than TU1800')
                else:
                    events.at[special, 'Part_assign'] = 'TU1800'
        if verbose:
            print(f'Special practices: \n{special_df}\n')
            print(f'Not compatible: \n{events["Incompatible"]}\n')
            print(f'Unwanted slots: \n{events["Unwanted"]}\n')
            print(f'Preferences: \n{events["Preference"]}\n')
            print(f'Pairs: \n{events["Pair_with"]}\n')
            print(f'Partial Assignments: {partial_assignments}')
        return events
    
    @staticmethod
    def get_index(event: str):
        """
        Convert a string to desired index in DataFrame
        """
        return event.replace(' ', '')
