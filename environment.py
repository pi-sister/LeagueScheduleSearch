# %% [markdown]
# # Imports and Variables

# %%
import os
import re
import pandas as pd
import numpy as np


# %%
class Environment:
    """
    Environment class for processing and managing game and practice schedules.
        w_minfilled (int): Weight for minimum filled slots.
        w_pref (int): Weight for preferences.
        w_pair (int): Weight for pairs.
        w_secdiff (int): Weight for section differences.
        pen_gamemin (int): Penalty for minimum games.
        pen_pracmin (int): Penalty for minimum practices.
        pen_notpaired (int): Penalty for not paired events.
        pen_section (int): Penalty for section violations.
        game_slots (pd.DataFrame): DataFrame containing game slots and their properties.
        practice_slots (pd.DataFrame): DataFrame containing practice slots and their properties.
        events (pd.DataFrame): DataFrame containing combined games and practices with additional properties.
    Methods:
        __init__(file_name: str = None, integers: list = None, verbose=1):
            Initializes the environment with the given file and integer parameters.
        w_minfilled:
            Returns the weight for minimum filled slots.
        w_pref:
            Returns the weight for preferences.
        w_pair:
            Returns the weight for pairs.
        w_secdiff:
            Returns the weight for section differences.
        pen_gamemin:
            Returns the penalty for minimum games.
        pen_pracmin:
            Returns the penalty for minimum practices.
        pen_notpaired:
            Returns the penalty for not paired events.
        pen_section:
            Returns the penalty for section violations.
        game_slots:
            Returns the DataFrame containing processed game slots.
        practice_slots:
            Returns the DataFrame containing processed practice slots.
        events:
            Returns the DataFrame containing combined games and practices with additional properties.
        __str__:
            Returns a string representation of the environment.
        event_length:
            Returns the number of events.
        game_slot_num:
            Returns the number of game slots.
        practice_slot_num:
            Returns the number of practice slots.
    """
    
    def __init__(self, file_name: str = None, integers: list = None, verbose=0):
        """
        Initializes the environment with the given file and integer parameters. Attributes do not change after initialization.
        Parameters:
            file_name (str, optional): The name of the file to be processed. If not provided, the user will be prompted to input it.
            integers (list, optional): A list of 8 positive integers representing weights and penalties. If not provided, the user will be prompted to input them.
            verbose (int, optional): Verbosity level for processing. Default is 1.
        Raises:
            ValueError: If any of the integers provided are negative or if the file cannot be opened.
        Attributes:
            __w_minfilled (int): Weight for minimum filled slots.
            __w_pref (int): Weight for preferences.
            __w_pair (int): Weight for pairs.
            __w_secdiff (int): Weight for section differences.
            __pen_gamemin (int): Penalty for minimum games.
            __pen_pracmin (int): Penalty for minimum practices.
            __pen_notpaired (int): Penalty for not paired events.
            __pen_section (int): Penalty for section violations.
            __game_slots (pd.DataFrame): DataFrame containing processed game slots.
            __practice_slots (pd.DataFrame): DataFrame containing processed practice slots.
            __events (pd.DataFrame): DataFrame containing combined games and practices with additional properties.
        Notes:
        - The file is expected to contain specific sections identified by key words (e.g. "Game slots:", "Practice slots:", etc.).
        - The integers list should contain exactly 8 positive integers.
        - The file content is split into sections and processed accordingly.
        - Game slots, practice slots, games, and practices are processed and combined into a DataFrame with additional properties.
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
        games = _PrivateParser.process_games_practices(split_data[4], 'G', verbose)

        # Process practices into a lookup table
        practices = _PrivateParser.process_games_practices(split_data[5], 'P', verbose)

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
    
    @property
    def preassigned_slots(self):
        """Returns the pre-assigned slots for each event.
        """
        return self.__events['Part_assign'].tolist()
    
    def __str__(self):
        return f'Environment: \n {self.__game_slots} \n {self.__practice_slots} \n {self.__events}'
    
    def event_length(self):
        return len(self.__events)
    
    def game_slot_num(self):
        return len(self.__game_slots)
    
    def practice_slot_num(self):
        return len(self.__practice_slots)
    
    def get_pslot_list(self):
        return self.__practice_slots.index.tolist()
    
    def get_gslot_list(self):
        return self.__game_slots.index.tolist()
    
    def overlaps(self, event1, event2):
        """
        Checks if two events overlap in time.
        Args:
            event1 (str): The first event to check.
            event2 (str): The second event to check.
        Returns:
            bool: True if the events overlap, False otherwise.
        """
        day1, start1 = self.__events.loc[event1, ['Day', 'Start']]
        day2, start2 = self.__events.loc[event2, ['Day', 'Start']]
        
        if day1 == day2:
            time_diff = abs(pd.to_datetime(start1) - pd.to_datetime(start2)).seconds / 60
            return time_diff < 60
        elif (day1 == 'MO' and day2 == 'F') or (day1 == 'F' and day2 == 'MO'):
            time_diff = abs(pd.to_datetime(start1) - pd.to_datetime(start2)).seconds / 60
            return time_diff < 120

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
    def process_games_practices(data, event_type, verbose = 0):
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
        
        # Hard constraint: nothing can be scheduled at this time. "Admin meeting"
        if 'TU11:00' in df.index:
            df = df.drop('TU11:00')
        if verbose:
            print(f'Processed {len(df)} {event_type} items: \n {df}\n')
        return df

        
    @staticmethod
    def add_properties(events: pd.DataFrame, split_data: list, verbose = 1):
        """
        Adds various properties to the events DataFrame based on the provided split_data.
        Parameters:
        events (pd.DataFrame): DataFrame containing event data.
        split_data (list): List containing various data segments used for processing.
        verbose (int, optional): Verbosity level. Defaults to 1.
        Returns:
        pd.DataFrame: Updated DataFrame with additional properties.
        The function performs the following operations:
        - Detects special practices and adds them to the DataFrame.
        - Prepares columns of empty lists for 'Unwanted', 'Incompatible', 'Pair_with', and 'Preference'.
        - Adds unwanted slots to the 'Unwanted' column for each event.
        - Adds slot preferences to the 'Preference' column for each event.
        - Adds pairings to the 'Pair_with' column for each event.
        - Processes partial assignments and updates the 'Part_assign' column.
        - Ensures special practices are assigned to a specific slot ('TU1800').
        If verbose is set to 1, prints detailed information about the special practices, incompatible events, unwanted slots, preferences, pairs, and partial assignments.
        """
        

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


        
        # Do we want to add the name of each team's game to the practice?
        def related_games(event):
            if event['Type'] == 'P':
                if 'PRC' in event.index:
                    return event['League'] + event['Tier'] + event['Div']
                else:
                    return event['League'] + event['Tier'] + event['Div']
            else:
                return ''
        
        events['Corresp_game'] = events.apply(related_games, axis=1)
        
        
        special_practices = []
        def special_detection(event):
            if event['League'] == 'CMSA' and event['Tier'] == 'U12T1':
                U12_practice = {'League':'CMSA',
                                'Tier': 'U12T1S', 
                                'Type': 'P',
                                'Div': '',
                                'Practice_Type': '',
                                'Num': '',
                                'Unwanted': [],
                                'Incompatible': [],
                                'Pair_with': [],
                                'Preference': [],
                                'Corresp_game': 'LeagueU12T1',
                                'Part_assign': 'TU1800'}
                special_practices.append(U12_practice)
            elif event['League'] == 'CMSA' and event['Tier'] == 'U13T1':
                U13_practice = {'League':'CMSA',
                                'Tier': 'U13T1S', 
                                'Type': 'P',
                                'Div': '',
                                'Practice_Type': '',
                                'Num': '',
                                'Unwanted': [],
                                'Incompatible': [],
                                'Pair_with': [],
                                'Preference': [],
                                'Corresp_game': 'LeagueU13T1',
                                'Part_assign': 'TU1800'}
                special_practices.append(U13_practice)

        events.apply(special_detection, axis=1)
        special_df = pd.DataFrame(special_practices)
        special_df = special_df.set_index(['League'] + special_df['Tier'])
        special_df = special_df.drop_duplicates(subset=['League', 'Tier'], keep='first')
        special_df = special_df.reindex(columns=events.columns, fill_value=0)
        
        events = pd.concat([events, special_df], axis=0)
        
        if verbose:
            print(events.head())
            print(f'Columns: {events.columns}\n')
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
