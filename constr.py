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
                evening: A time object that indicates the minimum time for an evening slot
                __u13t1s: A boolean indicating whether to consider the special U13T1S practices
                __u12t1s: A boolean indicating whether to consider the special U12T1S practices
                u15_plus_slots: A set of strings representing U15 - U19 game slots. 
                    Only considers game comparisons and not practices.
                incompatible_checker: A set of tuples representing already checked incompatible lists.
                    Considers both practices and game overlap.
        """

        self.evening = datetime.strptime("18:00", "%H:%M").time()

        self.__u13t1s = env.events["Tier"].isin(["U13T1"]).any()
        self.__u12t1s = env.events["Tier"].isin(["U12T1"]).any()

        self.u15_plus_slots = set()
        self.incompatible_checker = set()
    
    @property
    def special_events(self):
        """
        A boolean which indicates whether special events are considered or not
        """
        return (self.__u13t1s or self.__u12t1s)


    def reset_slots(self):
        """
        Indicates a new partial or complete schedule is being checked and to dump previously counted
        slots.
        """
        self.u15_plus_slots = set()
        self.incompatible_checker = set()
    
    
    def incompatible(self, event_assignments, incompatible_list, event_type, event_time):
        """
        Takes the incompatible list from an event, if there is one, and checks if there are overlaps
        in the scheduled times
        """
        if not incompatible_list:
            return True
        
        relevant_events = event_assignments.loc[incompatible_list]

        # Already compared this incompatible list
        if tuple(relevant_events) in self.incompatible_checker:
            return True
        
        self.incompatible_checker.add(tuple(relevant_events))

        return self.__check_time_overlap(self, event_time, event_type, relevant_events)
        
    def check_assign(self, df_info, curr_tier, curr_time, corresponding_game, mode):
        """
        Checks the assignments of events under 3 modes
        Mode: regcheck
            - Checks for time overlaps between U15 - U19 Games. (Not practices)
        Mode: specialcheck
            - Checks for time overlaps between the U13T1S/U12T1S special practices with the
              U13T1/U12T1 games/practices
        Mode: specialcheck
            - Checks for time overlaps between the U13T1S/U12T1S special practices with the
              U13T1/U12T1 games/practices
        Mode: pcheck
            - Checks for time overlaps between a practice booking and it's corresponding game
        """
        if mode == "regcheck":
            if not curr_tier.startswith(('U15', 'U16', 'U17', 'U18', 'U19')):
                return True

            if curr_time == "*":
                return True
            
            print(f'Time: {curr_time} \t Set{self.u15_plus_slots}')

            if curr_time in self.u15_plus_slots:
                return False
            
            self.u15_plus_slots.add(curr_time)
            return True

        related_events = None

        if mode == "specialcheck":
            if (self.__u13t1s and curr_tier.startswith('U13T1S')):
                related_events = df_info[
                    (df_info['Tier'].str.startswith('U13T1')) &
                    ~(df_info['Tier'] == 'U13T1S')
                ][['Assigned','Type']]

            if (self.__u12t1s and curr_tier.startswith('U12T1S')):
                related_events = df_info[
                    (df_info['Tier'].str.startswith('U12T1')) &
                    ~(df_info['Tier'] == 'U12T1S')
                ][['Assigned','Type']]

            if related_events is None:
                return True

            return self.__check_time_overlap(curr_time, "P", related_events)

        if mode == "pcheck":
            related_event = df_info.loc[corresponding_game, ['Assigned', 'Type']]

            return self.__check_time_overlap(curr_time, "P", related_event)
    
    def __check_time_overlap(self, curr_time, event_type, corresponding_events):
        """
        With the given input check if any of the corresponding events overlap with the current event
        """
        event_day = curr_time[:2]
        event_time = datetime.strptime(curr_time[2:], "%H:%M").time()
        event_datetime = datetime.combine(datetime.min, event_time)

        if event_type == "P":
            if (event_day == 'MO') or (event_day == 'TU'):
                event_duration = timedelta(hours=1)
            else:
                event_duration = timedelta(hours=2)
        else: 
            if (event_day == 'MO'):
                event_duration = timedelta(hours=1)
            else:
                event_duration = timedelta(hours=1,minutes=30)
        
        event_end_datetime = event_datetime + event_duration

        event_end = event_end_datetime.time()

        for _, detail in corresponding_events.iterrows():
            if event_type == "P":
                if not self.__practice_time_compare(curr_time, event_day, event_time, event_end, detail):
                    return False
            else:
                if not self.__game_time_compare(curr_time, event_day, event_time, event_end, detail):
                    return False
        
        return True
    
    def __practice_time_compare(self, practice_slot, practice_day, practice_time, practice_end, event_detail):
        """
        Compares time overlap between a practice and some event
        """
        result = True

        if event_detail['Type'] == 'G':

            if ((practice_day == 'FR') and (event_detail['Assigned'][:2] == 'MO')):
                game_start = datetime.strptime(event_detail['Assigned'][2:], "%H:%M").time()

                result = not((practice_time <= game_start) and (game_start < practice_end))
            
            elif ((practice_day == 'TU') and (event_detail['Assigned'][:2] == 'TU')):
                game_start = datetime.strptime(event_detail['Assigned'][2:], "%H:%M").time()

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
                result = not(practice_slot == event_detail['Assigned'])
        else: 
            result = not(practice_slot == event_detail['Assigned'])

        if not result:
            return False
        
        return True
    
    def __game_time_compare(self, game_slot, game_day, game_time, game_end, event_detail):
        """
        Compares time overlap between a game and some event
        """
        result = True
        
        if event_detail['Type'] == 'P':

            if ((event_detail['Assigned'][:2] == 'FR') and (game_day == 'MO')):
                practice_start = datetime.strptime(event_detail['Assigned'][2:], "%H:%M").time()

                # Convert time to datetime (use a dummy date, e.g., '1900-01-01')
                practice_start_datetime = datetime.combine(datetime.min, practice_start)

                # Add 2 hours
                practice_duration = timedelta(hours=2)
                practice_end_datetime = practice_start_datetime + practice_duration

                # Convert back to time object
                practice_end = practice_end_datetime.time()

                result = not((practice_start <= game_time) and (game_time < practice_end))
            
            elif ((event_detail['Assigned'][:2] == 'TU') and (game_day == 'TU')):
                practice_start = datetime.strptime(event_detail['Assigned'][2:], "%H:%M").time()

                # Convert time to datetime (use a dummy date, e.g., '1900-01-01')
                practice_start_datetime = datetime.combine(datetime.min, practice_start)

                # Add 1 hour 
                practice_duration = timedelta(hours=1)
                practice_end_datetime = practice_start_datetime + practice_duration

                # Convert back to time object
                practice_end = practice_end_datetime.time()

                result = not(
                    ((game_time <= practice_start) and (practice_start < game_end)) and 
                    ((game_time <= practice_end) and (practice_start < practice_end))
                )
            else: 
                result = not(game_slot == event_detail['Assigned'])
        else: 
            result = not(game_slot == event_detail['Assigned'])

        if not result:
            return False
        
        return True

    def check_evening_div(self, time_string, division):
        """
        Checks if the event is in division 9, and if so if it was assigned to
        and evening slot
        """
        if division != '09':
            return True
        
        event_time = datetime.strptime(time_string, "%H:%M").time()

        return self.evening >= event_time


