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

        self.environment = env

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

    def another_incompatible(self, scheduled_events, incompatible_list, event_type):

        valid_indices = [index for index in incompatible_list if index in scheduled_events.index]

        relevant_events = scheduled_events.loc[valid_indices, ['Assigned', 'Type']]

        overlapping_slots = []

        for _, detail in relevant_events.iterrows():
            overlaps = self.environment.overlaps(detail['Assigned'], detail['Type'], event_type)
            if overlaps:
                overlapping_slots.extend(overlaps)
            # overlapping_slots.extend(self.environment.overlaps(detail['Assigned'], detail['Type'], event_type))

        return overlapping_slots

    def avoid_u15_plus(self, scheduled_events):
        related_events = scheduled_events[
                    (scheduled_events['Tier'].str.startswith(('U15', 'U16', 'U17','U19'))) &
                    (scheduled_events['Type'] == 'G')
                ]
        
        overlapping_slots = []
        
        for _, detail in related_events.iterrows():
            overlaps = self.environment.overlaps(detail['Assigned'], detail['Type'], 'G')
            if overlaps:
                overlapping_slots.extend(overlaps)
            # overlapping_slots.extend(self.environment.overlaps(detail['Assigned'], detail['Type'], 'G'))

        return overlapping_slots
    
    def check_game_practice_pair(self, scheduled_events, event, event_type):

        overlapping_slots = []

        if event_type == 'P':
            #print("game to find:", event['Corresp_game'])
            #print("scheduled events: ", scheduled_events)
            if event['Corresp_game'] in scheduled_events.index:
                related_events = scheduled_events.loc[[event['Corresp_game']]]
                #print("related events in there", related_events)
            else:
                related_events = scheduled_events[
                    (scheduled_events.index.str.startswith(event['Corresp_game']))
                ]
                #print("related events not in there", related_events)
            for _, detail in related_events.iterrows():
                # overlapping_slots.extend(self.environment.overlaps(detail['Assigned'], detail['Type'], 'P'))
                overlaps = self.environment.overlaps(detail['Assigned'], detail['Type'], 'P')
                if overlaps:  # Check if overlaps is not None
                    overlapping_slots.extend(overlaps)
        else:
            matches = scheduled_events['Corresp_game'].apply(lambda x: event.name.startswith(x) if x else False)

            related_events = scheduled_events[matches]
            
            for _, detail in related_events.iterrows():
                # overlapping_slots.extend(self.environment.overlaps(detail['Assigned'], detail['Type'], 'G'))
                overlaps = self.environment.overlaps(detail['Assigned'], detail['Type'], 'G')
                if overlaps:  # Check if overlaps is not None
                    overlapping_slots.extend(overlaps)

        return overlapping_slots
    
    def another_check_evening_div(self,event_type):
        if event_type == 'G':
            return self.environment.not_evening_gslots()
        
        return self.environment.not_evening_pslots()
