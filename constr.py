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

    def __init__(self, gslot_df, pslot_df,events_df):
        """
        Initializes the Constr with the abstracted constraints from the environment and
        each game/practice and their slots variable.

            Parameters:
                game_slots (list): list of abstracted game slots
                practice_slots (list): list of abstracted practice slots
                env (dictionary): contains all the necessary information to complete orTree (particularly
                for the constr function)
        """
        # populate local variables
        self.gslots_and_info = gslot_df
        self.pslots_and_info = pslot_df
        self.events_and_info = events_df
        self.game_slot_num = len(gslot_df.index)
        self.practice_slot_num = len(pslot_df.index)

        # Variables for max exceeded
        self.game_counter = [0] * self.game_slot_num
        self.practice_counter = [0] * self.practice_slot_num

        self.game_slot_lookup = {}
        self.practice_slot_lookup = {}
    
    # Change this so we accept one slot and create a counter that will retunr T or F
    # Find a way to reset
    def max_exceeded_reset(self):
        # Lists to count the # of occurrences of each slot in the schedule
        self.game_counter = [0] * self.game_slot_num
        self.practice_counter = [0] * self.practice_slot_num

        # Disctionaries to lookup slot indices based on slot name
        self.game_slot_lookup = dict(zip(list(self.gslots_and_info.index), list(range(0,self.game_slot_num))))
        self.practice_slot_lookup = dict(zip(list(self.pslots_and_info.index), list(range(0,self.practice_slot_num))))


    def max_exceeded(self, slot, slot_type):
        if slot_type == 'G':
            self.game_counter[self.game_slot_lookup[slot]] += 1
            if self.game_counter[self.game_slot_lookup[slot]] > self.gslots_and_info.loc[slot,'Max']:
                return False
            
        if slot_type == 'P':
            self.practice_counter[self.practice_slot_lookup[slot]] += 1
            if self.practice_counter[self.practice_slot_lookup[slot]] > self.pslots_and_info.loc[slot,'Max']:
                return False
            
        return True