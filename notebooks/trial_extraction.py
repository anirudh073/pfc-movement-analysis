import numpy as np
import pandas as pd
import spyglass.common as sgc
import spyglass.position.v1 as sgp
import spyglass.linearization.v1 as sgpl
from spyglass.position import PositionOutput
from functools import reduce

'''
Achieves the following:
1. Identify DIO = 1 at ANY port
2. Consider DIO streams of less than a certain gap continous (consumption)
3. Store timestamps until a second port is licked. Store trial till here: start next trial
4. For every trial --> mark as correct or incorrect based on starting/ending ports.
5. Mark correct trials as outbound or inbound and leftward or rightward
6. Retrieve position information based on the stored timestamps
6. For each trial -> identify "consumption", "turning", and "run" period from linear position and orientation data.
7. Return a final dataframe with "trial number", "timestamps",  "position", "linear position", "velocity", "speed", "arm", "consumption (bool)", "turning (bool)", "correct/incorrect" , (for correct) "outbound/inbound"
'''


def prepare_DIO_data(session_restriction : dict, lick_event_threshold: float = 2):
    """ 
    Args:
        session_restriction (dict): {"nwb_file_name": nwb_file_name}
        lick_event_threshold (seconds): Min temporal gap after which a new lick event is registered 
    """

#---------------------------------------------- Internal functions
    def _drop_backward_timestamps(dataframe):
        """Remove rows from  sgc.DIOEvents() where timestamps drop below the first or the previous timestamp."""
        
        timestamps = dataframe["timestamps"]

        previous_max = timestamps.cummax().shift(fill_value=timestamps.iloc[0])
        mask = timestamps >= previous_max
        
        return dataframe.loc[mask].reset_index(drop=True)
    

    def _merge_on_timestamps(dataframe_list):
        """Merge multiple DIO dataframes by timestamps, sort in ascending order, keep all rows"""
        
        merged_dataframe = reduce(
            lambda left_df, right_df: pd.merge(
                left_df, right_df, on="timestamps", how="outer"
            ),
            dataframe_list,
        )
        merged_dataframe = merged_dataframe.sort_values("timestamps", kind="mergesort")
        merged_dataframe = merged_dataframe.reset_index(drop=True)
        return merged_dataframe.fillna(0)
    

    def _label_lick_events(input_dataframe, time_gap_threshold = 2):
        """Assign a lick number to each lick event separated by gaps > threshold."""
        data_columns = ["left", "middle", "right"]

        # Identify which side is 1 on each row 
        detection_dataframe = input_dataframe.copy()
        detection_dataframe["active_side"] = detection_dataframe[data_columns].idxmax(axis=1)
        detection_dataframe["active_side"] = detection_dataframe["active_side"].where(
            detection_dataframe[data_columns].max(axis=1) == 1,
            other=pd.NA,
        )

        # Keep only rows where some side is 1
        lick_events_dataframe = detection_dataframe[detection_dataframe["active_side"].notna()].copy()
        if lick_events_dataframe.empty:
            return lick_events_dataframe

        # Time gap to previous detection
        lick_events_dataframe["time_difference"] = lick_events_dataframe["timestamps"].diff().fillna(0)

        # Start a new lick when the gap is larger than the threshold
        lick_events_dataframe["lick_number"] = (
            (lick_events_dataframe["time_difference"] > time_gap_threshold).cumsum() + 1
        )

        return lick_events_dataframe[["timestamps", "active_side", "lick_number", *data_columns]]   
    
    
# ---------------------------------------------- Main
    
    # Get dio and time data for each dio event
    middle_dio = pd.DataFrame((sgc.DIOEvents() & session_restriction & {"dio_event_name": "poke_middle"}).fetch_nwb()[0]["dio"].data).rename(columns = {0:"middle"})
    middle_timestamps = pd.DataFrame((sgc.DIOEvents() & session_restriction & {"dio_event_name": "poke_middle"}).fetch_nwb()[0]["dio"].timestamps).rename(columns = {0: "timestamps"})
    
    left_dio = pd.DataFrame((sgc.DIOEvents() & session_restriction & {"dio_event_name": "poke_left"}).fetch_nwb()[0]["dio"].data).rename(columns = {0:"left"})
    left_timestamps = pd.DataFrame((sgc.DIOEvents() & session_restriction & {"dio_event_name": "poke_left"}).fetch_nwb()[0]["dio"].timestamps).rename(columns = {0: "timestamps"})  
    
    right_dio = pd.DataFrame((sgc.DIOEvents() & session_restriction & {"dio_event_name": "poke_right"}).fetch_nwb()[0]["dio"].data).rename(columns = {0:"right"})
    right_timestamps = pd.DataFrame((sgc.DIOEvents() & session_restriction & {"dio_event_name": "poke_right"}).fetch_nwb()[0]["dio"].timestamps).rename(columns = {0: "timestamps"})

    # Merge dio and time data
    poke_middle_df = _drop_backward_timestamps(
        pd.concat([middle_timestamps, middle_dio], axis = 1))
    
    poke_left_df = _drop_backward_timestamps(
        pd.concat([left_timestamps, left_dio], axis = 1))
    
    poke_right_df = _drop_backward_timestamps(
        pd.concat([right_timestamps, right_dio], axis = 1))
    

    #Merge dataframes
    dio_df = _merge_on_timestamps([poke_left_df, poke_right_df, poke_middle_df])
    #Compute lick events
    lick_events_dataframe = _label_lick_events(dio_df, lick_event_threshold)
    
    #Add trial numbers
    active_side_series = lick_events_dataframe["active_side"]
    change_mask = (
        active_side_series.ne(active_side_series.shift())
        & active_side_series.notna()
        & active_side_series.shift().notna()
    )
    increment_mask = change_mask.shift(fill_value=False)
    
    lick_events_dataframe["trial_number"] = (increment_mask.cumsum() + 1).where(active_side_series.notna())   
    
    return lick_events_dataframe



def prepare_trial_data(lick_events_dataframe: pd.DataFrame):
    """_summary_

    Args:
        lick_events_dataframe (pd.DataFrame): _description_
    """
    inbound_left_templates = [('middle', 'left', 'middle'), ('right', 'left', 'middle')]
    inbound_right_templates = [('left', 'right', 'middle'), ('middle', 'right', 'middle')]

    outbound_left_templates = [('right', 'middle', 'left')]
    outbound_right_templates = [('left', 'middle', 'right')]


    trials = []
    trial_starts = []
    trial_ends = []
    for trial in np.arange(1, lick_events_dataframe["trial_number"].max()):
        trial_start = (lick_events_dataframe[lick_events_dataframe["trial_number"]==trial].reset_index()["timestamps"].iloc[0])
        trial_end = (lick_events_dataframe[lick_events_dataframe["trial_number"]==trial].reset_index()["timestamps"].iloc[-1])
        trials.append(trial)
        trial_starts.append(trial_start)
        trial_ends.append(trial_end)

    
    
    trials_df = pd.DataFrame({"trial_number": trials, "trial_start": trial_starts, "trial_end": trial_ends})
    # compute duration for each trial (end - start)
    trials_df["trial_duration (s)"] = trials_df["trial_end"] - trials_df["trial_start"]

    trial_labels = []
    trial_types = []
    trial_directions = []
    for trial in np.arange(1, lick_events_dataframe["trial_number"].max()):

        arm = lick_events_dataframe[lick_events_dataframe["trial_number"]==trial]["active_side"].reset_index()["active_side"].iloc[0]
        next_arm = lick_events_dataframe[lick_events_dataframe["trial_number"]==(trial+1)]["active_side"].reset_index()["active_side"].iloc[0]


        try:
            previous_arm = lick_events_dataframe[lick_events_dataframe["trial_number"]==(trial-1)]["active_side"].reset_index()["active_side"].iloc[0]
        except IndexError:
            continue
        trial_directions.append((previous_arm, arm, next_arm))


        if (previous_arm, arm, next_arm) in [("left", "middle", "right"), ("right", "middle", "left")]:
            trial_labels.append("correct")
            trial_types.append("outbound")
        
        elif (arm, next_arm) in [("left", "middle"), ("right", "middle")]:
            trial_labels.append("correct")
            trial_types.append('inbound')
            
        else: 
            trial_labels.append("error")
            trial_types.append("NA")
        
        
    trial_labels.insert(0, "error")
    trial_types.insert(0, "NA")
    trial_directions.insert(0, (lick_events_dataframe[lick_events_dataframe["trial_number"]==1]["active_side"].reset_index()["active_side"].iloc[0],
                                lick_events_dataframe[lick_events_dataframe["trial_number"]==2]["active_side"].reset_index()["active_side"].iloc[0]))


    trials_df["trial_label"] = trial_labels
    trials_df["trial_type"] = trial_types
    trials_df["trial_direction (previous, current, next)"] = trial_directions

    trials_df["left/right"] = np.nan  
    td = trials_df['trial_direction (previous, current, next)']
    left_mask = td.isin(inbound_left_templates + outbound_left_templates)
    right_mask = td.isin(inbound_right_templates + outbound_right_templates)
    trials_df.loc[left_mask, "left/right"] = "left"
    trials_df.loc[right_mask, "left/right"] = "right"

    
    return trials_df



# def merge_trial_df_with_target(target_dataframe, trials_dataframe):
#     """ merge a trial dataframe with a target (containing 2D or linearized position data)
#         target_dataframe must share timestamps with trials_dataframe

#     Args:
#         position_dataframe (pd.DataFrame): Assuming a "time" index exists
#         trials_dataframe (pd.DataFrame): Output of prepare_trial_data()
#     """
    
#     trials_sorted = trials_dataframe.sort_values("trial_start")
#     targets_sorted = target_dataframe.sort_index()
    
#     merged_dataframe = pd.merge_asof(
#         targets_sorted,
#         trials_sorted,
#         left_index= True,
#         right_on = "trial_start",
#         direction = "backward"
#     )
    
#     merged_dataframe = merged_dataframe[merged_dataframe.index <= merged_dataframe["trial_end"]]
#     merged_dataframe["time_since_trial_start"] = merged_dataframe.index - merged_dataframe["trial_start"]
#     merged_dataframe["trial_progress"] = merged_dataframe["time_since_trial_start"]/merged_dataframe["trial_duration (s)"]


#     return merged_dataframe

def merge_trial_df_with_target(target_dataframe, trials_dataframe, reward_ranges = (
    (0, 25),
    (260, 285),
    (431, 456)
)):
    """ merge a trial dataframe with a target (containing 2D or linearized position data)
        target_dataframe must share timestamps with trials_dataframe

    Args:
        position_dataframe (pd.DataFrame): Indexed by time in UNIX format
        trials_dataframe (pd.DataFrame): Output of prepare_trial_data()
        reward_ranges (list of tuples): tuples representing reward zone ranges in terms of linear positons
    """
    
    trials_sorted = trials_dataframe.sort_values("trial_start")
    targets_sorted = target_dataframe.sort_index()

    trial_intervals = pd.IntervalIndex.from_arrays(
        trials_sorted["trial_start"],
        trials_sorted["trial_end"],
        closed="both",
    )
    interval_codes = trial_intervals.get_indexer(targets_sorted.index)
    valid_mask = interval_codes >= 0

    merged_dataframe = targets_sorted.loc[valid_mask].copy()
    trials_values = trials_sorted.reset_index(drop=True)
    for column_name in trials_sorted.columns:
        merged_dataframe[column_name] = trials_values.loc[
            interval_codes[valid_mask], column_name
        ].to_numpy()

    merged_dataframe["time_since_trial_start"] = merged_dataframe.index - merged_dataframe["trial_start"]
    merged_dataframe["trial_progress"] = merged_dataframe["time_since_trial_start"]/merged_dataframe["trial_duration (s)"]
    merged_dataframe = merged_dataframe.drop("time_since_trial_start", axis = 1)
    

    in_reward_range = False
    for start_pos, end_pos in reward_ranges:
        in_reward_range |= merged_dataframe["linear_position"].between(start_pos, end_pos, inclusive="both")

    forward_band = merged_dataframe["orientation"].between(-np.pi/6, np.pi/6, inclusive="both")
    turn_band = ~forward_band  

    merged_dataframe["zone"] = "run"
    merged_dataframe.loc[in_reward_range & forward_band, "zone"] = "reward"
    merged_dataframe.loc[in_reward_range & turn_band, "zone"] = "turn"


    return merged_dataframe


def plot_background_position(position_dataframe, axes, position_columns = (
    "position_x", "position_y"), background_color = "black"):
    """ Plot all position data from a dataframe as a black, low opacity background

    Args:
        position_dataframe (pd.DataFrame): Must contain x and y position coordinates
        axes (plt axes): axes the plot should be drawn on
        position_columns (tuple): x and y (in order) position column names in position_dataframe
    """
    
    position_dataframe.plot.scatter(x = position_columns[0],
                                    y = position_columns[1],
                                    s = 3,
                                    alpha = 0.1,
                                    color = background_color,
                                    ax = axes)
    
    
    
    
    
    
    
    
