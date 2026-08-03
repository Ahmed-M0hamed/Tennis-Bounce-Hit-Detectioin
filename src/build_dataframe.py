from typing import List 
from .utils import get_bottom_center_of_player , distance_between_two_points 
from .homography import transform_ball_players_court_position
from scipy.signal import savgol_filter 
import pandas as pd 
import numpy as np 
class DataFrameBuilding: 
    def __init__(self , max_gap_frames:int  , min_window_size :int ) : 
        self.max_gap_frames = max_gap_frames
        self.min_window_size = min_window_size 
    def _find_gaps(self , annotations : List ) -> List  :
        gap_start = 1
        index = 0 
        gaps = []
        while index < len(annotations): 
            if 'ball_position' in annotations[index] : 
                gap_end = annotations[index]['frame_id'] 
                if gap_end - gap_start > self.max_gap_frames : 
                    gaps.append((gap_start , gap_end)) 
                gap_start = gap_end 
            index += 1 
        return gaps 

    def _create_windows(self , annotations :List ) -> List : 
        gaps = self._find_gaps(annotations) 
        windows = []
        if gaps[0][0] != 1 :
            window_start = 1 
            for gap in gaps : 
                window_end = gap[0] 
                if window_end - window_start > self.min_window_size : 
                    windows.append((window_start , window_end)) 
                window_start = gap[1]
        else:  
            window_start = gaps[0][1]
            for gap in gaps[1:] : 
                window_end = gap[0] 
                if window_end - window_start > self.min_window_size : 
                    windows.append((window_start , window_end)) 
                window_start = gap[1]
        return windows 

    def _smooth_trajectory(self ,  df, window = 9, poly = 2) :
                """
                Fill small detection gaps then apply Savitzky-Golay smoothing.
                Why Savitzky-Golay?  It fits a polynomial locally, which preserves
                the sharp peak at a bounce better than a simple moving average.
                """
                # Reindex to dense frame range so gaps become NaN
                full_idx = pd.RangeIndex(df["frame"].min(), df["frame"].max() + 1)
                df = (
                    df.set_index("frame")
                    .reindex(full_idx)
                    .rename_axis("frame")
                    .reset_index()
                )
                for col in ["ball_x", "ball_y", "player_1_x", "player_1_y", "player_2_x", "player_2_y"]:
                    valid = df[col].notna() 
                    if valid.sum() > window:
                        df.loc[valid, f"{col}_smoothed"] = savgol_filter(
                            df.loc[valid, col], window_length=window, polyorder=poly
                        )
                    else:
                        df[f"{col}_smoothed"] = df[col]
                         
                
                return df
    def _turn_window_into_dataframe(self  , annotations_window ) : 
        dataframe_rows = []
        
        for annotation in annotations_window : 

            player_1_position = get_bottom_center_of_player(annotation['persons'][0]['xyxy']) if 'persons' in annotation and annotation['persons'] else [None , None]
            player_1_id = annotation['persons'][0]['id'] if 'persons' in annotation and annotation['persons'] else None 
            player_2_position = get_bottom_center_of_player(annotation['persons'][1]['xyxy']) if 'persons' in annotation and len(annotation['persons']) == 2 else [None , None]
            player_2_id = annotation['persons'][1]['id'] if 'persons' in annotation and len(annotation['persons']) == 2 else None
            if 'ball_position' in annotation and 'persons' in annotation and len(annotation['persons']) == 2 and 'court_points' in annotation : 
                player_1_position = get_bottom_center_of_player(annotation['persons'][0]['xyxy'])
                player_1_id = annotation['persons'][0]['id']
                player_2_position = get_bottom_center_of_player(annotation['persons'][1]['xyxy'])
                player_2_id = annotation['persons'][1]['id']
                transformed_ball , transformed_player_1, transformed_player_2 , transformed_corners = transform_ball_players_court_position(annotation['court_points'] ,
                                    annotation['ball_position'], player_1_position , player_2_position)
                flaten_corners = [item for corner in transformed_corners for item in corner]
                flaten_row = [*transformed_ball , *transformed_player_1 , player_1_id , *transformed_player_2 , player_2_id, *flaten_corners]
                flaten_row.append(annotation['frame_id'])
                if 'serve' in annotation or 'ball_hit' in annotation : 
                    flaten_row.append('ball_hit') 
                elif 'ball_bounced' in annotation : 
                    flaten_row.append('ball_bounced') 
                else :
                    flaten_row.append('no_event') 

                dataframe_rows.append(flaten_row) 
            
            elif 'ball_position' not in annotation and 'persons' in annotation and len(annotation['persons']) == 2 and 'court_points' in annotation : 
                    transformed_ball , transformed_player_1, transformed_player_2 , transformed_corners = transform_ball_players_court_position(annotation['court_points'] ,
                                        [None , None], player_1_position , player_2_position)
                    flaten_corners = [item for corner in transformed_corners for item in corner]
                    flaten_row = [*[None , None] , *transformed_player_1 , player_1_id, *transformed_player_2 , player_2_id, *flaten_corners]
                    flaten_row.append(annotation['frame_id'])
                    if 'serve' in annotation or 'ball_hit' in annotation : 
                        flaten_row.append('ball_hit') 
                    elif 'ball_bounced' in annotation : 
                        flaten_row.append('ball_bounced') 
                    else :
                        flaten_row.append('no_event') 

                    dataframe_rows.append(flaten_row) 
            else : 
                flaten_row = [*[None , None] , *[None , None] , None , *[None , None] , None , *[None , None , None , None , None , None, None ,None] ]
                flaten_row.append(annotation['frame_id'])
                if 'serve' in annotation or 'ball_hit' in annotation : 
                    flaten_row.append('ball_hit') 
                elif 'ball_bounced' in annotation : 
                    flaten_row.append('ball_bounced') 
                else :
                    flaten_row.append('no_event') 

                dataframe_rows.append(flaten_row)
       

        df_ball_positions = pd.DataFrame(dataframe_rows,columns=['ball_x' , 'ball_y' ,'player_1_x' ,
                        'player_1_y', 'player_1_id' , 'player_2_x' , 'player_2_y' , 'player_2_id', 'top_left_x' , 'top_left_y' 
                        , 'top_right_x' , 'top_right_y' , 'bottom_left_x' , 'bottom_left_y' 
                        , 'bottom_right_x' , 'bottom_right_y' ,'frame' , 'label']) 
        return df_ball_positions 
    def _data_engineering(self, df ) : 
        df_ball_positions = df
        df_ball_positions[['ball_x' , 'ball_y', 'player_1_x' , 'player_1_y' , 'player_2_x' , 'player_2_y' , 'top_left_x' , 'top_left_y' 
                        , 'top_right_x' , 'top_right_y' , 'bottom_left_x' , 'bottom_left_y' , 'bottom_right_x' , 'bottom_right_y']]= df_ball_positions[['ball_x' , 'ball_y' ,'player_1_x' , 'player_1_y' , 'player_2_x' , 'player_2_y' ,'top_left_x' , 'top_left_y' 
                        , 'top_right_x' , 'top_right_y' , 'bottom_left_x' , 'bottom_left_y' 
                        , 'bottom_right_x' , 'bottom_right_y']].interpolate()
        df_ball_positions[['ball_x' , 'ball_y','player_1_x' , 'player_1_y' , 'player_2_x' , 'player_2_y' , 'top_left_x' , 'top_left_y' 
                        , 'top_right_x' , 'top_right_y' , 'bottom_left_x' , 'bottom_left_y' 
                        , 'bottom_right_x' , 'bottom_right_y']] = df_ball_positions[['ball_x' , 'ball_y' , 'player_1_x' , 'player_1_y' , 'player_2_x' , 'player_2_y' , 'top_left_x' , 'top_left_y' 
                        , 'top_right_x' , 'top_right_y' , 'bottom_left_x' , 'bottom_left_y' 
                        , 'bottom_right_x' , 'bottom_right_y']].bfill()
        smothed_df = self._smooth_trajectory(df_ball_positions ) 
        smothed_df['ball_y_rolling_mean'] = smothed_df['y_ball_smooth'].rolling(window=5, min_periods=1, center=False).mean()
        smothed_df['ball_x_rolling_mean'] = smothed_df['x_ball_smooth'].rolling(window=5, min_periods=1, center=False).mean()
        smothed_df['vy_ball'] = np.gradient(smothed_df['ball_y_rolling_mean'].values)
        smothed_df['vx_ball'] = np.gradient(smothed_df['ball_x_rolling_mean'].values) 
        smothed_df['ay_ball'] = np.gradient(smothed_df['vy_ball'].values)
        smothed_df['ax_ball'] = np.gradient(smothed_df['vx_ball'].values)  
        smothed_df['ball_speed'] =np.sqrt(smothed_df["vx_ball"]**2 +smothed_df["vy_ball"]**2)
        smothed_df['ball_acc'] = np.sqrt(smothed_df["ax_ball"]**2 +smothed_df["ay_ball"]**2)
        smothed_df["ball_direction"] = np.arctan2(smothed_df["vy_ball"],smothed_df["vx_ball"])
        smothed_df["dir_sin"] = np.sin(smothed_df["ball_direction"])
        smothed_df["dir_cos"] = np.cos(smothed_df["ball_direction"])
        smothed_df["velocity_mean"] = (smothed_df["ball_speed"].rolling(5, center=True).mean())
        smothed_df["velocity_std"] = (smothed_df["ball_speed"].rolling(5, center=True).std())
        smothed_df["acc_mean"] = (smothed_df["ball_acc"].rolling(5, center=True).mean())
        smothed_df["acc_std"] = (smothed_df["ball_acc"].rolling(5, center=True).std())
        smothed_df['player_1_x_rolling_mean'] = smothed_df['player_1_x_smoothed'].rolling(window=5, min_periods=1, center=False).mean()
        smothed_df['player_1_y_rolling_mean'] = smothed_df['player_1_y_smoothed'].rolling(window=5, min_periods=1, center=False).mean()
        smothed_df['player_2_x_rolling_mean'] = smothed_df['player_2_x_smoothed'].rolling(window=5, min_periods=1, center=False).mean()
        smothed_df['player_2_y_rolling_mean'] = smothed_df['player_2_y_smoothed'].rolling(window=5, min_periods=1, center=False).mean()
        smothed_df['vy_player_2'] = np.gradient(smothed_df['player_2_y_rolling_mean'].values)
        smothed_df['vx_player_2'] = np.gradient(smothed_df['player_2_x_rolling_mean'].values) 
        smothed_df['ay_player_2'] = np.gradient(smothed_df['vy_player_2'].values)
        smothed_df['ax_player_2'] = np.gradient(smothed_df['vx_player_2'].values)  
        smothed_df['player_2_speed'] =np.sqrt(smothed_df["vx_player_2"]**2 +smothed_df["vy_player_2"]**2)
        smothed_df['player_2_acc'] = np.sqrt(smothed_df["ax_player_2"]**2 +smothed_df["ay_player_2"]**2)
        smothed_df['vy_player_1'] = np.gradient(smothed_df['player_1_y_rolling_mean'].values)
        smothed_df['vx_player_1'] = np.gradient(smothed_df['player_1_x_rolling_mean'].values) 
        smothed_df['ay_player_1'] = np.gradient(smothed_df['vy_player_1'].values)
        smothed_df['ax_player_1'] = np.gradient(smothed_df['vx_player_1'].values)  
        smothed_df['player_1_speed'] =np.sqrt(smothed_df["vx_player_1"]**2 +smothed_df["vy_player_1"]**2)
        smothed_df['player_1_acc'] = np.sqrt(smothed_df["ax_player_1"]**2 +smothed_df["ay_player_1"]**2)
        player_1_pos = list(zip(smothed_df['player_1_x'] , smothed_df['player_1_y']))
        player_2_pos = list(zip(smothed_df['player_2_x'] , smothed_df['player_2_y'])) 
        ball_pos = list(zip(smothed_df['ball_x'] , smothed_df['ball_y'])) 
        nearest_ball_player_distances = []
        for p1 ,p2 , b in zip(player_1_pos , player_2_pos , ball_pos) : 
            distances = [distance_between_two_points(b , p) for p in [p1,p2]] 
            index = np.argmin(distances) 
            nearest_ball_player_distances.append(distances[index]) 
        smothed_df['nearest_player_to_ball'] = nearest_ball_player_distances 
        smothed_df['nearest_player_rate'] = np.gradient(smothed_df['nearest_player_to_ball'].values) 
        return smothed_df

    def _check_window_events(self , annotations_window) : 
        events = ['serve' , 'ball_bounced' , 'ball_hit'] 
        state = False 
        for annotation in annotations_window : 
            if any(event in annotation for event in events ) : 
                state = True 
                break 
        return state 
    def build(self , annotations) :  
        video_df = None 
        windows = self._create_windows(annotations)

        for window in windows : 
            start_index = next(
            (i for i, d in enumerate(annotations) if int(d["frame_id"])  == window[0]))
            end_index = next(
            (i for i, d in enumerate(annotations) if int(d["frame_id"])  == window[1]))
            state = self._check_window_events(annotations[start_index : end_index]) 

            if state :
                df = self._turn_window_into_dataframe(annotations[start_index : end_index]) 
                df = self._data_engineering(df)

                if video_df is None : 
                    video_df = df 
                else : 
                    video_df = pd.concat([video_df , df])
        return video_df
