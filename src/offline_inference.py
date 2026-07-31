import pandas as pd 
import numpy as np 
import torch 
from .homography import compute_homography , transform_ball_players_court_position 
from .utils import get_bottom_center_of_player , distance_between_two_points 
from sklearn.preprocessing import StandardScaler 
from scipy.signal import savgol_filter 
from typing import List 
from .model import TennisEventCNN 
from transformers import pipeline  
from moviepy import VideoFileClip
label2id = {
    "no_event": 0,
    "ball_hit": 1,
    "ball_bounced": 2,
}
id2label = {
    0: "no_event",
    1: "ball_hit",
    2: "ball_bounced",
}

FEATURE_COLUMNS = [

    # Ball trajectory
    "y_ball_smooth",
    "x_ball_smooth",
    "vx_ball",
    "vy_ball",
    "ax_ball",
    "ay_ball",
    "ball_speed",
    "ball_acc" ,
    "dir_sin",
    "dir_cos",
    "velocity_mean" ,
    "velocity_std" ,
    "acc_mean" ,
    "acc_std" ,
    # Player interaction
    "nearest_player_to_ball",
    "nearest_player_rate",

    #court
    "top_left_x" ,
    "top_left_y" ,
    "bottom_left_x" ,
    "bottom_left_y" ,
    "bottom_right_x" ,
    "bottom_right_y" ,
    "top_right_x" ,
    "top_right_y" ,

]

class OfflineInference: 
    def __init__(self ,event_detector , video_path , id2label, audio_classifier : str = "ahmedmohamed55/ast-tennis" ,window_size :int = 15, stride:int =1  , min_gap_frames : int = 5 , FEATURES_COLUMNS:List = None):
        self.window_size = window_size 
        self.stride = stride 
        self.min_gap_frames = min_gap_frames
        self.FEATURES_COLUMNS = FEATURES_COLUMNS 
        self.event_detector = event_detector 
        self.id2label = id2label
        self.video_path = video_path 
        self.audio_classifier = pipeline("audio-classification", model=audio_classifier)
        

    def _get_window(self , annotations : List , last_window_center_index:int = None ) : 
        radius = int(self.window_size / 2) 
        if last_window_center_index is None or last_window_center_index < radius : 
            new_center_index =  radius
        else : 
            new_center_index = last_window_center_index + self.stride 

        window = annotations[new_center_index - radius : (new_center_index + radius)+1] 

        return window , new_center_index 
    def _check_window(self , window ) : 
        gap = 1 
        index = 0 
        while index < len(window): 
            if 'ball_position' not in window[index] : 
                gap +=1 
            else : 
                gap = 1 
            index += 1 
        if gap > self.min_gap_frames : 
            return False 
        else : 
            return True 
    def _turn_window_into_dataframe(self  , annotations_window ) : 
        dataframe_rows = []
        
        for annotation in annotations_window : 
            player_1_position = get_bottom_center_of_player(annotation['persons'][0]['xyxy']) if annotation['persons'] else [None , None]
            player_2_position = get_bottom_center_of_player(annotation['persons'][1]['xyxy']) if len(annotation['persons']) == 2 else [None , None]
            if 'ball_position' in annotation and len(annotation['persons']) == 2 : 
                player_1_position = get_bottom_center_of_player(annotation['persons'][0]['xyxy'])
                player_2_position = get_bottom_center_of_player(annotation['persons'][1]['xyxy'])
                transformed_ball , transformed_player_1, transformed_player_2 , transformed_corners = transform_ball_players_court_position(annotation['court_points'] ,
                                    annotation['ball_position'], player_1_position , player_2_position)
                flaten_corners = [item for corner in transformed_corners for item in corner]
                flaten_row = [*transformed_ball , *transformed_player_1 , *transformed_player_2 , *flaten_corners]
                flaten_row.append(annotation['frame_id'])


                dataframe_rows.append(flaten_row) 
            else : 
                transformed_ball , transformed_player_1, transformed_player_2 , transformed_corners = transform_ball_players_court_position(annotation['court_points'] ,
                                    [None , None], player_1_position , player_2_position)
                flaten_corners = [item for corner in transformed_corners for item in corner]
                flaten_row = [*[None , None] , *transformed_player_1 , *transformed_player_2 , *flaten_corners]
                flaten_row.append(annotation['frame_id'])
                dataframe_rows.append(flaten_row) 

       

        df_ball_positions = pd.DataFrame(dataframe_rows,columns=['ball_x' , 'ball_y' ,'player_1_x' ,
                        'player_1_y' , 'player_2_x' , 'player_2_y' , 'top_left_x' , 'top_left_y' 
                        , 'top_right_x' , 'top_right_y' , 'bottom_left_x' , 'bottom_left_y' 
                        , 'bottom_right_x' , 'bottom_right_y' ,'frame' ]) 
        return df_ball_positions 
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
    
            # Savitzky-Golay needs at least window+1 non-NaN points
            valid = df["ball_y"].notna()
            if valid.sum() > window:
                df.loc[valid, "y_ball_smooth"] = savgol_filter(
                    df.loc[valid, "ball_y"], window_length=window, polyorder=poly
                )
                df.loc[valid, "x_ball_smooth"] = savgol_filter(
                    df.loc[valid, "ball_x"], window_length=window, polyorder=poly
                )
            else:
                df["y_smooth"] = df["ball_y"]
                df["x_smooth"] = df["ball_x"]
    
            return df
    def _data_engineering(self, df ) : 
            df_ball_positions = df
            df_ball_positions[['ball_x' , 'ball_y']]= df_ball_positions[['ball_x' , 'ball_y']].interpolate()
            df_ball_positions[['ball_x' , 'ball_y']] = df_ball_positions[['ball_x' , 'ball_y']].bfill()
            smothed_df = self._smooth_trajectory(df_ball_positions ) 
            smothed_df['ball_y_rolling_mean'] = smothed_df['ball_y'].rolling(window=5, min_periods=1, center=False).mean()
            smothed_df['ball_x_rolling_mean'] = smothed_df['ball_x'].rolling(window=5, min_periods=1, center=False).mean()
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
    def _prepare_dataframe_for_inference(self , window_dataframe) : 
        # Columns identified to potentially have NaN values that need filling
        columns_to_fill = ['velocity_mean', 'velocity_std', 'acc_mean', 'acc_std']
        for col in columns_to_fill:
            if col in window_dataframe.columns:
                window_dataframe[col] = window_dataframe[col].fillna(window_dataframe[col].mean())

        # Scaling 
        scaler = StandardScaler()
        window_dataframe[self.FEATURES_COLUMNS] = scaler.fit_transform(
            window_dataframe[self.FEATURES_COLUMNS]
        )

        
        # convert to tensor 
        input_tensor = window_dataframe[FEATURE_COLUMNS].values.astype(np.float32) 
        input_tensor = torch.from_numpy(input_tensor).unsqueeze(0)
        return input_tensor
    def _extract_window_audio_file(self , window_center  ) :
        with VideoFileClip(self.video_path) as video:
            # Get the frame rate (FPS) of the video
            fps = video.fps
            total_frames = int(video.duration * video.fps)
            frame_id = window_center
            start_frame = (frame_id - int(self.window_size/2)  ) if (frame_id - int(self.window_size/2) ) > 0 else 0 
            end_frame = (frame_id + int(self.window_size/2) ) if (frame_id + int(self.window_size/2) ) < total_frames else total_frames
            start_time = start_frame / fps
            end_time = end_frame / fps            
                    # Subclip the video based on calculated timestamps
            sub_clip = video.subclipped(start_time, end_time)
            audio = sub_clip.audio
            audio_array = audio.to_soundarray(fps=16000)

            # Stereo -> Mono
            if audio_array.ndim == 2:
                audio_array = audio_array.mean(axis=1)

            audio_array = audio_array.astype(np.float32)
            return audio_array 
            

    def _event_detection_prediction(self , smooth_df) : 
        input_tensor = self._prepare_dataframe_for_inference(smooth_df)
        self.event_detector.train()
        with torch.no_grad() : 
            logits = self.event_detector(input_tensor)
        preds = torch.softmax(logits , 1 ) 
        return preds 
    def _audio_prediction(self,new_center ) : 
        audio_tensor = self._extract_window_audio_file(new_center)
        audio_logits = self.audio_classifier(audio_tensor) 
        return audio_logits 
    def _prediction_fusion(self , numric_logits , audio_logits) : 
        order = ['no_event' ,'ball_hit' , 'ball_bounced' ] 
        ordered_audio_logits = [] 
        row_result = []
        for o in order : 
            for pred in audio_logits : 
                if pred['label'] == o : 
                    ordered_audio_logits.append(pred['score'])

        wa = max(ordered_audio_logits)
        wt = max(numric_logits)

        wa /= wa + wt
        wt /= wa + wt
        final = wa * np.array(ordered_audio_logits) + wt * np.array(numric_logits)
        row_result.append(order[np.argmax(final)])
        row_result.extend(final) 
        row_result.append(order[np.argmax(numric_logits)]) 
        row_result.extend(numric_logits) 
        row_result.append(order[np.argmax(ordered_audio_logits)]) 
        row_result.extend(ordered_audio_logits)
        return row_result
    def infer(self ,annotations)  :
        window_center = 0 
        results = []
        while window_center + int(self.window_size / 2 ) < int(annotations[-1]['frame_id']): 

            window , new_center = self._get_window(annotations=annotations , last_window_center_index= window_center) 
            state = self._check_window(window)
            print(state)
            if state :
                try : 
                    df = self._turn_window_into_dataframe(window) 
                    smooth = self._data_engineering(df) 
                    numerical_logits = self._event_detection_prediction(smooth) 
                    audio_logits = self._audio_prediction(new_center) 
                    result = self._prediction_fusion(numerical_logits[0].tolist() ,audio_logits )
                    result.append(new_center) 
                    results.append(result) 
                except :
                    print('unvalid_window')


            window_center = new_center
        df_columns = ['combined_prediction' , 'combined_no_event_conf' ,
                      'combined_ball_hit_conf' , 'combined_ball_bounced_conf' ,'event_classifier_prediction' , 'event_classifier_no_event_conf' ,
                      'event_classifier_ball_hit_conf' , 'event_classifier_ball_bounced_conf' ,'audio_classifier_prediction' , 'audio_classifier_no_event_conf' ,
                      'audio_classifier_ball_hit_conf' , 'audio_classifier_ball_bounced_conf' , 'frame']
        result_df = pd.DataFrame(results , columns= df_columns)
        return result_df

        

        
