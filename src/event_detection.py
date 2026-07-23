from typing import List , Dist 
from scipy.signal import savgol_filter , find_peaks
import numpy as np 
import pandas as pd 
from .homography import transform_ball_position 

class EventDetection: 
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

        # Interpolate short gaps (≤5 frames) – handles momentary occlusion
        df["x"] = df["x"].interpolate(method="linear", limit=5)
        df["y"] = df["y"].interpolate(method="linear", limit=5)

        # Savitzky-Golay needs at least window+1 non-NaN points
        valid = df["y"].notna()
        if valid.sum() > window:
            df.loc[valid, "y_smooth"] = savgol_filter(
                df.loc[valid, "y"], window_length=window, polyorder=poly
            )
            df.loc[valid, "x_smooth"] = savgol_filter(
                df.loc[valid, "x"], window_length=window, polyorder=poly
            )
        else:
            df["y_smooth"] = df["y"]
            df["x_smooth"] = df["x"]

        return df
    def _turn_window_into_dataframe(self  , annotations_window ) : 
        ball_positions = []
        for annotation in annotations_window : 
            if 'ball_position' in annotation : 
                transformed_ball_position = transform_ball_position(annotation['court_points'] ,
                                                                     annotation['ball_position'])
                transformed_x , transformed_y = transformed_ball_position
                ball_positions.append([transformed_x , transformed_y , annotation['frame_id']]) 
            else : 
                ball_positions.append([None , None , annotation['frame_id']])
        # ball_positions = [
        # annotation["ball_position"] + [annotation["frame_id"]]
        # if "ball_position" in annotation
        # else [None, None, annotation["frame_id"]]
        # for annotation in annotations_window
        # ]

        df_ball_positions = pd.DataFrame(ball_positions,columns=['x' , 'y' , 'frame']) 
        return df_ball_positions 
    def _df_operations(self, df ) : 
        df_ball_positions = df.interpolate()
        df_ball_positions = df_ball_positions.bfill()
        smothed_df = self._smooth_trajectory(df_ball_positions ) 
        smothed_df['mid_y_rolling_mean'] = smothed_df['y_smooth'].rolling(window=5, min_periods=1, center=False).mean()
        smothed_df['delta_y'] = smothed_df['mid_y_rolling_mean'].diff() 
        return smothed_df
    def _detect_bounces(self, 
            df,min_drop_px = 10.0, min_gap_frames= 15 , court_y_min = 1,   # ignore bounces above this y (pixels)
        ) :
        # df = df.set_index("frame" , drop=False)
        cy = df["delta_y"].values
        # find_peaks on cy detects local maxima (lowest on screen = bounce)
        peaks, p_props = find_peaks(
            cy,
            height=court_y_min,           # optional lower bound on y
            prominence=min_drop_px,       # must stick out by N px
            distance=min_gap_frames,      # minimum spacing
        )
        y_min = -1 * court_y_min if court_y_min is not None else None 
        bottoms , b_props= find_peaks(
            -1 * cy,
            height= y_min,           # optional lower bound on y
            prominence=min_drop_px,       # must stick out by N px
            distance=min_gap_frames,      # minimum spacing
        )

        peaks_frames = df["frame"].iloc[peaks].values
        bottoms_frames = df["frame"].iloc[bottoms].values
        bounce_frames = np.concatenate((peaks_frames, bottoms_frames))

        return bounce_frames.tolist()

    def _detect_racket_hits(self , df) : 
        df = df.set_index('frame' , drop = False)
        df['ball_hit']= 0
        minimum_change_frames_for_hit = 20
        for i in range(1,len(df)- int(minimum_change_frames_for_hit*1.2) ):
            negative_position_change = df['delta_y'].iloc[i] >0 and df['delta_y'].iloc[i+1] <0
            positive_position_change = df['delta_y'].iloc[i] <0 and df['delta_y'].iloc[i+1] >0

            if negative_position_change or positive_position_change:
                change_count = 0 
                for change_frame in range(i+1, i+int(minimum_change_frames_for_hit*1.2)+1):
                    negative_position_change_following_frame = df['delta_y'].iloc[i] >0 and df['delta_y'].iloc[change_frame] <0
                    positive_position_change_following_frame = df['delta_y'].iloc[i] <0 and df['delta_y'].iloc[change_frame] >0

                    if negative_position_change and negative_position_change_following_frame:
                        change_count+=1
                    elif positive_position_change and positive_position_change_following_frame:
                        change_count+=1
            
                if change_count>minimum_change_frames_for_hit-1:
                    df.loc[df.index[i], 'ball_hit'] = 1

        frame_nums_with_ball_hits = df[df['ball_hit']==1].index.tolist()
    
        return frame_nums_with_ball_hits

    def test(self , annotations) :  
        windows = self._create_windows(annotations)
        events = []
        racket_hits = []
        for window in windows : 
            start_index = next(
            (i for i, d in enumerate(annotations) if int(d["frame_id"])  == window[0]))
            end_index = next(
            (i for i, d in enumerate(annotations) if int(d["frame_id"])  == window[1]))
            df = self._turn_window_into_dataframe(annotations[start_index : end_index]) 
            smoothed_df = self._df_operations(df) 
            detected_events = self._detect_bounces(smoothed_df,min_drop_px=5,
                                min_gap_frames=20,
                                court_y_min=1)
            events.extend(detected_events) 
            hits = self._detect_racket_hits(smoothed_df)
            racket_hits.extend(hits)
        return events  , racket_hits
        
