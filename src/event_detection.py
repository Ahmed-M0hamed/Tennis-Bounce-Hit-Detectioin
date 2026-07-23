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
        ball_positions = [
        annotation["ball_position"] + [annotation["frame_id"]]
        if "ball_position" in annotation
        else [None, None, annotation["frame_id"]]
        for annotation in annotations_window
        ]

        df_ball_positions = pd.DataFrame(ball_positions,columns=['x' , 'y' , 'frame']) 
        return df_ball_positions 
    def _df_operations(self, df ) : 
        df_ball_positions = df.interpolate()
        df_ball_positions = df_ball_positions.bfill()
        smothed_df = self._smooth_trajectory(df_ball_positions ) 
        smothed_df['mid_y_rolling_mean'] = smothed_df['y_smooth'].rolling(window=5, min_periods=1, center=False).mean()
        smothed_df['delta_y'] = smothed_df['mid_y_rolling_mean'].diff() 
        return smothed_df

    def test(self , annotations) :  
        windows = self._create_windows(annotations)
        for window in windows[:1] : 
            start_index = next(
            (i for i, d in enumerate(annotations) if int(d["frame_id"])  == window[0]))
            end_index = next(
            (i for i, d in enumerate(annotations) if int(d["frame_id"])  == window[1]))
            df = self._turn_window_into_dataframe(annotations[start_index : end_index]) 
            smoothed_df = self._df_operations(df) 
        return smoothed_df , df 
        
