import numpy as np
import pandas as pd


def temporal_event_clustering(df, gap=10):
    """
    Merge consecutive detections of the same event.

    Returns one event per cluster located at the confidence-weighted center.
    """

    confidence_map = {
        "ball_hit": "combined_ball_hit_conf",
        "ball_bounced": "combined_ball_bounced_conf",
    }

    df = df.copy()

    events = []

    for event_name in confidence_map.keys():

        conf_col = confidence_map[event_name]

        event_df = (
            df[df["combined_prediction"] == event_name]
            .sort_values("frame")
            .reset_index(drop=True)
        )

        if len(event_df) == 0:
            continue

        cluster = [event_df.iloc[0]]

        for i in range(1, len(event_df)):

            row = event_df.iloc[i]

            if row["frame"] - cluster[-1]["frame"] <= gap:

                cluster.append(row)

            else:

                cluster_df = pd.DataFrame(cluster)

                weighted_frame = int(round(np.average(
                    cluster_df["frame"],
                    weights=cluster_df[conf_col]
                )))

                best_idx = cluster_df[conf_col].idxmax()

                best_row = cluster_df.loc[best_idx].copy()

                # Replace frame by weighted center
                best_row["frame"] = weighted_frame

                events.append(best_row)

                cluster = [row]

        # Final cluster
        cluster_df = pd.DataFrame(cluster)

        weighted_frame = int(round(np.average(
            cluster_df["frame"],
            weights=cluster_df[conf_col]
        )))

        best_idx = cluster_df[conf_col].idxmax()

        best_row = cluster_df.loc[best_idx].copy()

        best_row["frame"] = weighted_frame

        events.append(best_row)

    return (
        pd.DataFrame(events)
        .sort_values("frame")
        .reset_index(drop=True)
    )

def filtering(df , audio_classifier_ball_hit_threshold , ball_hit_threshold , ball_bounce_threshold , tmp_gap) : 
    events_df = df[df['combined_prediction'] != 'no_event'] 
    bounces_df = events_df[events_df['combined_prediction'] == 'ball_bounced'] 
    hit_df = events_df[events_df['combined_prediction'] == 'ball_hit']

    bounces_df['combined_prediction_score'] = bounces_df[['combined_no_event_conf' , 'combined_ball_hit_conf','combined_ball_bounced_conf']].max( axis=1 )

    # filter ball hits 
    hit_df['combined_prediction_score'] = hit_df[['combined_no_event_conf' , 'combined_ball_hit_conf','combined_ball_bounced_conf']].max( axis=1 )
    hit_df = hit_df[hit_df['audio_classifier_ball_hit_conf'] > audio_classifier_ball_hit_threshold]
    hit_df = hit_df[hit_df['combined_prediction_score'] > ball_hit_threshold] 
    final_hit = temporal_event_clustering(hit_df , tmp_gap) 

    # filter ball bounces 
    bounces_df['combined_prediction_score'] = bounces_df[['combined_no_event_conf' , 'combined_ball_hit_conf','combined_ball_bounced_conf']].max( axis=1 )
    bounces_df = bounces_df[bounces_df['combined_prediction_score'] > ball_bounce_threshold]
    final_bounce = temporal_event_clustering(bounces_df , tmp_gap)

    fitered_events_df = pd.concat([final_hit , final_bounce]) 

    return fitered_events_df