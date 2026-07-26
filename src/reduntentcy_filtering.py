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