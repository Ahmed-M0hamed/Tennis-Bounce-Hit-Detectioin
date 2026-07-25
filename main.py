from src import load_urls ,write_labels , extract_audio_by_frames , extract_events_videos , TennisEventCNN , OfflineInference , read_annoations
import os 
import torch


def main():

    urls = {'Layal_vs_Fery' : {'video_url' : 'https://assets.dreamfight.io/tennis/cranbrook_special/atp_challenger_bloomfield_hills_usa_men_singles_lajal_vs_fery.mp4' , 
                           'annotations' : 'https://assets.dreamfight.io/tennis/cranbrook_special/atp_challenger_bloomfield_hills_usa_men_singles_lajal_vs_fery.json'} , 
                           
        'Layal_vs_Hsu' :  {'video_url' : 'https://assets.dreamfight.io/tennis/cranbrook_special/atp_challenger_bloomfield_hills_usa_men_singles_lajal_vs_hsu.mp4' , 
                           'annotations' : 'https://assets.dreamfight.io/tennis/cranbrook_special/atp_challenger_bloomfield_hills_usa_men_singles_lajal_vs_hsu.json'} , 
        
        'Layal_vs_Martin' : {'video_url' : 'https://assets.dreamfight.io/tennis/cranbrook_special/atp_challenger_bloomfield_hills_usa_men_singles_lajal_vs_martin.mp4' , 
                           'annotations' : 'https://assets.dreamfight.io/tennis/cranbrook_special/atp_challenger_bloomfield_hills_usa_men_singles_lajal_vs_martin.json'} } 

    # load_urls(urls) 
    # events = ['ball_bounced', 'ball_hit' , 'serve']
    # for video_name in urls.keys() : 
    #     video_path = os.path.join(os.getcwd() , 'data' ,f'{video_name}.mp4')
    #     annotations_path = os.path.join(os.getcwd() , 'data' ,f'{video_name}.json') 
    #     extract_audio_by_frames(video_path , annotations_path , events , 6) 
        # extract_events_videos(video_path , annotations_path , events , 3) 

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

    annotations = read_annoations(os.path.join(os.getcwd() , 'data' ,f'Layal_vs_Fery.json')) 
    event_detector_model = TennisEventCNN(input_features=len(FEATURE_COLUMNS) ) 
    checkpoint = torch.load(
    "best.pt",
    weights_only= False 
    )

    event_detector_model.load_state_dict(
    checkpoint["model"]
    )
    inferer = OfflineInference(event_detector=event_detector_model , video_path=os.path.join(os.getcwd() , 'data' ,f'Layal_vs_Fery.mp4' ) , id2label=id2label , stride=5, FEATURES_COLUMNS= FEATURE_COLUMNS)
    true_preds , preds = inferer.infer(annotations) 

    print(true_preds)
    print(preds)
    write_labels(true_preds , 'Layal_vs_Fery_results.json')
if __name__ == "__main__":
    main()
