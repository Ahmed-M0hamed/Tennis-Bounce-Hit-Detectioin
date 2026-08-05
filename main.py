from src import extract_croped_events_videos, DataFrameBuilding, extract_events_videos,filtering,load_urls ,write_labels , extract_audio_by_frames , extract_events_videos , TennisEventCNN , OfflineInference , read_annoations
import os 
import torch
import json 
import cv2
import pandas as pd 

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
    "ball_y_smoothed",
    "ball_x_smoothed",
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
    
    with open('data/novak_vs_thiem.jsonl' , 'rb') as f : 
        data = [json.loads(line) for line in f if line.strip()]
    # annotations = read_annoations(os.path.join(os.getcwd() , 'data' ,f'Layal_vs_Fery.json')) 
    event_detector_model = TennisEventCNN(input_features=len(FEATURE_COLUMNS) ) 
    checkpoint = torch.load(
    "best_model.pt",
    weights_only= False 
    )

    event_detector_model.load_state_dict(
    checkpoint["model"]
    )
    inferer = OfflineInference(event_detector=event_detector_model , video_path=os.path.join(os.getcwd() , 'data' ,f'novak_vs_thiem.mp4' ) , id2label=id2label, label2id= label2id , stride=5, FEATURES_COLUMNS= FEATURE_COLUMNS)
    resutls_df = inferer.infer(data) 
    print(resutls_df)
    # resutls_df.to_csv('pipeline_results/novak_vs_thiem.csv')
    # filtered_df = filtering(resutls_df , .98 , .85 , .6 , 7 ) 
    # filtered_df.to_csv('Layal_vs_Martin_filtered_results.csv' )


    # BUILD DATAFRAMES 
    # with open('data/paolini_vs_pegula.jsonl' , 'rb') as f : 
    #     data = [json.loads(line) for line in f if line.strip()]
    # data = read_annoations(os.path.join(os.getcwd() , 'data' ,f'novak_vs_thiem.json'))
    # builder = DataFrameBuilding(5 , 30)
    # df = builder.build(data)
    # df.to_csv('event_detections_dataset/Layal_vs_Hsu.csv')


    # VIDEO EXTRACTION 
    # events = ['ball_bounced', 'ball_hit' , 'serve'] 
    # labels = extract_croped_events_videos(os.path.join(os.getcwd() , 'data' ,f'paolini_vs_pegula.mp4') , data, events , 8 )
    # cv2.destroyAllWindows()
if __name__ == "__main__":
    main()
