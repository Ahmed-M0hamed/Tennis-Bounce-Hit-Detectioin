from src import load_urls , extract_audio_by_frames , extract_events_videos
import os 

def main():

    urls = {'Layal_vs_Fery' : {'video_url' : 'https://assets.dreamfight.io/tennis/cranbrook_special/atp_challenger_bloomfield_hills_usa_men_singles_lajal_vs_fery.mp4' , 
                           'annotations' : 'https://assets.dreamfight.io/tennis/cranbrook_special/atp_challenger_bloomfield_hills_usa_men_singles_lajal_vs_fery.json'} , 
                           
        'Layal_vs_Hsu' :  {'video_url' : 'https://assets.dreamfight.io/tennis/cranbrook_special/atp_challenger_bloomfield_hills_usa_men_singles_lajal_vs_hsu.mp4' , 
                           'annotations' : 'https://assets.dreamfight.io/tennis/cranbrook_special/atp_challenger_bloomfield_hills_usa_men_singles_lajal_vs_hsu.json'} , 
        
        'Layal_vs_Martin' : {'video_url' : 'https://assets.dreamfight.io/tennis/cranbrook_special/atp_challenger_bloomfield_hills_usa_men_singles_lajal_vs_martin.mp4' , 
                           'annotations' : 'https://assets.dreamfight.io/tennis/cranbrook_special/atp_challenger_bloomfield_hills_usa_men_singles_lajal_vs_martin.json'} } 

    # load_urls(urls) 
    events = ['ball_bounced', 'ball_hit' , 'serve']
    for video_name in urls.keys() : 
        video_path = os.path.join(os.getcwd() , 'data' ,f'{video_name}.mp4')
        annotations_path = os.path.join(os.getcwd() , 'data' ,f'{video_name}.json') 
        extract_audio_by_frames(video_path , annotations_path , events , 6) 
        # extract_events_videos(video_path , annotations_path , events , 3) 

if __name__ == "__main__":
    main()
