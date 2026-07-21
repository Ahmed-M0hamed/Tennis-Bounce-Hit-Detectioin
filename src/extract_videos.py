import os 
from typing import List 
from .utils import write_video  , write_labels , read_frames , read_annoations

def extract_events_videos(video_path : str , annotation_path : List , events :List , margin : int  ) : 
    frames , fps= read_frames(video_path ) 
    annotations = read_annoations(annotation_path)
    video_name = video_path.split('/')[-1].split('.')[0] 
    os.makedirs(os.path.join(os.getcwd() , 'dataset') , exist_ok = True)
    os.makedirs(os.path.join(os.getcwd() , 'dataset' , f'{video_name}') , exist_ok = True)
    os.makedirs(os.path.join(os.getcwd() , 'dataset' , f'{video_name}' , 'videos') , exist_ok = True)
    labels = []
    for annotation in annotations : 
        if any(event in annotation for event in events) : 
            event = list(set(events).intersection(annotation.keys()))[0] 
            event = 'ball_hit' if event == 'serve' else event 
            frame_id = annotation['frame_id']
            event_file_name = f"frame_{frame_id}_{event}"
            start_frame = (frame_id - margin ) if (frame_id - margin ) > 0 else 0 
            end_frame = (frame_id + margin ) if (frame_id + margin ) < len(frames) else len(frames) 
            frames_window = frames[start_frame : end_frame] 
            write_video(os.path.join(os.getcwd() , 'dataset', f'{video_name}' ,'videos' ,f'{event_file_name}.mp4' ) , frames_window , fps) 
            labels.append({'frame' : frame_id , 'label' :event}  ) 
    write_labels(labels , os.path.join(os.getcwd() , 'dataset' , f'{video_name}' , 'labels.json')) 
    