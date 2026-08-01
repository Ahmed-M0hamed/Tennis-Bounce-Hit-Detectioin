import os 
from typing import List 
from .utils import write_video  , write_labels , read_frames , read_annoations

def extract_events_videos(video_path : str , annotations: List , events :List , margin : int , noise_windows : int = 30 ) : 
    frames , fps= read_frames(video_path ) 

    video_name = video_path.split('/')[-1].split('.')[0] 
    os.makedirs(os.path.join(os.getcwd() , 'videos_dataset') , exist_ok = True)
    os.makedirs(os.path.join(os.getcwd() , 'videos_dataset' , f'{video_name}') , exist_ok = True)

    labels = []
    noise_count = 0
    i = 0
    last_event_frame = 0 
    while i < len(annotations) : 
        annotation = annotations[i]
        if any(event in annotation for event in events) : 
            event = list(set(events).intersection(annotation.keys()))[0] 
            event = 'ball_hit' if event == 'serve' else event 
            frame_id = annotation['frame_id']
            event_file_name = f"frame_{frame_id}_{event}"
            start_frame = (frame_id - margin ) if (frame_id - margin ) > 0 else 0 
            end_frame = (frame_id + margin ) if (frame_id + margin ) < len(frames) else len(frames) 
            frames_window = frames[start_frame : end_frame] 
            write_video(os.path.join(os.getcwd() , 'videos_dataset', f'{video_name}' ,f'{event_file_name}.mp4' ) , frames_window , fps) 
            labels.append({'frame' : frame_id , 'label' :event}  ) 
            last_event_frame = frame_id
            i += margin
        elif (annotation['frame_id'] - int(2*margin+1)) > int(last_event_frame ) : 
                j = 1 
                state = False 
                while j < margin : 
                    if (i+j) < len(annotations) : 
                        if any(event in annotations[i+j] for event in events) : 

                            state = True 
                            i += j - 1  
                            break 
                    j += 1
                if not state :
                    event = 'no_event'
                    frame_id = annotation['frame_id'] 
                    event_file_name = f"frame_{frame_id}_{event}"
                    start_frame = (frame_id - margin ) if (frame_id - margin ) > 0 else 0 
                    end_frame = (frame_id + margin ) if (frame_id + margin ) < len(frames) else len(frames) 
                    frames_window = frames[start_frame : end_frame] 
                    write_video(os.path.join(os.getcwd() , 'videos_dataset', f'{video_name}' ,f'{event_file_name}.mp4' ) , frames_window , fps) 
                    labels.append({'frame' : frame_id , 'label' :event}  ) 
                    noise_count += 1
                    i += margin
                

        i += 1
    print(f"Total number of noise windows : {noise_count} in video {video_name} and total number of events : {len(labels)}")
    return labels
    