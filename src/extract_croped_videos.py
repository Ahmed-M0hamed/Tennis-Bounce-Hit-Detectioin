import os 
from typing import List 
from .utils import write_video  , write_labels , read_frames , read_annoations , crop_frame
import random 

def extract_croped_events_videos(video_path : str , annotations: List , events :List  , margin : int , target_crop_size:tuple= (224,224), noise_windows : int = 30 ) : 
    frames , fps= read_frames(video_path ) 

    video_name = video_path.split('/')[-1].split('.')[0] 
    os.makedirs(os.path.join(os.getcwd() , 'croped_videos_dataset') , exist_ok = True)
    os.makedirs(os.path.join(os.getcwd() , 'croped_videos_dataset' , f'{video_name}') , exist_ok = True)
    annotation_map = {
                ann["frame_id"]: ann["ball_position"] if 'ball_position' in ann else None 
                for ann in annotations
            }
    labels = []
    noise_count = 0
    i = 0
    last_event_frame = 0 
    
    while i < len(annotations) : 
        annotation = annotations[i]
        last_valid_ball_pos = None 
        if any(event in annotation for event in events) : 
            event = list(set(events).intersection(annotation.keys()))[0] 
            event = 'ball_hit' if event == 'serve' else event 
            frame_id = annotation['frame_id']
            event_file_name = f"frame_{frame_id}_{event}"
            start_frame = (frame_id - margin ) if (frame_id - margin ) > 0 else 0 
            end_frame = (frame_id + margin ) if (frame_id + margin ) < len(frames) else len(frames) 
            frames_window = frames[start_frame : end_frame] 
            croped_frames_window = [] 
            for id , frame  in enumerate(frames_window) : 
                window_frame_id = start_frame + id 
                if window_frame_id in annotation_map and annotation_map[window_frame_id] is not None : 
                    crop_center = annotation_map[window_frame_id]
                    last_valid_ball_pos = annotation_map[window_frame_id]
                elif last_valid_ball_pos is not None : 
                    crop_center = last_valid_ball_pos
                else : 
                    h , w = frame.shape[:2]
                    crop_center = [h /2 , w/2] 
                croped_frame = crop_frame(frame , crop_center , target_crop_size) 

                croped_frames_window.append(croped_frame)

            write_video(os.path.join(os.getcwd() , 'croped_videos_dataset', f'{video_name}' ,f'{event_file_name}.mp4' ) , croped_frames_window , fps) 
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
                    croped_frames_window = []
                    for id , frame  in enumerate(frames_window) : 
                        window_frame_id = start_frame + id 
                        if window_frame_id in annotation_map and annotation_map[window_frame_id] is not None : 
                            crop_center = annotation_map[window_frame_id]
                            last_valid_ball_pos = annotation_map[window_frame_id]
                        elif last_valid_ball_pos is not None : 
                            crop_center = last_valid_ball_pos
                        else : 
                            h , w = frame.shape[:2]
                            random_center_x = random.randint(0 , w)
                            random_center_y = random.randint(0 , h)

                        croped_frame = crop_frame(frame , (random_center_x , random_center_y) , target_crop_size) 

                        croped_frames_window.append(croped_frame)

                    write_video(os.path.join(os.getcwd() , 'croped_videos_dataset', f'{video_name}' ,f'{event_file_name}.mp4' ) , croped_frames_window , fps) 
                    labels.append({'frame' : frame_id , 'label' :event}  ) 
                    noise_count += 1
                    i += margin
                

        i += 1
    print(f"Total number of noise windows : {noise_count} in video {video_name} and total number of events : {len(labels)}")
    return labels
    