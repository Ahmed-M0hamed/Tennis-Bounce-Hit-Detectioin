from moviepy import VideoFileClip
import os 
from .utils import read_annoations , write_labels 
def extract_audio_by_frames(video_path, annotations_path,events , margin):
    annotations = read_annoations(annotations_path) 
    video_name = video_path.split('/')[-1].split('.')[0] 
    os.makedirs(os.path.join(os.getcwd() , 'dataset') , exist_ok = True)
    os.makedirs(os.path.join(os.getcwd() , 'dataset' , f'{video_name}') , exist_ok = True)
    os.makedirs(os.path.join(os.getcwd() , 'dataset' , f'{video_name}' , 'audios') , exist_ok = True)
    labels = []
    for annotation in annotations : 
        if any(event in annotation for event in events) : 
            event = list(set(events).intersection(annotation.keys()))[0] 
            event = 'ball_hit' if event == 'serve' else event 
            frame_id = annotation['frame_id']
            labels.append({'frame' : frame_id , 'label' :event}  )

    with VideoFileClip(video_path) as video:
        # Get the frame rate (FPS) of the video
        fps = video.fps
        total_frames = int(video.duration * video.fps)
        for label in labels : 
            frame_id = label['frame']
            start_frame = (frame_id - margin ) if (frame_id - margin ) > 0 else 0 
            end_frame = (frame_id + margin ) if (frame_id + margin ) < total_frames else total_frames
            start_time = start_frame / fps
            end_time = end_frame / fps            
            # Subclip the video based on calculated timestamps
            sub_clip = video.subclipped(start_time, end_time)
            output_file_name = f'frame_{frame_id}_{label["label"]}'
            # Extract and save the audio component
            sub_clip.audio.write_audiofile(os.path.join(os.getcwd() , 'dataset' ,
                                            f'{video_name}' , 'audios' ,f'{output_file_name}.wav'))
    write_labels(labels ,os.path.join(os.getcwd() , 'dataset' ,
                                            f'{video_name}', 'labels.json' ))



