# this script download the videos and annotations 
import requests 
import os 
import json




def download_mp4(url : str, output_filename:str)-> None: 
    # Send a HTTP request to the URL with streaming enabled
    with requests.get(url, stream=True) as response:
        # Check if the request was successful
        response.raise_for_status()
        
        # Open the local file in Write-Binary (wb) mode
        with open(output_filename, 'wb') as file:
            # Iterate over the response data in 1MB chunks
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                # Filter out keep-alive new chunks
                if chunk:
                    file.write(chunk)
    
    print(f"Download complete: {output_filename}")

def download_json(url :str , output_filename:str) -> None: 
    response = requests.get(url)
    response.raise_for_status()
    data = []

    for line in response.iter_lines():
        if line:
            data.append(json.loads(line.decode('utf-8')))
    with open(output_filename, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)


def load_urls(urls : dict) -> None : 
    os.makedirs(os.path.join(os.getcwd() , 'data') , exist_ok=True) 

    for name , video_annotations_urls in urls.items() : 
        video_url = video_annotations_urls['video_url']
        annotations_url = video_annotations_urls['annotations']
        video_output_path = os.path.join(os.getcwd() , 'data' , f'{name}.mp4') 
        annotations_output_path = os.path.join(os.getcwd() , 'data' , f'{name}.json') 

        try : 
            # download_mp4(video_url , video_output_path) 
            download_json(annotations_url , annotations_output_path) 
        except : 
            print('could not fetch the data') 

        

