from src import load_urls
def main():

    urls = {'Layal_vs_Fery' : {'video_url' : 'https://assets.dreamfight.io/tennis/cranbrook_special/atp_challenger_bloomfield_hills_usa_men_singles_lajal_vs_fery.mp4' , 
                           'annotations' : 'https://assets.dreamfight.io/tennis/cranbrook_special/atp_challenger_bloomfield_hills_usa_men_singles_lajal_vs_fery.json'} , 
                           
        'Layal_vs_Hsu' :  {'video_url' : 'https://assets.dreamfight.io/tennis/cranbrook_special/atp_challenger_bloomfield_hills_usa_men_singles_lajal_vs_hsu.mp4' , 
                           'annotations' : 'https://assets.dreamfight.io/tennis/cranbrook_special/atp_challenger_bloomfield_hills_usa_men_singles_lajal_vs_hsu.json'} , 
        
        'Layal_vs_Martin' : {'video_url' : 'https://assets.dreamfight.io/tennis/cranbrook_special/atp_challenger_bloomfield_hills_usa_men_singles_lajal_vs_martin.mp4' , 
                           'annotations' : 'https://assets.dreamfight.io/tennis/cranbrook_special/atp_challenger_bloomfield_hills_usa_men_singles_lajal_vs_martin.json'} } 

    load_urls(urls) 
if __name__ == "__main__":
    main()
