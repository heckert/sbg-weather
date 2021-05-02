# -*- coding: utf-8 -*-
import pandas as pd

from src.globals import raw_data_dir, interim_data_dir, processed_data_dir,\
    data_source, logging

def main(input_url, output_filepath):
    """ 
    Loads raw csv file from url specified in globals.py
    Loaded file is stored in data directory
    """
    #df = pd.read_csv(input_url)
    #df.to_csv(output_filepath, index=False, encoding='utf8')
    pass
    

if __name__ == '__main__':
    for dir_ in [raw_data_dir, interim_data_dir, processed_data_dir]:
        if not dir_.exists():
            dir_.mkdir(parents=True)
            logging.debug(f'{dir_} created')
        else:
            logging.debug(f'{dir_} exists')
    logging.info('loading raw data from {}'.format(data_source['url']))
    main(data_source['url'], raw_data_dir/data_source['name'])
    logging.info('data stored at {}'.format(raw_data_dir/data_source['name']))
    
