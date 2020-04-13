import sys
import pandas as pd
from sqlalchemy import create_engine

# python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db


def load_data(messages_filepath, categories_filepath):
    '''
    input:
        messages_filepath: The path to the messages csv files.
        categories_filepath: The path to the categories csv files.
    output:
        df: The merged csv and categories dataset 
    '''
    
    #load data from csv
    messages = pd.read_csv(messages_filepath) 
    categories = pd.read_csv(categories_filepath)
    df = messages.merge(categories, how='outer', on=['id'])
    
    return df







def clean_data(df):
    '''
    input:
        df: The merged csv and categories dataset 
    output:
        df: The dataframe data after cleaning and elimination of unwanted texts.
    '''
    categories  = df['categories'].str.split(';', n=-1, expand=True)
    categories.columns = categories.iloc[1]
    
    categories.rename(columns=lambda x: x[0:-2], inplace=True)
    
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].str[-1]
    
        # convert column from string to numeric
        categories[column] = categories[column].astype(int)
    


    df = df.drop(['categories'], axis = 1) 
    df = pd.concat([df, categories], axis=1) 
    
    df_duplicate = df[df.duplicated()]
    
    df.drop_duplicates(inplace = True) 

    return df






def save_data(df, database_filepath):
    '''
    input:
        df: The cleaned dataframe
        database_filepath: Database where table could be created
        
    output:
        None
    '''
    engine = create_engine('sqlite:///' + database_filepath)
    df.to_sql('messages_disaster', engine, index=False)


    
    

def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
        
   
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()
    
    