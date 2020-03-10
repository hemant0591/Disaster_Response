import sys
import pandas as pd

def load_data(messages_filepath, categories_filepath):

    """
    This function reads csv file and merges them in a single pandas dataframe

    Args:

            messages_filepath: filepath where messages file is saved
            categories_filepath: filepath where categories file is saved

    Return:

            Pandas dataframe containing messages and categories

    """

    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)

    df = messages.merge(categories, on='id')

    return df


def clean_data(df):

    """
    This function cleans the dataframe so that we can analyze the data better

    Args:
            dataframe

    Returns:
            cleaned dataframe

    """

    categories = df['categories'].str.split(';', expand=True)
    row = categories.iloc[0]
    category_colnames = row.apply(lambda x: x.split('-')[0])
    categories.columns = category_colnames

    for column in categories:
        categories[column] = categories[column].astype(str)
        categories[column] = categories[column].apply(lambda x: x.split('-')[1])

        categories[column] = categories[column].astype(int)

    df.drop('categories', axis=1, inplace=True)

    df = pd.concat([df, categories], axis=1)

    dupli = df[df.duplicated()]

    df = df.drop(dupli.index)

    return df



def save_data(df, database_filename):

    """
    This function takes clean dataframe and stores the data into a sqlite database

    Args:
            df: dataframe
            database_filename: filename of the database

    Returns:
            Nothing
    
    """

    import sqlite3
    import sqlalchemy
    from sqlalchemy import create_engine

    engine = create_engine('sqlite:///{}'.format(database_filename))
    df.to_sql('DisasterResponse', engine, index=False, if_exists='replace')


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
