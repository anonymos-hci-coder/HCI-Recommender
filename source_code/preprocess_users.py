import pandas as pd
import requests
import os
import time
import sys
import json
from utils.utils import PROXY_SERVER


def load_cache(cache_path):
    """Loads the TMDB cache from a JSON file."""
    if os.path.exists(cache_path):
        print(f"Loading TMDB cache from {cache_path}...")
        with open(cache_path, 'r', encoding="utf8") as f:
            return json.load(f)
    print("No cache file found. A new one will be created.")
    return {}

def save_cache(cache, cache_path, flag=True):
    """Saves the TMDB cache to a JSON file."""
    if flag:
        print(f"\nSaving TMDB cache to {cache_path}...")
    with open(cache_path, 'w', encoding="utf8") as f:
        json.dump(cache, f, indent=4)
    if flag:
        print("Cache saved.")

def get_movie_data_from_tmdb(tmdb_id, api_key, cache):
    """
    Fetches movie data from the TMDB API, using a cache to avoid redundant calls.
    Returns the movie data dictionary on success, None on failure.
    """
    # TMDB IDs are numbers, but JSON keys must be strings for compatibility.
    tmdb_id_str = str(int(tmdb_id))

    # 1. Check the cache first
    if tmdb_id_str in cache:
        return cache[tmdb_id_str], True     #True means a cache-hit, False means cache-miss

    # 2. If not in cache, call the API
    try:
        api_url = f"{PROXY_SERVER}/3/movie/{tmdb_id_str}?api_key={api_key}"
        response = requests.get(api_url)
        response.raise_for_status() 
        
        movie_data = response.json()
        if movie_data:
            cache[tmdb_id_str] = movie_data
            
            save_cache(cache, TMDB_CACHE_FILE, False)
        return movie_data,  False

    except requests.exceptions.HTTPError as http_err:
        sys.stdout.write('\n')
        print(f"HTTP error for tmdbId {tmdb_id}: {http_err} - Skipping movie.")
        return None, None
    except Exception as e:
        sys.stdout.write('\n')
        print(f"An error occurred for tmdbId {tmdb_id}: {e} - Skipping movie.")
        return None, None


def enrich_movies_for_user(data_directory, output_file, api_key, tmdb_cache, specific_user_id):
    print("-" * 80)
    print(f"Starting process for User ID: {specific_user_id}")
    
    if not api_key or api_key == 'YOUR_TMDB_API_KEY':
        print("Error: Please provide a valid TMDB API key.")
        return
    
    try:
        movies_file = os.path.join(data_directory, 'movies.csv')
        links_file = os.path.join(data_directory, 'links.csv')
        ratings_file = os.path.join(data_directory, 'ratings.csv')

        # Load the necessary datasets
        print("Loading core MovieLens data...")
        movies_df = pd.read_csv(movies_file)
        links_df = pd.read_csv(links_file)
        ratings_df = pd.read_csv(ratings_file)

        # Merge movies and links to get tmdbId
        movies_with_links_df = pd.merge(movies_df, links_df, on='movieId')
        
        # --- Filter for the specific user and get their ratings ---
        print(f"Filtering for movies rated by user ID: {specific_user_id}")
        user_ratings_df = ratings_df[ratings_df['userId'] == specific_user_id][['movieId', 'rating']]
        
        user_movie_ids = user_ratings_df['movieId'].unique()
        
        movies_for_user_df = movies_with_links_df[
            movies_with_links_df['movieId'].isin(user_movie_ids)
        ].copy() 
        
        if len(movies_for_user_df) == 0:
            print(f"User {specific_user_id} has no rated movies to process. Skipping.")
            return

        print(f"Processing a sample of {len(movies_for_user_df)} movies for user {specific_user_id}.")

    except FileNotFoundError as e:
        print(f"Error: {e}. Please ensure the data directory and files are correct.")
        return

    processed_movies_df = None
    if os.path.exists(output_file):
        print(f"Output file found. Loading existing data to resume...")
        processed_movies_df = pd.read_csv(output_file)
        processed_movie_ids = set(processed_movies_df['movieId'])
        
        movies_to_process_df = movies_for_user_df[
            ~movies_for_user_df['movieId'].isin(processed_movie_ids)
        ].copy()
        print(f"{len(processed_movie_ids)} movies already processed. "
              f"Resuming with {len(movies_to_process_df)} remaining movies.")
    else:
        print("No existing output file found. Starting from scratch.")
        movies_to_process_df = movies_for_user_df.copy()

    # --- Iterate and fetch data from TMDB API ---
    total_movies = len(movies_to_process_df)
    if total_movies == 0:
        print("All movies for this user's sample have already been processed.")
        return
        
    print("Starting to fetch data from TMDB API (using cache)...")
    enriched_rows = []
    processed_count = 0
    cache_hit_count = 0
    for index, row in movies_to_process_df.iterrows():
        tmdb_id = row['tmdbId']
        movie_id = row['movieId']

        processed_count += 1
        percentage = (processed_count / total_movies) * 100
        bar_length = 50
        filled_length = int(bar_length * processed_count // total_movies)
        bar = '#' * filled_length + '-' * (bar_length - filled_length)
        sys.stdout.write(f'\rProgress: |{bar}| {percentage:.2f}% Complete ({processed_count}/{total_movies}) , (Cache hit {cache_hit_count})')
        sys.stdout.flush()

        if pd.isna(tmdb_id):
            continue
            
        # Use the caching function to get movie data
        movie_data, cache_hit = get_movie_data_from_tmdb(tmdb_id, api_key, tmdb_cache)
        
        if not movie_data:
            continue
        
        if cache_hit:
            cache_hit_count+=1    
            
        tmdb_genres_list = [genre['name'] for genre in movie_data.get('genres', [])]

        new_row = {
            'movieId': movie_id,
            'title': row['title'],
            'genres': row['genres'],
            'imdbId': row['imdbId'],
            'tmdbId': tmdb_id,
            'overview': movie_data.get('overview', ''),
            'tmdb_vote_average': movie_data.get('vote_average', 0),
            'tmdb_vote_count': movie_data.get('vote_count', 0),
            'tmdb_genres': tmdb_genres_list
        }
        enriched_rows.append(new_row)

    # --- Final Save for this user ---
    sys.stdout.write('\n')
    if enriched_rows:
        print("Saving enriched data...")
        temp_df = pd.DataFrame(enriched_rows)
        if processed_movies_df is not None:
            # Combine previously processed data with the new batch
            final_df = pd.concat([processed_movies_df, temp_df], ignore_index=True)
        else:
            final_df = temp_df
        
        final_df_with_ratings = pd.merge(final_df, user_ratings_df, on='movieId', how='left')
        
        
        final_df_with_ratings.to_csv(output_file, index=False)

    print(f"Enrichment process complete for User ID: {specific_user_id}")
    print("-" * 80 + "\n")


if __name__ == '__main__':
    # --- Configuration ---
    DATA_DIR = 'data/ml-32m' # Path to your MovieLens data directory
        
    
    if len(sys.argv) != 3:
        raise ValueError("not enough args")
    chunk = sys.argv[1]
    TMDB_API_KEY = sys.argv[2]
    
    USER_LIST_FILE = f'./data/final_users/split_{chunk}.csv'

    
    # 2. Path for the TMDB cache file (will be created if it doesn't exist)
    TMDB_CACHE_FILE = f'data/tmdb_api_cache_{chunk}.json'

    # --- Main Execution Logic ---
    
    # Load the TMDB cache once at the start
    tmdb_api_cache = load_cache(TMDB_CACHE_FILE)
    
    # Load the list of user IDs to process
    try:
        user_ids_df = pd.read_csv(USER_LIST_FILE)
        # Ensure the CSV has a column named 'userId'
        if 'userId' not in user_ids_df.columns:
            raise ValueError("CSV file must contain a 'userId' column.")
        user_ids_to_process = user_ids_df['userId'].tolist()
        print(f"Found {len(user_ids_to_process)} user IDs to process from {USER_LIST_FILE}.")
    except FileNotFoundError:
        print(f"Error: User list file not found at '{USER_LIST_FILE}'.")
        print("Please create this file with a 'userId' column.")
        sys.exit(1) # Exit the script if the user list is missing
    except Exception as e:
        print(f"An error occurred while reading the user list file: {e}")
        sys.exit(1)
        
    # 3. Loop through each user ID and run the enrichment process
    for user_id in user_ids_to_process:
        # Define the output file name dynamically for each user
        output_file_for_user = f'data/final_users/movies_enriched-user{user_id}.csv'

        enrich_movies_for_user(
            data_directory=DATA_DIR,
            output_file=output_file_for_user,
            api_key=TMDB_API_KEY,
            tmdb_cache=tmdb_api_cache,
            specific_user_id=user_id
        )

        # 4. Save the updated cache once at the very end of all processing
        save_cache(tmdb_api_cache, TMDB_CACHE_FILE)


    
    print("All specified users have been processed.")