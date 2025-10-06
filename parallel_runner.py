import os
import glob
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed

from multiprocessing import Manager
from filelock import FileLock 

from main import run_simulation_for_user, load_movie_database
import sys

JSON_PATH = "data/tmdb_api_cache.json"

try:
    MAX_WORKERS = max(1, 5)
except NotImplementedError:
    MAX_WORKERS = 4
    print("Could not determine CPU count. Defaulting to 4 workers.")


def main_parallel_runner(api_key, data_directory, tmdb_key):
    """
    Finds all user files and runs the simulation for them in parallel.
    """
    print(f"Starting parallel simulation for all users...")
    print(f"Using a maximum of {MAX_WORKERS} parallel processes.")

    # --- Create the Lock ---
    # The lock file itself is used to coordinate the processes.

    lock_path = "data/cache2.lock"

    with FileLock(lock_path) as lock:
        # 1. Load shared data ONCE before starting parallel processes
        all_movies_df = pd.read_csv('data/ml-32m/movies.csv')
        load_movie_database(JSON_PATH, lock=lock)

    # 2. Get the list of all user files to process
    user_files = glob.glob(os.path.join(data_directory, 'movies_enriched-user*.csv'))

    if not user_files:
        print("No user files found to process. Exiting.")
        return

    # 3. Use ProcessPoolExecutor to manage the parallel execution
    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        # Submit each user simulation as a separate job to the process pool
        future_to_user = {
            executor.submit(run_simulation_for_user, user_file, all_movies_df, lock_path, api_key , tmdb_key): user_file
            for user_file in user_files
        }

        # Process the results as they are completed for real-time feedback
        for future in as_completed(future_to_user):
            user_file = future_to_user[future]
            try:
                # Retrieve the result from the completed job
                result = future.result()
                print(f"COMPLETED: {os.path.basename(user_file)} -> {result}")
            except Exception as exc:
                print(f"ERROR: {os.path.basename(user_file)} generated an exception: {exc}")

    print("\nAll user simulations have been processed.")


if __name__ == "__main__":

    if len(sys.argv) != 4:
        raise ValueError("not enough args")
    username = sys.argv[1]
    API_KEY = sys.argv[2]
    TMDB_KEY = sys.argv[3]

    DATA_DIRECTORY = f"data/users_{username}/"
    print(DATA_DIRECTORY)

    main_parallel_runner(API_KEY, DATA_DIRECTORY, TMDB_KEY)
