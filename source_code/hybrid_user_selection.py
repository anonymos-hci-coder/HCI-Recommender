# -*- coding: utf-8 -*-
"""
Archetype-Based User Selection for Recommender System Experiments.

This script implements a sophisticated user selection strategy for experiments
aimed at measuring changes in user taste. Instead of selecting a generic
"diverse" set of users, it segments the user base into three distinct archetypes
based on the DIVERSITY and CONCENTRATION of their genre preferences:

1.  **Specialists**: Users with the least diverse tastes.
    Their taste profile is predictable, making them the ideal experimental group
    to observe taste changes.

2.  **Generalists**: Users with the most diverse tastes.
    They serve as a good control group.

3.  **Explorers**: Users who fall between Specialists and Generalists.

This version uses Shannon Entropy with quantiles to guarantee correctly sized
archetype pools, and then employs a cyclical sampling method to ensure balanced
genre representation within the final selected user set.
"""

import os
import pandas as pd
import numpy as np
from scipy.stats import entropy
import pandas as pd
from sklearn.model_selection import StratifiedKFold

# --- Configuration ---
# The total number of users to select for the experiment.
NUM_USERS_TO_SELECT = 150


# Defines the quantile boundaries for creating the archetype pools.
# e.g., [0.03, 0.50] means:
# - Bottom 3% of users by diversity score are Specialists.
# - Users between the 3rd and 50th percentile are Explorers.
# - Top 33% (above 50th percentile) are Generalists.
SEGMENT_QUANTILES = [0.03, 0.50]


# The proportions for each archetype in the FINAL user set.
# The order is [Specialists, Explorers, Generalists].
# These must sum to 1.0.
SELECTION_PROPORTIONS = [0.60, 0.25, 0.15]


# The local directory where the 'ml-32m' dataset is stored.
DATA_PATH = './data/ml-32m'
FINAL_RES_PATH = './data/final_users'


def load_data(data_directory):
    """
    Loads the MovieLens 32M dataset from a local directory.
    """
    print("--- 1. Data Loading ---")
    print(f"Attempting to load data from local path: {data_directory}")

    if not os.path.isdir(data_directory):
        print(f"Error: The specified data directory does not exist: {data_directory}")
        return None, None

    ratings_path = os.path.join(data_directory, "ratings.csv")
    movies_path = os.path.join(data_directory, "movies.csv")

    try:
        print("Loading ratings.csv and movies.csv...")
        ratings_df = pd.read_csv(ratings_path)
        movies_df = pd.read_csv(movies_path)
        print("DataFrames loaded successfully.")
        return ratings_df, movies_df
    except FileNotFoundError as e:
        print(f"Error: {e}. Please ensure the CSV files are in the directory.")
        return None, None


def filter_active_users(ratings_df, min_ratings=400, max_ratings=2500):
    """
    Filters for users with a rating count within the specified range.
    """
    print("\n--- 2. Filtering for Active Users ---")
    user_ratings_count = ratings_df['userId'].value_counts()
    
    active_users = user_ratings_count[
        (user_ratings_count >= min_ratings) & (user_ratings_count <= max_ratings)
    ].index
    
    filtered_ratings_df = ratings_df[ratings_df['userId'].isin(active_users)]
    
    print(f"Original user count: {ratings_df['userId'].nunique()}")
    print(f"Users after filtering (rated {min_ratings}-{max_ratings} movies): {len(active_users)}")
    return filtered_ratings_df


def calculate_user_genre_preferences(filtered_ratings_df, movies_df):
    """
    Calculates a normalized "taste profile" vector for each user.
    """
    print("\n--- 3. Calculating User Genre Preferences ---")
    
    movies_filtered = movies_df[
        (movies_df['genres'] != '(no genres listed)') &
        (~movies_df['genres'].str.contains('IMAX'))
    ].copy()
    
    merged_df = pd.merge(filtered_ratings_df, movies_filtered, on='movieId')
    
    merged_df['genre_list'] = merged_df['genres'].str.split('|')
    exploded_df = merged_df.explode('genre_list')
    
    user_genre_scores = exploded_df.pivot_table(
        index='userId', 
        columns='genre_list', 
        values='rating', 
        aggfunc='sum',
        fill_value=0
    )
    
    user_preferences = user_genre_scores.div(user_genre_scores.sum(axis=1), axis=0)
    
    print(f"Preference matrix created with shape: {user_preferences.shape}")
    return user_preferences


def calculate_diversity_and_dominant_genres(user_preferences_df, concentration_pct=0.75):
    """
    Calculates a diversity score (Entropy) and identifies the set of dominant
    genres for each user.
    """
    print("\n--- 4. Profiling Users with Diversity and Dominant Genres ---")

    profile_data = []
    for user_id, prefs in user_preferences_df.iterrows():
        # Calculate diversity score using Shannon Entropy
        diversity_score = entropy(prefs + 1e-9, base=2)
        
        # Identify dominant genres based on taste concentration
        sorted_prefs = prefs.sort_values(ascending=False)
        cumulative_prefs = sorted_prefs.cumsum()
        num_genres_for_concentration = (cumulative_prefs >= concentration_pct).idxmax()
        num_genres = sorted_prefs.index.get_loc(num_genres_for_concentration) + 1
        dominant_genres = sorted_prefs.head(num_genres).index.tolist()
        
        profile_data.append({
            'userId': user_id,
            'diversity_score': diversity_score,
            'dominant_genres': dominant_genres
        })

    user_profile_df = pd.DataFrame(profile_data).set_index('userId')
    print("User profiles calculated successfully.")
    return user_profile_df


def segment_users_into_archetypes(user_profile_df, segment_boundaries):
    """
    Segments users into archetypes based on diversity score
    quantiles to ensure the pools are correctly sized.
    """
    print("\n--- 5. Segmenting Users into Archetypes via Quantiles ---")
    
    quantiles = user_profile_df['diversity_score'].quantile(segment_boundaries)
    
    print(f"Defining archetype boundaries at diversity score quantiles: {segment_boundaries}")
    print(f"Specialist/Explorer boundary score: {quantiles.iloc[0]:.4f}")
    print(f"Explorer/Generalist boundary score: {quantiles.iloc[1]:.4f}")
    
    def assign_archetype(score):
        if score <= quantiles.iloc[0]:
            return 'Specialist'
        elif score <= quantiles.iloc[1]:
            return 'Explorer'
        else:
            return 'Generalist'

    user_profile_df['archetype'] = user_profile_df['diversity_score'].apply(assign_archetype)
    
    print("\nUsers successfully segmented. Total pool distribution:")
    print(user_profile_df['archetype'].value_counts(normalize=True).round(2))
    
    return user_profile_df


def select_final_users_cyclical(user_archetypes_df, total_users, selection_proportions):
    """
    Selects the final set of users using cyclical sampling for Specialists/Explorers
    to ensure balanced genre representation.
    """
    print("\n--- 6. Selecting Final Users (Cyclical Method) ---")
    
    final_users_list = []
    
    num_specialists = int(total_users * selection_proportions[0])
    num_explorers = int(total_users * selection_proportions[1])
    num_generalists = total_users - (num_specialists + num_explorers)
    
    archetype_targets = {
        'Specialist': num_specialists,
        'Explorer': num_explorers,
        'Generalist': num_generalists
    }

    print(f"Target selection counts per archetype: {archetype_targets}")
    
    selected_user_ids = set()

    for archetype, target_count in archetype_targets.items():
        print(f"\nSampling {target_count} users from the '{archetype}' pool...")
        
        archetype_pool = user_archetypes_df[user_archetypes_df['archetype'] == archetype]
        
        if archetype_pool.empty or target_count == 0:
            print(f"Skipping {archetype}.")
            continue
            
        archetype_selected_ids = []
        if archetype == 'Generalist':
            archetype_selected_ids = np.random.choice(
                archetype_pool.index, 
                size=min(target_count, len(archetype_pool)), 
                replace=False
            ).tolist()
        else:
            genre_to_user_map = {}
            for user_id, data in archetype_pool.iterrows():
                for genre in data['dominant_genres']:
                    if genre not in genre_to_user_map:
                        genre_to_user_map[genre] = []
                    genre_to_user_map[genre].append(user_id)

            available_genres = list(genre_to_user_map.keys())
            np.random.shuffle(available_genres)
            
            genre_idx = 0
            while len(archetype_selected_ids) < target_count and any(g for g in genre_to_user_map if genre_to_user_map[g]):
                current_genre = available_genres[genre_idx % len(available_genres)]
                
                if genre_to_user_map.get(current_genre):
                    user_to_add = np.random.choice(genre_to_user_map[current_genre])
                    
                    if user_to_add not in selected_user_ids:
                        archetype_selected_ids.append(user_to_add)
                        selected_user_ids.add(user_to_add)
                        
                        for g in genre_to_user_map:
                            if user_to_add in genre_to_user_map[g]:
                                genre_to_user_map[g].remove(user_to_add)
                
                genre_idx += 1
                if genre_idx > target_count * len(available_genres) * 2: 
                    print("Could not find enough unique users via cyclical sampling. Breaking.")
                    break

        print(f"Selected {len(archetype_selected_ids)} users for this archetype.")
        for user_id in archetype_selected_ids:
            final_users_list.append({'userId': user_id, 'archetype': archetype})

    if len(final_users_list) < total_users:
        print(f"\nBackfilling {total_users - len(final_users_list)} users...")
        all_pool = user_archetypes_df.index.difference(selected_user_ids)
        num_to_add = min(total_users - len(final_users_list), len(all_pool))
        backfill_ids = np.random.choice(all_pool, size=num_to_add, replace=False)
        for user_id in backfill_ids:
            archetype = user_archetypes_df.loc[user_id, 'archetype']
            final_users_list.append({'userId': user_id, 'archetype': archetype})

    return pd.DataFrame(final_users_list)


def display_archetype_samples(final_users_df, user_preferences_df, num_samples=3):
    """
    Displays sample users from each archetype to provide a qualitative view of taste.
    """
    print("\n--- 7. Qualitative Samples from Each Archetype ---")

    for archetype in ['Specialist', 'Explorer', 'Generalist']:
        print(f"\n-- Samples for '{archetype}' Archetype --")
        
        archetype_user_ids = final_users_df[final_users_df['archetype'] == archetype]['userId'].tolist()
        
        if not archetype_user_ids:
            print("No users selected for this archetype.")
            continue
            
        sample_size = min(num_samples, len(archetype_user_ids))
        sampled_ids = np.random.choice(archetype_user_ids, size=sample_size, replace=False)
        
        for user_id in sampled_ids:
            user_prefs = user_preferences_df.loc[user_id]
            top_genres = user_prefs.nlargest(7)
            
            print(f"\n  User ID: {user_id}")
            print("  Top 7 Genre Preferences:")
            for genre, score in top_genres.items():
                if score > 0:
                    print(f"    - {genre:<12s}: {score:.3f}")


def main():
    """Main execution block to orchestrate the user selection pipeline."""
    
    ratings_df, movies_df = load_data(DATA_PATH)
    if ratings_df is None: return
        
    filtered_ratings_df = filter_active_users(ratings_df)
    user_preferences = calculate_user_genre_preferences(filtered_ratings_df, movies_df)
    
    # Calculate diversity scores and dominant genres in one step
    user_profiles = calculate_diversity_and_dominant_genres(user_preferences)
    
    # Segment users into pools based on diversity quantiles
    user_archetypes = segment_users_into_archetypes(user_profiles, SEGMENT_QUANTILES)
    
    # Select final users from those pools
    final_selected_users_df = select_final_users_cyclical(
        user_archetypes,
        NUM_USERS_TO_SELECT,
        SELECTION_PROPORTIONS
    )

    # --- Final Verification and Save ---
    print("\n--- 8. Final Verification and Output ---")
    if not final_selected_users_df.empty:
        print(f"\nTotal users selected: {len(final_selected_users_df)}")
        print("Final distribution of users across archetypes:")
        print(final_selected_users_df['archetype'].value_counts())
        
        specialist_ids = final_selected_users_df[final_selected_users_df['archetype'] == 'Specialist']['userId']
        
        if not specialist_ids.empty:
            specialist_info = user_archetypes.loc[specialist_ids]
            dominant_genre_counts = specialist_info['dominant_genres'].explode().value_counts()
            print("\nDominant Genre distribution within the selected 'Specialist' group:")
            print(dominant_genre_counts)
        else:
            print("\nNo 'Specialist' users were selected.")

        display_archetype_samples(final_selected_users_df, user_preferences)

        output_filename = os.path.join(FINAL_RES_PATH,'archetype_selected_users.csv')
        final_selected_users_df.to_csv(output_filename, index=False)
        print(f"\nSuccessfully saved the final user list to '{output_filename}'.")
    else:
        print("\nNo users were selected. An issue occurred during sampling.")
        

    # # We can Split this dataset into 'n_splits' distinct parts, to be ready for multi-processing! 
    # df = final_selected_users_df

    # # Prepare stratified split (3 folds)
    # skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

    # # Split and save
    # for i, (_, test_idx) in enumerate(skf.split(df, df["archetype"]), start=1):
    #     split_df = df.iloc[test_idx]
    #     f_name = os.path.join(FINAL_RES_PATH,f"split_{i}.csv")
    #     split_df.to_csv(f_name, index=False)

    # print("Done! Created split_1.csv, split_2.csv, split_3.csv")



if __name__ == "__main__":
    main()

