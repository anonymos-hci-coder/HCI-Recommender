import os
import json
import re
import copy
import glob
import pandas as pd
import requests
import random
from termcolor import cprint
from typing import List, Dict, Any, Optional
from source_code.recommender.recommend import get_recommendation as algorithmic_recomendation
import json
from datetime import datetime
import openai
from tenacity import retry, stop_after_attempt, wait_fixed, retry_if_exception_type
import traceback
from filelock import FileLock
from openai import OpenAI
import ast

# --- LLM Client Configuration ---
# Attempts to import a predefined client
try:
    from source_code.utils.utils import SIMULATOR_LLM_NAME, TMDB_API_KEY, RESULTS_DIRECTORY, \
        RECOMMENDER_LLM_NAME, TEMPERATURE_SIM, TEMPERATURE_REC, PROXY_SERVER, BASE_URL, API_KEY

    print("Successfully imported custom client.")
except (TypeError, ImportError) as e:
    print("WARNING: Could not import custom client. Using a dummy client for demonstration.")
    raise e


# --- New Logging Function ---
def log(prompt, color="white", on_color=None, end="\n", file_path=None):
    """
    Prints a message to the console and optionally appends it to a log file.

    Args:
        prompt (str): The message to log.
        color (str, optional): Color for the console output. Defaults to "white".
        on_color (str, optional): Background color for the console output. Defaults to None.
        end (str, optional): String appended after the prompt. Defaults to "\n".
        file_path (str, optional): Path to the log file. Required if log_to_file is True.
    """
    # 1. Always print the message to the console using cprint
    cprint(text=prompt, color=color, on_color=on_color, force_color=True, end=end)

    # 2. If log_to_file is True, append the prompt to the specified file
    # Check if a file path was provided
    if file_path:
        try:
            # Ensure the directory for the log file exists
            log_dir = os.path.dirname(file_path)
            if log_dir and not os.path.exists(log_dir):
                os.makedirs(log_dir)

            # Open the file in append mode ('a') and write the uncolored prompt
            with open(file_path, 'a', encoding='utf-8') as f:
                f.write(str(prompt) + end)
        except IOError as e:
            cprint(f"Error: Could not write to file at {file_path}. Reason: {e}", color="red")


# --- System Prompt Templates ---

# 1. For the User Simulator LLM
# This prompt grounds the LLM in its role as a specific movie enthusiast.
USER_SIMULATOR_SYSTEM_PROMPT = """
### ROLE AND GOAL
You are an advanced AI simulating a specific human movie enthusiast. Your primary goal is to internalize the persona defined by the user data provided below and then act *exactly* as that person would. You must deduce their personality, tastes, biases, and viewing habits from their movie rating history, and genre preferences.

### PERSONA GROUNDING DATA
This is the complete history of the user you are simulating. Analyze it deeply to understand their character.

**User's Movie Ratings and History:**
```
{user_history}
```

**Key Personality Traits to Deduce:**
- **Rating Strictness:** Is this user a harsh critic (ratings are low) or easy to please (ratings are high)? What is their average rating?
- **Genre Affinity:** What are their favorite and least favorite genres? Do they prefer mainstream blockbusters or niche indie films?
- **Thematic Preferences:** Based on movie overviews, what themes and movie characteristics do they enjoy or dislike?
- **Conformity:** Do their ratings generally align with the global average rating for a movie, or do they have contrarian tastes?

### CORE TASK
Your task will be provided in the user message. It will typically involve one of two things:
1.  **Rating a New Movie:** You will be given a new movie's details and asked to predict a rating from 1-5, providing a brief justification based on your established persona.
2.  **Analyzing a Recommendation List:** You will be given a list of recommended movies and asked to perform a multi-step analysis.

### CONSTRAINTS AND BEHAVIORAL RULES
- **BE THE PERSONA:** Do not break character. Do not mention you are an AI. All responses must come from the point of view of the simulated user.
- **GROUNDING IS EVERYTHING:** Base all your decisions, ratings, and reasoning *exclusively* on the "Persona Grounding Data". Do not use any external knowledge.
- **FOLLOW FORMATTING:** Adhere strictly to any output format requested in the user message. This is critical for programmatic parsing of your response. Do not add extra text, apologies, or explanations outside the requested format.
"""

# 2. For the Standard Recommender LLM
RECOMMENDER_SYSTEM_PROMPT_STANDARD = """
### ROLE AND GOAL
You are a helpful, expert movie recommender AI. Your goal is to analyze a user's movie rating history and experience and recommend a list of new and unwatched movies they are most likely to enjoy.

### USER'S HISTORY
```
{user_history}
```

### CORE TASK
Analyze the user's tastes (genres, themes, actors, etc.) from their history and experience. Then, select exactly {movies_per_page} real, existing, and unwatched movies that best match their preferences.

### CONSTRAINTS AND BEHAVIORAL RULES
- **SUGGEST REAL MOVIES:** Recommend movies that are well-known and likely to exist. Do not invent movie titles.
- **AVOID REPEATS:** Do not recommend movies that are already in the user's history or experience.
- **FORMATTING:** Return only a bar-separated list of movie titles with their release year.
- **Example Output:** `The Matrix (1999)| Inception (2010)| Parasite (2019)| The Godfather (1972)`
"""

# 3. For the Persuasive Recommender LLM
RECOMMENDER_SYSTEM_PROMPT_PERSUASIVE = """
### ROLE AND GOAL
You are a strategic movie recommender AI. Your goal is to analyze a user's movie rating history and experience and recommend {movies_per_page} new and unwatched movies that will subtly broaden their tastes. You want to introduce them to genres they typically dislike by finding "bridge" movies.

### USER'S HISTORY
```
{user_history}
```

### CORE TASK
1.  **Analyze:** Identify the user's favorite genres (high ratings) and least favorite genres (low or no ratings).
2.  **Strategize:** Find "bridge" movies. A bridge movie belongs to a genre the user dislikes but has elements (like a favorite actor, director, or a secondary genre) that they do like.
3.  **Recommend:** Select exactly {movies_per_page} real, existing, and unwatched movies. Your list should be a mix of "safe" recommendations and strategic "bridge" movies.

### CONSTRAINTS AND BEHAVIORAL RULES
- **SUGGEST REAL MOVIES:** Recommend movies that are well-known and likely to exist. Do not invent titles.
- **AVOID REPEATS:** Do not recommend movies that are already in the user's history or experience.
- **FORMATTING:** Return only a bar-separated list of movie titles with their release year.
- **Example Output:** `Spirited Away (2001)| The Silence of the Lambs (1991)| Her (2013)`
"""

# --- Global Movie Database ---
# A dictionary to hold all movie details, loaded once to save memory and time.
MOVIE_DB = {}
DATA_CACHE = {}
JSON_PATH = "data/tmdb_api_cache.json"


def search_and_add_movie_to_db(title: str, year: Optional[str], log_file_path: Optional[str], tmdb_key):
    """Searches for a movie on TMDB and adds it to the global MOVIE_DB if found."""
    global MOVIE_DB
    global DATA_CACHE

    log(f"--> Movie '{title} ({year})' not in local DB. Searching TMDB...", color="blue", file_path=log_file_path)

    try:
        search_url = f"{PROXY_SERVER}/3/search/movie?query={requests.utils.quote(title)}&year={year}&api_key={tmdb_key}"
        search_response = requests.get(search_url, timeout=5)
        search_response.raise_for_status()
        search_results = search_response.json().get('results', [])

        if not search_results:
            log(f"--> Movie '{title} ({year})' Not found in TMDB! Skipping...", color="yellow",
                file_path=log_file_path)
            return False

        movie_id = search_results[0]['id']
        for sr in search_results:
            release_date = sr.get("release_date", "")
            release_year = None
            if release_date:
                try:
                    release_year = datetime.strptime(release_date, "%Y-%m-%d").year
                except ValueError:
                    pass  # invalid date format
                if release_year == int(year):
                    movie_id = sr['id']
                    break

        details_url = f"{PROXY_SERVER}/3/movie/{movie_id}?api_key={tmdb_key}"
        details_response = requests.get(details_url, timeout=5)
        details_response.raise_for_status()
        details = details_response.json()
        details['m_title'] = title
        DATA_CACHE[movie_id] = details

        # Use the official title and year from TMDB for consistency
        release_year = details.get('release_date', '????')[:4]
        official_title = details.get('title', title)
        full_title_key = f"{official_title} ({release_year})"
        full_title2_key = f"{title} ({release_year})"

        MOVIE_DB[full_title_key] = {
            "overview": details.get('overview', 'N/A'),
            "genres": [g['name'] for g in details.get('genres', [])],
            "tmdb_genres": [g['name'] for g in details.get('genres', [])],
            "avg_rating": round(details.get('vote_average', 0) / 2, 2)
        }
        MOVIE_DB[full_title2_key] = {
            "overview": details.get('overview', 'N/A'),
            "genres": [g['name'] for g in details.get('genres', [])],
            "tmdb_genres": [g['name'] for g in details.get('genres', [])],
            "avg_rating": round(details.get('vote_average', 0) / 2, 2)
        }
        log(f"--> Successfully found and added '{full_title_key}' to local DB.", color="blue", file_path=log_file_path)
        return True

    except requests.exceptions.RequestException as e:
        log(f"--> TMDB API request failed for '{title} ({year})': {e}", color="red", file_path=log_file_path)
        return False


def save_or_update_data_cache(lock: FileLock):
    global DATA_CACHE
    global JSON_PATH
    with lock:
        try:
            # It's safer to read the latest version of the file first,
            # update it in memory, and then write it back.
            if os.path.exists(JSON_PATH):
                with open(JSON_PATH, "r", encoding="utf-8") as f:
                    on_disk_cache = json.load(f)

                # Merge the current process's cache into what's on disk
                on_disk_cache.update(DATA_CACHE)
                DATA_CACHE = on_disk_cache

            with open(JSON_PATH, "w", encoding="utf-8") as json_file:
                json.dump(DATA_CACHE, json_file, indent=2)
        except (IOError, json.JSONDecodeError) as e:
            log(f"Error during cache save/update: {e}", color="red")
            raise


def clean_title(title):
    cleaned = re.sub(r'\s*\(a\.k\.a\..*?\)', '', title)
    if not title == cleaned:
        log(f"original title: {title} \t cleaned title: {cleaned}", color="red", on_color="on_white")
    return cleaned


def ensure_movies_in_db(movie_titles: List[str], log_file_path: Optional[str], lock: FileLock, tmdb_key):
    """Iterates through a list of 'Title (YYYY)' strings and ensures they exist in the global MOVIE_DB."""
    log(f"--- Checking new movies exist in the databse... ---", color="blue", file_path=log_file_path)
    for full_title in movie_titles:
        # Remove any (a.k.a. ...) block
        cleaned = clean_title(full_title)
        if cleaned in MOVIE_DB:
            continue

        match = re.match(r'^(.*?)\s*\((\d{4})\)$', cleaned.strip())
        if match:
            title, year = match.groups()
            search_and_add_movie_to_db(title=title.strip(), year=year, log_file_path=log_file_path, tmdb_key=tmdb_key)
        else:
            log(f"Warning: Could not parse title and year from '{full_title}'. Skipping TMDB search.", color="yellow",
                file_path=log_file_path)
    save_or_update_data_cache(lock=lock)


def load_movie_database(json_path: str, lock: FileLock):
    """Loads movies from a single JSON cache file into MOVIE_DB."""
    global MOVIE_DB
    global DATA_CACHE

    # Use the lock to ensure no other process is writing to the file while we read it.
    with lock:
        if not os.path.exists(json_path):
            log(f"JSON cache file not found: {json_path}", color="red")
            return

        try:
            with open(json_path, "r", encoding="utf-8") as f:
                DATA_CACHE = json.load(f)

            for movie_id, movie_data in DATA_CACHE.items():
                try:
                    if movie_data is None:
                        continue
                    original_title = movie_data.get("title")
                    m_title = movie_data.get("m_title")
                    release_year = datetime.strptime(movie_data.get("release_date"), "%Y-%m-%d").year
                    if not original_title:
                        continue
                    full_title_key = f"{original_title} ({release_year})"
                    full_m_title_key = f"{m_title} ({release_year})"
                    if full_title_key not in MOVIE_DB:
                        MOVIE_DB[full_title_key] = {
                            "overview": movie_data.get("overview", "N/A"),
                            "genres": [g.get("name") for g in movie_data.get("genres", [])],
                            "tmdb_genres": [g.get("name") for g in movie_data.get("genres", [])],
                            "avg_rating": round(float(movie_data.get("vote_average", 0)) / 2, 2)
                        }

                    if full_m_title_key not in MOVIE_DB:
                        MOVIE_DB[full_m_title_key] = {
                            "overview": movie_data.get("overview", "N/A"),
                            "genres": [g.get("name") for g in movie_data.get("genres", [])],
                            "tmdb_genres": [g.get("name") for g in movie_data.get("genres", [])],
                            "avg_rating": round(float(movie_data.get("vote_average", 0)) / 2, 2)
                        }
                except Exception as e:
                    log(f"Could not process Movie {movie_id}: {e}", color="red")
                    pass
            log(f"Movie database created with {len(MOVIE_DB)} unique movies.", color="blue")

        except Exception as e:
            log(f"Could not process JSON cache {json_path}: {e}", color="red")
            raise


def get_movie_details(movie_title: str) -> Dict[str, Any]:
    """Fetches movie data from the global in-memory database."""
    cleaned_title = clean_title(movie_title)
    return MOVIE_DB.get(cleaned_title,
                        {"overview": "No overview found.", "genres": ["Unknown"], "tmdb_genres": ["Unknown"],
                         "avg_rating": 3.0})


def parse_llm_output(response_text: str) -> Dict[str, Any]:
    """Parses the LLM's structured response to extract ratings and other data."""
    parsed_data = {"ratings": [], "alignments": [], "watch_decision": {}}

    align_pattern = r"MOVIE:\s*(.*?);\s*ALIGN:\s*(yes|no);\s*REASON:\s*(.*?)(?:;|$)"
    for title, align, reason in re.findall(align_pattern, response_text, re.DOTALL | re.MULTILINE):
        parsed_data["alignments"].append({"title": title.strip(), "align": align.strip(), "reason": reason.strip()})

    watch_pattern = r"NUM:\s*(\d+);\s*WATCH:\s*(.*?);\s*REASON:\s*(.*?)(?:;|$)"
    watch_match = re.search(watch_pattern, response_text, re.DOTALL | re.MULTILINE)
    if watch_match:
        parsed_data["watch_decision"] = {
            "num": int(watch_match.group(1).strip()),
            "watch_list": [m.strip() for m in watch_match.group(2).split('|')],
            "reason": watch_match.group(3).strip()
        }

    rating_pattern = r"MOVIE:\s*(.*?);\s*RATING:\s*([\d.]+);\s*FEELING:\s*(.*?)(?:;|$)"
    for title, rating, feeling in re.findall(rating_pattern, response_text, re.DOTALL | re.MULTILINE):
        try:
            parsed_data["ratings"].append({"title": title.strip(), "rating": float(rating), "feeling": feeling.strip()})
        except ValueError:
            continue

    return parsed_data


# --- Core Simulator Class ---
class UserSimulator:
    def __init__(self, user_data: Dict[str, Any], log_file_path: Optional[str],
                 site_url: str = "http://localhost:8080",
                 app_name: str = "LLM_User_Simulator", ):
        self.user_id = user_data.get("user_id", "unknown_user")
        self.site_url = site_url
        self.log_file_path = log_file_path
        self.app_name = app_name
        self.original_ratings = user_data.get("rated_movies", [])
        self.simulated_ratings = []
        self.test_ratings = []
        log(f"\n--- Initializing Persona for {self.user_id} with {len(self.original_ratings)} movies ---",
            file_path=log_file_path)
        self._update_persona_prompt()
        log(f"--- Persona for {self.user_id} is ready. ---", file_path=log_file_path)

    def _format_user_history(self) -> str:
        history_lines = ["**User Ratings:**"]
        for movie in self.original_ratings:
            genres = movie['tmdb_genres']
            history_lines.append(
                f"- Movie: {movie['title']}; User's Rating: {movie['rating']}/5; Genres: {', '.join(genres)}")
        history_lines.append("\n**My Experiences:**")
        if not self.simulated_ratings:
            history_lines.append("- None yet.")
        for movie in self.simulated_ratings:
            history_lines.append(
                f"- Movie: {movie['title']}; Genres: {', '.join(movie['genres'])}; My Rating: {movie['rating']}/5, My Feeling: \"{movie['feeling']}\"")
        return "\n".join(history_lines)

    def _update_persona_prompt(self):
        user_history_string = self._format_user_history()
        self.system_prompt = USER_SIMULATOR_SYSTEM_PROMPT.format(user_history=user_history_string)
        self.base_chat_history = [{"role": "system", "content": self.system_prompt}]

    def add_new_simulated_ratings(self, new_ratings: List[Dict[str, Any]], is_test=False):
        if not new_ratings: return

        if is_test:
            log(f"--> Adding {len(new_ratings)} new test rating(s) to {self.user_id}'s history.", color="blue",
                file_path=self.log_file_path)
            self.test_ratings.append(new_ratings)
        else:
            log(f"--> Adding {len(new_ratings)} new simulated rating(s) to {self.user_id}'s history.", color="blue",
                file_path=self.log_file_path)
            self.simulated_ratings.extend(new_ratings)
            self._update_persona_prompt()

    def copy(self):
        return copy.deepcopy(self)

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_fixed(60),  # Wait 1 minute between retries
        reraise=True  # Re-raise the exception if all retries fail
    )
    def _run_llm_completion(self, user_prompt: str, history: List[Dict], client) -> str:
        messages = history + [{"role": "user", "content": user_prompt}]
        log(f"History:\n", color="cyan", file_path=self.log_file_path)
        for h in history:
            log(f"{h['role']}:{h['content']}\n", color="cyan", file_path=self.log_file_path)
        log(f"User Prompt:\n {user_prompt}", color="cyan", file_path=self.log_file_path)
        try:
            response = client.chat.completions.create(
                model=SIMULATOR_LLM_NAME,
                messages=messages,
                temperature=TEMPERATURE_SIM,
                extra_headers={"HTTP-Referer": self.site_url, "X-Title": self.app_name}
            )
            return response.choices[0].message.content
        except Exception as e:
            log(f"ERROR: Simulator LLM call failed: {e}", color="red", file_path=self.log_file_path)
            raise e

    def analyze_recommendation_list(self, recommendation_list: List[str], lock: FileLock, client, tmdb_key,
                                    is_test=False) -> \
            Dict[
                str, Any]:
        reco_details = []
        title_to_genres = {}
        ensure_movies_in_db(recommendation_list, self.log_file_path, lock, tmdb_key)
        reco_details = []
        for title in recommendation_list:
            details = get_movie_details(movie_title=title)
            if details['overview'] == 'No overview found.':
                continue

            # Store the genres in our map using the title as the key
            title_to_genres[title] = details.get('tmdb_genres', [])

            reco_details.append(
                f"- Title: {title}; Genres: {', '.join(details['tmdb_genres'])}; Overview: {details['overview']}")

        task_description = f"""
Please respond to all the movies in the ## Recommended List ## and provide explanations.

## Recommended List ##
{"\n".join(reco_details)}
## End of List ##

Assume it's your first time watching the recommended movies, and rate them all on a scale from 0.0 to 5.0 (in 0.5 increments) to reflect different degrees of liking, considering your feeling and conformity trait. Use this format:
MOVIE:[Recommended Movie]; RATING: [a number from 0.0 to 5.0 in 0.5 increments]; FEELING: [aftermath sentence]; 

Do not include any additional information or explanations and stay grounded."""

        llm_response = self._run_llm_completion(user_prompt=task_description, history=self.base_chat_history,
                                                client=client)
        log("\n--- LLM Simulation Response ---", color="green", file_path=self.log_file_path)
        log(llm_response, color="green", file_path=self.log_file_path)
        log("-------------------------------\n", color="green", file_path=self.log_file_path)

        parsed_output = parse_llm_output(response_text=llm_response)

        # 2. Loop through the parsed ratings and add the stored genres from the map
        if "ratings" in parsed_output:
            for rating_dict in parsed_output["ratings"]:
                title = rating_dict.get("title")
                if title in title_to_genres:
                    # Add the 'genres' key to the dictionary
                    rating_dict["genres"] = title_to_genres[title]

        self.add_new_simulated_ratings(new_ratings=parsed_output.get("ratings", []), is_test=is_test)
        return parsed_output


# --- Helper and Experimental Path Functions ---

def log_path_step(log_filepath: str, step_name: str, recommended_movies: List[str], analysis_output: Dict[str, Any]):
    """Appends the results of a single recommendation step to a log file in a visually enhanced format."""
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # --- Build the log entry string ---
    log_lines = []

    # Header
    log_lines.append(f"\n{'=' * 80}")
    log_lines.append(f"LOG ENTRY @ {timestamp}")
    log_lines.append(f"STEP:      {step_name.upper()}")
    log_lines.append(f"{'-' * 80}")

    # Recommended Movies section
    log_lines.append("\n[ RECOMMENDED MOVIES ]")
    if recommended_movies:
        for i, movie in enumerate(recommended_movies, 1):
            log_lines.append(f"  {i:02d}. {movie}")
    else:
        log_lines.append("  No movies were recommended in this step.")

    # Simulator Analysis section
    log_lines.append("\n[ SIMULATOR ANALYSIS ]")
    if not analysis_output or not any(analysis_output.values()):
        log_lines.append("  No analysis output was generated.")
    else:
        # Alignments
        alignments = analysis_output.get("alignments")
        if alignments:
            log_lines.append("\n  - Taste Alignment:")
            for align in alignments:
                log_lines.append(f"    - Movie:  {align.get('title', 'N/A')}")
                log_lines.append(f"      Align:  {align.get('align', 'N/A').upper()}")
                log_lines.append(f"      Reason: {align.get('reason', 'N/A')}")

        # Watch Decision
        watch_decision = analysis_output.get("watch_decision")
        if watch_decision:
            log_lines.append("\n  - Watch Decision:")
            log_lines.append(f"    - Movies to Watch: {watch_decision.get('num', 0)}")
            log_lines.append(f"      Selection:       {' | '.join(watch_decision.get('watch_list', []))}")
            log_lines.append(f"      Reason:          {watch_decision.get('reason', 'N/A')}")

        # Ratings
        ratings = analysis_output.get("ratings")
        if ratings:
            log_lines.append("\n  - Simulated Ratings:")
            for rating in ratings:
                log_lines.append(f"    - Movie:   {rating.get('title', 'N/A')}")
                log_lines.append(f"      Rating:  {rating.get('rating', 'N/A')}/5")
                log_lines.append(f"      Feeling: \"{rating.get('feeling', 'N/A')}\"")

    # Footer
    log_lines.append(f"\n{'=' * 80}\n")
    log_entry_string = "\n".join(log_lines)

    # Append the formatted content to the log file
    with open(log_filepath, 'a', encoding='utf-8') as f:
        f.write(log_entry_string)


def select_diverse_movies(movie_df: pd.DataFrame, num_movies: int, log_file_path: Optional[str]) -> pd.DataFrame:
    """Selects a genre-diverse sample of movies from a user's history."""
    if len(movie_df) <= num_movies:
        return movie_df

    movie_df['primary_genre'] = movie_df['genres'].apply(lambda x: str(x).split('|')[0])

    try:
        stratified_sample = movie_df.groupby('primary_genre').apply(lambda x: x.sample(1)).reset_index(drop=True)

        remaining_needed = num_movies - len(stratified_sample)
        if remaining_needed > 0:
            remaining_indices = movie_df.index.difference(stratified_sample.index)
            remaining_df = movie_df.loc[remaining_indices]
            random_sample = remaining_df.sample(n=remaining_needed, random_state=42)
            final_sample = pd.concat([stratified_sample, random_sample])
        else:
            final_sample = stratified_sample.sample(n=num_movies, random_state=42)
    except:
        log("Warning: Stratified sampling failed. Falling back to random sampling.", color="yellow",
            file_path=log_file_path)
        final_sample = movie_df.sample(n=num_movies, random_state=42)

    return final_sample.drop(columns=['primary_genre'], errors='ignore')


@retry(
    stop=stop_after_attempt(3),
    wait=wait_fixed(60),  # Wait 1 minute between retries
    reraise=True  # Re-raise the exception if all retries fail
)
def get_recommendations(user_history_str: str, prompt_template: str, movies_per_page: int, log_file_path: Optional[str],
                        client,
                        recommender_llm: str = RECOMMENDER_LLM_NAME) -> List[str]:
    """Calls the Recommender LLM to get a page of movie recommendations."""
    log(f"\n--> Calling Recommender LLM ({recommender_llm}) to get {movies_per_page} movies...", color="blue",
        file_path=log_file_path)

    prompt = prompt_template.format(
        user_history=user_history_str,
        movies_per_page=movies_per_page
    )

    messages = [{"role": "system", "content": prompt}]
    log(prompt, color="cyan", file_path=log_file_path)

    try:
        response = client.chat.completions.create(model=recommender_llm, messages=messages, temperature=TEMPERATURE_REC)
        content = response.choices[0].message.content
        recommended_titles = [title.strip() for title in content.split('|')]
        log(f"--> Recommender LLM returned:", color="green", file_path=log_file_path)
        for rt in recommended_titles:
            log(rt, color="green", file_path=log_file_path)
        log("", color="green", file_path=log_file_path)
        return recommended_titles[:movies_per_page]
    except Exception as e:
        log(f"ERROR: Recommender LLM failed: {e}. Returning an empty list.", color="red", file_path=log_file_path)
        raise


def run_path_1_baseline(simulator: UserSimulator, test_movies: List[str], log_dir: str, log_file_path: Optional[str],
                        lock: FileLock, client, tmdb_key):
    log("\n" + "#" * 20 + " RUNNING PATH 1: BASELINE " + "#" * 20, color="white", file_path=log_file_path)
    log_filepath = os.path.join(log_dir, "path_1_baseline_log.txt")
    analysis_output = simulator.analyze_recommendation_list(recommendation_list=test_movies, is_test=True, lock=lock,
                                                            client=client, tmdb_key=tmdb_key)
    log_path_step(log_filepath=log_filepath, step_name="Final Test Evaluation", recommended_movies=test_movies,
                  analysis_output=analysis_output)


def run_path_2_standard_reco(simulator: UserSimulator, test_movies: List[str], num_pages: int, movies_per_page: int,
                             log_dir: str, log_file_path: Optional[str], lock: FileLock, client, tmdb_key):
    log("\n" + "#" * 20 + " RUNNING PATH 2: STANDARD RECOMMENDATION " + "#" * 20, color="white",
        file_path=log_file_path)
    log_filepath = os.path.join(log_dir, "path_2_standard_reco_log.txt")

    for i in range(num_pages):
        log(f"\n--- Path 2, Recommendation Page {i + 1}/{num_pages} ---", color="white", file_path=log_file_path)
        user_history_str = simulator._format_user_history()

        reco_page = get_recommendations(user_history_str=user_history_str,
                                        prompt_template=RECOMMENDER_SYSTEM_PROMPT_STANDARD,
                                        movies_per_page=movies_per_page, log_file_path=log_file_path, client=client)

        if not reco_page:
            log("Skipping page due to recommendation failure.", color="yellow", file_path=log_file_path)
            continue

        ensure_movies_in_db(movie_titles=reco_page, log_file_path=log_file_path, lock=lock, tmdb_key=tmdb_key)
        analysis_output = simulator.analyze_recommendation_list(recommendation_list=reco_page, lock=lock, client=client,
                                                                tmdb_key=tmdb_key)
        log_path_step(log_filepath=log_filepath, step_name=f"Recommendation Page {i + 1}", recommended_movies=reco_page,
                      analysis_output=analysis_output)

        log(f"\n--- Path 2: Running evaluation on test movies for page {i} ---", color="white", file_path=log_file_path)
        final_analysis_output = simulator.analyze_recommendation_list(recommendation_list=test_movies, is_test=True,
                                                                      lock=lock, client=client, tmdb_key=tmdb_key)
        log_path_step(log_filepath=log_filepath, step_name=f"Test Evaluation for page {i}",
                      recommended_movies=test_movies, analysis_output=final_analysis_output)


def run_path_3_persuasive_reco(simulator: UserSimulator, test_movies: List[str], num_pages: int, movies_per_page: int,
                               log_dir: str, log_file_path: Optional[str], lock: FileLock, client, tmdb_key):
    log("\n" + "#" * 20 + " RUNNING PATH 3: PERSUASIVE RECOMMENDATION " + "#" * 20, color="white",
        file_path=log_file_path)
    log_filepath = os.path.join(log_dir, "path_3_persuasive_reco_log.txt")

    for i in range(num_pages):
        log(f"\n--- Path 3, Recommendation Page {i + 1}/{num_pages} ---", color="white", file_path=log_file_path)
        user_history_str = simulator._format_user_history()

        reco_page = get_recommendations(user_history_str=user_history_str,
                                        prompt_template=RECOMMENDER_SYSTEM_PROMPT_PERSUASIVE,
                                        movies_per_page=movies_per_page, log_file_path=log_file_path, client=client)

        if not reco_page:
            log("Skipping page due to recommendation failure.", color="yellow", file_path=log_file_path)
            continue

        ensure_movies_in_db(movie_titles=reco_page, log_file_path=log_file_path, lock=lock, tmdb_key=tmdb_key)
        analysis_output = simulator.analyze_recommendation_list(recommendation_list=reco_page, lock=lock, client=client,
                                                                tmdb_key=tmdb_key)
        log_path_step(log_filepath=log_filepath, step_name=f"Recommendation Page {i + 1}", recommended_movies=reco_page,
                      analysis_output=analysis_output)

        log(f"\n--- Path 3: Running evaluation on test movies for page {i} ---", color="white", file_path=log_file_path)
        final_analysis_output = simulator.analyze_recommendation_list(recommendation_list=test_movies, is_test=True,
                                                                      lock=lock, client=client, tmdb_key=tmdb_key)
        log_path_step(log_filepath=log_filepath, step_name=f"Test Evaluation for page {i}",
                      recommended_movies=test_movies,
                      analysis_output=final_analysis_output)


def run_path_4_algorithmic_reco(simulator: UserSimulator, test_movies: List[str], num_pages: int, movies_per_page: int,
                                log_dir: str, log_file_path: Optional[str], lock: FileLock, client, tmdb_key):
    """PATH 4: Simulates interaction with an external algorithmic recommender."""
    log("\n" + "#" * 20 + " RUNNING PATH 4: ALGORITHMIC RECOMMENDATION " + "#" * 20, color="white",
        file_path=log_file_path)
    log_filepath = os.path.join(log_dir, "path_4_algorithmic_reco_log.txt")

    algo_ratings = {movie['title']: movie['rating'] for movie in simulator.original_ratings}
    all_movie_titles = list(MOVIE_DB.keys())

    for i in range(num_pages):
        log(f"\n--- Path 4, Recommendation Page {i + 1}/{num_pages} ---", color="white", file_path=log_file_path)
        seen_movies_in_path = list(algo_ratings.keys())

        reco_page = algorithmic_recomendation(
            history=algo_ratings,
            number_of_recommendations=movies_per_page
        )

        if not reco_page:
            log("Algorithmic recommender returned no movies. Skipping page.", color="yellow", file_path=log_file_path)
            continue

        log(f"--> Algorithmic recommender returned: {reco_page}", color="white", file_path=log_file_path)

        analysis_output = simulator.analyze_recommendation_list(recommendation_list=reco_page, lock=lock, client=client,
                                                                tmdb_key=tmdb_key)
        log_path_step(log_filepath=log_filepath, step_name=f"Recommendation Page {i + 1}", recommended_movies=reco_page,
                      analysis_output=analysis_output)

        new_ratings = analysis_output.get("ratings", [])
        for rating_info in new_ratings:
            algo_ratings[rating_info['title']] = rating_info['rating']
        log(f"--> Updated algo_ratings with {len(new_ratings)} new ratings for next iteration.", color="blue",
            file_path=log_file_path)

        log(f"\n--- Path 4: Running evaluation on test movies for page {i} ---", color="white", file_path=log_file_path)
        final_analysis_output = simulator.analyze_recommendation_list(recommendation_list=test_movies, is_test=True,
                                                                      lock=lock, client=client, tmdb_key=tmdb_key)
        log_path_step(log_filepath=log_filepath, step_name=f"Test Evaluation for page {i}",
                      recommended_movies=test_movies,
                      analysis_output=final_analysis_output)


def run_path_5_random_reco(simulator: UserSimulator, test_movies: List[str], all_movies_df: pd.DataFrame,
                           num_pages: int, movies_per_page: int, log_dir: str, log_file_path: Optional[str],
                           lock: FileLock, client, tmdb_key):
    """PATH 5: Simulates interaction with a purely random recommender based on movies.csv."""
    log("\n" + "#" * 20 + " RUNNING PATH 5: RANDOM RECOMMENDATION " + "#" * 20, color="white", file_path=log_file_path)
    log_filepath = os.path.join(log_dir, "path_5_random_reco_log.txt")

    all_movie_titles = all_movies_df['title'].tolist()

    for i in range(num_pages):
        log(f"\n--- Path 5, Recommendation Page {i + 1}/{num_pages} ---", color="white", file_path=log_file_path)
        original_titles = {m['title'] for m in simulator.original_ratings}
        simulated_titles = {m['title'] for m in simulator.simulated_ratings}
        seen_movies = original_titles.union(simulated_titles)

        unseen_movies = [title for title in all_movie_titles if title not in seen_movies]

        if len(unseen_movies) < movies_per_page:
            log("Not enough unseen movies to recommend. Stopping path early.", color="red", file_path=log_file_path)
            break

        reco_page = random.sample(unseen_movies, movies_per_page)
        log(f"--> Randomly selected: {reco_page}", color="white", file_path=log_file_path)

        analysis_output = simulator.analyze_recommendation_list(recommendation_list=reco_page, lock=lock, client=client,
                                                                tmdb_key=tmdb_key)
        log_path_step(log_filepath=log_filepath, step_name=f"Recommendation Page {i + 1}", recommended_movies=reco_page,
                      analysis_output=analysis_output)

        log(f"\n--- Path 5: Running evaluation on test movies for page {i} ---", color="white", file_path=log_file_path)
        final_analysis_output = simulator.analyze_recommendation_list(recommendation_list=test_movies, is_test=True,
                                                                      lock=lock, client=client, tmdb_key=tmdb_key)
        log_path_step(log_filepath=log_filepath, step_name=f"Test Evaluation for page {i}",
                      recommended_movies=test_movies,
                      analysis_output=final_analysis_output)


def parse_genres(val):
    if pd.isna(val):
        return []
    try:
        # Try JSON first
        return json.loads(val)
    except Exception:
        try:
            # Try Python literal
            return ast.literal_eval(val)
        except Exception:
            # Fallback: split by '|', ',' or return as single string
            return [val]


# --- NEW: Refactored main logic into a robust function ---
def run_simulation_for_user(user_filepath, all_movies_df, lock_path, api_key, tmdb_key):
    """
    Runs the full 5-path simulation for a single user with robust
    error handling and success/failure file signaling.
    """

    CLIENT = OpenAI(
        base_url=BASE_URL,
        api_key=api_key,
    )

    lock = FileLock(lock_path)

    user_id = os.path.basename(user_filepath).split('-user')[1].split('.csv')[0]
    user_output_dir = os.path.join(RESULTS_DIRECTORY, f"user_{user_id}")
    log_file_path = os.path.join(user_output_dir, "all_log.txt")

    # --- Define flag file paths ---
    success_file = os.path.join(user_output_dir, "_SUCCESS.log")
    error_file = os.path.join(user_output_dir, "_error.log")

    # 1. Check for success file first for efficient skipping
    if os.path.exists(success_file):
        log(f"User {user_id} already successfully completed. Skipping.", color="blue", file_path=log_file_path)
        return f"Success: Skipped: User {user_id} (already successful)"

    # Create output directory if it doesn't exist
    os.makedirs(user_output_dir, exist_ok=True)

    # Clean up old error file if it exists, to allow for a fresh run
    if os.path.exists(error_file):
        os.remove(error_file)

    try:
        # --- Start of the main simulation logic ---
        log("\n" + "=" * 80, color="blue", file_path=log_file_path)
        log(f"STARTING SIMULATION FOR USER: {user_id} at {datetime.now().isoformat()}", color="blue",
            file_path=log_file_path)
        log("=" * 80, color="blue", file_path=log_file_path)

        user_df = pd.read_csv(user_filepath)

        user_df["tmdb_genres"] = user_df["tmdb_genres"].apply(parse_genres)

        # Your constants for the simulation
        TEST_NUMBER = 30
        INTRODUCING_NUMBER = 40
        MOVIES_IN_PAGE = 30
        NUM_OF_PAGES = 4

        if len(user_df) < (TEST_NUMBER + INTRODUCING_NUMBER):
            raise ValueError(f"User has fewer than {TEST_NUMBER + INTRODUCING_NUMBER} movies.")

        # --- Data splitting and simulator setup ---
        test_df = select_diverse_movies(movie_df=user_df, num_movies=TEST_NUMBER, log_file_path=log_file_path)
        test_df.to_json(os.path.join(user_output_dir, "_test_movies.json"), orient='records', indent=2)
        test_movies = test_df['title'].tolist()

        remaining_df = user_df.drop(test_df.index)
        introducing_df = select_diverse_movies(movie_df=remaining_df, num_movies=INTRODUCING_NUMBER,
                                               log_file_path=log_file_path)
        introducing_df.to_json(os.path.join(user_output_dir, "_initial_persona_movies.json"), orient='records',
                               indent=2)

        initial_persona_data = {"user_id": f"user_{user_id}",
                                "rated_movies": introducing_df[
                                    ['title', 'rating', 'tmdb_genres', 'genres', 'overview']].to_dict('records')}
        base_simulator = UserSimulator(user_data=initial_persona_data, log_file_path=log_file_path)

        sim_path1 = base_simulator.copy()
        sim_path2 = base_simulator.copy()
        sim_path3 = base_simulator.copy()
        sim_path4 = base_simulator.copy()
        sim_path5 = base_simulator.copy()

        # --- Running all 5 paths ---
        run_path_1_baseline(simulator=sim_path1, test_movies=test_movies, log_dir=user_output_dir,
                            log_file_path=log_file_path, lock=lock, client=CLIENT, tmdb_key=tmdb_key)
        run_path_2_standard_reco(simulator=sim_path2, test_movies=test_movies, num_pages=NUM_OF_PAGES,
                                 movies_per_page=MOVIES_IN_PAGE, log_dir=user_output_dir, log_file_path=log_file_path,
                                 lock=lock, client=CLIENT, tmdb_key=tmdb_key)
        run_path_3_persuasive_reco(simulator=sim_path3, test_movies=test_movies, num_pages=NUM_OF_PAGES,
                                   movies_per_page=MOVIES_IN_PAGE, log_dir=user_output_dir, log_file_path=log_file_path,
                                   lock=lock, client=CLIENT, tmdb_key=tmdb_key)
        run_path_4_algorithmic_reco(simulator=sim_path4, test_movies=test_movies, num_pages=NUM_OF_PAGES,
                                    movies_per_page=MOVIES_IN_PAGE, log_dir=user_output_dir,
                                    log_file_path=log_file_path, lock=lock, client=CLIENT, tmdb_key=tmdb_key)
        run_path_5_random_reco(simulator=sim_path5, test_movies=test_movies, all_movies_df=all_movies_df,
                               num_pages=NUM_OF_PAGES, movies_per_page=MOVIES_IN_PAGE, log_dir=user_output_dir,
                               log_file_path=log_file_path, lock=lock, client=CLIENT, tmdb_key=tmdb_key)

        # --- Save Final Summary ---
        final_summary = {}
        all_sims = [sim_path1, sim_path2, sim_path3, sim_path4, sim_path5]
        for i, sim in enumerate(all_sims, 1):
            final_test_ratings = sim.test_ratings
            final_summary[f"path_{i}"] = final_test_ratings

        with open(os.path.join(user_output_dir, "_final_test_summary.json"), 'w', encoding='utf-8') as f:
            json.dump(final_summary, f, indent=2)

        log(f"\nSimulation for user {user_id} complete. Summary saved.", color="white",
            file_path=log_file_path)
        log("\n" + "=" * 80, color="white", file_path=log_file_path)

    except Exception as e:
        # --- 2. On ANY exception, create the error file ---
        error_message = f"Simulation for user {user_id} failed. at {datetime.now().isoformat()}.\n"
        error_message += f"Error: {str(e)}\n\n"
        error_message += "--- Traceback ---\n"
        error_message += traceback.format_exc()

        with open(error_file, 'w', encoding='utf-8') as f:
            f.write(error_message)

        log(f"FATAL ERROR for user {user_id}. Details logged to _error.log. Halting.\n Time: {datetime.now().isoformat()}",
            color="red", file_path=log_file_path)
        return f"Failed: User {user_id}"

    else:
        # --- 3. If the try block completes, create the success file ---
        with open(success_file, 'w', encoding='utf-8') as f:
            f.write(f"Simulation successfully completed on: {datetime.now().isoformat()}")

        log(f"\nSimulation for user {user_id} complete at {datetime.now().isoformat()}. Success flag created.",
            color="white", file_path=log_file_path)
        return f"Success: User {user_id}"


#
# You can keep the original main block for testing a single run if you want
if __name__ == "__main__":
    # This block will now only be used if you run main.py directly
    # print("This script is intended to be run via parallel_runner.py")
    # You could put a single user test run here for debugging
    # For example:

    DATA_DIRECTORY = "data/users_2/"
    JSON_PATH = "data/tmdb_api_cache.json"
    USER_LIMIT = 1  # Limit the number of users to process

    # --- Create the Lock ---
    # The lock file itself is used to coordinate the processes.
    # lock_path = os.path.join("data/cache.lock")
    with FileLock("data/cache2.lock") as LOCK:
        # print(f"Using lock file at: {lock_path}")
        # --- Setup ---
        # This data is loaded once and passed to the simulation function.
        os.makedirs(RESULTS_DIRECTORY, exist_ok=True)
        all_movies_df = pd.read_csv('data/ml-32m/movies.csv')
        load_movie_database(JSON_PATH, lock=LOCK)

    user_files = glob.glob(os.path.join(DATA_DIRECTORY, 'movies_enriched-user*.csv'))

    successful_runs = 0
    for user_filepath in sorted(user_files):
        # Check if we have reached the desired number of successful simulations.
        if successful_runs >= USER_LIMIT:
            print(f"User limit of {USER_LIMIT} successful simulations reached. Halting.")
            break

        print(f"\n--- Attempting simulation for: {os.path.basename(user_filepath)} ---")

        # The core logic is now encapsulated in this single function call.
        result = run_simulation_for_user(user_filepath, all_movies_df, lock_path="data/cache2.lock", api_key=API_KEY,
                                         tmdb_key=TMDB_API_KEY)

        # Increment the counter only if the simulation was a definitive success.
        if result.startswith("Success"):
            successful_runs += 1

        print(f"--- Result: {result} ---")

    print(f"\nDebug run finished. Total successful simulations: {successful_runs}")
