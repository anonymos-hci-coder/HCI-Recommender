from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()


BASE_URL = os.getenv("BASE_URL")
API_KEY = os.getenv("API_KEY")
SIMULATOR_LLM_NAME = os.getenv("SIMULATOR_LLM_NAME")
TMDB_API_KEY = os.getenv("TMDB_API_KEY")
RESULTS_DIRECTORY = os.getenv("RESULTS_DIRECTORY")
RECOMMENDER_LLM_NAME = os.getenv("RECOMMENDER_LLM_NAME")
TEMPERATURE_SIM = float(os.getenv("TEMPERATURE_SIM"))
TEMPERATURE_REC = float(os.getenv("TEMPERATURE_REC"))
PROXY_SERVER = os.getenv("PROXY_SERVER")

print(f"BASE_URL: {BASE_URL}")
print(f"API_KEY: {API_KEY}")
print(SIMULATOR_LLM_NAME)

CLIENT = OpenAI(
    base_url=BASE_URL,
    api_key=API_KEY,
)

