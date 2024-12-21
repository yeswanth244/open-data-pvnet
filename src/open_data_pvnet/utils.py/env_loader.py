from dotenv import load_dotenv
from pathlib import Path

# Define the project base directory
BASE_DIR = Path(__file__).resolve().parent.parent.parent


def load_environment_variables():
    """Load environment variables from a .env file."""
    env_path = BASE_DIR / ".env"
    if env_path.exists():
        load_dotenv(env_path)
        print(f"Environment variables loaded from {env_path}")
    else:
        raise FileNotFoundError(
            f"{env_path} file not found. Ensure you have a .env file in the project root."
        )
