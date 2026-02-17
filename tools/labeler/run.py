import sys
from pathlib import Path

if __name__ == "__main__":
    print("This script is deprecated. Please use Docker Compose to run the labeler application.")
    print(f"Navigate to '{Path(__file__).parent}' and run 'docker compose up -d'.")
    sys.exit(1)
