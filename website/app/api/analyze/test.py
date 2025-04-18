# test_rag.py
from rag import get_care_recommendations

if __name__ == "__main__":
    species = "tomato"
    condition = "early blight"

    print(f"Testing care recommendation for {species} with {condition}...\n")
    result = get_care_recommendations(species, condition)
    print("\n--- Recommendation ---\n")
    print(result)