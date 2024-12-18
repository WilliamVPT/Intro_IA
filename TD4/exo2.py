import random
from nltk.chat.util import Chat, reflections
import kagglehub

# Dataset information
print("Downloading IMDB movies dataset for potential integration...")
path = kagglehub.dataset_download("rajugc/imdb-movies-dataset-based-on-genre")
print("Path to dataset files:", path)

# Movie database
films = {
    "The Shawshank Redemption": ["drama", "crime"],
    "The Godfather": ["drama", "crime"],
    "The Dark Knight": ["action", "crime"],
    "Pulp Fiction": ["drama", "crime"],
    "Forrest Gump": ["drama", "comedy"],
    "Inception": ["action", "science-fiction"],
    "The Matrix": ["action", "science-fiction"],
    "Interstellar": ["drama", "science-fiction"]
}

# Movie recommendation function
def recommend_movie(genre):
    movies_genre = [movie for movie, genres in films.items() if genre.lower() in genres]
    if movies_genre:
        return random.choice(movies_genre)
    else:
        return "Sorry, I don’t have a recommendation for this genre."

# Chatbot responses
pairs = [
    (r"hello|hi|hey", ["Hello! Welcome to our movie recommendation chatbot.", "Hi there! How can I assist you today?"]),
    (r"what movies do you recommend for (.*)", 
     ["Let me see... I recommend you watch: %1.", "How about trying: %1?"]),
    (r"quit", ["Thank you for using our movie recommendation chatbot. Goodbye!"]),
]

# Define chatbot with custom reflections
chatbot = Chat(pairs, reflections)

# Start chatbot
print("\nWelcome to our movie recommendation chatbot!")
print("You can ask for movie recommendations based on genre.")
print("For example: 'What movies do you recommend for drama?' or 'quit' to exit.")

while True:
    user_input = input("You: ").strip()
    if user_input.lower() == "quit":
        print("Thank you for using our movie recommendation chatbot. Goodbye!")
        break
    
    # Check for genre recommendations
    if "recommend" in user_input.lower():
        # Extract genre
        for genre in ["action", "drama", "comedy", "crime", "science-fiction"]:
            if genre in user_input.lower():
                recommendation = recommend_movie(genre)
                print(f"I recommend you watch: {recommendation}")
                break
        else:
            print("Sorry, I couldn't identify the genre. Please specify one (e.g., action, drama).")
    else:
        response = chatbot.respond(user_input)
        if response:
            print(f"Bot: {response}")
        else:
            print("Bot: I didn’t understand that. Can you rephrase?")
