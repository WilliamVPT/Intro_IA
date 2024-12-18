import nltk
from nltk.chat.util import Chat, reflections

# Load and parse data from File.txt
try:
    with open('File.txt', 'r', encoding='utf-8') as file:
        lines = file.readlines()
except FileNotFoundError:
    print("Error: 'File.txt' not found. Please make sure the file exists in the current directory.")
    exit()

# Parse the lines into question-response pairs
pairs = []
buffer = ""
question = None

for line in lines:
    stripped_line = line.strip()
    # Detect a new question based on the colon pattern
    if stripped_line and ":" in stripped_line:
        if question and buffer:  # Save the previous question-response pair
            pairs.append((question.strip(), [buffer.strip()]))
        question, buffer = stripped_line.split(":", 1)
    elif question:  # Append additional lines to the current response
        buffer += " " + stripped_line

# Append the last pair if exists
if question and buffer:
    pairs.append((question.strip(), [buffer.strip()]))

# Check if pairs were successfully parsed
if not pairs:
    print("Error: No valid pairs found in 'File.txt'. Please ensure the file has the correct format.")
    exit()

# Initialize the chatbot
chatbot = Chat(pairs, reflections)

# Display the pairs for verification
print("Chatbot trained with the following pairs:")
for pair in pairs:
    print(f"Pattern: {pair[0]}, Response: {pair[1][0]}")

# Start the chatbot interaction
print("\nYou can start chatting with the bot now. Type 'quit' to stop.")
while True:
    try:
        user_input = input("You: ").strip()
        if user_input.lower() == 'quit':
            print("Goodbye!")
            break
        response = chatbot.respond(user_input)
        # Handle cases where the bot has no response
        if response:
            print(f"Bot: {response}")
        else:
            print("Bot: I didn't understand that. Can you rephrase?")
    except KeyboardInterrupt:
        print("\nGoodbye!")
        break
