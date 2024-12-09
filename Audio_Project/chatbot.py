import speech_recognition as sr
import pyttsx3
import nltk
from nltk.chat.util import Chat, reflections

# Define pairs for conversation
pairs = [
    [
        r"my name is (.*)",
        ["Hello %1, how are you today?",]
    ],
    [
        r"hi|hey|hello",
        ["Hello!", "Hey there!",]
    ],
    [
        r"what is your name ?",
        ["I am a chatbot created by Microsoft.",]
    ],
    [
        r"how are you ?",
        ["I'm doing well, thank you!", "I am good! How about you?",]
    ],
    [
        r"sorry (.*)",
        ["It's okay, no worries.", "No problem!",]
    ],
    [
        r"I am fine",
        ["Great to hear that!", "Awesome!",]
    ],
    [
        r"quit",
        ["Bye, take care.", "Goodbye!"]
    ],
]

# Initialize chatbot
chatbot = Chat(pairs, reflections)

def recognize_speech(recognizer, microphone):
    with microphone as source:
        recognizer.adjust_for_ambient_noise(source)
        print("Listening...")
        audio = recognizer.listen(source)

    try:
        response = recognizer.recognize_google(audio)
        print(f"You said: {response}")
        return response
    except sr.UnknownValueError:
        print("Sorry, I did not understand that.")
        return "Sorry, I did not understand that."
    except sr.RequestError:
        print("Sorry, there seems to be an issue with the request.")
        return "Sorry, there seems to be an issue with the request."

def speak_text(engine, text):
    engine.say(text)
    engine.runAndWait()

def main():
    recognizer = sr.Recognizer()
    microphone = sr.Microphone()
    engine = pyttsx3.init()

    # Set the voice to Microsoft Zira (female voice)
    voices = engine.getProperty('voices')
    for voice in voices:
        if 'Zira' in voice.name:
            engine.setProperty('voice', voice.id)
            break

    # Set properties for the text-to-speech engine
    engine.setProperty('rate', 150)  # Speed of speech
    engine.setProperty('volume', 0.9)  # Volume level (0.0 to 1.0)

    print("Starting the chatbot. Say 'quit' to exit.")
    
    while True:
        print("You can speak now...")
        user_input = recognize_speech(recognizer, microphone)
        
        if user_input.lower() in ["exit", "quit", "bye"]:
            speak_text(engine, "Goodbye!")
            break
        
        # Get response from the chatbot
        response = chatbot.respond(user_input)
        
        if response:
            speak_text(engine, response)
        else:
            speak_text(engine, "I didn't get that, could you please repeat?")

if __name__ == "__main__":
    main()
