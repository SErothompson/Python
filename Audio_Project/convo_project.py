import speech_recognition as sr
import pyttsx3

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

    # Set properties for the text-to-speech engine
    engine.setProperty('rate', 150)  # Speed of speech
    engine.setProperty('volume', 0.9)  # Volume level (0.0 to 1.0)

    while True:
        print("Say something...")
        user_input = recognize_speech(recognizer, microphone)
        
        if user_input.lower() in ["exit", "quit", "bye"]:
            speak_text(engine, "Goodbye!")
            break
        
        # Process user input and generate a response
        # For simplicity, let's just echo the input back
        response = f"You said: {user_input}"
        speak_text(engine, response)

if __name__ == "__main__":
    main()