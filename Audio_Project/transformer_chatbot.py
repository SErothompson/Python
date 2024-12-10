import speech_recognition as sr
import pyttsx3
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

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

    # Load the DialoGPT model and tokenizer
    model_name = "microsoft/DialoGPT-medium"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)

    # Initialize the text-generation pipeline
    chatbot = pipeline('text-generation', model=model, tokenizer=tokenizer)

    print("Starting the neural network chatbot. Say 'quit' to exit.")

    while True:
        print("You can speak now...")
        user_input = recognize_speech(recognizer, microphone)
        
        if user_input.lower() in ["exit", "quit", "bye"]:
            speak_text(engine, "Goodbye!")
            break
        
        # Generate a response from the chatbot
        inputs = tokenizer.encode(user_input + tokenizer.eos_token, return_tensors='pt')
        response_ids = model.generate(inputs, max_length=1000, pad_token_id=tokenizer.eos_token_id)
        response = tokenizer.decode(response_ids[:, inputs.shape[-1]:][0], skip_special_tokens=True)
        
        print(f"Bot: {response}")
        speak_text(engine, response)

if __name__ == "__main__":
    main()