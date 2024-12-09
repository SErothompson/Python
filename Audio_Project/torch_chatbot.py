import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import speech_recognition as sr
import pyttsx3

# Load the pre-trained model and tokenizer
model_name = "microsoft/DialoGPT-medium"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Initialize the text-to-speech engine
engine = pyttsx3.init()
voices = engine.getProperty('voices')
for voice in voices:
    if 'Zira' in voice.name:
        engine.setProperty('voice', voice.id)
        break
engine.setProperty('rate', 150)
engine.setProperty('volume', 0.9)

# Speech recognition function
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

# Text-to-speech function
def speak_text(text):
    engine.say(text)
    engine.runAndWait()

# Chat with model function
def chat_with_model(user_input, history=None):
    new_user_input_ids = tokenizer.encode(user_input + tokenizer.eos_token, return_tensors='pt')
    bot_input_ids = torch.cat([history, new_user_input_ids], dim=-1) if history is not None else new_user_input_ids
    chat_history_ids = model.generate(bot_input_ids, max_length=1000, pad_token_id=tokenizer.eos_token_id, no_repeat_ngram_size=3)
    response = tokenizer.decode(chat_history_ids[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True)
    return response, chat_history_ids

def main():
    recognizer = sr.Recognizer()
    microphone = sr.Microphone()

    print("Start chatting with the bot (say 'exit' to stop)")
    history = None

    while True:
        user_input = recognize_speech(recognizer, microphone)
        
        if user_input.lower() in ["exit", "quit", "bye"]:
            speak_text("Goodbye!")
            break
        
        response, history = chat_with_model(user_input, history)
        print(f"Bot: {response}")
        speak_text(response)

if __name__ == "__main__":
    main()