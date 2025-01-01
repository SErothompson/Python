import random
import streamlit as st

class SimpleBot:
    def __init__(self):
        self.responses = {
            "hello": ["Hi there!", "Hello!", "Hey!", "Greetings!"],
            "how are you": ["I'm doing well, thanks!", "I'm great, how are you?", "All good!"],
            "bye": ["Goodbye!", "See you later!", "Bye bye!"],
            "name": ["My name is SimpleBot.", "I'm SimpleBot!", "You can call me SimpleBot."],
            "help": ["I can help you with basic conversation. Try saying hello!"],
        }
        
    def get_response(self, user_input):
        user_input = user_input.lower()
        
        for key in self.responses:
            if key in user_input:
                return random.choice(self.responses[key])
        
        return "I'm not sure how to respond to that. Try asking something else!"

def create_chatbot_interface():
    st.title("Simple Rule-Based Chatbot")
    
    # Initialize bot and chat history
    if "bot" not in st.session_state:
        st.session_state.bot = SimpleBot()
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    # Accept user input
    user_input = st.chat_input("Type your message here...")
    
    if user_input:
        # Add user message to history
        st.session_state.messages.append({"role": "user", "content": user_input})
        
        # Display user message
        with st.chat_message("user"):
            st.write(user_input)

        # Get bot response
        with st.chat_message("assistant"):
            response = st.session_state.bot.get_response(user_input)
            st.write(response)
            st.session_state.messages.append({"role": "assistant", "content": response})

def main():
    st.set_page_config(
        page_title="Simple Chatbot",
        page_icon="🤖",
        layout="wide"
    )
    
    create_chatbot_interface()

if __name__ == "__main__":
    main()
