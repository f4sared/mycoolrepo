import streamlit as st
import numpy as np
import random
import time

st.title("Simple chat")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Streamed response emulator
def response_generator():
    response = random.choice(
        [
            "Hello there! How can I assist you today?",
            "Hi, human! Is there anything I can help you with?",
            "Do you need help?",
        ]
    )
    for word in response.split():
        yield word + " "
        time.sleep(0.05)

# Display assistant response in chat message container
with st.chat_message("assistant"):
    response = st.write_stream(response_generator())
# Add assistant response to chat history
st.session_state.messages.append({"role": "assistant", "content": response})

# # React to user input
# if prompt := st.chat_input("What is up?"):
#     # Display user message in chat message container
#     # without context
#     st.chat_message("user").markdown(prompt)
#     # st.line_chart(np.random.randn(10, 1))
#     # Add user message to chat history
#     st.session_state.messages.append({"role": "user", "content": prompt})

#     response = f"Echo: {prompt}"
#     # Display assistant response in chat message container
#     # with context
#     with st.chat_message("assistant"):
#         st.markdown(response)
#         # st.line_chart(np.random.randn(10, 1))
#     # Add assistant response to chat history
#     st.session_state.messages.append({"role": "assistant", "content": response})