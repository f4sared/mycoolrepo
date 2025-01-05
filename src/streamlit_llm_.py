import streamlit as st
import numpy as np

with st.chat_message("assistant"):
    st.write("Hello human")
    st.bar_chart(np.random.randn(30, 3))

message = st.chat_message("user")
message.write("Hello I am here")
message.bar_chart(np.random.randn(30, 3))