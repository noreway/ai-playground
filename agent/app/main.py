#!/usr/bin/python3

import streamlit as st
from streamlit_chat import message
from ai import IlluminatingAI

APP_NAME='AI Playground - Illuminating AI'

# ui handlers
def __on_ask_question():
    user_text = st.session_state['user_input']
    if user_text:
        with st.session_state['thinking_spinner'], st.spinner(f'Thinking'):
            response = st.session_state['assistant'].ask(user_text)
            agent_text = str(response)
        st.session_state['messages'].append((user_text, True))
        st.session_state['messages'].append((agent_text, False))

# ui renderers
def render_chat():
    for i, (msg, is_user) in enumerate(st.session_state['messages']):
        message(msg, is_user=is_user, key=str(i))
    st.session_state['thinking_spinner'] = st.empty()
    st.write(f'<font size="3">Question</font>', unsafe_allow_html=True)
    c1, c2 = st.columns([10,1])
    with c1:
        st.text_input('user_input', key='user_input',  label_visibility='collapsed')
    with c2:
        st.button('Ask', on_click=__on_ask_question, disabled=not st.session_state['user_input'], use_container_width=True)

def render_sidebar():
    pass

def render_page():
    # init first loop
    if len(st.session_state) == 0:
        st.session_state['messages'] = []
        st.session_state['assistant'] = IlluminatingAI()
    # init loop
    st.session_state['ingestion_spinner'] = st.empty()
    # render
    st.header(APP_NAME)
    render_chat()
    with st.sidebar:
        render_sidebar()

if __name__ == '__main__':
    st.set_page_config(page_title=APP_NAME)
    render_page()
