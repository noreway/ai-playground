#!/usr/bin/python3

import os
import tempfile
import streamlit as st
from streamlit_chat import message
from rag import ChatAssistant, OLLAMA_MODELS
from pprint import pformat

APP_NAME='AI Playground - RAG Chat'

st.set_page_config(page_title=APP_NAME)

def load_assistant():
    st.session_state['assistant'] = ChatAssistant(add_debug_message, st.session_state['selected_model'])

def create_collection():
    collection_name = st.session_state['new_collection'].strip()
    if collection_name and len(collection_name) > 0:
        st.session_state['assistant'].create_collections(collection_name)
        st.session_state['selected_collection'] = collection_name
        st.session_state['new_collection'] = ''

def delete_collection():
    st.session_state['assistant'].delete_collections(st.session_state['selected_collection'])
    st.session_state['selected_collection'] = None

def read_and_save_file():
    collection_name = st.session_state['selected_collection']
    
    if not collection_name:
        st.session_state['messages'].append(('Please, select a collection first.', False))
        return

    for file in st.session_state['file_uploader']:
        with tempfile.NamedTemporaryFile(delete=False) as tf:
            tf.write(file.getbuffer())
            file_path = tf.name

        with st.session_state['ingestion_spinner'], st.spinner(f'Ingesting {file.name}'):
            st.session_state['assistant'].ingest(collection_name, file_path, file.name)
        os.remove(file_path)

def process_input():
    if st.session_state['user_input'] and len(st.session_state['user_input'].strip()) > 0:
        user_text = st.session_state['user_input']

        collection_name = st.session_state['selected_collection']
        if not collection_name:
            st.session_state['messages'].append(('Please, select a collection first.', False))
            return

        with st.session_state['thinking_spinner'], st.spinner(f'Thinking'):
            agent_text = st.session_state['assistant'].ask(collection_name, user_text)

        st.session_state['messages'].append((user_text, True))
        st.session_state['messages'].append((agent_text, False))

def add_debug_message(msg, obj):
    st.session_state['debug'].append((msg, obj))

def display_chat(is_collection_selected):
    st.markdown(f'Selected language model: **{st.session_state["selected_model"]}**')
    st.markdown(f'Selected knowledge base: **{st.session_state["selected_collection"] or "Please select first!"}**')
    for i, (msg, is_user) in enumerate(st.session_state['messages']):
        message(msg, is_user=is_user, key=str(i))
    st.session_state['thinking_spinner'] = st.empty()
    if is_collection_selected:
        st.text_input('Question', key='user_input', on_change=process_input)

def display_sidebar(is_collection_selected):
    st.subheader('Language model')
    st.selectbox('Select base large language model', OLLAMA_MODELS.keys(), key='selected_model', on_change=load_assistant)

    st.subheader('Knowledge base')
    st.selectbox('Select existing knowledge base collection', st.session_state['assistant'].list_collections(), key='selected_collection')
    if is_collection_selected:
        for key, value in st.session_state['assistant'].get_collection_details(st.session_state['selected_collection']).items():
            st.caption(f'{key}: {value}')
        st.button(f'delete collection {st.session_state["selected_collection"]}', on_click=delete_collection)
        st.file_uploader(
            'Add document to knowledge base',
            type=['pdf'],
            key='file_uploader',
            on_change=read_and_save_file,
            accept_multiple_files=True,
        )
    st.text_input('Add new knowledge base collection', key='new_collection', on_change=create_collection)
    
    st.subheader('Debug output')
    for msg, obj in st.session_state['debug']:
        obj_str = pformat(obj).replace('\n', '  \n')
        st.write(f'{msg}:  \n{obj_str}')

def page():
    if len(st.session_state) == 0:
        st.session_state['messages'] = []
        st.session_state['debug'] = []
        st.session_state['selected_model'] = list(OLLAMA_MODELS.keys())[0]
        st.session_state['selected_collection'] = None
        load_assistant()

    is_collection_selected = st.session_state['selected_collection'] and len(st.session_state['selected_collection']) > 0

    st.header(APP_NAME)
    st.session_state['ingestion_spinner'] = st.empty()

    display_chat(is_collection_selected)

    with st.sidebar:
        display_sidebar(is_collection_selected)

if __name__ == '__main__':
    page()
