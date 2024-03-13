#!/usr/bin/python3

import os
import tempfile
import streamlit as st
from streamlit_chat import message
from rag import RagBuilder, RagAssistant
from pprint import pformat

APP_NAME='AI Playground - RAG Chat'

# ui handlers
def __on_nop():
    pass

def __on_debug(msg, obj):
    st.session_state['debug'].append((msg, obj))

def __on_change_llm_server():
    st.session_state['selected_llm'] = None
    st.session_state['assistant'] = None

def __on_change_kb_server():
    st.session_state['selected_kb'] = None
    st.session_state['assistant'] = None

def __on_change_llm_or_kb():
    if st.session_state['selected_llm'] and st.session_state['selected_kb']:
        st.session_state['assistant'] = rb.build_rag_assistant(
            st.session_state['selected_kb_server'], 
            st.session_state['selected_kb'], 
            st.session_state['selected_llm_server'], 
            st.session_state['selected_llm']
        )
    else:
        st.session_state['assistant'] = None

def __on_create_kb():
    kb_name = st.session_state["new_kb_name"].strip()
    if not kb_name:
        st.session_state['messages'].append(('Please, define a knowledge base name first.', False))
        return
    rb.create_knowledge_base(st.session_state['selected_kb_server'], kb_name)
    st.session_state['selected_kb'] = kb_name
    __on_change_llm_or_kb()

def __on_delete_kb():
    rb.delete_knowledge_base(st.session_state['selected_kb_server'], st.session_state['selected_kb'])
    st.session_state['selected_kb'] = None
    __on_change_llm_or_kb()

def __on_add_file_to_kb():
    kb_name = st.session_state['selected_kb']
    if not kb_name:
        st.session_state['messages'].append(('Please, select a knowledge base first.', False))
        return
    for file in st.session_state['file_uploader']:
        with tempfile.NamedTemporaryFile(delete=False) as tf:
            tf.write(file.getbuffer())
            file_path = tf.name
        with st.session_state['ingestion_spinner'], st.spinner(f'Adding file {file.name}'):
            st.session_state['assistant'].add_file(kb_name, file_path, file.name)
        os.remove(file_path)

def __on_add_confluence_space_to_kb():
    kb_name = st.session_state['selected_kb']
    if not kb_name:
        st.session_state['messages'].append(('Please, select a knowledge base first.', False))
        return
    confluence_space = st.session_state["add_confluence"].strip()
    if not confluence_space:
        st.session_state['messages'].append(('Please, define a Confluence space first.', False))
        return
    with st.session_state['ingestion_spinner'], st.spinner(f'Adding space {confluence_space}'):
        st.session_state['assistant'].add_confluence_page(kb_name, confluence_space)

def __on_ask_question():
    if not st.session_state['assistant']:
        st.session_state['messages'].append(('Please, select a language model and knowledge base first.', False))
        return
    user_text = st.session_state['user_input']
    if user_text:
        with st.session_state['thinking_spinner'], st.spinner(f'Thinking'):
            agent_text = st.session_state['assistant'].ask(user_text)
            agent_desc = f'{st.session_state["selected_llm_server"]}/{st.session_state["selected_llm"]} on {st.session_state["selected_kb_server"]}/{st.session_state["selected_kb"]}'
            agent_text = f'_{agent_desc}:_\n\n{agent_text}'
        st.session_state['messages'].append((user_text, True))
        st.session_state['messages'].append((agent_text, False))

# ui renderers
def render_chat():
    for i, (msg, is_user) in enumerate(st.session_state['messages']):
        message(msg, is_user=is_user, key=str(i))
    st.session_state['thinking_spinner'] = st.empty()
    if st.session_state['assistant']:
        agent_desc = f'{st.session_state["selected_llm_server"]}/{st.session_state["selected_llm"]} on {st.session_state["selected_kb_server"]}/{st.session_state["selected_kb"]}'
        st.write(f'<font size="3">Question to {agent_desc}</font>', unsafe_allow_html=True)
        c1, c2 = st.columns([10,1])
        with c1:
            st.text_input('user_input', key='user_input',  label_visibility='collapsed')
        with c2:
            st.button('Ask', on_click=__on_ask_question, disabled=not st.session_state['user_input'], use_container_width=True)


def render_sidebar():
    st.subheader('Language model')
    st.selectbox('selected_llm_server', rb.list_model_servers(), key='selected_llm_server', on_change=__on_change_llm_server, index=None, 
         placeholder='Select language model server', label_visibility='collapsed')
    st.selectbox('selected_llm', rb.list_models(st.session_state['selected_llm_server']), key='selected_llm', on_change=__on_change_llm_or_kb, index=None, 
        placeholder='Select language model', label_visibility='collapsed')

    st.subheader('Knowledge base')
    st.selectbox('selected_kb_server', rb.list_knowledge_base_servers(), key='selected_kb_server', on_change=__on_change_kb_server, index=None, 
        placeholder='Select knowledge base server', label_visibility='collapsed')
    st.selectbox('selected_kb', rb.list_knowledge_bases(st.session_state['selected_kb_server']), key='selected_kb', on_change=__on_change_llm_or_kb, index=None, 
        placeholder='Select knowledge base', label_visibility='collapsed')
    if st.session_state['selected_kb']:
        for key, value in rb.get_knowledge_base_details(st.session_state['selected_kb_server'], st.session_state['selected_kb']).items():
            st.caption(f'  {key}: {value}')

    if st.session_state['assistant']:
        st.write(f'<font size="3">Add file to knowledge base {st.session_state["selected_kb"]}</font>', unsafe_allow_html=True)
        st.file_uploader('file_uploader', type=['pdf', 'xls', 'xlsx', 'csv'], key='file_uploader', on_change=__on_add_file_to_kb,
            accept_multiple_files=True, label_visibility='collapsed')

        st.write(f'<font size="3">Add Confluence space to knowledge base {st.session_state["selected_kb"]}</font>', unsafe_allow_html=True)
        c1, c2 = st.columns([4,1])
        with c1:
            st.text_input('add_confluence', key='add_confluence', on_change=__on_nop, label_visibility='collapsed')
        with c2:
            st.button('Add', on_click=__on_add_confluence_space_to_kb, disabled=not st.session_state["add_confluence"], use_container_width=True)

    if st.session_state['selected_kb_server']:
        st.write(f'<font size="3">Add new knowledge base to {st.session_state["selected_kb_server"]}</font>', unsafe_allow_html=True)
        c1, c2 = st.columns([4,1])
        with c1:
            st.text_input('new_kb_name', key='new_kb_name', on_change=__on_nop, label_visibility='collapsed')
        with c2:
            st.button('Create', on_click=__on_create_kb, disabled=not st.session_state["new_kb_name"], use_container_width=True)            

    if st.session_state['selected_kb']:
        st.write(f'<font size="3">Delete knowledge base from {st.session_state["selected_kb_server"]}</font>', unsafe_allow_html=True)
        st.button(f'delete {st.session_state["selected_kb"]}', on_click=__on_delete_kb, type="primary", use_container_width=True)
    
    st.subheader('Debug output')
    for msg, obj in st.session_state['debug']:
        obj_str = pformat(obj).replace('\n', '  \n')
        st.write(f'{msg}:  \n{obj_str}')

def render_page():
    # init first loop
    if len(st.session_state) == 0:
        st.session_state['messages'] = []
        st.session_state['debug'] = []
        st.session_state['selected_kb_server'] = None
        st.session_state['selected_kb'] = None
        st.session_state['selected_llm_server'] = None
        st.session_state['selected_llm'] = None
        st.session_state['assistant'] = None
    # init loop
    st.session_state['ingestion_spinner'] = st.empty()
    # render
    st.header(APP_NAME)
    render_chat()
    with st.sidebar:
        render_sidebar()

if __name__ == '__main__':
    st.set_page_config(page_title=APP_NAME)
    rb = RagBuilder(__on_debug)
    render_page()
