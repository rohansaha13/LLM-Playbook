import streamlit as st

st.set_page_config(
    page_title="LLM_Playbook",
    page_icon="./logo.png",layout = "wide"
)
from streamlit_extras.app_logo import add_logo
add_logo("logo.png", height=60)
image = "./logo.png"
st.image(image, caption=None, width=None, use_column_width=None, clamp=False, channels="RGB", output_format="auto")
st.write("# Welcome to LLM_Playbook! ")

from streamlit_extras.switch_page_button import switch_page
col1, col2 = st.columns(2)
with col1:
    label = "Core Capabilities"
    state = st.button(label, type="primary", use_container_width=True)
    if state:
        switch_page("Text_Summarization")
with col2:
    label2 = "Usecases"
    state = st.button(label2, type="primary", use_container_width=True)
    if state:
        switch_page("Usecases")

st.markdown(
    """
    Navigate to the respective Core Capabilities to access various LLMs that aid you 
    with Machine Learning and Data Science projects.\n
    **👈 select Core Capabilities  from side bar** 
    """)
st.write("## Models Used Info- ")
st.markdown("""
   *-> wip models
    """) 
tab1,tab2,tab3,tab4 = st.tabs(["Text Summarization","Text Translator","NER","Q&A"])
with tab1:
    st.header("Text Summarization")
    st.markdown("""
    - pegasus-xsum
    - bart-large-cnn
    - dailymail-cnn (pegasus)
    - financial-summarization(pegasus)
    - bloom-7b*
    - roberta*
    - dolly-8b* 
    - falcon-7b*   
    """) 
with tab2:
    st.header("Text Translator")
    st.markdown("""
    - bloom-3b
    - t5-large
    - bigbird-pegasus*   
    """) 
with tab3:
    st.header("NER")
    st.markdown("""
    - roberta
    - bloom-7b*   
    """)     
with tab4: 
    st.header("Q&A")       
    st.markdown("""
    - flan-t5
    - falcon-7b_instruct
    - dolly-8b*   
    """)
# st.markdown(
#     """
#     Navigate to the respective Core Capabilities to access various LLMs that aid you 
#     with Machine Learning and Data Science projects. 
#     **👈 select Core Capabilities  from side bar** 
    
#     ### Text Summarization-
#     - falcon-7b
#     - pegasus-xsum
#     - bart-large-cnn
#     - dailymail-cnn (pegasus)
#     - financial-summarization(pegasus)
#     - bloom-7b*
#     - roberta*
#     - dolly-8b*
#     ### Text Translator-
#     - bloom-3b
#     - t5-large
#     - bigbird-pegasus*
#     ### NER-
#     - roberta \n
#     *-> wip models
#     """
# )