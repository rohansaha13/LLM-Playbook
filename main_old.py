import streamlit as st

st.set_page_config(
    page_title="LLM_Playbook",
    page_icon=""
)
image = "./logo.png"
st.image(image, caption=None, width=None, use_column_width=None, clamp=False, channels="RGB", output_format="auto")
st.write("# Welcome to LLM_Playbook! ")

st.sidebar.success("Select a demo above.")

st.markdown(
    """
    Navigate to the respective usecases to access various LLMs that aid you 
    with Machine Learning and Data Science projects. 
    **👈 select usecases from side bar** 
    
    ### Text Summarization-
    - falcon-7b
    - pegasus-xsum
    - bart-large-cnn
    - dailymail-cnn (pegasus)
    - financial-summarization(pegasus)
    - bloom-7b*
    - roberta*
    - dolly-8b*
    ### Text Translator-
    - bloom-3b
    - t5-large
    - bigbird-pegasus*
    ### NER-
    - roberta \n
    *-> wip models
    """
)