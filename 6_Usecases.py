import streamlit as st

st.set_page_config(
    page_title="LLM_Playbook",
    page_icon="./logo.png"
)
image = "./logo.png"
st.image(image, caption=None, width=None, use_column_width=None, clamp=False, channels="RGB", output_format="auto")
st.write("# LLM_Playbook - Usecases! ")
from streamlit_extras.switch_page_button import switch_page
col1, col2, col3 = st.columns(3)
with col1:
    label = "Conversational BI"
    state = st.button(label, type="primary", use_container_width=True)
    if state:
        switch_page("Text_Summarization")
with col2:
    label2 = "Model Documentation"
    state = st.button(label2, type="primary", use_container_width=True)
    if state:
        switch_page("Usecases")
with col3:
    label2 = "Servicing Letter"
    state = st.button(label2, type="primary", use_container_width=True)
    if state:
        switch_page("Usecases")

st.markdown(
    """
    **☝ select usecases from above** 
    \n Navigate to the respective Usecases to access various LLMs that aid you 
    with Machine Learning and Data Science projects.
    
    """)
st.write("## Usecase Desciptions ")

tab1,tab2,tab3 = st.tabs(["Conversational BI","Model Documentation","Servicing Letter"])
with tab1:
    st.header("Conversational BI")
    st.markdown("""
    ##### Conversational BI is an EXl's intelligent visualization capability that accelerates the data analysis and aids a data scientists to visualize the data in span of seconds
    """) 
with tab2:
    st.header("Model Documentation")
    st.markdown("""
    - bloom-3b
    - t5-large
    - bigbird-pegasus*   
    """) 
with tab3:
    st.header("Servicing Letter")
    st.markdown("""
    - roberta
    - bloom-7b*   
    """)     
