# import streamlit as st
# from langchain.docstore.document import Document
# from langchain.text_splitter import CharacterTextSplitter
# from langchain.chains.summarize import load_summarize_chain
# from langchain.chains.mapreduce import MapReduceChain
# from langchain import HuggingFacePipeline
# from langchain.llms import HuggingFaceHub
# from langchain.prompts import PromptTemplate
# from langchain import LLMChain
# import torch
# import textwrap
# from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM, AutoModelForSeq2SeqLM
# import datetime



# huggingfacehub_api_token="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"


# def generate_response_falcon(txt):
#     repo_id = "tiiuae/falcon-7b"

#     falcon_llm = HuggingFaceHub(huggingfacehub_api_token=huggingfacehub_api_token,
#                      repo_id=repo_id, task='text-generation',
#                      model_kwargs={"temperature":0.0001, "max_new_tokens":1000, "do_sample":False, "num_return_sequences":1, "repetition_penalty":100})
#     # print(txt)
#     sum_template = """Give detailed answer to the following question.:{text}
#     Answer: """
#     # print("\n\n",sum_template)
    
#     sum_prompt = PromptTemplate(template=sum_template, input_variables=["text"])
    
#     sum_llm_chain = LLMChain(prompt=sum_prompt, llm=falcon_llm)    
#     summary = sum_llm_chain.run(txt)
#     print(summary)
#     wrapped_text = textwrap.fill(summary, width=100, break_long_words=False, replace_whitespace=False)
#     # print(wrapped_text)
#     return str(wrapped_text)
# # Page title
# st.set_page_config(page_title='Text Summarization App',page_icon="./logo.png",layout ="wide")
# image = "./logo.png"
# st.image(image, caption=None, width=None, use_column_width=None, clamp=False, channels="RGB", output_format="auto")
# st.title('Text Summarization App')
# from streamlit_extras.app_logo import add_logo
# add_logo("logo.png", height=60)
# # Text input
# txt_input = st.text_area('Enter your text', '', height=200)
# # Form to accept user's text input for summarization
# result = []
# c1, c2 = st.columns([3, 1])
# with c1:
#     with st.form('summarize_form', clear_on_submit=True):
#         options = st.selectbox(
#         'Choose a model for summarization:',
#         options=["facebook/bart-large-cnn", "slauw87/bart_summarisation", "google/pegasus-xsum", "tiiuae/falcon-7b"])
#         st.write('You selected:', options)
#         submitted = st.form_submit_button('Submit')
#         response = ''
    
# with c2:
#     c2.subheader("Parameters")
#     option = c2.selectbox(
#     'Want to do sample ?',
#     ('True','False'))

#     st.write('You selected:', option)
#     option1 = c2.selectbox(
#     'Want to apply truncation?',
#     ('True','False'))

#     st.write('You selected:', option1)
#     max_length = c2.slider("What is the max length you want to choose for your document?",100,250,10)
#     st.write('max_length value :', max_length)
#     min_length = c2.slider("What is the min length you want to choose for your document?",5,30,5)
#     st.write('min_length value :', min_length)


#     if submitted:
#         with st.spinner('Calculating...'):
#             st.success(' AI Summarization started', icon="🆗")
#         if options=="human-centered-summarization/financial-summarization-pegasus":            
#             start = datetime.datetime.now()
#             response = response + generate_response_financial_summarization(txt_input)
#             stop = datetime.datetime.now() #not used now but useful
#             elapsed = stop - start
#             st.success(f'Summarization completed in {elapsed}', icon="🆗") 

#         elif options=="google/pegasus-xsum":
#             start = datetime.datetime.now()
#             response = response + generate_response_pegasus_xsum(txt_input)
#             stop = datetime.datetime.now() #not used now but useful
#             elapsed = stop - start
#             st.success(f'Summarization completed in {elapsed}', icon="🆗") 


#         elif options=="google/pegasus-cnn_dailymail":
#             start = datetime.datetime.now()
#             response = response + generate_response_pegasus_cnn(txt_input)
#             stop = datetime.datetime.now() #not used now but useful
#             elapsed = stop - start
#             st.success(f'Summarization completed in {elapsed}', icon="🆗")

#         elif options=="tiiuae/falcon-7b":
#             start = datetime.datetime.now()
#             response = response + generate_response_falcon(txt_input)
#             stop = datetime.datetime.now() #not used now but useful
#             elapsed = stop - start
#             st.success(f'Summarization completed in {elapsed}', icon="🆗")

#         elif options=="slauw87/bart_summarisation":
#             start = datetime.datetime.now()
#             response = response + generate_response_bart_summarisation(txt_input)
#             stop = datetime.datetime.now() #not used now but useful
#             elapsed = stop - start
#             st.success(f'Summarization completed in {elapsed}', icon="🆗")
            
#         else:
#             start = datetime.datetime.now()
#             response = response + generate_response_bart_large(txt_input)
#             stop = datetime.datetime.now() #not used now but useful
#             elapsed = stop - start
#             st.success(f'Summarization completed in {elapsed}', icon="🆗")
        
# st.info(response)
# response = ''
# import streamlit as st
# from streamlit_chat import message

# message("My message") 
# message("Hello bot!", is_user=True)  # align's the message to the right

import streamlit as st
from streamlit_chat import message
from streamlit_extras.colored_header import colored_header
from streamlit_extras.add_vertical_space import add_vertical_space
from hugchat import hugchat

st.set_page_config(page_title="HugChat - An LLM-powered Streamlit app")

# Sidebar contents
with st.sidebar:
    st.title('🤗💬 HugChat App')
    #st.markdown('''
    # ## About
    # This app is an LLM-powered chatbot built using:
    # - [Streamlit](https://streamlit.io/)
    # - [HugChat](https://github.com/Soulter/hugging-chat-api)
    # - [OpenAssistant/oasst-sft-6-llama-30b-xor](https://huggingface.co/OpenAssistant/oasst-sft-6-llama-30b-xor) LLM model
    
    # 💡 Note: No API key required!
    # ''')
    # add_vertical_space(5)
    # #st.write('Made with ❤️ by [Data Professor](https://youtube.com/dataprofessor)')

# Generate empty lists for generated and past.
## generated stores AI generated responses
if 'generated' not in st.session_state:
    st.session_state['generated'] = ["I'm HugChat, How may I help you?"]
## past stores User's questions
if 'past' not in st.session_state:
    st.session_state['past'] = ['Hi!']

# Layout of input/response containers
input_container = st.container()
colored_header(label='', description='', color_name='blue-30')
response_container = st.container()

# User input
## Function for taking user provided prompt as input
def get_text():
    input_text = st.text_input("You: ", "", key="input")
    return input_text
## Applying the user input box
with input_container:
    user_input = get_text()

# Response output
## Function for taking user prompt as input followed by producing AI generated responses
def generate_response(prompt):
    chatbot = hugchat.ChatBot()
    response = chatbot.chat(prompt)
    return response

## Conditional display of AI generated responses as a function of user provided prompts
with response_container:
    if user_input:
        response = generate_response(user_input)
        st.session_state.past.append(user_input)
        st.session_state.generated.append(response)
        
    if st.session_state['generated']:
        for i in range(len(st.session_state['generated'])):
            message(st.session_state['past'][i], is_user=True, key=str(i) + '_user')
            message(st.session_state["generated"][i], key=str(i))