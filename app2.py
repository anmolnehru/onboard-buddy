import streamlit as st
from openai import OpenAI
from streamlit_extras.app_logo import add_logo

logo_url = "onboard-buddy-logo.png"
st.sidebar.image(logo_url)
st.sidebar.text("Your helpful navigator at workplace")
# Code 

#Import Dependencies
import streamlit as sl
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings

# to create a new file named vectorstore in your current directory.
def load_knowledgeBase_HR():
        embeddings=OpenAIEmbeddings(api_key=st.secrets["OPENAI_API_KEY"])
        DB_FAISS_PATH = 'embed/HR_Information/'
        db = FAISS.load_local(DB_FAISS_PATH, embeddings)
        return db

def load_knowledgeBase_IT():
        embeddings=OpenAIEmbeddings(api_key=st.secrets["OPENAI_API_KEY"])
        DB_FAISS_PATH = 'embed/IT_Info/'
        db = FAISS.load_local(DB_FAISS_PATH, embeddings)
        return db

def load_knowledgeBase_IR():
        embeddings=OpenAIEmbeddings(api_key=st.secrets["OPENAI_API_KEY"])
        DB_FAISS_PATH = 'embed/Individual_Responsibility/'
        db = FAISS.load_local(DB_FAISS_PATH, embeddings)
        return db

def load_knowledgeBase_bene():
        embeddings=OpenAIEmbeddings(api_key=st.secrets["OPENAI_API_KEY"])
        DB_FAISS_PATH = 'embed/benefits/'
        db = FAISS.load_local(DB_FAISS_PATH, embeddings)
        return db

def load_knowledgeBase_prevproj():
        embeddings=OpenAIEmbeddings(api_key=st.secrets["OPENAI_API_KEY"])
        DB_FAISS_PATH = 'embed/prev-projects/'
        db = FAISS.load_local(DB_FAISS_PATH, embeddings)
        return db

def load_knowledgeBase_companyintro():
        embeddings=OpenAIEmbeddings(api_key=st.secrets["OPENAI_API_KEY"])
        DB_FAISS_PATH = 'embed/company-intro/'
        db = FAISS.load_local(DB_FAISS_PATH, embeddings)
        return db

def load_knowledgeBase_teamdes():
        embeddings=OpenAIEmbeddings(api_key=st.secrets["OPENAI_API_KEY"])
        DB_FAISS_PATH = 'embed/team-description/'
        db = FAISS.load_local(DB_FAISS_PATH, embeddings)
        return db

#Import Dependencies
from langchain.prompts import ChatPromptTemplate
def load_prompt():
        prompt = """ You need to answer the question in the sentence as same as in the  pdf content. . 
        Given below is the context and question of the user.
        context = {context}
        question = {question}
        if the answer is not in the context or related to company , you must answer "Sorry. I don't have the information you are asking for."
        if the question is about Company info or company intro you must answer in three sections: 
          1. Main work of the company. 
          2. Leadership priciples of the company. 
          3. Mission / values of of the company. 

        if the question is about Day1 you must answer in three sections: 
          1. Enrollment in company Benefits. 
          2. IT services and laptop setup. 
          3. HR checklist.

        if the question is about Team Info or team intro you must answer in three sections: 
          1. Previous Projects of the team. 
          2. Stepwise learning path to catch up with the pace of the team. 

         """
        prompt = ChatPromptTemplate.from_template(prompt)
        return prompt

#to load the OPENAI LLM
def load_llm():
        from langchain_openai import ChatOpenAI
        llm = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
        return llm



def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)



# Attach all DBs
knowledgeBase=load_knowledgeBase_HR()
knowledgeBase.merge_from(load_knowledgeBase_IT())
knowledgeBase.merge_from(load_knowledgeBase_IR())
knowledgeBase.merge_from(load_knowledgeBase_bene())
knowledgeBase.merge_from(load_knowledgeBase_prevproj())
knowledgeBase.merge_from(load_knowledgeBase_companyintro())
knowledgeBase.merge_from(load_knowledgeBase_teamdes())

llm=load_llm()
prompt1=load_prompt()

# query=sl.text_input('Enter some text')

#

st.title("onboard-buddy")
#st.image("onboard-buddy.jpg")
# Set OpenAI API key from Streamlit secrets
# client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
client = llm

# Set a default model
if "openai_model" not in st.session_state:
    st.session_state["openai_model"] = "gpt-3.5-turbo"

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

st.divider()  # 👈 Draws a horizontal rule
runshortcut = false;
def promptDay1():
    prompt="Give me day 1 checklist"
    runshortcut = true;
    runchat(prompt)
    
    
st.button("Day 1 checklist", on_click=promptDay1)

st.divider()  # 👈 Another horizontal rule


st.caption("Team info.")

st.divider()  # 👈 Another horizontal rule




st.caption("Benefits.")
st.divider()  # 👈 Another horizontal rule

def runchat(prompt):
    # Accept user input
    # Add user message to chat history
    __main__(prompt)

def __main__(prompt):
# Accept user input
    if  runshortcut == False:
        prompt = st.chat_input("What is up?")
        # Add user message to chat history
        similar_embeddings=knowledgeBase.similarity_search(prompt)
        similar_embeddings=FAISS.from_documents(documents=similar_embeddings, embedding=OpenAIEmbeddings(api_key=st.secrets["OPENAI_API_KEY"]))
        
        #creating the chain for integrating llm,prompt,stroutputparser
        retriever = similar_embeddings.as_retriever()
        ans = retriever.get_relevant_documents(prompt)
        prompt1 = (' ').join([doc.page_content for doc in ans])

        st.session_state.messages.append({"role": "user", "content": prompt})
        # Display user message in chat message container
        with st.chat_message("user"):
            st.markdown(prompt)
            
        with st.chat_message("assistant"):
            stream = client.chat.completions.create(
                model=st.session_state["openai_model"],
                messages=[
                    {"role": m["role"], "content": m["content"] + prompt1}
                    for m in st.session_state.messages
                ],
                stream=True,
            )

            #getting only the chunks that are similar to the query for llm to produce the output
            response = st.write_stream(stream)
            st.session_state.messages.append({"role": "assistant", "content": response})
            

            # rag_chain = (
            #         {"context": retriever | format_docs, "question": RunnablePassthrough()}
            #         | prompt1
            #         | llm
            #         | StrOutputParser()
            #     )
            
        
            # response=rag_chain.invoke(prompt)
            # # stream.append(response)
            # response = st.write(response)
    else:
        prompt="Day 1"
        similar_embeddings=knowledgeBase.similarity_search(prompt)
        similar_embeddings=FAISS.from_documents(documents=similar_embeddings, embedding=OpenAIEmbeddings(api_key=st.secrets["OPENAI_API_KEY"]))
        
        #creating the chain for integrating llm,prompt,stroutputparser
        retriever = similar_embeddings.as_retriever()
        ans = retriever.get_relevant_documents(prompt)
        prompt1 = (' ').join([doc.page_content for doc in ans])

        st.session_state.messages.append({"role": "user", "content": prompt})
        # Display user message in chat message container
        with st.chat_message("user"):
            st.markdown(prompt)
            
        with st.chat_message("assistant"):
            stream = client.chat.completions.create(
                model=st.session_state["openai_model"],
                messages=[
                    {"role": m["role"], "content": m["content"] + prompt1}
                    for m in st.session_state.messages
                ],
                stream=True,
            )

            #getting only the chunks that are similar to the query for llm to produce the output
            response = st.write_stream(stream)
            st.session_state.messages.append({"role": "assistant", "content": response})

