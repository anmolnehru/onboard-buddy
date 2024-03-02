#Import Dependencies
import streamlit as sl
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings

# to create a new file named vectorstore in your current directory.
def load_knowledgeBase():
        embeddings=OpenAIEmbeddings(api_key=st.secrets["OPENAI_API_KEY"])
        DB_FAISS_PATH = '/content/onboard-buddy/embed/HR_Information/'
        db = FAISS.load_local(DB_FAISS_PATH, embeddings)
        return db

#Import Dependencies
from langchain.prompts import ChatPromptTemplate

def load_prompt():
        prompt = """ You need to answer the question in the sentence as same as in the  pdf content. . 
        Given below is the context and question of the user.
        context = {context}
        question = {question}
        if the answer is not in the pdf , answer "Sorry, I don't know the anser to that"
         """
        prompt = ChatPromptTemplate.from_template(prompt)
        return prompt

#to load the OPENAI LLM
def load_llm():
        from langchain_openai import ChatOpenAI
        llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0, api_key=st.secrets["OPENAI_API_KEY"])
        return llm



def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

if __name__=='__main__':
        sl.header("welcome to the pdf bot")
        knowledgeBase=load_knowledgeBase()
        llm=load_llm()
        prompt=load_prompt()
        
        query=sl.text_input('Enter some text')
        
        
        if(query):
                #getting only the chunks that are similar to the query for llm to produce the output
                similar_embeddings=knowledgeBase.similarity_search(query)
                similar_embeddings=FAISS.from_documents(documents=similar_embeddings, embedding=OpenAIEmbeddings(api_key="Enter your API key"))
                
                #creating the chain for integrating llm,prompt,stroutputparser
                retriever = similar_embeddings.as_retriever()
                rag_chain = (
                        {"context": retriever | format_docs, "question": RunnablePassthrough()}
                        | prompt
                        | llm
                        | StrOutputParser()
                    )
                
                response=rag_chain.invoke(query)
                sl.write(response) 