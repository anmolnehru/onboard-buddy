import streamlit as st
from faiss import FAISS
from openai import OpenAI
from streamlit_extras.app_logo import add_logo

# Set web page title and icon.
st.set_page_config(
    page_title="Your virtual onboarding assistant",
    page_icon=":robot:"
)


st.title("onboard-buddy.jpg")
client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])


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

db = FAISS.load_local("emded/", embeddings)
retriever = db.as_retriever()    

qa_chain = RetrievalQA.from_chain_type(llm=chat, chain_type="stuff", retriever=retriever, verbose=True,chain_type_kwargs=chain_type_kwargs)


# Accept user input
if prompt := st.chat_input("Let's get you started with onboard buddy"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)
        
    with st.chat_message("assistant"):
        stream = client.chat.completions.create(
            model=st.session_state["openai_model"],
            messages=[
                {"role": m["role"], "content": m["content"]}
                for m in st.session_state.messages
            ],
            stream=True,
        )
        response = st.write_stream(stream)
    st.session_state.messages.append({"role": "assistant", "content": response})
