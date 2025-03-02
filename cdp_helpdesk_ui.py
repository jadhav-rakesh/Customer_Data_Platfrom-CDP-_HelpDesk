from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import ConversationalRetrievalChain
from langchain_community.vectorstores import FAISS

import streamlit as st
import os

from dotenv import load_dotenv

load_dotenv()

def query(question, chat_history):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004", google_api_key=os.getenv("GEMINI_API_KEY"))
    new_db = FAISS.load_local("Customer_Data_Platfrom-CDP-_HelpDesk/faiss_index", embeddings, allow_dangerous_deserialization=True)
    llm = ChatGoogleGenerativeAI(model="gemini-2.0-pro-exp-02-05", google_api_key=os.getenv("GEMINI_API_KEY"))

    relevance_prompt = f"Is the following question related to Segment, mParticle, Lytics, or Zeotap? Answer 'CDP-related' or 'Irrelevant'.\n\nQuestion: {question}"
    relevance_response = llm.invoke(relevance_prompt).content

    if "Irrelevant" in relevance_response:
        return {"answer": "I can only answer questions related to Segment, mParticle, Lytics, and Zeotap.", "source_documents": []}

    query_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=new_db.as_retriever(),
        return_source_documents=True
    )

    return query_chain({"question": question, "chat_history": chat_history})

def show_ui():
    st.title("CDP HelpDesk")
    st.image("Images/cdp.png")
    st.subheader("Please enter your Query")

    if "messages" not in st.session_state:
        st.session_state.messages = []
        st.session_state.chat_history = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Enter your CDP related Query: "):
        with st.spinner("Working on your query....."):
            response = query(question=prompt, chat_history=st.session_state.chat_history)
            with st.chat_message("user"):
                st.markdown(prompt)
            with st.chat_message("assistant"):
                st.markdown(response["answer"])
                if response["source_documents"]:
                    with st.expander("Source Documents"):
                        for doc in response["source_documents"]:
                            st.write(doc.page_content)

            st.session_state.messages.append({"role": "user", "content": prompt})
            st.session_state.messages.append({"role": "assistant", "content": response["answer"]})
            st.session_state.chat_history.extend([(prompt, response["answer"])])

if __name__ == "__main__":
    show_ui()