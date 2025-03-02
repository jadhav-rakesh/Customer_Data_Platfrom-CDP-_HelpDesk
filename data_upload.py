from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
import os
from dotenv import load_dotenv

load_dotenv()

def upload_html():
    cdp_paths = ["Data/Lytics_Doc", "Data/Segment_Doc", "Data/mParticle_Doc", "Data/Zeotap_Doc"]
    documents = []

    for path in cdp_paths:
        if "Lytics_Doc" in path:
            # Load all files from Lytics directory
            loader = DirectoryLoader(path=path, glob="**/*", show_progress=True)
            loaded_docs = loader.load()
            for doc in loaded_docs:
                if "Lytics_Doc" in doc.metadata['source']:
                    doc.metadata["cdp"] = "Lytics"
                    documents.append(doc)
        elif "Zeotap_Doc" in path:
            pdf_path = "Data/Zeotap_Doc/neotap_data.pdf"
            pdf_loader = PyPDFLoader(file_path=pdf_path)
            loaded_docs = pdf_loader.load()
            for doc in loaded_docs:
                doc.metadata["cdp"] = "Zeotap"
                documents.append(doc)
        else:
            # Load HTML documents for other CDPs
            loader = DirectoryLoader(path=path, glob="**/*.html", show_progress=True)
            documents.extend(loader.load())
            for doc in documents:
                if 'cdp' not in doc.metadata:
                    doc.metadata["cdp"] = path.split("_")[0].split("/")[1]

    print(f"{len(documents)} Pages Loaded")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500, chunk_overlap=50, separators=["\n\n", "\n", " ", ""]
    )

    split_documents = text_splitter.split_documents(documents=documents)
    print(f"Split into {len(split_documents)} Documents")

    embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004", google_api_key=os.getenv("GEMINI_API_KEY"))
    db = FAISS.from_documents(split_documents, embeddings)
    db.save_local("faiss_index")

#rest of the code remains the same

def faiss_query(query, chat_history=[]):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004", google_api_key=os.getenv("GEMINI_API_KEY"))
    new_db = FAISS.load_local("Customer_Data_Platfrom-CDP-_HelpDesk/faiss_index", embeddings, allow_dangerous_deserialization=True)

    model = ChatGoogleGenerativeAI(model="gemini-2.0-pro-exp-02-05", google_api_key=os.getenv("GEMINI_API_KEY"))
    relevance_prompt = f"Is the following question related to Segment, mParticle, Lytics, or Zeotap? Answer 'CDP-related' or 'Irrelevant'.\n\nQuestion: {query}"
    relevance_response = model.invoke(relevance_prompt).content

    if "Irrelevant" in relevance_response:
        return "I can only answer questions related to Segment, mParticle, Lytics, and Zeotap."

    docs = new_db.similarity_search(query)
    context = ""
    for doc in docs:
        context += f"CDP: {doc.metadata['cdp']}\n"
        context += doc.page_content + "\n\n"

    #Prompt engineering to compare CDPs.
    prompt = f"""
    Answer the user's question based on the provided context about the four CDPs: Segment, mParticle, Lytics, and Zeotap.

    Context:
    {context}

    Question:
    {query}

    If the question asks for a comparison, or differences between the cdp's, compare and contrast the different CDP's based on the context.

    Chat History:
    {chat_history}

    Answer:
    """
    response = model.invoke(prompt).content
    chat_history.append(f"User: {query}")
    chat_history.append(f"Chatbot: {response}")
    return response, chat_history

if __name__ == "__main__":
    #upload_html() #Run this only first time or when you change the data.
    chat_history = []
    while True:
        user_query = input("You: ")
        if user_query.lower() == "exit":
            break
        result = faiss_query(user_query, chat_history)
        if isinstance(result, str):
            print("Chatbot:", result)
        else:
            response, chat_history = result
            print("Chatbot:", response)