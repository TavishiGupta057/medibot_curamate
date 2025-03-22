import os
import pandas as pd
import streamlit as st
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEndpoint

DB_FAISS_PATH = "vectorstore/db_faiss"

@st.cache_resource
def get_vectorstore():
    """Load the FAISS vector store."""
    embedding_model = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
    try:
        db = FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)
        return db
    except Exception as e:
        st.error(f"Failed to load vector store: {e}")
        return None  # Return None if loading fails

def load_llm(huggingface_repo_id, HF_TOKEN):
    """Load the Hugging Face LLM."""
    try:
        llm = HuggingFaceEndpoint(
            repo_id=huggingface_repo_id,
            temperature=0.5,
            model_kwargs={"token": HF_TOKEN, "max_length": "512"}
        )
        return llm
    except Exception as e:
        st.error(f"Failed to load LLM: {e}")
        return None  # Return None if loading fails

def set_custom_prompt(custom_prompt_template):
    """Define a custom LLM prompt."""
    prompt = PromptTemplate(template=custom_prompt_template, input_variables=["context", "question"])
    return prompt

def get_specialist_for_symptom(user_symptoms):
    """Get recommended specialists based on symptoms from a CSV file."""
    df = pd.read_csv("symptom_specialists.csv")
    
    # Convert to lowercase for case-insensitive matching
    df['symptom'] = df['symptom'].str.lower()
    user_symptoms = user_symptoms.lower()

    # Find matching specialists
    matched_specialists = df[df["symptom"].apply(lambda x: any(symptom in user_symptoms for symptom in x.split(',')))]

    if not matched_specialists.empty:
        return ", ".join(matched_specialists["specialist"].unique())
    
    return "General Physician"  # Default fallback

def process_user_query(user_query, qa_chain):
    """Process user input and provide relevant answers."""
    
    if not qa_chain:
        return "Error: QA model failed to load."
    
    user_query = user_query.lower()
    
    # Handle "I'm healthy" cases
    healthy_statements = ["i am healthy", "i feel fine", "i have no symptoms", "i am okay", "i don't need a doctor"]
    if any(statement in user_query for statement in healthy_statements):
        return "You are healthy! No need to visit a doctor. 😊"
    
    # Get recommended doctor type
    recommended_specialists = get_specialist_for_symptom(user_query)
    
    # Use LLM for general medical queries
    response = qa_chain.invoke({'query': user_query})
    
    if response and "result" in response:
        return f"{response['result']}\n\n**Recommended Specialist:** {recommended_specialists}"
    else:
        return f"Unable to find an answer. **Recommended Specialist:** {recommended_specialists}"

def main():
    """Streamlit UI for the chatbot."""
    st.title("Ask MediBot! 🤖🩺")

    if 'messages' not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        st.chat_message(message['role']).markdown(message['content'])

    prompt = st.chat_input("Ask me about your symptoms or any medical query!")

    if prompt:
        st.chat_message('user').markdown(prompt)
        st.session_state.messages.append({'role': 'user', 'content': prompt})

        CUSTOM_PROMPT_TEMPLATE = """
                Use the pieces of information provided in the context to answer the user's question.
                If you don't know the answer, just say that you don't know. Don't try to make up an answer.
                Don't provide anything out of the given context.

                Context: {context}
                Question: {question}

                Start the answer directly. No small talk please.
                """
        
        HUGGINGFACE_REPO_ID = "mistralai/Mistral-7B-Instruct-v0.3"
        HF_TOKEN = os.environ.get("HF_TOKEN")

        try:
            vectorstore = get_vectorstore()
            if vectorstore is None:
                st.error("Failed to load the vector store.")
                return  # Stop execution if vectorstore is None

            llm_model = load_llm(huggingface_repo_id=HUGGINGFACE_REPO_ID, HF_TOKEN=HF_TOKEN)
            if llm_model is None:
                st.error("Failed to load LLM.")
                return  # Stop execution if llm_model is None

            qa_chain = RetrievalQA.from_chain_type(
                llm=llm_model,
                chain_type="stuff",
                retriever=vectorstore.as_retriever(search_kwargs={'k': 3}),
                return_source_documents=True,
                chain_type_kwargs={'prompt': set_custom_prompt(CUSTOM_PROMPT_TEMPLATE)}
            )

            response_text = process_user_query(prompt, qa_chain)

            st.chat_message('assistant').markdown(response_text)
            st.session_state.messages.append({'role': 'assistant', 'content': response_text})

        except Exception as e:
            st.error(f"Unexpected Error: {str(e)}")

if __name__ == "__main__":
    main()
