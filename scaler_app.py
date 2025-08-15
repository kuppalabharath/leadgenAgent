import os
import re
import streamlit as st
from dotenv import load_dotenv
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.vectorstores import FAISS
from groq import Groq

# App Configuration
st.set_page_config(page_title="LeadGen Agent", page_icon="💬")
load_dotenv()

# Groq API Setup
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
groq_client = Groq(api_key=GROQ_API_KEY)


# Load FAISS Vector Store
embedding_model = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
vector_store = FAISS.load_local(
    "faiss_index", embedding_model, allow_dangerous_deserialization=True
)

# System Prompt for AI Assistant
SYSTEM_PROMPT = """
You are Scaler’s friendly and knowledgeable course assistant.

Rules:
1. If the user asks about Scaler courses, instructors, curriculum, fees, schedules, or related details, use the retrieved knowledge base to answer naturally without mentioning that you used a database or retrieved documents.
2. If the user asks a general question not specific to Scaler (e.g., technology trends, career guidance, study tips, learning roadmaps), answer from your own knowledge.
3. Never use meta phrases like “Based on the information provided” or “According to the database”.
4. Keep all answers clear, concise, and positive, while sounding approachable and professional.
5. If the user asks for a roadmap or plan, give it as a logical step-by-step list that is easy to follow.
6. If you don’t have enough specific information, answer helpfully using general knowledge without refusing.
7. Always maintain a polite and engaging tone, encouraging the user’s learning journey.
"""

# Helper Functions
def ask_groq(prompt: str):
    """Send a prompt to Groq LLaMA model and return AI-generated response."""
    chat_completion = groq_client.chat.completions.create(
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt}
        ],
        model=GROQ_MODEL,
        temperature=0.7,
        max_tokens=512
    )
    return chat_completion.choices[0].message.content


def search_rag(query: str):
    """Search FAISS DB for context relevant to the query."""
    docs = vector_store.similarity_search(query, k=2)
    if not docs:
        return None
    return "\n".join([doc.page_content for doc in docs])


def save_contact_info(text: str):
    """Extract and store email or phone number from AI response."""
    emails = re.findall(r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+", text)
    phones = re.findall(r"\+?\d[\d -]{8,12}\d", text)

    with open("contacts.txt", "a", encoding="utf-8") as f:
        for email in emails:
            f.write(f"Email: {email}\n")
        for phone in phones:
            f.write(f"Phone: {phone}\n")

    return emails, phones


def detect_potential_lead(user_query: str):
    """Check if the query indicates interest in joining a course."""
    lead_keywords = ["price", "fee", "fees", "cost", "join", "enroll", "admission", "register"]
    return any(keyword in user_query.lower() for keyword in lead_keywords)


def handle_query(query: str):
    """Route query through RAG if relevant, else to general AI."""
    context = search_rag(query)
    if context:
        prompt = f"Use the following course information to answer:\n\n{context}\n\nUser question: {query}"
    else:
        prompt = query

    answer = ask_groq(prompt)
    save_contact_info(answer)
    return answer

# Streamlit UI Layout
st.title("LeadGen Agent")

st.markdown(
    """
    <div style="white-space: nowrap; font-size:16px;">
        Ask away, from course fees to career paths, learning roadmaps to pure curiosity. It starts here.
    </div>
    """,
    unsafe_allow_html=True
)

user_query = st.text_input("", placeholder="Go ahead, type it in — I’m here to help.")

if user_query:
    with st.spinner("Thinking..."):
        response = handle_query(user_query)
    st.success(response)

    # Lead capture section
    if detect_potential_lead(user_query):
        st.markdown("It seems you're interested in joining our course!")
        contact_info = st.text_input("Please share your email or phone number so we can send details:")
        if contact_info:
            with open("contacts.txt", "a", encoding="utf-8") as f:
                f.write(f"Lead Contact: {contact_info}\n")
            st.success("Thanks! Let's Scale up together. We'll get in touch with you soon.")