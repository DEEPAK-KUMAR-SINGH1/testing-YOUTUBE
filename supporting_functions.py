import time
import re
import streamlit as st
import os

from youtube_transcript_api import YouTubeTranscriptApi
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_mistralai import MistralAIEmbeddings, ChatMistralAI
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate

# 🔹 Load Mistral API key safely (Cloud + Local)
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY") or st.secrets["MISTRAL_API_KEY"]
os.environ["MISTRAL_API_KEY"] = MISTRAL_API_KEY

# ------------------ Helper Functions ------------------

def extract_video_id(url):
    """Extract YouTube video ID"""
    match = re.search(r"(?:v=|\/)([0-9A-Za-z_-]{11}).*", url)
    if match:
        return match.group(1)
    st.error("Invalid YouTube URL. Please enter a valid video link.")
    return None

def get_transcript(video_id, language="en"):
    """Fetch transcript from YouTube video"""
    ytt_api = YouTubeTranscriptApi()
    try:
        transcript = ytt_api.fetch(video_id, languages=[language])
        full_transcript = " ".join([i.text for i in transcript])
        time.sleep(2)  # Small delay
        return full_transcript
    except Exception as e:
        st.error(f"Error fetching transcript: {e}")
        return None

# ------------------ Initialize LLM ------------------

llm = ChatMistralAI(
    model="ministral-8b-2512",
    temperature=0.2
)

# ------------------ Core Functions ------------------

def translate_transcript(transcript):
    try:
        prompt = ChatPromptTemplate.from_template("""
        You are an expert translator with deep cultural and linguistic knowledge.
        I will provide you with a transcript. Your task is to translate it into English with absolute accuracy, preserving:
        - Full meaning and context (no omissions, no additions).
        - Tone and style (formal/informal, emotional/neutral as in original).
        - Nuances, idioms, and cultural expressions (adapt appropriately while keeping intent).
        - Speaker’s voice (same perspective, no rewriting into third-person).
        Do not summarize or simplify. The translation should read naturally in the target language but stay as close as possible to the original intent.

        Transcript:
        {transcript}
        """)
        chain = prompt | llm
        response = chain.invoke({"transcript": transcript})
        return response.content
    except Exception as e:
        st.error(f"Error translating transcript: {e}")
        return None

def get_important_topics(transcript):
    try:
        prompt = ChatPromptTemplate.from_template("""
        You are an assistant that extracts the 5 most important topics discussed in a video transcript or summary.

        Rules:
        - Summarize into exactly 5 major points.
        - Each point should represent a key topic or concept, not small details.
        - Keep wording concise and focused on the technical content.
        - Do not phrase them as questions or opinions.
        - Output should be a numbered list.
        - Show only points that are discussed in the transcript.
        Here is the transcript:
        {transcript}
        """)
        chain = prompt | llm
        response = chain.invoke({"transcript": transcript})
        return response.content
    except Exception as e:
        st.error(f"Error extracting topics: {e}")
        return None

def generate_notes(transcript):
    try:
        prompt = ChatPromptTemplate.from_template("""
        You are an AI note-taker. Your task is to read the following YouTube video transcript 
        and produce well-structured, concise notes.

        ⚡ Requirements:
        - Present the output as **bulleted points**, grouped into clear sections.
        - Highlight key takeaways, important facts, and examples.
        - Use **short, clear sentences** (no long paragraphs).
        - If the transcript includes multiple themes, organize them under **subheadings**.
        - Do not add information that is not present in the transcript.

        Here is the transcript:
        {transcript}
        """)
        chain = prompt | llm
        response = chain.invoke({"transcript": transcript})
        return response.content
    except Exception as e:
        st.error(f"Error generating notes: {e}")
        return None

def create_chunks(transcript):
    if not transcript:
        st.warning("Transcript is empty. Cannot create chunks.")
        return []
    text_splitters = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    doc = text_splitters.create_documents([transcript])
    return doc

def create_vector_store(docs):
    if not docs:
        st.warning("No documents to create vector store.")
        return None
    embedding = MistralAIEmbeddings(model="mistral-embed")
    vector_store = Chroma.from_documents(docs, embedding)
    return vector_store

def rag_answer(question, vectorstore):
    if not vectorstore:
        st.warning("Vector store is empty. Cannot perform RAG.")
        return "No context available."
    results = vectorstore.similarity_search(question, k=4)
    context_text = "\n".join([i.page_content for i in results])
    prompt = ChatPromptTemplate.from_template("""
    You are a kind, polite, and precise assistant.
    - Begin with a warm and respectful greeting (avoid repeating greetings every turn).
    - Understand the user’s intent even with typos or grammatical mistakes.
    - Answer ONLY using the retrieved context.
    - If answer not in context, say:
      "I couldn’t find that information in the database. Could you please rephrase or ask something else?"
    - Keep answers clear, concise, and friendly.

    Context:
    {context}

    User Question:
    {question}

    Answer:
    """)
    chain = prompt | llm
    response = chain.invoke({"context": context_text, "question": question})
    return response.content
