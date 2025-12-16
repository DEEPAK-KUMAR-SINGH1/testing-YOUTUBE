import os
import re
import streamlit as st

from youtube_transcript_api import YouTubeTranscriptApi
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_mistralai import MistralAIEmbeddings, ChatMistralAI
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate

# --------------------------------------------------
# MISTRAL API KEY (Streamlit Cloud way)
# --------------------------------------------------
os.environ["MISTRAL_API_KEY"] = st.secrets["MISTRAL_API_KEY"]

# --------------------------------------------------
# Helper: Extract YouTube Video ID
# --------------------------------------------------
def extract_video_id(url):
    match = re.search(r"(?:v=|\/)([0-9A-Za-z_-]{11}).*", url)
    if match:
        return match.group(1)
    st.error("Invalid YouTube URL.")
    return None

# --------------------------------------------------
# Get Transcript (May fail on Streamlit Cloud)
# --------------------------------------------------
def get_transcript(video_id, language="en"):
    try:
        transcript = YouTubeTranscriptApi.fetch(video_id, languages=[language])
        return " ".join([i.text for i in transcript])
    except Exception as e:
        st.error(
            "❌ Could not fetch transcript. "
            "YouTube blocks cloud IPs. Try another video or run locally."
        )
        return None

# --------------------------------------------------
# LLM (Mistral)
# --------------------------------------------------
llm = ChatMistralAI(
    model="ministral-8b-2512",
    temperature=0.2
)

# --------------------------------------------------
# Translate Transcript
# --------------------------------------------------
def translate_transcript(transcript):
    prompt = ChatPromptTemplate.from_template("""
    You are an expert translator.
    Translate the following transcript into English with full accuracy.

    Transcript:
    {transcript}
    """)
    chain = prompt | llm
    response = chain.invoke({"transcript": transcript})
    return response.content

# --------------------------------------------------
# Extract Important Topics
# --------------------------------------------------
def get_important_topics(transcript):
    prompt = ChatPromptTemplate.from_template("""
    Extract exactly 5 important topics from the transcript.
    Output as a numbered list.

    Transcript:
    {transcript}
    """)
    chain = prompt | llm
    response = chain.invoke({"transcript": transcript})
    return response.content

# --------------------------------------------------
# Generate Notes
# --------------------------------------------------
def generate_notes(transcript):
    prompt = ChatPromptTemplate.from_template("""
    Create clean, structured notes from the transcript.
    Use bullet points and headings.

    Transcript:
    {transcript}
    """)
    chain = prompt | llm
    response = chain.invoke({"transcript": transcript})
    return response.content

# --------------------------------------------------
# Create Chunks
# --------------------------------------------------
def create_chunks(transcript):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=10000,
        chunk_overlap=1000
    )
    return splitter.create_documents([transcript])

# --------------------------------------------------
# Create Vector Store
# --------------------------------------------------
def create_vector_store(docs):
    embeddings = MistralAIEmbeddings(model="mistral-embed")
    return Chroma.from_documents(docs, embeddings)

# --------------------------------------------------
# RAG Answer
# --------------------------------------------------
def rag_answer(question, vectorstore):
    results = vectorstore.similarity_search(question, k=4)
    context_text = "\n".join([r.page_content for r in results])

    prompt = ChatPromptTemplate.from_template("""
    Answer ONLY using the context below.

    Context:
    {context}

    Question:
    {question}
    """)

    chain = prompt | llm
    response = chain.invoke({
        "context": context_text,
        "question": question
    })

    return response.content
