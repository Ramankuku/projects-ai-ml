from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.tools import tool
from langchain_classic.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate
import pdfplumber

API_KEY=""


embeddings = OpenAIEmbeddings(
    api_key=API_KEY,
    model="text-embedding-3-small"
)

model = ChatOpenAI(
    api_key=API_KEY,
    model="gpt-4o-mini",
    temperature=0.4
)

def load_and_chunk_pdf(pdf_path):
    text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text += page.extract_text() or ""

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100
    )
    return splitter.split_text(text)

@tool
def summarization_tool(pdf_path: str) -> str:
    """Generate a structured summary from a PDF"""
    try:
        chunks = load_and_chunk_pdf(pdf_path)
        db = FAISS.from_texts(chunks, embeddings)
        retriever = db.as_retriever(search_kwargs={"k": 5})

        prompt = PromptTemplate(
            input_variables=["context"],
            template="""
Context:
{context}

Provide a detailed summary including:
- Key Ideas
- Main Focus
- Conclusion (if any)
"""
        )

        chain = RetrievalQA.from_chain_type(
            llm=model,
            retriever=retriever,
            chain_type="stuff",
            chain_type_kwargs={"prompt": prompt}
        )

        return chain.run("")

    except Exception as e:
        return f"Summary Error: {e}"

@tool
def mcq_generator_tool(pdf_path: str) -> str:
    """Generate MCQs from a PDF"""
    try:
        chunks = load_and_chunk_pdf(pdf_path)
        db = FAISS.from_texts(chunks, embeddings)
        retriever = db.as_retriever(search_kwargs={"k": 8})

        prompt = PromptTemplate(
            input_variables=["context"],
            template="""
Context:
{context}

Generate 5 multiple-choice questions.
Each question must include 4 options and the correct answer.
"""
        )

        chain = RetrievalQA.from_chain_type(
            llm=model,
            retriever=retriever,
            chain_type="stuff",
            chain_type_kwargs={"prompt": prompt}
        )

        return chain.run("")

    except Exception as e:
        return f"MCQ Error: {e}"

@tool
def concise_point_generator_tool(pdf_path: str) -> str:
    """Generate concise key points from a PDF"""
    try:
        chunks = load_and_chunk_pdf(pdf_path)
        db = FAISS.from_texts(chunks, embeddings)
        retriever = db.as_retriever(search_kwargs={"k": 5})

        prompt = PromptTemplate(
            input_variables=["context"],
            template="""
Context:
{context}

Generate clear and concise bullet points.
"""
        )

        chain = RetrievalQA.from_chain_type(
            llm=model,
            retriever=retriever,
            chain_type="stuff",
            chain_type_kwargs={"prompt": prompt}
        )

        return chain.run("")

    except Exception as e:
        return f"Key Points Error: {e}"
