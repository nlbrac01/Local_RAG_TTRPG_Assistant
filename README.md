# Local_RAG_TTRPG_Assistant

Feel free to use this code if you're looking to use a local LLM and RAG pipeline to answer questions directly from user defined source materials while checking document relevance and controlling for hallucination. Initially built for use with the Creative Commoins Licensed "Basic Fantasy RPG" which can be downloaded in pdf format for free @ https://www.basicfantasy.org/

Overview
---------------------------------------------------------------------
This notebook demonstrates the implementation of a Retrieval-Augmented Generation (RAG) pipeline using the LangChain framework. The pipeline processes a local PDF document, retrieves relevant chunks based on a user query, and generates answers using a local LLM model. The pipeline includes several stages, including document retrieval, relevance grading, generation, and hallucination detection.

Necessary Python packages: 
----------------------------------------------------------------------
langchain_community, langchain, langchain_core, PyMuPDF, dotenv
You can install the required packages using pip:
pip install langchain_community langchain langchain_core PyMuPDF python-dotenv

Setup Environment Variables
------------------------------------------------------------------------
Create a .env file in the root directory of your project and add the necessary environment variables. For this notebook, no specific environment variables are required, but ensure the setup can load environment variables if needed in the future.

Notebook Sections
----------------------------------------------------------------------
Imports and Setup:
-----------------------------------------------------------------------
import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain_community.embeddings import GPT4AllEmbeddings
from langchain_community.chat_models import ChatOllama
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from langchain_core.prompts import PromptTemplate
from langgraph.graph import END, StateGraph
from typing_extensions import TypedDict
from typing import List
from pprint import pprint


Load environment variables
-----------------------------------------------------------------
load_dotenv()
Define Constants and Load Document:

Define constants
-----------------------------------------------------------------
LOCAL_LLM = "phi3:3.8b"
PDF_PATH = 'Basic-Fantasy-RPG-Rules-r139.pdf'
CHUNK_SIZE = 500
CHUNK_OVERLAP = 150
VECTOR_COLLECTION_NAME = "rag-chroma"

Load and process documents
--------------------------------------------------------------------
loader = PyMuPDFLoader(PDF_PATH)
bf_data = loader.load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
bf_split_data = text_splitter.split_documents(bf_data)

Initialize Vectorstore and Retriever:
---------------------------------------------------------------------
vectorstore = Chroma.from_documents(
    documents=bf_split_data,
    collection_name=VECTOR_COLLECTION_NAME,
    embedding=GPT4AllEmbeddings(),
)
retriever = vectorstore.as_retriever()

Define retrieval grader + generation prompt + hallucination grader + answer grader
-----------------------------------------------------------------------
llm = ChatOllama(model=LOCAL_LLM, format="json", temperature=0)
retrieval_grader_prompt = PromptTemplate(
    template="""system You are a grader assessing relevance of a retrieved document to user prompts. 
    If the document contains context related to the user prompt, grade it as relevant. 
    It does not need to be a stringent test. 
    Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question. 
    Provide a short reason for your decision. 
    user Question: {question}
    Document: {document}
    assistant""",
    input_variables=["question", "document"],
)

Define the workflow
--------------------------------------------------------------------------
class GraphState(TypedDict):
    question: str
    generation: str
    documents: List[str]

Define nodes and their functions (refer to the notebook for complete implementations)
------------------------------------------------------------------------------

Compile and Test the Workflow:
--------------------------------------------------------------------------------------
app = workflow.compile()
inputs = {"question": "What are the stats of a Goblin?"}
for output in app.stream(inputs):
    for key, value in output.items():
        pprint(f"Finished running: {key}:")
        pprint(value)

Running the Notebook
-----------------------------------------------------------------------
Load the notebook in Jupyter or any compatible environment.
Execute each cell sequentially to ensure the workflow is compiled and tested properly.
Modify the input question to test different queries against your documents.

Notes:
------------------------------------------------------------------------
Ensure your desired PDF files are available in the same directory as the notebook.
The local LLM model (phi3:3.8b in this case, but would work with others) should be properly configured and accessible.

Adjust the CHUNK_SIZE and CHUNK_OVERLAP parameters based on your document and requirements.

By following the above steps, you can successfully run and test a local Retrieval-Augmented Generation pipeline that will pull directly from any sourcebook or pdf materials you have available. Offline and for free. Although the larger the datasource becomes, the more time vectorization will take.
