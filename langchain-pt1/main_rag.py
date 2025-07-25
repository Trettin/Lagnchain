from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import os

from pydantic import SecretStr


load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

if not api_key:
    raise ValueError("OPENAI_API_KEY não está configurada")

modelo = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0.5,
    api_key=SecretStr(api_key)
)

embeddings = OpenAIEmbeddings()

arquivos = [
    "documentos/GTB_standard_Nov23.pdf",
    "documentos/GTB_platinum_Nov23.pdf",
    "documentos/GTB_gold_Nov23.pdf"
]

documentos = sum([PyPDFLoader(arquivo).load() for arquivo in arquivos], [])

pedacos = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100).split_documents(documentos)

dados_recuperados = FAISS.from_documents(
    pedacos, embeddings
).as_retriever(search_kwargs={"k":4})

prompt_consulta_seguro = ChatPromptTemplate.from_messages(
    [
        ("system", "Responda usando exclusivamente o conteúdo fornecido"),
        ("human", "{query}\n\n{contexto}\n\nResposta:")
    ]
)

cadeia = prompt_consulta_seguro | modelo | StrOutputParser()

def responder(pergunta:str):
    trechos = dados_recuperados.invoke(pergunta)
    contexto = "\n\n".join(um_trecho.page_content for um_trecho in trechos)
    return cadeia.invoke({
        "query": pergunta, "contexto": contexto
    })


print(responder("Quais os benefícios do cartão gold?"))