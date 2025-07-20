import json
import os
from typing import List
from langchain.tools import BaseTool
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from pydantic import BaseModel, Field
import pandas as pd

load_dotenv()

def busca_dados_da_universidade(universidade):
    dados = pd.read_csv("documentos/universidades.csv")
    dados_com_essa_universidade = dados[dados["NOME_FACULDADE"].str.lower() == universidade]
    if dados_com_essa_universidade.empty:
        raise ValueError(f"Universidade '{universidade}' não encontrada nos dados.")
    return dados_com_essa_universidade.iloc[:1].to_dict()


def busca_dados_das_universidades():
    dados = pd.read_csv("documentos/universidades.csv")
    return dados.to_dict()

class ExtratorDeUniversidade(BaseModel):
    universidade:str = Field("Nome da universidade informado, sempre em letras minúsculas")

class DadosDeUniversidade(BaseTool):
    name:str = "DadosDeUniversidade"
    description:str = """Esta ferramenta extrai os dados de uma Universidade. Passe para essa ferramenta como argumento o nome da universidade.
    Passe para essa ferramente como argumento o nome da universidade."""

    def _run(self, input:str) -> str:
        try:
            parser = JsonOutputParser(pydantic_object=ExtratorDeUniversidade)

            template = PromptTemplate(
                template="""Você deve analisar a entrada a seguir e extrair o nome de universidade informado em minúsculo.
                Entrada:
                -----------------
                {input}
                -----------------
                Formato de saída:
                {formato_saida}""",

                input_variables=["input"],
                partial_variables={"formato_saida": parser.get_format_instructions()}
            )

            llm = ChatOpenAI(model="gpt-4o", api_key=os.getenv("OPENAI_API_KEY"), temperature=0.5)

            cadeia = template | llm | parser
            resposta = cadeia.invoke({"input": input})
            print("Resposta CADEIA", resposta)
            universidade = resposta['universidade']
            universidade = universidade.lower().strip()
            # estudante = input.lower().strip()
            dados = busca_dados_da_universidade(universidade)
            return json.dumps(dados)
        except Exception as e:
            print(f"Error: {e}")
            return "Error processing request"


class TodasUniversidades(BaseTool):
    name:str = "TodasUniversidades"
    description:str = """Carrega os dados de todas as universidade. Não é necessário parâmetro de entrada"""

    def _run(self, _:str):
        universidades = busca_dados_das_universidades()
        return json.dumps(universidades, ensure_ascii=False, indent=2)