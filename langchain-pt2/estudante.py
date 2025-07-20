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

def busca_dados_de_estudante(estudante):
    dados = pd.read_csv("documentos/estudantes.csv")
    dados_com_esse_estudante = dados[dados["USUARIO"] == estudante]
    if dados_com_esse_estudante.empty:
        return {}
    return dados_com_esse_estudante.iloc[:1].to_dict()

class ExtratorDeEstudante(BaseModel):
    estudante:str = Field("Nome do estudante informado, sempre em letras minúsculas. Exemplo: joão, carlos, joana, carla")

class DadosDeEstudante(BaseTool):
    name:str = "DadosDeEstudante"
    description:str = """Esta ferramenta extrai o histórico e preferências de um estudante de acordo com seu histórico. Passe para essa ferramenta como argumento o nome do estudante."""

    def _run(self, input) -> str:
        try:
            parser = JsonOutputParser(pydantic_object=ExtratorDeEstudante)

            template = PromptTemplate(
                template="""Você deve analisar a entrada a seguir e extrair o nome informado em minúsculo.
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
            estudante = resposta['estudante']
            estudante = estudante.lower().strip()
            # estudante = input.lower().strip()
            dados = busca_dados_de_estudante(estudante)
            return json.dumps(dados)
        except Exception as e:
            print(f"Error: {e}")
            return "Error processing request"

class Nota(BaseModel):
    area:str = Field("Nome da área de conhecimento")
    nota:float = Field("Nota na área de conhecimento")

class PerfilAcademicoDeEstudante(BaseModel):
    nome: str = Field("nome do estudante")
    ano_de_conclusao:int = Field("ano de conclusão")
    notas:List[Nota] = Field("Lista de notas das disciplinas e áreas de conhecimento.")
    resumo:str = Field("Resumo das principais características desse estudante de forma a torná-lo único e um óitmo potencial estudante para faculdades. Exemplo: só este estudante tem bla bla bla")


class PerfilAcademico(BaseTool):
    name: str = "PerfilAcademico"
    description: str = """Cria um perfil acadêmico de um estudante.
    Esta ferramenta requer como entrada todos os dados do estudante.
    Eu sou incapaz de buscar os dados do estudante.
    Você tem que buscar os dados do estudante antes de me invocar"""

    def _run(self, input:str) -> str:
        llm = ChatOpenAI(model="gpt-4o", api_key=os.getenv("OPENAI_API_KEY"), temperature=0.5)
        parser = JsonOutputParser(pydantic_object=PerfilAcademicoDeEstudante)
        template = PromptTemplate(template="""- Formate o estudante para seu perfil acadêmico
            - Com os dados, identifique as opções de universidades sugeridase  cursos compatíveis com o interesse do aluno.
            - Destaque o perfil do aluno dando enfase principalmente naquilo que faz sentido para as instituições de interesse do aluno.

            Persona: você é uma consultora de carreira e precisa indicar detalhes, riqueza, mas direta ao ponto para o estudante e faculdade as opções e consequências possíveis.
            Informações atuais:

            {dados_do_estudante}
            {formato_de_saida}
            """,
            input_variables=["dados_do_estudante"],
            partial_variables={"formato_de_saida": parser.get_format_instructions()}
        )
        cadeia = template | llm | parser
        return cadeia.invoke({"dados_do_estudante": input})

