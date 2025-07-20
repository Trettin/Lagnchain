import json
import os
from langchain.tools import BaseTool
from langchain.agents import Tool, create_openai_tools_agent, AgentExecutor
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain.globals import set_debug
from langchain import hub
import pandas as pd  # Corrigido: 'import' ao invés de 'from ... import ... as ...'

set_debug(False)

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
            dados = busca_dados_de_estudante(estudante)
            return json.dumps(dados)
        except Exception as e:
            print(f"Error: {e}")
            return "Error processing request"

pergunta = "Quais os dados da Bianca?"

llm = ChatOpenAI(model="gpt-4o", api_key=os.getenv("OPENAI_API_KEY"))

dados_de_estudante = DadosDeEstudante()

tools = [
    Tool(name = dados_de_estudante.name , func = dados_de_estudante._run , description = dados_de_estudante.description)
]

prompt = hub.pull("hwchase17/openai-functions-agent")

agente = create_openai_tools_agent(llm, tools, prompt)

executor = AgentExecutor(agent=agente, tools=tools, verbose=False)

resposta = executor.invoke({"input": pergunta})
print(resposta)


