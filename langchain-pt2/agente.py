import os
from typing import List
from langchain_core.tools import Tool
from langchain.agents import create_openai_tools_agent, create_react_agent
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain import hub
from estudante import DadosDeEstudante, PerfilAcademico
from universidade import DadosDeUniversidade, TodasUniversidades
from langsmith import traceable

load_dotenv()


@traceable
class AgenteOpenAIFunctions:
    def __init__(self) -> None:
        llm = ChatOpenAI(model="gpt-4o", api_key=os.getenv("OPENAI_API_KEY"))
        dados_de_estudante = DadosDeEstudante()
        perfil_de_estudante = PerfilAcademico()
        dados_de_univerisade = DadosDeUniversidade()
        todas_universidades = TodasUniversidades()
        self.tools: List[Tool] = [
            # return_direct=True faz a execução do AGENTE parar naquela ferramenta
            # Tool(name = dados_de_estudante.name , func = dados_de_estudante._run , description = dados_de_estudante.description, return_direct=True),
            Tool(name = dados_de_estudante.name , func = dados_de_estudante._run , description = dados_de_estudante.description, return_direct=False),
            Tool(name = perfil_de_estudante.name, func = perfil_de_estudante._run, description = perfil_de_estudante.description),
            Tool(name = dados_de_univerisade.name, func = dados_de_univerisade._run, description = dados_de_univerisade.description),
            Tool(name = todas_universidades.name, func = todas_universidades._run, description = todas_universidades.description),
        ]
        # prompt = hub.pull("hwchase17/openai-functions-agent")
        prompt = hub.pull("hwchase17/react")
        # self.agente = create_openai_tools_agent(llm, self.tools, prompt)
        self.agente = create_react_agent(llm, self.tools, prompt)