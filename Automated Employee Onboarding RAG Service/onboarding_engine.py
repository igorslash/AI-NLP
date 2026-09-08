import os
from typing import List, Optional
from pydantic import BaseModel, Field
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.core.query_engine import RouterQueryEngine
from llama_index.core.selectors import LLMSingleSelector
from llama_index.core.tools import QueryEngineTool, ToolMetadata
from llama_index.llms.openai import OpenAI
from llama_index.core.program import LLMTextCompletionProgram

 
Settings.llm = OpenAI(model="gpt-4o-mini", temperature=0.1)


class OnboardingStep(BaseModel):
    step_number: int = Field(description="Номер шага")
    action: str = Field(description="Что нужно сделать сотруднику")
    department: str = Field(description="Отдел, отвечающий за шаг (IT, HR, Security)")
    link_or_contact: Optional[str] = Field(None, description="Ссылка или контакт")

class OnboardingPlan(BaseModel):
    employee_role: str = Field(description="Роль сотрудника")
    summary: str = Field(description="Краткое приветствие и суть плана")
    steps: List[OnboardingStep] = Field(description="Пошаговый план действий")


class OnboardingRAG:
    def __init__(self, docs_root: str = "./docs"):
        """
        При старте строим отдельные индексы для каждого отдела.
        В продакшене они загружаются из VectorStore, здесь — из папок.
        """
        self.tools = []
        
        # Проходим по папкам (it, hr, security) и создаём индекс для каждой
        departments = [d for d in os.listdir(docs_root) if os.path.isdir(os.path.join(docs_root, d))]
        
        for dept in departments:
            dept_path = os.path.join(docs_root, dept)
            documents = SimpleDirectoryReader(dept_path).load_data()
            
            if not documents:
                continue
                
            index = VectorStoreIndex.from_documents(documents)
            query_engine = index.as_query_engine(similarity_top_k=3)
            
            # Оборачиваем индекс в Tool для роутера
            tool = QueryEngineTool(
                query_engine=query_engine,
                metadata=ToolMetadata(
                    name=f"{dept}_tool",
                    description=f"База знаний отдела {dept.upper()}. Используй для вопросов про {dept}.",
                ),
            )
            self.tools.append(tool)

        # Роутер для выбора индекса по роли
        self.router_engine = RouterQueryEngine(
            selector=LLMSingleSelector.from_defaults(),
            query_engine_tools=self.tools,
            verbose=True
        )

        #генерация JSON
        self.plan_program = LLMTextCompletionProgram.from_defaults(
            output_cls=OnboardingPlan,
            prompt_template_str="""
            Ты — AI-ассистент по онбордингу. На основе предоставленного контекста 
            составь пошаговый план адаптации для нового сотрудника.
            
            Роль сотрудника: {role}
            Контекст из баз знаний компании: {context}
            
            Верни результат СТРОГО в формате JSON согласно схеме. 
            Если в контексте нет информации для шага, укажи "Уточнить у ментора".
            """,
            verbose=True
        )

    def generate_plan(self, role: str, question: str) -> dict:
        """
        Главный метод: принимает роль и вопрос, возвращает структурированный план.
        """
        # Роутер  находит нужные документы через нужный индекс
        raw_response = self.router_engine.query(
            f"Собери информацию для нового сотрудника на позиции '{role}'. Запрос: {question}"
        )
        
        #Передаём найденный контекст в программу для генерации JSON-плана
        try:
            plan = self.plan_program(
                role=role,
                context=str(raw_response)
            )
            return plan.model_dump()
        except Exception as e:
            # Fallback, если LLM сломала формат JSON
            return {
                "employee_role": role,
                "summary": "Ошибка генерации структурированного плана.",
                "steps": [{"step_number": 1, "action": str(raw_response), "department": "N/A", "link_or_contact": None}],
                "error": str(e)
            }