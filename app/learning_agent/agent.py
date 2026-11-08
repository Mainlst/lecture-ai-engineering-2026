from google.adk.agents import Agent
from google.adk.integrations.openai import OpenAIResponsesLlm

from app import tools

root_agent = Agent(
    name="learning_agent",
    model=OpenAIResponsesLlm(model="gpt-6-luna", reasoning={"effort": "none"}),
    description="AIエンジニアリング講座の学習支援エージェント",
    instruction="""
あなたはAIエンジニアリング講座の学習支援エージェントです。

- 用語について答える前に search_glossary で用語集を確認する。
- 講義の回の内容を聞かれたら get_lecture_summary を使う。
- 確認問題を求められたら make_quiz を使う。
- 用語集にないことは推測で断定せず、その旨を伝える。
- 回答の最後に、関係する講義の回を示す。
- 宿題の答えそのものは書かず、考え方のヒントを示す。
""",
    tools=[
        tools.search_glossary,
        tools.get_lecture_summary,
        tools.make_quiz,
    ],
)
