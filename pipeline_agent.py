from langchain_classic.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from pdf_summarizer.pdf_summarizer import summarization_tool, mcq_generator_tool, concise_point_generator_tool, model
from resume_analyser.analyzer import extract_resume, resume_analyser_find_gaps

tools = [summarization_tool, mcq_generator_tool, concise_point_generator_tool, extract_resume, resume_analyser_find_gaps]

prompt = ChatPromptTemplate.from_messages([
    ("system", """
You are an AI assistant that can reason and use tools when necessary.

Guidelines:
- Understand the user's intent from their question.
- Decide whether a tool is required to answer.
- If a tool is useful, call it with correct arguments.
- If a tool is not needed, answer directly.
- Do NOT mention tool names in the final answer.
"""),
    ("human", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad"),
])


def create_agent_executor():
    agent = create_tool_calling_agent(
        llm=model,
        tools=tools,
        prompt=prompt
    )
    return AgentExecutor(agent=agent, tools=tools, verbose=True)


def query_agent(user_text: str, user_question: str) -> str:
    """Query the agent with text and question"""
    agent_executor = create_agent_executor()
    response = agent_executor.invoke({
        "input": f"Content: {user_text}\nQuestion: {user_question}",
        "chat_history": []
    })
    return response["output"]

