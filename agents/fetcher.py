from langchain_openrouter import ChatOpenRouter
from langchain.agents import create_agent
from tools.newsapi_tool import fetch_from_newsapi
from tools.google_news_tool import fetch_from_google_news
from langchain.messages import SystemMessage, HumanMessage, AIMessage
import os, ast
from dotenv import load_dotenv
load_dotenv()  # Load environment variables from .env file

prompt = """
You are a news fetcher agent. Your job is to collect news articles for a given topic.
Use BOTH tools — fetch from NewsAPI first, then from Google News.
After fetching from both, return ALL articles as a single combined Python list of dicts.
Each dict must have: title, description, url, source, published_at.
"""

llm = ChatOpenRouter(
    model = os.getenv("OPENROUTER_MODEL"),)

tools = [fetch_from_newsapi, fetch_from_google_news]

agent = create_agent(
    model = llm,
    tools = tools,
    system_prompt = prompt
)

def fetch_articles(topic: str) -> list[dict]:
    result = agent.invoke({"message":[{"role":"user","content":f"Fetch news articles for this topic: {topic}"}]})

    output = result["messages"][-1].content

    return output,topic

    '''

    try:
        start = output.find("[")
        end = output.rfind("]") + 1
        if start != -1 and end != 0:
            return ast.literal_eval(output[start:end])
    except Exception as e:
        print(f"[Fetcher] Parse error: {e}")

    return [] '''

