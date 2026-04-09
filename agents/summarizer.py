import os
from dotenv import load_dotenv
from langchain_openrouter import ChatOpenRouter
from langchain.messages import HumanMessage, SystemMessage
import json

load_dotenv()

SYSTEM_PROMPT = """
You are a news digest writer. Your job is to create a clean, readable news digest.

Write in this exact format:

## {topic}

A 2-3 sentence overview of the key developments across all articles.

Then list each article as:
**Article Title** — one sentence summary. (Source: source_name)

Be concise, informative, and neutral in tone.
Do not add any extra commentary or markdown outside this format.
"""

llm = ChatOpenRouter(model=os.getenv("OPENROUTER_MODEL"))

def summarize_articles(articles: list[dict], topic: str) -> str:
    if not articles:
        return f"## {topic}\n\nNo relevant articles found for this topic."

    articles_text = json.dumps(
        [{"title": a.get("title"), "description": a.get("description"), "source": a.get("source")} for a in articles],
        indent=2
    )

    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=f"""
Topic: {topic}

Articles:
{articles_text}

Write the digest now.
""")
    ]

    response = llm.invoke(messages)
    return response.content.strip()
