import os
import json
from dotenv import load_dotenv
from langchain_openrouter import ChatOpenRouter
from langchain.messages import HumanMessage, SystemMessage
from config import MAX_ARTICLES_AFTER_FILTER

load_dotenv()

SYSTEM_PROMPT = """
You are a news filter agent. Your job is to score articles by relevance to the user's topic and keywords.

For each article, assign a relevance score from 0.0 to 1.0.
Remove duplicates — same story from different sources, keep the better one.

Return ONLY a valid JSON array of the top articles sorted by score descending.
Each item must have: title, description, url, source, published_at, score.

Return only the JSON array, no explanation, no markdown, no code blocks.
"""

llm = ChatOpenRouter(model=os.getenv("OPENROUTER_MODEL"))

def filter_articles(articles: list[dict], topic: str, keywords: list[str]) -> list[dict]:
    if not articles:
        return []
    
    keywords_text = ", ".join(keywords) if keywords else topic
    articles_text = json.dumps(articles,indent=2)

    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=f"""
        Topic: {topic}
        Keywords: {keywords_text}
        Articles: {articles_text}
        Score and return the top articles as a JSON array.""")
    ]

    response = llm.invoke(messages)
    content = response.content.strip()

    try:
        start = content.find("[")
        end = content.rfind("]") + 1
        if start != -1 and end != 0:
            scored = json.loads(content[start:end])
            filtered = [a for a in scored if a.get("score", 0) >= 0.4]
            return filtered[:MAX_ARTICLES_AFTER_FILTER]
    except Exception as e:
        print(f"[Filter] Parse error: {e}")

    return articles[:MAX_ARTICLES_AFTER_FILTER]
    