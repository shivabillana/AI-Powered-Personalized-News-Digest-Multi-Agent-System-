from tools.newsapi_tool import fetch_from_newsapi
from tools.google_news_tool import fetch_from_google_news

def fetch(topic: str) -> list[dict]:
    newsapi_articles = fetch_from_newsapi.invoke({"query": topic})
    google_articles = fetch_from_google_news.invoke({"query": topic})

    return (newsapi_articles or []) + (google_articles or [])