#from agents.fetcher import fetch_articles
from agents.fetcher_copy import fetch as fetch_articles
from agents.filter import filter_articles
from agents.summarizer import summarize_articles


def run_digest(topics: list[str], keywords: list[str]) -> dict:
    """
    Coordinates the full pipeline for each topic:
    1. Fetch articles (Fetcher Agent)
    2. Filter & score relevance (Filter Agent)
    3. Summarize into digest (Summarizer Agent)

    Returns: { topic: { "digest": str, "articles": list } }
    """
    results = {}

    for topic in topics:
        print(f"\n[Orchestrator] Processing: {topic}")

        # Step 1: Fetch
        print(f"  → Fetching articles...")
        raw_articles = fetch_articles(topic)
        print(f"  → Got {len(raw_articles)} articles")

        # Step 2: Filter
        print(f"  → Filtering...")
        filtered = filter_articles(raw_articles, topic, keywords)
        print(f"  → {len(filtered)} articles after filter")

        # Step 3: Summarize
        print(f"  → Summarizing...")
        digest = summarize_articles(filtered, topic)

        results[topic] = {
            "digest": digest,
            "articles": filtered,
        }

    return results