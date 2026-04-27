import os,json
from dotenv import load_dotenv
from langchain_openrouter import ChatOpenRouter
from langchain.messages import SystemMessage, HumanMessage
from agents.graph import run_graph

load_dotenv()


TEST_CASES = [
    {
        "topic": "Artificial Intelligence",
        "keywords": ["LLM", "OpenAI"],
    },
    {
        "topic": "Space Exploration",
        "keywords": ["NASA", "SpaceX"],
    },
    {
        "topic": "Stock Market",
        "keywords": ["Tesla", "earnings"],
    },
    {
        "topic": "Cybersecurity",
        "keywords": ["data breach", "hacking"],
    },
]

llm = ChatOpenRouter(model=os.getenv("OPENROUTER_MODEL"))

def evaluate_output(topic: str,digest: str)-> float:
    prompt = f"""
Evaluate this news digest.

Topic: {topic}

Digest:
{digest}

Score from 1 to 10 based on:
- Relevance to topic
- Clarity
- Coverage

Return only a number (e.g., 7.5).

"""
    
    response = llm.invoke([
        SystemMessage(content="You are a strict evaluator."),
        HumanMessage(content=prompt)
    ])

    try:
        score = float(response.content.strip())
        return score
    except:
        return 0.0



def run_tests():
    results = []

    for test in TEST_CASES:
        print(f"\n[TEST] {test['topic']}")
        
        output = run_graph(test["topic"], test["keywords"])
        
        results.append({
            "input": test,
            "output": output
        })

    return results


def evaluate_tests():
    test_runs = run_tests()
    scores = []

    for run in test_runs:
        topic = run["input"]["topic"]
        digest = run["output"].get("digest","")

        score = evaluate_output(topic,digest)

        print(f"[Score] {topic}: {score}")
        scores.append(score)

    avg_score = sum(scores) / len(scores) if scores else 0

    print("\n======================")
    print(f"Average Score: {avg_score:.2f}")
    print("======================")

    return test_runs, scores


def save_results(results,scores):
    output_data = []

    for i, run in enumerate(results):
        output_data.append(
            {
            "topic": run["input"]["topic"],
            "keywords": run["input"]["keywords"],
            "digest": run["output"].get("digest", ""),
            "score": scores[i]
            }
        )

    with open("evaluation_results.json","w") as f:
        json.dump(output_data,f,indent=2)

    print("\n✅ Results saved to evaluation_results.json")


# ----------------------
# 6. MAIN
# ----------------------
if __name__ == "__main__":
    results, scores = evaluate_tests()
    save_results(results, scores)
