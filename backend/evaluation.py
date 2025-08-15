# evaluation.py
from get_vector_db import get_vector_db
from query import query
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_recall

# Example evaluation dataset
EVAL_DATA = [
    {
        "query": "What is machine learning?",
        "ground_truth": "Machine learning is a field of computer science that involves the development..."
    },
    {
        "query": "Give examples of ML applications",
        "ground_truth": "Image recognition, NLP, recommendation systems..."
    }
]

def evaluate_retrieval(domain="general"):
    db = get_vector_db(domain)
    results = []
    total_correct = 0
    total_docs = 0

    for item in EVAL_DATA:
        retrieved_docs = db.similarity_search(item["query"], k=3)
        retrieved_texts = [doc.page_content for doc in retrieved_docs]

        # Basic precision/recall check
        if any(gt.lower() in doc.lower() for doc in retrieved_texts for gt in [item["ground_truth"]]):
            total_correct += 1
        total_docs += 1

        results.append({
            "question": item["query"],
            "answer": query(item["query"], domain=domain),
            "contexts": retrieved_texts,
            "ground_truth": item["ground_truth"]
        })

    print(f"Basic Retrieval Accuracy: {total_correct}/{total_docs} = {total_correct/total_docs:.2%}")

    # RAGAS evaluation
    try:
        eval_result = evaluate(
            dataset=results,
            metrics=[faithfulness, answer_relevancy, context_recall]
        )
        print("RAGAS Results:\n", eval_result)
    except Exception as e:
        print("⚠ RAGAS evaluation failed or not installed:", e)

if __name__ == "__main__":
    evaluate_retrieval()
