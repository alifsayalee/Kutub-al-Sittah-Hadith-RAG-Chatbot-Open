from retrieval.retriever import get_query_embedding, search_database
from retrieval.generator import generate_answer

def test():
    question = "What is the reward for fasting during the month of Ramadan?"
    print(f"Testing RAG Pipeline...")
    print(f"Question: {question}\n")
    print(f"Connecting to Hugging Face API & ChromaDB...")
    
    try:
        # Step 3a: Retrieve
        vector = get_query_embedding(question)
        sources = search_database(vector)
        
        # Step 3b: Generate
        answer = generate_answer(question, sources)
        
        print("\n" + "="*50)
        print("GEMINI'S ANSWER:")
        print("="*50)
        print(answer)
        print("\n" + "="*50)
        print("SOURCES USED:")
        print("="*50)
        for s in sources:
            print(f"- {s['book']}, Vol {s['volume']}, Hadith {s['hadith_number']}")
            
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    test()
