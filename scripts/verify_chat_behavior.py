"""
scripts/verify_chat_behavior.py

Verifies the live behavior of the RAG chatbot (mirroring chat.py logic) 
using a series of:
1. Valid Islamic queries
2. Random/unrelated queries
3. Borderline/tricky queries

Limits LLM calls, prints step-by-step activations of the Distance Shield 
and Refusal Shield, and adds deliberate delays between generative calls.
"""

import os
import sys
import time
from dotenv import load_dotenv

# Add project root to sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
os.chdir(PROJECT_ROOT)

load_dotenv()

# Set UTF-8 encoding for standard output to support Arabic/special symbols safely
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

from retrieval.retriever import get_query_embedding, search_database
from retrieval.generator import generate_answer
from chat import sanitize_hadith_text

# Define test queries
TEST_QUERIES = [
    # --- Category 1: Valid Islamic Queries (Expected In-Scope & LLM Grounded) ---
    {"query": "how many times a day should a Muslim pray?", "category": "Valid Islamic"},
    {"query": "what is the reward of praying in congregation?", "category": "Valid Islamic"},
    {"query": "what did the Prophet say about treating neighbors kindly?", "category": "Valid Islamic"},
    {"query": "is fasting on Eid day allowed in Islam?", "category": "Valid Islamic"},
    {"query": "can a person pray without performing Wudu first?", "category": "Valid Islamic"},

    # --- Category 2: Random Non-Islamic Queries (Expected Retriever Shield Block, 0 LLM Calls) ---
    {"query": "what is the capital of France?", "category": "Random Non-Islamic"},
    {"query": "how do I write a quicksort algorithm in python?", "category": "Random Non-Islamic"},
    {"query": "who won the football world cup in 2022?", "category": "Random Non-Islamic"},
    {"query": "what is the distance from the earth to the sun?", "category": "Random Non-Islamic"},

    # --- Category 3: Borderline/Tricky Queries (Expected Shield/Refusal Coordination) ---
    {"query": "what is the meaning of life?", "category": "Borderline Existential"},
    {"query": "how do I cook chicken biryani?", "category": "Borderline Out-of-Scope"},
    {"query": "tell me about Jesus in Islam", "category": "Borderline Islamic Context"},
    {"query": "what is the ruling on cryptocurrency in Islam?", "category": "Borderline Modern Islamic"}
]

def run_chat_simulation():
    print("=" * 80)
    print("🕌 SIMULATING CHATBOT CLI BEHAVIOR & SHIELD ACTIVATIONS 🕌")
    print("=" * 80)
    
    llm_calls_count = 0
    max_llm_calls = 15
    
    for idx, test_case in enumerate(TEST_QUERIES):
        query = test_case["query"]
        category = test_case["category"]
        
        print(f"\n[{idx + 1}/{len(TEST_QUERIES)}] Query: \"{query}\"")
        print(f"    Category: {category}")
        
        # 1. Embed query
        t0 = time.time()
        query_vector = get_query_embedding(query)
        embed_duration = time.time() - t0
        
        # 2. Search database
        t0 = time.time()
        retrieved_hadiths = search_database(query_vector, book_filter=None)
        db_duration = time.time() - t0
        
        shield_triggered = False
        llm_called = False
        refusal_shield_triggered = False
        final_answer = ""
        top_distance = 1.0
        
        # chat.py logic simulation
        if not retrieved_hadiths:
            final_answer = "I could not find any relevant data to this, please consult a scholar."
            shield_triggered = True
            print("    [SHIELD] ❌ Layer 1 Retriever Shield Activated: No database matches returned.")
        else:
            top_distance = retrieved_hadiths[0].get("distance", 1.0)
            print(f"    [DB Hits] Found {len(retrieved_hadiths)} hits. Top distance: {top_distance:.4f} (embed time: {embed_duration:.2f}s, db time: {db_duration:.3f}s)")
            
            # Layer 1 Check: Distance-Based Retriever Shield
            if top_distance > 0.42:
                final_answer = "I could not find any relevant data to this, please consult a scholar."
                shield_triggered = True
                print(f"    [SHIELD] ❌ Layer 1 Retriever Shield Activated: Cosine distance {top_distance:.4f} > 0.42")
            else:
                # Deliberate sleep gap to satisfy instructions and prevent OpenRouter rate limits
                if llm_calls_count > 0:
                    print("    [API Delay] Sleeping 3.0 seconds to prevent API rate limiting...")
                    time.sleep(3.0)
                
                # Check maximum LLM calls cap
                if llm_calls_count >= max_llm_calls:
                    print("    [ALERT] Maximum allowed LLM calls reached. Skipping generation step.")
                    continue
                
                print("    [LLM] 📤 Invoking OpenRouter (Gemini 2.5 Flash)...")
                t_llm = time.time()
                llm_calls_count += 1
                llm_called = True
                
                answer = generate_answer(query, retrieved_hadiths)
                llm_duration = time.time() - t_llm
                print(f"    [LLM] Received response in {llm_duration:.2f}s")
                
                # Layer 2 Check: LLM Sentinel Refusal Guard
                if "[REFUSAL_SHIELD]" in answer:
                    final_answer = "I could not find any relevant data to this, please consult a scholar."
                    refusal_shield_triggered = True
                    print(f"    [SHIELD] ❌ Layer 2 Sentinel Shield Activated: LLM returned [REFUSAL_SHIELD] (Raw: \"{answer.strip()}\")")
                else:
                    final_answer = answer
                    print("    [SUCCESS] Grounded synthesis completed.")
                    
        # Output final synthesized answer
        print("-" * 80)
        print("🕌 CHATBOT RESPONSE:")
        print(final_answer.strip())
        print("-" * 80)
        
    print(f"\n================================================================================")
    print(f"✓ Simulation Completed successfully!")
    print(f"✓ Total Live LLM Calls Made: {llm_calls_count} / {max_llm_calls}")
    print(f"================================================================================")

if __name__ == "__main__":
    run_chat_simulation()
