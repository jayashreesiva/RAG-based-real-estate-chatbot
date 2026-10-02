import os
import sys
import time
from rag_chatbot import RealEstateRAG

def run_hallucination_tests():
    """
    Automated test suite verifying Hallucination Prevention and Grounded Generation.
    Validates that:
    1. In-domain queries are answered accurately using knowledge base facts.
    2. Non-existent locations and unrealistic criteria are rejected with clear refusal.
    3. Ungrounded attribute traps (hospitals, crime scores, swimming pool clubs, bank loan rates)
       are strictly refused rather than fabricated.
    """
    print("=" * 75)
    print("  AUTOMATED HALLUCINATION & GROUNDING VERIFICATION SUITE")
    print("=" * 75)
    print("Initializing RealEstateRAG pipeline with FAISS & Gemini 3.6 Flash...\n")

    try:
        bot = RealEstateRAG()
    except Exception as e:
        print(f"Failed to initialize RAG: {e}")
        return False

    test_cases = [
        # -------------------------------------------------------------
        # Category 1: Answerable Knowledge Base Queries (Should Pass)
        # -------------------------------------------------------------
        {
            "category": "In-Domain Answerable",
            "name": "3 BHK in Anna Nagar",
            "query": "Find 3 BHK properties in Anna Nagar",
            "expected_behavior": "Should retrieve real properties in Anna Nagar with prices and BHK.",
            "check": lambda resp: "anna nagar" in resp.lower() and ("3 bhk" in resp.lower() or "3-bhk" in resp.lower() or "lakhs" in resp.lower()) and "sorry" not in resp.lower()
        },
        {
            "category": "In-Domain Answerable",
            "name": "Properties under 50 Lakhs",
            "query": "Show properties under 50 lakhs in Chennai",
            "expected_behavior": "Should retrieve properties priced under 50 lakhs.",
            "check": lambda resp: "lakhs" in resp.lower() and "sorry, no matching" not in resp.lower()
        },
        # -------------------------------------------------------------
        # Category 2: Non-Existent / Out-of-Domain (Must Refuse)
        # -------------------------------------------------------------
        {
            "category": "Non-Existent Criteria",
            "name": "Out-of-Domain City (Coimbatore)",
            "query": "Show me 4 BHK luxury villas in Coimbatore for 20 lakhs",
            "expected_behavior": "Must refuse as Coimbatore is outside the Chennai knowledge base.",
            "check": lambda resp: "sorry" in resp.lower() or "not found" in resp.lower() or "not available" in resp.lower()
        },
        {
            "category": "Non-Existent Criteria",
            "name": "Unrealistic Price (Anna Nagar under 5 Lakhs)",
            "query": "Find properties in Anna Nagar under 5 lakhs",
            "expected_behavior": "Must refuse because no Anna Nagar property exists under 5 lakhs.",
            "check": lambda resp: "sorry" in resp.lower() or "no matching" in resp.lower() or "not found" in resp.lower()
        },
        # -------------------------------------------------------------
        # Category 3: Adversarial / Ungrounded Attribute Traps (Must Refuse)
        # -------------------------------------------------------------
        {
            "category": "Ungrounded Attribute Trap",
            "name": "Hospital Distance Hallucination Trap",
            "query": "Which property in Anna Nagar is nearest to Apollo Hospital and what is its exact walking distance in meters?",
            "expected_behavior": "Must refuse to invent hospital proximity or walking distances.",
            "check": lambda resp: ("not available" in resp.lower() or "does not contain" in resp.lower() or "no information" in resp.lower() or "not provide" in resp.lower() or "knowledge base" in resp.lower()) and "meters" not in resp.lower()
        },
        {
            "category": "Ungrounded Attribute Trap",
            "name": "Safety & Crime Rating Trap",
            "query": "What is the official safety crime index rating and school quality score for Property 1?",
            "expected_behavior": "Must refuse to fabricate crime safety ratings or school scores.",
            "check": lambda resp: ("not available" in resp.lower() or "not present" in resp.lower() or "knowledge base" in resp.lower() or "does not contain" in resp.lower() or "no information" in resp.lower())
        },
        {
            "category": "Ungrounded Attribute Trap",
            "name": "Fabricated Financing Terms Trap",
            "query": "Does the builder of Property 1 offer a 0% interest loan with zero down payment for 30 years?",
            "expected_behavior": "Must refuse to invent financing or loan schemes not in the dataset.",
            "check": lambda resp: ("not available" in resp.lower() or "not specified" in resp.lower() or "not mentioned" in resp.lower() or "knowledge base" in resp.lower() or "no information" in resp.lower() or "does not contain" in resp.lower())
        }
    ]

    results_summary = []
    all_passed = True

    for i, test in enumerate(test_cases, 1):
        bot.clear_memory() # Clear conversation history between independent tests
        print(f"\n[Test {i}/{len(test_cases)}] {test['category']}: '{test['name']}'")
        print(f"Query: \"{test['query']}\"")
        print(f"Expected: {test['expected_behavior']}")

        start_time = time.time()
        response = bot.ask(test['query'])
        elapsed = time.time() - start_time

        passed = test['check'](response)
        if not passed:
            all_passed = False

        status_str = "PASS" if passed else "FAIL"
        print(f"Result: [{status_str}] (in {elapsed:.2f}s)")
        print(f"Response Snippet:\n{response[:200]}...")

        results_summary.append({
            "id": i,
            "name": test['name'],
            "category": test['category'],
            "status": status_str,
            "time": f"{elapsed:.2f}s"
        })
        time.sleep(1) # Brief pause to avoid API rate limits

    # Print formatted summary table
    print("\n" + "=" * 75)
    print("                    HALLUCINATION TEST SUMMARY REPORT")
    print("=" * 75)
    print(f"{'#':<3} | {'Test Name':<35} | {'Category':<25} | {'Status':<6}")
    print("-" * 75)
    for r in results_summary:
        print(f"{r['id']:<3} | {r['name']:<35} | {r['category']:<25} | {r['status']:<6}")
    print("=" * 75)

    if all_passed:
        print(">> ALL 7 HALLUCINATION & GROUNDING TESTS PASSED PERFECTLY!")
    else:
        print(">> SOME TESTS FAILED. Please review the responses above.")

    return all_passed

if __name__ == "__main__":
    success = run_hallucination_tests()
    sys.exit(0 if success else 1)
