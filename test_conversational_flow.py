from pathlib import Path
from rag_chatbot import RealEstateRAG

PROJECT_ROOT = Path(__file__).resolve().parent

def run_tests():
    bot = RealEstateRAG(index_dir=PROJECT_ROOT / "faiss_index", require_gemini=False)
    bot.gemini_client = None  # Use deterministic local fallbacks for fast unit testing

    # 1. Greeting
    print("Testing 1: Greeting 'Hello'...")
    r1 = bot.ask("Hello")
    assert "welcome" in r1.lower() or "hello" in r1.lower()
    assert len(bot.last_retrieved_properties) == 0, "Greetings must not dump property results"
    print("PASS: Greeting does not dump properties")

    # 2. Broad intent
    print("Testing 2: Broad intent 'I\\'m looking for a property.'...")
    r2 = bot.ask("I'm looking for a property.")
    assert "location" in r2.lower() or "budget" in r2.lower() or "bhk" in r2.lower()
    assert len(bot.last_retrieved_properties) == 0, "Broad query must ask for criteria without dumping properties"
    print("PASS: Broad query asks for criteria")

    # 3. Specific query
    print("Testing 3: Specific search 'I\\'m looking for a 3 BHK in Velachery.'...")
    r3 = bot.ask("I'm looking for a 3 BHK in Velachery.")
    assert len(bot.last_retrieved_properties) > 0, "Specific search must retrieve properties"
    first_batch_ids = [str(item.get("source_id", item.get("property_id"))) for item in bot.last_retrieved_properties]
    print(f"PASS: Specific search returned {len(bot.last_retrieved_properties)} properties ({first_batch_ids})")

    # 4. Show more
    print("Testing 4: Show more 'Show me more.'...")
    r4 = bot.ask("Show me more.")
    second_batch_ids = [str(item.get("source_id", item.get("property_id"))) for item in bot.last_retrieved_properties]
    # Check that any returned properties are not duplicates
    intersection = set(first_batch_ids).intersection(set(second_batch_ids))
    assert len(intersection) == 0, f"Show more must not repeat same properties, got duplicates: {intersection}"
    if not second_batch_ids:
        assert "shown all the matching properties" in r4.lower()
    print("PASS: Show more handled without duplicates")

    # 5. Property selection: Tell me about property 1
    print("Testing 5: Property selection 'Tell me about property 1.'...")
    r5 = bot.ask("Tell me about property 1.")
    assert bot.selected_property_context is not None, "Property 1 must be selected"
    selected_id = bot.selected_property_context.get("property_id") or bot.selected_property_context.get("source_id")
    print(f"PASS: Property selection selected {selected_id}")

    # 6. Selected property follow-up: What schools are nearby?
    print("Testing 6: Schools nearby 'What schools are nearby?'...")
    r6 = bot.ask("What schools are nearby?")
    assert "school" in r6.lower() and "km" in r6.lower(), f"Expected school distances, got: {r6}"
    print("PASS: Grounded nearby schools answered")

    # 7. Selected property follow-up: What about hospitals?
    print("Testing 7: Hospitals nearby 'What about hospitals?'...")
    r7 = bot.ask("What about hospitals?")
    assert "hospital" in r7.lower() and "km" in r7.lower(), f"Expected hospital distances, got: {r7}"
    print("PASS: Grounded nearby hospitals answered")

    # 8. Selected property follow-up: Who is the broker?
    print("Testing 8: Broker inquiry 'Who is the broker?'...")
    r8 = bot.ask("Who is the broker?")
    assert "broker" in r8.lower() or "agency" in r8.lower() or "phone" in r8.lower(), f"Expected broker details, got: {r8}"
    print("PASS: Grounded broker answered")

    # 9. Selected property follow-up: How much is this property?
    print("Testing 9: Price inquiry 'How much is this property?'...")
    r9 = bot.ask("How much is this property?")
    assert "lakhs" in r9.lower() or "price" in r9.lower(), f"Expected price, got: {r9}"
    print("PASS: Grounded price answered")

    # 10. Selected property follow-up: Show me another property
    print("Testing 10: Switch property 'Show me another property.'...")
    r10 = bot.ask("Show me another property.")
    assert bot.selected_property_context is None, "Selected property must be cleared"
    assert "returning" in r10.lower() or "search" in r10.lower()
    print("PASS: Cleared selected property and returned to search results")

    print("\n" + "="*50)
    print("ALL CONVERSATIONAL FLOW TESTS PASSED SUCCESSFULLY!")
    print("="*50)

if __name__ == "__main__":
    run_tests()
