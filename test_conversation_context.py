from pathlib import Path

from rag_chatbot import RealEstateRAG


PROJECT_ROOT = Path(__file__).resolve().parent


def new_bot():
    return RealEstateRAG(index_dir=PROJECT_ROOT / "faiss_index", require_gemini=False)


def test_progressive_property_context():
    bot = new_bot()
    bot.update_conversation_context("I want a house in Velachery.")
    bot.update_conversation_context("3 BHK.")
    context = bot.update_conversation_context("Under 60 lakhs.")
    assert context["location"] == "Velachery"
    assert context["property_type"] == "house"
    assert context["bhk"] == 3
    assert context["max_price"] == 60


def test_land_followup_keeps_location():
    bot = new_bot()
    bot.update_conversation_context("I am looking for land in Tambaram.")
    query = bot.build_contextual_query("Show me some.")
    assert bot.conversation_context["location"] == "Tambaram"
    assert "Tambaram" in query
    assert bot.conversation_context["land_requirement"] is True


def test_location_replacement():
    bot = new_bot()
    bot.update_conversation_context("I need a property in Anna Nagar.")
    context = bot.update_conversation_context("Actually, change the location to Velachery.")
    assert context["location"] == "Velachery"


def test_nearby_without_location_requests_location():
    bot = new_bot()
    bot.update_conversation_context("Show me properties nearby.")
    assert bot.needs_location_for_nearby() is True
    assert bot.conversation_context["location"] is None


def test_facility_followup_uses_previous_location():
    bot = new_bot()
    bot.update_conversation_context("I am in Velachery.")
    query = bot.build_contextual_query("Are there schools nearby?")
    assert bot.conversation_context["location"] == "Velachery"
    assert bot.conversation_context["facility_type"] == "school"
    assert "Velachery" in query
    assert "school" in query


def main():
    tests = [
        test_progressive_property_context,
        test_land_followup_keeps_location,
        test_location_replacement,
        test_nearby_without_location_requests_location,
        test_facility_followup_uses_previous_location,
    ]
    for test in tests:
        test()
        print(f"PASS: {test.__name__}")
    print("ALL CONVERSATION CONTEXT TESTS PASSED")


if __name__ == "__main__":
    main()