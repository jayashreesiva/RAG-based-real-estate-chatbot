from pathlib import Path

import pandas as pd

from rag_chatbot import RealEstateRAG


PROJECT_ROOT = Path(__file__).resolve().parent


def new_bot():
    return RealEstateRAG(index_dir=PROJECT_ROOT / "faiss_index", require_gemini=False)


def scoped_result(bot, query):
    contextual_query = bot.build_contextual_query(query)
    results, _ = bot.retrieve(contextual_query, top_k=3, min_similarity=0.0, location_scope=bot.get_search_locations())
    return results


def test_nearby_properties_use_velachery_scope():
    bot = new_bot()
    bot.set_current_location("Velachery")
    bot.update_conversation_context("Show me houses nearby.")
    results = scoped_result(bot, "Show me houses nearby.")
    allowed = {value.lower() for value in bot.get_search_locations()}
    assert bot.conversation_context["current_location"] == "Velachery"
    assert allowed == {"velachery", "taramani", "adambakkam", "pallikaranai"}
    assert results and all(str(item["metadata"].get("location", "")).lower() in allowed for item in results)


def test_nearby_land_uses_velachery_scope():
    bot = new_bot()
    bot.set_current_location("Velachery")
    bot.update_conversation_context("Show me land nearby.")
    results = scoped_result(bot, "Show me land nearby.")
    assert results and all(item["source_type"] == "land" for item in results)


def test_nearby_schools_use_velachery_scope():
    bot = new_bot()
    bot.set_current_location("Velachery")
    bot.update_conversation_context("Are there schools nearby?")
    results = scoped_result(bot, "Are there schools nearby?")
    assert results and all(item["source_type"] == "school" for item in results)
    assert all(item["metadata"].get("area", "").lower() in {value.lower() for value in bot.get_search_locations()} for item in results)


def test_nearby_hospitals_use_velachery_scope():
    bot = new_bot()
    bot.set_current_location("Velachery")
    bot.update_conversation_context("Are there hospitals around here?")
    results = scoped_result(bot, "Are there hospitals around here?")
    assert results and all(item["source_type"] == "hospital" for item in results)


def test_location_change_reloads_taramani_scope():
    bot = new_bot()
    bot.set_current_location("Velachery")
    bot.update_conversation_context("Actually, search around Taramani.")
    bot.update_conversation_context("Show me properties nearby.")
    assert bot.conversation_context["current_location"] == "Taramani"
    assert bot.get_search_locations() == ["Taramani", "Velachery", "Perungudi", "Adambakkam"]


def test_nearby_without_location_does_not_invent_area():
    bot = new_bot()
    bot.update_conversation_context("Show me properties nearby.")
    assert bot.needs_location_for_nearby()
    assert bot.conversation_context["current_location"] is None


def test_existing_filters_are_preserved():
    bot = new_bot()
    bot.update_conversation_context("3 BHK under 60 lakhs")
    bot.set_current_location("Velachery")
    bot.update_conversation_context("Show me houses nearby.")
    assert bot.conversation_context["bhk"] == 3
    assert bot.conversation_context["max_price"] == 60
    assert bot.conversation_context["current_location"] == "Velachery"


def main():
    tests = [
        test_nearby_properties_use_velachery_scope,
        test_nearby_land_uses_velachery_scope,
        test_nearby_schools_use_velachery_scope,
        test_nearby_hospitals_use_velachery_scope,
        test_location_change_reloads_taramani_scope,
        test_nearby_without_location_does_not_invent_area,
        test_existing_filters_are_preserved,
    ]
    for test in tests:
        test()
        print(f"PASS: {test.__name__}")
    print("ALL LOCATION SEARCH TESTS PASSED")


if __name__ == "__main__":
    main()