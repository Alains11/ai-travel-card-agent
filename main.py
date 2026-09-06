import json
import sys
from pathlib import Path
from datetime import datetime
import ollama
from flight_search import search_flights
import argparse

print("main.py version: v3-duffel-integrated")

# Define paths to local card data
CARDS_FILE = Path(__file__).parent / "cards.json"
SEED_FILE = Path(__file__).parent / "card.json"


def _is_business_card(card_name: str) -> bool:
    """Simple keyword filter to exclude business/corporate cards from
    personal recommendations. Matches trip_planner.py's filter."""
    name = card_name.lower()
    return any(kw in name for kw in ["business", "corporate", "commercial", "ink "])

def load_card_data() -> tuple[str, str, str | None]:
    """Reads the JSON database updated by updater.py, falling back to the seed file.

    Returns (card_data_json_str, source_label, last_run_timestamp_or_None).
    """
    data = []
    source = None
    timestamp = None

    if CARDS_FILE.exists():
        with open(CARDS_FILE, "r", encoding="utf-8") as f:
            raw = json.load(f)
        if isinstance(raw, dict) and "cards" in raw:
            data = raw["cards"]
            timestamp = raw.get("metadata", {}).get("last_run_timestamp")
        else:
            data = raw
        if data:
            source = f"{CARDS_FILE.name} (live updater output)"

    if not data and SEED_FILE.exists():
        print(f"'{CARDS_FILE.name}' has no cards yet - using seed data from '{SEED_FILE.name}'.")
        with open(SEED_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        source = f"{SEED_FILE.name} (static seed data - NOT live)"
        timestamp = None

    if not data:
        print(f"Error: no card data found in '{CARDS_FILE.name}' or '{SEED_FILE.name}'. "
              f"Run 'updater.py' first.")
        sys.exit(1)

    return json.dumps(data, indent=2), source, timestamp

def run_agent(model_name: str = "gemma4:31b-cloud", debug: bool = True, origin: str = "JFK", destination: str = "LHR", date: str = "2026-12-01"):
    """Sends credit card data and real-time flight prices to local Gemma model for valuation analysis."""
    
    output = []
    output.append("--- Travel Destination Analysis ---")

    # 1. Flight Search
    flight_info = search_flights(origin.upper(), destination.upper(), date)
    flight_context = ""
    if flight_info:
        flight_context = f"The current cheapest cash price for a flight from {origin} to {destination} on {date} is {flight_info['price']} {flight_info['currency']}."
    else:
        flight_context = "Real-time flight pricing is currently unavailable."
    output.append(flight_context)

    # 2. Load Card Data
    card_data, source, timestamp = load_card_data()
    parsed = json.loads(card_data)

    all_count = len(parsed)
    parsed = [c for c in parsed if not _is_business_card(c.get("card_name", ""))]
    excluded = all_count - len(parsed)
    card_data = json.dumps(parsed, indent=2)

    output.append(f"Data source: {source}")
    if excluded:
        output.append(f"Excluded {excluded} business/corporate card(s) - personal recommendations only.")
    if timestamp:
        try:
            fetched_at = datetime.fromisoformat(timestamp)
            age = datetime.now() - fetched_at
            hours = age.total_seconds() / 3600
            output.append(f"Data last fetched: {timestamp} ({hours:.1f} hours ago)")
            if hours > 24:
                output.append("Warning: this data is more than 24 hours old - run updater.py to refresh it.")
        except ValueError:
            output.append(f"Data last fetched: {timestamp} (unparsed timestamp)")
    else:
        output.append("Data last fetched: unknown (seed data has no timestamp - this is NOT live data)")
    output.append("")

    if not parsed:
        return "Error: no personal (non-business) cards remain after filtering."

    if debug:
        debug_path = Path(__file__).parent / "last_sent_card_data.json"
        with open(debug_path, "w", encoding="utf-8") as f:
            f.write(card_data)

        unknown_heavy = 0
        for c in parsed:
            values = [v for k, v in c.items() if k in ("card_name", "welcome_bonus", "annual_fee")]
            if any(str(v).strip().lower() == "unknown" for v in values):
                unknown_heavy += 1

        output.append(f"[debug] {len(parsed)} card(s) loaded, {unknown_heavy} with at least one 'Unknown' field.")
        output.append(f"[debug] full dataset written to: {debug_path}")

    # 3. AI Analysis with Flight Context
    output.append(f"\nLoading local model '{model_name}' via Ollama...")
    output.append("Evaluating travel card valuations based on your destination...\n")

    system_prompt = (
        "You are an expert travel rewards strategist. "
        "Analyze the provided credit card dataset and select the best travel credit cards (up to 5). "
        "Rank them primarily by net signup bonus value, annual fee offset, and point transfer flexibility. "
        "Crucially, use the provided real-time flight price to explain the 'Real World Value' of the cards. "
        "For example, if a card's bonus is 60k points and the flight costs $800, explain how many points are needed "
        "and if the bonus covers the trip. "
        "Format the response into a clear, bulleted summary listing: "
        "1) Card Name, 2) Welcome Offer, 3) Estimated Net Value, and 4) Core Reason for Selection (referencing the flight price)."
    )

    user_prompt = (
        f"Flight Context: {flight_context}\n\n"
        f"Here is the latest credit card data:\n\n{card_data}\n\n"
        "Provide the top travel cards for this specific trip."
    )

    try:
        response = ollama.chat(
            model=model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
        )
        
        output.append("=== TOP TRAVEL CREDIT CARDS FOR YOUR TRIP ===\n")
        output.append(response['message']['content'])
        return "\n".join(output)
    except Exception as e:
        return f"Error calling Ollama: {str(e)}"

    except Exception as e:
        print(f"Error connecting to Ollama: {e}")
        print("Make sure Ollama is running locally ('ollama serve').")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Travel Card Valuation Agent")
    parser.add_argument("--origin", type=str, default="JFK", help="Origin airport code")
    parser.add_argument("--destination", type=str, default="LHR", help="Destination airport code")
    parser.add_argument("--date", type=str, default="2026-12-01", help="Departure date (YYYY-MM-DD)")
    parser.add_argument("--model", type=str, default="gemma4:31b-cloud", help="Ollama model name")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")

    args = parser.parse_args()
    
    result = run_agent(
        model_name=args.model,
        debug=args.debug,
        origin=args.origin,
        destination=args.destination,
        date=args.date
    )
    print(result)

