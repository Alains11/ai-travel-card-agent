import json
import sys
from pathlib import Path
import ollama

# Define paths to local card data
CARDS_FILE = Path(__file__).parent / "cards.json"
SEED_FILE = Path(__file__).parent / "card.json"

def load_card_data() -> str:
    """Reads the JSON database updated by updater.py, falling back to the seed file."""
    data = []

    if CARDS_FILE.exists():
        with open(CARDS_FILE, "r", encoding="utf-8") as f:
            raw = json.load(f)
        # Handle the format used by updater.py (dictionary with 'cards' key)
        data = raw["cards"] if isinstance(raw, dict) and "cards" in raw else raw

    if not data and SEED_FILE.exists():
        print(f"'{CARDS_FILE.name}' has no cards yet - using seed data from '{SEED_FILE.name}'.")
        with open(SEED_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)

    if not data:
        print(f"Error: no card data found in '{CARDS_FILE.name}' or '{SEED_FILE.name}'. "
              f"Run 'updater.py' first.")
        sys.exit(1)

    return json.dumps(data, indent=2)

def run_agent(model_name: str = "gemma4:31b-cloud", debug: bool = True):
    """Sends credit card data to local Gemma model via Ollama for valuation analysis."""
    card_data = load_card_data()
    parsed = json.loads(card_data)

    if debug:
        debug_path = Path(__file__).parent / "last_sent_card_data.json"
        with open(debug_path, "w", encoding="utf-8") as f:
            f.write(card_data)

        unknown_heavy = 0
        for c in parsed:
            values = [v for k, v in c.items() if k in ("card_name", "welcome_bonus", "annual_fee")]
            if any(str(v).strip().lower() == "unknown" for v in values):
                unknown_heavy += 1

        print(f"[debug] {len(parsed)} card(s) loaded, "
              f"{unknown_heavy} with at least one 'Unknown' field.")
        print(f"[debug] full dataset written to: {debug_path}")
        if parsed:
            print(f"[debug] first entry: {json.dumps(parsed[0], indent=2)}")
        print()

    print(f"Loading local model '{model_name}' via Ollama...")
    print("Evaluating travel card valuations...\n")

    system_prompt = (
        "You are an expert travel rewards strategist. "
        "Analyze the provided credit card dataset and select the best travel credit cards (up to 5). "
        "Rank them primarily by net signup bonus value, annual fee offset, and point transfer flexibility. "
        "Format the response into a clear, bulleted summary listing: "
        "1) Card Name, 2) Welcome Offer, 3) Estimated Net Value, and 4) Core Reason for Selection. "
        "Do not describe your methodology or ask for data - the dataset is provided in the user message "
        "below. Base your answer only on that dataset."
    )

    user_prompt = f"Here is the latest credit card data:\n\n{card_data}\n\nProvide the top 5 travel cards."

    try:
        response = ollama.chat(
            model=model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            # Gemma 4 supports thinking modes; if reasoning tokens eat the whole
            # output budget before the final answer is written, the response comes
            # back truncated/incomplete. Give it plenty of room.
            options={"num_predict": 4096}
        )

        message = response.get('message', {})

        # Some thinking-capable models return reasoning separately from the
        # final answer. Surface it if present so truncated-looking output is
        # easier to diagnose.
        if debug and message.get('thinking'):
            print("[debug] model thinking/reasoning trace:")
            print(message['thinking'])
            print("[debug] ---- end thinking ----\n")

        content = message.get('content', '')
        if not content.strip():
            print("Error: model returned an empty response (likely truncated by the "
                  "output token limit or a thinking-mode issue). Check [debug] thinking "
                  "output above if present, or try increasing num_predict further.")
            return

        print("=== TOP 5 TRAVEL CREDIT CARDS ===\n")
        print(content)

    except Exception as e:
        print(f"Error connecting to Ollama: {e}")
        print("Make sure Ollama is running locally ('ollama serve').")

if __name__ == "__main__":
    run_agent(model_name="gemma4:31b-cloud")