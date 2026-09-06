"""
trip_planner.py

Extends the same search -> scrape -> LLM-extract pipeline used in updater.py,
but pointed at a specific upcoming trip instead of generic card research.

Two things it does each run:
  1. Snapshots estimated flight and hotel prices for the trip legs, appending
     to a running price history (trip_prices.json) so you can see trends
     over the ~year of lead time instead of just a single point-in-time quote.
  2. Cross-references cards.json (produced by updater.py) against this trip's
     needs - no foreign transaction fees, transferable points useful for SE
     Asia routes - and recommends WHEN to apply so a card's minimum-spend
     window lines up with your actual flight/hotel purchases instead of being
     wasted on unrelated spending.

This does NOT book anything or check real-time fares precisely - DDG-scraped
search results and an LLM extracting numbers from articles/listings are
estimates, not live fare data. Treat price numbers as directional, not exact.
For actual booking, use Google Flights / ITA Matrix / the airline directly
once you're within a few months of departure.
"""

import json
import datetime
import random
import sys
import time
from pathlib import Path

import ollama
import trafilatura

try:
    from ddgs import DDGS
    from ddgs.exceptions import DDGSException, RatelimitException
except ImportError:
    from duckduckgo_search import DDGS
    from duckduckgo_search.exceptions import (
        DuckDuckGoSearchException as DDGSException,
        RatelimitException,
    )

print("trip_planner.py version: v3-wider-search")

# ---------------------------------------------------------------------------
# TRIP CONFIG - edit this section for your actual trip.
# ---------------------------------------------------------------------------
HOME_AIRPORT = "BOS"  # <-- change to your actual departure airport code
TRIP_START = datetime.date(2027, 9, 1)   # approximate - adjust to your real dates
TRIP_END = datetime.date(2027, 9, 22)    # 1.5 weeks Vietnam + 1.5 weeks Cambodia

LEGS = [
    {"label": "Flight to Vietnam", "query": f"cheap flights {HOME_AIRPORT} to Hanoi Vietnam September 2027"},
    {"label": "Flight Vietnam to Cambodia", "query": "cheap flights Ho Chi Minh City to Siem Reap Cambodia"},
    {"label": "Flight home from Cambodia", "query": f"cheap flights Phnom Penh to {HOME_AIRPORT}"},
    {"label": "Hotels - Hanoi", "query": "best mid-range hotels Hanoi Vietnam per night price"},
    {"label": "Hotels - Ho Chi Minh City", "query": "best mid-range hotels Ho Chi Minh City per night price"},
    {"label": "Hotels - Siem Reap", "query": "best mid-range hotels Siem Reap Cambodia per night price"},
]

CARDS_FILE = Path(__file__).parent / "cards.json"
TRIP_PRICES_FILE = Path(__file__).parent / "trip_prices.json"
OLLAMA_MODEL = "gemma4:31b-cloud"

MAX_SEARCH_RETRIES = 4
BASE_BACKOFF_SECONDS = 5
BACKENDS_TO_TRY = ["lite", "html", "auto"]


def _ddgs_text_with_retry(ddgs, query, max_results=4):
    for backend in BACKENDS_TO_TRY:
        for attempt in range(1, MAX_SEARCH_RETRIES + 1):
            try:
                res = list(ddgs.text(query, max_results=max_results, backend=backend))
                if res:
                    return res
                break
            except RatelimitException:
                if attempt == MAX_SEARCH_RETRIES:
                    break
                time.sleep(BASE_BACKOFF_SECONDS * (2 ** (attempt - 1)) + random.uniform(0, 2))
            except DDGSException:
                break
    return []


def _strip_json_fences(raw: str) -> str:
    s = raw.strip()
    if s.startswith("```"):
        s = s.split("\n", 1)[1] if "\n" in s else s.lstrip("`")
        if s.rstrip().endswith("```"):
            s = s.rstrip()[:-3]
    return s.strip()


def extract_price_estimate(label, title, text):
    """Asks the LLM to pull a rough price figure and one-line context out of
    scraped page text. Returns None if nothing usable is found."""
    truncated = text[:6000]
    prompt = (
        f"You are extracting a rough price estimate from travel content for: {label}.\n\n"
        f"Page title: {title}\nContent: {truncated}\n\n"
        "Respond ONLY in JSON with keys: 'price_estimate' (a short string like "
        "'$450-600 round trip' or '$25-40/night', or 'Unknown' if no price is mentioned), "
        "and 'note' (one short sentence of relevant context, or empty string). "
        "Do not include conversational text."
    )
    try:
        response = ollama.generate(
            model=OLLAMA_MODEL, prompt=prompt, format="json",
            options={"num_predict": 1024}
        )
        raw = response.get("response", "")
        if not raw.strip():
            return None
        cleaned = _strip_json_fences(raw)
        return json.loads(cleaned)
    except Exception as e:
        print(f"  Extraction error for '{label}': {e}")
        return None


def snapshot_prices():
    """Runs one search+scrape+extract pass per trip leg and returns a list of
    {leg, price_estimate, note, url, snapshot_date} entries. Tries up to 3
    results per leg (not just the top one) so a single thin/unparseable page
    doesn't leave that leg with no data for the whole run."""
    snapshot = []
    with DDGS() as ddgs:
        for leg in LEGS:
            print(f"Researching: {leg['label']}")
            results = _ddgs_text_with_retry(ddgs, leg["query"], max_results=6)
            if not results:
                print(f"  No search results for '{leg['label']}'.")
                continue

            found = False
            for r in results[:3]:  # try up to 3 results before giving up on this leg
                url = r["href"]
                downloaded = trafilatura.fetch_url(url)
                if not downloaded:
                    continue
                page_text = trafilatura.extract(downloaded)
                if not page_text:
                    continue

                extracted = extract_price_estimate(leg["label"], r["title"], page_text)
                if extracted and extracted.get("price_estimate", "Unknown").strip().lower() != "unknown":
                    snapshot.append({
                        "leg": leg["label"],
                        "price_estimate": extracted.get("price_estimate", "Unknown"),
                        "note": extracted.get("note", ""),
                        "url": url,
                        "snapshot_date": str(datetime.date.today()),
                    })
                    print(f"  -> {extracted.get('price_estimate', 'Unknown')} ({url})")
                    found = True
                    break  # got a usable price for this leg, move to the next leg
                time.sleep(1)

            if not found:
                print(f"  No usable price found for '{leg['label']}' after trying "
                      f"{min(3, len(results))} result(s).")
            time.sleep(1)
    return snapshot


def save_snapshot(snapshot):
    """Appends this run's snapshot to a running history file rather than
    overwriting, so price trends build up over the ~year of lead time."""
    history = {"trip": {"start": str(TRIP_START), "end": str(TRIP_END),
                         "home_airport": HOME_AIRPORT}, "snapshots": []}
    if TRIP_PRICES_FILE.exists():
        with open(TRIP_PRICES_FILE, "r", encoding="utf-8") as f:
            history = json.load(f)

    history["snapshots"].append({
        "run_timestamp": datetime.datetime.now().isoformat(),
        "prices": snapshot
    })

    with open(TRIP_PRICES_FILE, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)
    print(f"Saved snapshot with {len(snapshot)} price point(s) to {TRIP_PRICES_FILE}")


def _is_business_card(card_name: str) -> bool:
    """Simple keyword filter to exclude business/commercial cards from a
    personal trip's recommendations."""
    name = card_name.lower()
    return any(kw in name for kw in ["business", "corporate", "commercial", "ink "])


def card_timing_recommendation():
    """Reads cards.json and prints when to apply for cards relative to the
    trip, filtering for what actually matters on this specific trip:
    no foreign transaction fees + transferable points to SE Asia carriers.
    Business/corporate cards are excluded - this is personal trip spend."""
    if not CARDS_FILE.exists():
        print("No cards.json found - run updater.py first.")
        return

    with open(CARDS_FILE, "r", encoding="utf-8") as f:
        raw = json.load(f)
    all_cards = raw.get("cards", [])
    if not all_cards:
        print("cards.json has no cards yet.")
        return

    cards = [c for c in all_cards if not _is_business_card(c.get("card_name", ""))]
    excluded = len(all_cards) - len(cards)
    if excluded:
        print(f"(Excluded {excluded} business/corporate card(s) from recommendations - personal trip only.)\n")
    if not cards:
        print("No personal (non-business) cards found in cards.json.")
        return

    days_to_trip = (TRIP_START - datetime.date.today()).days
    # Rule of thumb: apply ~2-3 months before you expect to make the bulk of
    # trip-related spend (flights are usually bought first, often 4-8 months
    # out; hotels can be closer to departure). This gives the statement cycle
    # time to register the minimum spend from real trip purchases.
    recommended_apply_window_start = TRIP_START - datetime.timedelta(days=240)  # ~8 months out
    recommended_apply_window_end = TRIP_START - datetime.timedelta(days=120)    # ~4 months out

    print(f"\nTrip: {TRIP_START} to {TRIP_END} ({days_to_trip} days from today)")
    print(f"Recommended card application window (to align minimum spend with "
          f"flight/hotel purchases): {recommended_apply_window_start} to {recommended_apply_window_end}")
    print("(Flights are usually cheapest/booked 4-8 months out; book the flight "
          "shortly after opening the card so its spend counts toward the bonus.)\n")

    print("Cards from your latest scrape (verify FTF and transfer partners yourself - "
          "the scraper doesn't reliably extract those fields yet):")
    for c in cards:
        print(f"  - {c.get('card_name', 'Unknown')}: {c.get('welcome_bonus', 'Unknown')}, "
              f"fee {c.get('annual_fee', 'Unknown')} ({c.get('url', '')})")

    print("\nFor Vietnam/Cambodia specifically, prioritize cards with:")
    print("  - No foreign transaction fee (most premium travel cards qualify - verify each)")
    print("  - Transferable points (Amex Membership Rewards, Chase Ultimate Rewards, "
          "Citi ThankYou) rather than a fixed airline/hotel brand, since direct US-to-"
          "Vietnam/Cambodia award availability is limited on any single program")
    print("  - Amex MR transfers well to ANA and Cathay Pacific for Star Alliance/"
          "Oneworld routing through Asia; Chase UR transfers to Singapore Airlines "
          "KrisFlyer, which covers some SE Asia routing")


if __name__ == "__main__":
    if "--prices-only" in sys.argv:
        snap = snapshot_prices()
        if snap:
            save_snapshot(snap)
    elif "--recommend-only" in sys.argv:
        card_timing_recommendation()
    else:
        snap = snapshot_prices()
        if snap:
            save_snapshot(snap)
        card_timing_recommendation()

