import json
import datetime
import os
import random
import schedule
import sys
import time
import ollama
import trafilatura
from pathlib import Path

# Always resolve cards.json relative to this script's location, not the
# current working directory (which changes depending on how/where you run it).
CARDS_FILE = Path(__file__).parent / "cards.json"

# duckduckgo_search was renamed to `ddgs` - prefer the new package name,
# fall back to the old one if that's what's installed.
try:
    from ddgs import DDGS
    from ddgs.exceptions import DDGSException, RatelimitException
    _ACTIVE_SEARCH_PACKAGE = "ddgs"
except ImportError:
    from duckduckgo_search import DDGS
    from duckduckgo_search.exceptions import (
        DuckDuckGoSearchException as DDGSException,
        RatelimitException,
    )
    _ACTIVE_SEARCH_PACKAGE = "duckduckgo_search (deprecated - run: pip install ddgs)"

print(f"Search backend package in use: {_ACTIVE_SEARCH_PACKAGE}")
print("updater.py version: v7-catchup-if-stale")

# Ollama model tag - confirmed working: gemma4:31b-cloud (Ollama 0.33.3)
OLLAMA_MODEL = "gemma4:31b-cloud"

# DDG scraping is inherently rate-limited/bot-detected. These knobs make it
# more resilient but can't eliminate the risk - if it's blocking your IP
# consistently, consider a paid search API (Brave Search API, SerpAPI, Tavily)
# as a fallback backend.
MAX_SEARCH_RETRIES = 4
BASE_BACKOFF_SECONDS = 5
DELAY_BETWEEN_QUERIES = 3
BACKENDS_TO_TRY = ["lite", "html", "auto"]


def _ddgs_text_with_retry(ddgs, query, max_results):
    """Runs ddgs.text() across a few backends with backoff, and reports raw
    counts per backend so a silent zero-result response (no exception) is
    distinguishable from an actual rate-limit error."""
    for backend in BACKENDS_TO_TRY:
        for attempt in range(1, MAX_SEARCH_RETRIES + 1):
            try:
                res = list(ddgs.text(query, max_results=max_results, backend=backend))
                print(f"  [backend={backend}] returned {len(res)} raw result(s)")
                if res:
                    return res
                break  # zero results, no exception - move to next backend rather than retrying
            except RatelimitException:
                if attempt == MAX_SEARCH_RETRIES:
                    print(f"  [backend={backend}] still rate-limited after {MAX_SEARCH_RETRIES} attempts.")
                    break
                wait = BASE_BACKOFF_SECONDS * (2 ** (attempt - 1)) + random.uniform(0, 2)
                print(f"  [backend={backend}] rate-limited (attempt {attempt}/{MAX_SEARCH_RETRIES}). "
                      f"Backing off {wait:.1f}s...")
                time.sleep(wait)
            except DDGSException as e:
                print(f"  [backend={backend}] error: {e}")
                break
    print(f"  All backends ({', '.join(BACKENDS_TO_TRY)}) returned zero usable results for '{query}'. "
          f"This usually means DuckDuckGo is blocking this IP outright rather than just "
          f"rate-limiting it - a paid search API would be more reliable here.")
    return []

# 1. Fetch updated card data (using DuckDuckGo + Web Scraping)
def fetch_latest_card_data():
    print("Fetching latest data from DuckDuckGo...")
    results = []
    try:
        with DDGS() as ddgs:
            # Try a specific query first, then fallback to a broad one
            queries = [
                "best travel credit cards welcome bonus annual fee",
                "best travel credit cards"
            ]
            
            search_results = []
            for query in queries:
                print(f"Searching for: {query}")
                res = _ddgs_text_with_retry(ddgs, query, max_results=5)
                if res:
                    search_results = res
                    print(f"Found {len(search_results)} results with query: {query}")
                    break
                time.sleep(DELAY_BETWEEN_QUERIES)
            
            if not search_results:
                print("No results found for any query (see rate-limit messages above if any).")
                return []

            print(f"Scraping pages and extracting data via LLM...")
            for r in search_results:
                url = r["href"]
                print(f"Scraping: {url}")
                
                # Option 1: Web Scraping - Download the actual page content
                downloaded = trafilatura.fetch_url(url)
                if downloaded:
                    # Extract main text content from the page
                    page_text = trafilatura.extract(downloaded)
                    if page_text:
                        # Pass the full page text to the LLM for high-accuracy extraction
                        extracted = extract_card_details(r["title"], page_text)
                        if extracted:
                            results.append({
                                "card_name": extracted.get("card_name", r["title"]),
                                "welcome_bonus": extracted.get("welcome_bonus", "Unknown"),
                                "annual_fee": extracted.get("annual_fee", "Unknown"),
                                "last_updated": str(datetime.date.today()),
                                "snippet": r["body"],
                                "url": url,
                                "source": "Web Scraping + LLM"
                            })
                time.sleep(1)  # be polite to the sites we're scraping, avoid tripping their own bot protection
        return results
    except Exception as e:
        print(f"Error fetching data: {e}")
        return None

def _strip_json_fences(raw: str) -> str:
    """Some models wrap JSON output in markdown code fences even when asked
    not to. Strip ```json / ``` wrapping if present."""
    s = raw.strip()
    if s.startswith("```"):
        # Remove opening fence (with optional language tag) and closing fence
        s = s.split("\n", 1)[1] if "\n" in s else s.lstrip("`")
        if s.rstrip().endswith("```"):
            s = s.rstrip()[:-3]
    return s.strip()


def extract_card_details(title, text):
    """Uses Ollama to extract structured card data from full page text."""
    # We truncate the text to avoid exceeding LLM context limits while keeping the most relevant parts
    truncated_text = text[:8000] 
    prompt = (
        f"You are a data extraction expert. Extract the credit card name, welcome bonus, and annual fee "
        f"from the following webpage content. Focus on the most prominent travel card mentioned.\n\n"
        f"Page Title: {title}\n"
        f"Content: {truncated_text}\n\n"
        "Respond ONLY in JSON format with keys: 'card_name', 'welcome_bonus', 'annual_fee'. "
        "If a value is missing, use 'Unknown'. Do not include any conversational text."
    )
    try:
        response = ollama.generate(
            model=OLLAMA_MODEL,
            prompt=prompt,
            format="json",
            # Without this, thinking-mode reasoning can consume the whole default
            # output budget before the model ever writes the JSON answer, leaving
            # response['response'] empty and json.loads() failing on "".
            options={"num_predict": 2048}
        )
        raw = response.get('response', '')
        if not raw.strip():
            thinking = response.get('thinking', '')
            print(f"Extraction error: empty response for '{title}'."
                  + (f" Thinking trace: {thinking[:300]}..." if thinking else " No thinking trace returned either."))
            return None
        cleaned = _strip_json_fences(raw)
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            print(f"Extraction error: could not parse JSON for '{title}'. "
                  f"Raw model output (first 300 chars): {raw[:300]!r}")
            return None
    except Exception as e:
        print(f"Extraction error: {e}")
        return None

# How old cards.json needs to be before we bother doing a full refresh.
# Set below 24h so a daily cadence still gets hit reliably even if the exact
# same hour is missed on a given day (e.g. laptop asleep at that hour).
STALE_THRESHOLD_HOURS = 20


def _data_is_stale() -> bool:
    """Returns True if cards.json is missing, unreadable, or older than
    STALE_THRESHOLD_HOURS. Used to skip unnecessary scrape/LLM work when
    invoked frequently (e.g. hourly via launchd) but data is still fresh."""
    if not CARDS_FILE.exists():
        return True
    try:
        with open(CARDS_FILE, "r", encoding="utf-8") as f:
            raw = json.load(f)
        timestamp = raw.get("metadata", {}).get("last_run_timestamp")
        if not timestamp:
            return True
        age_hours = (datetime.datetime.now() - datetime.datetime.fromisoformat(timestamp)).total_seconds() / 3600
        return age_hours >= STALE_THRESHOLD_HOURS
    except Exception:
        # If the file is corrupt/unreadable, treat it as stale so we try to fix it.
        return True


# 2. Overwrite the local cards.json file
def update_json_file():
    data = fetch_latest_card_data()
    if data is None:
        print("Update failed due to network error.")
        return

    if not data:
        # Don't overwrite a previously good cards.json with an empty result -
        # this is what produced the misleading {"status": "success", "cards": []} you had before.
        print("No cards extracted this run - leaving existing cards.json untouched.")
        return

    output = {
        "metadata": {
            "last_run_timestamp": datetime.datetime.now().isoformat(),
            "timezone": "EST",
            "status": "success",
            "card_count": len(data)
        },
        "cards": data
    }

    with open(CARDS_FILE, "w") as f:
        json.dump(output, f, indent=4)
    print(f"[{datetime.datetime.now()}] {CARDS_FILE} successfully updated with {len(data)} card(s).")

if __name__ == "__main__":
    if "--once" in sys.argv:
        # Single run, for use with an external scheduler (e.g. launchd) that
        # owns the timing - avoids double-scheduling with the loop below.
        update_json_file()
    elif "--if-stale" in sys.argv:
        # Invoked frequently (e.g. hourly) by launchd. Only does the actual
        # scrape/LLM work if the existing data is old enough to need it - this
        # is what makes catch-up-after-sleep work without hammering DDG/Ollama
        # every hour.
        if _data_is_stale():
            print("Data is stale or missing - running full update.")
            update_json_file()
        else:
            print("Data is still fresh - skipping this run.")
    else:
        # Schedule the task for 12:00 PM EST
        # Note: This assumes the system clock is set to EST.
        # If not, you would need to adjust the time or use a library like pytz.
        schedule.every().day.at("12:00").do(update_json_file)

        print("Agent scheduler started. Running every day at 12:00 PM EST...")

        # Run once immediately on startup so we don't have to wait until 12pm for the first update
        update_json_file()

        while True:
            schedule.run_pending()
            time.sleep(60)  # Check every minute

