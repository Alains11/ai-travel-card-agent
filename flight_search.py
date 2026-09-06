import requests
import json
import os

# Duffel API Configuration
# It is best practice to use environment variables for API keys
DUFFEL_API_TOKEN = os.getenv("DUFFEL_API_TOKEN", "YOUR_TEST_TOKEN_HERE")
DUFFEL_API_URL = "https://api.duffel.com/air/offers"

def search_flights(origin, destination, departure_date):
    """
    Searches for the cheapest flights using the Duffel API.
    Returns the lowest cash price found.
    """
    print(f"Searching for flights from {origin} to {destination} on {departure_date}...")
    
    headers = {
        "Authorization": f"Bearer {DUFFEL_API_TOKEN}",
        "Duffel-Version": "v1",
        "Accept": "application/json",
        "Content-Type": "application/json"
    }
    
    payload = {
        "data": {
            "slices": [
                {
                    "origin": origin,
                    "destination": destination,
                    "departure_date": departure_date
                }
            ],
            "cabin_class": "ECONOMY"
        }
    }

    try:
        response = requests.post(DUFFEL_API_URL, headers=headers, json=payload)
        response.raise_for_status()
        data = response.json()
        
        offers = data.get("data", {}).get("offers", [])
        if not offers:
            print("No flight offers found.")
            return None
            
        # Find the cheapest offer
        cheapest_offer = min(offers, key=lambda x: x["total_amount"]["amount"])
        price = cheapest_offer["total_amount"]["amount"]
        currency = cheapest_offer["total_amount"]["currency"]
        
        print(f"Found cheapest flight: {price} {currency}")
        return {"price": price, "currency": currency}
        
    except Exception as e:
        print(f"Duffel API Error: {e}")
        return None

if __name__ == "__main__":
    # Quick test
    # Note: Use IATA codes (e.g., JFK, LHR, HND)
    result = search_flights("JFK", "LHR", "2026-12-01")
    print(f"Result: {result}")
