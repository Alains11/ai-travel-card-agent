# AI Travel Card Agent
An automated agent that fetches the latest travel credit card offers from the web, extracts structured data using an LLM, and analyzes the top cards for maximum value.

## 🚀 Features
- **Automated Data Fetching**: Uses DuckDuckGo to find the latest credit card offers.
- **Web Scraping**: Extracts full page content for higher accuracy.
- **LLM Extraction**: Uses `gemma4:31b-cloud` via Ollama to parse unstructured text into structured JSON.
- **Strategic Analysis**: Ranks cards based on net signup bonus value and annual fee offsets.

## 🛠️ Setup

### Prerequisites
- [Ollama](https://ollama.ai/) installed and running.
- Model `gemma4:31b-cloud` pulled: `ollama pull gemma4:31b-cloud`
- Python 3.10+

### Installation
1. Clone the repository:
   ```bash
   git clone <your-repo-url>
   cd "Ai Agent"
   ```

2. Create and activate a virtual environment:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```

3. Install dependencies:
   ```bash
   pip install duckduckgo_search ollama trafilatura beautifulsoup4 requests schedule
   ```

## 📖 Usage

### 1. Update the Database
Run the updater to fetch the latest data from the web:
```bash
python3 updater.py
```
This will create/update `cards.json`.

### 2. Run the Analysis Agent
Run the main agent to get the top 5 travel card recommendations:
```bash
python3 main.py
```

## 📁 Project Structure
- `main.py`: The analysis agent that interacts with the LLM.
- `updater.py`: The data pipeline that searches and scrapes the web.
- `cards.json`: Local database of extracted card offers.
