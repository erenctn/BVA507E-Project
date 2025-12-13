import requests
from bs4 import BeautifulSoup

class NewsAgent:
    def __init__(self):
        # MORE UP-TO-DATE AND FAST SOURCES
        # 1. Cointelegraph: Updates very frequently (Most up-to-date).
        # 2. Decrypt: Technology and culture focused.
        # 3. Yahoo Finance Crypto: Financially focused.
        self.rss_sources = [
            "https://cointelegraph.com/rss",
            "https://decrypt.co/feed",
            "https://finance.yahoo.com/news/rssindex"
        ]

    def fetch_latest_news(self, limit=10):
        """
        Fetches and combines the latest news from multiple RSS sources.
        """
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        
        all_news = []
        
        try:
            # Iterate through sources
            for url in self.rss_sources:
                try:
                    response = requests.get(url, headers=headers, timeout=5)
                    if response.status_code == 200:
                        soup = BeautifulSoup(response.content, features="xml")
                        items = soup.findAll('item')
                        
                        # Get the newest 4 news items from each source (To ensure diversity)
                        for item in items[:4]:
                            title = item.title.text.strip()
                            pub_date = item.pubDate.text.strip() if item.pubDate else ""
                            
                            # Determine source name (From link or URL)
                            source_name = "News"
                            if "cointelegraph" in url: source_name = "CoinTelegraph"
                            elif "decrypt" in url: source_name = "Decrypt"
                            elif "yahoo" in url: source_name = "YahooFin"
                            
                            # Add to list
                            all_news.append(f"- [{source_name}] {title} ({pub_date})")
                            
                except Exception as e:
                    print(f"Error ({url}): {e}")
                    continue # If one source fails, move to the next

            # If there is no news
            if not all_news:
                return ["News sources are currently unreachable."]

            # Return the requested limit from collected news
            # (Since we already took the newest ones from each source, it will be a mixed and up-to-date list)
            return all_news[:limit]

        except Exception as e:
            return [f"General News Error: {str(e)}"]

    def get_market_sentiment_prompt(self):
        """Converts news to text for LLM."""
        news = self.fetch_latest_news(limit=12) # Let's send more data
        return "\n".join(news)