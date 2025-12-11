import requests
from bs4 import BeautifulSoup

class NewsAgent:
    def __init__(self):
        # DAHA GÜNCEL VE HIZLI KAYNAKLAR
        # 1. Cointelegraph: Çok sık haber girer (En güncel).
        # 2. Decrypt: Teknoloji ve kültür odaklıdır.
        # 3. Yahoo Finance Crypto: Finansal odaklıdır.
        self.rss_sources = [
            "https://cointelegraph.com/rss",
            "https://decrypt.co/feed",
            "https://finance.yahoo.com/news/rssindex"
        ]

    def fetch_latest_news(self, limit=10):
        """
        Birden fazla RSS kaynağından en son haberleri çeker ve birleştirir.
        """
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        
        all_news = []
        
        try:
            # Kaynakları gez
            for url in self.rss_sources:
                try:
                    response = requests.get(url, headers=headers, timeout=5)
                    if response.status_code == 200:
                        soup = BeautifulSoup(response.content, features="xml")
                        items = soup.findAll('item')
                        
                        # Her kaynaktan en yeni 4 haberi al (Toplamda çeşitlilik olsun)
                        for item in items[:4]:
                            title = item.title.text.strip()
                            pub_date = item.pubDate.text.strip() if item.pubDate else ""
                            
                            # Kaynak ismini belirle (Linkten veya URL'den)
                            source_name = "News"
                            if "cointelegraph" in url: source_name = "CoinTelegraph"
                            elif "decrypt" in url: source_name = "Decrypt"
                            elif "yahoo" in url: source_name = "YahooFin"
                            
                            # Listeye ekle
                            all_news.append(f"- [{source_name}] {title} ({pub_date})")
                            
                except Exception as e:
                    print(f"Hata ({url}): {e}")
                    continue # Bir kaynak çalışmazsa diğerine geç

            # Eğer hiç haber yoksa
            if not all_news:
                return ["Haber kaynaklarına şu an ulaşılamıyor."]

            # Toplanan haberlerden istenen limit kadarını döndür
            # (Zaten her kaynaktan en yenileri aldığımız için karışık ve güncel bir liste olur)
            return all_news[:limit]

        except Exception as e:
            return [f"Genel Haber Hatası: {str(e)}"]

    def get_market_sentiment_prompt(self):
        """LLM için haberleri metne döker."""
        news = self.fetch_latest_news(limit=12) # Daha fazla veri gönderelim
        return "\n".join(news)