from newspaper import Article
import pandas as pd

def scrape_article(url):
    try:
        article = Article(url)
        article.download()
        article.parse()
        text = article.text.strip()
        if len(text) < 100:
            print(f" Possibly empty article for {url}")
        return {
            "title": article.title,
            "text": text,
            "source_url": url,
            "publish_date": article.publish_date,
            "authors": article.authors
        }
    except Exception as e:
        print(f"Failed to scrape {url}: {e}")
        return None

def ingest_articles(url_list):
    data = [result for url in url_list if (result := scrape_article(url))]
    return pd.DataFrame(data)

if __name__ == "__main__":
    urls = [
        "https://www.bbc.com/news/world-africa-66252047",
        "https://www.aljazeera.com/news/2023/8/1/kenya-loan-defaults-foreign-lenders",
        "https://www.reuters.com/world/africa/kenya-economy-outlook-2024-loan-demand-2024-05-10/"
    ]
    
    df = ingest_articles(urls)
    print(df.head())
    df.to_csv("data/raw/news_data.csv", index=False)
    print("Saved to data/raw/news_data.csv")
