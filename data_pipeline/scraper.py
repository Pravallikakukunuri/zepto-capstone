# Module 1: Data Pipeline - scrapes books.toscrape.com
import time
import requests
from bs4 import BeautifulSoup

BASE_URL = "https://books.toscrape.com/"
HOME_URL = BASE_URL + "index.html"
HEADERS = {"User-Agent": "Mozilla/5.0 (capstone-data-pipeline-scraper)"}


def get_soup(url, retries=3, delay=1.0):
    last_exc = None
    for _ in range(retries):
        try:
            resp = requests.get(url, headers=HEADERS, timeout=10)
            resp.raise_for_status()
            return BeautifulSoup(resp.content, "html.parser")
        except requests.RequestException as exc:
            last_exc = exc
            time.sleep(delay)
    raise RuntimeError(f"Failed to fetch {url} after {retries} attempts: {last_exc}")


def discover_categories():
    soup = get_soup(HOME_URL)
    sidebar_links = soup.select("div.side_categories ul li ul li a")
    categories = []
    for a in sidebar_links:
        name = a.get_text(strip=True)
        url = BASE_URL + a["href"]
        categories.append((name, url))
    return categories


def _parse_book_card(card, category_name):
    title = card.h3.a["title"]
    price_text = card.select_one("p.price_color").get_text(strip=True)
    rating_classes = card.select_one("p.star-rating")["class"]
    rating_word = rating_classes[1] if len(rating_classes) > 1 else None
    availability_text = card.select_one("p.instock.availability").get_text(strip=True)
    return {
        "title": title,
        "price": price_text,
        "star_rating": rating_word,
        "availability": availability_text,
        "category": category_name,
    }


def scrape_category(category_name, category_url, max_pages=None):
    books = []
    url = category_url
    page_count = 0
    while url:
        soup = get_soup(url)
        cards = soup.select("article.product_pod")
        for card in cards:
            books.append(_parse_book_card(card, category_name))
        page_count += 1
        if max_pages and page_count >= max_pages:
            break
        next_link = soup.select_one("li.next a")
        if next_link:
            url = url.rsplit("/", 1)[0] + "/" + next_link["href"]
        else:
            url = None
    return books


def scrape_books(min_total=60, min_categories=3):
    categories = discover_categories()
    all_books = []
    used_categories = 0
    for name, url in categories:
        all_books.extend(scrape_category(name, url))
        used_categories += 1
        if len(all_books) >= min_total and used_categories >= min_categories:
            break
    return all_books


if __name__ == "__main__":
    data = scrape_books()
    print(f"Scraped {len(data)} books across categories.")
    print(data[:2])