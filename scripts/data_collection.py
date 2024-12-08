import requests
import json
import logging
import time
import random
import sys
import pandas as pd
from pathlib import Path
from config import EBAY_APP_ID, RAW_DIR, JEWELRY_CATEGORIES, SUBCATEGORIES, SCRAPING_CONFIG

# data_collection.py: Scrape jewelry listings from eBay (or other sources).
# Uses fallback logic, logging, retries. CPU-only.

class EbayAPIScraper:
    def __init__(self):
        self.headers = {
            'X-EBAY-API-APP-ID': EBAY_APP_ID,
            'X-EBAY-API-CALL-NAME': 'findItemsByKeywords',
            'X-EBAY-API-RESPONSE-ENCODING': 'JSON',
            'X-EBAY-API-VERSION': '1.0.0',
            'X-EBAY-API-SITE-ID': '0',
            'Content-Type': 'application/json'
        }
        self.output_dir = RAW_DIR
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _perform_request(self, query: str, page: int) -> list:
        url = 'https://svcs.ebay.com/services/search/FindingService/v1'
        params = {
            'OPERATION-NAME': 'findItemsByKeywords',
            'SERVICE-VERSION': '1.0.0',
            'SECURITY-APPNAME': EBAY_APP_ID,
            'RESPONSE-DATA-FORMAT': 'JSON',
            'REST-PAYLOAD': '',
            'keywords': query,
            'paginationInput.entriesPerPage': '20',
            'paginationInput.pageNumber': str(page)
        }

        attempts = 0
        max_retries = SCRAPING_CONFIG['max_retries']
        delay = SCRAPING_CONFIG['delay_between_requests']
        timeout = SCRAPING_CONFIG['timeout']

        while attempts <= max_retries:
            attempts += 1
            try:
                logging.debug(f"Requesting page {page} for '{query}', attempt {attempts}.")
                response = requests.get(url, params=params, headers=self.headers, timeout=timeout)
                response.raise_for_status()
                data = response.json()
                items = data.get('findItemsByKeywordsResponse', [{}])[0].get('searchResult', [{}])[0].get('item', [])
                logging.debug(f"Received {len(items)} items for '{query}' page {page}.")
                return items
            except requests.exceptions.RequestException as e:
                logging.warning(f"Request error attempt {attempts}: {e}")
                if attempts > max_retries:
                    logging.error(f"Max retries exceeded for '{query}' page {page}.")
                    return []
                logging.info(f"Retrying in {delay}s...")
                time.sleep(delay)
        return []

    def scrape_category(self, category: str, subcat: str, pages: int):
        query = subcat if subcat else category
        logging.info(f"Starting scrape for category='{category}', subcategory='{subcat}', pages={pages}.")
        all_listings = []

        for page in range(1, pages+1):
            items = self._perform_request(query, page)
            for item in items:
                price = item.get('sellingStatus',[{}])[0].get('currentPrice',[{}])[0].get('__value__','N/A')
                condition = ''
                if 'condition' in item:
                    cval = item['condition'][0].get('conditionDisplayName',[''])
                    condition = cval[0] if cval else ''
                image = item.get('galleryURL',[''])[0] if 'galleryURL' in item else ''

                listing = {
                    'url': item.get('viewItemURL',[''])[0],
                    'title': item.get('title',[''])[0],
                    'category': category,
                    'subcategory': subcat,
                    'price': price,
                    'condition': condition.lower().strip(),
                    'images': json.dumps([image]),
                    'scraped_at': time.strftime('%Y-%m-%d %H:%M:%S')
                }
                all_listings.append(listing)

            sleep_time = random.uniform(1,3)
            logging.debug(f"Sleeping {sleep_time:.2f}s before next page.")
            time.sleep(sleep_time)

        logging.info(f"Completed scraping {category}-{subcat}, got {len(all_listings)} listings.")
        return all_listings

    def scrape_all_categories(self, category: str, subcat: str):
        if category not in JEWELRY_CATEGORIES:
            logging.error(f"Invalid category '{category}'")
            return
        if subcat not in SUBCATEGORIES.get(category, []):
            logging.error(f"Invalid subcategory '{subcat}' for '{category}'")
            return

        pages = SCRAPING_CONFIG['max_pages_per_category']
        listings = self.scrape_category(category, subcat, pages)
        if listings:
            fname = f"{category}_{subcat.replace(' ','_')}_listings.csv"
            df = pd.DataFrame(listings)
            df.to_csv(self.output_dir / fname, index=False)
            logging.info(f"Saved {len(listings)} listings to {fname}.")
        else:
            logging.warning(f"No listings found for {category}-{subcat}.")

if __name__ == '__main__':
    if len(sys.argv)<3:
        logging.error("Usage: python3 data_collection.py category subcategory")
        sys.exit(1)
    category_arg = sys.argv[1]
    subcat_arg = sys.argv[2]
    scraper = EbayAPIScraper()
    scraper.scrape_all_categories(category_arg, subcat_arg)
