import time
import logging
import requests
import hashlib
import json
from pathlib import Path
from typing import Dict, Any, Optional
from collections import deque
from threading import Lock
from config import BASE_DIR

API_TOKENS=[
    "token1",
    "token2"
] # replace with real tokens or leave empty
MAX_REQUESTS_PER_MINUTE=30
CACHE_DIR=BASE_DIR/'api_cache'
CACHE_EXPIRY_SECONDS=300
CACHE_DIR.mkdir(parents=True,exist_ok=True)

token_queue=deque(API_TOKENS)
token_lock=Lock()

request_times=[]
rate_lock=Lock()

MAX_RETRIES=5
INITIAL_BACKOFF=1.0
cache_lock=Lock()

def get_next_token()->Optional[str]:
    with token_lock:
        if not token_queue:
            return None
        token=token_queue[0]
        token_queue.rotate(-1)
        return token

def check_rate_limit():
    with rate_lock:
        now=time.time()
        while request_times and (now - request_times[0])>60:
            request_times.pop(0)
        if len(request_times)>=MAX_REQUESTS_PER_MINUTE:
            wait_time=60-(now-request_times[0])
            logging.warning(f"Rate limit hit. Sleeping {wait_time:.2f}s.")
            time.sleep(wait_time)
        request_times.append(time.time())

def cache_key(url:str,params:Dict[str,Any],headers:Dict[str,str])->str:
    key_str=json.dumps({'url':url,'params':params,'headers':headers},sort_keys=True)
    return hashlib.md5(key_str.encode('utf-8')).hexdigest()

def load_from_cache(key:str)->Optional[Dict[str,Any]]:
    with cache_lock:
        cache_path=CACHE_DIR/f"{key}.json"
        if cache_path.exists():
            mtime=cache_path.stat().st_mtime
            if (time.time()-mtime)<CACHE_EXPIRY_SECONDS:
                try:
                    with cache_path.open('r')as f:
                        data=json.load(f)
                    return data
                except Exception as e:
                    logging.error(f"Error reading cache {cache_path}: {e}")
                    return None
            else:
                cache_path.unlink(missing_ok=True)
        return None

def save_to_cache(key:str,data:Dict[str,Any]):
    with cache_lock:
        cache_path=CACHE_DIR/f"{key}.json"
        try:
            with cache_path.open('w')as f:
                json.dump(data,f)
        except Exception as e:
            logging.error(f"Error saving cache {cache_path}: {e}")

def get_json(url:str,params:Dict[str,Any]=None,headers:Dict[str,str]=None)->Optional[Dict[str,Any]]:
    if params is None: params={}
    if headers is None: headers={}

    ck=cache_key(url,params,headers)
    cached=load_from_cache(ck)
    if cached is not None:
        logging.debug(f"Returning cached response for {url} {params}")
        return cached

    check_rate_limit()
    token=get_next_token()
    if token:
        headers['Authorization']=f"Bearer {token}"

    backoff=INITIAL_BACKOFF
    for attempt in range(MAX_RETRIES):
        try:
            logging.debug(f"Fetching {url} attempt {attempt+1}, params={params}, headers={headers}")
            response=requests.get(url,params=params,headers=headers,timeout=10)
            if response.status_code==200:
                data=response.json()
                save_to_cache(ck,data)
                return data
            elif 500<=response.status_code<600:
                logging.warning(f"Server error {response.status_code} {url}. Retry in {backoff}s...")
                time.sleep(backoff)
                backoff*=2
            else:
                logging.error(f"HTTP {response.status_code} fetching {url}: {response.text}")
                return None
        except requests.exceptions.RequestException as e:
            logging.warning(f"Request exception {url}: {e}, retry in {backoff}s...")
            time.sleep(backoff)
            backoff*=2
        except Exception as e:
            logging.error(f"Unexpected error {url}: {e}")
            return None
    logging.error(f"All retries failed for {url}")
    return None
