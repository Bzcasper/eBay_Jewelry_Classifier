import torch
import torch.nn.functional as F
import logging
import json
from PIL import Image
from pathlib import Path
from typing import Optional, Tuple, List
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from torchvision import models, transforms
from config import MODELS_DIR, CLEANED_DIR
import pandas as pd
import statistics
import requests

CATEGORY_MAPPING={'ring':0,'necklace':1,'bracelet':2,'earring':3,'pendant':4,'wristwatch':5}
IDX_TO_CATEGORY={v:k for k,v in CATEGORY_MAPPING.items()}

def load_resnet_model(fallback=False):
    try:
        if not fallback:
            cdir=MODELS_DIR/'resnet50_lora_deepspeed_best'
            cpath=cdir/'resnet50_best.pth'
            logging.info(f"Loading LoRA-merged ResNet50 from {cpath}")
        else:
            cdir=MODELS_DIR/'resnet50'
            cpath=cdir/'best_model.pth'
            logging.info(f"Fallback: Loading baseline ResNet50 from {cpath}")
        model=models.resnet50(pretrained=False)
        num_ftrs=model.fc.in_features
        model.fc=torch.nn.Linear(num_ftrs,len(CATEGORY_MAPPING))
        model.load_state_dict(torch.load(cpath,map_location='cpu'))
        model.eval()
        logging.info("ResNet50 loaded.")
        return model
    except Exception as e:
        logging.error(f"Error loading ResNet50: {e}")
        if not fallback:
            logging.info("Trying fallback model for ResNet50...")
            return load_resnet_model(fallback=True)
        else:
            logging.critical("ResNet fallback also failed.")
            return None

def load_gpt2_model(fallback=False):
    try:
        if not fallback:
            cdir=MODELS_DIR/'gpt2_lora_deepspeed_best'
            logging.info(f"Loading LoRA-merged GPT-2 from {cdir}")
            model=GPT2LMHeadModel.from_pretrained(cdir)
            tokenizer=GPT2Tokenizer.from_pretrained(cdir)
        else:
            logging.info("Fallback: Loading baseline GPT-2")
            model=GPT2LMHeadModel.from_pretrained('gpt2')
            tokenizer=GPT2Tokenizer.from_pretrained('gpt2')
            tokenizer.pad_token=tokenizer.eos_token
        model.eval()
        logging.info("GPT-2 loaded.")
        return model,tokenizer
    except Exception as e:
        logging.error(f"Error loading GPT-2: {e}")
        if not fallback:
            logging.info("Trying fallback GPT-2...")
            return load_gpt2_model(fallback=True)
        else:
            logging.critical("GPT-2 fallback failed.")
            return None,None

def classify_image(model,image_path:str)->Tuple[str,float]:
    try:
        transform=transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
        ])
        img=Image.open(image_path).convert('RGB')
        input_tensor=transform(img).unsqueeze(0)
        with torch.no_grad():
            outputs=model(input_tensor)
            probs=F.softmax(outputs,dim=1)
            cidx=probs.argmax().item()
            confidence=probs[0,cidx].item()
        category=IDX_TO_CATEGORY.get(cidx,'unknown')
        return category,confidence
    except Exception as e:
        logging.error(f"Error classifying {image_path}: {e}")
        return 'unknown',0.0

def generate_description(model,tokenizer,category:str)->str:
    try:
        prompt=f"Generate a detailed description of a {category} piece of jewelry."
        inputs=tokenizer(prompt,return_tensors='pt')
        with torch.no_grad():
            outputs=model.generate(inputs.input_ids,max_length=50,temperature=0.7,top_p=0.9,do_sample=True,pad_token_id=tokenizer.eos_token_id)
        desc=tokenizer.decode(outputs[0],skip_special_tokens=True)
        return desc
    except Exception as e:
        logging.error(f"Error generating description for {category}: {e}")
        return f"This is a(n) {category} jewelry piece."

def fetch_sold_prices_local(category:str)->Optional[list]:
    sold_data_path=CLEANED_DIR/'sold_listings.csv'
    if not sold_data_path.exists():
        logging.warning("No local sold_listings.csv")
        return None
    try:
        df=pd.read_csv(sold_data_path)
        cat_df=df[(df['category'].str.lower()==category.lower())&(df['sold']==1)]
        if cat_df.empty:
            logging.info(f"No local sold data for {category}")
            return None
        prices=cat_df['price.current'].dropna().astype(float).tolist()
        if not prices:
            logging.info(f"No valid prices for {category}")
            return None
        return prices
    except Exception as e:
        logging.error(f"Error reading local sold data: {e}")
        return None

def fetch_sold_prices_api(category:str)->Optional[list]:
    api_url=f"https://example.com/api/sold_prices?category={category}"
    try:
        response=requests.get(api_url,timeout=10)
        response.raise_for_status()
        data=response.json()
        if 'prices' in data and isinstance(data['prices'],list) and data['prices']:
            return data['prices']
        logging.info(f"No prices from API for {category}")
        return None
    except Exception as e:
        logging.error(f"API error for {category}: {e}")
        return None

def fetch_sold_prices_for_category(category:str)->Optional[list]:
    prices=fetch_sold_prices_local(category)
    if prices: return prices
    logging.info("Local not found, trying API...")
    prices=fetch_sold_prices_api(category)
    if prices: return prices
    logging.warning(f"No sold prices from local or API for {category}")
    return None

def calculate_price_range(category:str)->Tuple[float,float]:
    baseline={
        'ring':(50,200),
        'necklace':(100,500),
        'bracelet':(30,150),
        'earring':(20,200),
        'pendant':(40,300),
        'wristwatch':(50,1000)
    }
    prices=fetch_sold_prices_for_category(category)
    if not prices:
        logging.info("Using baseline range.")
        return baseline.get(category,(50,200))
    prices_sorted=sorted(prices)
    median_price=statistics.median(prices_sorted)
    q1=statistics.median(prices_sorted[:len(prices_sorted)//2])
    q3=statistics.median(prices_sorted[len(prices_sorted)//2:])
    iqr=q3-q1
    lower=max(min(prices),median_price-1.5*iqr)
    upper=min(max(prices),median_price+1.5*iqr)
    if lower>=upper:
        lower,upper=min(prices),max(prices)
    logging.info(f"Price range for {category}: {lower}-{upper} from sold data.")
    return (float(lower),float(upper))

if __name__=='__main__':
    import sys
    logging.info("Starting combined predictions with enhanced pricing.")
    if len(sys.argv)<2:
        logging.error("Provide image path.")
        sys.exit(1)
    image_path=sys.argv[1]

    resnet_model=load_resnet_model()
    gpt2_model,gpt2_tokenizer=load_gpt2_model()
    if resnet_model is None or gpt2_model is None or gpt2_tokenizer is None:
        logging.critical("Models not loaded.")
        sys.exit(1)
    category,confidence=classify_image(resnet_model,image_path)
    description=generate_description(gpt2_model,gpt2_tokenizer,category)
    price_range=calculate_price_range(category)
    result={
        'category':category,
        'confidence':confidence,
        'description':description,
        'suggested_price_range':price_range
    }
    logging.info("Final result:")
    logging.info(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))
