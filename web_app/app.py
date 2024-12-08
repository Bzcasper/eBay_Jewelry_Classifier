import os
import logging
import json
import time
import subprocess
import threading
from pathlib import Path
from flask import Flask, render_template, request, redirect, url_for, flash, jsonify
from config import JEWELRY_CATEGORIES, SUBCATEGORIES

app=Flask(__name__)
app.secret_key='supersecretkey'

progress_dict={
    'scraping':0,
    'cleaning':0,
    'training':0,
    'inference':0
}
progress_lock=threading.Lock()

def run_command_with_progress(cmd,progress_key,steps=5):
    with progress_lock:
        progress_dict[progress_key]=0
    try:
        for i in range(steps):
            time.sleep(1)
            with progress_lock:
                progress_dict[progress_key]=int((i+1)/steps*100)
        subprocess.check_call(cmd)
        with progress_lock:
            progress_dict[progress_key]=100
    except subprocess.CalledProcessError as e:
        logging.error(f"Cmd {cmd} failed: {e}")
        with progress_lock:
            progress_dict[progress_key]=0
        flash(f"Error: {str(e)}",'danger')
    except Exception as e:
        logging.error(f"Unexpected error {cmd}: {e}")
        with progress_lock:
            progress_dict[progress_key]=0
        flash(f"Unexpected error: {str(e)}",'danger')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/run_scraping',methods=['GET','POST'])
def run_scraping():
    if request.method=='POST':
        category=request.form.get('category')
        subcat=request.form.get('subcategory')
        if not category or not subcat:
            flash("Select category & subcategory",'danger')
            return redirect(url_for('run_scraping'))
        cmd=['python3','scripts/data_collection.py',category,subcat]
        t=threading.Thread(target=run_command_with_progress,args=(cmd,'scraping',5))
        t.start()
        flash("Scraping started.",'info')
        return redirect(url_for('index'))
    return render_template('run_scraping.html',categories=JEWELRY_CATEGORIES,subcategories=SUBCATEGORIES)

@app.route('/clean_data',methods=['GET','POST'])
def clean_data():
    if request.method=='POST':
        cmd=['python3','scripts/data_cleaning.py']
        t=threading.Thread(target=run_command_with_progress,args=(cmd,'cleaning',5))
        t.start()
        flash("Data cleaning started",'info')
        return redirect(url_for('index'))
    return render_template('clean_data.html')

@app.route('/train',methods=['GET','POST'])
def train():
    if request.method=='POST':
        model_type=request.form.get('model_type','resnet')
        if model_type=='resnet':
            cmd=['deepspeed','--num_gpus=1','scripts/train_resnet50.py','--deepspeed_config','scripts/utils/deepspeed_config.json']
        else:
            cmd=['deepspeed','--num_gpus=1','scripts/fine_tune_gpt2.py','--deepspeed_config','scripts/utils/deepspeed_config.json']
        t=threading.Thread(target=run_command_with_progress,args=(cmd,'training',10))
        t.start()
        flash(f"{model_type} training started",'info')
        return redirect(url_for('index'))
    return render_template('train.html')

@app.route('/inference',methods=['GET','POST'])
def inference():
    if request.method=='POST':
        image_path=request.form.get('image_path')
        if not image_path or not Path(image_path).exists():
            flash("Valid image path needed",'danger')
            return redirect(url_for('inference'))
        cmd=['python3','scripts/combine_predictions.py',image_path]
        t=threading.Thread(target=run_command_with_progress,args=(cmd,'inference',5))
        t.start()
        flash("Inference started",'info')
        return redirect(url_for('index'))
    return render_template('inference.html')

@app.route('/progress/<task>')
def progress(task):
    with progress_lock:
        val=progress_dict.get(task,0)
    return jsonify({'progress':val})

if __name__=='__main__':
    app.run(host='0.0.0.0',port=5000)
