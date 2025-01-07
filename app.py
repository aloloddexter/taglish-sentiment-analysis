from flask import Flask, request, render_template, jsonify, send_from_directory
import requests
import pandas as pd
import re
import time
from random import randint
import os
from transformers import AutoModelForSequenceClassification
from transformers import XLMRobertaTokenizer
import torch

app = Flask(__name__)

# Load pre-trained XLM-RoBERTa model and tokenizer
MODEL_DIR = './xlm-r/'
tokenizer = XLMRobertaTokenizer.from_pretrained(MODEL_DIR)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)

def get_ids_from_url(url):
    match = re.search(r"i\.(\d+)\.(\d+)", url)
    if match:
        shop_id, item_id = match.groups()
        return int(shop_id), int(item_id)
    else:
        raise ValueError("Invalid Shopee URL format")

def fetch_comments(shop_id, item_id, limit=50, offset=0, retries=3):
    url = f"https://shopee.ph/api/v2/item/get_ratings?itemid={item_id}&shopid={shop_id}&limit={limit}&offset={offset}"
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'
    }
    for attempt in range(retries):
        try:
            response = requests.get(url, headers=headers)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.HTTPError as http_err:
            print(f"HTTP error occurred: {http_err}")
            if attempt < retries - 1:
                time.sleep(2)
        except Exception as err:
            print(f"An error occurred: {err}")
            if attempt < retries - 1:
                time.sleep(2)
    return None

def extract_comments(data):
    comments = []
    if data and 'data' in data and 'ratings' in data['data']:
        for rating in data['data']['ratings']:
            comment_parts = []
            if 'tag_info' in rating:
                for tag in rating['tag_info']:
                    tag_text = f"{tag.get('tag_name', '')}: {tag.get('tag_value', '')}"
                    comment_parts.append(tag_text)
            main_comment = rating.get('comment', '').strip()
            if main_comment:
                comment_parts.append(main_comment)
            full_comment = "\n".join(comment_parts)
            comment = {
                'Username': rating.get('author_username', ''),
                'Rating': rating.get('rating_star', 0),
                'Date and Time': pd.to_datetime(rating.get('ctime', 0), unit='s').strftime('%Y-%m-%d %H:%M'),
                'Comment': full_comment
            }
            comments.append(comment)
    return comments

def clean_data(df):
    # Remove special characters from comments
    df['Comment'] = df['Comment'].apply(lambda x: re.sub(r'[^a-zA-Z0-9\s]', '', x))

    # Drop rows with empty comments
    df = df[df['Comment'].str.strip() != '']

    return df

def classify_sentiment(text):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    outputs = model(**inputs)
    prediction = torch.argmax(outputs.logits, dim=-1)
    return prediction.item()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/scrape', methods=['POST'])
def scrape():
    url = request.form.get('url')
    try:
        shop_id, item_id = get_ids_from_url(url)
    except ValueError as e:
        return jsonify({'error': str(e)})

    all_comments = []
    offset = 0
    limit = 50

    while True:
        data = fetch_comments(shop_id, item_id, limit=limit, offset=offset)
        if data is None:
            break
        comments = extract_comments(data)
        if not comments:
            break
        all_comments.extend(comments)

        if len(comments) < limit:
            break
        offset += limit
        time.sleep(randint(2, 5))

    if all_comments:
        df = pd.DataFrame(all_comments)

        # Clean data before analysis
        df = clean_data(df)

        # Perform sentiment analysis
        df['Sentiment'] = df['Comment'].apply(classify_sentiment)

        # Count sentiment occurrences
        sentiment_counts = df['Sentiment'].value_counts().to_dict()

        # sentiment distribution
        chart_data = [["Sentiment", "Count"]]
        for sentiment, count in sentiment_counts.items():
            sentiment_label = "Positive" if sentiment == 2 else "Neutral" if sentiment == 1 else "Negative"
            chart_data.append([sentiment_label, count])

        csv_filename = 'shopee_comments_formatted.csv'
        csv_filepath = os.path.join('static', csv_filename)
        df.to_csv(csv_filepath, index=False)

        return jsonify({
            'message': f'Successfully scraped and analyzed comments.',
            'filename': csv_filename,
            'chart_data': chart_data  # Send chart data
        })
    else:
        return jsonify({'error': 'No comments found or unable to fetch comments.'})

@app.route('/download/<filename>')
def download_file(filename):
    return send_from_directory('static', filename, as_attachment=True)

if __name__ == '__main__':
     app.run(debug=True, host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))
