from transformers import AutoTokenizer, AutoModelForSequenceClassification
from scipy.special import softmax
import pandas as pd
import os
import time
from datetime import datetime
import logging
import torch
import traceback
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
import numpy as np
from io import BytesIO
import base64
from django.utils import timezone

# Set matplotlib to use non-interactive backend for web deployment
matplotlib.use('Agg')

logger = logging.getLogger(__name__)

# Global variables for model
MODEL_NAME = "cardiffnlp/twitter-roberta-base-sentiment"
tokenizer = None
model = None

def load_model():
    """Load the RoBERTa model and tokenizer"""
    global tokenizer, model
    
    if tokenizer is None or model is None:
        try:
            logger.info("Loading RoBERTa sentiment model...")
            tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
            model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
            logger.info("Model loaded successfully!")
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise

def polarity_scores_roberta(text):
    """
    Analyze sentiment of input text using RoBERTa model
    Returns dict with sentiment scores or None if error
    """
    global tokenizer, model
    
    if not tokenizer or not model:
        logger.error("Model not loaded")
        return None
        
    try:
        # Handle empty or invalid text
        if not text or pd.isna(text) or len(str(text).strip()) == 0:
            return None
            
        # Clean and prepare text
        text = str(text).strip()
        
        # Skip very short texts
        if len(text) < 3:
            return None
        
        # Tokenize and encode the text
        encoded_text = tokenizer(
            text, 
            return_tensors='pt', 
            truncation=True, 
            max_length=512,
            padding=True
        )
        
        # Get model output
        with torch.no_grad():  # Disable gradient computation for inference
            output = model(**encoded_text)
            scores = output.logits[0].detach().numpy()
            scores = softmax(scores)
        
        # Return scores dictionary
        return {
            'roberta_neg': float(scores[0]),
            'roberta_neu': float(scores[1]),
            'roberta_pos': float(scores[2])
        }
    
    except Exception as e:
        logger.error(f"Error in sentiment analysis for text '{text[:50]}...': {e}")
        return None

def create_sentiment_pie_chart(results_df, save_path=None):
    try:
        plt.style.use('default')
        fig, ax = plt.subplots(figsize=(5, 4))  # Smaller size

        sentiment_counts = results_df['dominant_sentiment'].value_counts()
        colors = {
            'positive': '#28a745',
            'negative': '#dc3545',
            'neutral': '#6c757d'
        }
        color_list = [colors.get(sentiment, '#17a2b8') for sentiment in sentiment_counts.index]

        wedges, texts, autotexts = ax.pie(
            sentiment_counts.values,
            labels=[f'{sentiment.title()}\n({count})' for sentiment, count in sentiment_counts.items()],
            colors=color_list,
            autopct='%1.1f%%',
            startangle=90,
            explode=(0.02, 0.02, 0.02),
            shadow=True,
            textprops={'fontsize': 11, 'fontweight': 'bold'}
        )

        for text in texts:
            text.set_fontsize(12)
            text.set_fontweight('bold')
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontsize(10)
            autotext.set_fontweight('bold')

        ax.set_title('Sentiment Distribution', fontsize=16, fontweight='bold', pad=20)
        total = len(results_df)
        ax.text(0, 0, f'Total\n{total}', ha='center', va='center',
                fontsize=16, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.9, edgecolor='gray'))
        ax.axis('equal')
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
            plt.close()
            return save_path
        else:
            buffer = BytesIO()
            plt.savefig(buffer, format='png', dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
            buffer.seek(0)
            image_base64 = base64.b64encode(buffer.getvalue()).decode()
            plt.close()
            return f"data:image/png;base64,{image_base64}"

    except Exception as e:
        logger.error(f"Error creating pie chart: {e}")
        if 'plt' in locals():
            plt.close()
        return None


def analyze_dataset_file(file_path, max_rows=500):

    try:
        start_time = time.time()
        _, ext = os.path.splitext(file_path)
        ext = ext.lower()
        logger.info(f"Attempting to read file: {file_path} (ext: {ext})")
        if ext == '.csv':
            df = pd.read_csv(file_path)
        elif ext in ['.xlsx', '.xls']:
            df = pd.read_excel(file_path)
        else:
            logger.error("Unsupported file type.")
            return None

        logger.info(f"File read: {df.shape[0]} rows, {df.shape[1]} columns. Columns: {list(df.columns)}")
        # Check for empty dataframe
        if df.empty:
            logger.error("Dataset is empty")
            return None

        # Better column detection - case insensitive and more flexible
        text_col = None
        possible_text_columns = [
            'text', 'review', 'comment', 'content', 'message', 'description', 
            'feedback', 'opinion', 'tweet', 'post', 'body', 'summary'
        ]
        
        # First try exact matches (case insensitive)
        for col in df.columns:
            col_clean = str(col).lower().strip()
            if col_clean in possible_text_columns:
                text_col = col
                logger.info(f"Found text column (exact match): {text_col}")
                break
        
        # If no exact match, try partial matches
        if not text_col:
            for col in df.columns:
                col_lower = str(col).lower().strip()
                for possible_col in possible_text_columns:
                    if possible_col in col_lower or col_lower in possible_col:
                        text_col = col
                        logger.info(f"Found text column (partial match): {text_col}")
                        break
                if text_col:
                    break
        
        # If still no match, use the first column that seems to contain text
        if not text_col:
            for col in df.columns:
                # Check if column contains mostly string data
                sample_data = df[col].dropna().head(10)
                if len(sample_data) > 0:
                    string_count = sum(1 for x in sample_data if isinstance(x, str) and len(str(x).strip()) > 10)
                    if string_count > len(sample_data) * 0.7:  # 70% of samples are strings
                        text_col = col
                        logger.info(f"Found text column (heuristic): {text_col}")
                        break
        
        if not text_col:
            logger.error(f"No text column found. Available columns: {list(df.columns)}")
            logger.error("File must contain a column with text data. Try renaming your text column to 'text' or 'review'")
            return None

        # Check if the text column has data
        non_empty_count = df[text_col].dropna().apply(lambda x: len(str(x).strip()) > 0).sum()
        logger.info(f"Text column '{text_col}' has {non_empty_count} non-empty entries")
        
        if non_empty_count == 0:
            logger.error(f"Text column '{text_col}' contains no valid text data")
            return None

        # Limit rows for processing
        original_rows = len(df)
        if len(df) > max_rows:
            df = df.head(max_rows)
            logger.info(f"Limited to {max_rows} rows for processing (original: {original_rows})")

        results_list = []
        processed_count = 0
        skipped_count = 0
        error_count = 0

        # Analyze each review
        for idx, row in df.iterrows():
            try:
                text = row.get(text_col, '')
                summary = row.get('Summary', row.get('summary', ''))

                # Skip empty texts - be more thorough
                if pd.isna(text) or not text or len(str(text).strip()) == 0:
                    skipped_count += 1
                    continue

                # Convert to string and clean
                text = str(text).strip()
                if len(text) < 3:  # Skip very short texts
                    skipped_count += 1
                    continue

                # Analyze the review text
                result = polarity_scores_roberta(text)

                if result:
                    # Determine dominant sentiment
                    max_key = max(result, key=result.get)
                    sentiment_map = {
                        'roberta_neg': 'negative',
                        'roberta_neu': 'neutral',
                        'roberta_pos': 'positive'
                    }
                    dominant_sentiment = sentiment_map[max_key]
                    confidence = result[max_key]*100  # Convert to percentage

                    result_entry = {
                        'Id': row.get('Id', row.get('id', row.get('ID', idx + 1))),
                        'Summary': str(summary)[:500] if summary and not pd.isna(summary) else '',
                        'Text': text,
                        'negative_score': round(result['roberta_neg'], 4),
                        'neutral_score': round(result['roberta_neu'], 4),
                        'positive_score': round(result['roberta_pos'], 4),
                        'dominant_sentiment': dominant_sentiment,
                        'confidence': round(confidence, 4)
                    }
                    results_list.append(result_entry)
                    processed_count += 1
                else:
                    error_count += 1
                    logger.debug(f"Failed to analyze text at row {idx}: {text[:50]}...")

                # Log progress every 1000 items
                if (processed_count + error_count ) % 1000 == 0:
                    logger.info(f"Progress: processed {processed_count}, errors {error_count}")
                    
            except Exception as e:
                logger.error(f"Error processing row {idx}: {e}")
                error_count += 1
                continue

        processing_time = time.time() - start_time
        logger.info(f"Analysis completed in {processing_time:.2f} seconds")
        logger.info(f"Total processed: {processed_count}")

        if not results_list:
            logger.error("No valid text entries found to analyze")
            return None

        results_df = pd.DataFrame(results_list)
        results_df.attrs['processing_time'] = processing_time

        return results_df

    except Exception as e:
        logger.error(f"Error analyzing dataset: {e}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        return None
    


def create_excel_report(results_df, filename_prefix="sentiment_analysis"):
    try:
        from django.conf import settings
        
        # Create media directory if it doesn't exist
        os.makedirs(settings.MEDIA_ROOT, exist_ok=True)
        
        # Generate filename
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        excel_filename = f"{filename_prefix}_{timestamp}.xlsx"
        excel_path = os.path.join(settings.MEDIA_ROOT, excel_filename)
        
        # Create charts directory for this report
        charts_dir = os.path.join(settings.MEDIA_ROOT, f"charts_{timestamp}")
        os.makedirs(charts_dir, exist_ok=True)
        
        # Generate visualizations
        charts = create_sentiment_pie_chart(results_df, charts_dir)
        
        with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
            # Main results sheet
            results_df.to_excel(writer, sheet_name='All_Results', index=False)
            
            # Summary statistics sheet
            if 'dominant_sentiment' in results_df.columns:
                sentiment_counts = results_df['dominant_sentiment'].value_counts()
                total_count = len(results_df)
                
                summary_data = []
                for sentiment, count in sentiment_counts.items():
                    percentage = (count / total_count) * 100
                    avg_confidence = results_df[results_df['dominant_sentiment'] == sentiment]['confidence'].mean()
                    summary_data.append({
                        'Sentiment': sentiment.title(),
                        'Count': count,
                        'Percentage': f"{percentage:.1f}%",
                        'Average_Confidence': f"{avg_confidence:.1f}%"
                    })
                
                summary_df = pd.DataFrame(summary_data)
                summary_df.to_excel(writer, sheet_name='Summary_Stats', index=False)
                
                # Add processing info
                processing_info = pd.DataFrame({
                    'Metric': ['Total Processed', 'Processing Time (seconds)', 'Average Confidence'],
                    'Value': [
                        total_count,
                        f"{getattr(results_df, 'attrs', {}).get('processing_time', 0):.2f}",
                        f"{results_df['confidence'].mean():.1f}%"
                    ]
                })
                processing_info.to_excel(writer, sheet_name='Processing_Info', index=False)
                
        logger.info(f"Excel report created: {excel_filename}")
        logger.info(f"Charts created in: {charts_dir}")
        return excel_filename
    
    except Exception as e:
        logger.error(f"Error creating Excel report: {e}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        return None
def get_client_ip(request):
    x_forwarded_for = request.META.get('HTTP_X_FORWARDED_FOR')
    if x_forwarded_for:
        ip = x_forwarded_for.split(',')[0].strip()
    else:
        ip = request.META.get('REMOTE_ADDR')
    return ip

try:
    load_model()
except Exception as e:
    logger.warning(f"Could not load AI model on import: {e}")
    logger.warning("Model will be loaded on first use")