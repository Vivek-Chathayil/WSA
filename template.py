import os
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s: %(message)s'
)

# Project configuration
project_name = "sentiment_analyzer"
app_name = "analyzer"

# Complete Django project structure based on your sentiment analysis app
list_of_files = [
    # Root level files
    "manage.py",
    "requirements.txt",
    "README.md",
    ".gitignore",
    ".env.example",
    
    # Main project directory
    f"{project_name}/__init__.py",
    f"{project_name}/settings.py",
    f"{project_name}/urls.py",
    f"{project_name}/wsgi.py",
    f"{project_name}/asgi.py",
    
    # Django app structure
    f"{app_name}/__init__.py",
    f"{app_name}/admin.py",
    f"{app_name}/apps.py",
    f"{app_name}/models.py",
    f"{app_name}/views.py",
    f"{app_name}/urls.py",
    f"{app_name}/forms.py",
    f"{app_name}/utils.py",
    f"{app_name}/tests.py",
    f"{app_name}/migrations/__init__.py",
    
    # Templates directory structure
    f"{app_name}/templates/{app_name}/base.html",
    f"{app_name}/templates/{app_name}/index.html",
    f"{app_name}/templates/{app_name}/results.html",
    f"{app_name}/templates/{app_name}/dataset_analysis.html",
    f"{app_name}/templates/{app_name}/dataset_results.html",
    f"{app_name}/templates/{app_name}/error.html",
    
    # Static files structure
    "static/css/style.css",
    "static/js/main.js",
    "static/js/charts.js",
    "static/images/.gitkeep",
    
    # Media directory
    "media/.gitkeep",
    "media/uploads/.gitkeep",
    "media/reports/.gitkeep",
    
    # Logs directory
    "logs/.gitkeep",
]

def create_django_project_structure():
    """Create the complete Django project structure"""
    
    logging.info(f"Starting creation of Django project: {project_name}")
    logging.info(f"Main app name: {app_name}")
    logging.info(f"Total files to create: {len(list_of_files)}")
    
    created_files = 0
    created_directories = 0
    existing_files = 0
    
    for filepath in list_of_files:
        filepath = Path(filepath)
        filedir, filename = os.path.split(filepath)
        
        # Create directory if it doesn't exist
        if filedir != "":
            if not os.path.exists(filedir):
                os.makedirs(filedir, exist_ok=True)
                created_directories += 1
                logging.info(f"Created directory: {filedir}")
        
        # Create file if it doesn't exist or is empty
        if (not os.path.exists(filepath)) or (os.path.getsize(filepath) == 0):
            with open(filepath, 'w', encoding='utf-8') as f:
                # Add appropriate content based on file type
                content = get_file_content(filepath, filename)
                if content:
                    f.write(content)
                
            created_files += 1
            logging.info(f"Created file: {filepath}")
        else:
            existing_files += 1
            logging.info(f"File already exists: {filename}")
    
    # Print summary
    logging.info("=" * 60)
    logging.info("PROJECT STRUCTURE CREATION COMPLETED")
    logging.info("=" * 60)
    logging.info(f"Directories created: {created_directories}")
    logging.info(f"Files created: {created_files}")
    logging.info(f"Files already existing: {existing_files}")
    logging.info(f"Total files processed: {len(list_of_files)}")
    logging.info("=" * 60)
    
    return {
        'created_files': created_files,
        'created_directories': created_directories,
        'existing_files': existing_files,
        'total_files': len(list_of_files)
    }

def get_file_content(filepath, filename):
    """Return appropriate content for specific file types"""
    
    filepath_str = str(filepath)
    
    # Python files
    if filename.endswith('.py'):
        if filename == '__init__.py':
            return '"""Package initialization file"""\n'
        elif filename == 'manage.py':
            return '''#!/usr/bin/env python
"""Django's command-line utility for administrative tasks."""
import os
import sys

if __name__ == '__main__':
    os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'sentiment_analyzer.settings')
    try:
        from django.core.management import execute_from_command_line
    except ImportError as exc:
        raise ImportError(
            "Couldn't import Django. Are you sure it's installed and "
            "available on your PYTHONPATH environment variable? Did you "
            "forget to activate a virtual environment?"
        ) from exc
    execute_from_command_line(sys.argv)
'''
        elif filename == 'settings.py':
            return '''import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

SECRET_KEY = 'django-insecure-your-secret-key-change-this-in-production'

DEBUG = True

ALLOWED_HOSTS = ['localhost', '127.0.0.1', '0.0.0.0']

INSTALLED_APPS = [
    'django.contrib.admin',
    'django.contrib.auth',
    'django.contrib.contenttypes',
    'django.contrib.sessions',
    'django.contrib.messages',
    'django.contrib.staticfiles',
    'analyzer',
]

MIDDLEWARE = [
    'django.middleware.security.SecurityMiddleware',
    'django.contrib.sessions.middleware.SessionMiddleware',
    'django.middleware.common.CommonMiddleware',
    'django.middleware.csrf.CsrfViewMiddleware',
    'django.contrib.auth.middleware.AuthenticationMiddleware',
    'django.contrib.messages.middleware.MessageMiddleware',
    'django.middleware.clickjacking.XFrameOptionsMiddleware',
]

ROOT_URLCONF = 'sentiment_analyzer.urls'

TEMPLATES = [
    {
        'BACKEND': 'django.template.backends.django.DjangoTemplates',
        'DIRS': [BASE_DIR / 'templates'],
        'APP_DIRS': True,
        'OPTIONS': {
            'context_processors': [
                'django.template.context_processors.debug',
                'django.template.context_processors.request',
                'django.contrib.auth.context_processors.auth',
                'django.contrib.messages.context_processors.messages',
            ],
        },
    },
]

WSGI_APPLICATION = 'sentiment_analyzer.wsgi.application'

DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.sqlite3',
        'NAME': BASE_DIR / 'db.sqlite3',
    }
}

AUTH_PASSWORD_VALIDATORS = [
    {
        'NAME': 'django.contrib.auth.password_validation.UserAttributeSimilarityValidator',
    },
    {
        'NAME': 'django.contrib.auth.password_validation.MinimumLengthValidator',
    },
    {
        'NAME': 'django.contrib.auth.password_validation.CommonPasswordValidator',
    },
    {
        'NAME': 'django.contrib.auth.password_validation.NumericPasswordValidator',
    },
]

LANGUAGE_CODE = 'en-us'
TIME_ZONE = 'UTC'
USE_I18N = True
USE_TZ = True

STATIC_URL = '/static/'
STATICFILES_DIRS = [BASE_DIR / 'static']
STATIC_ROOT = BASE_DIR / 'staticfiles'

MEDIA_URL = '/media/'
MEDIA_ROOT = BASE_DIR / 'media'

DEFAULT_AUTO_FIELD = 'django.db.models.BigAutoField'

# Logging configuration
LOGGING = {
    'version': 1,
    'disable_existing_loggers': False,
    'handlers': {
        'file': {
            'level': 'INFO',
            'class': 'logging.FileHandler',
            'filename': BASE_DIR / 'logs' / 'django.log',
        },
        'console': {
            'level': 'INFO',
            'class': 'logging.StreamHandler',
        },
    },
    'loggers': {
        'analyzer': {
            'handlers': ['file', 'console'],
            'level': 'INFO',
            'propagate': True,
        },
    },
}
'''
        elif filename == 'urls.py' and 'sentiment_analyzer' in filepath_str:
            return '''from django.contrib import admin
from django.urls import path, include
from django.conf import settings
from django.conf.urls.static import static

urlpatterns = [
    path('admin/', admin.site.urls),
    path('', include('analyzer.urls')),
]

if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
    urlpatterns += static(settings.STATIC_URL, document_root=settings.STATIC_ROOT)
'''
        elif filename == 'urls.py' and app_name in filepath_str:
            return '''from django.urls import path
from . import views

app_name = 'analyzer'

urlpatterns = [
    path('', views.index, name='index'),
    path('analyze/', views.analyze_text, name='analyze_text'),
    path('results/<int:analysis_id>/', views.results, name='results'),
    path('dataset/', views.dataset_analysis, name='dataset_analysis'),
    path('dataset-results/<int:dataset_id>/', views.dataset_results, name='dataset_results'),
]
'''
        elif filename == 'wsgi.py':
            return '''"""
WSGI config for sentiment_analyzer project.

It exposes the WSGI callable as a module-level variable named ``application``.

For more information on this file, see
https://docs.djangoproject.com/en/4.2/howto/deployment/wsgi/
"""

import os

from django.core.wsgi import get_wsgi_application

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'sentiment_analyzer.settings')

application = get_wsgi_application()
'''
        elif filename == 'asgi.py':
            return '''"""
ASGI config for sentiment_analyzer project.

It exposes the ASGI callable as a module-level variable named ``application``.

For more information on this file, see
https://docs.djangoproject.com/en/4.2/howto/deployment/asgi/
"""

import os

from django.core.asgi import get_asgi_application

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'sentiment_analyzer.settings')

application = get_asgi_application()
'''
        elif filename == 'models.py':
            return '''from django.db import models
from django.utils import timezone
from django.core.validators import MinValueValidator, MaxValueValidator

class SentimentAnalysis(models.Model):
    SENTIMENT_CHOICES = [
        ('positive', 'Positive'),
        ('negative', 'Negative'),
        ('neutral', 'Neutral'),
    ]
    
    text = models.TextField(help_text="Original text analyzed")
    summary = models.CharField(max_length=500, blank=True, help_text="Summary if available")
    
    # Sentiment scores (0.0 to 1.0)
    negative_score = models.FloatField(
        validators=[MinValueValidator(0.0), MaxValueValidator(1.0)]
    )
    neutral_score = models.FloatField(
        validators=[MinValueValidator(0.0), MaxValueValidator(1.0)]
    )
    positive_score = models.FloatField(
        validators=[MinValueValidator(0.0), MaxValueValidator(1.0)]
    )
    
    # Results
    dominant_sentiment = models.CharField(max_length=20, choices=SENTIMENT_CHOICES)
    confidence = models.FloatField(
        validators=[MinValueValidator(0.0), MaxValueValidator(1.0)]
    )
    
    # Metadata
    created_at = models.DateTimeField(default=timezone.now)
    ip_address = models.GenericIPAddressField(null=True, blank=True)
    
    class Meta:
        ordering = ['-created_at']
        indexes = [
            models.Index(fields=['dominant_sentiment']),
            models.Index(fields=['created_at']),
            models.Index(fields=['confidence']),
        ]
    
    def __str__(self):
        return f"{self.dominant_sentiment.title()} - {self.confidence:.1%} ({self.created_at.strftime('%Y-%m-%d %H:%M')})"
    
    @property
    def confidence_percentage(self):
        return f"{self.confidence:.1%}"
    
    @property
    def text_preview(self):
        return self.text[:100] + "..." if len(self.text) > 100 else self.text

class DatasetAnalysis(models.Model):
    name = models.CharField(max_length=200)
    description = models.TextField(blank=True)
    
    # Statistics
    total_reviews = models.IntegerField()
    positive_count = models.IntegerField(default=0)
    negative_count = models.IntegerField(default=0)
    neutral_count = models.IntegerField(default=0)
    
    # File information
    original_filename = models.CharField(max_length=255)
    file_size = models.BigIntegerField(help_text="File size in bytes")
    
    # Processing information
    processing_time = models.FloatField(help_text="Processing time in seconds", null=True, blank=True)
    created_at = models.DateTimeField(default=timezone.now)
    
    class Meta:
        ordering = ['-created_at']
    
    def __str__(self):
        return f"{self.name} - {self.total_reviews} reviews"
    
    @property
    def positive_percentage(self):
        return (self.positive_count / self.total_reviews * 100) if self.total_reviews > 0 else 0
    
    @property
    def negative_percentage(self):
        return (self.negative_count / self.total_reviews * 100) if self.total_reviews > 0 else 0
    
    @property
    def neutral_percentage(self):
        return (self.neutral_count / self.total_reviews * 100) if self.total_reviews > 0 else 0
'''
        elif filename == 'views.py':
            return '''from django.shortcuts import render, redirect, get_object_or_404
from django.contrib import messages
from django.http import JsonResponse, HttpResponse
from django.core.files.storage import default_storage
from django.core.files.base import ContentFile
import os
import time
import logging

from .models import SentimentAnalysis, DatasetAnalysis
from .forms import TextAnalysisForm, DatasetUploadForm
from .utils import analyze_single_text, analyze_dataset_file, create_excel_report, get_client_ip

logger = logging.getLogger(__name__)

def index(request):
    """Home page with text analysis form"""
    form = TextAnalysisForm()
    context = {
        'form': form,
        'recent_analyses': SentimentAnalysis.objects.all()[:5]
    }
    return render(request, 'analyzer/index.html', context)

def analyze_text(request):
    """Analyze single text input"""
    if request.method == 'POST':
        form = TextAnalysisForm(request.POST)
        if form.is_valid():
            text = form.cleaned_data['text']
            
            # Analyze the text
            result = analyze_single_text(text)
            
            if result:
                # Save to database
                analysis = SentimentAnalysis.objects.create(
                    text=result['text'],
                    negative_score=result['negative_score'],
                    neutral_score=result['neutral_score'],
                    positive_score=result['positive_score'],
                    dominant_sentiment=result['dominant_sentiment'],
                    confidence=result['confidence'],
                    ip_address=get_client_ip(request)
                )
                
                messages.success(request, 'Text analysis completed successfully!')
                return redirect('analyzer:results', analysis_id=analysis.id)
            else:
                messages.error(request, 'Error analyzing text. Please try again.')
    else:
        form = TextAnalysisForm()
    
    context = {
        'form': form,
        'recent_analyses': SentimentAnalysis.objects.all()[:5]
    }
    return render(request, 'analyzer/index.html', context)

def results(request, analysis_id):
    """Display analysis results"""
    analysis = get_object_or_404(SentimentAnalysis, id=analysis_id)
    context = {
        'analysis': analysis
    }
    return render(request, 'analyzer/results.html', context)

def dataset_analysis(request):
    """Dataset upload and analysis page"""
    if request.method == 'POST':
        form = DatasetUploadForm(request.POST, request.FILES)
        if form.is_valid():
            uploaded_file = form.cleaned_data['dataset_file']
            max_rows = form.cleaned_data['max_rows']
            
            # Save uploaded file temporarily
            file_path = default_storage.save(
                f'uploads/{uploaded_file.name}',
                ContentFile(uploaded_file.read())
            )
            full_file_path = default_storage.path(file_path)
            
            try:
                # Analyze the dataset
                results_df = analyze_dataset_file(full_file_path, max_rows)
                
                if results_df is not None and not results_df.empty:
                    # Calculate statistics
                    sentiment_counts = results_df['dominant_sentiment'].value_counts()
                    total_reviews = len(results_df)
                    
                    # Save dataset analysis
                    dataset_analysis = DatasetAnalysis.objects.create(
                        name=f"Analysis of {uploaded_file.name}",
                        description=f"Sentiment analysis of {total_reviews} reviews",
                        total_reviews=total_reviews,
                        positive_count=sentiment_counts.get('positive', 0),
                        negative_count=sentiment_counts.get('negative', 0),
                        neutral_count=sentiment_counts.get('neutral', 0),
                        original_filename=uploaded_file.name,
                        file_size=uploaded_file.size,
                        processing_time=getattr(results_df, 'attrs', {}).get('processing_time', 0)
                    )
                    
                    # Create Excel report
                    excel_filename = create_excel_report(results_df, "dataset_analysis")
                    
                    messages.success(request, f'Dataset analysis completed! Processed {total_reviews} reviews.')
                    return redirect('analyzer:dataset_results', dataset_id=dataset_analysis.id)
                else:
                    messages.error(request, 'Error processing dataset. Please check file format.')
            except Exception as e:
                logger.error(f"Error processing dataset: {e}")
                messages.error(request, 'Error processing dataset. Please try again.')
            finally:
                # Clean up uploaded file
                if default_storage.exists(file_path):
                    default_storage.delete(file_path)
    else:
        form = DatasetUploadForm()
    
    context = {
        'form': form,
        'recent_datasets': DatasetAnalysis.objects.all()[:5]
    }
    return render(request, 'analyzer/dataset_analysis.html', context)

def dataset_results(request, dataset_id):
    """Display dataset analysis results"""
    dataset = get_object_or_404(DatasetAnalysis, id=dataset_id)
    context = {
        'dataset': dataset
    }
    return render(request, 'analyzer/dataset_results.html', context)
'''
        elif filename == 'forms.py':
            return '''from django import forms
from django.core.validators import FileExtensionValidator

class TextAnalysisForm(forms.Form):
    text = forms.CharField(
        widget=forms.Textarea(attrs={
            'class': 'form-control form-control-lg',
            'rows': 6,
            'placeholder': 'Enter your text here for sentiment analysis...',
            'maxlength': '5000'
        }),
        label='Text to Analyze',
        max_length=5000,
        help_text='Enter up to 5000 characters for analysis'
    )
    
    def clean_text(self):
        text = self.cleaned_data['text']
        if len(text.strip()) < 10:
            raise forms.ValidationError("Text must be at least 10 characters long.")
        return text.strip()

class DatasetUploadForm(forms.Form):
    dataset_file = forms.FileField(
        widget=forms.FileInput(attrs={
            'class': 'form-control',
            'accept': '.csv'
        }),
        label='Upload CSV Dataset',
        help_text='Upload a CSV file with "Text" column for batch analysis (max 10MB)',
        validators=[FileExtensionValidator(allowed_extensions=['csv'])]
    )
    
    max_rows = forms.IntegerField(
        widget=forms.NumberInput(attrs={
            'class': 'form-control',
            'min': '1',
            'max': '1000',
            'value': '500'
        }),
        label='Maximum Rows to Process',
        min_value=1,
        max_value=1000,
        initial=500,
        help_text='Limit the number of rows to process (1-1000)'
    )
    
    def clean_dataset_file(self):
        file = self.cleaned_data['dataset_file']
        if file.size > 10 * 1024 * 1024:  # 10MB limit
            raise forms.ValidationError("File size must be less than 10MB.")
        return file
'''
        elif filename == 'utils.py':
            return '''from transformers import AutoTokenizer, AutoModelForSequenceClassification
from scipy.special import softmax
import pandas as pd
import os
import time
from datetime import datetime
import logging
import torch

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
        logger.error(f"Error in sentiment analysis: {e}")
        return None

def analyze_single_text(text):
    """
    Analyze a single text and return formatted results
    """
    result = polarity_scores_roberta(text)
    
    if not result:
        return None
    
    # Determine dominant sentiment
    max_key = max(result, key=result.get)
    sentiment_map = {
        'roberta_neg': 'negative', 
        'roberta_neu': 'neutral', 
        'roberta_pos': 'positive'
    }
    
    dominant_sentiment = sentiment_map[max_key]
    confidence = result[max_key]
    
    return {
        'text': text,
        'negative_score': result['roberta_neg'],
        'neutral_score': result['roberta_neu'],
        'positive_score': result['roberta_pos'],
        'dominant_sentiment': dominant_sentiment,
        'confidence': confidence
    }

def analyze_dataset_file(file_path, max_rows=500):
    """
    Analyze a CSV dataset file
    Returns DataFrame with results or None if error
    """
    try:
        start_time = time.time()
        
        # Read CSV file
        logger.info(f"Reading CSV file: {file_path}")
        df = pd.read_csv(file_path)
        logger.info(f"Loaded dataset with {len(df)} rows")
        
        # Check for required columns
        if 'Text' not in df.columns:
            logger.error("CSV file must contain 'Text' column")
            return None
        
        # Limit rows for processing
        if len(df) > max_rows:
            df = df.head(max_rows)
            logger.info(f"Limited to {max_rows} rows for processing")
        
        results_list = []
        processed_count = 0
        
        # Analyze each review
        for idx, row in df.iterrows():
            text = row.get('Text', '')
            summary = row.get('Summary', '')
            
            # Skip empty texts
            if not text or pd.isna(text) or len(str(text).strip()) == 0:
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
                confidence = result[max_key]
                
                result_entry = {
                    'Id': row.get('Id', idx),
                    'Summary': str(summary)[:500] if summary and not pd.isna(summary) else '',
                    'Text': str(text),
                    'negative_score': result['roberta_neg'],
                    'neutral_score': result['roberta_neu'],
                    'positive_score': result['roberta_pos'],
                    'dominant_sentiment': dominant_sentiment,
                    'confidence': confidence
                }
                results_list.append(result_entry)
                processed_count += 1
            
            # Log progress every 50 items
            if processed_count % 50 == 0:
                logger.info(f"Processed {processed_count} items...")
        
        processing_time = time.time() - start_time
        logger.info(f"Analysis completed in {processing_time:.2f} seconds")
        
        results_df = pd.DataFrame(results_list)
        results_df.attrs['processing_time'] = processing_time
        
        return results_df
    
    except Exception as e:
        logger.error(f"Error analyzing dataset: {e}")
        return None

def create_excel_report(results_df, filename_prefix="sentiment_analysis"):
    """
    Create Excel report from results DataFrame
    Returns filename if successful, None if error
    """
    try:
        from django.conf import settings
        
        # Create media directory if it doesn't exist
        os.makedirs(settings.MEDIA_ROOT, exist_ok=True)
        
        # Generate filename
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        excel_filename = f"{filename_prefix}_{timestamp}.xlsx"
        excel_path = os.path.join(settings.MEDIA_ROOT, excel_filename)
        
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
                    summary_data.append({
                        'Sentiment': sentiment.title(),
                        'Count': count,
                        'Percentage': f"{percentage:.1f}%"
                    })
                
                summary_df = pd.DataFrame(summary_data)
                summary_df.to_excel(writer, sheet_name='Summary_Stats', index=False)
        
        logger.info(f"Excel report created: {excel_filename}")
        return excel_filename
    
    except Exception as e:
        logger.error(f"Error creating Excel report: {e}")
        return None

def get_client_ip(request):
    """Get client IP address from request"""
    x_forwarded_for = request.META.get('HTTP_X_FORWARDED_FOR')
    if x_forwarded_for:
        ip = x_forwarded_for.split(',')[0]
    else:
        ip = request.META.get('REMOTE_ADDR')
    return ip

# Load model when module is imported
try:
    load_model()
except Exception as e:
    logger.warning(f"Could not load AI model on import: {e}")
'''
        elif filename == 'admin.py':
            return '''from django.contrib import admin
from .models import SentimentAnalysis, DatasetAnalysis

@admin.register(SentimentAnalysis)
class SentimentAnalysisAdmin(admin.ModelAdmin):
    list_display = ['id', 'dominant_sentiment', 'confidence_percentage', 'text_preview', 'created_at']
    list_filter = ['dominant_sentiment', 'created_at']
    search_fields = ['text', 'summary']
    readonly_fields = ['created_at', 'ip_address']
    ordering = ['-created_at']
    
    fieldsets = (
        ('Content', {
            'fields': ('text', 'summary')
        }),
        ('Analysis Results', {
            'fields': ('dominant_sentiment', 'confidence', 'negative_score', 'neutral_score', 'positive_score')
        }),
        ('Metadata', {
            'fields': ('created_at', 'ip_address'),
            'classes': ('collapse',)
        }),
    )

@admin.register(DatasetAnalysis)
class DatasetAnalysisAdmin(admin.ModelAdmin):
    list_display = ['name', 'total_reviews', 'positive_count', 'negative_count', 'neutral_count', 'created_at']
    list_filter = ['created_at']
    search_fields = ['name', 'description', 'original_filename']
    readonly_fields = ['created_at', 'processing_time']
    
    fieldsets = (
        ('Dataset Info', {
            'fields': ('name', 'description', 'original_filename', 'file_size')
        }),
        ('Analysis Results', {
            'fields': ('total_reviews', 'positive_count', 'negative_count', 'neutral_count')
        }),
        ('Processing Info', {
            'fields': ('processing_time', 'created_at'),
            'classes': ('collapse',)
        }),
    )
'''
        elif filename == 'apps.py':
            return '''from django.apps import AppConfig

class AnalyzerConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'analyzer'
    verbose_name = 'Sentiment Analyzer'
    
    def ready(self):
        # Import the model loading function to initialize models on startup
        try:
            from .utils import load_model
            load_model()
        except Exception as e:
            print(f"Warning: Could not load AI model on startup: {e}")
'''
    
    # HTML files
    elif filename.endswith('.html'):
        if filename == 'base.html':
            return '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{% block title %}Sentiment Analyzer{% endblock %}</title>
    {% load static %}
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" rel="stylesheet">
    <link href="{% static 'css/style.css' %}" rel="stylesheet">
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
</head>
<body>
    <nav class="navbar navbar-expand-lg navbar-dark bg-primary">
        <div class="container">
            <a class="navbar-brand" href="{% url 'analyzer:index' %}">Sentiment Analyzer</a>
            <div class="navbar-nav">
                <a class="nav-link" href="{% url 'analyzer:index' %}">Text Analysis</a>
                <a class="nav-link" href="{% url 'analyzer:dataset_analysis' %}">Dataset Analysis</a>
            </div>
        </div>
    </nav>

    <div class="container mt-4">
        {% if messages %}
            {% for message in messages %}
                <div class="alert alert-{{ message.tags }} alert-dismissible fade show" role="alert">
                    {{ message }}
                    <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
                </div>
            {% endfor %}
        {% endif %}

        {% block content %}
        {% endblock %}
    </div>

    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/js/bootstrap.bundle.min.js"></script>
    <script src="{% static 'js/main.js' %}"></script>
    {% block extra_js %}{% endblock %}
</body>
</html>
'''
        else:
            page_name = filename.split('.')[0].title().replace('_', ' ')
            template_content = f'''{{{{ extends "{app_name}/base.html" }}}}

{{{{ block title }}}}{page_name} - Sentiment Analyzer{{{{ endblock }}}}

{{{{ block content }}}}
<div class="row">
    <div class="col-12">
        <h1 class="mb-4">{page_name}</h1>
        <!-- TODO: Add content for {filename} -->
        <p>This is the {page_name.lower()} page.</p>
    </div>
</div>
{{{{ endblock }}}}
'''
            return template_content
    
    # CSS files
    elif filename.endswith('.css'):
        return '''/* Sentiment Analyzer Custom Styles */

body {
    background-color: #f8f9fa;
}

.sentiment-positive {
    color: #28a745;
    font-weight: bold;
}

.sentiment-negative {
    color: #dc3545;
    font-weight: bold;
}

.sentiment-neutral {
    color: #6c757d;
    font-weight: bold;
}

.confidence-bar {
    height: 20px;
    border-radius: 10px;
    background-color: #e9ecef;
    overflow: hidden;
}

.confidence-fill {
    height: 100%;
    transition: width 0.3s ease;
}

.analysis-card {
    border: none;
    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    transition: transform 0.2s ease;
}

.analysis-card:hover {
    transform: translateY(-2px);
    box-shadow: 0 4px 8px rgba(0,0,0,0.15);
}

.stats-card {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
    border-radius: 10px;
    padding: 1.5rem;
}

.chart-container {
    position: relative;
    height: 300px;
    margin: 20px 0;
}

.btn-primary {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    border: none;
}

.btn-primary:hover {
    background: linear-gradient(135deg, #5a6fd8 0%, #6a4190 100%);
}
'''
    
    # JavaScript files
    elif filename.endswith('.js'):
        if filename == 'main.js':
            return '''// Sentiment Analyzer Main JavaScript

document.addEventListener('DOMContentLoaded', function() {
    console.log('Sentiment Analyzer loaded');
    
    // Initialize tooltips
    var tooltipTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="tooltip"]'));
    var tooltipList = tooltipTriggerList.map(function (tooltipTriggerEl) {
        return new bootstrap.Tooltip(tooltipTriggerEl);
    });
    
    // Auto-hide alerts after 5 seconds
    setTimeout(function() {
        var alerts = document.querySelectorAll('.alert');
        alerts.forEach(function(alert) {
            var bsAlert = new bootstrap.Alert(alert);
            bsAlert.close();
        });
    }, 5000);
});

// Utility functions
function updateConfidenceBar(elementId, confidence) {
    const bar = document.getElementById(elementId);
    if (bar) {
        const fill = bar.querySelector('.confidence-fill');
        if (fill) {
            fill.style.width = (confidence * 100) + '%';
        }
    }
}

function formatConfidence(confidence) {
    return (confidence * 100).toFixed(1) + '%';
}
'''
        elif filename == 'charts.js':
            return '''// Chart.js configurations for sentiment analysis

function createSentimentChart(canvasId, data) {
    const ctx = document.getElementById(canvasId).getContext('2d');
    
    return new Chart(ctx, {
        type: 'doughnut',
        data: {
            labels: ['Positive', 'Negative', 'Neutral'],
            datasets: [{
                data: [data.positive, data.negative, data.neutral],
                backgroundColor: [
                    '#28a745',
                    '#dc3545',
                    '#6c757d'
                ],
                borderWidth: 2,
                borderColor: '#fff'
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    position: 'bottom',
                    labels: {
                        padding: 20,
                        usePointStyle: true
                    }
                }
            }
        }
    });
}

function createConfidenceChart(canvasId, scores) {
    const ctx = document.getElementById(canvasId).getContext('2d');
    
    return new Chart(ctx, {
        type: 'bar',
        data: {
            labels: ['Negative', 'Neutral', 'Positive'],
            datasets: [{
                label: 'Confidence Score',
                data: [scores.negative, scores.neutral, scores.positive],
                backgroundColor: [
                    'rgba(220, 53, 69, 0.8)',
                    'rgba(108, 117, 125, 0.8)',
                    'rgba(40, 167, 69, 0.8)'
                ],
                borderColor: [
                    '#dc3545',
                    '#6c757d',
                    '#28a745'
                ],
                borderWidth: 1
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                y: {
                    beginAtZero: true,
                    max: 1.0,
                    ticks: {
                        callback: function(value) {
                            return (value * 100) + '%';
                        }
                    }
                }
            },
            plugins: {
                legend: {
                    display: false
                }
            }
        }
    });
}
'''
        else:
            return '''// Auto-generated JavaScript file
// TODO: Add functionality

console.log('JavaScript file loaded');
'''
    
    # Configuration files
    elif filename == 'requirements.txt':
        return '''Django
transformers
torch
scipy
pandas
numpy
openpyxl
pillow
tqdm
scikit-learn
'''
    
    elif filename == '.gitignore':
        return '''# Django
*.log
*.pot
*.pyc
__pycache__/
local_settings.py
db.sqlite3
db.sqlite3-journal
media/
staticfiles/

# Environment
.env
.venv
env/
venv/
ENV/
env.bak/
venv.bak/

# IDE
.vscode/
.idea/
*.swp
*.swo
*~

# OS
.DS_Store
.DS_Store?
._*
.Spotlight-V100
.Trashes
ehthumbs.db
Thumbs.db

# Models
*.pkl
*.h5
*.model
models/

# Logs
logs/
*.log
'''
    
    elif filename == 'README.md':
        return f'''# {project_name.title()}

A Django-based sentiment analysis application with AI-powered text analysis capabilities.

## Features

- Individual text sentiment analysis
- Batch CSV file processing
- Beautiful web interface with Bootstrap
- Real-time results with Chart.js visualizations
- Excel report generation
- Admin panel for data management
- Responsive design
- Error handling and logging

## Setup Instructions

### 1. Create Project Environment
```bash
mkdir {project_name}
cd {project_name}
python -m venv venv
source venv/bin/activate  # On Windows: venv\\Scripts\\activate
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Database Setup
```bash
python manage.py makemigrations
python manage.py migrate
python manage.py createsuperuser
```

### 4. Run Development Server
```bash
python manage.py runserver
```

### 5. Access Application
- Main app: http://127.0.0.1:8000/
- Admin panel: http://127.0.0.1:8000/admin/

## Project Structure

```
{project_name}/
├── manage.py
├── requirements.txt
├── {project_name}/
│   ├── settings.py
│   ├── urls.py
│   └── ...
├── {app_name}/
│   ├── models.py
│   ├── views.py
│   ├── forms.py
│   ├── utils.py
│   └── templates/
└── static/
    ├── css/
    └── js/
```

## Usage

1. **Text Analysis**: Enter text in the main form for individual sentiment analysis
2. **Dataset Analysis**: Upload CSV files with 'Text' column for batch processing
3. **View Results**: See detailed analysis with confidence scores and visualizations
4. **Admin Panel**: Manage analysis records and view statistics

## API Endpoints

- `/` - Home page with text analysis form
- `/analyze/` - Process text analysis
- `/results/<id>/` - View analysis results
- `/dataset/` - Dataset upload page
- `/dataset-results/<id>/` - View dataset analysis results

## License

MIT License
'''
    elif filename == '.env.example':
            return '''# Django Environment Variables
DEBUG=True
SECRET_KEY=your-secret-key-here
ALLOWED_HOSTS=localhost,127.0.0.1

# Database (optional - defaults to SQLite)
# DATABASE_URL=sqlite:///db.sqlite3

# Media and Static Files
MEDIA_ROOT=media/
STATIC_ROOT=staticfiles/

# Logging
LOG_LEVEL=INFO
'''
    else:
            return ""

if __name__ == "__main__":
    try:
        result = create_django_project_structure()
        
        # Additional setup instructions
        print("\n" + "="*60)
        print("NEXT STEPS:")
        print("="*60)
        print("1. Navigate to project directory")
        print("2. Create virtual environment: python -m venv venv")
        print("3. Activate virtual environment:")
        print("   - Linux/Mac: source venv/bin/activate")
        print("   - Windows: venv\\Scripts\\activate")
        print("4. Install dependencies: pip install -r requirements.txt")
        print("5. Run migrations: python manage.py makemigrations")
        print("6. Apply migrations: python manage.py migrate")
        print("7. Create superuser: python manage.py createsuperuser")
        print("8. Run development server: python manage.py runserver")
        print("9. Visit: http://127.0.0.1:8000/")
        print("="*60)
        print(f"✓ Created {result['created_files']} files")
        print(f"✓ Created {result['created_directories']} directories")
        print(f"✓ Project '{project_name}' is ready!")
        print("="*60)
        
    except Exception as e:
        logging.error(f"Error creating project structure: {e}")
        raise