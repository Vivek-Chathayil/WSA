from django.db import models
from django.utils import timezone
from django.core.validators import MinValueValidator, MaxValueValidator
from django.urls import reverse
import os

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
        validators=[MinValueValidator(0.0), MaxValueValidator(1.0)],
        help_text="Negative sentiment score (0.0 to 1.0)"
    )
    neutral_score = models.FloatField(
        validators=[MinValueValidator(0.0), MaxValueValidator(1.0)],
        help_text="Neutral sentiment score (0.0 to 1.0)"
    )
    positive_score = models.FloatField(
        validators=[MinValueValidator(0.0), MaxValueValidator(1.0)],
        help_text="Positive sentiment score (0.0 to 1.0)"
    )
    
    # Results
    dominant_sentiment = models.CharField(
        max_length=20, 
        choices=SENTIMENT_CHOICES,
        help_text="The sentiment with the highest score"
    )
    confidence = models.FloatField(
        validators=[MinValueValidator(0.0), MaxValueValidator(1.0)],
        help_text="Confidence level of the dominant sentiment"
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
            models.Index(fields=['dominant_sentiment', 'created_at']),
        ]
        verbose_name = "Sentiment Analysis"
        verbose_name_plural = "Sentiment Analyses"
    
    def __str__(self):
        return f"{self.dominant_sentiment.title()} - {self.confidence:.1%} ({self.created_at.strftime('%Y-%m-%d %H:%M')})"
    
    @property
    def confidence_percentage(self):
        """Return confidence as percentage string"""
        return f"{self.confidence:.1%}"
    
    @property
    def text_preview(self):
        """Return truncated text for display"""
        return self.text[:100] + "..." if len(self.text) > 100 else self.text
    
    @property
    def sentiment_color(self):
        """Return CSS color class for sentiment"""
        color_map = {
            'positive': 'text-success',
            'negative': 'text-danger',
            'neutral': 'text-warning'
        }
        return color_map.get(self.dominant_sentiment, 'text-muted')
    
    @property
    def sentiment_icon(self):
        """Return icon class for sentiment"""
        icon_map = {
            'positive': 'fas fa-smile',
            'negative': 'fas fa-frown',
            'neutral': 'fas fa-meh'
        }
        return icon_map.get(self.dominant_sentiment, 'fas fa-question')
    
    def get_absolute_url(self):
        return reverse('analyzer:results', kwargs={'analysis_id': self.id})

class DatasetAnalysis(models.Model):
    name = models.CharField(max_length=200, help_text="Analysis name/description")
    description = models.TextField(blank=True, help_text="Detailed description of the analysis")
    chart_data = models.TextField(blank=True, null=True)
    # Statistics
    total_reviews = models.IntegerField(help_text="Total number of reviews processed")
    positive_count = models.IntegerField(default=0, help_text="Number of positive reviews")
    negative_count = models.IntegerField(default=0, help_text="Number of negative reviews")
    neutral_count = models.IntegerField(default=0, help_text="Number of neutral reviews")
    
    # File information
    original_filename = models.CharField(max_length=255, help_text="Original uploaded filename")
    file_size = models.BigIntegerField(help_text="File size in bytes")
    
    # Processing information
    processing_time = models.FloatField(
        help_text="Processing time in seconds", 
        null=True, 
        blank=True,
        validators=[MinValueValidator(0.0)]
    )
    created_at = models.DateTimeField(default=timezone.now)
    
    # Excel report file
    excel_report = models.CharField(
        max_length=255, 
        blank=True, 
        null=True,
        help_text="Path to generated Excel report file"
    )
    
    class Meta:
        ordering = ['-created_at']
        indexes = [
            models.Index(fields=['created_at']),
            models.Index(fields=['total_reviews']),
            models.Index(fields=['original_filename']),
        ]
        verbose_name = "Dataset Analysis"
        verbose_name_plural = "Dataset Analyses"
    
    def __str__(self):
        return f"{self.name} - {self.total_reviews} reviews ({self.created_at.strftime('%Y-%m-%d')})"
    
    @property
    def positive_percentage(self):
        """Return positive sentiment percentage"""
        return (self.positive_count / self.total_reviews * 100) if self.total_reviews > 0 else 0
    
    @property
    def negative_percentage(self):
        """Return negative sentiment percentage"""
        return (self.negative_count / self.total_reviews * 100) if self.total_reviews > 0 else 0
    
    @property
    def neutral_percentage(self):
        """Return neutral sentiment percentage"""
        return (self.neutral_count / self.total_reviews * 100) if self.total_reviews > 0 else 0
    
    @property
    def file_size_mb(self):
        """Return file size in MB"""
        return self.file_size / (1024 * 1024)
    
    @property
    def processing_time_formatted(self):
        """Return formatted processing time"""
        if not self.processing_time:
            return "Unknown"
        
        if self.processing_time < 60:
            return f"{self.processing_time:.1f} seconds"
        elif self.processing_time < 3600:
            minutes = self.processing_time / 60
            return f"{minutes:.1f} minutes"
        else:
            hours = self.processing_time / 3600
            return f"{hours:.1f} hours"
    
    @property
    def dominant_sentiment(self):
        """Return the dominant sentiment across all reviews"""
        counts = {
            'positive': self.positive_count,
            'negative': self.negative_count,
            'neutral': self.neutral_count
        }
        return max(counts, key=counts.get) if self.total_reviews > 0 else 'neutral'
    
    @property
    def sentiment_distribution(self):
        """Return sentiment distribution as a list of tuples"""
        return [
            ('Positive', self.positive_count, self.positive_percentage),
            ('Negative', self.negative_count, self.negative_percentage),
            ('Neutral', self.neutral_count, self.neutral_percentage),
        ]
    
    @property
    def has_excel_report(self):
        """Check if Excel report exists and is accessible"""
        if not self.excel_report:
            return False
        
        from django.conf import settings
        file_path = os.path.join(settings.MEDIA_ROOT, self.excel_report)
        return os.path.exists(file_path)
    
    def get_absolute_url(self):
        return reverse('analyzer:dataset_results', kwargs={'dataset_id': self.id})
    
    def get_download_url(self):
        return reverse('analyzer:download_excel_report', kwargs={'dataset_id': self.id})
    
    def delete(self, *args, **kwargs):
        """Override delete to clean up Excel report file"""
        if self.excel_report:
            from django.conf import settings
            file_path = os.path.join(settings.MEDIA_ROOT, self.excel_report)
            if os.path.exists(file_path):
                try:
                    os.unlink(file_path)
                except OSError:
                    pass  # File already deleted or permission issue
        
        super().delete(*args, **kwargs)