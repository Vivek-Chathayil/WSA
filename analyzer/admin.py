from django.contrib import admin
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
