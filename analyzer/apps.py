from django.apps import AppConfig

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
