from django.shortcuts import render, redirect, get_object_or_404
from django.contrib import messages
from django.http import JsonResponse, HttpResponse
from django.core.files.storage import default_storage
from django.core.files.base import ContentFile
from django.conf import settings
from django.http import FileResponse, Http404
from django.views.decorators.csrf import csrf_exempt
import os
import time
import logging
import tempfile
import shutil
import traceback
import pandas as pd

from .models import SentimentAnalysis, DatasetAnalysis
from .forms import TextAnalysisForm, DatasetUploadForm
from .utils import (
    get_client_ip, analyze_dataset_file, create_excel_report, load_model, create_sentiment_pie_chart,
)

logger = logging.getLogger(__name__)



def index(request):
    """Home page with text analysis form"""
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

            # Create a more reliable temporary file path
            temp_file_path = None
            
            try:
                # Ensure model is loaded
                load_model()
                
                # Create temporary file with proper extension
                file_ext = os.path.splitext(uploaded_file.name)[1].lower()
                
                # Validate file extension
                if file_ext not in ['.csv', '.xlsx', '.xls']:
                    messages.error(request, 'Invalid file type. Please upload a CSV or Excel file.')
                    return render(request, 'analyzer/dataset_analysis.html', {
                        'form': form,
                        'recent_datasets': DatasetAnalysis.objects.all()[:5]
                    })
                
                with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as temp_file:
                    # Copy uploaded file to temporary file
                    uploaded_file.seek(0)  # Reset file pointer
                    shutil.copyfileobj(uploaded_file, temp_file)
                    temp_file_path = temp_file.name

                logger.info(f"Temporary file created at: {temp_file_path}")
                logger.info(f"Original filename: {uploaded_file.name}")
                logger.info(f"File size: {uploaded_file.size} bytes")
                logger.info(f"Temp file size: {os.path.getsize(temp_file_path)} bytes")

                # Verify file was written correctly
                if os.path.getsize(temp_file_path) == 0:
                    logger.error("Temporary file is empty")
                    messages.error(request, 'Error: Uploaded file appears to be empty.')
                    return render(request, 'analyzer/dataset_analysis.html', {
                        'form': form,
                        'recent_datasets': DatasetAnalysis.objects.all()[:5]
                    })

                # Analyze the dataset
                logger.info("Starting dataset analysis...")
                results_df = analyze_dataset_file(temp_file_path, max_rows)
                
                if results_df is None or results_df.empty:
                    logger.error("Dataset could not be read or contains no valid data.")
                    messages.error(request, "Error: The uploaded file could not be processed. Please check the file format and content.")
                    return render(request, 'analyzer/dataset_analysis.html', {
                        'form': form,
                        'recent_datasets': DatasetAnalysis.objects.all()[:5]
                    })
                else:
                    logger.info(f"Dataset read successfully with {len(results_df)} rows.")

                # Calculate statistics
                sentiment_counts = results_df['dominant_sentiment'].value_counts()
                total_reviews = len(results_df)
                logger.info(f"Total reviews processed: {total_reviews}")
                logger.info(f"Sentiment counts: {sentiment_counts.to_dict()}")

                # Create Excel report
                excel_filename = create_excel_report(results_df, "dataset_analysis")
                
                if not excel_filename:
                    logger.warning("Failed to create Excel report")

                # Save dataset analysis
                dataset_analysis = DatasetAnalysis.objects.create(
                    name=f"Analysis of {uploaded_file.name}",
                    description=f"Sentiment analysis of {total_reviews} reviews from {uploaded_file.name}",
                    total_reviews=total_reviews,
                    positive_count=sentiment_counts.get('positive', 0),
                    negative_count=sentiment_counts.get('negative', 0),
                    neutral_count=sentiment_counts.get('neutral', 0),
                    original_filename=uploaded_file.name,
                    file_size=uploaded_file.size,
                    processing_time=getattr(results_df, 'attrs', {}).get('processing_time', 0),
                    excel_report=excel_filename,
                )
                logger.info(f"Dataset analysis saved with ID: {dataset_analysis.id}")

                # Store results in session for display
                try:
                    request.session['last_results'] = results_df.to_dict('split')
                    request.session['last_excel'] = excel_filename
                    request.session.modified = True
                    logger.info("Results stored in session")
                except Exception as session_error:
                    logger.warning(f"Could not store results in session: {session_error}")

                success_message = f'Dataset analysis completed successfully! Processed {total_reviews} reviews.'
                if excel_filename:
                    success_message += ' Excel report generated.'
                
                messages.success(request, success_message)
                return redirect('analyzer:dataset_results', dataset_id=dataset_analysis.id)
                
            except Exception as e:
                logger.error(f"Error processing dataset: {e}")
                logger.error(f"Traceback: {traceback.format_exc()}")
                messages.error(request, f'Error processing dataset: {str(e)[:100]}...')
                
            finally:
                # Clean up temporary file
                if temp_file_path and os.path.exists(temp_file_path):
                    try:
                        os.unlink(temp_file_path)
                        logger.info("Temporary file cleaned up")
                    except Exception as cleanup_error:
                        logger.error(f"Error cleaning up temporary file: {cleanup_error}")
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
    results_df = None
    excel_filename = None

    # Try to get results from session (if just analyzed)
    try:
        last_results = request.session.pop('last_results', None)
        last_excel = request.session.pop('last_excel', None)
        if last_results:
            results_df = pd.DataFrame(**last_results)
            excel_filename = last_excel
            logger.info("Retrieved results from session")
    except Exception as e:
        logger.warning(f"Could not retrieve results from session: {e}")

    # Prepare Excel download URL
    excel_download_url = None
    if dataset.excel_report:
        excel_path = os.path.join(settings.MEDIA_ROOT, dataset.excel_report)
        if os.path.exists(excel_path):
            excel_download_url = f"{settings.MEDIA_URL}{dataset.excel_report}"

    # Convert results_df to a list of tuples for display
    if results_df is not None:
        results_table = list(results_df[['Text', 'dominant_sentiment', 'confidence']].itertuples(index=False, name=None))
    else:
        results_table = None

    context = {
        'dataset': dataset,
        'results_table': results_table,
        'excel_filename': excel_filename,
        'excel_download_url': excel_download_url,
        'has_results': results_df is not None and not results_df.empty,
    }
    return render(request, 'analyzer/dataset_results.html', context)

def download_excel_report(request, dataset_id):
    """Download Excel report for dataset analysis"""
    dataset = get_object_or_404(DatasetAnalysis, id=dataset_id)
    if not dataset.excel_report:
        raise Http404("No report available.")
    file_path = os.path.join(settings.MEDIA_ROOT, dataset.excel_report)
    if not os.path.exists(file_path):
        raise Http404("Report file not found.")
    return FileResponse(open(file_path, 'rb'), as_attachment=True, filename=os.path.basename(file_path))

@csrf_exempt
def debug_dataset(request):
    """Debug endpoint to test dataset file reading"""
    if request.method == 'POST' and request.FILES.get('debug_file'):
        uploaded_file = request.FILES['debug_file']
        temp_file_path = None
        
        try:
            # Create temporary file
            file_ext = os.path.splitext(uploaded_file.name)[1].lower()
            with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as temp_file:
                shutil.copyfileobj(uploaded_file, temp_file)
                temp_file_path = temp_file.name
            
            # Try to read the file
            if file_ext == '.csv':
                df = pd.read_csv(temp_file_path)
            elif file_ext in ['.xlsx', '.xls']:
                df = pd.read_excel(temp_file_path)
            else:
                return JsonResponse({
                    'success': False,
                    'error': f'Unsupported file type: {file_ext}'
                })
            
            # Prepare debug information
            debug_info = {
                'success': True,
                'file_info': {
                    'name': uploaded_file.name,
                    'size': uploaded_file.size,
                    'type': file_ext
                },
                'dataframe_info': {
                    'shape': df.shape,
                    'columns': list(df.columns),
                    'dtypes': {col: str(dtype) for col, dtype in df.dtypes.items()},
                    'non_null_counts': df.count().to_dict(),
                    'memory_usage': df.memory_usage(deep=True).sum()
                },
                'sample_data': df.head(3).to_dict('records') if not df.empty else [],
                'text_columns_detected': []
            }
            
            # Detect potential text columns
            for col in df.columns:
                col_lower = str(col).lower().strip()
                sample_data = df[col].dropna().head(10)
                if len(sample_data) > 0:
                    string_count = sum(1 for x in sample_data if isinstance(x, str) and len(str(x).strip()) > 10)
                    if string_count > len(sample_data) * 0.5:
                        debug_info['text_columns_detected'].append({
                            'column': col,
                            'sample_text': str(sample_data.iloc[0])[:100] if len(sample_data) > 0 else ''
                        })
            
            logger.info(df.head().to_string())
            
            return JsonResponse(debug_info)
            
        except Exception as e:
            return JsonResponse({
                'success': False,
                'error': str(e),
                'traceback': traceback.format_exc()
            })
        finally:
            if temp_file_path and os.path.exists(temp_file_path):
                os.unlink(temp_file_path)
    
    return JsonResponse({
        'success': False,
        'error': 'No file provided or invalid request method'
    })

def health_check(request):
    """Health check endpoint"""
    try:
        # Check if model is loaded
        from .utils import tokenizer, model
        model_loaded = tokenizer is not None and model is not None
        
        # Check database connectivity
        db_connected = True
        try:
            SentimentAnalysis.objects.count()
        except Exception:
            db_connected = False
        
        # Check media directory
        media_accessible = os.path.exists(settings.MEDIA_ROOT) and os.access(settings.MEDIA_ROOT, os.W_OK)
        
        status = {
            'status': 'healthy' if all([model_loaded, db_connected, media_accessible]) else 'unhealthy',
            'model_loaded': model_loaded,
            'database_connected': db_connected,
            'media_directory_accessible': media_accessible,
            'recent_analyses_count': SentimentAnalysis.objects.count(),
            'recent_datasets_count': DatasetAnalysis.objects.count(),
        }
        
        return JsonResponse(status)
        
    except Exception as e:
        return JsonResponse({
            'status': 'error',
            'error': str(e)
        })

from django.shortcuts import render

def custom_404(request, exception):
    return render(request, 'analyzer/error.html', {
        'error_message': 'Page not found (404).'
    }, status=404)

def custom_500(request):
    return render(request, 'analyzer/error.html', {
        'error_message': 'Internal server error (500).'
    }, status=500)

# Add these additional views to your views.py file

from django.shortcuts import render
from django.http import Http404

def custom_404(request, exception):
    """Custom 404 error page"""
    return render(request, 'analyzer/404.html', status=404)

def custom_500(request):
    """Custom 500 error page"""
    return render(request, 'analyzer/500.html', status=500)

def test_csv_upload(request):
    """Test view for CSV upload debugging"""
    if request.method == 'POST':
        form = DatasetUploadForm(request.POST, request.FILES)
        if form.is_valid():
            # Just validate, don't process
            return JsonResponse({
                'success': True,
                'message': 'File validation successful',
                'filename': form.cleaned_data['dataset_file'].name,
                'size': form.cleaned_data['dataset_file'].size,
                'max_rows': form.cleaned_data['max_rows']
            })
        else:
            return JsonResponse({
                'success': False,
                'errors': form.errors
            })
    
    return render(request, 'analyzer/test_upload.html', {
        'form': DatasetUploadForm()
    })

def model_status(request):
    """Check model loading status"""
    try:
        from .utils import tokenizer, model, load_model
        
        if tokenizer is None or model is None:
            try:
                load_model()
                status = 'loaded'
                message = 'Model loaded successfully'
            except Exception as e:
                status = 'failed'
                message = f'Failed to load model: {str(e)}'
        else:
            status = 'loaded'
            message = 'Model already loaded'
        
        return JsonResponse({
            'status': status,
            'message': message,
            'model_name': 'cardiffnlp/twitter-roberta-base-sentiment'
        })
        
    except Exception as e:
        return JsonResponse({
            'status': 'error',
            'message': str(e)
        })

def clear_old_reports(request):
    """Admin view to clear old Excel reports"""
    if not request.user.is_staff:
        return JsonResponse({'error': 'Permission denied'}, status=403)
    
    try:
        import os
        from datetime import datetime, timedelta
        from django.conf import settings
        
        # Delete reports older than 30 days
        cutoff_date = timezone.now() - timedelta(days=30)
        old_datasets = DatasetAnalysis.objects.filter(
            created_at__lt=cutoff_date,
            excel_report__isnull=False
        )
        
        deleted_files = []
        for dataset in old_datasets:
            if dataset.excel_report:
                file_path = os.path.join(settings.MEDIA_ROOT, dataset.excel_report)
                if os.path.exists(file_path):
                    try:
                        os.unlink(file_path)
                        deleted_files.append(dataset.excel_report)
                    except OSError:
                        pass
                dataset.excel_report = None
                dataset.save()
        
        return JsonResponse({
            'success': True,
            'deleted_files': len(deleted_files),
            'cleaned_datasets': old_datasets.count()
        })
        
    except Exception as e:
        return JsonResponse({
            'success': False,
            'error': str(e)
        })


# Updated dataset_results view
def dataset_results(request, dataset_id):
    """Display dataset analysis results with visualizations"""
    dataset = get_object_or_404(DatasetAnalysis, id=dataset_id)
    results_df = None
    excel_filename = None
    charts = {}

    # Try to get results from session (if just analyzed)
    try:
        last_results = request.session.pop('last_results', None)
        last_excel = request.session.pop('last_excel', None)
        if last_results:
            results_df = pd.DataFrame(**last_results)
            excel_filename = last_excel
            logger.info("Retrieved results from session")
            
            # Generate visualizations
            try:
                charts = create_sentiment_pie_chart(results_df)
                logger.info("Visualizations created successfully")
            except Exception as viz_error:
                logger.error(f"Error creating visualizations: {viz_error}")
                charts = {}
                
    except Exception as e:
        logger.warning(f"Could not retrieve results from session: {e}")

    # Prepare Excel download URL
    excel_download_url = None
    if dataset.excel_report:
        excel_path = os.path.join(settings.MEDIA_ROOT, dataset.excel_report)
        if os.path.exists(excel_path):
            excel_download_url = f"{settings.MEDIA_URL}{dataset.excel_report}"

    # Convert results_df to a list of tuples for display
    if results_df is not None:
        results_table = list(results_df[['Text', 'dominant_sentiment', 'confidence']].itertuples(index=False, name=None))
        
        # Calculate additional statistics
        stats = {
            'avg_confidence': results_df['confidence'].mean(),
            'highest_confidence': results_df['confidence'].max(),
            'lowest_confidence': results_df['confidence'].min(),
            'sentiment_counts': results_df['dominant_sentiment'].value_counts().to_dict()
        }
    else:
        results_table = None
        stats = {}

    context = {
        'dataset': dataset,
        'results_table': results_table,
        'excel_filename': excel_filename,
        'excel_download_url': excel_download_url,
        'has_results': results_df is not None and not results_df.empty,
        'stats': stats,
        # Visualization charts
        'pie_chart': charts,
    }
    return render(request, 'analyzer/dataset_results.html', context)


# New API endpoint for generating charts
def generate_chart_api(request):
    """API endpoint to generate charts for existing dataset"""
    if request.method == 'POST':
        try:
            dataset_id = request.POST.get('dataset_id')
            chart_type = request.POST.get('chart_type', 'pie')
            
            dataset = get_object_or_404(DatasetAnalysis, id=dataset_id)
            
            # This would require storing the results data or regenerating it
            # For now, return an error message
            return JsonResponse({
                'success': False,
                'message': 'Chart generation requires recent analysis data in session'
            })
            
        except Exception as e:
            return JsonResponse({
                'success': False,
                'error': str(e)
            })
    
    return JsonResponse({
        'success': False,
        'error': 'Invalid request method'
    })

# Utility view to test visualization functions
def test_visualization(request):
    """Test view for visualization functions"""
    try:
        # Create sample data
        import pandas as pd
        import numpy as np
        
        np.random.seed(42)
        sample_data = {
            'Text': [f'Sample text {i}' for i in range(100)],
            'negative_score': np.random.random(100) * 0.4,
            'neutral_score': np.random.random(100) * 0.3,
            'positive_score': np.random.random(100) * 0.6,
            'dominant_sentiment': np.random.choice(['positive', 'negative', 'neutral'], 100, p=[0.5, 0.3, 0.2]),
            'confidence': np.random.random(100) * 0.4 + 0.6
        }
        
        results_df = pd.DataFrame(sample_data)
        
        # Generate all charts
        charts =create_sentiment_pie_chart(results_df)
        
        context = {
            'charts': charts,
            'sample_data_size': len(results_df),
            'has_charts': bool(charts)
        }
        
        return render(request, 'analyzer/test_visualization.html', context)
        
    except Exception as e:
        logger.error(f"Error in test visualization: {e}")
        return JsonResponse({
            'success': False,
            'error': str(e)
        })