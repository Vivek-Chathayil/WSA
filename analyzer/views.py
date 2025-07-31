from django.shortcuts import render, redirect, get_object_or_404
from django.contrib import messages
from django.http import JsonResponse, FileResponse, Http404
from django.conf import settings
from django.views.decorators.csrf import csrf_exempt
import os
import logging
import tempfile
import shutil
import traceback
import pandas as pd
from .models import SentimentAnalysis, DatasetAnalysis
from .forms import DatasetUploadForm
from .utils import (
    get_client_ip, analyze_dataset_file, create_excel_report, load_model, create_sentiment_pie_chart,
)

logger = logging.getLogger(__name__)

def index(request):
    return render(request, 'analyzer/index.html')

def results(request, analysis_id):
    analysis = get_object_or_404(SentimentAnalysis, id=analysis_id)
    return render(request, 'analyzer/results.html', {'analysis': analysis})

def dataset_analysis(request):
    if request.method == 'POST':
        form = DatasetUploadForm(request.POST, request.FILES)
        if form.is_valid():
            uploaded_file = form.cleaned_data['dataset_file']
            max_rows = form.cleaned_data['max_rows']
            keyword = form.cleaned_data.get('keyword', '').strip()
            temp_file_path = None
            try:
                load_model()
                file_ext = os.path.splitext(uploaded_file.name)[1].lower()
                if file_ext not in ['.csv', '.xlsx', '.xls']:
                    messages.error(request, 'Invalid file type. Please upload a CSV or Excel file.')
                    return render(request, 'analyzer/dataset_analysis.html', {'form': form, 'recent_datasets': DatasetAnalysis.objects.all()[:5]})
                with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as temp_file:
                    uploaded_file.seek(0)
                    shutil.copyfileobj(uploaded_file, temp_file)
                    temp_file_path = temp_file.name
                if os.path.getsize(temp_file_path) == 0:
                    logger.error("Temporary file is empty")
                    messages.error(request, 'Error: Uploaded file appears to be empty.')
                    return render(request, 'analyzer/dataset_analysis.html', {'form': form, 'recent_datasets': DatasetAnalysis.objects.all()[:5]})
                if file_ext == '.csv':
                    df = pd.read_csv(temp_file_path)
                else:
                    df = pd.read_excel(temp_file_path)
                possible_text_columns = [
                    'text', 'review', 'comment', 'content', 'message', 'description', 'feedback', 'opinion', 'tweet', 'post', 'body', 'summary'
                ]
                text_col = next((col for col in df.columns if str(col).lower().strip() in possible_text_columns), None)
                if not text_col:
                    for col in df.columns:
                        sample_data = df[col].dropna().head(10)
                        if len(sample_data) > 0:
                            string_count = sum(1 for x in sample_data if isinstance(x, str) and len(str(x).strip()) > 10)
                            if string_count > len(sample_data) * 0.7:
                                text_col = col
                                break
                if not text_col:
                    logger.error("No text column found for keyword filtering.")
                    messages.error(request, "No text column found for keyword filtering.")
                    return render(request, 'analyzer/dataset_analysis.html', {'form': form, 'recent_datasets': DatasetAnalysis.objects.all()[:5]})
                if keyword:
                    df = df[df[text_col].astype(str).str.contains(keyword, case=False, na=False)]
                    if df.empty:
                        logger.error(f"No rows found containing keyword: {keyword}")
                        messages.error(request, f"No rows found containing the keyword: '{keyword}'")
                        return render(request, 'analyzer/dataset_analysis.html', {'form': form, 'recent_datasets': DatasetAnalysis.objects.all()[:5]})
                    if file_ext == '.csv':
                        df.to_csv(temp_file_path, index=False)
                    else:
                        df.to_excel(temp_file_path, index=False)
                results_df = analyze_dataset_file(temp_file_path, max_rows)
                if results_df is None or results_df.empty:
                    logger.error("Dataset could not be read or contains no valid data.")
                    messages.error(request, "Error: The uploaded file could not be processed. Please check the file format and content.")
                    return render(request, 'analyzer/dataset_analysis.html', {'form': form, 'recent_datasets': DatasetAnalysis.objects.all()[:5]})
                pie_chart = create_sentiment_pie_chart(results_df)
                sentiment_counts = results_df['dominant_sentiment'].value_counts()
                total_reviews = len(results_df)
                excel_filename = create_excel_report(results_df, "dataset_analysis")
                if not excel_filename:
                    logger.warning("Failed to create Excel report")
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
                try:
                    request.session['last_results'] = results_df.to_dict('split')
                    request.session['last_excel'] = excel_filename
                    request.session.modified = True
                except Exception as session_error:
                    logger.warning(f"Could not store results in session: {session_error}")
                messages.success(request, f'Dataset analysis completed successfully! Processed {total_reviews} reviews.' + (' Excel report generated.' if excel_filename else ''))
                return redirect('analyzer:dataset_results', dataset_id=dataset_analysis.id)
            except Exception as e:
                logger.error(f"Error processing dataset: {e}")
                logger.error(f"Traceback: {traceback.format_exc()}")
                messages.error(request, f'Error processing dataset: {str(e)[:100]}...')
            finally:
                if temp_file_path and os.path.exists(temp_file_path):
                    try:
                        os.unlink(temp_file_path)
                    except Exception as cleanup_error:
                        logger.error(f"Error cleaning up temporary file: {cleanup_error}")
    else:
        form = DatasetUploadForm()
    return render(request, 'analyzer/dataset_analysis.html', {'form': form, 'recent_datasets': DatasetAnalysis.objects.all()[:5]})

def dataset_results(request, dataset_id):
    dataset = get_object_or_404(DatasetAnalysis, id=dataset_id)
    results_df = None
    excel_filename = None
    charts = {}
    try:
        last_results = request.session.pop('last_results', None)
        last_excel = request.session.pop('last_excel', None)
        if last_results:
            results_df = pd.DataFrame(**last_results)
            excel_filename = last_excel
            try:
                charts = create_sentiment_pie_chart(results_df)
            except Exception as viz_error:
                logger.error(f"Error creating visualizations: {viz_error}")
                charts = {}
    except Exception as e:
        logger.warning(f"Could not retrieve results from session: {e}")
    excel_download_url = None
    if dataset.excel_report:
        excel_path = os.path.join(settings.MEDIA_ROOT, dataset.excel_report)
        if os.path.exists(excel_path):
            excel_download_url = f"{settings.MEDIA_URL}{dataset.excel_report}"
    if results_df is not None:
        results_table = list(results_df[['Text', 'dominant_sentiment', 'confidence']].itertuples(index=False, name=None))
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
        'pie_chart': charts,
    }
    return render(request, 'analyzer/dataset_results.html', context)

def download_excel_report(request, dataset_id):
    dataset = get_object_or_404(DatasetAnalysis, id=dataset_id)
    if not dataset.excel_report:
        raise Http404("No report available.")
    file_path = os.path.join(settings.MEDIA_ROOT, dataset.excel_report)
    if not os.path.exists(file_path):
        raise Http404("Report file not found.")
    return FileResponse(open(file_path, 'rb'), as_attachment=True, filename=os.path.basename(file_path))

@csrf_exempt
def debug_dataset(request):
    if request.method == 'POST' and request.FILES.get('debug_file'):
        uploaded_file = request.FILES['debug_file']
        temp_file_path = None
        try:
            file_ext = os.path.splitext(uploaded_file.name)[1].lower()
            with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as temp_file:
                shutil.copyfileobj(uploaded_file, temp_file)
                temp_file_path = temp_file.name
            if file_ext == '.csv':
                df = pd.read_csv(temp_file_path)
            elif file_ext in ['.xlsx', '.xls']:
                df = pd.read_excel(temp_file_path)
            else:
                return JsonResponse({'success': False, 'error': f'Unsupported file type: {file_ext}'})
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
            for col in df.columns:
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
            return JsonResponse({'success': False, 'error': str(e), 'traceback': traceback.format_exc()})
        finally:
            if temp_file_path and os.path.exists(temp_file_path):
                os.unlink(temp_file_path)
    return JsonResponse({'success': False, 'error': 'No file provided or invalid request method'})

def health_check(request):
    try:
        from .utils import tokenizer, model
        model_loaded = tokenizer is not None and model is not None
        db_connected = True
        try:
            SentimentAnalysis.objects.count()
        except Exception:
            db_connected = False
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
        return JsonResponse({'status': 'error', 'error': str(e)})

def custom_404(request, exception):
    return render(request, 'analyzer/error.html', {'error_message': 'Page not found (404).'}, status=404)

def custom_500(request):
    return render(request, 'analyzer/error.html', {'error_message': 'Internal server error (500).'}, status=500)
