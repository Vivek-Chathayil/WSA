from django.urls import path
from . import views

app_name = 'analyzer'

urlpatterns = [
    # Main pages
    path('', views.index, name='index'),
    
    # Dataset analysis
    path('dataset/', views.dataset_analysis, name='dataset_analysis'),
    path('dataset-results/<int:dataset_id>/', views.dataset_results, name='dataset_results'),
    path('dataset/<int:dataset_id>/download/', views.download_excel_report, name='download_excel_report'),
    
    path('generate-chart/', views.generate_chart_api, name='generate_chart_api'),
    path('test-visualization/', views.test_visualization, name='test_visualization'),

]