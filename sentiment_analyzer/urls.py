# main/urls.py or your project's main urls.py file

from django.contrib import admin
from django.urls import path, include
from django.conf import settings
from django.conf.urls.static import static

urlpatterns = [
    path('admin/', admin.site.urls),
    path('', include('analyzer.urls')),
    # Add other URL patterns here
]

# Serve media files during development
if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)

# Custom error handlers (optional)
handler404 = 'analyzer.views.custom_404'
handler500 = 'analyzer.views.custom_500'