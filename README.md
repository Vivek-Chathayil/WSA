# WSA
AI driven Sentiment Analyzer Web App using Django



## 1. Project Purpose  
This project is an AI-driven Sentiment Analyzer Web App built with Django. It allows users to upload CSV or Excel datasets containing text data, performs sentiment analysis using a transformer-based model (RoBERTa), and provides results, statistics, and downloadable reports. The domain is NLP (Natural Language Processing) for sentiment analysis at scale.

## 2. Project Structure
- **WSA/**: Main Django project directory.
  - **analyzer/**: Core app for dataset upload, analysis, reporting, and web views.
    - `views.py`: Handles web requests, file uploads, analysis logic, and result rendering.
    - `utils.py`: Contains model loading, sentiment scoring, analysis, and reporting utilities.
    - `forms.py`: Django forms for validating and processing dataset uploads.
    - `models.py`: ORM models for storing analysis results and metadata.
    - `templates/analyzer/`: HTML templates for UI (dataset upload, results, errors, etc.).
    - `static/`: JS and CSS for frontend (charts, styles).
    - `tests.py`: Placeholder for tests (should be expanded).
  - **sentiment_analyzer/**: Django project settings, URLs, and WSGI/ASGI entry points.
  - **media/**: Stores generated Excel reports and uploaded files.
  - **logs/**: Application and Django logs.
  - **requirements.txt**: Python dependencies.
  - **manage.py**: Django management script.

## 3. Test Strategy
- **Framework**: Django's built-in test framework (unittest style).
- **Location**: Tests should be placed in `analyzer/tests.py` or split into multiple files as the app grows.
- **Naming**: Test methods should start with `test_` and classes with `Test`.
- **Mocking**: Use `unittest.mock` for mocking file uploads, model inference, and external dependencies.
- **Philosophy**: Focus on unit tests for forms, utils, and views logic. Add integration tests for end-to-end dataset analysis. Aim for high coverage of data validation, error handling, and model inference.

## 4. Code Style
- **Language**: Python 3.8+ (Django 4.x compatible).
- **Typing**: Use type hints where practical, especially in utility functions.
- **Async**: Not used; Django views are synchronous.
- **Naming**: 
  - Classes: `CamelCase` (e.g., `DatasetUploadForm`)
  - Functions/variables: `snake_case`
  - Constants: `UPPER_SNAKE_CASE`
- **Comments**: Use docstrings for public functions/classes. Inline comments for complex logic.
- **Error Handling**: Use try/except with logging. Show user-friendly messages in the UI. Clean up temp files in finally blocks.

## 5. Common Patterns
- **Reusable Utilities**: All model and analysis logic is in `utils.py` for reuse.
- **Design Patterns**: Factory pattern for model loading; property methods for computed model fields.
- **Idioms**: Use of Django ORM, form validation, and context-driven template rendering. Pandas for data processing.

## 6. Do's and Don'ts
- ✅ Validate all uploaded files for type, size, and structure.
- ✅ Use logging for all errors and important events.
- ✅ Clean up temporary files after processing.
- ✅ Use Django forms for all user input.
- ✅ Document new utilities and models.
- ❌ Do not hardcode file paths or secrets.
- ❌ Do not process unvalidated user files.
- ❌ Do not bypass Django's ORM for database access.
- ❌ Do not leave temp files or logs unrotated.

## 7. Tools & Dependencies
- **Django**: Web framework and ORM.
- **transformers**: For RoBERTa model inference.
- **torch**: Backend for model inference.
- **pandas, numpy, scipy**: Data processing and statistics.
- **openpyxl**: Excel report generation.
- **matplotlib**: Chart generation for reports.
- **tqdm**: (Optional) Progress bars for batch processing.
- **Setup**: Install dependencies with `pip install -r requirements.txt`. Run migrations before first use.

## 8. Other Notes
- LLMs should respect the file validation and error handling patterns in forms and views.
- Sentiment analysis is limited to English and the RoBERTa model's domain.
- Uploaded files and reports are stored in `media/` and may be cleaned up periodically.
- For large files, limit processing to a reasonable number of rows (default: 500).
- Security: Never expose the secret key or sensitive settings in code or logs.
