from django import forms
from django.core.validators import FileExtensionValidator
from django.core.exceptions import ValidationError
import pandas as pd
import tempfile
import os

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
        
        # Remove extra whitespace
        text = text.strip()
        
        # Check minimum length
        if len(text) < 10:
            raise ValidationError("Text must be at least 10 characters long.")
        
        # Check for potentially harmful content (basic check)
        if text.lower().count('script') > 3 or '<' in text and '>' in text:
            raise ValidationError("Text contains potentially harmful content.")
        
        return text

class DatasetUploadForm(forms.Form):
    dataset_file = forms.FileField(
        widget=forms.FileInput(attrs={
            'class': 'form-control',
            'accept': '.csv,.xlsx,.xls',
        }),
        label='Upload CSV or Excel Dataset',
        help_text='Upload a CSV or Excel file with a text column for batch analysis (max 100MB)',
        validators=[FileExtensionValidator(allowed_extensions=['csv', 'xlsx', 'xls'])]
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
        help_text='Maximum number of rows to process from the dataset (default: 500, max: 1000)'
    )
    
    def clean_dataset_file(self):
        file = self.cleaned_data['dataset_file']

        # Check file size (500MB limit)
        if file.size > 500 * 1024 * 1024:
            raise ValidationError("File size must be less than 500MB.")

        # Check minimum file size (avoid empty files)
        if file.size < 100:  # Less than 100 bytes
            raise ValidationError("File appears to be empty or too small.")
        
        # Validate file extension
        file_ext = os.path.splitext(file.name)[1].lower()
        if file_ext not in ['.csv', '.xlsx', '.xls']:
            raise ValidationError("Invalid file type. Only CSV and Excel files are supported.")
        
        # Try to read the file to validate structure
        try:
            # Reset file pointer
            file.seek(0)
            
            # Create temporary file for validation
            with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as temp_file:
                # Write uploaded file to temp file
                for chunk in file.chunks():
                    temp_file.write(chunk)
                temp_file_path = temp_file.name
            
            try:
                # Try to read the file
                if file_ext == '.csv':
                    # Try different encodings
                    df = None
                    for encoding in ['utf-8', 'latin-1', 'cp1252']:
                        try:
                            df = pd.read_csv(temp_file_path, encoding=encoding, nrows=5)
                            break
                        except UnicodeDecodeError:
                            continue
                    
                    if df is None:
                        raise ValidationError("Could not read CSV file. Please check the file encoding.")
                        
                elif file_ext in ['.xlsx', '.xls']:
                    df = pd.read_excel(temp_file_path, nrows=5)
                
                # Check if dataframe is empty
                if df.empty:
                    raise ValidationError("The uploaded file contains no data.")
                
                # Check if there are any columns
                if len(df.columns) == 0:
                    raise ValidationError("The uploaded file has no columns.")
                
                # Check for potential text columns
                possible_text_columns = [
                    'text', 'review', 'comment', 'content', 'message', 
                    'description', 'feedback', 'opinion', 'tweet', 'post'
                ]
                
                text_column_found = False
                for col in df.columns:
                    col_lower = str(col).lower().strip()
                    if col_lower in possible_text_columns:
                        text_column_found = True
                        break
                
                # If no standard text column found, check for columns with text data
                if not text_column_found:
                    for col in df.columns:
                        sample_data = df[col].dropna().head(3)
                        if len(sample_data) > 0:
                            string_count = sum(1 for x in sample_data 
                                             if isinstance(x, str) and len(str(x).strip()) > 10)
                            if string_count > 0:
                                text_column_found = True
                                break
                
                if not text_column_found:
                    raise ValidationError(
                        "No text column found in the file. Please ensure your file contains "
                        "a column with text data (preferably named 'text', 'review', or 'comment')."
                    )
                
            finally:
                # Clean up temporary file
                if os.path.exists(temp_file_path):
                    os.unlink(temp_file_path)
            
            # Reset file pointer for later use
            file.seek(0)
            
        except ValidationError:
            raise  # Re-raise validation errors
        except Exception as e:
            raise ValidationError(f"Error validating file: {str(e)}")
        
        return file
    
    def clean_max_rows(self):
        max_rows = self.cleaned_data['max_rows']
        
        # Ensure reasonable limits based on file size if available
        if hasattr(self, 'cleaned_data') and 'dataset_file' in self.cleaned_data:
            file_size = self.cleaned_data['dataset_file'].size
            # Rough estimate: limit rows based on file size to prevent memory issues
            if file_size > 50 * 1024 * 1024 and max_rows > 10000:  # > 50MB
                raise ValidationError(
                    "For files larger than 50MB, please limit processing to 10,000 rows or less."
                )
        
        return max_rows