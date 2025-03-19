import pandas as pd
import numpy as np
import torch
from transformers import RobertaModel, RobertaTokenizer
import pickle
import json
from torch.utils.data import Dataset, DataLoader
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class PRDataset(Dataset):
    def __init__(self, texts, tokenizer, max_length=128):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx][:1000] if self.texts[idx] else ""
        encoded = self.tokenizer(
            text,
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoded['input_ids'].squeeze(),
            'attention_mask': encoded['attention_mask'].squeeze()
        }

def safe_int_conversion(value, default=0):
    try:
        if value is None:
            return default
        return int(value)
    except (ValueError, TypeError):
        return default

def extract_pr_features(pr):
    """Extract features from a single PR object"""
    return {
        'id': str(pr.get('id', 'unknown')),
        'body': str(pr.get('body', '')),
        'title': str(pr.get('title', '')),
        'line': safe_int_conversion(pr.get('line')),
        'author_association': str(pr.get('author_association', 'NONE')),
        'comments': safe_int_conversion(pr.get('comments')),
        'additions': safe_int_conversion(pr.get('additions')),
        'deletions': safe_int_conversion(pr.get('deletions')),
        'changed_files': safe_int_conversion(pr.get('changed_files'))
    }

def generate_embeddings_with_dataloader(texts, tokenizer, model, device, batch_size=8):
    """Generate embeddings for a list of texts using the provided model"""
    dataset = PRDataset(texts, tokenizer)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    embeddings_list = []
    
    for batch in dataloader:
        try:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            
            with torch.no_grad():
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                last_hidden_state = outputs[0]
                batch_embeddings = last_hidden_state[:, 0, :].cpu().numpy()
                embeddings_list.append(batch_embeddings)
                
                del outputs, input_ids, attention_mask
                if torch.cuda.is_available() and len(embeddings_list) % 50 == 0:
                    torch.cuda.empty_cache()
        
        except RuntimeError as e:
            logger.warning(f"Error processing batch: {str(e)}")
            # Fall back to CPU if OOM error
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            input_ids = batch['input_ids'].cpu()
            attention_mask = batch['attention_mask'].cpu()
            model_cpu = model.cpu()
            
            with torch.no_grad():
                outputs = model_cpu(input_ids=input_ids, attention_mask=attention_mask)
                last_hidden_state = outputs[0]
                batch_embeddings = last_hidden_state[:, 0, :].numpy()
                embeddings_list.append(batch_embeddings)
            
            if torch.cuda.is_available():
                model.to(device)  # Move model back to GPU
            continue
    
    return np.vstack(embeddings_list) if embeddings_list else np.array([])

class PRPrioritizer:
    def __init__(self, model_path='./app/models/pr_priority_model.pkl'):
        """Initialize the PR Prioritizer with a trained model"""
        logger.info(f"Loading model from {model_path}")
        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)
        
        self.model = model_data['model']
        self.label_encoder = model_data['label_encoder']
        
        # Load feature information (now includes more metadata)
        self.feature_columns = model_data.get('feature_columns', [])
        self.author_association_columns = model_data.get('author_association_columns', [])
        self.feature_means = model_data.get('feature_means', {})
        self.feature_stds = model_data.get('feature_stds', {})
        self.expected_feature_count = model_data.get('expected_feature_count', 1548)  # Based on error message
        
        # Initialize RoBERTa model
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")
        
        # Load tokenizer
        self.tokenizer = RobertaTokenizer.from_pretrained('distilroberta-base', do_lower_case=True)
        
        # Load RoBERTa model
        self.bert_model = RobertaModel.from_pretrained(
            'distilroberta-base',
            output_hidden_states=True,
            torchscript=True
        ).to(self.device)
        self.bert_model.eval()
    
    def process_prs(self, prs_data):
        """Process a list of pull request dictionaries and return a DataFrame"""
        processed_data = []
        
        for pr in prs_data:
            pr_features = extract_pr_features(pr)
            processed_data.append(pr_features)
        
        df = pd.DataFrame(processed_data)
        
        # Add derived features
        df['total_changes'] = df['additions'] + df['deletions']
        df['change_ratio'] = df['additions'] / (df['deletions'] + 1)  # Add 1 to avoid division by zero
        
        return df
    
    def extract_features(self, df, body_embeddings, title_embeddings):
        """Extract and normalize features from PR data with consistent feature set"""
        logger.info("Extracting features...")
        
        # Define the numeric features we want to use
        numeric_features = ['line', 'comments', 'additions', 'deletions', 
                            'changed_files', 'total_changes', 'change_ratio']
        
        # Process numeric features with consistent normalization
        numeric_feature_arrays = []
        for feature in numeric_features:
            if feature in df.columns:
                feature_data = df[feature].fillna(0).values
                
                # Use stored means and stds if available, otherwise compute them
                if feature in self.feature_means and feature in self.feature_stds:
                    mean = self.feature_means[feature]
                    std = self.feature_stds[feature]
                else:
                    mean = feature_data.mean()
                    std = feature_data.std()
                
                # Normalize the feature
                normalized_feature = (feature_data - mean) / (std + 1e-8)
                numeric_feature_arrays.append(normalized_feature)
        
        # Stack all numeric features
        if numeric_feature_arrays:
            numeric_features_array = np.column_stack(numeric_feature_arrays)
        else:
            numeric_features_array = np.empty((len(df), 0))
        
        # Process categorical features with consistent one-hot encoding
        # Create one-hot encoded author_association with consistent columns
        author_dummies = pd.get_dummies(df['author_association'].fillna('NONE'), prefix='author')
        
        # Ensure all expected columns are present
        for col in self.author_association_columns:
            if col not in author_dummies.columns:
                author_dummies[col] = 0
        
        # Ensure only expected columns are included and in the correct order
        author_values = author_dummies[self.author_association_columns].values if self.author_association_columns else pd.get_dummies(df['author_association'].fillna('NONE'), prefix='author').values
        
        # Process in smaller chunks to save memory
        chunk_size = 1000
        final_features_list = []
        
        for i in range(0, len(df), chunk_size):
            end_idx = min(i + chunk_size, len(df))
            
            # Combine features in the correct order
            chunk_features = np.hstack([
                body_embeddings[i:end_idx],
                title_embeddings[i:end_idx],
                numeric_features_array[i:end_idx],
                author_values[i:end_idx]
            ])
            
            final_features_list.append(chunk_features)
        
        final_features = np.vstack(final_features_list) if final_features_list else np.array([])
        
        # Check if we have the right number of features and adjust if necessary
        feature_count = final_features.shape[1]
        logger.info(f"Feature matrix shape: {final_features.shape}, expected: {self.expected_feature_count}")
        
        if feature_count != self.expected_feature_count:
            logger.warning(f"Feature count mismatch: got {feature_count}, expected {self.expected_feature_count}")
            
            if feature_count < self.expected_feature_count:
                # Add padding features
                padding = np.zeros((final_features.shape[0], self.expected_feature_count - feature_count))
                final_features = np.hstack([final_features, padding])
                logger.info(f"Added padding. New shape: {final_features.shape}")
            else:
                # Truncate features
                final_features = final_features[:, :self.expected_feature_count]
                logger.info(f"Truncated features. New shape: {final_features.shape}")
        
        return final_features
    
    def predict_priorities(self, prs_data):
        """Predict priorities for a list of PRs"""
        # Process PRs
        df = self.process_prs(prs_data)
        
        # Generate embeddings
        logger.info("Generating body embeddings...")
        body_embeddings = generate_embeddings_with_dataloader(
            df['body'].tolist(), self.tokenizer, self.bert_model, self.device, batch_size=16
        )
        
        logger.info("Generating title embeddings...")
        title_embeddings = generate_embeddings_with_dataloader(
            df['title'].tolist(), self.tokenizer, self.bert_model, self.device, batch_size=32
        )
        
        # Extract features with consistent feature set
        features = self.extract_features(df, body_embeddings, title_embeddings)
        
        # Debug information
        logger.info(f"Final features shape: {features.shape}")
        
        # Make predictions
        logger.info("Making predictions...")
        try:
            predictions_encoded = self.model.predict(features)
            
            # Get prediction probabilities if available
            confidence_scores = np.ones(len(predictions_encoded))
            if hasattr(self.model, 'predict_proba'):
                try:
                    probabilities = self.model.predict_proba(features)
                    confidence_scores = np.max(probabilities, axis=1)
                except Exception as e:
                    logger.warning(f"Could not get prediction probabilities: {str(e)}")
            
            # Convert predictions to labels
            predictions = self.label_encoder.inverse_transform(predictions_encoded)
            
        except ValueError as e:
            logger.error(f"Prediction error: {str(e)}")
            if 'Feature shape mismatch' in str(e):
                # Extract the expected feature count from the error message
                import re
                match = re.search(r'expected: (\d+), got (\d+)', str(e))
                if match:
                    expected = int(match.group(1))
                    got = int(match.group(2))
                    logger.info(f"Updating feature count from error: expected={expected}, got={got}")
                    
                    # Adjust features to match the expected shape
                    if got < expected:
                        padding = np.zeros((features.shape[0], expected - got))
                        features = np.hstack([features, padding])
                    else:
                        features = features[:, :expected]
                    
                    logger.info(f"Adjusted features shape: {features.shape}")
                    
                    # Try prediction again with adjusted features
                    predictions_encoded = self.model.predict(features)
                    
                    # Get prediction probabilities if available
                    confidence_scores = np.ones(len(predictions_encoded))
                    if hasattr(self.model, 'predict_proba'):
                        try:
                            probabilities = self.model.predict_proba(features)
                            confidence_scores = np.max(probabilities, axis=1)
                        except Exception as e:
                            logger.warning(f"Could not get prediction probabilities: {str(e)}")
                    
                    # Convert predictions to labels
                    predictions = self.label_encoder.inverse_transform(predictions_encoded)
                else:
                    # If we can't parse the error, raise it
                    raise
            else:
                # If it's not a feature shape mismatch error, raise it
                raise
        
        # Create results
        results = []
        for i, pred in enumerate(predictions):
            result = {
                'id': df['id'].iloc[i],
                'title': df['title'].iloc[i],
                'predicted_priority': pred,
                'confidence': float(confidence_scores[i])
            }
            results.append(result)
        
        logger.info(f"Completed predictions for {len(results)} PRs")
        return results
    
    def get_model_info(self):
        """Return metadata about the model"""
        return {
            'model_type': type(self.model).__name__,
            'priority_classes': self.label_encoder.classes_.tolist(),
            'feature_count': self.expected_feature_count,
            'feature_columns': self.feature_columns,
            'author_association_columns': self.author_association_columns,
            'device': str(self.device)
        }