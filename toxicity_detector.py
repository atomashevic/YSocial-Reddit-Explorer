#!/usr/bin/env python3
"""
Perspective API Toxicity Detection Script

This script processes posts.csv files in data folders and creates toxigen.csv files
using Google's Perspective API for toxicity detection.
"""

import os
import sys
import time
import json
import argparse
import logging
from typing import List, Dict, Optional, Tuple
from pathlib import Path
import pandas as pd
from googleapiclient import discovery
from googleapiclient.errors import HttpError
from tqdm import tqdm
import pickle


class Config:
    """Configuration management for the toxicity detector."""
    
    def __init__(self):
        self.api_key = os.getenv('PERSPECTIVE_API_KEY')
        self.rate_limit_delay = 1.5  # seconds between requests
        self.max_retries = 3
        self.batch_size = 100  # Save progress every N posts
        self.do_not_store = True  # Privacy: don't store comments on Google servers
        
    def validate(self):
        """Validate configuration."""
        if not self.api_key:
            raise ValueError(
                "PERSPECTIVE_API_KEY environment variable must be set.\n"
                "Get your API key from: https://developers.google.com/codelabs/setup-perspective-api"
            )


class PerspectiveAPIClient:
    """Client for Google Perspective API with rate limiting and error handling."""
    
    def __init__(self, config: Config):
        self.config = config
        self.client = discovery.build(
            "commentanalyzer",
            "v1alpha1",
            developerKey=config.api_key,
            discoveryServiceUrl="https://commentanalyzer.googleapis.com/$discovery/rest?version=v1alpha1",
            static_discovery=False,
        )
        
    def analyze_toxicity(self, text: str) -> float:
        """
        Analyze toxicity of text using Perspective API.
        
        Args:
            text: Text to analyze
            
        Returns:
            Toxicity score (0.0 to 1.0)
        """
        if not text or not text.strip():
            return 0.0
            
        # Clean text for API
        clean_text = text.strip()
        if len(clean_text) > 20000:  # API limit
            clean_text = clean_text[:20000]
            
        analyze_request = {
            'comment': {'text': clean_text},
            'requestedAttributes': {'TOXICITY': {}},
            'doNotStore': self.config.do_not_store,
            'languages': ['en']  # Assuming English content
        }
        
        for attempt in range(self.config.max_retries):
            try:
                response = self.client.comments().analyze(body=analyze_request).execute()
                toxicity_score = response['attributeScores']['TOXICITY']['summaryScore']['value']
                return float(toxicity_score)
                
            except HttpError as e:
                if e.resp.status == 429:  # Rate limit
                    wait_time = (2 ** attempt) * self.config.rate_limit_delay
                    logging.warning(f"Rate limit hit, waiting {wait_time:.1f}s (attempt {attempt+1})")
                    time.sleep(wait_time)
                elif e.resp.status == 400:  # Bad request (invalid text)
                    logging.warning(f"Invalid text for API: {text[:50]}...")
                    return 0.0
                else:
                    logging.error(f"API error {e.resp.status}: {e}")
                    if attempt == self.config.max_retries - 1:
                        return 0.0
                    time.sleep(self.config.rate_limit_delay)
                    
            except Exception as e:
                logging.error(f"Unexpected error: {e}")
                if attempt == self.config.max_retries - 1:
                    return 0.0
                time.sleep(self.config.rate_limit_delay)
        
        # Rate limiting between requests
        time.sleep(self.config.rate_limit_delay)
        return 0.0


class DataProcessor:
    """Process posts.csv files and generate toxigen.csv format."""
    
    @staticmethod
    def determine_post_type(row: pd.Series) -> str:
        """Determine post type based on data structure."""
        if row['comment_to'] != -1:
            return 'comment'
        elif row['news_id'] != -1 and pd.notna(row['news_id']):
            return 'news_share'
        else:
            return 'regular_post'
    
    @staticmethod
    def extract_text_content(row: pd.Series) -> str:
        """Extract text content from post."""
        text = str(row.get('tweet', ''))
        
        # Remove TITLE: prefix if present
        if text.startswith('TITLE: '):
            text = text[7:]
            
        return text.strip()
    
    @staticmethod
    def load_posts(csv_path: Path) -> pd.DataFrame:
        """Load posts CSV file."""
        try:
            df = pd.read_csv(csv_path)
            logging.info(f"Loaded {len(df)} posts from {csv_path}")
            return df
        except Exception as e:
            logging.error(f"Error loading {csv_path}: {e}")
            raise


class ProgressTracker:
    """Track progress and enable resumption of processing."""
    
    def __init__(self, checkpoint_path: Path):
        self.checkpoint_path = checkpoint_path
        self.processed_ids = set()
        self.results = []
        self.load_checkpoint()
    
    def load_checkpoint(self):
        """Load progress from checkpoint file."""
        if self.checkpoint_path.exists():
            try:
                with open(self.checkpoint_path, 'rb') as f:
                    data = pickle.load(f)
                    self.processed_ids = data.get('processed_ids', set())
                    self.results = data.get('results', [])
                logging.info(f"Resumed from checkpoint: {len(self.processed_ids)} posts already processed")
            except Exception as e:
                logging.warning(f"Could not load checkpoint: {e}")
    
    def save_checkpoint(self):
        """Save current progress."""
        try:
            with open(self.checkpoint_path, 'wb') as f:
                pickle.dump({
                    'processed_ids': self.processed_ids,
                    'results': self.results
                }, f)
        except Exception as e:
            logging.error(f"Could not save checkpoint: {e}")
    
    def add_result(self, post_id: int, toxicity: float, post_type: str, is_comment: bool):
        """Add a processed result."""
        self.processed_ids.add(post_id)
        self.results.append({
            'id': post_id,
            'toxicity': toxicity,
            'post_type': post_type,
            'is_comment': is_comment
        })
    
    def is_processed(self, post_id: int) -> bool:
        """Check if post has been processed."""
        return post_id in self.processed_ids
    
    def get_results_df(self) -> pd.DataFrame:
        """Get results as DataFrame."""
        return pd.DataFrame(self.results)


class ToxicityDetector:
    """Main toxicity detection orchestrator."""
    
    def __init__(self, config: Config):
        self.config = config
        self.api_client = PerspectiveAPIClient(config)
        self.data_processor = DataProcessor()
    
    def process_folder(self, data_folder: Path, force_reprocess: bool = False) -> bool:
        """
        Process a single data folder.
        
        Args:
            data_folder: Path to data folder
            force_reprocess: If True, ignore existing perspective.csv
            
        Returns:
            True if successful, False otherwise
        """
        posts_csv = data_folder / 'posts.csv'
        perspective_csv = data_folder / 'perspective.csv'
        checkpoint_file = data_folder / '.perspective_checkpoint.pkl'
        
        if not posts_csv.exists():
            logging.warning(f"No posts.csv found in {data_folder}")
            return False
            
        if perspective_csv.exists() and not force_reprocess:
            logging.info(f"perspective.csv already exists in {data_folder}, skipping (use --force to reprocess)")
            return True
        
        logging.info(f"Processing {data_folder}")
        
        # Load data
        posts_df = self.data_processor.load_posts(posts_csv)
        
        # Initialize progress tracker
        progress_tracker = ProgressTracker(checkpoint_file)
        
        # Filter unprocessed posts
        unprocessed_posts = posts_df[~posts_df['id'].isin(progress_tracker.processed_ids)]
        
        if len(unprocessed_posts) == 0:
            logging.info("All posts already processed")
        else:
            logging.info(f"Processing {len(unprocessed_posts)} unprocessed posts")
            
            # Process posts with progress bar
            with tqdm(total=len(unprocessed_posts), desc="Analyzing toxicity") as pbar:
                for idx, (_, row) in enumerate(unprocessed_posts.iterrows()):
                    post_id = row['id']
                    text = self.data_processor.extract_text_content(row)
                    post_type = self.data_processor.determine_post_type(row)
                    is_comment = row['comment_to'] != -1
                    
                    # Get toxicity score
                    toxicity = self.api_client.analyze_toxicity(text)
                    
                    # Record result
                    progress_tracker.add_result(post_id, toxicity, post_type, is_comment)
                    
                    # Save checkpoint periodically
                    if (idx + 1) % self.config.batch_size == 0:
                        progress_tracker.save_checkpoint()
                    
                    pbar.update(1)
        
        # Save final results
        results_df = progress_tracker.get_results_df()
        results_df = results_df.sort_values('id')  # Sort by ID to match original format
        results_df.to_csv(perspective_csv, index=False)
        
        # Clean up checkpoint
        if checkpoint_file.exists():
            checkpoint_file.unlink()
            
        logging.info(f"Saved {len(results_df)} results to {perspective_csv}")
        return True
    
    def process_all_folders(self, data_dir: Path, specific_folders: Optional[List[str]] = None, 
                          force_reprocess: bool = False) -> List[str]:
        """
        Process all data folders.
        
        Args:
            data_dir: Path to data directory
            specific_folders: If provided, only process these folders
            force_reprocess: If True, reprocess existing perspective.csv files
            
        Returns:
            List of successfully processed folder names
        """
        if not data_dir.exists():
            raise FileNotFoundError(f"Data directory not found: {data_dir}")
        
        # Find data folders
        all_folders = [f for f in data_dir.iterdir() if f.is_dir()]
        
        if specific_folders:
            folders_to_process = [data_dir / name for name in specific_folders if (data_dir / name).exists()]
            missing = [name for name in specific_folders if not (data_dir / name).exists()]
            if missing:
                logging.warning(f"Folders not found: {missing}")
        else:
            folders_to_process = all_folders
        
        if not folders_to_process:
            logging.error("No folders to process")
            return []
        
        logging.info(f"Found {len(folders_to_process)} folders to process")
        
        successful = []
        for folder in folders_to_process:
            try:
                if self.process_folder(folder, force_reprocess):
                    successful.append(folder.name)
            except Exception as e:
                logging.error(f"Failed to process {folder}: {e}")
        
        return successful


def setup_logging(verbose: bool = False):
    """Setup logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler('toxicity_detector.log')
        ]
    )


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Run Perspective API toxicity detection on posts')
    parser.add_argument('--data-dir', type=Path, default='data', 
                      help='Path to data directory (default: data)')
    parser.add_argument('--folders', nargs='+', 
                      help='Specific folders to process (default: all folders)')
    parser.add_argument('--force', action='store_true',
                      help='Force reprocessing of existing perspective.csv files')
    parser.add_argument('--verbose', '-v', action='store_true',
                      help='Enable verbose logging')
    
    args = parser.parse_args()
    
    # Setup
    setup_logging(args.verbose)
    
    # Load configuration
    try:
        config = Config()
        config.validate()
    except ValueError as e:
        logging.error(e)
        sys.exit(1)
    
    # Process data
    detector = ToxicityDetector(config)
    
    try:
        successful = detector.process_all_folders(
            data_dir=args.data_dir,
            specific_folders=args.folders,
            force_reprocess=args.force
        )
        
        if successful:
            logging.info(f"Successfully processed {len(successful)} folders: {successful}")
        else:
            logging.error("No folders were successfully processed")
            sys.exit(1)
            
    except Exception as e:
        logging.error(f"Processing failed: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()