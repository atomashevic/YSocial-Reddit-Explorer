#!/usr/bin/env python3
"""
OpenAI Moderation API Content Analysis Script

This script processes posts.csv files in data folders and creates moderation.csv files
using OpenAI's Moderation API for harmful content detection.
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
from openai import OpenAI
from openai import RateLimitError, APIError, APIConnectionError
from tqdm import tqdm
import pickle


class Config:
    """Configuration management for the moderation detector."""
    
    def __init__(self):
        self.api_key = os.getenv('OPENAI_API_KEY')
        self.rate_limit_delay = 1.0  # seconds between requests
        self.max_retries = 3
        self.batch_size = 100  # Save progress every N posts
        self.max_text_length = 32000  # OpenAI moderation API limit
        
    def validate(self):
        """Validate configuration."""
        if not self.api_key:
            raise ValueError(
                "OPENAI_API_KEY environment variable must be set.\n"
                "Get your API key from: https://platform.openai.com/account/api-keys"
            )


class OpenAIModerationClient:
    """Client for OpenAI Moderation API with rate limiting and error handling."""
    
    def __init__(self, config: Config):
        self.config = config
        self.client = OpenAI(api_key=config.api_key)
        self.logger = logging.getLogger(__name__)
        
    def moderate_text(self, text: str) -> Optional[Dict]:
        """
        Moderate text using OpenAI Moderation API.
        
        Args:
            text: Text content to moderate
            
        Returns:
            Dict with moderation results or None if failed
        """
        if not text or not text.strip():
            return self._empty_moderation_result()
            
        # Truncate text if too long
        if len(text) > self.config.max_text_length:
            text = text[:self.config.max_text_length]
            self.logger.warning(f"Text truncated to {self.config.max_text_length} characters")
            
        for attempt in range(self.config.max_retries):
            try:
                response = self.client.moderations.create(input=text)
                result = response.results[0]
                
                # Convert to our format
                moderation_data = {
                    'flagged': result.flagged,
                    'categories': {
                        'sexual': result.category_scores.sexual,
                        'sexual_minors': result.category_scores.sexual_minors,
                        'harassment': result.category_scores.harassment,
                        'harassment_threatening': result.category_scores.harassment_threatening,
                        'hate': result.category_scores.hate,
                        'hate_threatening': result.category_scores.hate_threatening,
                        'illicit': result.category_scores.illicit,
                        'illicit_violent': result.category_scores.illicit_violent,
                        'self_harm': result.category_scores.self_harm,
                        'self_harm_intent': result.category_scores.self_harm_intent,
                        'self_harm_instructions': result.category_scores.self_harm_instructions,
                        'violence': result.category_scores.violence,
                        'violence_graphic': result.category_scores.violence_graphic
                    }
                }
                
                # Add delay to respect rate limits
                time.sleep(self.config.rate_limit_delay)
                return moderation_data
                
            except RateLimitError as e:
                wait_time = 2 ** attempt
                self.logger.warning(f"Rate limit hit, waiting {wait_time}s (attempt {attempt + 1})")
                time.sleep(wait_time)
                
            except (APIError, APIConnectionError) as e:
                self.logger.error(f"API error (attempt {attempt + 1}): {e}")
                if attempt == self.config.max_retries - 1:
                    return None
                time.sleep(2 ** attempt)
                
            except Exception as e:
                self.logger.error(f"Unexpected error: {e}")
                return None
                
        return None
    
    def _empty_moderation_result(self) -> Dict:
        """Return empty moderation result for empty/invalid text."""
        return {
            'flagged': False,
            'categories': {
                'sexual': 0.0,
                'sexual_minors': 0.0,
                'harassment': 0.0,
                'harassment_threatening': 0.0,
                'hate': 0.0,
                'hate_threatening': 0.0,
                'illicit': 0.0,
                'illicit_violent': 0.0,
                'self_harm': 0.0,
                'self_harm_intent': 0.0,
                'self_harm_instructions': 0.0,
                'violence': 0.0,
                'violence_graphic': 0.0
            }
        }


class DataProcessor:
    """Handles reading posts data and preparing it for moderation."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
    def load_posts(self, posts_file: Path) -> pd.DataFrame:
        """Load posts from CSV file."""
        try:
            df = pd.read_csv(posts_file)
            self.logger.info(f"Loaded {len(df)} posts from {posts_file}")
            return df
        except Exception as e:
            self.logger.error(f"Error loading {posts_file}: {e}")
            raise
            
    def detect_post_type(self, row: pd.Series) -> str:
        """Detect post type based on data."""
        if pd.notna(row.get('is_comment')) and row['is_comment']:
            return 'comment'
        elif pd.notna(row.get('news_id')):
            return 'news_share'
        else:
            return 'regular_post'
            
    def prepare_moderation_data(self, df: pd.DataFrame) -> List[Dict]:
        """Prepare data for moderation processing."""
        posts_data = []
        
        for _, row in df.iterrows():
            post_data = {
                'id': row['id'],
                'content': str(row.get('content', '')),
                'post_type': self.detect_post_type(row),
                'is_comment': bool(row.get('is_comment', False))
            }
            posts_data.append(post_data)
            
        return posts_data
        
    def save_moderation_results(self, results: List[Dict], output_file: Path):
        """Save moderation results to CSV."""
        if not results:
            self.logger.warning("No results to save")
            return
            
        # Convert results to DataFrame
        rows = []
        for result in results:
            row = {
                'id': result['id'],
                'sexual': result['moderation']['categories']['sexual'],
                'sexual_minors': result['moderation']['categories']['sexual_minors'],
                'harassment': result['moderation']['categories']['harassment'],
                'harassment_threatening': result['moderation']['categories']['harassment_threatening'],
                'hate': result['moderation']['categories']['hate'],
                'hate_threatening': result['moderation']['categories']['hate_threatening'],
                'illicit': result['moderation']['categories']['illicit'],
                'illicit_violent': result['moderation']['categories']['illicit_violent'],
                'self_harm': result['moderation']['categories']['self_harm'],
                'self_harm_intent': result['moderation']['categories']['self_harm_intent'],
                'self_harm_instructions': result['moderation']['categories']['self_harm_instructions'],
                'violence': result['moderation']['categories']['violence'],
                'violence_graphic': result['moderation']['categories']['violence_graphic'],
                'flagged': result['moderation']['flagged'],
                'post_type': result['post_type'],
                'is_comment': result['is_comment']
            }
            rows.append(row)
            
        df = pd.DataFrame(rows)
        df.to_csv(output_file, index=False)
        self.logger.info(f"Saved {len(df)} moderation results to {output_file}")


class ProgressTracker:
    """Tracks processing progress with checkpoint/resume capability."""
    
    def __init__(self, checkpoint_file: Path):
        self.checkpoint_file = checkpoint_file
        self.logger = logging.getLogger(__name__)
        
    def load_progress(self) -> Tuple[int, List[Dict]]:
        """Load progress from checkpoint file."""
        if not self.checkpoint_file.exists():
            return 0, []
            
        try:
            with open(self.checkpoint_file, 'rb') as f:
                data = pickle.load(f)
                return data.get('processed_count', 0), data.get('results', [])
        except Exception as e:
            self.logger.warning(f"Error loading checkpoint: {e}")
            return 0, []
            
    def save_progress(self, processed_count: int, results: List[Dict]):
        """Save progress to checkpoint file."""
        try:
            data = {
                'processed_count': processed_count,
                'results': results
            }
            with open(self.checkpoint_file, 'wb') as f:
                pickle.dump(data, f)
        except Exception as e:
            self.logger.error(f"Error saving checkpoint: {e}")
            
    def cleanup(self):
        """Remove checkpoint file after successful completion."""
        if self.checkpoint_file.exists():
            try:
                self.checkpoint_file.unlink()
            except Exception as e:
                self.logger.warning(f"Error removing checkpoint file: {e}")


class ModerationDetector:
    """Main class that orchestrates the moderation detection process."""
    
    def __init__(self, config: Config):
        self.config = config
        self.client = OpenAIModerationClient(config)
        self.processor = DataProcessor()
        self.logger = logging.getLogger(__name__)
        
    def process_folder(self, data_folder: Path, force: bool = False) -> bool:
        """Process a single data folder."""
        posts_file = data_folder / 'posts.csv'
        moderation_file = data_folder / 'moderation.csv'
        checkpoint_file = data_folder / '.moderation_checkpoint.pkl'
        
        # Check if posts.csv exists
        if not posts_file.exists():
            self.logger.warning(f"No posts.csv found in {data_folder}")
            return False
            
        # Check if moderation.csv already exists
        if moderation_file.exists() and not force:
            self.logger.info(f"moderation.csv already exists in {data_folder}, skipping (use --force to reprocess)")
            return True
            
        self.logger.info(f"Processing {data_folder}")
        
        # Load posts data
        try:
            df = self.processor.load_posts(posts_file)
            posts_data = self.processor.prepare_moderation_data(df)
        except Exception as e:
            self.logger.error(f"Error processing {posts_file}: {e}")
            return False
            
        # Set up progress tracking
        tracker = ProgressTracker(checkpoint_file)
        processed_count, results = tracker.load_progress()
        
        if processed_count > 0:
            self.logger.info(f"Resuming from post {processed_count + 1}")
            
        # Process posts
        total_posts = len(posts_data)
        with tqdm(total=total_posts, initial=processed_count, desc=f"Processing {data_folder.name}") as pbar:
            for i in range(processed_count, total_posts):
                post = posts_data[i]
                
                # Get moderation for this post
                moderation_result = self.client.moderate_text(post['content'])
                
                if moderation_result is None:
                    self.logger.error(f"Failed to moderate post {post['id']}")
                    continue
                    
                # Store result
                result = {
                    'id': post['id'],
                    'moderation': moderation_result,
                    'post_type': post['post_type'],
                    'is_comment': post['is_comment']
                }
                results.append(result)
                
                pbar.update(1)
                processed_count += 1
                
                # Save progress periodically
                if processed_count % self.config.batch_size == 0:
                    tracker.save_progress(processed_count, results)
                    
        # Save final results
        try:
            self.processor.save_moderation_results(results, moderation_file)
            tracker.cleanup()
            self.logger.info(f"Successfully processed {data_folder}")
            return True
        except Exception as e:
            self.logger.error(f"Error saving results for {data_folder}: {e}")
            return False
            
    def process_all_folders(self, data_dir: Path, target_folders: Optional[List[str]] = None, force: bool = False):
        """Process all data folders."""
        if not data_dir.exists():
            raise FileNotFoundError(f"Data directory not found: {data_dir}")
            
        # Find all data folders
        folders = []
        for item in data_dir.iterdir():
            if item.is_dir() and not item.name.startswith('.'):
                if target_folders is None or item.name in target_folders:
                    folders.append(item)
                    
        if not folders:
            self.logger.warning("No data folders found to process")
            return
            
        folders.sort()  # Process in consistent order
        
        success_count = 0
        for folder in folders:
            try:
                if self.process_folder(folder, force):
                    success_count += 1
            except KeyboardInterrupt:
                self.logger.info("Processing interrupted by user")
                break
            except Exception as e:
                self.logger.error(f"Error processing {folder}: {e}")
                
        self.logger.info(f"Successfully processed {success_count}/{len(folders)} folders")


def setup_logging(verbose: bool = False):
    """Set up logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO
    format_str = '%(asctime)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=level, format=format_str)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="OpenAI Moderation API content analysis for Reddit simulation data"
    )
    parser.add_argument(
        '--folders',
        nargs='+',
        help='Specific folder names to process (e.g., reddit-technology-v2)'
    )
    parser.add_argument(
        '--force',
        action='store_true',
        help='Force reprocessing even if moderation.csv already exists'
    )
    parser.add_argument(
        '--delay',
        type=float,
        default=1.0,
        help='Delay between API requests in seconds (default: 1.0)'
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )
    
    args = parser.parse_args()
    
    # Set up logging
    setup_logging(args.verbose)
    logger = logging.getLogger(__name__)
    
    try:
        # Initialize configuration
        config = Config()
        config.rate_limit_delay = args.delay
        config.validate()
        
        # Initialize detector
        detector = ModerationDetector(config)
        
        # Process folders
        data_dir = Path('data')
        detector.process_all_folders(
            data_dir,
            target_folders=args.folders,
            force=args.force
        )
        
    except KeyboardInterrupt:
        logger.info("Script interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Script failed: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()