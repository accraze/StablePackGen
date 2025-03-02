#!/usr/bin/env python3
"""
Batch Sample Pack Generator for StablePackGen.

This script allows queuing multiple sample packs for generation overnight.
It processes each pack in sequence, saving the results to their respective directories.
"""

import os
import sys
import time
import json
import argparse
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta

# Import from stablepackgen
from stablepackgen import run_agent_with_prompt, generate_sample_pack_plan
from genre_templates import get_available_genres

class BatchGenerator:
    """Manages batch generation of multiple sample packs."""
    
    def __init__(self, queue_file: str = "generation_queue.json"):
        """
        Initialize the batch generator.
        
        Args:
            queue_file: Path to the JSON file storing the generation queue
        """
        self.queue_file = queue_file
        self.queue = self._load_queue()
        
    def _load_queue(self) -> List[Dict[str, Any]]:
        """Load the generation queue from file."""
        if os.path.exists(self.queue_file):
            try:
                with open(self.queue_file, 'r') as f:
                    return json.load(f)
            except json.JSONDecodeError:
                print(f"Error: Queue file {self.queue_file} is corrupted. Creating a new queue.")
                return []
        return []
    
    def _save_queue(self) -> None:
        """Save the generation queue to file."""
        with open(self.queue_file, 'w') as f:
            json.dump(self.queue, f, indent=2)
    
    def add_to_queue(self, genre: str, output_dir: Optional[str] = None) -> None:
        """
        Add a sample pack to the generation queue.
        
        Args:
            genre: The genre of the sample pack to generate
            output_dir: Optional custom output directory
        """
        # Validate genre
        available_genres = get_available_genres()
        if genre.lower() not in available_genres:
            print(f"Error: Unknown genre '{genre}'. Available genres: {', '.join(available_genres)}")
            return
        
        # Create queue entry
        entry = {
            "genre": genre.lower(),
            "status": "pending",
            "added_at": datetime.now().isoformat(),
            "output_dir": output_dir
        }
        
        self.queue.append(entry)
        self._save_queue()
        print(f"Added {genre} sample pack to generation queue.")
    
    def remove_from_queue(self, index: int) -> None:
        """
        Remove a sample pack from the generation queue.
        
        Args:
            index: The index of the sample pack to remove
        """
        if 0 <= index < len(self.queue):
            removed = self.queue.pop(index)
            self._save_queue()
            print(f"Removed {removed['genre']} sample pack from queue.")
        else:
            print(f"Error: Invalid index {index}. Queue has {len(self.queue)} entries.")
    
    def list_queue(self) -> None:
        """Display the current generation queue."""
        if not self.queue:
            print("Generation queue is empty.")
            return
        
        print("\nSample Pack Generation Queue:")
        print("-----------------------------")
        for i, entry in enumerate(self.queue):
            status = entry["status"]
            genre = entry["genre"]
            added = datetime.fromisoformat(entry["added_at"]).strftime("%Y-%m-%d %H:%M")
            
            # Format status with color if terminal supports it
            if status == "pending":
                status_str = "\033[33mPending\033[0m"  # Yellow
            elif status == "completed":
                status_str = "\033[32mCompleted\033[0m"  # Green
            elif status == "failed":
                status_str = "\033[31mFailed\033[0m"  # Red
            elif status == "in_progress":
                status_str = "\033[36mIn Progress\033[0m"  # Cyan
            else:
                status_str = status
            
            # Check if terminal supports colors
            if not sys.stdout.isatty():
                status_str = status
            
            print(f"{i+1}. {genre.title()} - Status: {status_str} - Added: {added}")
            
            # Show output directory if custom
            if entry.get("output_dir"):
                print(f"   Output: {entry['output_dir']}")
            
            # Show completion time if available
            if entry.get("completed_at"):
                completed = datetime.fromisoformat(entry["completed_at"]).strftime("%Y-%m-%d %H:%M")
                print(f"   Completed: {completed}")
            
            # Show error if failed
            if status == "failed" and entry.get("error"):
                print(f"   Error: {entry['error']}")
        
        print()
    
    def process_queue(self, limit: Optional[int] = None, until: Optional[str] = None) -> None:
        """
        Process the generation queue.
        
        Args:
            limit: Maximum number of sample packs to generate
            until: Process queue until this time (format: HH:MM)
        """
        pending_entries = [entry for entry in self.queue if entry["status"] == "pending"]
        
        if not pending_entries:
            print("No pending sample packs in the queue.")
            return
        
        print(f"Starting batch generation of {len(pending_entries)} sample packs...")
        
        # Calculate end time if specified
        end_time = None
        if until:
            try:
                hour, minute = map(int, until.split(':'))
                now = datetime.now()
                end_time = now.replace(hour=hour, minute=minute)
                
                # If the specified time is earlier than now, assume it's for tomorrow
                if end_time < now:
                    end_time += timedelta(days=1)
                
                print(f"Will process queue until {end_time.strftime('%Y-%m-%d %H:%M')}")
            except ValueError:
                print(f"Error: Invalid time format '{until}'. Please use HH:MM format.")
                return
        
        # Process queue
        processed_count = 0
        for i, entry in enumerate(self.queue):
            # Skip if not pending
            if entry["status"] != "pending":
                continue
            
            # Check if we've reached the limit
            if limit is not None and processed_count >= limit:
                print(f"Reached limit of {limit} sample packs. Stopping.")
                break
            
            # Check if we've reached the end time
            if end_time and datetime.now() >= end_time:
                print(f"Reached end time {until}. Stopping.")
                break
            
            # Update status to in_progress
            entry["status"] = "in_progress"
            self._save_queue()
            
            genre = entry["genre"]
            output_dir = entry.get("output_dir")
            
            print(f"\n[{i+1}/{len(pending_entries)}] Generating {genre} sample pack...")
            
            try:
                # If custom output directory is specified, temporarily change working directory
                original_cwd = None
                if output_dir:
                    print(f"Using custom output directory: {output_dir}")
                    os.makedirs(output_dir, exist_ok=True)
                    original_cwd = os.getcwd()
                    os.chdir(output_dir)
                
                # Generate sample pack
                start_time = time.time()
                result = run_agent_with_prompt(f"Generate a complete {genre} sample pack", genre=genre)
                end_time = time.time()
                
                # Restore original working directory if changed
                if original_cwd:
                    os.chdir(original_cwd)
                
                # Update queue entry
                entry["status"] = "completed"
                entry["completed_at"] = datetime.now().isoformat()
                entry["duration"] = round(end_time - start_time, 2)
                
                # Count samples by category
                samples_by_category = {}
                for action in result["actions"]:
                    if hasattr(action, 'tool_input') and action.tool == "generate_audio":
                        path = action.tool_input["output_path"]
                        path_parts = path.split(os.sep)
                        category = path_parts[-2]
                        
                        if category not in samples_by_category:
                            samples_by_category[category] = 0
                        samples_by_category[category] += 1
                
                entry["samples"] = samples_by_category
                entry["total_samples"] = sum(samples_by_category.values())
                
                print(f"✓ Completed {genre} sample pack with {entry['total_samples']} samples in {entry['duration']}s")
                
                processed_count += 1
                
            except Exception as e:
                # Update queue entry with error
                entry["status"] = "failed"
                entry["error"] = str(e)
                print(f"✗ Failed to generate {genre} sample pack: {str(e)}")
                
                # Restore original working directory if changed
                if original_cwd:
                    os.chdir(original_cwd)
            
            # Save queue after each sample pack
            self._save_queue()
        
        print("\nBatch generation completed.")
        self.list_queue()

def main():
    """Main entry point for the batch generator."""
    parser = argparse.ArgumentParser(description="Batch Sample Pack Generator")
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")
    
    # Add command
    add_parser = subparsers.add_parser("add", help="Add a sample pack to the queue")
    add_parser.add_argument("genre", help="Genre of the sample pack")
    add_parser.add_argument("--output-dir", help="Custom output directory")
    
    # Remove command
    remove_parser = subparsers.add_parser("remove", help="Remove a sample pack from the queue")
    remove_parser.add_argument("index", type=int, help="Index of the sample pack to remove (1-based)")
    
    # List command
    subparsers.add_parser("list", help="List the current generation queue")
    
    # Process command
    process_parser = subparsers.add_parser("process", help="Process the generation queue")
    process_parser.add_argument("--limit", type=int, help="Maximum number of sample packs to generate")
    process_parser.add_argument("--until", help="Process queue until this time (format: HH:MM)")
    
    # Parse arguments
    args = parser.parse_args()
    
    # Create batch generator
    batch_gen = BatchGenerator()
    
    # Execute command
    if args.command == "add":
        batch_gen.add_to_queue(args.genre, args.output_dir)
    elif args.command == "remove":
        batch_gen.remove_from_queue(args.index - 1)  # Convert to 0-based index
    elif args.command == "list":
        batch_gen.list_queue()
    elif args.command == "process":
        batch_gen.process_queue(args.limit, args.until)
    else:
        # If no command is specified, show help
        parser.print_help()

if __name__ == "__main__":
    main()
