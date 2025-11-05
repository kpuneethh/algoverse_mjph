#!/usr/bin/env python3
"""
Helper script to list and explore available problems in ImpossibleBench.

Usage:
    # List all problems in conflicting split
    python explore/list_impossible_problems.py --split conflicting
    
    # List first 10 problems with details
    python explore/list_impossible_problems.py --split conflicting --limit 10 --details
    
    # Show specific problem by task_id
    python explore/list_impossible_problems.py --split conflicting --task-id lcbhard_0
    
    # Export task_ids to a file for later use
    python explore/list_impossible_problems.py --split conflicting --export task_ids.txt
"""

import sys
from pathlib import Path
from typing import Optional
import json

# Add paths
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

possible_paths = [
    project_root / "algoverse_mjph" / "impossiblebench" / "src",
    project_root / "impossiblebench" / "src",
]
for path in possible_paths:
    path_resolved = path.resolve() if path.exists() else None
    if path_resolved and path_resolved.exists():
        sys.path.insert(0, str(path_resolved))
        break

try:
    from datasets import load_dataset
    HAS_IMPOSSIBLEBENCH = True
except ImportError:
    print("Error: datasets library not installed. Install with: pip install datasets")
    sys.exit(1)


def list_problems(
    split: str = "conflicting",
    limit: Optional[int] = None,
    task_id: Optional[str] = None,
    show_details: bool = False,
    export_file: Optional[str] = None
):
    """List problems in the dataset."""
    
    # Load dataset directly from HuggingFace (not through inspect_ai)
    print(f"Loading dataset: fjzzq2002/impossible_livecodebench (split: {split})")
    dataset = load_dataset(
        "fjzzq2002/impossible_livecodebench",
        split=split,
    )
    
    print(f"Total problems in '{split}' split: {len(dataset)}\n")
    
    problems = []
    found_task = None
    
    for i, record in enumerate(dataset):
        current_task_id = record.get("task_id", f"unknown_{i}")
        
        # If looking for specific task_id
        if task_id and current_task_id == task_id:
            found_task = {
                "index": i,
                "task_id": current_task_id,
                "entry_point": record.get("entry_point", "unknown"),
                "impossible_type": record.get("impossible_type", split),
                "prompt": record.get("prompt", ""),
                "test": record.get("test", ""),
                "original_test": record.get("original_test", ""),
            }
            break
        
        # Collect problem info
        if not task_id:  # Only collect if not searching for specific task
            problem_info = {
                "index": i,
                "task_id": current_task_id,
                "entry_point": record.get("entry_point", "unknown"),
                "impossible_type": record.get("impossible_type", split),
            }
            
            if show_details:
                problem_info.update({
                    "prompt_preview": record.get("prompt", "")[:150] + "..." if len(record.get("prompt", "")) > 150 else record.get("prompt", ""),
                    "test_preview": record.get("test", "")[:150] + "..." if len(record.get("test", "")) > 150 else record.get("test", ""),
                })
            
            problems.append(problem_info)
            
            if limit and len(problems) >= limit:
                break
    
    # Display results
    if task_id:
        if found_task:
            print(f"Found task: {task_id}\n")
            print("=" * 80)
            print(f"Index: {found_task['index']}")
            print(f"Task ID: {found_task['task_id']}")
            print(f"Entry Point: {found_task['entry_point']}")
            print(f"Type: {found_task['impossible_type']}")
            print("\nPrompt:")
            print("-" * 80)
            print(found_task['prompt'])
            print("\nTest (may be impossible):")
            print("-" * 80)
            print(found_task['test'])
            print("\nOriginal Test:")
            print("-" * 80)
            print(found_task['original_test'])
        else:
            print(f"Task ID '{task_id}' not found in '{split}' split")
    else:
        print(f"Showing {len(problems)} problem(s):\n")
        for prob in problems:
            print(f"Index {prob['index']:3d} | Task ID: {prob['task_id']:15s} | Entry: {prob['entry_point']:20s} | Type: {prob['impossible_type']}")
            if show_details:
                print(f"  Prompt: {prob.get('prompt_preview', 'N/A')}")
                print(f"  Test: {prob.get('test_preview', 'N/A')}")
                print()
        
        # Export if requested
        if export_file:
            task_ids = [p['task_id'] for p in problems]
            with open(export_file, 'w') as f:
                json.dump(task_ids, f, indent=2)
            print(f"\nExported {len(task_ids)} task_ids to {export_file}")
            print(f"You can use these in your experiments like:")
            print(f"  --task-ids {','.join(task_ids[:3])}...")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description="List and explore ImpossibleBench problems",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # List first 10 problems
  python explore/list_impossible_problems.py --split conflicting --limit 10
  
  # Show detailed view
  python explore/list_impossible_problems.py --split conflicting --limit 5 --details
  
  # Find specific problem
  python explore/list_impossible_problems.py --split conflicting --task-id lcbhard_0
  
  # Export task_ids to file
  python explore/list_impossible_problems.py --split conflicting --export my_problems.txt
        """
    )
    
    parser.add_argument(
        "--split",
        type=str,
        choices=["original", "oneoff", "conflicting"],
        default="conflicting",
        help="Dataset split (default: conflicting)"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Maximum number of problems to show"
    )
    parser.add_argument(
        "--task-id",
        type=str,
        default=None,
        help="Show details for specific task_id"
    )
    parser.add_argument(
        "--details",
        action="store_true",
        help="Show detailed problem information"
    )
    parser.add_argument(
        "--export",
        type=str,
        default=None,
        help="Export task_ids to JSON file"
    )
    
    args = parser.parse_args()
    
    list_problems(
        split=args.split,
        limit=args.limit,
        task_id=args.task_id,
        show_details=args.details,
        export_file=args.export
    )


if __name__ == "__main__":
    main()
