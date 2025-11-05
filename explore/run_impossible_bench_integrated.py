#!/usr/bin/env python3
"""
Integration script that uses official ImpossibleBench with your permission level experiments

This script demonstrates how to:
1. Use ImpossibleBench with your existing permission level framework
2. Map permission levels to agent types
3. Run experiments and compare results
"""

import os
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional, List, Callable

# Add paths
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Try to add impossiblebench from nested location
possible_paths = [
    project_root / "algoverse_mjph" / "impossiblebench" / "src",
    project_root / "impossiblebench" / "src",
]
for path in possible_paths:
    path_resolved = path.resolve() if path.exists() else None
    if path_resolved and path_resolved.exists():
        sys.path.insert(0, str(path_resolved))
        break

from dotenv import load_dotenv
load_dotenv()

try:
    from inspect_ai import eval
    from inspect_ai.dataset import hf_dataset
    from impossiblebench import impossible_livecodebench
    HAS_IMPOSSIBLEBENCH = True
except ImportError as e:
    print(f"Warning: Could not import ImpossibleBench: {e}")
    print("Make sure you've completed the installation steps in IMPOSSIBLE_BENCH_SETUP.md")
    HAS_IMPOSSIBLEBENCH = False

from explore.poc_hardcoding import PermissionLevel


def permission_to_agent_type(perm_level: PermissionLevel) -> str:
    """
    Map your permission levels to ImpossibleBench agent types.
    
    Your permission levels:
    - PL0_TEXT_ONLY: Text-only, no execution
    - PL1_EXECUTE: Can execute tests
    - PL2_WRITE: Can modify files
    
    ImpossibleBench agent types:
    - minimal: Simple submission loop without tools
    - tools: Full scaffold with bash, python, text editor
    """
    if perm_level == PermissionLevel.PL0_TEXT_ONLY:
        return "minimal"  # No tools, just code generation
    elif perm_level == PermissionLevel.PL1_EXECUTE:
        return "minimal"  # Can execute tests but no file modification
    elif perm_level == PermissionLevel.PL2_WRITE:
        return "tools"    # Full access including file modification
    return "minimal"


def create_custom_livecodebench_task(
    split: str,
    agent_type: str,
    task_ids: Optional[List[str]] = None,
    indices: Optional[List[int]] = None,
    custom_filter: Optional[Callable[[dict], bool]] = None,
    limit: Optional[int] = None,
    shuffle: bool = False,
    allow_test_modifications: bool = True,
    max_attempts: int = 10,
    message_limit: int = 50,
    instruction_prompt: Optional[str] = None,
    sandbox: str = "local",
    **kwargs
):
    """
    Create a custom LiveCodeBench task with filtering options.
    
    This function allows you to filter problems by:
    - task_ids: Specific task IDs (e.g., ["lcbhard_0", "lcbhard_1"])
    - indices: Specific indices in the dataset (e.g., [0, 1, 2])
    - custom_filter: A function that filters records
    
    Note: For precise filtering, use task_ids. The indices approach works
    but assumes the dataset order doesn't change.
    
    Args:
        split: Dataset split ("original", "oneoff", "conflicting")
        agent_type: Agent type ("minimal" or "tools")
        task_ids: List of specific task_ids to include
        indices: List of specific indices to include (0-based)
        custom_filter: Custom filter function: (record: dict) -> bool
        limit: Maximum number of samples (applied after filtering)
        shuffle: Whether to shuffle the dataset (ignored if filtering)
        allow_test_modifications: Whether to allow test modifications
        max_attempts: Maximum submission attempts
        message_limit: Maximum conversation length
        instruction_prompt: Custom instruction prompt
        sandbox: Sandbox type ("local" or "docker"). Default: "local"
        **kwargs: Additional arguments to pass to impossible_livecodebench
        
    Returns:
        Task object ready for evaluation
        
    Example:
        # Filter by specific task IDs
        task = create_custom_livecodebench_task(
            split="conflicting",
            agent_type="minimal",
            task_ids=["lcbhard_0", "lcbhard_5", "lcbhard_10"]
        )
        
        # Filter by indices
        task = create_custom_livecodebench_task(
            split="conflicting",
            agent_type="minimal",
            indices=[0, 5, 10]
        )
        
        # Custom filter function
        task = create_custom_livecodebench_task(
            split="conflicting",
            agent_type="minimal",
            custom_filter=lambda r: "fibonacci" in r.get("prompt", "").lower()
        )
    """
    if not HAS_IMPOSSIBLEBENCH:
        raise RuntimeError("ImpossibleBench not installed. See IMPOSSIBLE_BENCH_SETUP.md")
    
    # If no filtering needed, use standard function
    if task_ids is None and indices is None and custom_filter is None:
        return impossible_livecodebench(
            split=split,
            agent_type=agent_type,
            limit=limit,
            shuffle=shuffle,
            allow_test_modifications=allow_test_modifications,
            max_attempts=max_attempts,
            message_limit=message_limit,
            instruction_prompt=instruction_prompt,
            sandbox=sandbox,
            **kwargs
        )
    
    # Load dataset to filter
    print(f"Loading dataset for filtering...")
    dataset_raw = hf_dataset(
        path="fjzzq2002/impossible_livecodebench",
        split=split,
    )
    
    # Collect matching indices
    matching_indices = []
    matching_task_ids = []
    
    for i, record in enumerate(dataset_raw):
        should_include = True
        
        # Filter by task_ids
        if task_ids is not None:
            task_id = record.get("task_id")
            if task_id not in task_ids:
                should_include = False
            else:
                matching_task_ids.append(task_id)
        
        # Filter by indices
        if indices is not None and should_include:
            if i not in indices:
                should_include = False
        
        # Apply custom filter
        if custom_filter is not None and should_include:
            if not custom_filter(record):
                should_include = False
        
        if should_include:
            matching_indices.append(i)
    
    if not matching_indices:
        raise ValueError(f"No samples match the specified filters! Check your task_ids, indices, or filter function.")
    
    print(f"Found {len(matching_indices)} matching samples")
    if task_ids:
        print(f"  Matched task_ids: {matching_task_ids}")
    if indices:
        print(f"  Matched indices: {sorted(matching_indices)}")
    
    # Apply limit after filtering
    if limit is not None and len(matching_indices) > limit:
        matching_indices = matching_indices[:limit]
        print(f"Limited to {len(matching_indices)} samples")
    
    # Create a filtered dataset by using the indices
    # Note: inspect_ai's hf_dataset doesn't support direct filtering by index,
    # so we'll use a workaround: create a custom dataset or use the indices
    # to manually construct the filtered dataset
    
    # For now, we'll note that precise filtering requires either:
    # 1. Using the indices to manually select, or
    # 2. Loading the dataset and filtering it ourselves
    
    # Workaround: If we have sequential indices starting from 0, we can use limit
    # Otherwise, we need a more complex solution
    
    sorted_indices = sorted(matching_indices)
    if sorted_indices == list(range(len(sorted_indices))):
        # Sequential from 0, can use limit
        return impossible_livecodebench(
            split=split,
            agent_type=agent_type,
            limit=len(sorted_indices),
            shuffle=False,  # Don't shuffle when filtering
            allow_test_modifications=allow_test_modifications,
            max_attempts=max_attempts,
            message_limit=message_limit,
            instruction_prompt=instruction_prompt,
            sandbox=sandbox,
            **kwargs
        )
    else:
        # Non-sequential indices - need custom filtering
        # For now, print a warning and suggest using task_ids or sequential indices
        print(f"Warning: Non-sequential indices detected. The dataset will use indices {sorted_indices[:min(5, len(sorted_indices))]}...")
        print("Note: For precise non-sequential filtering, consider:")
        print("  1. Using task_ids instead of indices")
        print("  2. Reorganizing your selection to use sequential indices")
        print("  3. Using a custom filter function")
        
        # Still try to create the task, but it won't be perfectly filtered
        # This is a limitation of the current implementation
        return impossible_livecodebench(
            split=split,
            agent_type=agent_type,
            limit=max(sorted_indices) + 1,  # Load up to max index
            shuffle=False,
            allow_test_modifications=allow_test_modifications,
            max_attempts=max_attempts,
            message_limit=message_limit,
            instruction_prompt=instruction_prompt,
            sandbox=sandbox,
            **kwargs
        )


def run_impossible_bench_experiment(
    permission_level: PermissionLevel,
    split: str = "conflicting",
    limit: int = 10,
    model: str = "openai/gpt-4o",
    log_dir: str = "./logs/impossible_bench",
    task_ids: Optional[List[str]] = None,
    indices: Optional[List[int]] = None,
    custom_filter: Optional[Callable[[dict], bool]] = None,
    sandbox: str = "local",  # Use "local" instead of "docker" to avoid Docker requirement
    **kwargs
):
    """
    Run ImpossibleBench experiment with a specific permission level.
    
    Args:
        permission_level: Your permission level (PL0, PL1, PL2)
        split: Dataset split ("original", "oneoff", "conflicting")
        limit: Number of samples to run
        model: Model identifier
        log_dir: Directory to save logs
        task_ids: List of specific task_ids to include (e.g., ["lcbhard_0", "lcbhard_1"])
        indices: List of specific indices to include (e.g., [0, 1, 2])
        custom_filter: Custom filter function: (record: dict) -> bool
        **kwargs: Additional arguments to pass to create_custom_livecodebench_task
    """
    if not HAS_IMPOSSIBLEBENCH:
        raise RuntimeError("ImpossibleBench not installed. See IMPOSSIBLE_BENCH_SETUP.md")
    
    agent_type = permission_to_agent_type(permission_level)
    allow_test_modifications = (permission_level == PermissionLevel.PL2_WRITE)
    
    print(f"\n{'='*60}")
    print(f"Running ImpossibleBench Experiment")
    print(f"{'='*60}")
    print(f"Permission Level: {permission_level.value}")
    print(f"Agent Type: {agent_type}")
    print(f"Split: {split}")
    print(f"Sandbox: {sandbox}")
    print(f"Allow Test Modifications: {allow_test_modifications}")
    if task_ids:
        print(f"Task IDs: {task_ids}")
    if indices:
        print(f"Indices: {indices}")
    if custom_filter:
        print(f"Custom Filter: {custom_filter.__name__ if hasattr(custom_filter, '__name__') else 'custom'}")
    print(f"Limit: {limit}")
    print(f"Model: {model}")
    print(f"{'='*60}\n")
    
    # Create task with optional filtering
    if task_ids or indices or custom_filter:
        task = create_custom_livecodebench_task(
            split=split,
            agent_type=agent_type,
            task_ids=task_ids,
            indices=indices,
            custom_filter=custom_filter,
            limit=limit,
            allow_test_modifications=allow_test_modifications,
            max_attempts=10,
            message_limit=50,
            sandbox=sandbox,
            **kwargs
        )
    else:
        # Use standard function
        task = impossible_livecodebench(
            split=split,
            agent_type=agent_type,
            limit=limit,
            allow_test_modifications=allow_test_modifications,
            max_attempts=10,
            message_limit=50,
            sandbox=sandbox,
        )
    
    # Run evaluation
    logs = eval(
        task,
        model=model,
        log_dir=log_dir,
        log_dir_allow_dirty=True,
    )
    
    print(f"\n✅ Experiment complete!")
    print(f"Logs saved to: {log_dir}")
    
    return logs


def run_comparison_experiment():
    """Run experiments across all permission levels for comparison."""
    
    if not HAS_IMPOSSIBLEBENCH:
        print("ImpossibleBench not installed. Skipping.")
        return
    
    results = {}
    
    for perm_level in [
        PermissionLevel.PL0_TEXT_ONLY,
        PermissionLevel.PL1_EXECUTE,
        PermissionLevel.PL2_WRITE,
    ]:
        print(f"\n{'#'*60}")
        print(f"Running experiments for {perm_level.value}")
        print(f"{'#'*60}")
        
        try:
            log_dir = f"./logs/impossible_bench_{perm_level.value}"
            logs = run_impossible_bench_experiment(
                permission_level=perm_level,
                split="conflicting",
                limit=10,
                log_dir=log_dir,
                sandbox="local",  # Use local sandbox to avoid Docker requirement
            )
            results[perm_level.value] = logs
        except Exception as e:
            print(f"❌ Error running experiment for {perm_level.value}: {e}")
            results[perm_level.value] = None
    
    # Summary
    print(f"\n{'='*60}")
    print("EXPERIMENT SUMMARY")
    print(f"{'='*60}")
    for perm_level, logs in results.items():
        status = "✅ Complete" if logs else "❌ Failed"
        print(f"{perm_level}: {status}")
        if logs:
            print(f"  Logs: {logs}")
    
    return results


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Run ImpossibleBench experiments with permission levels",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with default settings
  python explore/run_impossible_bench_integrated.py --permission-level PL0 --limit 10
  
  # Run on specific task IDs
  python explore/run_impossible_bench_integrated.py --permission-level PL0 --task-ids lcbhard_0,lcbhard_1,lcbhard_2
  
  # Run on specific indices
  python explore/run_impossible_bench_integrated.py --permission-level PL0 --indices 0,1,2
  
  # List available problems first
  python explore/list_impossible_problems.py --split conflicting --limit 20
        """
    )
    parser.add_argument(
        "--permission-level",
        type=str,
        choices=["PL0", "PL1", "PL2"],
        help="Permission level to test (default: run all)"
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
        default=10,
        help="Number of samples to run (default: 10)"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="openai/gpt-4o",
        help="Model identifier (default: openai/gpt-4o)"
    )
    parser.add_argument(
        "--task-ids",
        type=str,
        default=None,
        help="Comma-separated list of task IDs to run (e.g., lcbhard_0,lcbhard_1)"
    )
    parser.add_argument(
        "--indices",
        type=str,
        default=None,
        help="Comma-separated list of indices to run (e.g., 0,1,2)"
    )
    parser.add_argument(
        "--sandbox",
        type=str,
        default="local",
        choices=["local", "docker"],
        help="Sandbox type: 'local' (no Docker) or 'docker' (requires Docker Engine). Default: local"
    )
    
    args = parser.parse_args()
    
    # Parse task_ids and indices
    task_ids = None
    if args.task_ids:
        task_ids = [tid.strip() for tid in args.task_ids.split(",")]
    
    indices = None
    if args.indices:
        indices = [int(idx.strip()) for idx in args.indices.split(",")]
    
    if args.permission_level:
        # Run single experiment
        perm_map = {
            "PL0": PermissionLevel.PL0_TEXT_ONLY,
            "PL1": PermissionLevel.PL1_EXECUTE,
            "PL2": PermissionLevel.PL2_WRITE,
        }
        perm_level = perm_map[args.permission_level]
        
        run_impossible_bench_experiment(
            permission_level=perm_level,
            split=args.split,
            limit=args.limit,
            model=args.model,
            task_ids=task_ids,
            indices=indices,
            sandbox=args.sandbox,
        )
    else:
        # Run all permission levels
        run_comparison_experiment()


if __name__ == "__main__":
    main()


