

## Files

- `explore/run_impossible_bench_integrated.py` - Integration script for your experiments
- `algoverse_mjph/impossiblebench/` - Official ImpossibleBench repository

## Customizing Problems

You can customize which problems to run tests on using several methods:

### 1. List Available Problems

First, explore what problems are available:

```bash
# List first 20 problems
python explore/list_impossible_problems.py --split conflicting --limit 20

# Show detailed view of a problem
python explore/list_impossible_problems.py --split conflicting --task-id lcbhard_0

# Export task_ids to a file
python explore/list_impossible_problems.py --split conflicting --export my_problems.json
```

### 2. Filter by Task IDs

Run experiments on specific problems using their task_ids:

```bash
python explore/run_impossible_bench_integrated.py \
    --permission-level PL0 \
    --task-ids lcbhard_0,lcbhard_1,lcbhard_2
```

### 3. Filter by Indices

Run experiments on specific positions in the dataset:

```bash
python explore/run_impossible_bench_integrated.py \
    --permission-level PL0 \
    --indices 0,1,2,5,10
```

### 4. Custom Filter Functions (Python)

For more complex filtering, you can use Python code:

```python
from explore.run_impossible_bench_integrated import run_impossible_bench_experiment
from explore.poc_hardcoding import PermissionLevel

# Filter problems containing "fibonacci" in the prompt
def fibonacci_filter(record):
    return "fibonacci" in record.get("prompt", "").lower()

run_impossible_bench_experiment(
    permission_level=PermissionLevel.PL0_TEXT_ONLY,
    split="conflicting",
    custom_filter=fibonacci_filter,
    limit=10
)
```

### Tips

- **Use task_ids for precise selection**: Task IDs are stable identifiers that won't change if the dataset is reordered
- **Use indices for quick testing**: Good for testing specific positions, but less stable
- **Use limit + shuffle for random sampling**: `--limit 10 --shuffle` (if shuffle is supported)
- **Combine with splits**: Try different splits (`original`, `oneoff`, `conflicting`) to test different impossible test scenarios
