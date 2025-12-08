# Performance Report Template

## Experiment Information

| Field | Value |
|-------|-------|
| Experiment ID | |
| Date | |
| Config File | |
| Model | |
| Training Mode | |
| Dataset | |

## Training Summary

### Configuration

```yaml
# Paste relevant config here
```

### Training Metrics

| Epoch | Train Loss | Val Loss | Learning Rate |
|-------|------------|----------|---------------|
| 1 | | | |
| 2 | | | |
| 3 | | | |

### Resource Usage

| Metric | Value |
|--------|-------|
| Total Time | |
| GPU Memory Peak | |
| Total Steps | |
| Samples/Second | |

## Evaluation Results

### Automatic Metrics

| Metric | Value |
|--------|-------|
| Harmlessness Rate | |
| ROUGE-L | |
| BERTScore F1 | |

### Per-paradigm Comparison

| Paradigm | Primary Metric | Secondary Metric |
|----------|----------------|------------------|
| SFT | | |
| RL | | |
| Staged | | |

## Sample Outputs

### Example 1

**Prompt:**
```
[Insert prompt]
```

**SFT Output:**
```
[Insert output]
```

**RL Output:**
```
[Insert output]
```

**Staged Output:**
```
[Insert output]
```

## Statistical Analysis

### Paired Comparison

| Comparison | Mean Diff | 95% CI | p-value |
|------------|-----------|--------|---------|
| SFT vs RL | | | |
| SFT vs Staged | | | |
| RL vs Staged | | | |

## Observations

### Strengths

- 

### Weaknesses

- 

### Recommendations

- 

## Checkpoints

| Checkpoint | Path | Description |
|------------|------|-------------|
| Best | | |
| Final | | |
| LoRA | | |
