# Benchmark Tools

This directory contains tools for benchmarking AudioPlayground patches and comparing performance across multiple runs.

## Available Tools

### 1. benchmark_compare.py - Multi-Patch Benchmarking

Benchmark multiple patches and create comprehensive comparison plots.

**Features:**
- Benchmark all patches or specific patches
- Generate detailed comparison plots
- Save results to JSON for later analysis
- Performance metrics: min, mean, median, max, 95th/99th percentiles
- CPU headroom analysis
- Complexity vs performance analysis

**Usage:**

```bash
# Benchmark all patches
python scripts/benchmarks/benchmark_compare.py --all

# Benchmark specific patches
python scripts/benchmarks/benchmark_compare.py patch1.apr patch2.apr patch3.apr

# Custom settings
python scripts/benchmarks/benchmark_compare.py --all \
    --buffer-size 1024 \
    --iterations 2000 \
    --sample-rate 48000 \
    --output-dir ./my_results

# Regenerate plots from saved results
python scripts/benchmarks/benchmark_compare.py \
    --load benchmark_results_20260706_084743.json
```

**Output Files:**
- `benchmark_comparison.png` - Main comparison dashboard with 6 subplots
- `benchmark_percentiles.png` - Detailed percentile analysis
- `benchmark_results_[timestamp].json` - Raw results data

**Generated Plots:**

1. **Mean Processing Time** - Bar chart comparing processing times
2. **CPU Headroom** - Available CPU capacity per patch
3. **Processing Time Distribution** - Box plots showing variability
4. **Complexity vs Performance** - Scatter plot (modules vs time)
5. **Performance Metrics Table** - Top 10 performing patches
6. **Summary Statistics** - Overall benchmark summary

### 2. compare_benchmark_runs.py - Multi-Run Comparison

Compare benchmark results across multiple runs to track performance changes over time or between different configurations.

**Features:**
- Compare 2+ benchmark runs side-by-side
- Automatic change detection and analysis
- Identify improvements and regressions
- Detailed change analysis for 2-run comparisons
- Summary statistics and visualizations

**Usage:**

```bash
# Compare two runs
python scripts/benchmarks/compare_benchmark_runs.py \
    baseline.json optimized.json

# Compare multiple runs with custom labels
python scripts/benchmarks/compare_benchmark_runs.py \
    --labels "Baseline" "After Threading Fix" "After Component Cache" \
    run1.json run2.json run3.json

# Custom output directory
python scripts/benchmarks/compare_benchmark_runs.py \
    --output-dir ./comparisons \
    before.json after.json
```

**Output Files:**
- `benchmark_run_comparison.png` - Main comparison dashboard
- `benchmark_change_analysis.png` - Detailed change analysis (2 runs only)

**Generated Plots:**

1. **Processing Time Comparison** - Side-by-side bar charts
2. **CPU Headroom Comparison** - Headroom across all runs
3. **Performance Change Heatmap** - Percentage changes (2 runs)
4. **Summary Metrics** - Average time, headroom, passing patches
5. **95th Percentile - Top Variable Patches** - Focus on high-variance patches
6. **Run Comparison Summary Table** - Key statistics per run

**Change Analysis Plots (2 runs):**
1. **Top Changes** - Biggest improvements and regressions
2. **Before vs After Scatter** - Visual correlation
3. **Change Distribution** - Histogram of all changes
4. **Summary Statistics** - Overall change metrics

### 3. benchmark_patch.py - Single Patch Benchmarking

Original tool for benchmarking individual patches or all patches sequentially.

**Usage:**

```bash
# Benchmark single patch
python scripts/benchmarks/benchmark_patch.py examples/patches/demo.apr

# Benchmark all patches
python scripts/benchmarks/benchmark_patch.py --all

# Custom settings
python scripts/benchmarks/benchmark_patch.py \
    --buffer-size 512 \
    --iterations 1000 \
    examples/patches/303_voice.apr
```

## Performance Metrics Explained

### Processing Time Metrics
- **Min/Max** - Best and worst case performance
- **Mean** - Average processing time (most important)
- **Median** - Middle value, less affected by outliers
- **95th Percentile** - 95% of renders complete within this time
- **99th Percentile** - 99% of renders complete within this time

### CPU Headroom
CPU Headroom represents how much processing capacity remains available:

```
Headroom = 100% × (1 - mean_time / time_budget)
```

- **>50%** - ✅ Excellent (plenty of CPU available)
- **20-50%** - ✓ Good (acceptable margin)
- **0-20%** - ⚠️ Marginal (may cause dropouts under load)
- **<0%** - ❌ Overbudget (real-time processing not possible)

### Time Budget
The maximum time available to render a buffer:

```
Time Budget = (buffer_size / sample_rate) × 1,000,000 µs
```

Example: 512 samples @ 44100 Hz = 11,609.98 µs

## Interpreting Results

### Good Performance Signs
- Mean time well below budget (>50% headroom)
- Low variance (min/max close together)
- Consistent percentiles
- No failed renders

### Warning Signs
- Mean time close to budget (<20% headroom)
- High variance (max >> mean)
- Failed renders during benchmark
- 99th percentile exceeding budget

### Optimization Workflow

1. **Baseline Benchmark**
   ```bash
   python scripts/benchmarks/benchmark_compare.py --all
   ```

2. **Make Optimizations**
   - Optimize hot paths
   - Reduce allocations
   - Vectorize operations
   - Cache computations

3. **Run New Benchmark**
   ```bash
   python scripts/benchmarks/benchmark_compare.py --all
   ```

4. **Compare Results**
   ```bash
   python scripts/benchmarks/compare_benchmark_runs.py \
       --labels "Before" "After" \
       baseline.json optimized.json
   ```

5. **Analyze Changes**
   - Check overall average change
   - Identify improved patches
   - Look for regressions
   - Verify no new failures

## Example Results

After running benchmarks on 20 patches:

```
Total Patches:           20
Passing (>0% headroom):  20 (100%)
Excellent (>50% headroom): 15 (75%)

Average Processing Time: 3,421.5 µs
Average CPU Headroom:    70.5%

Buffer Size:             512 samples
Sample Rate:             44100 Hz
Time Budget:             11,610.0 µs
```

## Tips for Accurate Benchmarks

1. **Close Other Applications** - Minimize background processes
2. **Consistent Environment** - Same machine, same settings
3. **Sufficient Iterations** - 500-1000 iterations for stable results
4. **Warmup** - Scripts include automatic warmup passes
5. **Multiple Runs** - Average results from 3-5 runs for best accuracy

## Troubleshooting

### "Failed to load patch"
- Patch may be incompatible with headless rendering
- Check for GUI-specific dependencies
- Verify patch file is not corrupted

### "No successful renders"
- Patch may have runtime errors
- Check mod_synth.log for details
- Try running patch in GUI first

### Matplotlib warnings
- Font warnings can be ignored
- Plots will still generate correctly
- Update matplotlib to suppress warnings

## Files in This Directory

- `benchmark_compare.py` - Multi-patch benchmarking with plots
- `compare_benchmark_runs.py` - Multi-run comparison tool
- `benchmark_patch.py` - Original single/all patch tool
- `benchmark_engine_presets.py` - Engine-level benchmarks
- `benchmark_plots/` - Generated plots and results
- `README.md` - This file

## Dependencies

Required packages (already in AudioPlayground):
- matplotlib
- numpy
- statistics (stdlib)
- json (stdlib)

## See Also

- `docs/BENCHMARK_BASELINES.md` - Historical benchmark results
- `examples/patches/` - Available test patches
- Main project README for general usage

