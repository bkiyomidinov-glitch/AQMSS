"""
Dataset Builder for AMQSS AI Layer (Simplified & Robust)
Purpose: Build a clean, time-aligned dataset from AMQSS output + future performance data.
Constraints: No data leakage, no shuffling, time-based split only.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os


def load_amqss_results(csv_path):
    """Load AMQSS results (market_scores.csv). Robust to CSV issues."""
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Results file not found: {csv_path}")
    
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"[DatasetBuilder] Warning: CSV parsing error ({e}). Attempting lenient read...")
        try:
            df = pd.read_csv(csv_path, on_bad_lines='skip')
        except:
            df = pd.read_csv(csv_path, error_bad_lines=False)
    
    # Convert timestamp to datetime - handle both formats (with/without microseconds)
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'], format='mixed', utc=False)
    df = df.sort_values('timestamp').reset_index(drop=True)
    return df


def load_price_data(csv_path):
    """Load raw candle data (EURUSD_60_2025-01-20_2026-01-19.csv)."""
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Price data not found: {csv_path}")
    
    # Read with header=None because the CSV has corrupted headers (first row is data)
    df = pd.read_csv(csv_path, header=None)
    
    # Assign column names
    df.columns = ['time', 'open', 'high', 'low', 'close', 'volume', 'volume2', 'other']
    
    # The first row is actually a header row with column names - skip it
    df = df.iloc[1:].reset_index(drop=True)
    
    # Ensure numeric
    for col in ['open', 'high', 'low', 'close', 'volume']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    df = df.dropna()
    
    # Parse time - handle format like "2025.01.20 02:00:00"
    df['time'] = pd.to_datetime(df['time'].str.replace('.', '-'), format='mixed', errors='coerce')
    df = df.dropna(subset=['time'])
    
    df = df.sort_values('time').reset_index(drop=True)
    return df


def compute_forward_performance(price_df, lookback_bars=20):
    """
    Compute forward_return and max_drawdown over the next N bars.
    
    Args:
        price_df: DataFrame with 'close' column, sorted by time
        lookback_bars: Number of future bars to look ahead
    
    Returns:
        Same dataframe with additional columns: forward_return, max_drawdown
    """
    df = price_df.copy()
    
    forward_returns = []
    max_drawdowns = []
    
    for i in range(len(df)):
        # Cannot compute forward performance if not enough bars remain
        if i + lookback_bars >= len(df):
            forward_returns.append(np.nan)
            max_drawdowns.append(np.nan)
            continue
        
        # Get future window (strictly forward, no look-ahead bias)
        future_close = df['close'].iloc[i+1:i+1+lookback_bars].values
        entry_price = df['close'].iloc[i]
        
        # Forward return: (future_end - entry) / entry
        exit_price = future_close[-1]
        forward_ret = (exit_price - entry_price) / entry_price if entry_price != 0 else 0
        forward_returns.append(forward_ret)
        
        # Max drawdown: (lowest - entry) / entry
        min_price = future_close.min()
        max_dd = (min_price - entry_price) / entry_price if entry_price != 0 else 0
        max_drawdowns.append(max_dd)
    
    df['forward_return'] = forward_returns
    df['max_drawdown'] = max_drawdowns
    
    return df


def build_dataset(
    amqss_csv_path,
    price_csv_path,
    output_csv_path,
    lookback_bars=20,
    return_threshold=0.002,
    drawdown_threshold=-0.01
):
    """
    Full pipeline: Load, align, compute forward perf, generate labels, save.
    
    Args:
        amqss_csv_path: Path to market_scores.csv
        price_csv_path: Path to price candle data
        output_csv_path: Path to save cleaned dataset
        lookback_bars: Window for forward performance
        return_threshold: Min return for positive label
        drawdown_threshold: Max drawdown for positive label
    """
    print("[DatasetBuilder] Loading AMQSS results...")
    amqss_df = load_amqss_results(amqss_csv_path)
    print(f"  Loaded {len(amqss_df)} AMQSS records")
    
    print("[DatasetBuilder] Loading price data...")
    price_df = load_price_data(price_csv_path)
    print(f"  Loaded {len(price_df)} price candles")
    
    print("[DatasetBuilder] Computing forward performance...")
    price_df = compute_forward_performance(price_df, lookback_bars=lookback_bars)
    valid_perf = price_df['forward_return'].notna().sum()
    print(f"  Computed forward performance for {valid_perf} candles")
    
    # Align AMQSS with price data
    print("[DatasetBuilder] Aligning AMQSS with price data...")
    result = []
    
    for _, amqss_row in amqss_df.iterrows():
        amqss_time = amqss_row['timestamp']
        
        # Find closest price candle (increase tolerance to 24 hours since data spans different periods)
        price_df['time_diff'] = (price_df['time'] - amqss_time).abs()
        closest_idx = price_df['time_diff'].idxmin()
        time_diff_seconds = price_df.loc[closest_idx, 'time_diff'].total_seconds()
        
        # If no match within 1 day, skip this AMQSS record
        if time_diff_seconds > 24 * 60 * 60:
            print(f"  Warning: No price data match for {amqss_time}, gap={time_diff_seconds/3600:.1f}h")
            continue
        
        # Get aligned row
        price_row = price_df.iloc[closest_idx]
        
        # Merge AMQSS + price performance
        merged_row = dict(amqss_row)
        merged_row['forward_return'] = price_row['forward_return']
        merged_row['max_drawdown'] = price_row['max_drawdown']
        
        result.append(merged_row)
    
    aligned_df = pd.DataFrame(result)
    print(f"  Aligned {len(aligned_df)} records")
    
    # Generate labels
    print("[DatasetBuilder] Generating labels...")
    
    if len(aligned_df) == 0:
        print("  ERROR: No AMQSS records aligned with price data!")
        print("  Check that AMQSS timestamps fall within price data range.")
        print(f"  This often happens when using datetime.now() vs historical data.")
        return None
    
    aligned_df['label'] = (
        (aligned_df['forward_return'] >= return_threshold) & 
        (aligned_df['max_drawdown'] >= drawdown_threshold)
    ).astype(int)
    label_dist = aligned_df['label'].value_counts()
    print(f"  Label distribution: {label_dist.to_dict()}")
    
    # Extract features (AMQSS metrics only)
    print("[DatasetBuilder] Building feature set...")
    feature_cols = [
        'volatility_regime', 'true_liquidity', 'structure', 
        'momentum', 'volatility_level', 'current_atr'
    ]
    
    # Verify all features exist
    missing = [c for c in feature_cols if c not in aligned_df.columns]
    if missing:
        raise ValueError(f"Missing feature columns: {missing}")
    
    # Keep only valid rows (where forward performance was computed)
    dataset_df = aligned_df[aligned_df['forward_return'].notna()].copy()
    
    if len(dataset_df) == 0:
        print("  ERROR: No samples with valid forward performance!")
        return None
    
    # Select output columns: features + label + metadata
    output_cols = feature_cols + ['label', 'timestamp']
    dataset_df = dataset_df[output_cols].reset_index(drop=True)
    
    print(f"  Final dataset: {len(dataset_df)} samples")
    
    # Save
    os.makedirs(os.path.dirname(output_csv_path) or '.', exist_ok=True)
    dataset_df.to_csv(output_csv_path, index=False)
    print(f"[DatasetBuilder] Dataset saved to {output_csv_path}")
    
    return dataset_df


# Keep old functions for reference (not used in main pipeline)
def align_amqss_with_prices(amqss_df, price_df, tolerance_minutes=5):
    """Legacy function - not used."""
    pass


def generate_labels(df, return_threshold=0.002, drawdown_threshold=-0.01):
    """Legacy function - not used."""
    pass


def build_feature_set(df, feature_cols=None):
    """Legacy function - not used."""
    pass


if __name__ == '__main__':
    # Example usage
    amqss_path = r'C:\Users\bkiyo\Desktop\Project.X\results\market_scores.csv'
    price_path = r'C:\Users\bkiyo\Downloads\EURUSD_60_2025-01-20_2026-01-19.csv'
    output_path = r'C:\Users\bkiyo\Desktop\Project.X\data\dataset.csv'
    
    try:
        dataset = build_dataset(amqss_path, price_path, output_path)
        print(f"\n✓ Dataset ready: {output_path}")
    except Exception as e:
        print(f"\n✗ Error: {e}")
