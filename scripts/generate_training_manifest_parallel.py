#!/usr/bin/env python3
"""
Generate Training Manifest from Local Cutouts (Parallel Version)

Scans .npz files on local NFS and extracts metadata in parallel.
Creates a parquet manifest for training.

Usage:
    python scripts/generate_training_manifest_parallel.py \
        --positives /lambda/nfs/.../cutouts/positives/TIMESTAMP/ \
        --negatives /lambda/nfs/.../cutouts/negatives/TIMESTAMP/ \
        --output /lambda/nfs/.../manifests/training_manifest.parquet \
        --workers 16

Author: Generated for stronglens_calibration project
Date: 2026-02-10
"""
import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime, timezone
from multiprocessing import Pool, cpu_count
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# =============================================================================
# CONSTANTS
# =============================================================================

PIPELINE_VERSION = "1.0.0"

# Tier-based sample weights (from project guidelines)
TIER_WEIGHTS = {
    "A": 1.0,      # Confirmed lenses - full weight
    "B": 0.5,      # Probable candidates - reduced weight
    "N1": 1.0,     # Random negatives
    "N2": 1.0,     # Hard confusers
}

# =============================================================================
# LOGGING
# =============================================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [ManifestGen] %(levelname)s: %(message)s",
)
logger = logging.getLogger("ManifestGen")


# =============================================================================
# METADATA EXTRACTION
# =============================================================================

def extract_metadata_from_npz(npz_path: str) -> Optional[Dict]:
    """Extract metadata from a single .npz file."""
    try:
        data = np.load(npz_path, allow_pickle=True)
        
        # Base info from path
        path = Path(npz_path)
        is_positive = "positives" in str(path)
        
        result = {
            "cutout_path": str(path),
            "filename": path.name,
            "galaxy_id": path.stem,
            "label": 1 if is_positive else 0,
            "cutout_type": "positive" if is_positive else "negative",
        }
        
        # Extract all meta_ prefixed fields
        for key in data.files:
            if key.startswith("meta_"):
                field_name = key[5:]  # Remove 'meta_' prefix
                value = data[key]
                
                # Handle numpy types
                if isinstance(value, np.ndarray):
                    if value.ndim == 0:
                        value = value.item()
                    elif len(value) == 1:
                        value = value[0]
                    else:
                        value = str(value)  # Convert arrays to string
                
                result[field_name] = value
        
        # Ensure critical fields exist
        if "ra" not in result:
            result["ra"] = None
        if "dec" not in result:
            result["dec"] = None
        if "split" not in result:
            result["split"] = None  # Will be assigned later if missing
        
        # Determine tier and sample weight
        if is_positive:
            # Check if tier is stored in metadata
            tier = result.get("tier", "B")  # Default to Tier-B if unknown
            result["tier"] = tier
            result["sample_weight"] = TIER_WEIGHTS.get(tier, 0.5)
        else:
            pool = result.get("pool", "N1")
            result["tier"] = pool
            result["sample_weight"] = TIER_WEIGHTS.get(pool, 1.0)
        
        return result
        
    except Exception as e:
        return {
            "cutout_path": npz_path,
            "filename": Path(npz_path).name,
            "galaxy_id": Path(npz_path).stem,
            "error": str(e),
        }


def extract_batch(paths: List[str]) -> List[Dict]:
    """Extract metadata from a batch of files (for Pool.map)."""
    return [extract_metadata_from_npz(p) for p in paths]


# =============================================================================
# FILE DISCOVERY
# =============================================================================

def find_npz_files(directory: str) -> List[str]:
    """Find all .npz files in directory recursively."""
    path = Path(directory)
    if not path.exists():
        logger.warning(f"Directory not found: {directory}")
        return []
    
    files = list(path.rglob("*.npz"))
    return [str(f) for f in files]


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Generate training manifest (parallel)")
    parser.add_argument("--positives", required=True, help="Path to positive cutouts directory")
    parser.add_argument("--negatives", required=True, help="Path to negative cutouts directory")
    parser.add_argument("--output", required=True, help="Output parquet file path")
    parser.add_argument("--workers", type=int, default=16, help="Number of parallel workers")
    parser.add_argument("--batch-size", type=int, default=1000, help="Batch size for progress reporting")
    args = parser.parse_args()
    
    start_time = time.time()
    logger.info("=" * 60)
    logger.info("Training Manifest Generator (Parallel)")
    logger.info("=" * 60)
    logger.info(f"Positives: {args.positives}")
    logger.info(f"Negatives: {args.negatives}")
    logger.info(f"Output: {args.output}")
    logger.info(f"Workers: {args.workers}")
    
    # Find all files
    logger.info("\nDiscovering cutout files...")
    pos_files = find_npz_files(args.positives)
    neg_files = find_npz_files(args.negatives)
    
    logger.info(f"  Positives: {len(pos_files):,}")
    logger.info(f"  Negatives: {len(neg_files):,}")
    
    if len(pos_files) == 0:
        logger.error("No positive cutouts found!")
        return 1
    if len(neg_files) == 0:
        logger.error("No negative cutouts found!")
        return 1
    
    all_files = pos_files + neg_files
    total = len(all_files)
    logger.info(f"  Total: {total:,}")
    
    # Extract metadata in parallel
    logger.info(f"\nExtracting metadata with {args.workers} workers...")
    
    results = []
    processed = 0
    
    with Pool(args.workers) as pool:
        # Process in batches for progress reporting
        batch_size = args.batch_size
        for i in range(0, total, batch_size):
            batch = all_files[i:i+batch_size]
            batch_results = pool.map(extract_metadata_from_npz, batch)
            results.extend(batch_results)
            processed += len(batch)
            
            if processed % 10000 == 0 or processed == total:
                elapsed = time.time() - start_time
                rate = processed / elapsed
                eta = (total - processed) / rate if rate > 0 else 0
                logger.info(f"  Progress: {processed:,}/{total:,} ({100*processed/total:.1f}%) - {rate:.0f} files/s - ETA: {eta:.0f}s")
    
    # Build dataframe
    logger.info("\nBuilding manifest dataframe...")
    df = pd.DataFrame(results)
    
    # Check for errors
    if "error" in df.columns:
        errors = df[df["error"].notna()]
        if len(errors) > 0:
            logger.warning(f"  {len(errors)} files had extraction errors")
            # Remove error rows
            df = df[df["error"].isna()].drop(columns=["error"])
    
    # Verify splits are assigned (from cutout metadata)
    if "split" in df.columns:
        split_counts = df["split"].value_counts()
        logger.info(f"  Split distribution from metadata: {dict(split_counts)}")
    
    # Statistics
    pos_count = (df["label"] == 1).sum()
    neg_count = (df["label"] == 0).sum()
    
    logger.info(f"\n  Final manifest: {len(df):,} samples")
    logger.info(f"    Positives: {pos_count:,}")
    logger.info(f"    Negatives: {neg_count:,}")
    logger.info(f"    Ratio: {neg_count/pos_count:.1f}:1")
    
    # Ensure output directory exists
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save manifest
    logger.info(f"\nSaving to {args.output}...")
    df.to_parquet(args.output, index=False)
    
    # Save summary JSON
    summary_path = output_path.with_suffix(".json")
    summary = {
        "pipeline_version": PIPELINE_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "total_samples": len(df),
        "positives": int(pos_count),
        "negatives": int(neg_count),
        "ratio": round(neg_count / pos_count, 1) if pos_count > 0 else 0,
        "splits": {
            split: int((df["split"] == split).sum())
            for split in ["train", "val", "test"]
            if "split" in df.columns
        },
        "columns": list(df.columns),
        "source_paths": {
            "positives": args.positives,
            "negatives": args.negatives,
        },
    }
    
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    
    elapsed = time.time() - start_time
    logger.info("\n" + "=" * 60)
    logger.info("MANIFEST GENERATION COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Total samples: {len(df):,}")
    logger.info(f"Output: {args.output}")
    logger.info(f"Elapsed: {elapsed:.1f}s ({len(df)/elapsed:.0f} files/s)")
    logger.info("=" * 60)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
