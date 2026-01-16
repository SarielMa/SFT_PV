%%writefile /nfs/roberts/scratch/pi_sjf37/gp528/FinBen/Llama_eightb_train/prepare_targeted_dpo_data_revised.py
#!/usr/bin/env python3
"""
REVISED: Prepare TARGETED DPO Dataset Using Actual Model Confusion Matrices

IMPROVEMENTS:
1. GUARANTEED different chosen vs rejected (retry logic + validation)
2. RE-WEIGHTED strategies based on actual confusion matrix analysis
3. Automatic filtering of identical pairs
4. Better error handling and logging

Key Features:
- Loads confusion matrices (code and sub-code level)
- Generates negatives weighted by actual confusion frequency
- Prioritizes most common confusions
- Creates realistic error patterns your model actually makes

Usage:
    python prepare_targeted_dpo_data_REVISED.py \
        --input_dir /path/to/arrow/dataset \
        --output_dir /path/to/dpo/dataset \
        --code_confusion_file code_confusion.csv \
        --subcode_confusion_file subcode_confusion.csv
"""

import os
import json
import random
import argparse
import csv
from typing import List, Dict, Any, Tuple, Optional
from pathlib import Path
from datasets import Dataset, load_from_disk
from collections import defaultdict


# ============================================================================
# LOAD CONFUSION MATRICES
# ============================================================================

def load_confusion_matrix(filepath: str) -> Dict[Tuple[str, str], int]:
    """
    Load confusion matrix from CSV file
    
    Returns:
        {(ground_truth, predicted): count}
    """
    
    confusion_dict = {}
    
    with open(filepath, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if 'Ground_Truth_Code' in row:
                # Code-level confusion
                gt = row['Ground_Truth_Code']
                pred = row['Predicted_Code']
            elif 'Ground_Truth_SubCode' in row:
                # Sub-code level confusion
                gt = row['Ground_Truth_SubCode']
                pred = row['Predicted_SubCode']
            else:
                continue
            
            count = int(row['Count'])
            confusion_dict[(gt, pred)] = count
    
    return confusion_dict


def build_weighted_confusion_pairs(confusion_dict: Dict[Tuple[str, str], int]) -> Dict[str, List[Tuple[str, float]]]:
    """
    Build confusion pairs with weights based on frequency
    
    Returns:
        {ground_truth: [(confused_value, weight), ...]}
    """
    
    confusion_pairs = defaultdict(list)
    
    for (gt, pred), count in confusion_dict.items():
        if gt != pred:  # Skip correct predictions
            confusion_pairs[gt].append((pred, count))
    
    # Normalize weights for each ground truth
    normalized_pairs = {}
    for gt, confusions in confusion_pairs.items():
        total = sum(count for _, count in confusions)
        normalized = [(pred, count / total) for pred, count in confusions]
        # Sort by weight descending
        normalized_pairs[gt] = sorted(normalized, key=lambda x: x[1], reverse=True)
    
    return normalized_pairs


def print_top_confusions(confusion_dict: Dict[Tuple[str, str], int], title: str, top_n: int = 10):
    """Print top N confusions for inspection"""
    
    print(f"\n{title}")
    print("=" * 80)
    sorted_confusions = sorted(confusion_dict.items(), key=lambda x: x[1], reverse=True)
    
    total_errors = sum(count for (gt, pred), count in confusion_dict.items() if gt != pred)
    
    for i, ((gt, pred), count) in enumerate(sorted_confusions[:top_n], 1):
        if gt != pred:  # Only show actual errors
            pct = 100 * count / total_errors if total_errors > 0 else 0
            print(f"{i:2d}. {gt:40s} → {pred:40s} : {count:3d} errors ({pct:4.1f}%)")


# ============================================================================
# CODE & SUB-CODE VALIDITY CHECKING
# ============================================================================

CODE_SUBCODES = {
    "InfoGive": ["Symptoms", "Diagnostics", "Drugs", "Prognosis", "Generalinformation", 
                 "Instruction", "carecoordination", "SchedulingAppt"],
    "InfoSeek": ["Symptoms", "Diagnostics", "Drugs", "Generalinformation", 
                 "carecoordination", "SchedulingAppt"],
    "InfoGiveSDOH": ["EconomicStability", "HealthCareAccessAndQuality", "NeighborhoodAndBuiltEnvironment"],
    "InfoSeekSDOH": ["EconomicStability", "HealthCareAccessAndQuality"],
    "PartnershipProvider": ["salutation", "signoff", "connection", "Appreciation/Gratitude",
                           "maintainCommunication", "build trust", "Alignment", 
                           "inviteCollaboration", "checkingUnderstanding/clarification"],
    "PartnershipPatient": ["salutation", "signoff", "Appreciation/Gratitude", "connection",
                          "Alignment", "build trust", "statePreferences", "expressOpinions",
                          "activeParticipation/involvement"],
    "SharedDecisionProvider": ["ShareOptions", "Approval/Reinforcement", "EncourageQuestions",
                               "SummarizeConfirmUnderstanding", "MakeDecision"],
    "SharedDecisionPatient": ["ApprovalofDecision/Reinforcement", "ExploreOptions", "SeekingApproval"],
    "Socioemotional/Empathy": ["ExpressConcern/unease", "Sadness/fear", "Reassurance",
                               "AcknowledgeSituation", "PositiveRemarks"]
}


def is_valid_code_subcode_pair(code: str, subcode: str) -> bool:
    """Check if code-subcode pairing is valid"""
    return code in CODE_SUBCODES and subcode in CODE_SUBCODES[code]


def annotations_are_identical(ann1: List[Dict], ann2: List[Dict]) -> bool:
    """
    Check if two annotation lists are identical
    
    Compares as sets to handle order differences
    """
    if len(ann1) != len(ann2):
        return False
    
    # Convert to comparable tuples
    set1 = {(a['Code'], a['Sub-code'], a['Span']) for a in ann1}
    set2 = {(a['Code'], a['Sub-code'], a['Span']) for a in ann2}
    
    return set1 == set2


# ============================================================================
# TARGETED NEGATIVE GENERATION (REVISED)
# ============================================================================

class TargetedNegativeGenerator:
    """
    Generate negatives based on actual model confusion patterns
    
    REVISED VERSION:
    - Re-weighted strategies based on confusion matrix analysis
    - Guaranteed different outputs with retry logic
    - Better handling of edge cases
    """
    
    def __init__(
        self, 
        code_confusion_file: str,
        subcode_confusion_file: str,
        seed: int = 42
    ):
        """
        Args:
            code_confusion_file: Path to code-level confusion CSV
            subcode_confusion_file: Path to sub-code level confusion CSV
            seed: Random seed
        """
        self.rng = random.Random(seed)
        
        # Load confusion matrices
        print("\nLoading confusion matrices...")
        self.code_confusions = load_confusion_matrix(code_confusion_file)
        self.subcode_confusions = load_confusion_matrix(subcode_confusion_file)
        
        # Build weighted confusion pairs
        self.code_confusion_pairs = build_weighted_confusion_pairs(self.code_confusions)
        self.subcode_confusion_pairs = build_weighted_confusion_pairs(self.subcode_confusions)
        
        # Print statistics
        total_code_errors = sum(count for (gt, pred), count in self.code_confusions.items() if gt != pred)
        total_subcode_errors = sum(count for (gt, pred), count in self.subcode_confusions.items() if gt != pred)
        
        print(f"✓ Loaded {len(self.code_confusions)} code-level confusion pairs")
        print(f"  Total code errors: {total_code_errors}")
        print(f"✓ Loaded {len(self.subcode_confusions)} sub-code level confusion pairs")
        print(f"  Total subcode errors: {total_subcode_errors}")
        
        print_top_confusions(self.code_confusions, "TOP 10 CODE CONFUSIONS", 10)
        print_top_confusions(self.subcode_confusions, "TOP 10 SUB-CODE CONFUSIONS", 10)
        
        # REVISED STRATEGY WEIGHTS based on confusion matrix analysis:
        # 
        # Code confusions: 508 total errors, max single error = 59 (11.6%)
        # Subcode confusions: 868 total errors, max single error = 29 (3.3%)
        # 
        # Code errors are more severe (higher individual frequency)
        # Subcode errors are more numerous (more diversity)
        # 
        # Strategy re-weighting:
        print("\n" + "="*80)
        print("STRATEGY WEIGHTS (REVISED based on confusion analysis)")
        print("="*80)
        
        self.strategies = [
            ("code_confusion", 0.25),              # ✓ Reduced but still substantial
            ("subcode_confusion", 0.35),           # ✓ Increased (more granular)
            ("code_and_subcode_confusion", 0.20),  # ✓ Keep (compound errors important)
            ("missing_annotation", 0.08),          # ⚠️ Keep at 8% (not 2%!)
            ("extra_annotation", 0.12),            # ✓ Increase to 12% (not 18%)
        ]
        
        for strategy, weight in self.strategies:
            print(f"  {strategy:35s}: {weight:.0%}")
        
        print("="*80)
    
    def generate_negative_with_retry(
        self, 
        ground_truth_annotations: List[Dict],
        context_info: Dict,
        max_attempts: int = 10
    ) -> Tuple[List[Dict], str]:
        """
        Generate negative with guaranteed difference from ground truth
        
        CRITICAL FIX: Retries until chosen != rejected
        
        Returns:
            (negative_annotations, strategy_used)
        """
        
        if not ground_truth_annotations:
            # Empty ground truth - add spurious annotation
            neg, strategy = self._add_spurious_annotation(context_info), "spurious_annotation"
            return neg, strategy
        
        for attempt in range(max_attempts):
            # Select strategy weighted by importance
            strategy = self.rng.choices(
                [s for s, _ in self.strategies],
                weights=[w for _, w in self.strategies]
            )[0]
            
            # Apply strategy
            if strategy == "subcode_confusion":
                result = self._apply_subcode_confusion(ground_truth_annotations)
            elif strategy == "code_confusion":
                result = self._apply_code_confusion(ground_truth_annotations)
            elif strategy == "code_and_subcode_confusion":
                result = self._apply_code_and_subcode_confusion(ground_truth_annotations)
            elif strategy == "missing_annotation":
                result = self._missing_annotation(ground_truth_annotations)
            elif strategy == "extra_annotation":
                result = self._extra_annotation(ground_truth_annotations, context_info)
            else:
                result = ground_truth_annotations
            
            # CRITICAL CHECK: Verify result is different
            if not annotations_are_identical(result, ground_truth_annotations):
                return result, strategy
        
        # Fallback after max_attempts: force a difference by removing one annotation
        print(f"⚠️  WARNING: Could not generate different negative after {max_attempts} attempts")
        print(f"   Forcing difference by removing one annotation")
        
        if len(ground_truth_annotations) > 1:
            result = ground_truth_annotations[:-1]  # Remove last
            return result, "forced_missing_annotation"
        else:
            # Single annotation - change its subcode to first confused option
            result = ground_truth_annotations.copy()
            original_subcode = result[0]["Sub-code"]
            
            # Try to find a confused subcode
            if original_subcode in self.subcode_confusion_pairs:
                confused_subcodes = [pred for pred, _ in self.subcode_confusion_pairs[original_subcode]]
                for confused in confused_subcodes:
                    if is_valid_code_subcode_pair(result[0]["Code"], confused):
                        result[0]["Sub-code"] = confused
                        return result, "forced_subcode_change"
            
            # Last resort: change to Generalinformation
            if is_valid_code_subcode_pair(result[0]["Code"], "Generalinformation"):
                result[0]["Sub-code"] = "Generalinformation"
                return result, "forced_subcode_to_general"
            
            # Ultimate fallback
            result[0]["Code"] = "InfoGive"
            result[0]["Sub-code"] = "Generalinformation"
            return result, "forced_code_change"
    
    def _apply_subcode_confusion(self, annotations: List[Dict]) -> List[Dict]:
        """
        Replace sub-code with commonly confused one (from actual errors)
        
        Example: Appreciation/Gratitude → signoff (29 actual errors)
        """
        if not annotations:
            return annotations
        
        # Try each annotation until we find one with confusion data
        indices = list(range(len(annotations)))
        self.rng.shuffle(indices)
        
        for idx in indices:
            original = annotations[idx]
            gt_subcode = original["Sub-code"]
            
            # Check if we have confusion data for this sub-code
            if gt_subcode in self.subcode_confusion_pairs:
                confusions = self.subcode_confusion_pairs[gt_subcode]
                
                # Sample confused sub-code weighted by actual error frequency
                confused_subcodes = [pred for pred, _ in confusions]
                weights = [weight for _, weight in confusions]
                
                confused_subcode = self.rng.choices(confused_subcodes, weights=weights)[0]
                
                # Only apply if valid with current code AND different from original
                if confused_subcode != gt_subcode and is_valid_code_subcode_pair(original["Code"], confused_subcode):
                    modified = annotations.copy()
                    modified[idx] = {
                        "Code": original["Code"],
                        "Sub-code": confused_subcode,
                        "Span": original["Span"]
                    }
                    return modified
                else:
                    # Try other valid confused subcodes
                    valid_confused = [
                        sc for sc in confused_subcodes 
                        if sc != gt_subcode and is_valid_code_subcode_pair(original["Code"], sc)
                    ]
                    if valid_confused:
                        modified = annotations.copy()
                        modified[idx] = {
                            "Code": original["Code"],
                            "Sub-code": self.rng.choice(valid_confused),
                            "Span": original["Span"]
                        }
                        return modified
        
        # No confusion found - return unchanged (will be caught by retry logic)
        return annotations
    
    def _apply_code_confusion(self, annotations: List[Dict]) -> List[Dict]:
        """
        Replace code with commonly confused one (from actual errors)
        
        Example: PartnershipPatient → InfoGive (59 actual errors)
        """
        if not annotations:
            return annotations
        
        # Try each annotation until we find one with confusion data
        indices = list(range(len(annotations)))
        self.rng.shuffle(indices)
        
        for idx in indices:
            original = annotations[idx]
            gt_code = original["Code"]
            
            # Check if we have confusion data for this code
            if gt_code in self.code_confusion_pairs:
                confusions = self.code_confusion_pairs[gt_code]
                
                # Sample confused code weighted by actual error frequency
                confused_codes = [pred for pred, _ in confusions]
                weights = [weight for _, weight in confusions]
                
                confused_code = self.rng.choices(confused_codes, weights=weights)[0]
                
                # Must be different from original
                if confused_code != gt_code:
                    # Pick a valid sub-code for the new code
                    # Prefer keeping same subcode if valid, otherwise pick random
                    if is_valid_code_subcode_pair(confused_code, original["Sub-code"]):
                        new_subcode = original["Sub-code"]
                    else:
                        valid_subcodes = CODE_SUBCODES.get(confused_code, ["Generalinformation"])
                        new_subcode = self.rng.choice(valid_subcodes)
                    
                    modified = annotations.copy()
                    modified[idx] = {
                        "Code": confused_code,
                        "Sub-code": new_subcode,
                        "Span": original["Span"]
                    }
                    return modified
        
        # No confusion found - return unchanged (will be caught by retry logic)
        return annotations
    
    def _apply_code_and_subcode_confusion(self, annotations: List[Dict]) -> List[Dict]:
        """
        Apply both code and sub-code confusion (compound error)
        """
        # First apply code confusion
        modified = self._apply_code_confusion(annotations)
        
        # Check if it changed
        if not annotations_are_identical(modified, annotations):
            # Then apply sub-code confusion with 60% probability
            if self.rng.random() > 0.4:
                modified2 = self._apply_subcode_confusion(modified)
                # Use second modification if it's different
                if not annotations_are_identical(modified2, modified):
                    return modified2
            return modified
        
        # If code confusion didn't work, try subcode confusion
        return self._apply_subcode_confusion(annotations)
    
    def _missing_annotation(self, annotations: List[Dict]) -> List[Dict]:
        """Remove one annotation (under-labeling)"""
        if len(annotations) <= 1:
            return annotations  # Can't remove from single annotation
        
        modified = annotations.copy()
        idx = self.rng.randint(0, len(modified) - 1)
        modified.pop(idx)
        return modified
    
    def _extra_annotation(self, annotations: List[Dict], context_info: Dict) -> List[Dict]:
        """Add an incorrect annotation (over-labeling)"""
        modified = annotations.copy()
        
        # Pick a random code that appears in confusions
        if self.code_confusion_pairs:
            all_confused_codes = []
            for gt_code, confusions in self.code_confusion_pairs.items():
                all_confused_codes.extend([pred for pred, _ in confusions])
            
            if all_confused_codes:
                spurious_code = self.rng.choice(all_confused_codes)
                
                # Pick valid sub-code
                valid_subcodes = CODE_SUBCODES.get(spurious_code, ["Generalinformation"])
                spurious_subcode = self.rng.choice(valid_subcodes)
                
                # Use part of existing span if available
                if annotations:
                    existing_span = self.rng.choice(annotations)["Span"]
                    words = existing_span.split()
                    if len(words) > 2:
                        # Take first half
                        span = " ".join(words[:len(words)//2])
                    else:
                        span = existing_span + " extra"
                else:
                    span = "spurious annotation"
                
                modified.append({
                    "Code": spurious_code,
                    "Sub-code": spurious_subcode,
                    "Span": span
                })
        
        return modified
    
    def _add_spurious_annotation(self, context_info: Dict) -> List[Dict]:
        """Add spurious annotation when ground truth is empty"""
        
        # Pick random code from confusions
        all_codes = list(self.code_confusion_pairs.keys())
        if all_codes:
            code = self.rng.choice(all_codes)
            valid_subcodes = CODE_SUBCODES.get(code, ["Generalinformation"])
            subcode = self.rng.choice(valid_subcodes)
        else:
            code = "InfoGive"
            subcode = "Generalinformation"
        
        return [{
            "Code": code,
            "Sub-code": subcode,
            "Span": "spurious annotation"
        }]


# ============================================================================
# DATASET CONVERSION (REVISED)
# ============================================================================

def extract_context_from_prompt(prompt: str) -> Dict[str, Any]:
    """Extract context information from the instruction prompt"""
    context = {}
    
    # Extract TO_PAT_YN
    if "TO_PAT_YN: Y" in prompt:
        context["TO_PAT_YN"] = "Y"
    elif "TO_PAT_YN: N" in prompt:
        context["TO_PAT_YN"] = "N"
    
    # Extract sentences
    for line in prompt.split("\n"):
        if line.startswith("Previous sentence:"):
            context["previous"] = line.replace("Previous sentence:", "").strip()
        elif line.startswith("Current sentence:"):
            context["current"] = line.replace("Current sentence:", "").strip()
        elif line.startswith("Next sentence:"):
            context["next"] = line.replace("Next sentence:", "").strip()
    
    return context


def convert_to_dpo_format(
    dataset: Dataset, 
    negative_generator: TargetedNegativeGenerator,
    negatives_per_sample: int = 1
) -> Dataset:
    """
    Convert standard dataset to DPO format with targeted negatives
    
    REVISED: Guarantees chosen != rejected for every pair
    """
    
    dpo_examples = []
    strategy_counts = defaultdict(int)
    skipped_identical = 0
    skipped_invalid = 0
    
    for sample in dataset:
        # Extract prompt and ground truth
        prompt = sample["entries"][0]["content"]
        chosen_response = sample["entries"][1]["content"]
        
        # Parse ground truth annotations
        try:
            chosen_annotations = json.loads(chosen_response)["results"]
        except:
            print(f"⚠️  Skipping sample {sample['id']}: invalid JSON")
            skipped_invalid += 1
            continue
        
        # Extract context
        context = extract_context_from_prompt(prompt)
        
        # Generate multiple negative examples
        for neg_idx in range(negatives_per_sample):
            # CRITICAL: Use retry logic to guarantee different annotations
            rejected_annotations, strategy = negative_generator.generate_negative_with_retry(
                chosen_annotations, context, max_attempts=10
            )
            
            strategy_counts[strategy] += 1
            
            rejected_response = json.dumps({"results": rejected_annotations}, indent=2)
            
            # VALIDATION: Double-check they're different
            if chosen_response == rejected_response:
                print(f"⚠️  CRITICAL: Identical pair detected for sample {sample['id']} (attempt {neg_idx+1})")
                skipped_identical += 1
                continue  # Skip this pair
            
            dpo_examples.append({
                "prompt": prompt,
                "chosen": chosen_response,
                "rejected": rejected_response,
                "metadata": json.dumps({
                    "strategy": strategy,
                    "context": context,
                    "original_id": sample["id"],
                    "num_chosen": len(chosen_annotations),
                    "num_rejected": len(rejected_annotations)
                })
            })
    
    print(f"\n✓ Generated {len(dpo_examples)} DPO pairs")
    if skipped_identical > 0:
        print(f"⚠️  Skipped {skipped_identical} identical pairs")
    if skipped_invalid > 0:
        print(f"⚠️  Skipped {skipped_invalid} samples with invalid JSON")
    
    print("\nStrategy Distribution:")
    for strategy, count in sorted(strategy_counts.items(), key=lambda x: x[1], reverse=True):
        pct = 100 * count / len(dpo_examples) if len(dpo_examples) > 0 else 0
        print(f"  {strategy:40s}: {count:5d} ({pct:5.1f}%)")
    
    return Dataset.from_list(dpo_examples)


# ============================================================================
# VERIFICATION & FILTERING (NEW)
# ============================================================================

def filter_identical_pairs(dpo_dataset: Dataset) -> Dataset:
    """
    Final safety check: Remove any remaining identical pairs
    
    This should catch zero pairs if generation logic is working correctly
    """
    
    print("\n" + "="*80)
    print("FILTERING IDENTICAL PAIRS (Final Safety Check)")
    print("="*80)
    
    filtered = []
    removed = 0
    
    for sample in dpo_dataset:
        if sample["chosen"] != sample["rejected"]:
            filtered.append(sample)
        else:
            removed += 1
            metadata = json.loads(sample["metadata"])
            print(f"⚠️  Removed identical pair: ID {metadata.get('original_id', 'unknown')}, Strategy: {metadata.get('strategy', 'unknown')}")
    
    if removed == 0:
        print("✅ PERFECT! No identical pairs found (generation logic working correctly)")
    else:
        print(f"⚠️  Filtered {removed} identical pairs ({100*removed/len(dpo_dataset):.1f}%)")
    
    print(f"✓ Kept {len(filtered)} valid pairs ({100*len(filtered)/len(dpo_dataset):.1f}%)")
    print("="*80)
    
    return Dataset.from_list(filtered)


def verify_and_print_samples(dpo_dataset: Dataset, n_samples: int = 3):
    """Print first N samples for manual inspection"""
    
    print("\n" + "="*80)
    print(f"SHOWING FIRST {n_samples} DPO SAMPLES")
    print("="*80)
    
    for i in range(min(n_samples, len(dpo_dataset))):
        sample = dpo_dataset[i]
        metadata = json.loads(sample["metadata"])
        
        # Parse annotations
        chosen = json.loads(sample["chosen"])["results"]
        rejected = json.loads(sample["rejected"])["results"]
        
        # Extract current sentence from prompt
        current_sentence = "N/A"
        for line in sample["prompt"].split("\n"):
            if line.startswith("Current sentence:"):
                current_sentence = line.replace("Current sentence:", "").strip()
                break
        
        print(f"\n{'='*80}")
        print(f"SAMPLE {i+1} (Strategy: {metadata['strategy']})")
        print(f"{'='*80}")
        print(f"Original ID: {metadata['original_id']}")
        print(f"TO_PAT_YN: {metadata['context'].get('TO_PAT_YN', 'N/A')}")
        print(f"Current sentence: {current_sentence[:100]}...")
        
        print(f"\n✓ CHOSEN ({len(chosen)} annotations):")
        for j, ann in enumerate(chosen, 1):
            print(f"  {j}. Code: {ann['Code']:30s} | Sub-code: {ann['Sub-code']:35s}")
            print(f"     Span: {ann['Span'][:70]}...")
        
        print(f"\n✗ REJECTED ({len(rejected)} annotations):")
        for j, ann in enumerate(rejected, 1):
            print(f"  {j}. Code: {ann['Code']:30s} | Sub-code: {ann['Sub-code']:35s}")
            print(f"     Span: {ann['Span'][:70]}...")
        
        # Highlight differences
        print(f"\n🔍 DIFFERENCES:")
        
        # Verification
        if chosen == rejected:
            print(f"  ❌ ERROR: Chosen and rejected are IDENTICAL!")
        else:
            print(f"  ✅ Confirmed: Chosen and rejected are DIFFERENT")
        
        # Find modified annotations
        chosen_set = {(a['Code'], a['Sub-code'], a['Span']) for a in chosen}
        rejected_set = {(a['Code'], a['Sub-code'], a['Span']) for a in rejected}
        
        added = rejected_set - chosen_set
        removed = chosen_set - rejected_set
        
        if added:
            print(f"  ➕ Added in rejected: {len(added)}")
            for code, subcode, span in added:
                print(f"     → {code}/{subcode}: {span[:50]}...")
        
        if removed:
            print(f"  ➖ Removed in rejected: {len(removed)}")
            for code, subcode, span in removed:
                print(f"     → {code}/{subcode}: {span[:50]}...")
        
        # Check for code/subcode changes on same span
        for ann_c in chosen:
            for ann_r in rejected:
                if ann_c['Span'] == ann_r['Span']:
                    if ann_c['Code'] != ann_r['Code']:
                        print(f"  🔄 Code change on '{ann_c['Span'][:40]}...':")
                        print(f"     {ann_c['Code']} → {ann_r['Code']}")
                    elif ann_c['Sub-code'] != ann_r['Sub-code']:
                        print(f"  🔄 Sub-code change on '{ann_c['Span'][:40]}...':")
                        print(f"     {ann_c['Code']}/{ann_c['Sub-code']} → {ann_c['Code']}/{ann_r['Sub-code']}")
    
    print("\n" + "="*80)


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Prepare targeted DPO dataset using confusion matrices (REVISED)")
    parser.add_argument("--input_dir", type=str, required=True,
                       help="Path to input Arrow dataset")
    parser.add_argument("--output_dir", type=str, required=True,
                       help="Path to save DPO dataset")
    parser.add_argument("--code_confusion_file", type=str, required=True,
                       help="Path to code-level confusion matrix CSV")
    parser.add_argument("--subcode_confusion_file", type=str, required=True,
                       help="Path to sub-code level confusion matrix CSV")
    parser.add_argument("--negatives_per_sample", type=int, default=1,
                       help="Number of negative examples per sample")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed")
    parser.add_argument("--print_samples", type=int, default=3,
                       help="Number of samples to print for inspection")
    
    args = parser.parse_args()
    
    print("="*80)
    print("TARGETED DPO DATASET PREPARATION (REVISED)")
    print("Using Actual Model Confusion Matrices")
    print("="*80)
    print("IMPROVEMENTS:")
    print("  ✓ Guaranteed different chosen vs rejected")
    print("  ✓ Re-weighted strategies based on confusion analysis")
    print("  ✓ Automatic filtering of identical pairs")
    print("  ✓ Better error handling and logging")
    print("="*80)
    print(f"Input: {args.input_dir}")
    print(f"Output: {args.output_dir}")
    print(f"Code confusions: {args.code_confusion_file}")
    print(f"Sub-code confusions: {args.subcode_confusion_file}")
    print(f"Negatives per sample: {args.negatives_per_sample}")
    print(f"Random seed: {args.seed}")
    print("="*80)
    
    # Load dataset
    print("\nLoading dataset...")
    dataset = load_from_disk(args.input_dir)
    print(f"✓ Loaded {len(dataset)} samples")
    
    # Create negative generator with confusion matrices
    negative_generator = TargetedNegativeGenerator(
        code_confusion_file=args.code_confusion_file,
        subcode_confusion_file=args.subcode_confusion_file,
        seed=args.seed
    )
    
    # Convert to DPO format
    print("\n" + "="*80)
    print("GENERATING DPO PREFERENCE PAIRS")
    print("="*80)
    
    dpo_dataset = convert_to_dpo_format(
        dataset,
        negative_generator,
        negatives_per_sample=args.negatives_per_sample
    )
    
    # Filter identical pairs (final safety check)
    dpo_dataset = filter_identical_pairs(dpo_dataset)
    
    # Print sample examples
    verify_and_print_samples(dpo_dataset, n_samples=args.print_samples)
    
    # Save
    print(f"\n{'='*80}")
    print(f"SAVING DPO DATASET")
    print(f"{'='*80}")
    print(f"Output directory: {args.output_dir}")
    
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    dpo_dataset.save_to_disk(args.output_dir)
    print("✓ Dataset saved")
    
    # Save configuration
    config = {
        "code_confusion_file": args.code_confusion_file,
        "subcode_confusion_file": args.subcode_confusion_file,
        "negatives_per_sample": args.negatives_per_sample,
        "seed": args.seed,
        "total_pairs": len(dpo_dataset),
        "original_samples": len(dataset),
        "strategies": dict(negative_generator.strategies),
        "version": "REVISED_v1.0",
        "improvements": [
            "Guaranteed different chosen vs rejected (retry logic)",
            "Re-weighted strategies based on confusion matrix analysis",
            "Automatic filtering of identical pairs",
            "Better validation and error handling"
        ]
    }
    
    with open(os.path.join(args.output_dir, "dpo_config.json"), "w") as f:
        json.dump(config, f, indent=2)
    
    print("✓ Configuration saved")
    
    print("\n" + "="*80)
    print("✅ TARGETED DPO DATASET PREPARATION COMPLETE")
    print("="*80)
    print(f"\nDataset statistics:")
    print(f"  Original samples: {len(dataset)}")
    print(f"  DPO pairs generated: {len(dpo_dataset)}")
    print(f"  Negatives per sample: {args.negatives_per_sample}")
    print(f"  Quality: ALL pairs verified different ✓")
    print(f"\nNext step:")
    print(f"  python train_dpo_annotator.py --train_data_path {args.output_dir}")


if __name__ == "__main__":
    main()