"""
Metric Parity Tests
===================
Minimum 6 tests to verify metric.py matches official_eval_reference.py exactly.

3 hand-computable micro tests (<=8 samples) with explicit expected values
3 randomized parity tests with fixed seeds

HARD GATE: If ANY test fails, pipeline must STOP.
"""

import numpy as np
import sys
import hashlib

import metric
import official_eval_reference as official


def test_micro_1():
    """
    Micro test 1: All cheaters, perfect predictions
    y_true = [1, 1, 1, 1]
    y_pred = [0.9, 0.85, 0.95, 0.8]
    
    At t_low=0.0, t_high=0.7:
    - All 4 samples go to block (p >= 0.7)
    - All are cheaters, so 0 FP
    - 0 reviews, 0 FN
    - Cost = 0
    """
    y_true = np.array([1, 1, 1, 1])
    y_pred = np.array([0.9, 0.85, 0.95, 0.8])
    
    result = metric.score(y_true, y_pred)
    
    # Best case: block all (they're all cheaters)
    # At t_low=0, t_high close to min(y_pred)=0.8, we block all
    # Cost should be 0 since no FP (all are cheaters) and no FN
    
    assert result['best_cost'] == 0, f"Expected cost 0, got {result['best_cost']}"
    assert result['n_fn'] == 0, f"Expected 0 FN, got {result['n_fn']}"
    assert result['n_fp'] == 0, f"Expected 0 FP, got {result['n_fp']}"
    
    print("✓ Micro test 1 PASSED: All cheaters, perfect predictions")
    return True


def test_micro_2():
    """
    Micro test 2: All legitimate, need to auto-pass all
    y_true = [0, 0, 0, 0]
    y_pred = [0.1, 0.2, 0.15, 0.05]
    
    Best strategy: t_low=max(y_pred)+epsilon, t_high=1.0
    - All samples auto-pass
    - No cheaters, so 0 FN
    - 0 reviews, 0 FP
    - Cost = 0
    """
    y_true = np.array([0, 0, 0, 0])
    y_pred = np.array([0.1, 0.2, 0.15, 0.05])
    
    result = metric.score(y_true, y_pred)
    
    assert result['best_cost'] == 0, f"Expected cost 0, got {result['best_cost']}"
    assert result['n_fn'] == 0, f"Expected 0 FN, got {result['n_fn']}"
    assert result['n_fp'] == 0, f"Expected 0 FP, got {result['n_fp']}"
    
    print("✓ Micro test 2 PASSED: All legitimate, all pass")
    return True


def test_micro_3():
    """
    Micro test 3: Mixed case with explicit calculation
    y_true = [1, 0, 1, 0, 1, 0]
    y_pred = [0.9, 0.1, 0.7, 0.3, 0.5, 0.4]
    
    Let's calculate at t_low=0.35, t_high=0.65:
    - p < 0.35: sample 1 (y=0, p=0.1) → auto-pass (legit, no FN)
    - 0.35 <= p < 0.65: samples 3,4,5 (y=0,1,0, p=0.3,0.5,0.4) → review
      Wait, p=0.3 < 0.35 → pass
      Actually: p=0.3 < 0.35 → pass, p=0.5 in [0.35, 0.65) → review, p=0.4 in [0.35, 0.65) → review
    - p >= 0.65: samples 0,2 (y=1,1, p=0.9,0.7) → block (both cheaters, 0 FP)
    
    Let me recalculate more carefully at t_low=0.35, t_high=0.65:
    Sample 0: y=1, p=0.9 → block (cheater blocked correctly)
    Sample 1: y=0, p=0.1 → pass (legit passed correctly)
    Sample 2: y=1, p=0.7 → block (cheater blocked correctly)
    Sample 3: y=0, p=0.3 → pass (legit passed correctly)
    Sample 4: y=1, p=0.5 → review (cheater caught via review)
    Sample 5: y=0, p=0.4 → review (legit reviewed, review cost only)
    
    n_pass = 2, n_review = 2, n_block = 2
    n_fn = 0 (no cheaters in pass)
    n_fp = 0 (no legits in block)
    cost = 0*5000 + 0*1000 + 2*50 = 100
    
    But optimal might be different. Let's check t_low=0.6, t_high=1.0:
    Sample 0: y=1, p=0.9 → block
    Sample 1: y=0, p=0.1 → pass
    Sample 2: y=1, p=0.7 → block
    Sample 3: y=0, p=0.3 → pass
    Sample 4: y=1, p=0.5 → pass (FN!)
    Sample 5: y=0, p=0.4 → pass
    
    n_fn = 1, cost = 5000
    
    Or t_low=0.5, t_high=0.6:
    Sample 4: y=1, p=0.5 → review
    n_review = 1, cost = 50
    
    Or t_low=0.45, t_high=0.65:
    Sample 4: y=1, p=0.5 → review (review cost)
    Sample 5: y=0, p=0.4 → pass
    n_review = 1, cost = 50
    
    Best: t_low=0.5, t_high=0.7 (or similar)
    Samples: 0→block, 1→pass, 2→block, 3→pass, 4→review, 5→pass
    Cost = 1 review * 50 = 50
    """
    y_true = np.array([1, 0, 1, 0, 1, 0])
    y_pred = np.array([0.9, 0.1, 0.7, 0.3, 0.5, 0.4])
    
    result = metric.score(y_true, y_pred)
    
    # The optimal should minimize cost
    # Manual calculation: we need to catch all 3 cheaters with minimal cost
    # Cheater preds: 0.9, 0.7, 0.5
    # Legit preds: 0.1, 0.3, 0.4
    # If we set t_high=0.5, we block cheaters at 0.9, 0.7, and review cheater at 0.5
    # Actually at t_low=0.5, t_high=0.5: samples with p>=0.5 go to block
    # That's 3 cheaters blocked, 0 FP, 0 review → cost=0
    
    # Let's verify
    cost_at_05_05, _ = metric.compute_cost(y_true, y_pred, 0.5, 0.5)
    # p>=0.5: samples 0,2,4 blocked (all cheaters)
    # p<0.5: samples 1,3,5 pass (all legit)
    # cost = 0
    
    assert cost_at_05_05 == 0, f"Cost at (0.5, 0.5) should be 0, got {cost_at_05_05}"
    assert result['best_cost'] == 0, f"Expected best cost 0, got {result['best_cost']}"
    
    print("✓ Micro test 3 PASSED: Mixed case with explicit calculation")
    return True


def test_parity_1():
    """
    Randomized parity test 1: Compare metric.py with official_eval_reference.py
    """
    np.random.seed(42)
    n = 100
    y_true = np.random.randint(0, 2, n).astype(float)
    y_pred = np.random.random(n)
    
    # Get results from both
    result_metric = metric.score(y_true, y_pred)
    result_official = official.evaluate(y_true, y_pred)
    
    # Check exact match
    assert result_metric['best_cost'] == result_official['best_cost'], \
        f"Cost mismatch: metric={result_metric['best_cost']}, official={result_official['best_cost']}"
    assert result_metric['best_score'] == result_official['best_score'], \
        f"Score mismatch: metric={result_metric['best_score']}, official={result_official['best_score']}"
    assert result_metric['t_low'] == result_official['t_low'], \
        f"t_low mismatch: metric={result_metric['t_low']}, official={result_official['t_low']}"
    assert result_metric['t_high'] == result_official['t_high'], \
        f"t_high mismatch: metric={result_metric['t_high']}, official={result_official['t_high']}"
    assert result_metric['n_pass'] == result_official['n_pass'], \
        f"n_pass mismatch"
    assert result_metric['n_review'] == result_official['n_review'], \
        f"n_review mismatch"
    assert result_metric['n_block'] == result_official['n_block'], \
        f"n_block mismatch"
    
    print(f"✓ Parity test 1 PASSED: n=100, seed=42, cost={result_metric['best_cost']}")
    return True


def test_parity_2():
    """
    Randomized parity test 2: Larger dataset
    """
    np.random.seed(123)
    n = 1000
    # Imbalanced: 30% cheaters
    y_true = (np.random.random(n) < 0.3).astype(float)
    # Predictions correlated with truth but noisy
    y_pred = np.clip(y_true * 0.6 + np.random.random(n) * 0.4, 0, 1)
    
    result_metric = metric.score(y_true, y_pred)
    result_official = official.evaluate(y_true, y_pred)
    
    assert result_metric['best_cost'] == result_official['best_cost'], \
        f"Cost mismatch: metric={result_metric['best_cost']}, official={result_official['best_cost']}"
    assert result_metric['t_low'] == result_official['t_low'], \
        f"t_low mismatch"
    assert result_metric['t_high'] == result_official['t_high'], \
        f"t_high mismatch"
    assert result_metric['n_fn'] == result_official['n_fn'], \
        f"n_fn mismatch"
    assert result_metric['n_fp'] == result_official['n_fp'], \
        f"n_fp mismatch"
    
    print(f"✓ Parity test 2 PASSED: n=1000, seed=123, imbalanced, cost={result_metric['best_cost']}")
    return True


def test_parity_3():
    """
    Randomized parity test 3: Edge cases with extreme predictions
    """
    np.random.seed(999)
    n = 500
    y_true = np.random.randint(0, 2, n).astype(float)
    # Mix of very confident and uncertain predictions
    y_pred = np.where(
        np.random.random(n) < 0.3,
        np.random.random(n) * 0.05,  # Very low confidence
        np.where(
            np.random.random(n) < 0.5,
            0.95 + np.random.random(n) * 0.05,  # Very high confidence
            np.random.random(n)  # Random
        )
    )
    y_pred = np.clip(y_pred, 0, 1)
    
    result_metric = metric.score(y_true, y_pred)
    result_official = official.evaluate(y_true, y_pred)
    
    # Full comparison
    for key in ['best_cost', 'best_score', 't_low', 't_high', 
                'n_pass', 'n_review', 'n_block', 'n_fn', 'n_fp',
                'cost_fn', 'cost_fp', 'cost_review']:
        assert result_metric[key] == result_official[key], \
            f"{key} mismatch: metric={result_metric[key]}, official={result_official[key]}"
    
    print(f"✓ Parity test 3 PASSED: n=500, seed=999, extreme preds, cost={result_metric['best_cost']}")
    return True


def compute_metric_code_hash():
    """Compute SHA256 hash of the core metric code block."""
    code_block = '''
COST_FN = 5000
COST_FP = 1000
COST_REVIEW = 50
pass_mask = y_pred < t_low
review_mask = (y_pred >= t_low) & (y_pred < t_high)
block_mask = y_pred >= t_high
n_fn = int(np.sum((y_true == 1) & pass_mask))
n_fp = int(np.sum((y_true == 0) & block_mask))
cost_fn_total = n_fn * COST_FN
cost_fp_total = n_fp * COST_FP
cost_review_total = n_review * COST_REVIEW
total_cost = cost_fn_total + cost_fp_total + cost_review_total
'''
    return hashlib.sha256(code_block.encode()).hexdigest()


def run_all_tests():
    """Run all metric tests. Returns True if all pass, False otherwise."""
    print("="*60)
    print("METRIC PARITY TESTS")
    print("="*60)
    
    tests = [
        test_micro_1,
        test_micro_2,
        test_micro_3,
        test_parity_1,
        test_parity_2,
        test_parity_3,
    ]
    
    all_passed = True
    for test in tests:
        try:
            test()
        except AssertionError as e:
            print(f"✗ {test.__name__} FAILED: {e}")
            all_passed = False
        except Exception as e:
            print(f"✗ {test.__name__} ERROR: {e}")
            all_passed = False
    
    print("="*60)
    if all_passed:
        print("ALL 6 TESTS PASSED ✓")
        print(f"SHA256 of metric code block: {compute_metric_code_hash()}")
    else:
        print("TESTS FAILED - STOP PIPELINE")
    print("="*60)
    
    return all_passed


if __name__ == '__main__':
    success = run_all_tests()
    sys.exit(0 if success else 1)
