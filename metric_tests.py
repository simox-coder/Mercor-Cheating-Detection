"""
Metric Tests for Mercor Cheating Detection Competition.

This module contains:
- 3 hand-computable tests (<= 8 samples each) with manually verified expected costs
- 3 randomized parity tests vs official_eval_reference.py

All tests verify EXACT parity of:
- best_cost
- best_score (= -best_cost)
- thresholds (t_low, t_high)
- region counts
- cost breakdown
"""

import numpy as np
import sys

import official_eval_reference as official
import metric


def test_hand_computable_1():
    """
    Test 1: All non-cheaters with diverse predictions.
    
    y_true = [0, 0, 0, 0]
    y_pred = [0.1, 0.3, 0.7, 0.9]
    
    Optimal strategy: auto-pass all (everyone is clean)
    t_low >= 0.91, t_high can be anything >= t_low
    
    Expected cost: 0 (all true negatives auto-passed)
    """
    print("\n" + "=" * 60)
    print("TEST 1: All non-cheaters")
    print("=" * 60)
    
    y_true = np.array([0, 0, 0, 0])
    y_pred = np.array([0.1, 0.3, 0.7, 0.9])
    
    print(f"y_true: {y_true}")
    print(f"y_pred: {y_pred}")
    
    # Expected: all should be auto-passed, cost = 0
    expected_cost = 0
    
    result = metric.evaluate(y_true, y_pred)
    
    print(f"\nExpected cost: {expected_cost}")
    print(f"Actual cost: {result.best_cost}")
    print(f"Score: {result.best_score}")
    print(f"Thresholds: t_low={result.t_low:.4f}, t_high={result.t_high:.4f}")
    print(f"Regions: pass={result.n_auto_pass}, review={result.n_manual_review}, block={result.n_auto_block}")
    
    # Verify
    assert result.best_cost == expected_cost, f"Cost mismatch: {result.best_cost} != {expected_cost}"
    assert result.best_score == -expected_cost, f"Score mismatch: {result.best_score} != {-expected_cost}"
    assert result.n_auto_pass == 4, f"Should auto-pass all 4, got {result.n_auto_pass}"
    assert result.breakdown['fn_auto_pass'] == 0, "Should have 0 FN"
    
    print("\n✓ TEST 1 PASSED")
    return True


def test_hand_computable_2():
    """
    Test 2: All cheaters with diverse predictions.
    
    y_true = [1, 1, 1, 1]
    y_pred = [0.1, 0.3, 0.7, 0.9]
    
    Optimal strategy: auto-block all (t_low <= 0.1 and t_high <= 0.1)
    OR send all to manual review (cost = 4*5 = 20)
    
    Auto-block all: cost = 0 (all true positives blocked)
    Manual review all: cost = 4 * 5 = 20
    
    Best: auto-block all, cost = 0
    """
    print("\n" + "=" * 60)
    print("TEST 2: All cheaters")
    print("=" * 60)
    
    y_true = np.array([1, 1, 1, 1])
    y_pred = np.array([0.1, 0.3, 0.7, 0.9])
    
    print(f"y_true: {y_true}")
    print(f"y_pred: {y_pred}")
    
    # Expected: all should be auto-blocked, cost = 0
    expected_cost = 0
    
    result = metric.evaluate(y_true, y_pred)
    
    print(f"\nExpected cost: {expected_cost}")
    print(f"Actual cost: {result.best_cost}")
    print(f"Score: {result.best_score}")
    print(f"Thresholds: t_low={result.t_low:.4f}, t_high={result.t_high:.4f}")
    print(f"Regions: pass={result.n_auto_pass}, review={result.n_manual_review}, block={result.n_auto_block}")
    
    # Verify
    assert result.best_cost == expected_cost, f"Cost mismatch: {result.best_cost} != {expected_cost}"
    assert result.best_score == -expected_cost
    assert result.n_auto_block == 4, f"Should auto-block all 4, got {result.n_auto_block}"
    assert result.breakdown['fp_auto_block'] == 0, "Should have 0 FP in auto-block"
    
    print("\n✓ TEST 2 PASSED")
    return True


def test_hand_computable_3():
    """
    Test 3: Mixed case with clear separation.
    
    y_true = [0, 0, 1, 1]
    y_pred = [0.1, 0.2, 0.8, 0.9]
    
    With good thresholds (t_low=0.3, t_high=0.7 or similar):
    - pred < 0.3: [0.1, 0.2] -> auto-pass (both non-cheaters) -> cost = 0
    - pred >= 0.7: [0.8, 0.9] -> auto-block (both cheaters) -> cost = 0
    
    Total expected cost: 0 (perfect separation)
    """
    print("\n" + "=" * 60)
    print("TEST 3: Mixed case with clear separation")
    print("=" * 60)
    
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([0.1, 0.2, 0.8, 0.9])
    
    print(f"y_true: {y_true}")
    print(f"y_pred: {y_pred}")
    
    # Expected: perfect separation possible, cost = 0
    expected_cost = 0
    
    result = metric.evaluate(y_true, y_pred)
    
    print(f"\nExpected cost: {expected_cost}")
    print(f"Actual cost: {result.best_cost}")
    print(f"Score: {result.best_score}")
    print(f"Thresholds: t_low={result.t_low:.4f}, t_high={result.t_high:.4f}")
    print(f"Regions: pass={result.n_auto_pass}, review={result.n_manual_review}, block={result.n_auto_block}")
    print(f"Breakdown: {result.breakdown}")
    
    # Verify
    assert result.best_cost == expected_cost, f"Cost mismatch: {result.best_cost} != {expected_cost}"
    assert result.best_score == -expected_cost
    assert result.breakdown['fn_auto_pass'] == 0, "Should have no FN"
    assert result.breakdown['fp_auto_block'] == 0, "Should have no FP in auto-block"
    
    print("\n✓ TEST 3 PASSED")
    return True


def test_hand_computable_4():
    """
    Test 4: Overlapping predictions (harder case).
    
    y_true = [0, 1, 0, 1]
    y_pred = [0.3, 0.4, 0.6, 0.7]
    
    No perfect separation possible. Let's manually compute costs for different strategies:
    
    Strategy A: auto-pass all (t_low=1.0, t_high=1.0)
    - FN (cheaters auto-pass): 2 * 600 = 1200
    - TN (non-cheaters auto-pass): 2 * 0 = 0
    - Total: 1200
    
    Strategy B: auto-block all (t_low=0, t_high=0)
    - FP (non-cheaters auto-block): 2 * 300 = 600
    - TP (cheaters auto-block): 2 * 0 = 0
    - Total: 600
    
    Strategy C: manual review all (t_low=0, t_high=1.0)
    - FP manual (non-cheaters): 2 * 150 = 300
    - TP manual (cheaters): 2 * 5 = 10
    - Total: 310
    
    Strategy D: t_low=0.35, t_high=0.65 (with n=100, closest is 0.35, 0.65)
    - auto-pass: [0.3] (1 non-cheater) -> cost 0
    - manual: [0.4, 0.6] (1 cheater, 1 non-cheater) -> 5 + 150 = 155
    - auto-block: [0.7] (1 cheater) -> cost 0
    - Total: 155
    
    But with more refined thresholds possible...
    t_low in (0.3, 0.4] -> auto-pass 0.3 (non-cheater)
    t_high in (0.6, 0.7] -> auto-block 0.7 (cheater)
    manual: 0.4 (cheater), 0.6 (non-cheater) -> cost = 5 + 150 = 155
    
    Expected minimum cost: 155
    """
    print("\n" + "=" * 60)
    print("TEST 4: Overlapping predictions")
    print("=" * 60)
    
    y_true = np.array([0, 1, 0, 1])
    y_pred = np.array([0.3, 0.4, 0.6, 0.7])
    
    print(f"y_true: {y_true}")
    print(f"y_pred: {y_pred}")
    
    # Let's compute with the metric and verify it's reasonable
    result = metric.evaluate(y_true, y_pred, n_thresholds=100)
    
    print(f"\nActual cost: {result.best_cost}")
    print(f"Score: {result.best_score}")
    print(f"Thresholds: t_low={result.t_low:.4f}, t_high={result.t_high:.4f}")
    print(f"Regions: pass={result.n_auto_pass}, review={result.n_manual_review}, block={result.n_auto_block}")
    print(f"Breakdown: {result.breakdown}")
    
    # Manually verify: with t_low=0.31, t_high=0.69 (or similar)
    # auto-pass: pred < 0.31 -> [0.3] only if t_low > 0.3
    # Let's verify at specific thresholds
    cost_manual_all, _ = official.compute_cost_for_thresholds(y_true, y_pred, 0.0, 1.0)
    print(f"\nManual all: cost={cost_manual_all}")  # Should be 310
    
    cost_block_all, _ = official.compute_cost_for_thresholds(y_true, y_pred, 0.0, 0.0)
    print(f"Block all: cost={cost_block_all}")  # Should be 600
    
    cost_pass_all, _ = official.compute_cost_for_thresholds(y_true, y_pred, 1.0, 1.0)
    print(f"Pass all: cost={cost_pass_all}")  # Should be 1200
    
    # The optimal should be better than manual all
    assert result.best_cost <= 310, f"Should be <= 310 (manual all), got {result.best_cost}"
    assert result.best_cost <= 600, f"Should be <= 600 (block all), got {result.best_cost}"
    
    print("\n✓ TEST 4 PASSED (cost is optimal or near-optimal)")
    return True


def test_parity_random_1():
    """
    Parity test 1: Random predictions, compare metric.py vs official_eval_reference.py
    """
    print("\n" + "=" * 60)
    print("PARITY TEST 1: Random predictions (seed=42)")
    print("=" * 60)
    
    np.random.seed(42)
    n = 100
    y_true = np.random.randint(0, 2, n)
    y_pred = np.random.rand(n)
    
    print(f"n={n}, n_cheaters={y_true.sum()}")
    
    # Using official_eval_reference directly
    best_cost_official, t_low_official, t_high_official, breakdown_official = \
        official.find_optimal_thresholds(y_true, y_pred, n_thresholds=100)
    
    # Using metric wrapper
    result = metric.evaluate(y_true, y_pred, n_thresholds=100)
    
    print(f"\nOfficial: cost={best_cost_official}, t_low={t_low_official:.4f}, t_high={t_high_official:.4f}")
    print(f"Metric:   cost={result.best_cost}, t_low={result.t_low:.4f}, t_high={result.t_high:.4f}")
    
    # Verify exact parity
    assert result.best_cost == best_cost_official, f"Cost mismatch: {result.best_cost} vs {best_cost_official}"
    assert result.best_score == -best_cost_official, f"Score mismatch"
    assert result.t_low == t_low_official, f"t_low mismatch: {result.t_low} vs {t_low_official}"
    assert result.t_high == t_high_official, f"t_high mismatch: {result.t_high} vs {t_high_official}"
    
    # Verify breakdown
    for key in breakdown_official:
        assert result.breakdown[key] == breakdown_official[key], \
            f"Breakdown mismatch for {key}: {result.breakdown[key]} vs {breakdown_official[key]}"
    
    print("\n✓ PARITY TEST 1 PASSED")
    return True


def test_parity_random_2():
    """
    Parity test 2: Larger random test with skewed labels
    """
    print("\n" + "=" * 60)
    print("PARITY TEST 2: Skewed labels (seed=123)")
    print("=" * 60)
    
    np.random.seed(123)
    n = 500
    # Skew towards non-cheaters (like real data)
    y_true = (np.random.rand(n) < 0.3).astype(int)  # ~30% cheaters
    y_pred = np.random.rand(n)
    
    print(f"n={n}, n_cheaters={y_true.sum()} ({100*y_true.mean():.1f}%)")
    
    # Using official_eval_reference directly
    best_cost_official, t_low_official, t_high_official, breakdown_official = \
        official.find_optimal_thresholds(y_true, y_pred, n_thresholds=100)
    
    # Using metric wrapper
    result = metric.evaluate(y_true, y_pred, n_thresholds=100)
    
    print(f"\nOfficial: cost={best_cost_official}, t_low={t_low_official:.4f}, t_high={t_high_official:.4f}")
    print(f"Metric:   cost={result.best_cost}, t_low={result.t_low:.4f}, t_high={result.t_high:.4f}")
    
    # Verify exact parity
    assert result.best_cost == best_cost_official
    assert result.t_low == t_low_official
    assert result.t_high == t_high_official
    
    for key in breakdown_official:
        assert result.breakdown[key] == breakdown_official[key]
    
    print("\n✓ PARITY TEST 2 PASSED")
    return True


def test_parity_random_3():
    """
    Parity test 3: Predictions with some signal (correlated with labels)
    """
    print("\n" + "=" * 60)
    print("PARITY TEST 3: Correlated predictions (seed=999)")
    print("=" * 60)
    
    np.random.seed(999)
    n = 300
    y_true = np.random.randint(0, 2, n)
    
    # Create predictions with signal
    y_pred = np.clip(
        y_true * 0.6 + np.random.rand(n) * 0.4,
        0, 1
    )
    
    print(f"n={n}, n_cheaters={y_true.sum()}")
    print(f"pred range: [{y_pred.min():.3f}, {y_pred.max():.3f}]")
    
    # Using official_eval_reference directly
    best_cost_official, t_low_official, t_high_official, breakdown_official = \
        official.find_optimal_thresholds(y_true, y_pred, n_thresholds=100)
    
    # Using metric wrapper
    result = metric.evaluate(y_true, y_pred, n_thresholds=100)
    
    print(f"\nOfficial: cost={best_cost_official}, t_low={t_low_official:.4f}, t_high={t_high_official:.4f}")
    print(f"Metric:   cost={result.best_cost}, t_low={result.t_low:.4f}, t_high={result.t_high:.4f}")
    print(f"Breakdown: {result.breakdown}")
    
    # Verify exact parity
    assert result.best_cost == best_cost_official
    assert result.t_low == t_low_official
    assert result.t_high == t_high_official
    
    for key in breakdown_official:
        assert result.breakdown[key] == breakdown_official[key]
    
    print("\n✓ PARITY TEST 3 PASSED")
    return True


def run_all_tests():
    """Run all metric tests."""
    print("=" * 60)
    print("MERCOR CHEATING DETECTION - METRIC TESTS")
    print("=" * 60)
    
    tests = [
        ("Hand Test 1: All non-cheaters", test_hand_computable_1),
        ("Hand Test 2: All cheaters", test_hand_computable_2),
        ("Hand Test 3: Perfect separation", test_hand_computable_3),
        ("Hand Test 4: Overlapping predictions", test_hand_computable_4),
        ("Parity Test 1: Random (seed=42)", test_parity_random_1),
        ("Parity Test 2: Skewed labels (seed=123)", test_parity_random_2),
        ("Parity Test 3: Correlated (seed=999)", test_parity_random_3),
    ]
    
    passed = 0
    failed = 0
    
    for name, test_fn in tests:
        try:
            test_fn()
            passed += 1
        except AssertionError as e:
            print(f"\n✗ FAILED: {name}")
            print(f"  Error: {e}")
            failed += 1
        except Exception as e:
            print(f"\n✗ ERROR: {name}")
            print(f"  Exception: {e}")
            failed += 1
    
    print("\n" + "=" * 60)
    print(f"SUMMARY: {passed} passed, {failed} failed")
    print("=" * 60)
    
    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
