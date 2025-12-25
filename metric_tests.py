"""
Metric tests to verify parity with official evaluation.
Tests include:
1. Hand-computable tests (at least 3)
2. Parity tests vs official_eval_reference.py (at least 3)
"""

import numpy as np
import sys

from official_eval_reference import compute_cost, evaluate, find_optimal_thresholds
from official_eval_reference import MISS_CHEATER_COST, BLOCK_CLEAN_COST, MANUAL_REVIEW_COST
from metric import score, score_with_details, score_at_thresholds, get_thresholds


def test_hand_computable_1():
    """
    Test 1: All predictions below threshold (all auto-pass)
    
    Setup:
    - y_true = [1, 1, 0, 0]  (2 cheaters, 2 clean)
    - y_pred = [0.1, 0.1, 0.1, 0.1]  (all low confidence)
    - t_low = 0.5, t_high = 0.8
    
    Expected:
    - All 4 samples in auto-pass region
    - 2 missed cheaters -> cost = 2 * 500 = 1000
    - 0 manual reviews
    - 0 blocked clean
    - Total = 1000
    """
    y_true = np.array([1, 1, 0, 0])
    y_pred = np.array([0.1, 0.1, 0.1, 0.1])
    
    result = compute_cost(y_true, y_pred, t_low=0.5, t_high=0.8)
    
    assert result['n_auto_pass'] == 4, f"Expected 4 auto-pass, got {result['n_auto_pass']}"
    assert result['n_manual_review'] == 0, f"Expected 0 manual review, got {result['n_manual_review']}"
    assert result['n_auto_block'] == 0, f"Expected 0 auto-block, got {result['n_auto_block']}"
    assert result['missed_cheaters'] == 2, f"Expected 2 missed cheaters, got {result['missed_cheaters']}"
    assert result['cost_missed_cheaters'] == 1000, f"Expected cost 1000, got {result['cost_missed_cheaters']}"
    assert result['total_cost'] == 1000, f"Expected total 1000, got {result['total_cost']}"
    
    print("✓ Test 1 PASSED: All auto-pass scenario")


def test_hand_computable_2():
    """
    Test 2: All predictions in manual review region
    
    Setup:
    - y_true = [1, 1, 0, 0]  (2 cheaters, 2 clean)
    - y_pred = [0.5, 0.6, 0.55, 0.65]  (all medium confidence)
    - t_low = 0.4, t_high = 0.7
    
    Expected:
    - All 4 samples in manual review region
    - Manual review cost = 4 * 50 = 200
    - 0 missed cheaters, 0 blocked clean
    - Total = 200
    """
    y_true = np.array([1, 1, 0, 0])
    y_pred = np.array([0.5, 0.6, 0.55, 0.65])
    
    result = compute_cost(y_true, y_pred, t_low=0.4, t_high=0.7)
    
    assert result['n_auto_pass'] == 0, f"Expected 0 auto-pass, got {result['n_auto_pass']}"
    assert result['n_manual_review'] == 4, f"Expected 4 manual review, got {result['n_manual_review']}"
    assert result['n_auto_block'] == 0, f"Expected 0 auto-block, got {result['n_auto_block']}"
    assert result['cost_manual_review'] == 200, f"Expected manual cost 200, got {result['cost_manual_review']}"
    assert result['total_cost'] == 200, f"Expected total 200, got {result['total_cost']}"
    
    print("✓ Test 2 PASSED: All manual review scenario")


def test_hand_computable_3():
    """
    Test 3: Mixed regions
    
    Setup:
    - y_true = [1, 1, 0, 0, 1, 0]  (3 cheaters, 3 clean)
    - y_pred = [0.1, 0.5, 0.9, 0.1, 0.9, 0.5]
    - t_low = 0.3, t_high = 0.7
    
    Expected regions:
    - y_pred[0]=0.1 < 0.3: auto-pass, y_true=1 -> missed cheater
    - y_pred[1]=0.5 in [0.3, 0.7): manual review
    - y_pred[2]=0.9 >= 0.7: auto-block, y_true=0 -> blocked clean
    - y_pred[3]=0.1 < 0.3: auto-pass, y_true=0 -> correct
    - y_pred[4]=0.9 >= 0.7: auto-block, y_true=1 -> correct
    - y_pred[5]=0.5 in [0.3, 0.7): manual review
    
    Costs:
    - 1 missed cheater: 1 * 500 = 500
    - 2 manual reviews: 2 * 50 = 100
    - 1 blocked clean: 1 * 100 = 100
    - Total = 700
    """
    y_true = np.array([1, 1, 0, 0, 1, 0])
    y_pred = np.array([0.1, 0.5, 0.9, 0.1, 0.9, 0.5])
    
    result = compute_cost(y_true, y_pred, t_low=0.3, t_high=0.7)
    
    assert result['n_auto_pass'] == 2, f"Expected 2 auto-pass, got {result['n_auto_pass']}"
    assert result['n_manual_review'] == 2, f"Expected 2 manual review, got {result['n_manual_review']}"
    assert result['n_auto_block'] == 2, f"Expected 2 auto-block, got {result['n_auto_block']}"
    assert result['missed_cheaters'] == 1, f"Expected 1 missed cheater, got {result['missed_cheaters']}"
    assert result['blocked_clean'] == 1, f"Expected 1 blocked clean, got {result['blocked_clean']}"
    assert result['cost_missed_cheaters'] == 500, f"Expected missed cost 500, got {result['cost_missed_cheaters']}"
    assert result['cost_manual_review'] == 100, f"Expected manual cost 100, got {result['cost_manual_review']}"
    assert result['cost_blocked_clean'] == 100, f"Expected block cost 100, got {result['cost_blocked_clean']}"
    assert result['total_cost'] == 700, f"Expected total 700, got {result['total_cost']}"
    
    print("✓ Test 3 PASSED: Mixed regions scenario")


def test_parity_1():
    """
    Parity test 1: Verify metric.score matches official_eval_reference.evaluate
    """
    np.random.seed(42)
    y_true = np.random.randint(0, 2, 100)
    y_pred = np.random.random(100)
    
    # Get score from metric.py
    metric_score = score(y_true, y_pred, n_thresholds=50)
    
    # Get score from official_eval_reference.py
    official_result = evaluate(y_true, y_pred, n_thresholds=50)
    official_score = official_result['total_cost']
    
    assert metric_score == official_score, f"Parity failed: metric={metric_score}, official={official_score}"
    
    print("✓ Parity test 1 PASSED: score() matches evaluate()")


def test_parity_2():
    """
    Parity test 2: Verify score_at_thresholds matches compute_cost
    """
    np.random.seed(123)
    y_true = np.random.randint(0, 2, 50)
    y_pred = np.random.random(50)
    
    t_low, t_high = 0.3, 0.7
    
    # Get score from metric.py
    metric_score = score_at_thresholds(y_true, y_pred, t_low, t_high)
    
    # Get score from official_eval_reference.py
    official_result = compute_cost(y_true, y_pred, t_low, t_high)
    official_score = official_result['total_cost']
    
    assert metric_score == official_score, f"Parity failed: metric={metric_score}, official={official_score}"
    
    print("✓ Parity test 2 PASSED: score_at_thresholds() matches compute_cost()")


def test_parity_3():
    """
    Parity test 3: Verify get_thresholds matches find_optimal_thresholds
    """
    np.random.seed(456)
    y_true = np.random.randint(0, 2, 75)
    y_pred = np.random.random(75)
    
    # Get thresholds from metric.py
    metric_t_low, metric_t_high = get_thresholds(y_true, y_pred, n_thresholds=30)
    
    # Get thresholds from official_eval_reference.py
    official_t_low, official_t_high, _ = find_optimal_thresholds(y_true, y_pred, n_thresholds=30)
    
    assert metric_t_low == official_t_low, f"t_low mismatch: metric={metric_t_low}, official={official_t_low}"
    assert metric_t_high == official_t_high, f"t_high mismatch: metric={metric_t_high}, official={official_t_high}"
    
    print("✓ Parity test 3 PASSED: get_thresholds() matches find_optimal_thresholds()")


def test_edge_cases():
    """
    Test edge cases
    """
    # Edge case 1: All same predictions
    y_true = np.array([1, 0, 1, 0])
    y_pred = np.array([0.5, 0.5, 0.5, 0.5])
    result = evaluate(y_true, y_pred, n_thresholds=50)
    assert result['total_cost'] >= 0, "Cost should be non-negative"
    
    # Edge case 2: Perfect predictions
    y_true = np.array([1, 1, 0, 0])
    y_pred = np.array([1.0, 1.0, 0.0, 0.0])
    result = evaluate(y_true, y_pred, n_thresholds=100)
    # With perfect predictions, optimal is t_low=0.01, t_high=0.99 or similar
    # This should give very low cost
    assert result['total_cost'] == 0 or result['total_cost'] < 100, f"Perfect predictions should have low cost, got {result['total_cost']}"
    
    # Edge case 3: All cheaters
    y_true = np.array([1, 1, 1, 1])
    y_pred = np.array([0.1, 0.9, 0.5, 0.3])
    result = evaluate(y_true, y_pred, n_thresholds=50)
    assert result['blocked_clean'] == 0, "No clean users to block"
    
    print("✓ Edge cases PASSED")


def test_cost_constants():
    """
    Verify cost constants are set correctly
    """
    assert MISS_CHEATER_COST == 500, f"MISS_CHEATER_COST should be 500, got {MISS_CHEATER_COST}"
    assert BLOCK_CLEAN_COST == 100, f"BLOCK_CLEAN_COST should be 100, got {BLOCK_CLEAN_COST}"
    assert MANUAL_REVIEW_COST == 50, f"MANUAL_REVIEW_COST should be 50, got {MANUAL_REVIEW_COST}"
    
    print("✓ Cost constants verified")


def run_all_tests():
    """Run all tests and report results."""
    tests = [
        ("Hand-computable Test 1", test_hand_computable_1),
        ("Hand-computable Test 2", test_hand_computable_2),
        ("Hand-computable Test 3", test_hand_computable_3),
        ("Parity Test 1", test_parity_1),
        ("Parity Test 2", test_parity_2),
        ("Parity Test 3", test_parity_3),
        ("Edge Cases", test_edge_cases),
        ("Cost Constants", test_cost_constants),
    ]
    
    passed = 0
    failed = 0
    
    print("=" * 60)
    print("RUNNING METRIC TESTS")
    print("=" * 60)
    print()
    
    for name, test_func in tests:
        try:
            test_func()
            passed += 1
        except AssertionError as e:
            print(f"✗ {name} FAILED: {e}")
            failed += 1
        except Exception as e:
            print(f"✗ {name} ERROR: {e}")
            failed += 1
    
    print()
    print("=" * 60)
    print(f"RESULTS: {passed}/{len(tests)} tests passed, {failed} failed")
    print("=" * 60)
    
    if failed > 0:
        print("\n*** TESTS FAILED - FIX BEFORE PROCEEDING ***")
        sys.exit(1)
    else:
        print("\n*** ALL TESTS PASSED - METRIC VERIFIED ***")
        sys.exit(0)


if __name__ == "__main__":
    run_all_tests()
