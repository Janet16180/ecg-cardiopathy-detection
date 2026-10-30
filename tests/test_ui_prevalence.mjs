import assert from 'node:assert/strict';
import test from 'node:test';
import {adjustedIndex, atPrevalence, predictive, wilson} from '../ui/static/prevalence.mjs';

test('Bayes rule gives the textbook predictive values', () => {
  const value = predictive(0.9, 0.95, 0.01);
  assert.ok(Math.abs(value.ppv - 0.009 / (0.009 + 0.0495)) < 1e-12);
  assert.ok(Math.abs(value.npv - 0.9405 / (0.9405 + 0.001)) < 1e-12);
});

test('Wilson interval brackets the observed proportion', () => {
  const [low, high] = wilson(2460, 3584);
  assert.ok(low < 2460 / 3584 && 2460 / 3584 < high);
  assert.deepEqual(wilson(0, 0), [null, null]);
});

test('expected counts add up to the screened population', () => {
  const matrix = {tp:70, fn:30, fp:50, tn:950, positives:100, negatives:1000, sensitivity:0.7, specificity:0.95, threshold:0.5};
  const value = atPrevalence(matrix, 0.01, 1000);
  assert.ok(Math.abs(value.found + value.missed - 10) < 1e-12);
  assert.ok(Math.abs(value.referred - value.found - value.falseAlarms) < 1e-9);
  assert.throws(() => atPrevalence(matrix, 0, 1000), RangeError);
});

test('a larger referral tolerance never picks a higher cutoff', () => {
  const sweep = {threshold:[0.2, 0.5, 0.8], tp:[95, 70, 30], fn:[5, 30, 70], fp:[400, 50, 5], tn:[600, 950, 995]};
  const strict = adjustedIndex(sweep, 0.01, 2), loose = adjustedIndex(sweep, 0.01, 200);
  assert.ok(loose.index <= strict.index);
});
