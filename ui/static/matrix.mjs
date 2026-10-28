/* Exact saved cutoffs: no score interpolation or threshold fitting in the browser. */
export function fraction(numerator, denominator) {
  return denominator ? numerator / denominator : null;
}

export function matrixAt(sweep, index) {
  if (!Number.isInteger(index) || index < 0 || index >= sweep.threshold.length) {
    throw new RangeError('Choose a saved threshold index.');
  }
  const result = Object.fromEntries(['tp', 'tn', 'fp', 'fn'].map(key => [key, sweep[key][index]]));
  result.positives = result.tp + result.fn;
  result.negatives = result.tn + result.fp;
  result.ecgs = result.positives + result.negatives;
  result.sensitivity = fraction(result.tp, result.positives);
  result.specificity = fraction(result.tn, result.negatives);
  result.precision = fraction(result.tp, result.tp + result.fp);
  result.referral_fraction = fraction(result.tp + result.fp, result.ecgs);
  result.threshold = sweep.threshold[index];
  return result;
}

export function nearestIndex(thresholds, value) {
  let selected = 0;
  thresholds.forEach((threshold, index) => {
    if (Math.abs(threshold - value) < Math.abs(thresholds[selected] - value)) selected = index;
  });
  return selected;
}

export function parseCanonicalCSV(text) {
  const leads = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6'];
  const lines = text.replace(/^\uFEFF/, '').trim().split(/\r?\n/);
  if (lines.length !== 5001 || lines[0].split(',').map(s => s.trim()).join(',') !== leads.join(',')) {
    throw new Error('CSV must have the canonical 12-lead header and exactly 5000 sample rows.');
  }
  const samples = leads.map(() => []);
  lines.slice(1).forEach((line, row) => {
    const values = line.split(',').map(s => s.trim());
    if (values.length !== 12) throw new Error(`Expected twelve values on sample row ${row + 1}.`);
    values.forEach((raw, lead) => {
      if (raw === '' || /^(?:[+-]?(?:nan|inf(?:inity)?))$/i.test(raw)) samples[lead].push(null);
      else {
        const value = Number(raw);
        if (!Number.isFinite(value)) throw new Error(`Invalid sample on row ${row + 1}.`);
        samples[lead].push(value);
      }
    });
  });
  return {samples, leads, sample_rate: 500, units: 'mV'};
}
