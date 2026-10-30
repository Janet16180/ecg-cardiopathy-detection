/* Bayes estimates for an assumed screening prevalence, from exact saved cutoff matrices. */
import {matrixAt} from './matrix.mjs';

export function wilson(successes, total, z = 1.96) {
  if (!total) return [null, null];
  const p = successes / total, scale = 1 + z * z / total;
  const centre = (p + z * z / (2 * total)) / scale;
  const half = z * Math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / scale;
  return [Math.max(0, centre - half), Math.min(1, centre + half)];
}

export function predictive(sensitivity, specificity, prevalence) {
  const flagged = prevalence * sensitivity + (1 - prevalence) * (1 - specificity);
  const cleared = 1 - flagged;
  return {
    ppv: flagged > 0 ? prevalence * sensitivity / flagged : null,
    npv: cleared > 0 ? (1 - prevalence) * specificity / cleared : null,
    flagged,
  };
}

export function atPrevalence(matrix, prevalence, people) {
  if (!(prevalence > 0 && prevalence < 1)) throw new RangeError('Prevalence must be between 0 and 1.');
  const [sensitivityLow, sensitivityHigh] = wilson(matrix.tp, matrix.positives);
  const [specificityLow, specificityHigh] = wilson(matrix.tn, matrix.negatives);
  const point = predictive(matrix.sensitivity, matrix.specificity, prevalence);
  const low = predictive(sensitivityLow, specificityLow, prevalence);
  const high = predictive(sensitivityHigh, specificityHigh, prevalence);
  const cases = prevalence * people;
  return {
    threshold: matrix.threshold, sensitivity: matrix.sensitivity, specificity: matrix.specificity,
    cases, found: cases * matrix.sensitivity, missed: cases * (1 - matrix.sensitivity),
    falseAlarms: (people - cases) * (1 - matrix.specificity), referred: people * point.flagged,
    ppv: point.ppv, npv: point.npv, ppvRange: [low.ppv, high.ppv], npvRange: [low.npv, high.npv],
  };
}

export function netBenefit(matrix, prevalence, acceptable) {
  return prevalence * matrix.sensitivity - (1 - prevalence) * (1 - matrix.specificity) / acceptable;
}

export function adjustedIndex(sweep, prevalence, acceptable) {
  if (!(acceptable > 0)) throw new RangeError('Acceptable referrals per case must be positive.');
  let best = null;
  sweep.threshold.forEach((_, index) => {
    const benefit = netBenefit(matrixAt(sweep, index), prevalence, acceptable);
    if (best === null || benefit > best.benefit) best = {index, benefit};
  });
  return best;
}

const escape = value => String(value).replace(/[&<>"']/g, char => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[char]));

export function ppvCurve(lines, prevalence, sample, labels, format) {
  const low = Math.log10(0.001), high = Math.log10(0.5);
  const x = value => 70 + 500 * (Math.log10(value) - low) / (high - low);
  const y = value => 240 - 205 * value;
  const grid = [0, .25, .5, .75, 1].map(value => `<line x1="70" x2="570" y1="${y(value)}" y2="${y(value)}"/><text x="60" y="${y(value) + 4}" text-anchor="end">${format.percent(value)}</text>`).join('');
  const ticks = [0.001, 0.003, 0.01, 0.03, 0.1, 0.3].map(value => `<text x="${x(value)}" y="262" text-anchor="middle">${format.share(value)}</text>`).join('');
  const steps = Array.from({length: 121}, (_, index) => 10 ** (low + (high - low) * index / 120));
  const paths = lines.map(line => {
    const d = steps.map((value, index) => `${index ? 'L' : 'M'}${x(value).toFixed(1)},${y(predictive(line.sensitivity, line.specificity, value).ppv).toFixed(1)}`).join('');
    const here = predictive(line.sensitivity, line.specificity, prevalence).ppv;
    return `<path class="ppv-line ${line.className}" d="${d}"/><circle class="ppv-point ${line.className}" cx="${x(prevalence)}" cy="${y(here)}" r="5"><title>${escape(line.label)} · ${escape(labels.ppv)} ${format.percent(here)}</title></circle><text class="ppv-label" x="${x(prevalence) + (line.below ? 10 : -10)}" y="${y(here) + (line.below ? 20 : -8)}" text-anchor="${line.below ? 'start' : 'end'}">${escape(line.short)} ${format.percent(here)}</text>`;
  }).join('');
  const hover = steps.filter((_, index) => index % 4 === 0).map(value => `<rect class="hover-band" x="${x(value) - 8}" y="30" width="16" height="210"><title>${escape(labels.assumedPrevalence)} ${format.share(value)} · ${lines.map(line => `${escape(line.short)}: ${escape(labels.ppv)} ${format.percent(predictive(line.sensitivity, line.specificity, value).ppv)}`).join(' · ')}</title></rect>`).join('');
  const sampleLine = sample > 0.001 && sample < 0.5 ? `<line class="sample-line" x1="${x(sample)}" x2="${x(sample)}" y1="30" y2="240"/><text x="${x(sample) - 6}" y="26" text-anchor="end">${escape(labels.samplePrevalence)} ${format.share(sample)}</text>` : '';
  const chosen = `<line class="current-line" x1="${x(prevalence)}" x2="${x(prevalence)}" y1="30" y2="240"/><text x="${x(prevalence) + 6}" y="44">${escape(labels.assumedPrevalence)} ${format.share(prevalence)}</text>`;
  return `<svg class="impact-plot ppv-plot" viewBox="0 0 640 290" role="img" aria-labelledby="ppv-title ppv-desc"><title id="ppv-title">${escape(labels.ppvCurve)}</title><desc id="ppv-desc">${lines.map(line => `${escape(line.label)}: ${escape(labels.ppv)} ${format.percent(predictive(line.sensitivity, line.specificity, prevalence).ppv)}`).join('. ')}</desc>${grid}${ticks}${sampleLine}${chosen}${paths}${hover}<text x="320" y="285" text-anchor="middle">${escape(labels.assumedPrevalence)} (log)</text></svg>`;
}

function segments(points, x, y) {
  let open = false;
  return points.map(([position, value]) => {
    if (value === null) { open = false; return ''; }
    const step = `${open ? 'L' : 'M'}${x(position).toFixed(1)},${y(value).toFixed(1)}`;
    open = true;
    return step;
  }).join('');
}

export function thresholdCurve(sweep, marks, selected, prevalence, labels, format) {
  const points = sweep.threshold.map((_, index) => matrixAt(sweep, index));
  const low = points[0].threshold, high = points.at(-1).threshold;
  const x = value => 70 + 470 * (value - low) / (high - low || 1);
  const y = value => 250 - 205 * value;
  const series = [
    {key:'sensitivity', className:'series-sens', value:point => point.sensitivity},
    {key:'specificity', className:'series-spec', value:point => point.specificity},
    {key:'youngPPV', className:'series-ppv', value:point => predictive(point.sensitivity, point.specificity, prevalence).ppv},
  ];
  const grid = [0, .25, .5, .75, 1].map(value => `<line x1="70" x2="540" y1="${y(value)}" y2="${y(value)}"/><text x="60" y="${y(value) + 4}" text-anchor="end">${format.percent(value)}</text>`).join('');
  const ticks = [0.2, 0.4, 0.6, 0.8].filter(value => value > low && value < high).map(value => `<text x="${x(value)}" y="272" text-anchor="middle">${value.toFixed(1)}</text>`).join('');
  const lines = series.map(item => {
    const values = points.map(point => [point.threshold, item.value(point)]);
    return `<path class="curve ${item.className}" d="${segments(values, x, y)}"/>`;
  }).join('');
  const verticals = marks.map((mark, order) => `<line class="mark-line ${mark.className}" x1="${x(points[mark.index].threshold)}" x2="${x(points[mark.index].threshold)}" y1="40" y2="250"/><text x="${x(points[mark.index].threshold)}" y="${16 + 13 * order}" text-anchor="middle">${escape(mark.label)}</text>`).join('');
  const current = points[selected];
  const rings = series.map(item => item.value(current) === null ? '' : `<circle class="current-point" cx="${x(current.threshold)}" cy="${y(item.value(current))}" r="6"/>`).join('');
  const hover = points.map((point, index) => {
    const next = points[index + 1] ?? point, previous = points[index - 1] ?? point;
    const left = x((previous.threshold + point.threshold) / 2), right = x((point.threshold + next.threshold) / 2);
    const text = series.map(item => `${labels[item.key]} ${format.percent(item.value(point))}`).join(' · ');
    return `<rect class="hover-band" data-index="${index}" x="${left}" y="40" width="${Math.max(right - left, 1)}" height="210"><title>${escape(labels.threshold)} ${format.cutoff(point.threshold)} · ${escape(text)}</title></rect>`;
  }).join('');
  return `<svg class="impact-plot threshold-curve" viewBox="0 0 680 300" role="img" aria-labelledby="threshold-curve-title threshold-curve-desc"><title id="threshold-curve-title">${escape(labels.curveTitle)}</title><desc id="threshold-curve-desc">${escape(labels.threshold)} ${format.cutoff(current.threshold)}: ${series.map(item => `${escape(labels[item.key])} ${format.percent(item.value(current))}`).join(', ')}.</desc>${grid}${ticks}${verticals}<line class="current-line" x1="${x(current.threshold)}" x2="${x(current.threshold)}" y1="40" y2="250"/>${lines}${rings}${hover}<text x="305" y="294" text-anchor="middle">${escape(labels.threshold)}</text></svg>`;
}
