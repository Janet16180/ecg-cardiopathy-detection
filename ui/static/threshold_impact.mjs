import {matrixAt} from './matrix.mjs';

export function thresholdImpact(pool, index) {
  const frozenIndex = pool.sweep.threshold.indexOf(pool.threshold);
  if (frozenIndex < 0) throw new RangeError('The calibrated cutoff must be a saved point.');
  const selected = matrixAt(pool.sweep, index);
  const points = pool.sweep.threshold.map((_, position) => matrixAt(pool.sweep, position));
  if (points.some(point => point.positives !== selected.positives || point.negatives !== selected.negatives)) {
    throw new Error('Saved cutoff matrices must describe the same reference population.');
  }
  return {points, selected, frozen:points[frozenIndex], index, frozenIndex};
}

const escape = value => String(value).replace(/[&<>"']/g, char => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[char]));
const coordinate = (value, low, high, start, span) => start + (high === low ? .5 : (value - low) / (high - low)) * span;

function markers(point, index, keys, x, y, kind) {
  return keys.filter(key => point[key] !== null).map(key => {
    const cx = x(point), cy = y(point[key]);
    const attributes = `data-index="${index}" data-series="${key}" data-threshold="${point.threshold}" data-value="${point[key]}"`;
    return kind === 'frozen'
      ? `<rect class="frozen-point" ${attributes} x="${cx - 4}" y="${cy - 4}" width="8" height="8"/>`
      : `<circle class="current-point" ${attributes} cx="${cx}" cy="${cy}" r="6"/>`;
  }).join('');
}

function countChart(data, keys, denominator, id, labels, format) {
  const low = data.points[0].threshold, high = data.points.at(-1).threshold;
  const x = point => coordinate(point.threshold, low, high, 65, 525);
  const y = value => 240 - 205 * value / (denominator || 1);
  const ticks = denominator ? [0, Math.round(denominator / 2), denominator] : [0];
  const grid = [...new Set(ticks)].map(value => `<line x1="65" x2="590" y1="${y(value)}" y2="${y(value)}"/><text x="55" y="${y(value) + 4}" text-anchor="end">${format.integer(value)}</text>`).join('');
  const points = keys.map(key => data.points.map((point, index) => `<circle class="saved-point count-${key}" data-index="${index}" data-series="${key}" cx="${x(point)}" cy="${y(point[key])}" r="2"><title>${escape(labels.threshold)} ${format.cutoff(point.threshold)} · ${key.toUpperCase()} ${format.integer(point[key])}</title></circle>`).join('')).join('');
  const lines = `<line class="frozen-line" x1="${x(data.frozen)}" x2="${x(data.frozen)}" y1="30" y2="240"/><line class="current-line" x1="${x(data.selected)}" x2="${x(data.selected)}" y1="30" y2="240"/>`;
  const summary = keys.map(key => `${key.toUpperCase()} ${format.integer(data.selected[key])}`).join(' · ');
  return `<section class="impact-chart"><h3>${escape(labels[id])}</h3><p>${escape(labels[id === 'positiveCounts' ? 'positive' : 'negative'])}: n = ${format.integer(denominator)}</p><svg class="impact-plot" viewBox="0 0 620 295" role="img" aria-labelledby="${id}-title ${id}-desc"><title id="${id}-title">${escape(labels[id])}</title><desc id="${id}-desc">${escape(labels.savedPoints)} ${escape(labels.currentCutoff)} ${format.cutoff(data.selected.threshold)}. ${summary}. ${escape(labels.frozen)} ${format.cutoff(data.frozen.threshold)}.</desc>${grid}${points}${lines}${markers(data.frozen, data.frozenIndex, keys, x, y, 'frozen')}${markers(data.selected, data.index, keys, x, y, 'current')}<text x="65" y="263">${format.cutoff(low)}</text><text x="590" y="263" text-anchor="end">${format.cutoff(high)}</text><text x="325" y="288" text-anchor="middle">${escape(labels.threshold)}</text></svg><p class="impact-values">${escape(labels.currentCutoff)}: ${summary}</p><div class="impact-series">${keys.map(key => `<span class="count-${key}">${key.toUpperCase()}</span>`).join('')}</div></section>`;
}

function workloadChart(data, labels, format) {
  const x = point => 65 + 525 * point.referral_fraction;
  const y = value => 240 - 205 * value;
  const valid = point => point.sensitivity !== null && point.referral_fraction !== null;
  const grid = [0,.5,1].map(value => `<line x1="65" x2="590" y1="${y(value)}" y2="${y(value)}"/><text x="55" y="${y(value) + 4}" text-anchor="end">${format.percent(value)}</text><text x="${65 + 525 * value}" y="263" text-anchor="middle">${format.percent(value)}</text>`).join('');
  const points = data.points.map((point, index) => valid(point) ? `<circle class="saved-point series-0" data-index="${index}" cx="${x(point)}" cy="${y(point.sensitivity)}" r="2.5"><title>${escape(labels.threshold)} ${format.cutoff(point.threshold)} · ${escape(labels.sensitivity)} ${format.percent(point.sensitivity)} · ${escape(labels.referral)} ${format.percent(point.referral_fraction)}</title></circle>` : '').join('');
  const frozen = valid(data.frozen) ? markers(data.frozen, data.frozenIndex, ['sensitivity'], x, y, 'frozen') : '';
  const current = valid(data.selected) ? markers(data.selected, data.index, ['sensitivity'], x, y, 'current') : '';
  return `<section class="impact-chart workload-chart"><h3>${escape(labels.workloadSensitivity)}</h3><p>${escape(labels.sensitivity)}: TP / ${format.integer(data.selected.positives)} · ${escape(labels.referral)}: (TP + FP) / ${format.integer(data.selected.ecgs)}</p><svg class="impact-plot" viewBox="0 0 620 295" role="img" aria-labelledby="workload-title workload-desc"><title id="workload-title">${escape(labels.workloadSensitivity)}</title><desc id="workload-desc">${escape(labels.savedPoints)} ${escape(labels.currentCutoff)} ${format.cutoff(data.selected.threshold)}. ${escape(labels.sensitivity)} ${format.percent(data.selected.sensitivity)}. ${escape(labels.referral)} ${format.percent(data.selected.referral_fraction)}.</desc>${grid}${points}${frozen}${current}<text x="65" y="18">${escape(labels.sensitivity)}</text><text x="325" y="288" text-anchor="middle">${escape(labels.referral)}</text></svg><p class="impact-values">${escape(labels.currentCutoff)}: ${escape(labels.sensitivity)} ${format.percent(data.selected.sensitivity)} · ${escape(labels.referral)} ${format.percent(data.selected.referral_fraction)}</p></section>`;
}

export function impactPlots(pool, index, labels, format) {
  const data = thresholdImpact(pool, index);
  return `<h3>${escape(labels.thresholdImpact)}</h3><p class="muted">${escape(labels.savedPoints)}</p><div class="impact-legend"><span class="current-key">${escape(labels.currentCutoff)}: ${format.cutoff(data.selected.threshold)}</span><span class="frozen-key">${escape(labels.frozen)}: ${format.cutoff(data.frozen.threshold)}</span></div><div class="impact-grid">${countChart(data, ['tp','fn'], data.selected.positives, 'positiveCounts', labels, format)}${countChart(data, ['fp','tn'], data.selected.negatives, 'negativeCounts', labels, format)}${workloadChart(data, labels, format)}</div>`;
}
