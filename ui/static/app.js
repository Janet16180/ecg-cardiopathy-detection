import {copy} from './i18n.mjs';
import {fraction, matrixAt, nearestIndex, parseCanonicalCSV} from './matrix.mjs';
import {ageComparisonRows} from './comparison.mjs';
import {impactPlots} from './threshold_impact.mjs';
import {modelResults} from './model_results.mjs';
import {adjustedIndex, atPrevalence, ppvCurve, thresholdCurve} from './prevalence.mjs';
import {ecgPlot, flagged, regionTable, swapArtifact, topRegions} from './replacement.mjs';

const state = {lang:'en', evaluation:null, cohorts:null, status:null, model:0, pool:0, cutoff:0, sort:'sensitivity', direction:'desc', prevalence:0.01, people:1000, acceptable:20, youngCutoff:null, nbr:null, nbrExample:0, nbrModel:'all_age_blend', nbrView:'blend', nbrSelection:null, nbrCounterpart:true, nbrState:'loading'};
const content = document.querySelector('#content');
const t = key => copy[state.lang][key] ?? copy.en[key] ?? key;
const escape = value => String(value ?? '').replace(/[&<>"']/g, char => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[char]));
const integer = value => Number(value).toLocaleString(state.lang);
const percent = value => value === null || value === undefined ? t('undefined') : `${(100 * value).toFixed(1)}%`;
const decimal = value => value === null || value === undefined ? '—' : Number(value).toFixed(4);
const cutoff = value => Number(value).toPrecision(7);
const heading = (title, intro) => `<h1>${t(title)}</h1><p class="lead">${t(intro)}</p>`;
const notice = text => `<div class="notice">${t(text)}</div>`;
const panel = html => `<section class="panel">${html}</section>`;

async function request(path, options) {
  const response = await fetch(path, options);
  const value = await response.json();
  if (!response.ok) throw new Error(value.detail ?? 'Local request did not complete.');
  return value;
}

function overview() {
  const cards = ['quality', 'scores', 'budget'].map(key => `<section class="card"><h3>${t(key)}</h3><p class="muted">${t(`${key}Text`)}</p></section>`).join('');
  const statuses = ['cohorts', 'evaluation'].map(key => `<div class="status"><strong>${t(key)}</strong><span>${t(state.status?.[key]?.state === 'completed' ? 'completed' : 'waiting')}</span></div>`).join('');
  const comparison = state.evaluation?.primary;
  const interval = comparison?.sensitivity_difference;
  const primary = comparison ? `<div class="notice"><strong>${t('primaryComparison')}</strong><p>${t(comparison.development_advantage ? 'advantage' : 'noAdvantage')} ${t('pairedDifference')}: ${(100 * interval.difference).toFixed(2)} ${t('percentagePoints')} · ${t('interval')}: [${(100 * interval.ci_low).toFixed(2)}, ${(100 * interval.ci_high).toFixed(2)}].</p></div>` : '';
  const results = state.evaluation ? summaryHTML() + primary + modelResultsHTML(true) : '';
  return heading('title', 'intro') + notice('scope') + results + `<div class="cards">${cards}</div>` + panel(`<h2>${t('execution')}</h2>${statuses}`);
}

function summaryHTML() {
  const model = state.evaluation.models.find(value => value.id === state.evaluation.default_model_id);
  const pool = model.evaluations.find(value => value.pool === 'sph') ?? model.evaluations[0];
  const matrix = matrixAt(pool.sweep, pool.sweep.threshold.indexOf(pool.threshold));
  const young = atPrevalence(matrix, 0.01, 1000);
  const cards = `<div class="cards four"><div class="card"><small>AUROC</small><div class="value">${decimal(pool.metrics.auroc)}</div></div><div class="card"><small>${t('sensitivity')}</small><div class="value">${percent(matrix.sensitivity)}</div></div><div class="card"><small>${t('specificity')}</small><div class="value">${percent(matrix.specificity)}</div></div><div class="card"><small>${t('ppvYoung')}, 1%</small><div class="value">${percent(young.ppv)}</div></div></div>`;
  return panel(`<h2>${t('summaryTitle')}</h2><p>${t('summaryText')}</p><p class="muted">${escape(model.label)} · ${escape(pool.label)} · score &gt; ${cutoff(pool.threshold)}</p>${cards}<div class="grid-two live-grid"><section class="live-column"><h3>${t('measuredTitle')}</h3>${matrixHTML(matrix)}</section><section class="live-column estimate-column"><h3>${t('youngTitle')}</h3><p class="muted">${t('perPeople')} 1000 ${t('students')}, ${t('assumedPrevalence').toLowerCase()} 1%. ${t('estimateShort')}</p>${expectedMatrixHTML(young, 1000)}</section></div><p><a href="#prevalence">${t('summaryLink')}</a></p>`);
}

function matrixHTML(matrix) {
  const cells = (left, right, denominator, errorLeft) => `<td class="${errorLeft ? 'error' : ''}">${left}<strong>${integer(matrix[left.toLowerCase()])}</strong><span>/ ${integer(denominator)}</span></td><td class="${errorLeft ? '' : 'error'}">${right}<strong>${integer(matrix[right.toLowerCase()])}</strong><span>/ ${integer(denominator)}</span></td>`;
  return `<table class="matrix"><thead><tr><th>${t('reference')}</th><th>${t('flagged')}</th><th>${t('notFlagged')}</th></tr></thead><tbody><tr><th>${t('positive')}<br><small>n = ${integer(matrix.positives)}</small></th>${cells('TP', 'FN', matrix.positives, false)}</tr><tr><th>${t('negative')}<br><small>n = ${integer(matrix.negatives)}</small></th>${cells('FP', 'TN', matrix.negatives, true)}</tr></tbody></table>`;
}

function expectedMatrixHTML(young, people) {
  const healthy = people - young.cases, tn = healthy - young.falseAlarms;
  const cell = (label, value, denominator, error) => `<td class="${error ? 'error' : ''}">${label}<strong>${estimate(value)}</strong><span>/ ${estimate(denominator)}</span></td>`;
  return `<table class="matrix expected"><thead><tr><th>${t('expectedReference')}</th><th>${t('flagged')}</th><th>${t('notFlagged')}</th></tr></thead><tbody><tr><th>${t('withCondition')}<br><small>n = ${estimate(young.cases)}</small></th>${cell('TP', young.found, young.cases, false)}${cell('FN', young.missed, young.cases, true)}</tr><tr><th>${t('withoutCondition')}<br><small>n = ${estimate(healthy)}</small></th>${cell('FP', young.falseAlarms, healthy, true)}${cell('TN', tn, healthy, false)}</tr></tbody></table>`;
}

function ratesHTML(matrix) {
  const denominators = {
    sensitivity:`${integer(matrix.tp)} / ${integer(matrix.positives)} ${t('positiveDenominator')}`,
    specificity:`${integer(matrix.tn)} / ${integer(matrix.negatives)} ${t('normalDenominator')}`,
    precision:`${integer(matrix.tp)} / ${integer(matrix.tp + matrix.fp)} ${t('flagged').toLowerCase()}`,
    referral:`${integer(matrix.tp + matrix.fp)} / ${integer(matrix.ecgs)} ${t('records')}`,
  };
  return `<div class="rates">${['sensitivity','specificity','precision','referral'].map(key => `<div class="metric"><span>${t(key)}</span><b>${percent(matrix[key === 'referral' ? 'referral_fraction' : key])}</b><small>${denominators[key]}</small></div>`).join('')}</div>`;
}


function selectedPool() {
  return state.evaluation.models[state.model].evaluations[state.pool];
}

function primaryHTML() {
  const value = state.evaluation.primary, interval = value.sensitivity_difference;
  const labels = Object.fromEntries(state.evaluation.models.map(model => [model.id,model.label]));
  return panel(`<h2>${t('primaryComparison')}</h2><p>${escape(labels[value.first])} vs ${escape(labels[value.second])} · ${escape(value.site.toUpperCase())}</p><p>${t('pairedDifference')}: <strong>${(100 * interval.difference).toFixed(2)} ${t('percentagePoints')}</strong></p><p>${t('interval')}: [${(100 * interval.ci_low).toFixed(2)}, ${(100 * interval.ci_high).toFixed(2)}] ${t('percentagePoints')} · ${integer(interval.draws)} ${t('draws')}. ${t('conditional')}</p><div class="notice">${t(value.development_advantage ? 'advantage' : 'noAdvantage')}</div>`);
}

function ageComparisonHTML(pool) {
  const rows = ageComparisonRows(state.evaluation, pool.pool);
  if (!rows.length) return '';
  const headings = [t('cohort'), t('threshold'), 'TP', 'FN', 'FP', 'TN', t('sensitivity'), t('specificity'), 'AUROC', t('referral')];
  const body = rows.map(row => {
    const m = row.metrics;
    const values = [cutoff(row.threshold), ...['tp','fn','fp','tn'].map(key => integer(m[key])), percent(m.sensitivity), percent(m.specificity), decimal(m.auroc), percent(m.referral_fraction)];
    return `<tr data-model-id="${row.id}" data-pool="${escape(row.pool)}"><th scope="row">${t(row.labelKey)}</th>${values.map(value => `<td>${value}</td>`).join('')}</tr>`;
  }).join('');
  return panel(`<h2>${t('ageComparison')}</h2><p>${t('ageComparisonText')}</p><p class="muted">${t('ageComparisonFixed')}</p><p>${escape(pool.label)} · ${integer(pool.metrics.ecgs)} ${t('records')} · ${t('positive')}: ${integer(pool.metrics.positives)} · ${t('negative')}: ${integer(pool.metrics.negatives)}</p><div class="table-wrap" tabindex="0" role="region" aria-label="${t('ageComparison')}"><table class="data age-comparison" id="age-comparison"><thead><tr>${headings.map(value => `<th scope="col">${value}</th>`).join('')}</tr></thead><tbody>${body}</tbody></table></div>`);
}

function modelResultsHTML(withPool = false) {
  const pool = selectedPool(), rows = modelResults(state.evaluation, pool.pool, state.sort, state.direction);
  const leaders = ['sensitivity','auroc'].map(key => {
    const best = rows.filter(row => row[`highest_${key}`]);
    const value = best.length ? (key === 'sensitivity' ? percent(best[0][key]) : decimal(best[0][key])) : t('undefined');
    return `<div class="card result-leader"><small>${t(key === 'sensitivity' ? 'highestSensitivity' : 'highestAUROC')}</small><div class="value">${value}</div><p>${best.map(row => escape(row.label)).join('<br>')}</p></div>`;
  }).join('');
  const site = withPool ? `<label>${t('pool')}<select id="results-pool">${state.evaluation.models[state.model].evaluations.map((value,index) => `<option value="${index}" ${index === state.pool ? 'selected' : ''}>${escape(value.label)}</option>`).join('')}</select></label>` : '';
  const sorting = `<div class="controls">${site}<label>${t('sortResults')}<select id="results-sort">${['sensitivity','auroc','fn','fp','referral_fraction'].map(key => `<option value="${key}" ${key === state.sort ? 'selected' : ''}>${t({auroc:'auc',fn:'falseNegatives',fp:'falsePositives',referral_fraction:'referral'}[key] ?? key)}</option>`).join('')}</select></label><label>${t('order')}<select id="results-order"><option value="desc" ${state.direction === 'desc' ? 'selected' : ''}>${t('descending')}</option><option value="asc" ${state.direction === 'asc' ? 'selected' : ''}>${t('ascending')}</option></select></label></div>`;
  const headers = [t('model'),t('threshold'),'TP','FN','FP','TN',t('sensitivity'),'AUROC',t('evaluationFPR'),t('referral')];
  const body = rows.map(row => {
    const badges = ['sensitivity','auroc'].filter(key => row[`highest_${key}`]).map(key => `<small class="result-badge">${t(key === 'sensitivity' ? 'highestSensitivity' : 'highestAUROC')}</small>`).join('');
    const values = [cutoff(row.threshold),...['tp','fn','fp','tn'].map(key => integer(row[key])),percent(row.sensitivity),decimal(row.auroc),percent(row.false_positive_rate),percent(row.referral_fraction)];
    return `<tr data-model-id="${escape(row.id)}" data-pool="${escape(pool.pool)}"><th scope="row">${escape(row.label)}${badges}</th>${values.map(value => `<td>${value}</td>`).join('')}</tr>`;
  }).join('');
  return panel(`<h2 id="all-results-title">${t('allResults')} · ${integer(rows.length)}</h2><p>${t('resultsDescriptive')}</p><div class="leader-grid">${leaders}</div>${sorting}<p class="muted">${escape(pool.label)} · n = ${integer(rows[0].ecgs)} · ${t('positive')}: ${integer(rows[0].positives)} · ${t('negative')}: ${integer(rows[0].negatives)}</p><p class="muted">${t('budgetCompact')} ${t('metricsCompact')}</p><div class="table-wrap" tabindex="0" role="region" aria-labelledby="all-results-title"><table class="data all-results" id="all-results"><thead><tr>${headers.map(value => `<th scope="col">${value}</th>`).join('')}</tr></thead><tbody>${body}</tbody></table></div>`);
}

function evaluation() {
  const header = heading('evaluation', 'evalIntro') + notice('scope') + `<p class="muted">${t('offlineEvaluation')}</p>`;
  if (!state.evaluation) return header + panel(`<h2>${t('noEvaluation')}</h2><p>${t('noEvaluationText')}</p>`);
  const model = state.evaluation.models[state.model], pool = selectedPool(), matrix = pool.metrics;
  const controls = `<div class="controls"><label>${t('model')}<select id="model">${state.evaluation.models.map((value, index) => `<option value="${index}" ${index === state.model ? 'selected' : ''}>${escape(value.label)}</option>`).join('')}</select></label><label>${t('pool')}<select id="pool">${model.evaluations.map((value, index) => `<option value="${index}" ${index === state.pool ? 'selected' : ''}>${escape(value.label)}</option>`).join('')}</select></label></div>`;
  const descriptors = `<div class="cards"><div class="card"><small>AUROC · ${escape(pool.label)}</small><div class="value">${decimal(matrix.auroc)}</div><small>Average precision: ${decimal(matrix.average_precision)}</small></div><div class="card"><small>${t('records')}</small><div class="value">${integer(matrix.ecgs)}</div><small>${t('positive')}: ${integer(matrix.positives)} · ${t('negative')}: ${integer(matrix.negatives)}</small></div><div class="card"><small>${t('prevalence')}</small><div class="value">${percent(fraction(matrix.positives, matrix.ecgs))}</div><small>${integer(matrix.positives)} / ${integer(matrix.ecgs)} ${t('records')}</small></div></div>`;
  const fixed = panel(`<h2>${t('frozen')}</h2><p class="muted">${t('frozenText')}</p><p class="threshold">score &gt; ${cutoff(pool.threshold)}</p><p class="muted">${t('calibration')}: ${integer(pool.calibration.flagged_normals)} / ${integer(pool.calibration.normals)} · ${t('allowed')}: ${integer(pool.calibration.allowed_normals)} · ${t('tied')}: ${integer(pool.calibration.tied_normals)}</p><div class="grid-two">${matrixHTML(matrix)}${ratesHTML(matrix)}</div>`);
  const explore = panel(`<h2>${t('explore')}</h2><p class="muted">${t('exploreText')}</p><div class="slider-label"><label for="cutoff">${t('threshold')}</label><output id="cutoff-value" class="threshold"></output></div><input id="cutoff" class="slider" type="range" min="0" max="${pool.sweep.threshold.length - 1}" step="1" value="${state.cutoff}" aria-describedby="cutoff-help"><div class="slider-label"><small id="cutoff-help">${t('comparison')}</small><button class="secondary" id="reset">${t('reset')}</button></div><div class="grid-two"><div id="exploratory-matrix"></div><div id="exploratory-rates"></div></div><div id="threshold-impact"></div><p id="threshold-summary" class="sr-only" aria-live="polite" aria-atomic="true"></p>`);
  return header + primaryHTML() + controls + modelResultsHTML() + ageComparisonHTML(pool) + descriptors + fixed + explore + `<details><summary>${t('provenance')}</summary><p><small>${t('artifactHash')}</small><br><code>${escape(state.evaluation.artifact_sha256)}</code></p></details>`;
}

const share = value => `${Number((100 * value).toFixed(1))}%`;
const estimate = value => value < 10 ? String(Number(value.toFixed(1))) : integer(Math.round(value));
const range = values => values.some(value => value === null) ? '' : `<br><small>${percent(values[0])}–${percent(values[1])}</small>`;
const sources = [
  {prevalence:0.003, key:'sourceSerious', cite:'Sarto P, et al. Eur Heart J 2023;44:1084-92', doi:'10.1093/eurheartj/ehad017'},
  {prevalence:0.0038, key:'sourceSoccer', cite:'Malhotra A, et al. N Engl J Med 2018;379:524-34', doi:'10.1056/NEJMoa1714719'},
  {prevalence:0.009, key:'sourceCongenital', cite:'van der Linde D, et al. J Am Coll Cardiol 2011;58:2241-7', doi:'10.1016/j.jacc.2011.08.025'},
  {prevalence:0.02, key:'sourceDisqualified', cite:'Corrado D, et al. JAMA 2006;296:1593-601', doi:'10.1001/jama.296.13.1593'},
  {prevalence:0.118, key:'sourceAbnormalECG', cite:'Pelliccia A, et al. Eur Heart J 2007;28:2006-10', doi:'10.1093/eurheartj/ehm219'},
];
const limits = [
  {key:'limitSheikh', cite:'Sheikh N, et al. Circulation 2014;129:1637-49', doi:'10.1161/CIRCULATIONAHA.113.006179'},
  {key:'limitRiding', cite:'Riding NR, et al. Heart 2015;101:384-90', doi:'10.1136/heartjnl-2014-306437'},
  {key:'limitMacLachlan', cite:'MacLachlan H, et al. J Am Coll Cardiol 2026;87:756-68', doi:'10.1016/j.jacc.2025.11.049'},
  {key:'limitMalhotra', cite:'Malhotra A, et al. N Engl J Med 2018;379:524-34', doi:'10.1056/NEJMoa1714719'},
];

function prevalence() {
  const header = heading('prevalenceTitle', 'prevalenceIntro') + `<div class="notice"><strong>${t('estimateOnly')}</strong><p>${t('estimateOnlyText')}</p></div>`;
  if (!state.evaluation) return header + panel(`<h2>${t('noEvaluation')}</h2><p>${t('noEvaluationText')}</p>`);
  const model = state.evaluation.models[state.model], pool = selectedPool();
  const presets = [{prevalence:0.003, key:'presetSerious'}, {prevalence:0.01, key:'presetUser'}, {prevalence:0.02, key:'presetBroad'}, {prevalence:0.118, key:'presetECG'}, {prevalence:pool.metrics.positives / pool.metrics.ecgs, key:'presetSample'}];
  const controls = `<div class="controls"><label>${t('model')}<select id="prev-model">${state.evaluation.models.map((value, index) => `<option value="${index}" ${index === state.model ? 'selected' : ''}>${escape(value.label)}</option>`).join('')}</select></label><label>${t('pool')}<select id="prev-pool">${model.evaluations.map((value, index) => `<option value="${index}" ${index === state.pool ? 'selected' : ''}>${escape(value.label)}</option>`).join('')}</select></label></div>`;
  const inputs = `<div class="preset-row" role="group" aria-label="${t('assumedPrevalence')}">${presets.map(value => `<button class="secondary preset" data-prevalence="${value.prevalence}">${share(value.prevalence)} · ${t(value.key)}</button>`).join('')}</div><div class="controls"><label>${t('assumedPrevalence')} (%)<input id="prev-value" type="number" min="0.1" max="50" step="0.1" value="${Number((100 * state.prevalence).toFixed(2))}"></label><label>${t('screened')}<input id="prev-people" type="number" min="100" max="100000" step="100" value="${state.people}"></label><label>${t('acceptable')}<span class="inline-range"><input id="prev-acceptable" type="range" min="1" max="100" step="1" value="${state.acceptable}"><output id="prev-acceptable-value">${state.acceptable}</output></span></label></div><p class="muted">${t('acceptableText')}</p>`;
  const evidence = panel(`<h2>${t('sourcesTitle')}</h2><p class="muted">${t('sourcesText')}</p><table class="data"><thead><tr><th scope="col">${t('assumedPrevalence')}</th><th scope="col">${t('definition')}</th><th scope="col">${t('source')}</th></tr></thead><tbody>${sources.map(value => `<tr><td>${share(value.prevalence)}</td><td>${t(value.key)}</td><td><a href="https://doi.org/${value.doi}" target="_blank" rel="noopener">${escape(value.cite)}</a></td></tr>`).join('')}</tbody></table>`);
  const hard = panel(`<h2>${t('limitsTitle')}</h2><p class="muted">${t('limitsText')}</p><table class="data"><thead><tr><th scope="col">${t('finding')}</th><th scope="col">${t('source')}</th></tr></thead><tbody>${limits.map(value => `<tr><td>${t(value.key)}</td><td><a href="https://doi.org/${value.doi}" target="_blank" rel="noopener">${escape(value.cite)}</a></td></tr>`).join('')}</tbody></table>`);
  const age = panel(`<h2>${t('ageTitle')}</h2><p>${t('ageText')}</p>`);
  if (state.youngCutoff === null) state.youngCutoff = pool.sweep.threshold.indexOf(pool.threshold);
  const live = panel(`<h2>${t('liveTitle')}</h2><p class="muted">${t('liveText')}</p><div class="slider-label"><label for="young-cutoff">${t('threshold')}</label><output id="young-cutoff-value" class="threshold"></output></div><input id="young-cutoff" class="slider" type="range" min="0" max="${pool.sweep.threshold.length - 1}" step="1" value="${state.youngCutoff}"><div class="controls"><button class="secondary" id="young-frozen">${t('goFrozen')}</button><button class="secondary" id="young-adjusted">${t('goAdjusted')}</button></div><div id="young-live"></div>`);
  return header + controls + panel(`<h2>${t('assumptions')}</h2>${inputs}`) + live + `<div id="prevalence-results"></div>` + hard + evidence + age;
}

function updatePrevalence() {
  const pool = selectedPool(), sweep = pool.sweep;
  const frozenIndex = sweep.threshold.indexOf(pool.threshold);
  const adjusted = adjustedIndex(sweep, state.prevalence, state.acceptable);
  const rows = [
    {key:'frozenRow', className:'frozen-series', index:frozenIndex},
    {key:'adjustedRow', className:'adjusted-series', index:adjusted.index},
  ].map(row => ({...row, value:atPrevalence(matrixAt(sweep, row.index), state.prevalence, state.people)}));
  const sample = pool.metrics.positives / pool.metrics.ecgs;
  const headers = [t('cutoffRule'), t('threshold'), t('sensitivity'), t('specificity'), t('referredTile'), t('foundTile'), t('missedTile'), t('falseTile'), 'VPP / PPV', 'VPN / NPV'];
  const body = rows.map(row => {
    const v = row.value;
    const cells = [cutoff(v.threshold), percent(v.sensitivity), percent(v.specificity), estimate(v.referred), estimate(v.found), estimate(v.missed), estimate(v.falseAlarms), `${percent(v.ppv)}${range(v.ppvRange)}`, `${percent(v.npv)}${range(v.npvRange)}`];
    return `<tr><th scope="row"><span class="series-key ${row.className}"></span>${t(row.key)}</th>${cells.map(value => `<td>${value}</td>`).join('')}</tr>`;
  }).join('');
  const noBenefit = adjusted.benefit <= 0 ? `<div class="notice">${t('noBenefit')}</div>` : '';
  const lines = rows.map(row => ({label:t(row.key), short:t(`${row.key}Short`), className:row.className, below:row.key === 'frozenRow', sensitivity:row.value.sensitivity, specificity:row.value.specificity}));
  const labels = Object.fromEntries(['ppv','assumedPrevalence','samplePrevalence','ppvCurve'].map(key => [key, t(key)]));
  document.querySelector('#prevalence-results').innerHTML =
    panel(`<h2>${t('adjustedTitle')}</h2><p class="muted">${t('adjustedText')}</p>${noBenefit}<div class="table-wrap" tabindex="0" role="region" aria-label="${t('adjustedTitle')}"><table class="data" id="prevalence-table"><thead><tr>${headers.map(value => `<th scope="col">${value}</th>`).join('')}</tr></thead><tbody>${body}</tbody></table></div><p class="muted">${t('intervalNote')}</p><button class="secondary" id="prev-explore">${t('openExplorer')}</button>`) +
    panel(`<h2>${t('ppvCurve')}</h2><p class="muted">${t('ppvCurveText')}</p><div class="impact-legend">${lines.map(line => `<span class="series-key-label"><span class="series-key ${line.className}"></span>${escape(line.label)}</span>`).join('')}</div>${ppvCurve(lines, state.prevalence, sample, labels, {percent, share})}`);
  document.querySelector('#prev-explore').addEventListener('click', () => { state.cutoff = adjusted.index; location.hash = '#evaluation'; });
  updateYoung();
}

function updateYoung() {
  const pool = selectedPool(), sweep = pool.sweep, matrix = matrixAt(sweep, state.youngCutoff);
  const frozenIndex = sweep.threshold.indexOf(pool.threshold);
  const adjusted = adjustedIndex(sweep, state.prevalence, state.acceptable);
  const young = atPrevalence(matrix, state.prevalence, state.people);
  document.querySelector('#young-cutoff-value').textContent = cutoff(matrix.threshold);
  document.querySelector('#young-cutoff').setAttribute('aria-valuetext', `${t('threshold')} ${cutoff(matrix.threshold)}`);
  const tiles = `<div class="metric"><span>${t('referredTile')}</span><b>${t('about')} ${estimate(young.referred)}</b><small>TP + FP, ${percent(young.referred / state.people)}</small></div><div class="metric"><span>${t('sensitivity')} / ${t('specificity')}</span><b>${percent(young.sensitivity)} / ${percent(young.specificity)}</b><small>${t('fromValidation')}</small></div>`;
  const predictiveValues = `<div class="metric"><span>${t('ppv')}</span><b>${percent(young.ppv)}</b><small>${t('range95')} ${percent(young.ppvRange[0])} ${t('to')} ${percent(young.ppvRange[1])}</small></div><div class="metric"><span>${t('npv')}</span><b>${percent(young.npv)}</b><small>${t('range95')} ${percent(young.npvRange[0])} ${t('to')} ${percent(young.npvRange[1])}</small></div>`;
  const sentence = young.ppv ? t('ppvSentence').replace('{ppv}', percent(young.ppv)).replace('{one}', estimate(1 / young.ppv)) : t('noReferrals');
  const measured = `<section class="live-column"><h3>${t('measuredTitle')}</h3><p class="muted">${escape(pool.label)} · n = ${integer(matrix.ecgs)}</p>${matrixHTML(matrix)}${ratesHTML(matrix)}</section>`;
  const estimated = `<section class="live-column estimate-column"><h3>${t('youngTitle')}</h3><p class="muted">${t('perPeople')} ${integer(state.people)} ${t('students')}, ${t('assumedPrevalence').toLowerCase()} ${share(state.prevalence)}: ${t('about')} ${estimate(young.cases)} ${t('trueCases')}. ${t('estimateShort')}</p>${expectedMatrixHTML(young, state.people)}<div class="rates">${tiles}${predictiveValues}</div><p>${sentence}</p></section>`;
  const labels = {sensitivity:t('sensitivityMeasured'), specificity:t('specificityMeasured'), youngPPV:t('ppvYoung'), threshold:t('threshold'), curveTitle:t('liveTitle')};
  const marks = [{index:frozenIndex, label:t('frozenRowShort'), className:'frozen-mark'}, {index:adjusted.index, label:t('adjustedRowShort'), className:'adjusted-mark'}];
  const legend = `<div class="impact-legend">${[['series-sens','sensitivityMeasured'],['series-spec','specificityMeasured'],['series-ppv','ppvYoung']].map(([className, key]) => `<span class="series-key-label"><span class="series-key ${className}"></span>${t(key)}</span>`).join('')}</div>`;
  document.querySelector('#young-live').innerHTML = `<div class="grid-two live-grid">${measured}${estimated}</div><h3>${t('liveCurve')}</h3><p class="muted">${t('liveCurveText')}</p>${legend}${thresholdCurve(sweep, marks, state.youngCutoff, state.prevalence, labels, {percent, cutoff})}`;
}

function setupPrevalence() {
  if (!state.evaluation) return;
  document.querySelector('#prev-model').addEventListener('change', event => {
    state.model = Number(event.target.value); state.pool = 0; state.youngCutoff = null;
    state.cutoff = nearestIndex(selectedPool().sweep.threshold, selectedPool().threshold); render();
  });
  document.querySelector('#prev-pool').addEventListener('change', event => {
    state.pool = Number(event.target.value); state.youngCutoff = null;
    state.cutoff = nearestIndex(selectedPool().sweep.threshold, selectedPool().threshold); render();
  });
  document.querySelector('#young-cutoff').addEventListener('input', event => { state.youngCutoff = Number(event.target.value); updateYoung(); });
  const jump = index => { state.youngCutoff = index; document.querySelector('#young-cutoff').value = index; updateYoung(); };
  document.querySelector('#young-frozen').addEventListener('click', () => jump(selectedPool().sweep.threshold.indexOf(selectedPool().threshold)));
  document.querySelector('#young-adjusted').addEventListener('click', () => jump(adjustedIndex(selectedPool().sweep, state.prevalence, state.acceptable).index));
  document.querySelectorAll('.preset').forEach(button => button.addEventListener('click', () => {
    state.prevalence = Number(button.dataset.prevalence);
    document.querySelector('#prev-value').value = Number((100 * state.prevalence).toFixed(2)); updatePrevalence();
  }));
  document.querySelector('#prev-value').addEventListener('input', event => {
    const value = Number(event.target.value) / 100;
    if (value >= 0.001 && value <= 0.5) { state.prevalence = value; updatePrevalence(); }
  });
  document.querySelector('#prev-people').addEventListener('input', event => {
    const value = Math.round(Number(event.target.value));
    if (value >= 100 && value <= 100000) { state.people = value; updatePrevalence(); }
  });
  document.querySelector('#prev-acceptable').addEventListener('input', event => {
    state.acceptable = Number(event.target.value);
    document.querySelector('#prev-acceptable-value').textContent = state.acceptable; updatePrevalence();
  });
  updatePrevalence();
}

function recording() {
  const models = state.evaluation?.models ?? [];
  return heading('recording', 'recordingIntro') + notice('scoresText') + panel(`<h2>${t('format')}</h2><p>${t('formatText')}</p><p><code>I,II,III,aVR,aVL,aVF,V1,V2,V3,V4,V5,V6</code></p><p class="muted">${t('localSignal')}</p><label for="ecg-file">${t('chooseCSV')}</label><input id="ecg-file" type="file" accept=".csv,text/csv"><div class="controls"><label>${t('model')}<select id="recording-model">${models.length ? models.map(value => `<option value="${escape(value.id)}" ${value.id === state.evaluation.default_model_id ? 'selected' : ''}>${escape(value.label)}</option>`).join('') : `<option value="unavailable">${t('noModel')}</option>`}</select></label><button class="primary" id="analyze">${t('analyze')}</button></div><div id="recording-result" aria-live="polite"></div>`);
}

function cohortTable(value) {
  const rows = [
    ['records',value.records],['targetPositive',value.target['1'] ?? 0],['targetNegative',value.target['0'] ?? 0],
    ['patients',value.known_patients],['exactAge',value.age.exact_count],['youngAge',value.age.exact_18_25_count],
    ['missingAge',value.age.missing_count],['censoredAge',value.age.right_censored_over_89_count],
    ['minors',value.age.exact_under_18_count],
  ];
  return `<table class="data"><tbody>${rows.map(([key,count]) => `<tr><th>${t(key)}</th><td>${integer(count)}</td></tr>`).join('')}</tbody></table><p class="muted">${t('exactRange')}: ${decimal(value.age.exact_min)}–${decimal(value.age.exact_max)}<br>${t('exactMean')}: ${decimal(value.age.exact_mean)} · ${t('exactMedian')}: ${decimal(value.age.exact_median)}</p><h3>${t('evidenceRank')}</h3><table class="data"><tbody>${Object.entries(value.label_evidence_rank).map(([rank,count]) => `<tr><th>${escape(t(`rank${rank}`))}</th><td>${integer(count)}</td></tr>`).join('')}</tbody></table><h3>${t('source')}</h3><table class="data"><tbody>${Object.entries(value.sources).map(([source,count]) => `<tr><th>${escape(source)}</th><td>${integer(count)}</td></tr>`).join('')}</tbody></table>`;
}

function cohorts() {
  const header = heading('cohorts', 'cohortIntro');
  if (!state.cohorts) return header + panel(`<p>${t('noCohorts')}</p>`);
  const primary = state.cohorts.primary, secondary = state.cohorts.secondary;
  const comparison = (values, names, labels) => `<div class="grid-two">${names.map((key,index) => panel(`<h2>${t(labels[index])}</h2>${cohortTable(values.cohorts[key])}`)).join('')}</div><p class="muted">${t('overlap')} ${t('records')}: ${integer(values.overlap_between_training_cohorts.records)} · canonical windows: ${integer(values.overlap_between_training_cohorts.canonical_windows)} · ${t('patients')}: ${integer(values.overlap_between_training_cohorts.known_patients)}</p>`;
  return header + `<h2>${t('primary')}</h2>` + notice('matched') + `<p>${t('useSeed')}: ${integer(primary.seed)}</p>` + comparison(primary,['strict_18_25','general_matched_size'],['strictYoung','matchedControl']) + notice('heldoutScope') + (secondary ? `<h2>${t('secondary')}</h2>` + comparison(secondary,['younger','all_age'],['younger','allAge']) : '');

}

const groups = {inferior:'groupInferior', anterior:'groupAnterior', lateral:'groupLateral', rbbb:'groupRbbb', lbbb:'groupLbbb', stt:'groupStt', normal:'groupNormal'};
const nbrModels = {all_age_blend:'modelAllAge', ptb_baseline_blend:'modelPtb'};
const nbrViews = [['blend','viewBlend'],['hgb','viewHgb'],['xecg','viewXecg'],['energy','viewEnergy'],['swap','viewSwap']];
const nbrFormat = {score: value => Number(value).toFixed(3), value: (value, view) => view === 'energy' ? Number(value).toFixed(1) : (Math.abs(value) < 0.0005 ? 0 : Number(value)).toFixed(3).replace('-', '−')};

function localization() {
  const header = heading('nbrTitle', 'nbrIntro') + `<span class="badge">${t('nbrBadge')}</span>`;
  const how = panel(`<h2>${t('nbrHowTitle')}</h2><p>${t('nbrHowText')}</p>`);
  const limits = panel(`<h2>${t('nbrLimitsTitle')}</h2><ul>${[1,2,3,4,5,6,7].map(index => `<li>${t(`nbrLimit${index}`)}</li>`).join('')}</ul>`);
  const data = state.nbr;
  if (!data) return header + panel(state.nbrState === 'loading' ? `<h2>${t('nbrLoading')}</h2><p>${t('nbrLoadingText')}</p>` : `<h2>${t('nbrUnavailable')}</h2><p>${t('nbrUnavailableText')}</p>`) + how + limits;
  const options = data.examples.map((example, index) => `<option value="${index}" ${index === state.nbrExample ? 'selected' : ''}>${escape(t(groups[example.group] ?? example.group))} · ${escape(example.id)} · ${escape(example.age ?? '')} ${t('years')} · ${escape(example.sex ?? '')}</option>`).join('');
  const models = Object.keys(data.models).map(id => `<option value="${escape(id)}" ${id === state.nbrModel ? 'selected' : ''}>${escape(nbrModels[id] ? t(nbrModels[id]) : id)}</option>`).join('');
  const views = nbrViews.map(([id, label]) => `<button class="preset" data-view="${id}" aria-pressed="${state.nbrView === id}">${t(label)}</button>`).join('');
  const controls = `<div class="controls"><label>${t('nbrExample')}<select id="nbr-example">${options}</select></label><label>${t('nbrModel')}<select id="nbr-model">${models}</select></label></div><p class="muted">${t('nbrView')}</p><div class="preset-row" id="nbr-views">${views}</div>`;
  return header + (data.mock ? notice('nbrMock') : '') + panel(controls + `<div id="nbr-live" aria-live="polite"></div>`) + `<div id="nbr-study">${nbrStudyHTML(data.summary)}</div>` + how + limits;
}

function nbrStudyHTML(summary) {
  if (!summary) return '';
  const signed = value => `${value < 0 ? '−' : '+'}${Math.abs(value).toFixed(3)}`;
  const share = value => `${(100 * value).toFixed(1)}%`;
  const points = value => `${value < 0 ? '−' : '+'}${Math.abs(100 * value).toFixed(1)} pp`;
  const range = (value, format) => value ? `${format(value.point)} <span class="muted">[${format(value.low)}, ${format(value.high)}]</span>` : '—';
  const fill = (key, values) => Object.entries(values).reduce((text, [name, value]) => text.replaceAll(`{${name}}`, value), t(key));
  const stat = key => summary.statistics[key], primary = summary.primary_statistic.split('/')[0];
  const model = summary.statistics[`${state.nbrModel}/territory`] ? state.nbrModel : primary;
  const passed = summary.decisions.primary === 'territory localization shown' && summary.decisions.qualifier === 'beyond replaced energy';
  const verdict = fill(passed ? 'nbrVerdictPass' : 'nbrVerdictFail', {territory:signed(stat(`${primary}/territory`).point), energy:signed(stat('energy/territory').point), gap:signed(stat(`${primary}/territory_minus_energy`).point), inferior:integer(summary.n.inferior.records), anterior:integer(summary.n.anterior.records)});
  const rows = [
    ['nbrRowTerritory', 'territory', signed], ['nbrRowWave', 'wave', signed], ['nbrRowTop1', 'top1_hit', share], ['nbrRowTop3', 'top3_hit', share],
  ].map(([label, key, format]) => `<tr><th scope="row">${t(label)}</th><td>${range(stat(`${model}/${key}`), format)}</td><td>${range(stat(`energy/${key}`), format)}</td><td>${range(stat(`${model}/${key}_minus_energy`), format === share ? points : format)}</td></tr>`).join('');
  const swap = `<tr><th scope="row">${t('nbrRowSwap')}</th><td>${range(stat(`${model}/swap_territory`), signed)}</td><td>—</td><td>—</td></tr>`;
  const table = `<div class="table-wrap"><table class="data"><thead><tr><th scope="col">${t('nbrMeasure')}</th><th scope="col">${escape(t(nbrModels[model] ?? model))}</th><th scope="col">${t('nbrEnergyMap')}</th><th scope="col">${t('nbrGap')}</th></tr></thead><tbody>${rows}${swap}</tbody></table></div>`;
  const abnormal = Object.entries(summary.n).filter(([group]) => group !== 'normal').map(([, counts]) => counts);
  const flagged = abnormal.reduce((total, counts) => total + counts[`${model.replace('_blend', '')}_flagged`], 0);
  const records = abnormal.reduce((total, counts) => total + counts.records, 0);
  const flips = [['nbrFlipGrid', 'grid_flip'], ['nbrFlipLead', 'lead_flip'], ['nbrFlipComponent', 'component_flip'], ['nbrFlipFull', 'full_flip']].map(([label, key]) => `<tr><th scope="row">${t(label)}</th><td>${range(stat(`${model}/abnormal/${key}`), share)}</td></tr>`).join('');
  const flipTable = `<h3>${t('nbrFlipTitle')}</h3><p class="muted">${fill('nbrFlipText', {flagged:integer(flagged), records:integer(records)})}</p><table class="data"><tbody>${flips}</tbody></table>`;
  const read = `<h3>${t('nbrReadTitle')}</h3><ul>${[1, 2, 3].map(index => `<li>${t(`nbrRead${index}`)}</li>`).join('')}</ul>`;
  const total = Object.values(summary.n).reduce((sum, counts) => sum + counts.records, 0);
  return panel(`<h2>${t('nbrStudyTitle')}</h2><p>${verdict}</p>${table}${flipTable}${read}<p class="muted">${fill('nbrStudyNote', {records:integer(total)})}</p>`);
}

function nbrExample() {
  const data = state.nbr, example = data.examples[state.nbrExample];
  return Object.assign(example, {leads:data.leads, componentNames:data.components, fs:data.fs});
}

function regionName(example, selection) {
  if (selection.kind === 'full') return t('nbrFull');
  if (selection.kind === 'component') return `${example.componentNames[selection.component]} · ${t('nbrAllLeads')}`;
  if (selection.kind === 'lead') return `${example.leads[selection.lead]} · ${t('nbrWholeLead')}`;
  return `${example.componentNames[selection.component]} · ${example.leads[selection.lead]}`;
}

function updateReplacement() {
  const example = nbrExample(), scores = example.scores[state.nbrModel], threshold = state.nbr.models[state.nbrModel].threshold;
  const top = topRegions(scores, threshold);
  state.nbrSelection ??= {kind:'cell', lead:top[0].lead, component:top[0].component};
  const artifact = swapArtifact(scores);
  const statements = example.statements?.length ? `<p class="muted">${example.statements.map(escape).join(' · ')}</p>` : '';
  const status = `<p><strong>${t('nbrOriginal')} ${nbrFormat.score(scores.original)}</strong> (${t('nbrThreshold')} ${nbrFormat.score(threshold)}): ${t(flagged(scores.original, threshold) ? 'nbrReferred' : 'nbrNotReferred')}. ${t('nbrFullSentence')} ${nbrFormat.score(scores.full)}.</p>`;
  const saturated = Math.min(scores.original, 1 - scores.original) < 0.02 ? `<p class="muted saturated-note">${t('nbrSaturated')}</p>` : '';
  const list = `<h3>${t('nbrTopTitle')}</h3><ol class="top-regions">${top.map(region => `<li>${escape(example.componentNames[region.component])} · ${escape(example.leads[region.lead])}: ${t('nbrFalls')} ${nbrFormat.value(region.drop)} (${t('nbrAfter')} ${nbrFormat.score(region.after)}); ${t(region.flips ? 'nbrTopFlip' : 'nbrTopNoFlip')}.</li>`).join('')}</ol>`;
  const swap = `<p class="muted">${t('nbrSwapNote').replace('{mean}', nbrFormat.value(artifact.mean)).replace('{max}', nbrFormat.value(artifact.max))}</p>`;
  const caption = {energy:'nbrCaptionEnergy', swap:'nbrCaptionSwap'}[state.nbrView] ?? 'nbrCaption';
  const table = regionTable(example, scores, threshold, state.nbrView, state.nbrSelection, {lead:t('nbrLead'), wholeLead:t('nbrWholeLead'), allLeads:t('nbrAllLeads'), full:t('nbrFull'), after:t('nbrAfter'), flip:t('nbrFlip'), caption:t(caption)}, nbrFormat);
  const keys = `<p class="muted"><span class="series-key-label"><span class="series-key original-series"></span>${t('nbrKeyOriginal')}</span> · <label class="series-key-label counterpart-toggle"><input type="checkbox" id="nbr-counterpart" ${state.nbrCounterpart ? 'checked' : ''}><span class="series-key counterpart-series"></span>${t('nbrKeyCounterpart')}</label> · <span class="series-key-label"><span class="band-key"></span>${t('nbrKeyBand')}</span></p>`;
  const plot = ecgPlot(example, state.nbrSelection, {title:t('nbrPlotTitle'), replaced:t('nbrKeyBand'), scale:t('nbrScale')}, state.nbrCounterpart);
  document.querySelector('#nbr-live').innerHTML = `<div class="grid-two nbr-grid"><section>${statements}${status}${saturated}${list}${swap}</section><section class="region-scroll">${table}</section></div><h3>${t('nbrSelected')}: ${escape(regionName(example, state.nbrSelection))}</h3>${keys}<div class="ecg-scroll">${plot}</div>`;
}

function setupReplacement() {
  if (!state.nbr) return;
  document.querySelector('#nbr-example').addEventListener('change', event => { state.nbrExample = Number(event.target.value); state.nbrSelection = null; updateReplacement(); });
  document.querySelector('#nbr-model').addEventListener('change', event => { state.nbrModel = event.target.value; state.nbrSelection = null; updateReplacement(); document.querySelector('#nbr-study').innerHTML = nbrStudyHTML(state.nbr.summary); });
  document.querySelector('#nbr-views').addEventListener('click', event => {
    const button = event.target.closest('[data-view]');
    if (!button) return;
    state.nbrView = button.dataset.view;
    document.querySelectorAll('#nbr-views [data-view]').forEach(element => element.setAttribute('aria-pressed', element === button));
    updateReplacement();
  });
  document.querySelector('#nbr-live').addEventListener('click', event => {
    const button = event.target.closest('button[data-lead],button[data-component],button[data-full]');
    if (!button) return;
    const {lead, component, full} = button.dataset;
    if (full) state.nbrSelection = {kind:'full'};
    else if (lead !== undefined && component !== undefined) state.nbrSelection = {kind:'cell', lead:Number(lead), component:Number(component)};
    else if (lead !== undefined) state.nbrSelection = {kind:'lead', lead:Number(lead)};
    else state.nbrSelection = {kind:'component', component:Number(component)};
    updateReplacement();
  });
  document.querySelector('#nbr-live').addEventListener('change', event => {
    if (event.target.id !== 'nbr-counterpart') return;
    state.nbrCounterpart = event.target.checked;
    updateReplacement();
  });
  updateReplacement();
}

function methods() {
  return heading('methodsIntro', 'scope') + `<div class="methods">${['proxy','offlineEvaluation','evidence','xecg','legacy','wipMethod','privacy'].map(key => panel(`<p>${t(key)}</p>`)).join('')}</div>`;
}

function updateMatrix() {
  const pool = selectedPool(), matrix = matrixAt(pool.sweep, state.cutoff);
  document.querySelector('#cutoff-value').textContent = cutoff(matrix.threshold);
  document.querySelector('#cutoff').setAttribute('aria-valuetext', `${t('threshold')} ${cutoff(matrix.threshold)}`);
  document.querySelector('#exploratory-matrix').innerHTML = matrixHTML(matrix);
  document.querySelector('#exploratory-rates').innerHTML = ratesHTML(matrix);
  const labels = Object.fromEntries(['thresholdImpact','positiveCounts','negativeCounts','workloadSensitivity','savedPoints','currentCutoff','frozen','threshold','positive','negative','sensitivity','referral'].map(key => [key,t(key)]));
  document.querySelector('#threshold-impact').innerHTML = impactPlots(pool, state.cutoff, labels, {integer,percent,cutoff});
  document.querySelector('#threshold-summary').textContent = `${t('threshold')} ${cutoff(matrix.threshold)}: TP ${integer(matrix.tp)}, FN ${integer(matrix.fn)}, FP ${integer(matrix.fp)}, TN ${integer(matrix.tn)}.`;
}

function setupEvaluation() {
  if (!state.evaluation) return;
  document.querySelector('#model').addEventListener('change', event => {
    state.model = Number(event.target.value); state.pool = 0;
    state.cutoff = nearestIndex(selectedPool().sweep.threshold, selectedPool().threshold); render();
  });
  document.querySelector('#pool').addEventListener('change', event => {
    state.pool = Number(event.target.value);
    state.cutoff = nearestIndex(selectedPool().sweep.threshold, selectedPool().threshold); render();
  });
  document.querySelector('#cutoff').addEventListener('input', event => { state.cutoff = Number(event.target.value); updateMatrix(); });
  document.querySelector('#reset').addEventListener('click', () => {
    state.cutoff = nearestIndex(selectedPool().sweep.threshold, selectedPool().threshold);
    document.querySelector('#cutoff').value = state.cutoff; updateMatrix();
  });
  updateMatrix();
}

function setupModelResults() {
  if (!state.evaluation) return;
  document.querySelector('#results-pool')?.addEventListener('change', event => {
    state.pool = Number(event.target.value);
    state.cutoff = nearestIndex(selectedPool().sweep.threshold, selectedPool().threshold); render();
  });
  document.querySelector('#results-sort').addEventListener('change', event => {
    state.sort = event.target.value;
    state.direction = ['fn','fp','referral_fraction'].includes(state.sort) ? 'asc' : 'desc'; render();
  });
  document.querySelector('#results-order').addEventListener('change', event => {
    state.direction = event.target.value; render();
  });
}

function recordingResult(result) {
  const quality = result.quality;
  const warnings = quality.review_flags.length ? `<p>${t('flags')}: ${quality.review_flags.map(escape).join(', ')}</p>` : '';
  if (result.status === 'abstained') return `<div class="notice"><h3>${t('abstained')}</h3><p>${t('noScore')}</p><p>${t('exclusions')}: ${quality.reasons.map(escape).join(', ')}</p><p>${t('abstentionAction')}</p></div>`;
  if (result.status !== 'scored') return `<div class="notice"><h3>${t('passed')}</h3><p>${t('unavailable')}</p><p>${t('noScore')}</p>${warnings}</div>`;
  return `<div class="notice"><h3>${t(result.flagged ? 'flaggedResult' : 'notFlaggedResult')}</h3><small>${t('score')}</small><div class="result-score">${decimal(result.score)}</div><p>score &gt; ${cutoff(result.threshold)}</p><p>${t('notDiagnosis')}</p><p>${t('inferenceCalibration')}</p>${warnings}</div>`;
}

async function analyzeRecording() {
  const file = document.querySelector('#ecg-file').files[0];
  const output = document.querySelector('#recording-result'), button = document.querySelector('#analyze');
  if (!file) { output.textContent = t('chooseFirst'); return; }
  if (file.size > 4 * 1024 * 1024) { output.textContent = 'CSV exceeds the 4 MiB local request limit.'; return; }
  button.disabled = true; button.textContent = t('working');
  try {
    const payload = parseCanonicalCSV(await file.text());
    payload.model_id = document.querySelector('#recording-model').value;
    const result = await request('/api/analyze', {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify(payload)});
    output.innerHTML = recordingResult(result);
  } catch (error) {
    output.innerHTML = `<p class="error-text">${escape(error.message)}</p>`;
  } finally { button.disabled = false; button.textContent = t('analyze'); }
}

function render() {
  const page = location.hash.slice(1) || 'overview';
  const routes = {overview,evaluation,prevalence,recording,cohorts,localization,methods};
  const active = Object.hasOwn(routes, page) ? page : 'overview';
  document.documentElement.lang = state.lang;
  document.querySelectorAll('[data-i18n]').forEach(element => { element.textContent = t(element.dataset.i18n); });
  document.querySelectorAll('nav a').forEach(element => {
    if (element.dataset.page === active) element.setAttribute('aria-current', 'page');
    else element.removeAttribute('aria-current');
  });
  document.querySelector('#english').setAttribute('aria-pressed', state.lang === 'en');
  document.querySelector('#spanish').setAttribute('aria-pressed', state.lang === 'es');
  document.querySelector('#connection').textContent = t(state.status ? 'connected' : 'disconnected');
  document.querySelector('#artifact-mode').textContent = state.status ? t(state.status.artifact_mode === 'from_raw' ? 'modeFromRaw' : 'modeCached') : '';
  content.innerHTML = routes[active]();
  if (['overview','evaluation'].includes(active)) setupModelResults();
  if (active === 'evaluation') setupEvaluation();
  if (active === 'prevalence') setupPrevalence();
  if (active === 'localization') setupReplacement();
  if (active === 'recording') document.querySelector('#analyze').addEventListener('click', analyzeRecording);
}

window.addEventListener('hashchange', render);
document.querySelector('#english').addEventListener('click', () => { state.lang = 'en'; render(); });
document.querySelector('#spanish').addEventListener('click', () => { state.lang = 'es'; render(); });
document.querySelector('#theme').addEventListener('click', () => {
  document.documentElement.dataset.theme = document.documentElement.dataset.theme === 'dark' ? 'light' : 'dark';
});
render();
const [status, evaluationData, cohortData, replacementData] = await Promise.all([
  request('/api/status').catch(() => null), request('/api/evaluation').catch(() => null), request('/api/cohorts').catch(() => null),
  request('/api/localization').catch(() => null),
]);
state.status = status; state.evaluation = evaluationData; state.cohorts = cohortData; state.nbr = replacementData; state.nbrState = replacementData ? 'ready' : 'unavailable';
if (replacementData && !replacementData.models[state.nbrModel]) state.nbrModel = Object.keys(replacementData.models)[0];
if (evaluationData) {
  state.model = evaluationData.models.findIndex(model => model.id === evaluationData.default_model_id);
  state.cutoff = nearestIndex(selectedPool().sweep.threshold, selectedPool().threshold);
}
render();
