import {copy} from './i18n.mjs';
import {fraction, matrixAt, nearestIndex, parseCanonicalCSV} from './matrix.mjs';
import {ageComparisonRows} from './comparison.mjs';
import {impactPlots} from './threshold_impact.mjs';

const state = {lang:'en', evaluation:null, cohorts:null, status:null, model:0, pool:0, cutoff:0};
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
  return heading('title', 'intro') + notice('scope') + `<div class="cards">${cards}</div>` + panel(`<h2>${t('execution')}</h2>${statuses}`);
}

function matrixHTML(matrix) {
  const cells = (left, right, denominator, errorLeft) => `<td class="${errorLeft ? 'error' : ''}">${left}<strong>${integer(matrix[left.toLowerCase()])}</strong><span>/ ${integer(denominator)}</span></td><td class="${errorLeft ? '' : 'error'}">${right}<strong>${integer(matrix[right.toLowerCase()])}</strong><span>/ ${integer(denominator)}</span></td>`;
  return `<table class="matrix"><thead><tr><th>${t('reference')}</th><th>${t('flagged')}</th><th>${t('notFlagged')}</th></tr></thead><tbody><tr><th>${t('positive')}<br><small>n = ${integer(matrix.positives)}</small></th>${cells('TP', 'FN', matrix.positives, false)}</tr><tr><th>${t('negative')}<br><small>n = ${integer(matrix.negatives)}</small></th>${cells('FP', 'TN', matrix.negatives, true)}</tr></tbody></table>`;
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

function evaluation() {
  const header = heading('evaluation', 'evalIntro') + notice('scope') + `<p class="muted">${t('offlineEvaluation')}</p>`;
  if (!state.evaluation) return header + panel(`<h2>${t('noEvaluation')}</h2><p>${t('noEvaluationText')}</p>`);
  const model = state.evaluation.models[state.model], pool = selectedPool(), matrix = pool.metrics;
  const controls = `<div class="controls"><label>${t('model')}<select id="model">${state.evaluation.models.map((value, index) => `<option value="${index}" ${index === state.model ? 'selected' : ''}>${escape(value.label)}</option>`).join('')}</select></label><label>${t('pool')}<select id="pool">${model.evaluations.map((value, index) => `<option value="${index}" ${index === state.pool ? 'selected' : ''}>${escape(value.label)}</option>`).join('')}</select></label></div>`;
  const descriptors = `<div class="cards"><div class="card"><small>AUROC · ${escape(pool.label)}</small><div class="value">${decimal(matrix.auroc)}</div><small>Average precision: ${decimal(matrix.average_precision)}</small></div><div class="card"><small>${t('records')}</small><div class="value">${integer(matrix.ecgs)}</div><small>${t('positive')}: ${integer(matrix.positives)} · ${t('negative')}: ${integer(matrix.negatives)}</small></div><div class="card"><small>${t('prevalence')}</small><div class="value">${percent(fraction(matrix.positives, matrix.ecgs))}</div><small>${integer(matrix.positives)} / ${integer(matrix.ecgs)} ${t('records')}</small></div></div>`;
  const fixed = panel(`<h2>${t('frozen')}</h2><p class="muted">${t('frozenText')}</p><p class="threshold">score &gt; ${cutoff(pool.threshold)}</p><p class="muted">${t('calibration')}: ${integer(pool.calibration.flagged_normals)} / ${integer(pool.calibration.normals)} · ${t('allowed')}: ${integer(pool.calibration.allowed_normals)} · ${t('tied')}: ${integer(pool.calibration.tied_normals)}</p><div class="grid-two">${matrixHTML(matrix)}${ratesHTML(matrix)}</div>`);
  const explore = panel(`<h2>${t('explore')}</h2><p class="muted">${t('exploreText')}</p><div class="slider-label"><label for="cutoff">${t('threshold')}</label><output id="cutoff-value" class="threshold"></output></div><input id="cutoff" class="slider" type="range" min="0" max="${pool.sweep.threshold.length - 1}" step="1" value="${state.cutoff}" aria-describedby="cutoff-help"><div class="slider-label"><small id="cutoff-help">${t('comparison')}</small><button class="secondary" id="reset">${t('reset')}</button></div><div class="grid-two"><div id="exploratory-matrix"></div><div id="exploratory-rates"></div></div><div id="threshold-impact"></div><p id="threshold-summary" class="sr-only" aria-live="polite" aria-atomic="true"></p>`);
  return header + primaryHTML() + controls + ageComparisonHTML(pool) + descriptors + fixed + explore + `<details><summary>${t('provenance')}</summary><p><small>${t('artifactHash')}</small><br><code>${escape(state.evaluation.artifact_sha256)}</code></p></details>`;
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

function localization() {
  return heading('wipTitle', 'wipIntro') + `<span class="badge">WIP · UNVALIDATED</span>` + panel(`<h2>${t('wipStatus')}</h2><p>${t('wipText')}</p>`);
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
  const routes = {overview,evaluation,recording,cohorts,localization,methods};
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
  if (active === 'evaluation') setupEvaluation();
  if (active === 'recording') document.querySelector('#analyze').addEventListener('click', analyzeRecording);
}

window.addEventListener('hashchange', render);
document.querySelector('#english').addEventListener('click', () => { state.lang = 'en'; render(); });
document.querySelector('#spanish').addEventListener('click', () => { state.lang = 'es'; render(); });
document.querySelector('#theme').addEventListener('click', () => {
  document.documentElement.dataset.theme = document.documentElement.dataset.theme === 'dark' ? 'light' : 'dark';
});
render();
const [status, evaluationData, cohortData] = await Promise.all([
  request('/api/status').catch(() => null), request('/api/evaluation').catch(() => null), request('/api/cohorts').catch(() => null),
]);
state.status = status; state.evaluation = evaluationData; state.cohorts = cohortData;
if (evaluationData) {
  state.model = evaluationData.models.findIndex(model => model.id === evaluationData.default_model_id);
  state.cutoff = nearestIndex(selectedPool().sweep.threshold, selectedPool().threshold);
}
render();
