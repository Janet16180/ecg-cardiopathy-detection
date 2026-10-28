import test from 'node:test';
import assert from 'node:assert/strict';
import {matrixAt, nearestIndex, parseCanonicalCSV} from '../ui/static/matrix.mjs';
import {ageComparisonRows} from '../ui/static/comparison.mjs';
import {thresholdImpact, impactPlots} from '../ui/static/threshold_impact.mjs';
import {copy} from '../ui/static/i18n.mjs';

test('saved cutoff extremes, ties and zero denominator use exact matrices', () => {
  // Strict >: score .5 ties are not referred at threshold .5.
  const sweep = {threshold:[-.01,.5,1],tp:[2,1,0],tn:[0,2,2],fp:[2,0,0],fn:[0,1,2]};
  assert.equal(matrixAt(sweep,0).referral_fraction,1);
  assert.equal(matrixAt(sweep,1).tp,1);
  assert.equal(matrixAt(sweep,1).sensitivity,.5);
  assert.equal(matrixAt(sweep,2).referral_fraction,0);
  assert.equal(matrixAt(sweep,2).precision,null);
  assert.throws(() => matrixAt(sweep,.5),RangeError);
  assert.equal(nearestIndex(sweep.threshold,.49),1);
});

test('empty and single-class samples preserve undefined rates', () => {
  const sweep = {threshold:[.2],tp:[0],tn:[4],fp:[0],fn:[0]};
  assert.equal(matrixAt(sweep,0).sensitivity,null);
  assert.equal(matrixAt(sweep,0).specificity,1);
  assert.equal(matrixAt({...sweep,tn:[0]},0).referral_fraction,null);
});

test('CSV never infers lead names, units or missing samples', () => {
  const header='I,II,III,aVR,aVL,aVF,V1,V2,V3,V4,V5,V6';
  const text=header+'\n'+Array(5000).fill('NaN,0,0,0,0,0,0,0,0,0,0,0').join('\n');
  const value=parseCanonicalCSV(text);
  assert.equal(value.samples.length,12);
  assert.equal(value.samples[0][0],null);
  assert.equal(value.samples[1].length,5000);
  assert.equal(value.units,'mV');
  assert.throws(() => parseCanonicalCSV(text.replace('aVR','unknown')));
  assert.throws(() => parseCanonicalCSV(text.replace('NaN','not-a-number')));
});

test('age comparison pairs actual model IDs at their own fixed cutoff in the same site', () => {
  const point = (pool, threshold, tp) => ({pool, comparison:'>', threshold,
    metrics:{tp,fn:10-tp,fp:1,tn:19,ecgs:30,positives:10,negatives:20},
    sweep:{threshold:[-1,2],tp:[10,0],fn:[0,10],fp:[20,0],tn:[0,20]}});
  const data = {models:[
    {id:'all_age_blend',evaluations:[point('ptb',.3,6),point('sph',.4,7)]},
    {id:'younger_hgb',evaluations:[point('sph',.9,1)]},
    {id:'younger_blend',evaluations:[point('sph',.6,8),point('ptb',.7,9)]},
  ]};
  const before = JSON.stringify(data);
  const sph = ageComparisonRows(data,'sph');
  assert.deepEqual(sph.map(row => [row.id,row.pool,row.threshold,row.metrics.tp,row.metrics.fn]),
    [['younger_blend','sph',.6,8,2],['all_age_blend','sph',.4,7,3]]);
  assert.deepEqual(ageComparisonRows(data,'ptb').map(row => [row.threshold,row.metrics.tp]),[[.7,9],[.3,6]]);
  assert.equal(sph[0].metrics,data.models[2].evaluations[0].metrics);
  assert.equal(JSON.stringify(data),before);
  assert.deepEqual(ageComparisonRows(data,'unknown'),[]);
  assert.deepEqual(ageComparisonRows({models:data.models.slice(1)},'sph'),[]);
  data.models[0].evaluations[1].metrics.ecgs = 31;
  assert.deepEqual(ageComparisonRows(data,'sph'),[]);
});

const impactPool = () => ({threshold:.5,sweep:{threshold:[-.01,.5,1],
  tp:[2,1,0],fn:[0,1,2],fp:[3,0,0],tn:[0,3,3]}});
const plotFormat = {integer:String,cutoff:value=>Number(value).toPrecision(7),
  percent:value=>value === null ? 'undefined' : `${(100*value).toFixed(1)}%`};

test('threshold plots use exact matrix points and a separate fixed calibration marker', () => {
  const pool=impactPool(), before=JSON.stringify(pool);
  const low=thresholdImpact(pool,0), high=thresholdImpact(pool,2);
  assert.deepEqual(low.selected,matrixAt(pool.sweep,0));
  assert.deepEqual(high.selected,matrixAt(pool.sweep,2));
  assert.deepEqual(low.frozen,high.frozen);
  assert.equal(low.frozenIndex,1);
  assert.equal(low.selected.referral_fraction,1);
  assert.equal(high.selected.referral_fraction,0);
  assert.equal(high.selected.sensitivity,0);
  assert.equal(JSON.stringify(pool),before);
  const html=impactPlots(pool,0,copy.en,plotFormat);
  assert.equal((html.match(/class="current-point"/g)||[]).length,5);
  assert.equal((html.match(/class="frozen-point"/g)||[]).length,5);
  assert.equal((html.match(/class="saved-point /g)||[]).length,15);
  assert.match(html,/class="current-point" data-index="0" data-series="tp" data-threshold="-0.01" data-value="2"/);
  assert.match(html,/class="frozen-point" data-index="1" data-series="tp" data-threshold="0.5" data-value="1"/);
  assert.doesNotMatch(html,/<path|<polyline/);
});

test('threshold plots preserve undefined sensitivity and reject inconsistent populations', () => {
  const pool={threshold:.5,sweep:{threshold:[.5],tp:[0],fn:[0],fp:[0],tn:[4]}};
  assert.equal(thresholdImpact(pool,0).selected.sensitivity,null);
  const html=impactPlots(pool,0,copy.en,plotFormat);
  assert.doesNotMatch(html,/NaN|Infinity|data-value="null"/);
  assert.equal((html.match(/class="current-point"/g)||[]).length,4);
  assert.throws(()=>thresholdImpact({...pool,threshold:.6},0),RangeError);
  assert.throws(()=>thresholdImpact(impactPool(),3),RangeError);
  const inconsistent=impactPool();inconsistent.sweep.tp[0]=3;
  assert.throws(()=>thresholdImpact(inconsistent,0),/same reference population/);
});

test('threshold graphs provide translated accessible descriptions and class denominators', () => {
  const html=impactPlots(impactPool(),1,copy.es,plotFormat);
  assert.match(html,/¿Qué cambia con el umbral\?/);
  assert.match(html,/Sensibilidad frente a carga de derivación/);
  assert.match(html,/role="img" aria-labelledby="positiveCounts-title positiveCounts-desc"/);
  assert.match(html,/n = 2/);
  assert.match(html,/n = 3/);
  assert.match(html,/TP \/ 2/);
  assert.match(html,/\(TP \+ FP\) \/ 5/);
});
