import test from 'node:test';
import assert from 'node:assert/strict';
import {matrixAt, nearestIndex, parseCanonicalCSV} from '../ui/static/matrix.mjs';

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
