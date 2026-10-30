import assert from 'node:assert/strict';
import test from 'node:test';
import {ecgPlot, flips, regionTable, regionValues, replacedWindows, swapArtifact, topRegions} from '../ui/static/replacement.mjs';

const scores = {
  original: 0.8, full: 0.1, lead: Array(12).fill(0.7), component: [0.79, 0.75, 0.5],
  grid: Array.from({length: 12}, (_, lead) => [0.79, 0.78, lead === 1 ? 0.6 : 0.77]),
  self_swap_grid: Array.from({length: 12}, () => [0.81, 0.79, 0.8]),
  hgb_original: 0.9, hgb_grid: Array.from({length: 12}, () => [0.85, 0.85, 0.85]),
  xecg_original: 0.7, xecg_grid: Array.from({length: 12}, () => [0.7, 0.6, 0.7]),
};
const example = {
  leads: ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6'], componentNames: ['P', 'QRS', 'ST-T'], fs: 250,
  signal: Array.from({length: 12}, () => Array(500).fill(0)), counterpart: Array.from({length: 12}, () => Array(500).fill(0.1)),
  components: {P: [[10, 40]], QRS: [[50, 70]], 'ST-T': [[71, 150]]}, energy_grid: Array.from({length: 12}, () => [1, 2, 3]),
};
const labels = {lead: 'Lead', wholeLead: 'Whole lead', allLeads: 'All', full: 'Full', after: 'after', flip: 'flip', caption: 'c', title: 't', replaced: 'r', scale: 's'};
const format = {score: value => value.toFixed(3), value: value => value.toFixed(3)};

test('drops are original minus score after replacement', () => {
  const values = regionValues(scores, example.energy_grid, 'blend');
  assert.ok(Math.abs(values.cells[1][2] - 0.2) < 1e-12);
  assert.ok(Math.abs(values.full - 0.7) < 1e-12);
  assert.ok(Math.abs(regionValues(scores, null, 'hgb').cells[0][0] - 0.05) < 1e-12);
  assert.deepEqual(regionValues(scores, example.energy_grid, 'energy').cells, example.energy_grid);
});

test('a flip needs the original above and the replaced score at or below the threshold', () => {
  const changed = flips(scores, 0.65);
  assert.equal(changed[1][2], true);
  assert.equal(changed.flat().filter(Boolean).length, 1);
  assert.equal(flips({...scores, original: 0.6}, 0.65).flat().some(Boolean), false);
  assert.equal(flips(scores, 0.6)[1][2], true);
  assert.equal(flips(scores, 0.59).flat().some(Boolean), false);
});

test('top regions are ordered by drop', () => {
  const [first, second] = topRegions(scores, 0.65);
  assert.deepEqual([first.lead, first.component, first.flips], [1, 2, true]);
  assert.ok(first.drop >= second.drop);
});

test('splice artefact is the mean and largest absolute change', () => {
  const artifact = swapArtifact(scores);
  assert.ok(Math.abs(artifact.max - 0.01) < 1e-12);
  assert.ok(artifact.mean > 0 && artifact.mean <= artifact.max);
});

test('selections map to the right replaced windows', () => {
  assert.deepEqual(replacedWindows(example, {kind: 'cell', lead: 7, component: 2})[7], [[71, 150]]);
  assert.deepEqual(replacedWindows(example, {kind: 'cell', lead: 7, component: 2})[6], []);
  assert.equal(replacedWindows(example, {kind: 'lead', lead: 0})[0].length, 3);
  assert.ok(replacedWindows(example, {kind: 'component', component: 1}).every(windows => windows.length === 1));
  assert.deepEqual(replacedWindows(example, {kind: 'full'})[11], [[10, 40], [50, 70], [71, 150]]);
});

test('table and plot render without undefined values', () => {
  for (const view of ['blend', 'hgb', 'xecg', 'energy', 'swap']) {
    const html = regionTable(example, scores, 0.65, view, {kind: 'cell', lead: 1, component: 2}, labels, format);
    assert.ok(!/undefined|NaN/.test(html), view);
  }
  const svg = ecgPlot(example, {kind: 'cell', lead: 1, component: 2}, labels);
  assert.equal((svg.match(/counterpart-trace/g) ?? []).length, 1);
  assert.ok(!/undefined|NaN/.test(svg));
  const hidden = ecgPlot(example, {kind: 'cell', lead: 1, component: 2}, labels, false);
  assert.equal((hidden.match(/counterpart-trace/g) ?? []).length, 0);
  assert.equal((hidden.match(/replaced-band/g) ?? []).length, (svg.match(/replaced-band/g) ?? []).length);
});
