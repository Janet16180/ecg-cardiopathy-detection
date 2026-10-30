/* Normal-beat replacement (NBR-01): score after replacing each lead and wave by its normal counterpart. */
const escape = value => String(value ?? '').replace(/[&<>"']/g, char => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[char]));

export const flagged = (score, threshold) => score > threshold;

export function regionValues(scores, energy, view) {
  if (view === 'energy') return {cells: energy, leads: null, components: null, full: null};
  if (view === 'swap') return {
    cells: scores.self_swap_grid.map(row => row.map(value => scores.original - value)),
    leads: scores.self_swap_lead?.map(value => scores.original - value) ?? null, components: null,
    full: scores.self_swap_full === undefined ? null : scores.original - scores.self_swap_full,
  };
  if (view === 'hgb' || view === 'xecg') {
    const base = scores[`${view}_original`];
    return {cells: scores[`${view}_grid`].map(row => row.map(value => base - value)), leads: null, components: null, full: null};
  }
  return {
    cells: scores.grid.map(row => row.map(value => scores.original - value)),
    leads: scores.lead.map(value => scores.original - value),
    components: scores.component.map(value => scores.original - value),
    full: scores.original - scores.full,
  };
}

export function flips(scores, threshold) {
  const before = flagged(scores.original, threshold);
  return scores.grid.map(row => row.map(value => before && !flagged(value, threshold)));
}

export function topRegions(scores, threshold, count = 3) {
  const before = flagged(scores.original, threshold), regions = [];
  scores.grid.forEach((row, lead) => row.forEach((after, component) => regions.push({lead, component, after, drop: scores.original - after, flips: before && !flagged(after, threshold)})));
  return regions.sort((a, b) => b.drop - a.drop).slice(0, count);
}

export function swapArtifact(scores) {
  const changes = scores.self_swap_grid.flat().map(value => Math.abs(scores.original - value));
  return {mean: changes.reduce((sum, value) => sum + value, 0) / changes.length, max: Math.max(...changes)};
}

export function replacedWindows(example, selection) {
  const all = example.components;
  const last = example.signal[0].length - 1;
  const clamp = windows => windows.map(([start, end]) => [Math.max(0, start), Math.min(last, end)]).filter(([start, end]) => end > start);
  const union = clamp(Object.values(all).flat());
  return example.leads.map((_, lead) => {
    if (selection.kind === 'full') return union;
    if (selection.kind === 'component') return clamp(all[example.componentNames[selection.component]]);
    if (selection.lead !== lead) return [];
    return selection.kind === 'lead' ? union : clamp(all[example.componentNames[selection.component]]);
  });
}

export function regionTable(example, scores, threshold, view, selection, labels, format) {
  const values = regionValues(scores, example.energy_grid, view);
  const changes = view === 'blend' ? flips(scores, threshold) : null;
  const largest = Math.max(...[...values.cells.flat(), ...(values.leads ?? [])].filter(value => value > 0), 1e-9);
  const largestAll = Math.max(...[...(values.components ?? []), values.full ?? 0].filter(value => value > 0), 1e-9);
  const pressed = (kind, lead, component) => selection.kind === kind && (lead === undefined || selection.lead === lead) && (component === undefined || selection.component === component);
  const bar = (value, scale) => value > 0 ? `<svg class="region-bar" viewBox="0 0 100 1" preserveAspectRatio="none" aria-hidden="true"><rect width="${Math.min(100, 100 * value / scale).toFixed(1)}" height="1"/></svg>` : '';
  const cell = (value, attributes, isPressed, flip, title, scale = largest) => `<button class="region-cell${value < 0 ? ' negative' : ''}" ${attributes} aria-pressed="${isPressed}" title="${escape(title)}">${bar(value, scale)}<span class="region-number">${value === null ? '—' : format.value(value, view)}</span>${flip ? `<span class="region-flip">${escape(labels.flip)}</span>` : ''}</button>`;
  const header = `<tr><th scope="col">${escape(labels.lead)}</th>${example.componentNames.map((name, component) => `<th scope="col"><button class="region-head" data-component="${component}" aria-pressed="${pressed('component', undefined, component)}">${escape(name)}</button></th>`).join('')}<th scope="col">${escape(labels.wholeLead)}</th></tr>`;
  const rows = example.leads.map((name, lead) => `<tr><th scope="row"><button class="region-head" data-lead="${lead}" aria-pressed="${pressed('lead', lead)}">${escape(name)}</button></th>${values.cells[lead].map((value, component) => `<td>${cell(value, `data-lead="${lead}" data-component="${component}"`, pressed('cell', lead, component), changes?.[lead][component], `${name} · ${example.componentNames[component]} · ${labels.after} ${view === 'blend' ? format.score(scores.grid[lead][component]) : format.value(value, view)}`)}</td>`).join('')}<td>${values.leads ? cell(values.leads[lead], `data-lead="${lead}" data-whole="1"`, pressed('lead', lead), view === 'blend' && flagged(scores.original, threshold) && !flagged(scores.lead[lead], threshold), `${name} · ${labels.wholeLead}`) : '—'}</td></tr>`).join('');
  const footer = values.components ? `<tr><th scope="row">${escape(labels.allLeads)}</th>${values.components.map((value, component) => `<td>${cell(value, `data-component="${component}" data-whole="1"`, pressed('component', undefined, component), view === 'blend' && flagged(scores.original, threshold) && !flagged(scores.component[component], threshold), `${example.componentNames[component]} · ${labels.allLeads}`, largestAll)}</td>`).join('')}<td>${cell(values.full, 'data-full="1"', selection.kind === 'full', false, labels.full, largestAll)}</td></tr>` : '';
  return `<table class="data region-table"><caption>${escape(labels.caption)}</caption><thead>${header}</thead><tbody>${rows}${footer}</tbody></table>`;
}

export function ecgPlot(example, selection, labels, showCounterpart = true) {
  const width = 520, height = 78, gap = 16, left = 34, perMv = 24, samples = example.signal[0].length;
  const x = index => left + width * index / (samples - 1);
  const windows = replacedWindows(example, selection);
  const panels = example.leads.map((name, lead) => {
    const column = lead < 6 ? 0 : 1, row = lead % 6;
    const offsetX = column * (width + left + gap), top = 8 + row * (height + 6), centre = top + height / 2;
    const y = value => (centre - perMv * Math.max(-1.6, Math.min(1.6, value))).toFixed(1);
    const path = (values, start, end) => values.slice(start, end + 1).map((value, index) => `${index ? 'L' : 'M'}${(offsetX + x(start + index)).toFixed(1)},${y(value)}`).join('');
    const seconds = Array.from({length: Math.floor((samples - 1) / (example.fs / 5)) + 1}, (_, index) => index).map(index => {
      const position = offsetX + x(index * example.fs / 5);
      return `<line class="${index % 5 ? 'grid-minor' : 'grid-major'}" x1="${position.toFixed(1)}" x2="${position.toFixed(1)}" y1="${top}" y2="${top + height}"/>`;
    }).join('');
    const shade = windows[lead].map(([start, end]) => `<rect class="replaced-band" x="${(offsetX + x(start)).toFixed(1)}" y="${top}" width="${Math.max(1, x(end) - x(start)).toFixed(1)}" height="${height}"/>`).join('');
    const replaced = !showCounterpart ? '' : windows[lead].map(([start, end]) => `<path class="counterpart-trace" d="${path(example.counterpart[lead], start, end)}"/>`).join('');
    return `<g><title>${escape(name)}${windows[lead].length ? ` · ${escape(labels.replaced)}` : ''}</title><rect class="lead-frame" x="${offsetX + left}" y="${top}" width="${width}" height="${height}"/>${seconds}<line class="grid-zero" x1="${offsetX + left}" x2="${offsetX + left + width}" y1="${centre}" y2="${centre}"/>${shade}<path class="original-trace" d="${path(example.signal[lead], 0, samples - 1)}"/>${replaced}<text class="lead-name" x="${offsetX + 4}" y="${centre + 4}">${escape(name)}</text></g>`;
  }).join('');
  const total = 2 * (width + left) + gap, tall = 8 + 6 * (height + 6) + 18;
  return `<svg class="ecg-plot" viewBox="0 0 ${total} ${tall}" role="img" aria-labelledby="ecg-plot-title"><title id="ecg-plot-title">${escape(labels.title)}</title>${panels}<text class="axis-note" x="${left}" y="${tall - 4}">${escape(labels.scale)}</text></svg>`;
}
