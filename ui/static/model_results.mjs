import {fraction} from './matrix.mjs';

export function modelResults(evaluation, pool, sort = 'sensitivity', direction = 'desc') {
  if (!['sensitivity','auroc','fn','fp','referral_fraction'].includes(sort) || !['asc','desc'].includes(direction)) {
    throw new RangeError('Choose an available descriptive result ordering.');
  }
  const rows = evaluation.models.map(model => {
    const point = model.evaluations.find(value => value.pool === pool);
    if (!point || point.comparison !== '>') throw new Error('Each model needs the same development site.');
    return {id:model.id,label:model.label,threshold:point.threshold,...point.metrics,
      false_positive_rate:fraction(point.metrics.fp,point.metrics.negatives)};
  });
  if (rows.some(row => ['ecgs','positives','negatives'].some(key => row[key] !== rows[0][key]))) {
    throw new Error('Model results must share reference denominators.');
  }
  for (const metric of ['sensitivity','auroc']) {
    const values = rows.map(row => row[metric]).filter(value => value !== null);
    const maximum = values.length ? Math.max(...values) : null;
    rows.forEach(row => { row[`highest_${metric}`] = maximum !== null && row[metric] === maximum; });
  }
  return rows.sort((a,b) => {
    if (a[sort] === null || b[sort] === null) return (a[sort] === null) - (b[sort] === null) || a.id.localeCompare(b.id,'en');
    return (direction === 'asc' ? 1 : -1) * (a[sort] - b[sort]) || a.id.localeCompare(b.id,'en');
  });
}
