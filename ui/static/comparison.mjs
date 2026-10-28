/* Compare fixed operating points by model identity and evaluation site. */
export function ageComparisonRows(evaluation, pool) {
  const names = [['younger_blend', 'agePreferred25k'], ['all_age_blend', 'noAgePreference25k']];
  const rows = names.map(([id, labelKey]) => {
    const model = evaluation.models.find(value => value.id === id);
    const point = model?.evaluations.find(value => value.pool === pool);
    if (!point || point.comparison !== '>') return null;
    return {id, labelKey, pool, threshold:point.threshold, metrics:point.metrics};
  });
  if (!rows.every(Boolean)) return [];
  return ['ecgs', 'positives', 'negatives'].every(key => rows[0].metrics[key] === rows[1].metrics[key]) ? rows : [];
}
