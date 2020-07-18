import pandas as pd

results = pd.read_csv('categorical_results.csv')

grouped = results.groupby(['model', 'partition', 'subtask'])

results_meaned = []

for group, data in grouped:
  model, partition, subtask = group
  meaned = data.mean()
  nll, accuracy, abs_error, prop_best, worst, best, aarons_metric = \
    meaned['nll'], meaned['accuracy'], meaned['abs_error'], meaned['prop_best'], meaned['worst'], meaned['best'], meaned['aarons_metric']
  score_mod = (accuracy - worst) / (best - worst)
  results_meaned.append([model, partition, subtask, score_mod, nll, accuracy, abs_error, prop_best, worst, best, aarons_metric])

df = pd.DataFrame(results_meaned, columns=['model','partition','subtask','score_mod', 'nll','accuracy','abs_error','prop_best','worst','best','aarons_metric'])
df.to_csv('categorical_results_mean.csv', sep=',',index=False)

