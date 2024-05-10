import pandas as pd
import matplotlib.pyplot as plt
plt.style.use('ggplot')

exclude_rc = True
plot_best_accuracy = False
plot_avg_accuracy = True
num_of_samples_to_skip_for_avg_calc = 0
file_path = '~/Downloads/2024S-BSc-experimental_results_mn10 - mn10-test-instance-acc.csv'
rolling_avg = 150
font_size = 6
whitelist_categories = []
# whitelist_categories = ['ref']
whitelist_categories = ['ref', 'synth05', 'dist20', 'dist50', 'noise25']  # best_only
# whitelist_categories = ['synth05', 'synth10', 'synth15']  # synth
# whitelist_categories = ['noise20', 'noise25', 'noise25_rc', 'noise50', 'noise50_rc', 'noise75', 'noise100']  # noise
# whitelist_categories = ['dist10', 'dist20', 'dist25', 'dist30', 'dist40', 'dist50', 'dist60', 'dist70', 'dist75', 'dist80', 'dist90', 'dist100']  # dist
blacklist_categories = []
blacklist_categories = ['noise70', 'noise100']  # remove outliers

data = pd.read_csv(file_path)
print(data)

best_values = {}
avg_values = {}
xticks = []
for key in data:
    # prepare box plots
    category = key[:key.rindex('_')]
    if exclude_rc and category.endswith('_rc'):
        continue

    if len(whitelist_categories) > 0 and category not in whitelist_categories:
        continue

    if category in blacklist_categories:
        continue

    # plot rolling avg line
    plt.plot(data[key].rolling(rolling_avg).mean(), label=key, linestyle='dashed' if category == 'ref' else 'solid')

    if category not in best_values:
        best_values[category] = []
        avg_values[category] = []

    max = data[key].max()
    mean = data[key][num_of_samples_to_skip_for_avg_calc:].mean()
    best_values[category].append(max)
    avg_values[category].append(mean)
    xticks.append(key)

plt.legend(loc='center left')
plt.title('Rolling avg accuracy')
plt.show()

# generate best acc plot
if plot_best_accuracy:
    _, ax = plt.subplots()
    boxplot_data = [best_values[key] for key in best_values]
    ax.boxplot(boxplot_data)
    ax.set_xticklabels(
        [key for key in best_values],
        fontsize=font_size,
        rotation=90
    )
    plt.title('MAX accuracy')
    plt.show()

# generate avg acc plot
if plot_avg_accuracy:
    _, ax = plt.subplots()
    boxplot_data = [avg_values[key] for key in avg_values]
    ax.boxplot(boxplot_data)
    ax.set_xticklabels(
        [key for key in avg_values],
        fontsize=font_size,
        rotation=90
    )
    plt.title('AVG accuracy')
    plt.show()
