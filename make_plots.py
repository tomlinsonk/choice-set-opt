import os
import pickle

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
from scipy.stats import gaussian_kde


def plot_proof_functions():
    """
    Plot the disagreement functions used in the proofs of Theorems 1 and 2.
    """

    def thm1_D(x):
        return abs(1 / (2 + x) - 3 / (5 + x)) + abs(1 / (2 + x) - 2 / (5 + x))

    def thm2_D(x):
        return abs(2 / (2 + x) - (1 / 2) / ((1 / 2) + x))

    plt.figure(figsize=(2, 1.5))
    x = np.linspace(0, 2, 1000)

    plt.plot(x, thm1_D(x))
    plt.xlim(0, 2)
    plt.ylim(0.15, 0.22)
    plt.vlines(1, 0, 1, linestyles='dashed', colors='grey', alpha=0.5)
    plt.hlines(1 / 6, 0, 2, linestyles='dashed', colors='grey', alpha=0.5)
    plt.ylabel('$D(Z)$')
    plt.xlabel('$s_Z / t$')
    plt.savefig('plots/thm1_D.pdf', bbox_inches='tight')
    plt.xticks(range(3), range(3))
    plt.close()

    print(f'Saved plot to: plots/thm1_D.pdf')

    plt.figure(figsize=(2, 1.5))
    x = np.linspace(0, 5, 1000)
    plt.vlines(1, 0, 1, linestyles='dashed', colors='grey', alpha=0.5)
    plt.hlines(1 / 3, 0, 5, linestyles='dashed', colors='grey', alpha=0.5)
    plt.plot(x, thm2_D(x))
    plt.xlim(0, 5)
    plt.ylim(0, 0.4)
    plt.xticks(range(6), range(6))

    plt.ylabel('$D(Z)$')
    plt.xlabel('$s_Z / t$')
    plt.savefig('plots/thm2_D.pdf', bbox_inches='tight')
    plt.close()

    print(f'Saved plot to: plots/thm2_D.pdf')


def plot_allstate():
    with open('results/allstate_all_pairs_promo_all_epsilon_results.pickle', 'rb') as f:
        data = pickle.load(f)

    with open('results/allstate_all_pairs_promo_brute_force_results.pickle', 'rb') as f:
        brute_force_data = pickle.load(f)

    epsilons = sorted(data.keys())
    sets_computed = [
        [num_sets_computed for _, _, _, _, _, _, num_sets_computed, _ in data[epsilon] if num_sets_computed > 0] for
        epsilon in epsilons]
    approx_num_solved = [np.count_nonzero([approx_success for _, _, _, approx_success, _, _, _, _ in data[epsilon]]) for
                         epsilon in epsilons]
    greedy_num_solved = [np.count_nonzero([greedy_success for _, _, greedy_success, _, _, _, _, _ in data[epsilon]]) for
                         epsilon in epsilons]
    opt_num_solved = np.count_nonzero([opt_success for _, _, _, opt_success, _, _, _, _ in brute_force_data])

    attempts = np.count_nonzero(sets_computed, axis=1)[0]

    plt.figure(figsize=(3, 2.2))

    # Good boxplot widths (https://stackoverflow.com/questions/46687062/matplotlib-boxplot-width-in-log-scale)
    w = 0.2
    width = lambda p, w: 10 ** (np.log10(p) + w / 2) - 10 ** (np.log10(p) - w / 2)

    plt.boxplot(sets_computed, positions=epsilons, widths=width(epsilons, w), showfliers=False)
    plt.hlines(2 ** 22, 0.05, 1500, ls='dashed', alpha=0.2)
    plt.xscale('log')
    plt.yscale('log')

    plt.xlim(0.1, 1400)
    plt.xlabel('Approximation $\\varepsilon$', fontsize=12)
    plt.ylabel('# Sets Computed', fontsize=12)
    plt.tick_params(labelsize=12)
    plt.xticks([10 ** i for i in range(-1, 4)], ['$10^{{{}}}$'.format(i) for i in range(-1, 4)])
    plt.yticks([10 ** i for i in range(2, 8)], ['$10^{{{}}}$'.format(i) for i in range(2, 8)])

    plt.savefig('plots/allstate_all_pairs_promo_sets_computed.pdf', bbox_inches='tight')
    print(f'Saved plot to: plots/allstate_all_pairs_promo_sets_computed.pdf')

    plt.close()

    plt.figure(figsize=(3, 2.2))

    plt.plot(epsilons, np.array(approx_num_solved) / attempts, '.-', label='Algorithm 1')
    plt.plot(epsilons, np.array(greedy_num_solved) / attempts, label='Greedy', ls='dotted')
    plt.hlines(opt_num_solved / attempts, 0.05, 1500, label='Brute force', ls='dashed')

    plt.xscale('log')
    plt.ylim(0, 1)
    plt.xlim(0.1, 1400)
    plt.xlabel('Approximation $\\varepsilon$', fontsize=12)
    plt.ylabel('Frac. Instances Solved', fontsize=12)
    plt.xticks([10 ** i for i in range(-1, 4)], ['$10^{{{}}}$'.format(i) for i in range(-1, 4)])
    plt.tick_params(labelsize=12)
    plt.legend(loc='lower left')
    plt.savefig('plots/allstate_all_pairs_promo_successes.pdf', bbox_inches='tight')
    print(f'Saved plot to: plots/allstate_all_pairs_promo_successes.pdf')

    plt.close()


def plot_all_pairs_agree():
    np.random.seed(0)
    label_coords = {'Allstate': {'mnl': (0.03, 0.83), 'cdm': (0.03, 0.83)},
                    'YOOCHOOSE': {'mnl': (0.03, 0.83), 'cdm': (0.03, 0.83)}}
    label_align = {'Allstate': {'mnl': 'left', 'cdm': 'left'},
                   'YOOCHOOSE': {'mnl': 'left', 'cdm': 'left'}}

    fig, axes = plt.subplots(2, 2, figsize=(5, 4.8), sharey='row')

    ip_fig, ip_axes = plt.subplots(2, 1, figsize=(3.5, 4.8))

    for row, dataset in enumerate(('Allstate', 'YOOCHOOSE')):
        with open(f'results/{dataset.lower()}_all_pairs_agreement.pickle', 'rb') as f:
            data = pickle.load(f)

        epsilons = {model_name: epsilon for model_name, epsilon in data.keys()}

        for col, model in enumerate(('mnl', 'cdm')):
            agree_approx_Ds = []
            agree_greedy_Ds = []
            agree_ip_Ds = []
            disagree_approx_Ds = []
            disagree_greedy_Ds = []
            disagree_ip_Ds = []
            num_sets_computeds = []
            for choice_set, greedy_min_D, approx_min_D, ip_min_D, greedy_min_Z, approx_min_Z, ip_min_Z, greedy_max_D, approx_max_D, ip_max_D, greedy_max_Z, approx_max_Z, ip_max_Z, num_sets_computed, total_sets in data[model, epsilons[model]]:
                agree_approx_Ds.append(approx_min_D)
                agree_greedy_Ds.append(greedy_min_D)
                agree_ip_Ds.append(ip_min_D)
                disagree_approx_Ds.append(approx_max_D)
                disagree_greedy_Ds.append(greedy_max_D)
                disagree_ip_Ds.append(ip_max_D)
                num_sets_computeds.append(num_sets_computed)

            agree_approx_Ds = np.array(agree_approx_Ds)
            agree_greedy_Ds = np.array(agree_greedy_Ds)
            agree_ip_Ds = np.array(agree_ip_Ds)
            disagree_approx_Ds = np.array(disagree_approx_Ds)
            disagree_greedy_Ds = np.array(disagree_greedy_Ds)
            disagree_ip_Ds = np.array(disagree_ip_Ds)

            pct_sets_computed = np.mean(num_sets_computeds) / total_sets * 100

            if model == 'mnl':
                ip_axes[row].boxplot([agree_approx_Ds - agree_ip_Ds, disagree_approx_Ds - disagree_ip_Ds],
                                       showfliers=False, zorder=10, widths=0.3, boxprops={'linewidth': 2},
                                       whiskerprops={'linewidth': 2}, medianprops={'linewidth': 2},
                                       capprops={'linewidth': 2})
                ip_axes[row].plot((1, 2), (
                    np.mean(agree_approx_Ds - agree_ip_Ds), np.mean(disagree_approx_Ds - disagree_ip_Ds)), '.',
                                    markerfacecolor='white', markersize=8, marker='X', zorder=100, markeredgewidth=1,
                                    markeredgecolor='black')
                ip_axes[row].scatter((0.75 + np.random.random(len(agree_approx_Ds)) / 2, 1.75 + np.random.random(len(agree_approx_Ds)) / 2),
                                     [agree_approx_Ds - agree_ip_Ds, disagree_approx_Ds - disagree_ip_Ds],
                                       color='darkblue', alpha=0.2, s=8, zorder=0, marker='o', linewidths=0)

            axes[row, col].boxplot([agree_approx_Ds - agree_greedy_Ds, disagree_approx_Ds - disagree_greedy_Ds],
                                   showfliers=False, zorder=10, widths=0.3, boxprops={'linewidth': 2},
                                   whiskerprops={'linewidth': 2}, medianprops={'linewidth': 2},
                                   capprops={'linewidth': 2})
            axes[row, col].plot((1, 2), (
                np.mean(agree_approx_Ds - agree_greedy_Ds), np.mean(disagree_approx_Ds - disagree_greedy_Ds)), '.',
                                markerfacecolor='white', markersize=8, marker='X', zorder=100, markeredgewidth=1,
                                markeredgecolor='black')

            # Gaussian KDE
            # (https://stackoverflow.com/questions/8671808/matplotlib-avoiding-overlapping-datapoints-in-a-scatter-dot-beeswarm-plot)
            agree_data = agree_approx_Ds - agree_greedy_Ds
            try:
                agree_density = gaussian_kde(agree_data)(agree_data)
            except np.linalg.LinAlgError:
                agree_density = np.full_like(agree_data, 1)
            agree_jitter = np.random.rand(*agree_data.shape) - 0.5
            agree_xvals = 1 + (agree_density * agree_jitter / max(agree_density) / 1.5)

            disagree_data = disagree_approx_Ds - disagree_greedy_Ds
            try:
                disagree_density = gaussian_kde(disagree_data)(disagree_data)
            except np.linalg.LinAlgError:
                disagree_density = np.full_like(disagree_data, 1)
            disagree_jitter = np.random.rand(*disagree_data.shape) - 0.5
            disagree_xvals = 2 + (disagree_density * disagree_jitter / max(disagree_density) / 1.5)

            axes[row, col].scatter((agree_xvals, disagree_xvals), (agree_data, disagree_data),
                                   color='darkblue', alpha=0.2, s=8, zorder=0, marker='o', linewidths=0)
            pct_sets_string = f'{pct_sets_computed:.6f}% sets' if dataset == 'YOOCHOOSE' else f'{pct_sets_computed:.0f}% sets'
            axes[row, col].annotate(f'$\\varepsilon={epsilons[model]}$\n{pct_sets_string}',
                                    xy=label_coords[dataset][model], xycoords='axes fraction', fontsize=10,
                                    ha=label_align[dataset][model])

            axes[row, col].tick_params(labelsize=9)
            if col == 0:
                axes[row, col].set_ylabel('Alg. 1 $D(Z)$ - Greedy $D(Z)$')
                ip_axes[row].set_ylabel(f'{dataset}\nAlg. 1 $D(Z)$ - MIBLP $D(Z)$')
            else:
                axes[row, col].annotate(dataset, xy=(1, 0.5), xytext=(-axes[row, col].yaxis.labelpad + 20, 0),
                                        xycoords='axes fraction', textcoords='offset points',
                                        fontsize=14, ha='right', va='center', rotation=270)

            if row == 0:
                axes[row, col].annotate('MNL' if model == 'mnl' else 'Rank-2 CDM', xy=(0.5, 1), xytext=(0, 5),
                                        xycoords='axes fraction', textcoords='offset points',
                                        fontsize=14, ha='center', va='baseline')
                axes[row, col].set_xticks([])
                axes[row, col].set_xticklabels([])
                ip_axes[row].set_xticks([])
                ip_axes[row].set_xticklabels([])
            else:
                axes[row, col].set_xticks([1, 2])
                axes[row, col].set_xticklabels(['Agreement', 'Disagreement'])
                ip_axes[row].set_xticks([1, 2])
                ip_axes[row].set_xticklabels(['Agreement', 'Disagreement'])

            axes[row, col].axhline(0, 0, 3, color='black', linewidth=0.8)
            axes[row, col].set_yscale('symlog', linthreshy=0.0011)
            ip_axes[row].set_yscale('symlog', linthreshy=0.0011)


    plt.figure(fig.number)
    plt.tight_layout(h_pad=0, w_pad=0.3)
    plt.savefig('plots/all_pairs_agree.pdf', bbox_inches='tight')
    print('Saved plot to: plots/all_pairs_agree.pdf')
    plt.close()

    plt.figure(ip_fig.number)
    ip_axes[0].yaxis.set_major_formatter(ticker.FormatStrFormatter('%.2e'))
    plt.tight_layout(h_pad=0, w_pad=0.3)
    plt.savefig('plots/ip_all_pairs_agree.pdf', bbox_inches='tight')
    print('Saved plot to: plots/ip_all_pairs_agree.pdf')
    plt.close()


if __name__ == '__main__':
    os.makedirs('plots', exist_ok=True)
    plot_all_pairs_agree()
    plot_allstate()
    plot_proof_functions()
