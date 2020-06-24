import argparse
import glob
import itertools
import os
import pickle
from functools import partial
from multiprocessing import cpu_count
from multiprocessing.pool import Pool

import dateutil.parser
import numpy as np
import scipy.io as sio
import torch
from tqdm import tqdm

import optimize_choice_sets
from choice_models import DiscreteChoiceSetting, CDM, MNL, NLNode, NL
from inference import TorchLowRankCDM, fit, TorchMNL, TorchNL


def get_sf_work_tree():
    """
    Build the Model 28W tree from p. 185 of:
    Koppelman, F. S. and Bhat, C. A self instructing course in mode choice modeling:
        Multinomial and nested logit models, 2006.
    :return: A NLNode
    """
    items = {entry: i for i, entry in enumerate(
        ['Drive Alone', 'Shared Ride 2', 'Shared Ride 3+', 'Transit', 'Bike', 'Walk']
    )}

    return NLNode(
        NLNode(
            NLNode(item=items['Drive Alone']),
            NLNode(item=items['Transit']),
            NLNode(
                NLNode(item=items['Shared Ride 2']),
                NLNode(item=items['Shared Ride 3+'])
            )
        ),
        NLNode(
            NLNode(item=items['Walk']),
            NLNode(item=items['Bike'])
        )
    )


def convert_data_to_torch_format(choice_sets, choices):
    """
    Convert a binary choice sets matrix and choice indices into a format for the PyTorch optimizer
    :param choice_sets: (num observations) x |U| binary array indicating which items were present in each choice set
    :param choices: (num observations) x 1 array storing the index of the item chosen in each observation
    :return: a triple of tensors (choice sets, choice set sizes, choices), with the item len(choice_sets[0]) as padding
    """
    m = len(choice_sets)
    n = len(choice_sets[0])

    choice_sets_tensor = torch.empty(m, n, dtype=torch.long)
    choice_set_sizes_tensor = torch.empty(m, dtype=torch.long)
    choices_tensor = torch.empty(m, dtype=torch.long)

    for i in range(m):
        size = 0
        for j in range(n):
            if choice_sets[i, j] == 1:
                if choices[i] == j:
                    choices_tensor[i] = size
                choice_sets_tensor[i, size] = j
                size += 1

        choice_set_sizes_tensor[i] = size

        while size < n:
            choice_sets_tensor[i, size] = n
            size += 1

    return choice_sets_tensor, choice_set_sizes_tensor, choices_tensor.long()


def load_allstate():
    """
    Load the Allstate dataset (https://www.kaggle.com/c/allstate-purchase-prediction-challenge), which must be placed
    in data/ with train.csv uncompressed.
    :return:
    """
    data = np.genfromtxt('data/allstate-purchase-prediction-challenge/train.csv', delimiter=',', names=True, dtype=int,
                         usecols=(0, 2, 8, 17, 18, 19, 20, 21, 22, 23, 24))

    unique_items = np.unique(data[['A', 'B', 'C']], axis=0)
    indivs = np.unique(data['customer_ID'])

    item_indices = {tuple(item): i for i, item in enumerate(unique_items)}
    indiv_indices = {name: i for i, name in enumerate(indivs)}

    item_costs = np.zeros(len(unique_items))
    item_counts = np.zeros(len(unique_items))

    homeown = np.zeros(len(indivs), dtype=bool)
    choice_sets = np.zeros((len(indivs), len(unique_items)), dtype=int)
    choices = np.zeros((len(indivs), 1), dtype=int)
    for row in data:
        item_index = item_indices[tuple(row[['A', 'B', 'C']])]
        indiv_index = indiv_indices[row['customer_ID']]

        item_costs[item_index] += row['cost']
        item_counts[item_index] += 1

        choice_sets[indiv_index, item_index] = 1
        if row['record_type'] == 1:
            choices[indiv_index] = item_index
            homeown[indiv_index] = row['homeowner']

    item_costs /= item_counts

    print('# Items:', len(unique_items))
    print('# Observations:', len(choice_sets))
    print('# Choice Sets:', len(np.unique(choice_sets, axis=0)))

    print('Inferring rank 2 CDM (all data)... ')
    torch_cdm = TorchLowRankCDM(len(unique_items), 2)
    fit(torch_cdm, convert_data_to_torch_format(choice_sets, choices))
    one_agent_cdm = CDM(DiscreteChoiceSetting([0], unique_items, []), np.array([torch_cdm.get_u_p()]))

    def allstate_load_fn():
        return choice_sets, choices, homeown, unique_items

    return one_agent_cdm, allstate_load_fn, data, unique_items, item_indices, choice_sets, choices, item_costs, item_counts


def load_yoochoose():
    """
    Run all experiments on the YOOCHOOSE dataset. For info on this dataset, see:
        Ben-Shimon, David, et al. RecSys Challenge 2015 and the YOOCHOOSE Dataset.
            In Proceedings of the 9th ACM Conference on Recommender Systems. 2015.
    """
    print('Loading YOOCHOOSE dataset...')
    yoochoose_buys = np.genfromtxt('data/yoochoose-data/yoochoose-buys.dat', delimiter=',',
                                   dtype=[('session_id', int), ('time', '|U23'), ('item_id', int), ('price', int),
                                          ('quantity', int)])

    yoochoose_clicks = np.loadtxt('data/yoochoose-data/yoochoose-filtered-clicks.dat',
                                  dtype=[('session_id', int), ('time', '|U23'), ('item_id', int), ('category', int)])

    indivs = np.intersect1d(yoochoose_buys['session_id'], yoochoose_clicks['session_id'])
    unique_items, counts = np.unique(yoochoose_clicks['category'], return_counts=True)

    item_indices = {item: i for i, item in enumerate(unique_items)}
    indiv_indices = {name: i for i, name in enumerate(indivs)}

    raw_choice_sets = np.zeros((len(indivs), len(unique_items)), dtype=int)

    item_id_to_category_map = dict()

    for row in yoochoose_clicks:
        if row['session_id'] in indiv_indices:
            indiv = indiv_indices[row['session_id']]
            item = item_indices[row['category']]
            raw_choice_sets[indiv, item] = 1
            item_id_to_category_map[row['item_id']] = row['category']

    # Filter out items that don't appear enough times
    min_item_count = 20
    item_appearances = np.count_nonzero(raw_choice_sets, axis=0)
    filtered_choice_sets = raw_choice_sets[:, item_appearances > min_item_count]
    filtered_items = unique_items[item_appearances > min_item_count]
    filtered_item_indices = {item: i for i, item in enumerate(filtered_items)}

    # Filter out agents with small choice sets
    sizes = np.count_nonzero(filtered_choice_sets, axis=1)
    filtered_indivs = indivs[sizes > 1]
    filtered_indiv_indices = {name: i for i, name in enumerate(filtered_indivs)}
    filtered_choice_sets = filtered_choice_sets[sizes > 1]

    times = np.zeros(len(filtered_indivs))
    time_half = dateutil.parser.parse('2014-08-09').timestamp()
    indiv_choices = [[] for _ in filtered_indivs]
    for row in yoochoose_buys:
        if row['session_id'] in filtered_indivs:
            indiv = filtered_indiv_indices[row['session_id']]
            if row['item_id'] in item_id_to_category_map and item_id_to_category_map[row['item_id']] in filtered_items:
                selection = filtered_item_indices[item_id_to_category_map[row['item_id']]]
                indiv_choices[indiv].append(selection)
                times[indiv] = dateutil.parser.parse(row['time']).timestamp()

                # In case the item was on sale and has 'S' for its category in this case
                filtered_choice_sets[indiv, selection] = 1

    times = np.repeat(times, [len(buys) for buys in indiv_choices], axis=0)
    choice_sets = np.repeat(filtered_choice_sets, [len(buys) for buys in indiv_choices], axis=0)
    choices = np.array([[item] for this_indiv_choices in indiv_choices for item in this_indiv_choices], dtype=int)

    print('# Items:', len(filtered_items))
    print('# Observations', len(choice_sets))
    print('# Choice sets', len(np.unique(choice_sets, axis=0)))

    return choice_sets, choices, times < time_half, filtered_items


def load_sf_work():
    """
    Load and parse SFWork dataset from original .mat file. Each row in the table is a single survey answer. Individuals
    are identified by a "household id" (hhid) and a "person id" (perid) within each household in the original data.
    :return: the tuple (
                (num observations) x |U| binary array indicating which items were present in each choice set,
                (num observations) x 1 array storing the index of the item chosen in each observation,
                mask indicating who lives in the core of SF/Berkeley
                labels for the items
            )
    """

    sf_work = sio.loadmat('data/SFWork.mat')

    indivs = np.unique(np.concatenate((sf_work['perid'], sf_work['hhid']), axis=1), axis=0)
    indiv_indices = {(perid, hhid): [] for perid, hhid in indivs}

    for i in range(len(sf_work['hhid'])):
        indiv_indices[sf_work['perid'][i][0], sf_work['hhid'][i][0]].append(i)

    choice_sets = np.zeros((len(indivs), 6), dtype=int)
    for i, (perid, hhid) in enumerate(indivs):
        for j in indiv_indices[perid, hhid]:
            choice_sets[i, sf_work['alt'][j] - 1] = 1

    choices = np.zeros((len(indivs), 1), dtype=int)
    for i, (perid, hhid) in enumerate(indivs):
        for j in indiv_indices[perid, hhid]:
            if sf_work['chosen'][j]:
                choices[i] = sf_work['alt'][j] - 1

    in_core = np.zeros(len(indivs), dtype=int)
    for i, (perid, hhid) in enumerate(indivs):
        in_core[i] = sf_work['corredis'][indiv_indices[perid, hhid][0]][0]

    choice_labels = ['Drive Alone', 'Shared Ride 2', 'Shared Ride 3+', 'Transit', 'Bike', 'Walk']

    return choice_sets, choices, in_core == 1, choice_labels


def infer_two_agent_models(load_fn, dataset, nl_tree_fn=None):
    """
    Infer MNL, CDM, and (NL) models for the data loaded by load_fn.
    :param load_fn: the function to load the dataset, which should return the tuple below
    :param dataset: the name of the dataset (for filename  to save the models in)
    :param nl_tree_fn: function to get a nested logit tree for this dataset. If none, don't infer nested logit.
    :return: a tuple (MNL, CDM, NL)
    """
    choice_sets, choices, agent_attr, choice_labels = load_fn()

    n = len(choice_sets[0])

    mnl_params = [None, None]
    cdm_params = [None, None]
    nl_params = [None, None]

    for group, indices in enumerate((agent_attr, ~agent_attr)):
        group_choice_sets = choice_sets[indices]
        group_choices = choices[indices]
        group_data = convert_data_to_torch_format(group_choice_sets, group_choices)

        print(f'Group {group} size: {len(group_choices)}')

        # Infer MNL params
        print('Inferring MNL for group', group)
        group_mnl_model = TorchMNL(n)
        fit(group_mnl_model, group_data)
        mnl_params[group] = group_mnl_model.get_utilities()

        # Infer CDM params
        print('Inferring CDM for group', group)
        group_cdm_model = TorchLowRankCDM(n, 2)
        fit(group_cdm_model, group_data)
        cdm_params[group] = group_cdm_model.get_u_p()

        if nl_tree_fn is not None:
            # Infer NL params
            print('Inferring NL for group', group)
            group_nl_tree = nl_tree_fn()
            group_nl_model = TorchNL(n, group_nl_tree)
            fit(group_nl_model, group_data)
            group_nl_model.populate_tree_utilities()
            nl_params[group] = group_nl_tree

    models = MNL(
        DiscreteChoiceSetting(['a', 'b'], choice_labels, []), np.array(mnl_params)
    ), CDM(
        DiscreteChoiceSetting(['a', 'b'], choice_labels, []), np.array(cdm_params)
    )

    if nl_tree_fn is not None:
        models = models[0], models[1], NL(
            DiscreteChoiceSetting(['a', 'b'], choice_labels, []), nl_params
        )

    with open(f'models/{dataset}_models.pickle', 'wb') as f:
        pickle.dump(models, f)
    print(f'Saved inferred models to models/{dataset}_models.pickle')

    return models


def load_two_agent_models(dataset):
    try:
        with open(f'models/{dataset}_models.pickle', 'rb') as f:
            return pickle.load(f)
    except FileNotFoundError:
        print(f'Error: No file names models/{dataset}_models.pickle. Run without --saved-models.')
        exit(1)


def sets_to_tabular(sets, opt_sets, item_labels):
    """
    Convert solution sets into tabular format
    :param sets: A 2d array storing the solutions sets found given each pair of alternatives
    :param opt_sets: A 2d array storing the optimal sets given each pair of alternatives
    :param item_labels: an array of labels for the items
    :return: a tabular string with the input data in pretty form
    """
    newline = '\\\\\n'

    string = f'\\scalebox{{0.6}}{{\n\\begin{{tabular}}{{{"c" * (len(item_labels) + 1)}}}\n'

    string += '\\toprule\n' + '&' + ' & '.join(
        [f'${label}$' for label in item_labels]) + newline + '\\midrule\n'

    string += newline.join([f'${item_labels[i]}$' + ' &' + ' & '.join(
        ('$\\bm{\\{' + ','.join(item_labels[item] for item in Z) + '\\}}$'
         if set(Z) == set(opt_sets[i][j]) else
         '$\\{' + ','.join(item_labels[item] for item in Z) + '\\}$')
        if Z is not None else '' for j, Z in enumerate(row))
                            for i, row in enumerate(sets)]) + newline + '\\bottomrule\n'

    string += '\\end{tabular}}'

    return string


def run_sf_work_experiments(args):
    """
    Infer SF models and find optimal choice sets.
    :param args: command-line args
    """
    all_item_labels = ['d_1', 'd_2', 'd_{+}', 't', 'b', 'w']

    print('Running SFWork experiments...')

    latex = ''

    if args.saved_models:
        mnl, cdm, nl = load_two_agent_models('sfwork')
    else:
        mnl, cdm, nl = infer_two_agent_models(load_sf_work, 'sfwork', nl_tree_fn=get_sf_work_tree)

    for model in mnl, cdm, nl:

        for maximize in [False, True]:

            num_alts = len(model.dcs.U)

            greedy_Zs = [[None for _ in range(num_alts)] for _ in range(num_alts)]
            approx_Zs = [[None for _ in range(num_alts)] for _ in range(num_alts)]
            opt_Zs = [[None for _ in range(num_alts)] for _ in range(num_alts)]

            approx_is_opt = 0
            greedy_is_opt = 0

            approx_error = 0
            greedy_error = 0

            # Integer programming method, only for MNL
            ip_error = 0
            ip_is_opt = 0

            for x, y in itertools.combinations(range(num_alts), 2):
                model.set_C([x, y])

                optimize_choice_sets.greedy_agreement(model, maximize=maximize)
                greedy_Z = model.dcs.Z
                greedy_D = model.disagreement()

                optimize_choice_sets.opt_agreement(model, maximize=maximize)
                opt_Z = model.dcs.Z
                opt_D = model.disagreement()

                optimize_choice_sets.approx_agreement(model, 0.01, maximize=maximize)
                approx_Z = model.dcs.Z
                approx_D = model.disagreement()

                if model.short_name == 'mnl':
                    optimize_choice_sets.mnl_integer_program_agreement(model, maximize=maximize)
                    if set(opt_Z) == set(model.dcs.Z):
                        ip_is_opt += 1
                    ip_D = model.disagreement()
                    ip_error += abs(ip_D - opt_D)

                greedy_Zs[x][y] = greedy_Z
                opt_Zs[x][y] = opt_Z
                approx_Zs[x][y] = approx_Z

                if set(opt_Z) == set(approx_Z):
                    approx_is_opt += 1

                if set(opt_Z) == set(greedy_Z):
                    greedy_is_opt += 1

                approx_error += abs(approx_D - opt_D)
                greedy_error += abs(greedy_D - opt_D)

            num_choice_sets = num_alts * (num_alts - 1) // 2
            print('SFWork', model.short_name.upper(), 'Disagreement' if maximize else 'Agreement')
            print(f'\tGreedy is opt: {greedy_is_opt}/{num_choice_sets} (error: {greedy_error:.3f})\n'
                  f'\tApprox is opt: {approx_is_opt}/{num_choice_sets} (error: {approx_error:.3f})')
            if model.short_name == 'mnl':
                print(f'\tIP is opt: {ip_is_opt}/{num_choice_sets} (error: {ip_error:.3f})')

            greedy_tabular_string = sets_to_tabular(greedy_Zs, opt_Zs, all_item_labels)
            approx_tabular_string = sets_to_tabular(approx_Zs, opt_Zs, all_item_labels)

            caption = f'\\textsc{{SFWork}}, {model.short_name.upper()}, \\textsc{{{"Disagreement" if maximize else "Agreement"}}}'

            greedy_table_string = f'\\begin{{table}}[ht!]\n\\centering\n\\caption{{{caption}, greedy}}\n {greedy_tabular_string}\n\\end{{table}}'
            approx_table_string = f'\\begin{{table}}[ht!]\n\\centering\n\\caption{{{caption}, approx}}\n {approx_tabular_string}\n\\end{{table}}'

            latex += greedy_table_string + '\n\n' + approx_table_string + '\n\n'

    with open('results/sf_work_tables.tex', 'w') as f:
        f.write(latex)

    print('\nSaved SFWork solution sets to results/sf_work_tables.tex')


def allstate_promo_helper(args, cdm, epsilon, brute_force):
    """
    Helper to try promoting the most expensive item in parallel with approx and greedy
    :param args: the tuple (the choice set to promote from, the index of the item to promote)
    :param cdm: a CDM
    :param epsilon: the parameter for approx
    :param brute_force: if true, use brute force. Otherwise, use approx with given epsilon
    :return: a tuple of results
    """
    choice_set, target = args

    greedy_solution = None
    approx_solution = None

    greedy_success = False
    approx_success = False
    cdm.set_C(choice_set)
    target = cdm.dcs.C.index(target)
    num_sets_computed = 0
    num_possible_sets = 0

    if cdm.favorite(0) != target:
        optimize_choice_sets.greedy_promotion(cdm, target)
        if cdm.favorite(0) == target:
            greedy_success = True
            greedy_solution = cdm.dcs.Z

        if brute_force:
            optimize_choice_sets.opt_promotion(cdm, target)
        else:
            _, num_sets_computed, num_possible_sets = optimize_choice_sets.cdm_approx_promotion(cdm, target,
                                                                                                epsilon=epsilon)

        if cdm.favorite(0) == target:
            approx_success = True
            approx_solution = cdm.dcs.Z

    return choice_set, target, greedy_success, approx_success, greedy_solution, approx_solution, num_sets_computed, num_possible_sets


def agreement_helper(choice_set, model, epsilon):
    """
    Helper to optimize agreement in parallel with approx and greedy
    :param choice_set: the choice set to optimize agreement in
    :param model: a DiscreteChoiceModel
    :param epsilon: the parameter for approx
    :return: a tuple of results
    """
    model.set_C(choice_set)

    optimize_choice_sets.greedy_agreement(model, maximize=False)
    greedy_min_D = model.disagreement()
    greedy_min_Z = model.dcs.Z

    optimize_choice_sets.greedy_agreement(model, maximize=True)
    greedy_max_D = model.disagreement()
    greedy_max_Z = model.dcs.Z

    approx_min_Z, approx_min_D, approx_max_Z, approx_max_D, num_sets_computed = optimize_choice_sets.approx_agreement(
        model, epsilon=epsilon)

    # Integer programming approach: only run for MNL
    ip_min_D = ip_min_Z = ip_max_D = ip_max_Z = None
    if model.short_name == 'mnl':
        optimize_choice_sets.mnl_integer_program_agreement(model, maximize=False)
        ip_min_D = model.disagreement()
        ip_min_Z = model.dcs.Z

        optimize_choice_sets.mnl_integer_program_agreement(model, maximize=True)
        ip_max_D = model.disagreement()
        ip_max_Z = model.dcs.Z

    return choice_set, greedy_min_D, approx_min_D, ip_min_D, greedy_min_Z, approx_min_Z, ip_min_Z, \
           greedy_max_D, approx_max_D, ip_max_D, greedy_max_Z, approx_max_Z, ip_max_Z, num_sets_computed, \
           2 ** len(model.dcs.not_C)


def allstate_all_pairs_promo(num_threads, cdm, unique_items, item_indices, epsilon, brute_force=False):
    """
    For every two item choice set, try to promote the less-favored item using approx and greedy. Save results to a file.
    :param num_threads: number of threads to use
    :param cdm: a fitted CDM
    :param unique_items: the items in the allstate data
    :param item_indices: indices of each item in cdm.dcs.U
    :param epsilon: approximation parameter
    :param brute_force: if true, run brute force instead of approx
    """
    print(f'Running allstate all pairs promotion experiment (epsilon={epsilon})...')

    promo_attempts = [([item_indices[tuple(x)], item_indices[tuple(y)]], target)
                      for x, y in itertools.combinations(unique_items, 2)
                      for target in [item_indices[tuple(x)], item_indices[tuple(y)]]]

    pool = Pool(num_threads)
    helper_partial = partial(allstate_promo_helper, cdm=cdm, epsilon=epsilon, brute_force=brute_force)
    results = []

    for result in tqdm(pool.imap_unordered(helper_partial, promo_attempts), total=len(promo_attempts)):
        results.append(result)

    pool.close()
    pool.join()

    filename = f'results/allstate_all_pairs_promo_epsilon_{epsilon}_results.pickle' if not brute_force else \
        f'results/allstate_all_pairs_promo_brute_force_results.pickle'
    with open(filename, 'wb') as f:
        pickle.dump(results, f)

    print(f'Saved results to {filename}')


def all_pairs_agreement(num_threads, model, epsilon):
    """
    For every two item choice set, optimize agreement.
    :param num_threads: number of threads to use
    :param model: a fitted DiscreteChoiceModel with two agents
    :param epsilon: approximation parameter
    """
    all_pairs = list(itertools.combinations(range(len(model.dcs.U)), 2))

    pool = Pool(num_threads)
    helper_partial = partial(agreement_helper, model=model, epsilon=epsilon)
    results = []

    for result in tqdm(pool.imap_unordered(helper_partial, all_pairs), total=len(all_pairs)):
        results.append(result)

    pool.close()
    pool.join()

    return results


def sampled_choice_sets_agreement(choice_sets, num_threads, model, epsilon):
    """
    Optimize agreement for 500 randomly sampled choice sets.
    :param choice_sets: the choice sets to sample from
    :param num_threads: number of threads to use
    :param model: a fitted DiscreteChoiceModel with two agents
    :param epsilon: approximation parameter
    """

    filtered_choice_sets = [x for x in choice_sets if 1 < np.count_nonzero(x) <= 5]

    choice_set_indices = np.random.choice(range(len(filtered_choice_sets)), 500, replace=False)
    sampled_choice_sets = [tuple(np.nonzero(filtered_choice_sets[i])[0]) for i in choice_set_indices]

    pool = Pool(num_threads)
    helper_partial = partial(agreement_helper, model=model, epsilon=epsilon)
    results = []

    for result in tqdm(pool.imap_unordered(helper_partial, sampled_choice_sets), total=len(sampled_choice_sets)):
        results.append(result)

    pool.close()
    pool.join()

    return results


def run_allstate_experiments(args):
    """
    Run experiments on Allstate dataset in parallel
    :param args: command-line args
    """
    print('Loading Allstate data...')

    cdm, allstate_load_fn, data, unique_items, item_indices, choice_sets, choices, costs, item_counts = load_allstate()

    if args.saved_models:
        models = load_two_agent_models('allstate')
    else:
        models = infer_two_agent_models(allstate_load_fn, 'allstate')

    epsilons = {'mnl': 0.12, 'cdm': 80}

    # All pairs agreement optimization
    results = dict()
    for model in models:
        print(f'Running Allstate all pairs {model.short_name.upper()}')
        results[model.short_name, epsilons[model.short_name]] = all_pairs_agreement(args.threads, model,
                                                                                    epsilons[model.short_name])

    with open('results/allstate_all_pairs_agreement.pickle', 'wb') as f:
        pickle.dump(results, f)

    # All pairs promo, approx with varying epsilon
    for epsilon in (2 ** i for i in range(10, -3, -1)):
        allstate_all_pairs_promo(args.threads, cdm, unique_items, item_indices, epsilon)

    all_epsilon_data = dict()
    for file in glob.glob('results/allstate_all_pairs_promo_epsilon_*.pickle'):
        epsilon = float(file.split('_')[5])
        with open(file, 'rb') as f:
            all_epsilon_data[epsilon] = pickle.load(f)

    with open('results/allstate_all_pairs_promo_all_epsilon_results.pickle', 'wb') as f:
        pickle.dump(all_epsilon_data, f)
    for file in glob.glob('results/allstate_all_pairs_promo_epsilon_*.pickle'):
        os.remove(file)
    print('Merged results/allstate_all_pairs_promo_epsilon_*.pickle into '
          'results/allstate_all_pairs_promo_all_epsilon_results.pickle')

    # All pairs promo, brute force
    allstate_all_pairs_promo(args.threads, cdm, unique_items, item_indices, epsilon=0, brute_force=True)

    # Sampled choice sets agreement optimization
    epsilons = {'mnl': 2, 'cdm': 500}

    sampled_choice_set_results = dict()
    for model in models:
        print(f'Running Allstate sampled choice sets {model.short_name.upper()}')
        sampled_choice_set_results[model.short_name, epsilons[model.short_name]] = sampled_choice_sets_agreement(
            choice_sets, args.threads, model, epsilons[model.short_name])

    with open('results/allstate_sampled_choice_sets_agreement.pickle', 'wb') as f:
        pickle.dump(sampled_choice_set_results, f)


def run_yoochoose_experiments(args):
    """
    Run all experiments on the YOOCHOOSE dataset. For info on this dataset, see:
        Ben-Shimon, David, et al. RecSys Challenge 2015 and the YOOCHOOSE Dataset.
            In Proceedings of the 9th ACM Conference on Recommender Systems. 2015.
    """
    if args.saved_models:
        models = load_two_agent_models('yoochoose')
    else:
        models = infer_two_agent_models(load_yoochoose, 'yoochoose')
    epsilons = {'mnl': 0.4, 'cdm': 130}
    results = dict()
    for model in models:
        print(f'Running YOOCHOOSE all pairs {model.short_name.upper()}')
        results[model.short_name, epsilons[model.short_name]] = all_pairs_agreement(args.threads, model,
                                                                                    epsilons[model.short_name])

    with open('results/yoochoose_all_pairs_agreement.pickle', 'wb') as f:
        pickle.dump(results, f)

    # Sampled choice sets agreement optimization
    epsilons = {'mnl': 2, 'cdm': 500}
    choice_sets, _, _, _ = load_yoochoose()
    sampled_choice_set_results = dict()
    for model in models:
        print(f'Running Yoochoose sampled choice sets {model.short_name.upper()}')
        sampled_choice_set_results[model.short_name, epsilons[model.short_name]] = sampled_choice_sets_agreement(
            choice_sets, args.threads, model, epsilons[model.short_name])

    with open('results/yoochoose_sampled_choice_sets_agreement.pickle', 'wb') as f:
        pickle.dump(sampled_choice_set_results, f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run experiments on Allstate, SFWork, and YOOCHOOSE data.')
    parser.add_argument('--threads', default=cpu_count() // 2,
                        help='number of threads to use (default: # logical cores / 2)', type=int)
    parser.add_argument('--saved-models', action='store_true', help='use models saved in models/ for all-pairs agree')
    args = parser.parse_args()

    os.makedirs('results', exist_ok=True)
    os.makedirs('models', exist_ok=True)

    torch.manual_seed(0)
    np.random.seed(0)
    run_yoochoose_experiments(args)

    # reset seed so commenting experiments doesn't affect reproducibility
    torch.manual_seed(0)
    np.random.seed(0)
    run_sf_work_experiments(args)

    torch.manual_seed(0)
    np.random.seed(0)
    run_allstate_experiments(args)
