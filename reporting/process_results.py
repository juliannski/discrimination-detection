import src.paths as paths
from pathlib import Path
from reporting.processing import process_participant_data, append_participant_data
from reporting.utils import make_report, open_file, merge_pdfs
import dill
import numpy as np

RESULTS_FOLDER = "25_18_03_results_strong_proxy_shap"
REPORT_FOLDER = "25_18_03_report_strong_proxy_shap"
PROCESS = True
EXP_PROCESS = True
BONUS = True
PART_REPORTS = True
EXP_REPORT = False

def process_and_save(files):
    early_droppouts = []
    failed_quizzes = []
    for num_data, f in enumerate(files):
        with open(f, 'rb') as r:
            print('Processing raw results for participant %s' % f.stem)
            raw = dill.load(r)
        results, pid = process_participant_data(results=raw['results'], params=raw['params'])
        if results:
            if not results['completion_check']:
                failed_quizzes.append(pid)
            output_file_name = paths.results_dir / RESULTS_FOLDER / f'participant_{results["participant_id"]}.results'
            with open(output_file_name, 'wb') as o:
                print('Saving processed results for participant %s' % results["participant_id"])
                dill.dump(results, o)
        else:
            early_droppouts.append(pid)
    print()
    print('EARLY DROPOUTS: ', early_droppouts)
    print('FAILED QUIZ: ', failed_quizzes)
    return list(Path(paths.results_dir / RESULTS_FOLDER).glob('participant_*.results'))


def create_reports(files):
    for f in files:
        with open(f, 'rb') as r:
            results = dill.load(r)
        if results['rejected']:
            continue
        else:
            print('Creating report for participant %s' % results["participant_id"])
            make_report(template_file=paths.report_templates / 'participant_report.Rmd',
                        report_data_file=f,
                        report_python_dir=paths.repo_dir,
                        build_dir=paths.reports_dir,
                        output_dir=paths.reports_dir / REPORT_FOLDER,
                        output_file=f'participant_{results["participant_id"]}_report.pdf',
                        clean=True,
                        quiet=False,
                        remove_build=False,
                        remove_output=False)


def combine_and_save(files):
    experiment_results = append_participant_data(files)
    output_file_name = paths.results_dir / RESULTS_FOLDER / 'experiment.results'
    with open(output_file_name, 'wb') as f:
        dill.dump(experiment_results, f)
    return output_file_name


def create_experiment_report(file, def_cond=False, cond_params=[]):
    output_file_name = file
    filename = 'experiment_report.pdf' if not def_cond else 'experiment_report_cond.pdf'
    make_report(template_file=paths.report_templates / 'experiment_report.Rmd',
                report_data_file=output_file_name,
                report_python_dir=paths.repo_dir,
                build_dir=paths.reports_dir,
                output_dir=paths.reports_dir / REPORT_FOLDER,
                output_file=filename,
                clean=True,
                quiet=False,
                remove_build=False,
                remove_output=False,
                define_conditions=def_cond,
                condition_params=cond_params)
    return filename


def create_bonuses(files, mean_rew=3.0, max_bonus=6.0):
    all_earnings, all_pids = [], []
    for f in files:
        with open(f, 'rb') as r:
            results = dill.load(r)
        if not(results['rejected']):
            earnings = results['round_df']['total_earnings'][0]
            all_earnings.append(earnings)
            all_pids.append(results['participant_id'])

    min_earnings = min(all_earnings)
    normalized_earnings = [earnings - min_earnings for earnings in all_earnings]
    mean_earnings = np.mean(normalized_earnings)
    pay_per_unit = mean_rew / mean_earnings
    bonuses = [min(round(pay_per_unit * earnings, 2), max_bonus) if max_bonus else round(pay_per_unit * earnings, 2)
               for earnings in normalized_earnings]

    # Iteratively adjust bonuses to ensure the mean
    while np.mean(bonuses) < mean_rew:
        for i, bonus in enumerate(bonuses):
            if bonus < max_bonus:
                bonuses[i] = min(bonuses[i] + 0.01, max_bonus)
            if np.mean(bonuses) >= mean_rew:
                break

    monies, print_to_file = [], []
    for i, (pid, earnings) in enumerate(zip(all_pids, normalized_earnings)):
        mony = round(bonuses[i], 2)
        monies.append(mony)
        input = pid + ',' + str(mony)
        if (mony) > 0.0:
            print_to_file.append(input)

    to_file = '\n'.join(print_to_file)
    with open('results/bonuses.txt', 'w') as f:
        f.write(to_file)
    print("NUMBER OF BONUSES: ", len(monies))
    print("BONUSES: ", monies)
    print('MEAN BONUS: ', np.mean(monies))
    print('STD BONUS: ', np.std(monies))


def get_bonuses(files):
    all_earnings, all_pids = [], []
    for f in files:
        with open(f, 'rb') as r:
            results = dill.load(r)
        if not (results['rejected']):
            earnings = results.get('bonus', 0)
            all_earnings.append(earnings)
            all_pids.append(results['participant_id'])

    monies, print_to_file = [], []
    for pid, earnings in zip(all_pids, all_earnings):
        mony = round(earnings, 2)
        monies.append(mony)
        input = pid + ',' + str(mony)
        if (mony) > 0.0:
            print_to_file.append(input)

    to_file = '\n'.join(print_to_file)
    with open('results/bonuses.txt', 'w') as f:
        f.write(to_file)
    print("NUMBER OF BONUSES: ", len(monies))
    print("BONUSES: ", monies)
    print('MEAN BONUS: ', np.mean(monies))
    print('STD BONUS: ', np.std(monies))


if PROCESS:
    # process files
    all_raw_results_files = Path(paths.results_dir / RESULTS_FOLDER).glob('*.pickle')
    all_processed_files = process_and_save(all_raw_results_files)
    exp_file = combine_and_save(all_processed_files)
elif EXP_PROCESS:
    all_processed_files = list(Path(paths.results_dir / RESULTS_FOLDER).glob('participant_*.results'))
    exp_file = combine_and_save(all_processed_files)
else:
    all_processed_files = list(Path(paths.results_dir / RESULTS_FOLDER).glob('participant_*.results'))
    exp_file = paths.results_dir / RESULTS_FOLDER / 'experiment.results'

if BONUS:
    get_bonuses(all_processed_files)

if PART_REPORTS:
    # create participant reports
    create_reports(all_processed_files)

if EXP_REPORT:
    # create experiment report
    #report_to_open = create_experiment_report(exp_file)
    report_to_open = create_experiment_report(exp_file, def_cond=True, cond_params=['named', 'target_accuracy'])

    #report_to_open = merge_pdfs([paths.reports_dir / REPORT_FOLDER / report_name,
    #                             paths.reports_dir / REPORT_FOLDER / report_name2],
    #                            merged_file=paths.reports_dir / REPORT_FOLDER / 'experiment_report_merged.pdf')

    open_file(paths.reports_dir / REPORT_FOLDER / report_to_open)
