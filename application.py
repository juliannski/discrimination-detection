import copy
import src.paths as paths
import datetime
import time
import logging
import dill
import shortuuid
import random
import psutil

from flask import Flask, render_template, request, session, url_for, redirect, jsonify
from flask_session import Session
from collections import defaultdict
from scripts.dev_robots import *
from src.quiz import PopQuiz
from src.ce_quiz import PopChoice, PopChoice2, PopChoiceShap, FeatureRankingForm
from src.questionnaire import PopQuest, PopInitQuest
from src.classifier import INTERCEPT_NAME
from src.utils import compute_model_info, linear_model_to_shap

application = Flask(__name__, template_folder=paths.template_dir, static_folder=paths.static_dir)
application.secret_key = "what_is_this"
application.config["SESSION_PERMANENT"] = False
application.config["SESSION_TYPE"] = "filesystem"

Session(application)

# local testing
LOCAL = False
if LOCAL:
    from pyngrok import ngrok
    public_url = ngrok.connect(5000)
    print("Public URL:", public_url)

MODE = 'test'
#MODE = 'debug'
PARAMS = {
    'seed': 3456, # for all: 345; weak:3456
    'base_pay': 3,
    'bonus': 3,
    'delay': 2 if not MODE == 'debug' else 0,
    ##
    'y_func': {INTERCEPT_NAME: -2,            # linear model parameters for the ground truth Y(x) = logit(y_func(x))
               'body': 1,
               'legs': 1,
               #'sign': 1,
               #'grip': 1,
               'antenna': 1,
               'head': 1},
    'yhat_func': {INTERCEPT_NAME: -0.48,      # linear model parameters for the predicted Y_hat(x) = logit(yhat_func(x))
                  'body': 0.24,
                  'legs': 0.24,
                  #'sign': 0.16,
                  #'grip': 0.16,
                  'antenna': 0.36,
                  'head': 0.18},
    'yhat_categorical': 'Prediction\n\n3*Antenna=Yes + 1*HeadShape=Round + 1*BodyShape=Round + 1*BaseType=Wheels >= 4',
    'color_features': ('head', 'body'),        # features that are used for the color of the robots
    'proxy_strength': ['strong_proxy'],
    'proxy_params': {'strong_proxy': ([0.95, 0.05], [0.05, 0.95]),    # probabilities for proxy values under different
                     'medium_proxy': ([0.95, 0.05], [0.45, 0.55]),    # values of the protected attribute
                     'weak_proxy': ([0.95, 0.05], [0.9, 0.1])},
    'proxy': 'antenna',
    'fairness_threshold': 0.2,                                        # threshold for the value of CE-Fairness
    'CE_method': ['proxy'],                                          # 'proxy', 'non_proxy', 'competing', 'multiple', 'random'
    'explanation': 'SHAP',                                            # 'CE', 'SHAP'
    ##
    'n_robots': 64,
    'n_probing_repeat': 2,                                           # number of robots repeated in anchoring stage 2
    'n_anchoring': 10,
    'n_auditing': 16,
    'max_attempts': 3,                                        # maximum number of attempts for the quiz
    'sample_CEs': True,                                       # whether to sample CEs or use the ones in the parameters
    ##
    'use_handle': True,                                       # whether to use the experiment handle or not
    'sample_sets': True,                                      # whether the sets in the parameters are to be used or not
    'probing': {'strong_proxy': ((118, 22), (2, 34),  # ids of robots in the Probing set
                                 (90, 122), (110, 14),
                                 (17, 113), (114, 18),
                                 (19, 115), (123, 27))},
    'anchoring': {'strong_proxy': {0: [9, 15, 30, 31],  # ids of robots in the Anchoring set
                                   1: [105, 111, 126, 127]}},
    'repeat': ((90, 122), (114, 18)),  # ids of robots to repeat in the Probing set

}


def pull_experiment_state(session):
    """
    pulls experimental parameters and results into a dictionary object that can be saved
    :param session: Flask session object
    :return:
    """
    state = {
        'pid': session['pid'],
        'params': session['params'],
        'results': session['results'],
        }
    return state


def render(template_file, **template_args):
    """
    This is a wrapper function to render Jinja templates.
    We use it to perform basic operations each time a new template is rendered
    (e.g., save results to disk, write results to a log)
    :param template_file:
    :param template_args:
    :return:
    """
    session = template_args.pop('args')

    # todo berk: print results on log..
    application.logger.debug('session', session)

    save = template_args.pop('save', False)
    if save:
        save_to_disk(session)

    # add params to template args
    template_args['params'] = session['params']

    return render_template(template_file, args=session, **template_args)


def save_to_disk(session, **kwargs):
    """
    :param session: Flask session object
    :return: name of the saved file
    """

    # parse file name
    path = kwargs.get('path', paths.results_dir)
    name = kwargs.get('name',  '{}_session'.format(session['pid']))
    suffix = kwargs.get('suffix', '.pickle')
    filename = (Path(path) / Path(name)).with_suffix(suffix)

    overwrite = kwargs.get('overwrite', True)
    if not overwrite and filename.exists():
        application.logger.warning('failed to save session file\nfile already exists: {}'.format(filename))
    else:
        # pull objects to save
        contents = pull_experiment_state(session)
        with open(filename, 'wb') as f:
            dill.dump(contents, file=f, protocol=dill.HIGHEST_PROTOCOL)
        application.logger.info("saved session content in file: {}".format(filename))

    return filename


@application.before_first_request
def before_first_request():

    # Set up the logger
    application.logger.setLevel(logging.INFO)
    application.logger.info('starting up Flask')

    # clear session
    session.clear()

    # set parameters
    session['params'] = copy.deepcopy(PARAMS)

    # experimental results
    session['results'] = {
        'time_start': str(datetime.datetime.now()),
        'experiment_handled': session['params']['use_handle'],
        'early_exit': False
    }


@application.route('/cpu_usage')
def get_cpu_usage():
    results = session.get('results', {})
    cpu_percent = psutil.cpu_percent(2)
    application.logger.info(f'[{results["pid"]}]: CPU usage: %s' % cpu_percent)
    return jsonify(cpu_percent=cpu_percent)


@application.route('/', methods=['GET', 'POST'])
def loading():
    """ page for loading the consent """
    before_first_request()

    pid = request.args.get('PROLIFIC_PID', shortuuid.uuid())
    study_id = request.args.get('STUDY_ID', shortuuid.uuid())
    session_id = request.args.get('SESSION_ID', shortuuid.uuid())

    params = session.pop('params')

    def random_select_from_params(d):
        sampled_items = {}
        for param_key, param_list in d.items():
            list_test = type(param_list) is list and len(param_list) > 0
            if list_test:
                list_el = param_list[random.sample(range(len(param_list)), 1)[0]]
                d[param_key] = list_el
                sampled_items[param_key] = list_el
        return d, sampled_items

    params, s = random_select_from_params(params)

    sampled_params = {**s}
    application.logger.info(f'[{pid}]: Sampling experiment parameters \n\n%s\n\n' % sampled_params)

    for key, val in zip(['pid', 'study_id', 'session_id'], [pid, study_id, session_id]):
        session[key] = val
        session['results'][key] = val

    session['params'] = params

    application.logger.info(f'Loading up the consent form for participant {session["pid"]}')

    return render('00_welcome.jinja', args=session, starting_screen=url_for('interim_loading'), save=False)

@application.route('/interim_loading', methods=['GET', 'POST'])
def interim_loading():
    next_page_nav = url_for('consent')
    next_page_load = url_for('consent')
    return render('01_interim.jinja', next_page_load=next_page_load, next_page_nav=next_page_nav,
                  args=session, save=True)

@application.route('/consent', methods=['GET', 'POST'])
def consent():
    """ landing page for the experiment """

    params, results = session.pop('params'), session.pop('results')
    application.logger.info(f'[{results["pid"]}]: Showing consent form for participant PID %s' % session['pid'])

    # Experiment handle randomly selects params['n_auditing'] robots from the "allowed" robots in the catalog
    if not params['sample_sets']:
        params['n_anchoring'] = sum([len(v) for v in params['anchoring'].values()])
    params['n_probing'] = 2 ** (len(params['y_func'].values())-1) + params['n_probing_repeat']

    if not results["experiment_handled"] and params['sample_CEs']:
            experiment_handle_l, new_params = setup_experiment(params)
            with open("results/experiments/experiment_handle_app_" + params['proxy_strength'] + "_CE_" + params['CE_method'] + ".pkl", 'wb') as f:
                dill.dump(experiment_handle_l, f)
            results['experiment_handled'] = True

            robot_colors = generate_color_combinations(ids=sorted(experiment_handle_l.robot_catalog.df['id']),
                                                       colored_features=params['color_features'],
                                                       colors=SEL_COLORS,
                                                       random_state=np.random.RandomState(params['seed']))
            params['robot_colors'] = robot_colors
    else:
        with open("results/experiments/experiment_handle_app_" + params['proxy_strength'] + "_CE_" + params['CE_method'] + ".pkl", 'rb') as f:
            base_handle = dill.load(f)
            experiment_handle_l = copy.deepcopy(base_handle)

            if "robot_colors" not in params.keys():
                robot_colors = generate_color_combinations(ids=sorted(experiment_handle_l.robot_catalog.df['id']),
                                                           colored_features=params['color_features'],
                                                           colors=SEL_COLORS,
                                                           random_state=np.random.RandomState(params['seed']))
                params['robot_colors'] = robot_colors

        experiment_handle_add = None
        if type(experiment_handle_l) is list:
            experiment_handle, experiment_handle_add = experiment_handle_l
        else:
            experiment_handle = experiment_handle_l

        params['model_info'] = compute_model_info(params['yhat_categorical'], "score_function")

        anchor, probe, probe_repeat, audit = experiment_handle.get_sets()
        print("\n\n\n\n")
        print(len(probe))
        print(len(probe_repeat))
        if experiment_handle_add:
            # adhoc column-name change due to some Flask errors (renaming parts of the columns with no rhyme-or-reason)
            experiment_handle_add.robot_catalog.df.columns = ['id', 'A', 'antenna', 'body', 'legs', 'sign', 'grip', 'head', 'Y'] + \
                                                              list(experiment_handle.robot_catalog.df.columns)[9:]

        # flattening the list of ID pairs to generate a list of Robot instances so that each consecutive 2 created a pair
        application.logger.info(f'[{results["pid"]}]: GETTING STAGE 2 ROBOTS: {probe}')
        anchoring_robots = {k: experiment_handle.robot_catalog.get_robot_set(robot_ids=anchor[k]) for k in anchor.keys()}
        probing_robots = experiment_handle.robot_catalog.get_robot_set(robot_ids=sum(probe, ()))
        application.logger.info(f'[{results["pid"]}]: GETTING REPEATED: {probe_repeat}')
        probing_repeated_robots = experiment_handle.robot_catalog.get_robot_set(robot_ids=sum(probe_repeat, ()))

        model_anchoring_set = experiment_handle.sample_model_anchoring(params['n_anchoring'])
        model_anchoring_robots = {k: experiment_handle.robot_catalog.get_robot_set(robot_ids=model_anchoring_set[k])
                                  for k in model_anchoring_set.keys()}

        # generating auditing robots
        application.logger.info(f'[{results["pid"]}]: GETTING THESE MANY AUDITING ROBOTS: {params["n_auditing"]}')
        experiment_handle.sample_auditing(params['n_auditing'])
        _, _, _, audit = experiment_handle.get_sets()
        print(f"Audit: {audit}")
        auditing_robots = experiment_handle.robot_catalog.get_robot_set(robot_ids=audit)
        if experiment_handle_add:
            auditing_robots += experiment_handle_add.robot_catalog.get_robots(robot_ids=audit)
            params['n_auditing'] *= 2

        results['selection_df'] = experiment_handle.robot_catalog.df
        results['robot_catalog'] = experiment_handle.robot_catalog

        # Adding Robot objects/path to robot image use in the next stages of the experiment
        # Take care of JSON serialization...
        application.logger.info(f'[{results["pid"]}]: Adding Robots to session')
        results['pid'] = session['pid']

        # Get the indices that would sort the robots in group 0 by outcome
        first_class_indices = sorted(range(len(anchoring_robots[0])),
                                     key=lambda i: anchoring_robots[0][i].outcome)
        # Sort robots in all groups by outcome
        results['anchoring'] = {
            k: [anchoring_robots[k][i].__dict__ for i in first_class_indices] for k in anchoring_robots.keys()
        }
        params['anchoring'] = results['anchoring']

        results['model_anchoring'] = {k: [robot.__dict__ for robot in model_anchoring_robots[k]]
                                        for k in model_anchoring_robots.keys()}
        params['model_anchoring'] = results['model_anchoring']
        results['probing'] = defaultdict(list)
        results['probing']['robots'] = [robot.__dict__ for robot in probing_robots] + \
                                                [robot.__dict__ for robot in probing_repeated_robots]
        results['quiz_attempts'] = 0
        results['auditing'] = defaultdict(list)
        results['auditing']['robots'] = [robot.__dict__ for robot in auditing_robots]

        results['audit_training'] = defaultdict(list)
        # noinspection PyTypeChecker
        results['audit_training']['robots'] = [('static/images/test_' + str(i) + '.png',
                                                'static/images/test_cf_' + str(i) + '.png') for i in range(3)] + \
                                              [('static/images/test_3.png',
                                                'static/images/test_cf_3_0.png',
                                                'static/images/test_cf_3_1.png'),
                                               ('static/images/test_4.png',
                                                'static/images/test_cf_4_0.png',
                                                'static/images/test_cf_4_1.png')
                                               ]
        results['audit_training']['fairness'] = ['UNFAIR', 'FAIR', 'UNFAIR', 'UNFAIR', 'FAIR']

    session['params'] = params
    session['results'] = results

    return render('01_consent.jinja', args=session)


@application.route('/pre_instructions', methods=['GET', 'POST'])
def pre_instructions():
    """ initial instructions """

    results = session.pop('results')
    results['completed'] = ['intro']
    session['results'] = results
    application.logger.info(f'[{results["pid"]}]: Showing initial info about the experiment')

    save_to_disk(session)

    if MODE == 'debug':
        next_page = url_for('anchoring', page=0)
    else:
        next_page = url_for('instructions', index=1)

    return render('02_instructions_0.jinja', next_page=next_page, args=session)


@application.route('/instructions/<int:index>', methods=['GET', 'POST'])
def instructions(index=1, message=None):
    """ screens for more detailed back-story instructions """

    results = session.pop('results')
    params = session.pop('params')

    session['results'] = results

    prev_page = url_for('instructions', index=index - 1, message='Correct!') if index != 1 else None

    if params['explanation'] == 'CE':
        form = PopChoice() if index < 6 else PopChoice2()
        image_id = form.q.id
        params['image_id'] = image_id
        next_page = url_for('instructions', index=index + 1) if index != 9 else url_for('auditing_train', index=0)
    else:
        form = PopChoiceShap()
        form = FeatureRankingForm()
        image_id = form.id
        params['image_id'] = image_id
        next_page = url_for('instructions', index=index + 1) if index != 8 else url_for('auditing_train', index=0)

    if message:
        message_neg, message_pos = None, message
    else:
        message_neg, message_pos = None, None

    session['params'] = params

    if index in [5, 6]:

        if form.is_submitted():

            application.logger.info(f'[{results["pid"]}]: Reading quiz answer instructions {index}')

            results = session.pop('results')

            new_data = form.q.data
            if 'quiz' not in results.keys():
                results['quiz_ce' + str(index)] = [new_data]
            else:
                results['quiz_ce' + str(index)].append(new_data)

            print(f"SUBMITTED: {new_data}")

            session['results'] = results
            save_to_disk(session)

            if form.validate():
                message_pos = "Correct! Clink 'Next' to continue your training."
            else:
                message_neg = 'Incorrect! ' + str(form.q.label)

            if params['explanation'] == 'SHAP':
                next_page = url_for('instructions', index=index + 2)

            results['completed'].append(f'quiz instructions {index}')

        else:
            application.logger.info(f'[{results["pid"]}]: Showing detailed instructions: page {index}')
            results['completed'].append(f'instructions {index}')

    else:
        application.logger.info(f'[{results["pid"]}]: Showing detailed instructions: page {index}')
        results['completed'].append(f'instructions {index}')

    return render('02_instructions_'+str(index)+'.jinja', args=session, next_page=next_page, prev_page=prev_page,
                  form=form, message_pos=message_pos, message_neg=message_neg, save=True)


@application.route('/instructions_final', methods=['GET', 'POST'])
def instructions_final():
    """ screens for more detailed back-story instructions """

    results = session.pop('results')
    params = session.pop('params')
    application.logger.info(f'[{results["pid"]}]: Showing final instructions')
    results['completed'].append(f'instructions final')
    bon = params['bonus']
    bon_corr = round(float(bon) / params['n_auditing'], 2)

    session['results'] = results
    session['params'] = params

    next_page = url_for('quiz')

    return render('04_instructions_final.jinja', args=session, next_page=next_page, bonus_per_correct=bon_corr,
                  save=True)


@application.route('/quiz', methods=['GET', 'POST'])
def quiz():
    """ quiz to test the instructions stage """
    form = PopQuiz()

    results = session.pop('results')
    results['completed'].append(f'quiz')
    session['results'] = results

    application.logger.info(f'[{results["pid"]}]: Showing the quiz')

    if MODE == 'debug':
        return redirect(url_for('auditing', index=0))

    if form.is_submitted():

        results = session.pop('results')

        new_data = [form.q1.data, form.q2.data, form.q3.data, form.q4.data]
        if 'quiz' not in results.keys():
            results['quiz'] = [new_data]
        else:
            results['quiz'].append(new_data)

        session['results'] = results

        if form.validate():
            return redirect(url_for('quiz_result', result='passed'))
        else:
            return redirect(url_for('quiz_result', result='failed'))

    return render('05_quiz.jinja', args=session, form=form, save=True)


@application.route('/quiz_result/<result>', methods=['GET', 'POST'])
def quiz_result(result):
    """ failed or passed the quiz """

    results, params = session.pop('results'), session.pop('params')
    results['quiz_attempts'] += 1
    remaining_attempts = params['max_attempts'] - results['quiz_attempts']
    application.logger.info(f'[{results["pid"]}]: Quiz {result}')

    if results['quiz_attempts'] >= params['max_attempts'] and result == 'failed':
        session['params'] = params
        session['results'] = results
        application.logger.info(f'[{results["pid"]}]: {params["max_attempts"]} quiz attempts reached. Ending experiment.')
        return redirect(url_for('debrief', status="rejected"))

    next_page = url_for('anchoring', page=0) if result == "passed" else url_for('instructions', index=1)
    results['completed'].append('quiz_result')

    session['results'] = results
    session['params'] = params

    return render(f'quiz_{result}.jinja', args=session, next_page=next_page, remaining_attempts=remaining_attempts, save=True)

@application.route('/interim_audit/<int:page>', methods=['GET', 'POST'])
def interim_audit(page=0):
    """ screen after the fairness training """

    results = session.pop('results')

    next_page = url_for('auditing', index=0) if page > 0 else url_for('interim_audit', page=1)

    results['completed'].append(f'instructions audit {page}')
    session['results'] = results

    return render(f'08_instructions_auditing_0{page}.jinja', args=session, next_page=next_page, save=True)


@application.route('/anchoring/<int:page>', methods=['GET', 'POST'])
def anchoring(page=0):
    """ anchoring stage of the experiment -- showing the anchoring set """

    results = session.pop('results')
    results['completed'].append(f'anchoring_{page}')
    application.logger.info(f'[{results["pid"]}]: Presenting the Anchoring set page {page}')

    next_page = url_for('probing', index=0) if page == 2 else url_for('anchoring', page=page+1)

    session['results'] = results

    return render(f'06_anchoring_0{page+1}.jinja', args=session, next_page=next_page, save=True)


@application.route('/probing/<int:index>', methods=['GET', 'POST'])
def probing(index):
    """ page for probing """
    results = session.pop('results')
    params = session.pop("params")

    p = results['probing']
    robot = p['robots'][index * 2]
    robot_pair = p['robots'][index * 2 + 1]
    proxy = params['proxy']
    total = params['n_probing']

    # we save the robot guess from previous round in
    region = request.form.get('region', None)
    reliability = request.form.get('reliability', None)
    determinism = request.form.get('determinism', None)
    if reliability:
        p['reliability'].append(reliability)
        p['first_robots'].append(robot)
        p['second_robots'].append(robot_pair)

        results['completed'].append(f'reliability probing {index}')

        params['probing_question'] = 'check_determinism'

    elif region:
        p['companies'].append(region)
        results['completed'].append(f'region probing {index}')
        if robot_pair['region_name'] != region:
            results['proxy_strength'] += 1

        params['probing_question'] = 'check_reliability'

    elif determinism:
        time_now = time.time()
        elapsed = time_now - results['time_prev']
        p['times'].append(elapsed)
        results['time_prev'] = time_now

        p['determinism'].append(determinism)
        results['completed'].append(f'determinism probing {index}')

        if index+1 >= params['n_probing']:
            session['results'] = results
            session['params'] = params
            return redirect(url_for('interim_audit', page=0))

        params['probing_question'] = 'check_region'
        index += 1
        robot = p['robots'][index * 2]
        robot_pair = p['robots'][index * 2 + 1]

    else: # index == 0
        results['time_prev'] = time.time()
        results['proxy_strength'] = 0
        params['probing_question'] = 'check_region'

    next_page = url_for('probing', index=index)

    session['results'] = results
    session['params'] = params

    application.logger.info(f'[{results["pid"]}]: Probing %s' % index)

    return render('07_probing.jinja', robot=robot, robot_pair=robot_pair,
                  next_page=next_page, index=index+1, proxy=proxy, total=total, args=session, save=True)


@application.route('/auditing_train/<int:index>', methods=['GET', 'POST'])
def auditing_train(index):
    """ page for determining prediction fairness """
    results = session.pop('results')
    params = session.pop("params")

    if index == 0:
        results['correct_train'] = 0
        results['incorrect_train'] = 0

    # if index > 0, then we save the robot guess from previous round in
    if index > 0:
        results['audit_training']['judgements'].append(request.form['decision'])
        real_f = results['audit_training']['fairness'][index-1]
        if real_f == request.form['decision']:
            results['correct_train'] += 1
        else:
            results['incorrect_train'] += 1

    if index < 4:
        robot_url = results['audit_training']['robots'][index][0]
        robot_cf_url = results['audit_training']['robots'][index][1]
        robot_cf_url2 = None
        if index in [3, 4]:
            robot_cf_url2 = results['audit_training']['robots'][index][2]

        results['completed'].append(f'auditing train {index}')
        next_page = url_for('auditing_train', index=index + 1)
    else:
        session['results'] = results
        session['params'] = params
        return redirect(url_for('instructions_final'))

    session['results'] = results
    session['params'] = params

    application.logger.info(f'[{results["pid"]}]: Fairness audit train %s' % index)

    return render('03_auditing_traing.jinja', robot_url=robot_url, robot_cf_url=robot_cf_url, next_page=next_page,
                  example_num=index, add_robot_url=robot_cf_url2, args=session, save=True)


@application.route('/auditing/<int:index>', methods=['GET', 'POST'])
def auditing(index):
    """ page for determining prediction fairness """
    results = session.pop('results')
    params = session.pop("params")

    total = params['n_auditing']

    if index == 0:
        results['correct_audit'] = 0
        results['bonus_correct_audit'] = 0
        results['time_prev'] = time.time()

    # if index > 0, then we save the robot guess from previous round in
    if index > 0:
        results['auditing']['judgements'].append(request.form['decision'])
        prev_robot = results['auditing']['robots'][index-1]
        f = prev_robot['ground_truth_fairness']
        of = prev_robot['observed_fairness']
        real_f = 'FAIR' if f == 1 else 'UNFAIR'
        obs_f = 'FAIR' if of == 1 else 'UNFAIR'
        if real_f == request.form['decision']:
            results['correct_audit'] += 1
        if obs_f == request.form['decision']:
            results['bonus_correct_audit'] += 1

        time_now = time.time()
        elapsed = time_now - results['time_prev']
        results['auditing']['times'].append(elapsed)
        results['time_prev'] = time_now

    if index < params['n_auditing']:
        application.logger.info(f'[{results["pid"]}]: Generating CF robot')
        # going over the auditing set again if too few robots
        robot = results['auditing']['robots'][index]
        robot_paths = []
        for i, feature_set in enumerate(robot['unchanged_cf_features']):
            print(robot['robot_url'])
            print(robot['counterfactuals'][i])
            print(robot['feature_values'])
            print(feature_set)
            _, robot_filename = print_robot(point=robot['feature_values'], colors=params['robot_colors'],
                                            alpha_features=feature_set, replace=True, add=i, id=robot['robot_id'])
            cf_robot_url = str(image_dir.relative_to(paths.repo_dir) / robot_filename)
            robot_paths.append(cf_robot_url)

        namechange = lambda x: 'BodyShape' if x == 'body' else 'HeadShape' if x == 'head' else 'BaseType' if x == 'legs' else x.capitalize()
        categorical_values = {'legs': ['Legs', 'Wheels'], 'head': ['Round', 'Square'],
                              'body': ['Square', 'Round'], 'antenna': ['No', 'Yes']}
        print(params['model_info'])
        print(robot['feature_values'])
        rfeats = {namechange(k): categorical_values[k][v] for k, v in robot['feature_values'].items() if k != "A"}
        print(rfeats)

        params["shap_explanation"] = linear_model_to_shap(params['model_info'], rfeats,
                                                          random_state=np.random.RandomState(seed=params['seed'] + index + 123))
        print(params['shap_explanation'])

        results['completed'].append(f'auditing {index+1}')
        application.logger.info(f'[{results["pid"]}]: Fairness audit test %s' % str(index+1))
        next_page = url_for('auditing', index=index + 1)
    else:
        session['results'] = results
        session['params'] = params
        return redirect(url_for('questionnaire'))

    session['results'] = results
    session['params'] = params

    return render('09_auditing.jinja', robot=robot, cf_robot_urls=robot_paths, next_page=next_page, index=index+1,
                  args=session, total=total, save=True)


@application.route('/questionnaire', methods=['GET', 'POST'])
def questionnaire():
    """ final questionnaire """

    results = session.pop('results')

    form = PopQuest()

    if form.is_submitted():

        answers = {
            form.q1.label: form.q1.data,
            form.q2.label: form.q2.data,
            form.q3.label: form.q3.data,
            form.q4.label: form.q4.data,
            form.q5.label: form.q5.data
        }
        results['questionnaire'] = answers
        session['results'] = results

        return redirect(url_for('debrief', status='complete'))

    application.logger.info(f'[{results["pid"]}]: Showing the questionnaire')
    results['completed'].append('questionnaire')

    session['results'] = results

    return render('10_questionnaire.jinja', args=session, form=form, save=True)

@application.route('/debrief/<string:status>', methods = ['GET', 'POST'])
def debrief(status="incomplete"):
    """ anchoring stage of the experiment -- showing the anchoring set """

    params = session.pop("params")
    results = session.pop('results')
    results['completed'].append('debrief_'+status)
    application.logger.info(f'[{results["pid"]}]: Showing the bonus and exiting the experiment')

    # Obtain the scaling for the bonus
    corr = results['bonus_correct_audit'] if 'bonus_correct_audit' in results.keys() else 0
    scale = round(float(corr) / params['n_auditing'], 2)
    bonus = round(params['bonus'] * scale, 1)

    if status == "complete":
        results['finished'] = True

    if status == "incomplete":
        results['bonus'] = 0
        results['early_exit'] = True
    else:
        results['bonus'] = bonus

    results['time_end'] = str(datetime.datetime.now())
    total_time_in_minutes = round((datetime.datetime.strptime(results['time_end'], '%Y-%m-%d %H:%M:%S.%f') -
                             datetime.datetime.strptime(results['time_start'], '%Y-%m-%d %H:%M:%S.%f')).total_seconds() / 60, 2)
    results['total_time'] = total_time_in_minutes

    session['results'] = results
    session['params'] = params
    session['earned_bonus'] = bonus

    return render('11_debrief.jinja', args=session, bonus=bonus, accuracy=scale, status=status, save=True)


application.config['APP'] = "application"

if __name__ == '__main__':
    if LOCAL:
        application.run(port=5000)
    else:
        application.run()