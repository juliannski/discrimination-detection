"""
This file prints all possible robots
"""
import itertools

from scripts.dev_experiment import *
import pickle
import numpy as np
from PIL import Image, ImageFont, ImageDraw

from matplotlib import colors
from pathlib import Path
from src.paths import image_dir, static_dir

# Default non-mutable parameters
PROXY = DEFAULT_SETTINGS['proxy']
IMAGE_SHAPE = (1485, 2420)                                    # size of the robot image
DEFAULT_COLORS = {'body': 'custom_blue',
                  'legs': 'black',
                  'antenna': 'black',
                  'dot': 'black',
                  'head': 'custom_blue',
                  'grip': 'custom_blue',
                  'neck': 'black',
                  'sign': 'black'}                             # colors for robots' parts

# Mutable parameters
ALPHA = 128
ROBOT_COLORS = {'body': DEFAULT_COLORS['body'],
                'legs': DEFAULT_COLORS['legs'],
                'antenna': DEFAULT_COLORS['antenna'],
                'dot': DEFAULT_COLORS['dot'],
                'head': DEFAULT_COLORS['head'],
                'grip': DEFAULT_COLORS['grip'],
                'neck': DEFAULT_COLORS['neck'],
                'sign': DEFAULT_COLORS['sign']}                 # colors for the robot parts
POSITIONS = {'body': (0, 0),
             'legs': (0, 0),
             'antenna': (0, 0),
             'dot': (650, 1100),
             'head': (0, 0),
             'grip': (0, 0),
             'neck': (0, 0),
             'sign': ((510, 1080), (550, 1080))}                # pixel positions for pasting robot parts when printing

FEATURE_VALS = {'body': ['body_rect.png', 'body_round.png'],    # image to paste for each part based on feature values
                'legs': ['legs.png', 'rotors_black.png'],
                'antenna': ['', 'antenna_black.png'],
                'dot': ['', 'dot_black.png'],
                'head': ['head_round.png', 'head_tube.png'],
                'grip': ['arms.png', 'arms.png'],
                'neck': ['neck.png', 'neck_round.png'],
                'sign': ['', 'lightning_black.png']
                }

ROBOT_FEATURES = ['body', 'legs', 'antenna', 'head']  # names of the USED robot features
CONST_ROBOT_FEATURES = ['grip', 'neck']  # names of the robot features that are not used in the DAG_experiment
NUM_UNIQUE_ROBOTS = np.product([len(set(FEATURE_VALS[f])) for f in ROBOT_FEATURES]) * 2 # 2 since there are 2 companies

# Parameters for printing the anchoring set in one image
PROTECTED_ATT_NAMES = ['COMPANY X', 'COMPANY S']
IMAGE_VERTICAL_BREAK = 400                         # How much vertical space to make between one robot and another
IMAGE_HORIZONTAL_BREAK = 150                       # How much horizontal space to make between one robot and another
NAME_SPACE = 800                                   # How much space to use for the protected attribute name printing

parts_dir = image_dir / 'parts'


def is_dark(color_hex):
    # Convert hex to RGB
    r, g, b = colors.hex2color(color_hex)
    # Calculate brightness
    brightness = 0.299 * r + 0.587 * g + 0.114 * b
    # Consider dark if brightness is less than 0.5 (on a scale from 0 to 1)
    return brightness < 0.7


SEL_COLORS = ['custom_blue', 'custom_red']
COLORS = colors.cnames
COLORS['custom_blue'] = '#1ba1be'
COLORS['custom_red'] = '#e51400'


def is_coordinates(var):
    return isinstance(var, tuple) and all(isinstance(item, int) for item in var)


def print_element(image, element, val, color=None, alph=255):
    """
    Print a proper version of image named 'element' onto a canvas named 'image', possibly changing the color of the
    element to 'color'

    :param image: PIL.Image
    :param element: name of the element to stack onto the canvas
    :param val: integer indicating the version of the element to choose
    :param color: one of the matplotlib colors
    :return: PIL.Image with the element pasted onto the original canvas
    """
    # Create an image if it doesn't exist yet
    if image is None:
        image = Image.new(mode="RGBA", size=IMAGE_SHAPE, color=(255, 255, 255, 0))

    # Load the part
    part_name = FEATURE_VALS[element][int(val)]

    # Stack it on top of the image
    if part_name:
        part = Image.open(parts_dir / part_name)

        alpha = part.getchannel('A')
        # Make all opaque pixels into semi-opaque
        new_alpha = alpha.point(lambda i: alph if i > 0 else 0)
        # Put new alpha channel back into original image and save
        part.putalpha(new_alpha)

        position = POSITIONS[element] if is_coordinates(POSITIONS[element]) else POSITIONS[element][val]

        part = change_color(part, to_=color, from_=DEFAULT_COLORS[element])
        image.paste(part, position, part)
    return image


def print_robot(point, colors, alpha_features=None, location=parts_dir, replace=False, add='', id=0):
    """
    Draw a robot given by a vector of variables and values 'point' and a directory with robot parts corresponding to
    those variables

    :param point: { var: val } dict with values for different robot features
    :param colors: { id: { feature: color } } dict with colors for the robot features
    :param location: folder with the elements making up the robot's image
    :param replace: whether to generate new images
    :param alpha_features: name of the features to print unchanged and with the alpha channel
    :param add: custom text to add to the robot filename
    :return: PIL.Image with a robot drawn from parts
    """

    assert isinstance(point, dict), \
        "Correct the input, it should be a dict with keys corresponding to features in the DAG_experiment"

    location = Path(location)
    location.mkdir(exist_ok=True)

    if alpha_features:
        id_ = id
        filename = 'temp_robot' + str(add )+ '.png'
    else:
        id_ = point['id']
        filename = 'robot_%03d' % id_ + str(add) + '.png'
    filepath = location.parent / filename

    robot_colors = ROBOT_COLORS.copy()
    print(f"Robot id: {id_} with colors {colors[id_]}.")
    robot_colors.update(colors[id_])
    print(robot_colors)


    if not filepath.exists() or replace:
        img = None
        for part in ROBOT_FEATURES:
            if alpha_features and part in alpha_features:
                img = print_element(img, part, point[part], color=robot_colors[part], alph=ALPHA)
            elif alpha_features:
                img = print_element(img, part, 1-point[part], color=robot_colors[part])
            else:
                img = print_element(img, part, point[part], color=robot_colors[part])
        for part in CONST_ROBOT_FEATURES:
            if alpha_features:
                img = print_element(img, part, 1, color=robot_colors[part], alph=ALPHA)
            else:
                img = print_element(img, part, 1, color=robot_colors[part])
        if not(alpha_features): print(id_)
        img.save(filepath)

    return filepath, filename


def change_color(image, to_, from_='custom_blue'):
    """
    Change color from_ in image 'image' to color 'to_'.

    Supports only 4 custom colors picked by hand when creating robot parts.

    :param image: PIL.Image
    :param to_: one of the matplotlib colors
    :param from_: one of the custom colors to change
    :return: original image with the changed color
    """
    if to_ is None or to_ == from_:
        return image
    assert to_ in COLORS.keys() or to_ == 'black', f"Color {to_} is undefined. Try a different name."
    assert from_ in ['custom_blue', 'custom_red', 'black'], \
        f"Cannot change from color {from_}, only black, custom_red and custom_blue available"

    if isinstance(image, str):
        im = Image.open(image)
    else:
        im = image.copy()

    im = im.convert('RGBA')

    data = np.array(im)  # "data" is a height x width x 4 numpy array
    red, green, blue, alpha = data.T  # Temporarily unpack the bands for readability

    # Replace color "from_" with color "to_"... (leaves alpha values alone...)
    if from_ == 'custom_blue':
        from_areas = (red == 27) & (green == 161) & (blue == 226)
    elif from_ == 'custom_red':
        from_areas = (red == 229) & (green == 20) & (blue == 0)
    else:  # from_ == 'black':
        from_areas = (red == 0) & (green == 0) & (blue == 0) & (alpha == 255)

    if to_ == 'custom_blue':
        data[..., :-1][from_areas.T] = (27, 161, 226)
    elif to_ == 'custom_red':
        data[..., :-1][from_areas.T] = (229, 20, 0)
    else:
        rgb_to_ = tuple([c * 255 for c in colors.to_rgb(to_)])
        data[..., :-1][from_areas.T] = rgb_to_  # Transpose back needed

    im2 = Image.fromarray(data)
    return im2


def print_anchoring_set(robot_ids, replace=False):
    """
    Create an image with robots assigned to the anchoring set and passed as an argument.

    Each robot with the same value of the protected attribute is printed in one line with a text description of the
    attribute value (here, region of robot's production) at the beginning.

    Robots from another region are printed below.

    :param robot_ids: { protected_var : [ id ] } dictionary with robots ids where the key is the value of the protected
                      attribute shared by those robots
    :param replace: whether to generate a new image
    """
    path = image_dir / "anchoring.png"

    if not path.exists() or replace:
        # Selecting image size to evenly distribute robot images across rows and columns
        print(robot_ids)
        levels = len(list(robot_ids.keys()))
        num_robots = len(list(robot_ids.values())[0])
        image_height = IMAGE_SHAPE[1] * levels + IMAGE_VERTICAL_BREAK * (levels - 1)
        image_width = int(NAME_SPACE*1.5) + IMAGE_SHAPE[0] * num_robots + IMAGE_HORIZONTAL_BREAK * num_robots

        image = Image.new('RGBA', (image_width, image_height), (255, 255, 255, 0))
        draw = ImageDraw.Draw(image)

        # Proper font size selection to take up all the NAME_SPACE space
        len_names = [len(name) for name in PROTECTED_ATT_NAMES]
        max_len_index = len_names.index(max(len_names))
        max_len_name = PROTECTED_ATT_NAMES[max_len_index]
        font = ImageFont.truetype("Keyboard.ttf", 1)
        fontsize = 1
        while font.getsize(max_len_name)[0] < NAME_SPACE:
            # iterate until the text size is just larger than the criteria
            fontsize += 1
            font = ImageFont.truetype("Keyboard.ttf", fontsize)
        font = ImageFont.truetype("Keyboard.ttf", int(fontsize*1.5))

        for level, (name_id, robot_id_list) in enumerate(robot_ids.items()):
            name = PROTECTED_ATT_NAMES[name_id]
            mid_level = IMAGE_SHAPE[1] * level + IMAGE_VERTICAL_BREAK * level + int(float(IMAGE_SHAPE[1]) / 2)
            draw.text((0, mid_level), name, (0, 0, 0), font=font)
            for num, robot_num in enumerate(robot_id_list):
                robot_name = "%03d" % robot_num
                robot = Image.open(image_dir / f'robot_{robot_name}.png')
                image.paste(robot, (int(NAME_SPACE*1.5) + IMAGE_HORIZONTAL_BREAK * (num + 1) + IMAGE_SHAPE[0] * num,
                                    IMAGE_SHAPE[1] * level + IMAGE_VERTICAL_BREAK * level))
        image.show()
        image.save(path)

    else:
        image = Image.open(path)
        image.show()
        return


def print_robots(robot_catalog, alpha_features, color_features, subset=None, val=1, replace=False, random_state=None):
    """
    Create images of robots based on the points from a set where all points have value 1 for column 'subset'.

    :param robot_catalog: Experiment instance with sampled anchoring set stage 1
    :param alpha_features: names of the features to print with the alpha channel
    :param color_features: names of the features to print with the color changed
    :param subset: name of the relevant column in the dataframe, e.g. 'Anchoring'
    :param val: value of the parameter for column subset
    :param replace: whether to generate new images
    :return: list of filenames with the robots from the set (or all filenames)
    """
    if subset is not None:
        points = robot_catalog.df[robot_catalog.df[subset] == val]
    else:
        points = robot_catalog.df

    ids = sorted(points['id'])
    colors_dict = generate_color_combinations(ids, color_features, SEL_COLORS, random_state)

    for _, point in points.iterrows():
        print(point.to_dict())
        print_robot(point=point.to_dict(), replace=replace, alpha_features=alpha_features, colors=colors_dict)


def print_assets(experiment_handle, replace, anchoring=True):
    """
    Print all the robots and the anchoring set for stage 1 of anchoring.

    :param experiment_handle: Experiment instance with sampled anchoring set stage 1
    :param replace: whether to generate new images
    """
    print_robots(experiment_handle.robot_catalog, replace=replace, alpha_features=None)
    anchoring_ids = experiment_handle.get_sets()[0]
    print(experiment_handle.get_sets()[1])
    print(anchoring_ids)
    if anchoring:
        print_anchoring_set(anchoring_ids, replace=True)


def generate_color_combinations(ids, colored_features, colors, random_state):
    id_color_combinations = {}
    print("INPUT IDS: ", ids)
    print("INPUT FEATURES: ", colored_features)
    print("INPUT COLORS: ", colors)
    color_combs = list(itertools.combinations(colors, len(colored_features)))
    random_state.shuffle(color_combs)

    if len(color_combs) < len(ids):
        color_combs = list(itertools.product(colors, repeat=len(colored_features)))
        color_combs = [(a,b) for a,b in color_combs if a == b]
        i = 0
        new_color_combs = []
        while len(new_color_combs) < len(ids):
            new_color_combs += [color_combs[i]] * NUM_UNIQUE_ROBOTS
            i += 1
        color_combs = new_color_combs
    else:
        color_combs = color_combs[:len(ids)]

    for i, robot_id in enumerate(ids):
        robot_color_comb = color_combs[i]
        colors_features = {f: robot_color_comb[j] for j, f in enumerate(colored_features)}
        id_color_combinations[robot_id] = colors_features
    return id_color_combinations


if __name__ == "__main__":
    yf = {INTERCEPT_NAME: -3,  # linear model parameters for the ground truth Y(x) = logit(y_func(x))
               'body': 1,
               'legs': 1,
               # 'sign': 1,
               # 'grip': 1,
               'antenna': 1,
               'head': 1}
    yhat = {INTERCEPT_NAME: -0.48,  # linear model parameters for the predicted Y_hat(x) = logit(yhat_func(x))
                  'body': 0.16,
                  'legs': 0.16,
                  # 'sign': 0.16,
                  # 'grip': 0.16,
                  'antenna': 0.36,
                  'head': 0.16}
    proxy = "antenna"

    seed = 145425
    rs = np.random.RandomState(seed)

    # Define the predictor
    model = LinearClassificationModel.from_dict(**yhat)

    # Variable definition and DAG definition
    feature_nodes = [k for k in yhat.keys() if k not in (INTERCEPT_NAME, proxy)]
    A = Variable(name='A', values=[0, 1], probabilities=[0.5, 0.5])
    B = Variable(name=proxy, values=[0, 1],
                 probabilities=dict({'A_0': [0.95, 0.05],
                                     'A_1': [0.05, 0.95]}))
    X = {k: Variable(name=k, values=[0, 1], probabilities=[0.5, 0.5]) for k in feature_nodes}
    Y = FinalBooleanVariable(name='Y', vars_ordering=[B.name] + [x.name for x in X.values()],
                             model_parameters=yf)
    dag = FairnessDAG(features=X, outcome=Y, protected=A, proxy=B)
    dag.add_edges_from_coefficients(coefficients=yf)

    # Create the robot catalog
    # Generate a giant data frame of attributes, robots, likelihoods, true labels
    catalog = RobotCatalog(dag=dag, model=model, random_state=np.random.RandomState(0))
    catalog.generate(n=dag.n_distinct, with_replacement=False)
    catalog.adapt(to_size=64)

    print("Printing all the robots...")
    print_robots(catalog, alpha_features=None, color_features=['head', 'body'], subset=None, val=1, replace=True, random_state=rs)
