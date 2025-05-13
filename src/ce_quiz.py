from flask_wtf import FlaskForm as Form
from wtforms import SelectMultipleField, widgets, SelectField, FormField
from wtforms.validators import ValidationError


class CorrectAnswer(object):
    def __init__(self, checkbox_list):
        self.checkboxes = checkbox_list

    def __call__(self, form, field):
        message = 'Incorrect answer. Try again.'

        if field.data is None or set(list(field.data)) != set(self.checkboxes):
            raise ValidationError(message)


class MultiCheckboxField(SelectMultipleField):
    widget = widgets.ListWidget(html_tag='ul', prefix_label=False)
    option_widget = widgets.CheckboxInput()


class PopChoice(Form):
    class Meta:
        csrf = False
    q = MultiCheckboxField(
        label="Which parts need to be changed in the input picture to modify the algorithm's prediction according to the explanation above?",
        id="../static/images/cf_example.png",
        choices=[('legs', 'BaseType'),
                 ('torso', 'BodyShape'),
                 ('antenna', 'Antenna'),
                 ('head', 'HeadShape')],
        validators=[CorrectAnswer(['legs', 'antenna'])]
    )


class PopChoice2(Form):
    class Meta:
        csrf = False
    q = MultiCheckboxField(
        label="Select all changes that if applied to the picture on the left, modify the algorithm's prediction." + \
              " There may be more than one.",
        id="../static/images/cf_example2.png",
        choices=[('legs', 'Swap BaseType and HeadShape'),
                 ('torso', 'Swap BodyShape and HeadShape'),
                 ('antenna', 'Add Antenna, swap BaseType and HeadShape'),
                 ('antenna2', 'Add Antenna'),
                 ('head', 'Swap HeadShape')],
        validators=[CorrectAnswer(['legs', 'antenna2'])]
    )


class PopChoiceShap(Form):
    class Meta:
        csrf = False
    q = MultiCheckboxField(
        label="Which parts influence the algorithm toward predicting the robot is Defective? Select all that apply.",
        id="../static/images/shap_example.png",
        choices=[('antenna', 'Antenna'),
                 ('head', 'HeadShape'),
                 ('torso', 'BodyShape'),
                 ('legs', 'BaseType')],
        validators=[CorrectAnswer(['legs', 'antenna', 'torso'])]
    )


class CorrectRanking(object):
    def __init__(self, correct_rankings):
        # Dictionary mapping feature to correct rank
        self.correct_rankings = correct_rankings

    def __call__(self, form, field):
        message = 'Incorrect ranking. Try again.'

        # Check if all rankings match the expected values
        for feature, rank in self.correct_rankings.items():
            if getattr(form, feature).data != rank:
                raise ValidationError(message)


class RankingFormField(FormField):
    """Custom FormField that mimics MultiCheckboxField's label rendering"""

    def __init__(self, form_class, **kwargs):
        super(RankingFormField, self).__init__(form_class, **kwargs)
        self.image_id = kwargs.get('id', '')
        self.field_label = kwargs.get('label', '')

    def __call__(self, **kwargs):
        html = []

        html.append(super(RankingFormField, self).__call__(**kwargs))

        return ''.join(html)


class RankingForm(Form):
    class Meta:
        csrf = False

    antenna = SelectField(
        'Antenna',
        choices=[(str(i), str(i)) for i in range(1, 5)]
    )

    head = SelectField(
        'HeadShape',
        choices=[(str(i), str(i)) for i in range(1, 5)]
    )

    body = SelectField(
        'BodyShape',
        choices=[(str(i), str(i)) for i in range(1, 5)]
    )

    base = SelectField(
        'BaseType',
        choices=[(str(i), str(i)) for i in range(1, 5)]
    )


class FeatureRankingForm(Form):
    class Meta:
        csrf = False

    id = "../static/images/shap_example2.png"

    q = RankingFormField(
        RankingForm,
        label="Rank these features from least influential (1) to most influential (4) in the algorithm's prediction.",
    )

    def validate(self, extra_validators=None):
        if not super().validate(extra_validators=extra_validators):
            return False

        ranking_form = self.q

        ranks = [ranking_form.antenna.data, ranking_form.head.data,
                 ranking_form.body.data, ranking_form.base.data]
        if len(set(ranks)) != 4:
            return False

        # Check correct order
        correct_rankings = {
            'antenna': '4',
            'head': '1',
            'body': '3',
            'base': '2'
        }

        for feature, rank in correct_rankings.items():
            if getattr(ranking_form, feature).data != rank:
                return False

        return True