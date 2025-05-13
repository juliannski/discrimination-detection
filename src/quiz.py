from flask_wtf import FlaskForm as Form
from wtforms import RadioField
from wtforms.validators import ValidationError


class CorrectAnswer(object):
    def __init__(self, answer):
        self.answer = answer

    def __call__(self, form, field):
        message = 'Incorrect answer.'
        if field.data != self.answer:
            raise ValidationError(message)


class PopQuiz(Form):
    class Meta:
        csrf = False
    q1 = RadioField(
        "What is your goal in the experiment?",
        choices=[('check_predictions', "Decide which predictions made by an algorithm are fair and which are not"),
                 ('best_robot', 'Decide which robot has the biggest antenna'),
                 ('mean_robot', 'Decide which robots from Company X are nice and which are mean')],
        validators=[CorrectAnswer('check_predictions')]
        )

    q2 = RadioField(
        "What does it mean that the algorithm's prediction is unfair?",
        choices=[('tech', 'The prediction is unfair if a person helped to reach it.'),
                 ('autonomous', 'The prediction is unfair if it was made autonomously by the algorithm.'),
                 ('brand_info', 'The prediction is unfair if it is based on the Company type.')],
        validators=[CorrectAnswer('brand_info')]
    )
    q3 = RadioField(
        "How will your bonus be determined?",
        choices=[('fairness', 'I am given bonus for each correct fairness decision.'),
                 ('experts', 'After being evaluated by a university committee.'),
                 ('no', 'There is no bonus')],
        validators=[CorrectAnswer('fairness')]
    )
    q4 = RadioField(
        "How does the algorithm explain its prediction?",
        choices=[('none', "It does not explain its prediction, it's an algorithm, not a person."),
                 #('image', "By showing how to change the parts in the robot's picture in order to modify the prediction."),
                 ('image', "By showing the influence of each part in favor of the Reliable or Defective prediction."),
                 ('other', 'By showing one or more other robots that were given the same prediction')],
        validators=[CorrectAnswer('image')]
    )

    q5 = RadioField(
        "If the NASA's algorithm says '> Defective' is it the robot really defective?",
        choices=[('yes', "Yes"),
                 ('no', 'No, it may still be reliable')],
        validators=[CorrectAnswer('no')]
    )

    q6 = RadioField(
        "How accurate is the NASA's algorithm?",
        choices=[('50', "50%"),
                 ('90', "90%"),
                 ('100', '100%')],
        validators=[CorrectAnswer('90')]
    )

    # q7 = RadioField(
    #     "Say that took a robot and changed its parts according to the picture in the algorithm's explanation. Would that make the robot more reliable?",
    #     choices=[('yes', "Yes"),
    #              ('no', "No, it would only change the algorithm's prediction to '> Reliable', but not necessarily the reliability.")],
    #     validators=[CorrectAnswer('no')]
    # )