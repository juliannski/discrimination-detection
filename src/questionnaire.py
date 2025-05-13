from flask_wtf import FlaskForm as Form
from wtforms import TextAreaField, RadioField
from src.ce_quiz import MultiCheckboxField


class PopInitQuest(Form):
    class Meta:
        csrf = False

    q1 = RadioField(
        label="Do you think that having an antenna makes robots more likely to be reliable?",
        choices=[('causal', 'Yes'),
                 ('non-causal', 'No')],
    )


class PopQuest(Form):
    class Meta:
        csrf = False

    q1 = MultiCheckboxField(
        label="What do you think is the characteristic that differentiates Company S and X? Select all that apply.",
        choices=[('legs', 'BaseType'),
                 ('body', 'BodyShape'),
                 ('antenna', 'Antenna'),
                 ('head', 'HeadShape')],
    )

    q2 = MultiCheckboxField(
        label="Did you decide that predictions that can be flipped by manipulating the Antenna alone are unfair?",
        choices=[('Always', 'Yes, I did that each time I saw a prediction that could be flipped by changing the Antenna alone.'),
                 ('Sometimes', 'Yes, but I did that only sometimes because other times I thought the prediction was fair'),
                 ('No', 'No, I used different criteria to decide which predictions were fair and which weren’t')],
    )

    q3 = TextAreaField(
        label="Please, elaborate how exactly you decided on fairness when the prediction could be flipped by manipulating the Antenna alone.",
        id='text'
    )

    q4 = TextAreaField(
        label="Please, elaborate how exactly you decided on fairness in all other cases.",
        id='text'
    )

    q5 = RadioField(
        "How certain were you of your judgments?",
        id='likert',
        choices=[('uncertain', 'Uncertain'),
                 ('fairly uncertain', 'Fairly uncertain'),
                 ('neither', 'Neither'),
                 ('fairly certain', 'Fairly certain'),
                 ('certain', 'Certain')],
    )