import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
from flask_wtf import FlaskForm
from wtforms import SubmitField, SelectField, IntegerField
from wtforms.validators import DataRequired, NumberRange

available_models = [
                    "dcgan",
                    "wganv4"
                    ]

class MainForm(FlaskForm):
    gen_num_field = IntegerField(
        label="Number of faces",
        validators=[
            DataRequired(),
            NumberRange(min=1, max=32)
            ]
        )

    models_list_field = SelectField("Select model",
                                    choices=[(m, m) for m in available_models])
    # resize obrazkow???????????
    submit_field = SubmitField("Submit")