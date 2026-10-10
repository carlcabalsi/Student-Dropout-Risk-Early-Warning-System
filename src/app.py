from pathlib import Path

import dash
from dash import Dash, Input, Output, State, dcc, html
import joblib
import numpy as np


# ---------------------------------------------------------
# MODEL LOADING
# ---------------------------------------------------------

# app.py is inside src/, so parent.parent points to the
# repository root.
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "dropout_model.pkl"

model = joblib.load(MODEL_PATH)


# ---------------------------------------------------------
# ENCODING MAPPINGS USED DURING MODEL TRAINING
# ---------------------------------------------------------

GENDER_MAPPING = {
    "Female": 0,
    "Male": 1,
    "Other": 2,
}

DEPARTMENT_MAPPING = {
    "Arts": 0,
    "Business": 1,
    "Engineering": 2,
    "Law": 3,
    "Medicine": 4,
    "Science": 5,
}

RESIDENCE_MAPPING = {
    "Day Scholar": 0,
    "Hostel": 1,
    "PG/Rented": 2,
}

PART_TIME_JOB_MAPPING = {
    "No": 0,
    "Yes": 1,
}

INCOME_MAPPING = {
    "High": 0,
    "Low": 1,
    "Lower-Middle": 2,
    "Middle": 3,
    "Upper-Middle": 4,
}

COUNSELING_MAPPING = {
    "No": 0,
    "Yes": 1,
}

BURNOUT_MAPPING = {
    "High": 0,
    "Low": 1,
    "Medium": 2,
}


# ---------------------------------------------------------
# DASH APPLICATION
# ---------------------------------------------------------

app = Dash(__name__)
server = app.server

app.title = "Student Dropout Risk Predictor"


def numeric_input(label, input_id, minimum, maximum, step, value):
    return html.Div(
        [
            html.Label(
                label,
                style={
                    "display": "block",
                    "fontWeight": "600",
                    "marginBottom": "6px",
                },
            ),
            dcc.Input(
                id=input_id,
                type="number",
                min=minimum,
                max=maximum,
                step=step,
                value=value,
                required=True,
                style={
                    "width": "100%",
                    "padding": "10px",
                    "boxSizing": "border-box",
                    "border": "1px solid #cbd5e1",
                    "borderRadius": "6px",
                },
            ),
        ],
        style={"marginBottom": "16px"},
    )


def dropdown_input(label, input_id, options, default_value):
    return html.Div(
        [
            html.Label(
                label,
                style={
                    "display": "block",
                    "fontWeight": "600",
                    "marginBottom": "6px",
                },
            ),
            dcc.Dropdown(
                id=input_id,
                options=[
                    {"label": option, "value": option}
                    for option in options
                ],
                value=default_value,
                clearable=False,
            ),
        ],
        style={"marginBottom": "16px"},
    )


app.layout = html.Div(
    [
        html.Div(
            [
                html.H1(
                    "Student Dropout Risk Early Warning System",
                    style={
                        "color": "#12364B",
                        "marginBottom": "8px",
                    },
                ),
                html.P(
                    (
                        "Enter the student information below to estimate "
                        "dropout risk using the trained Logistic Regression model."
                    ),
                    style={
                        "color": "#475569",
                        "fontSize": "17px",
                    },
                ),
                html.Div(
                    (
                        "This application is a decision-support prototype. "
                        "Predictions should be reviewed by qualified staff "
                        "and should not be used as the sole basis for decisions."
                    ),
                    style={
                        "backgroundColor": "#E6F5F3",
                        "borderLeft": "5px solid #0B7A75",
                        "padding": "14px",
                        "marginTop": "18px",
                        "marginBottom": "24px",
                        "color": "#12364B",
                    },
                ),
            ]
        ),

        html.Div(
            [
                # LEFT COLUMN
                html.Div(
                    [
                        html.H2(
                            "Student and Academic Information",
                            style={"color": "#0B5A7A"},
                        ),

                        numeric_input(
                            "Age",
                            "age",
                            17,
                            25,
                            1,
                            20,
                        ),

                        dropdown_input(
                            "Gender",
                            "gender",
                            list(GENDER_MAPPING.keys()),
                            "Female",
                        ),

                        numeric_input(
                            "Year of Study",
                            "year-of-study",
                            1,
                            4,
                            1,
                            2,
                        ),

                        dropdown_input(
                            "Department",
                            "department",
                            list(DEPARTMENT_MAPPING.keys()),
                            "Engineering",
                        ),

                        dropdown_input(
                            "Residence Type",
                            "residence-type",
                            list(RESIDENCE_MAPPING.keys()),
                            "Day Scholar",
                        ),

                        numeric_input(
                            "Attendance Percentage",
                            "attendance",
                            30.0,
                            100.0,
                            0.1,
                            75.0,
                        ),

                        numeric_input(
                            "Study Hours per Day",
                            "study-hours",
                            0.0,
                            8.1,
                            0.1,
                            3.0,
                        ),

                        numeric_input(
                            "Previous GPA",
                            "previous-gpa",
                            2.42,
                            10.0,
                            0.01,
                            7.0,
                        ),

                        numeric_input(
                            "Number of Backlogs",
                            "backlogs",
                            0,
                            6,
                            1,
                            0,
                        ),

                        numeric_input(
                            "Sleep Hours",
                            "sleep-hours",
                            2.0,
                            10.0,
                            0.1,
                            7.0,
                        ),

                        numeric_input(
                            "Screen Time Hours",
                            "screen-time",
                            0.5,
                            12.6,
                            0.1,
                            5.0,
                        ),

                        numeric_input(
                            "Exercise Frequency per Week",
                            "exercise-frequency",
                            0,
                            7,
                            1,
                            3,
                        ),

                        numeric_input(
                            "Social Activity Score",
                            "social-activity",
                            0.0,
                            10.0,
                            0.1,
                            5.0,
                        ),
                    ],
                    style={
                        "backgroundColor": "white",
                        "padding": "24px",
                        "borderRadius": "10px",
                        "boxShadow": "0 2px 8px rgba(15, 23, 42, 0.08)",
                    },
                ),

                # RIGHT COLUMN
                html.Div(
                    [
                        html.H2(
                            "Well-being and Support Information",
                            style={"color": "#0B7A75"},
                        ),

                        dropdown_input(
                            "Part-Time Job",
                            "part-time-job",
                            list(PART_TIME_JOB_MAPPING.keys()),
                            "No",
                        ),

                        dropdown_input(
                            "Family Income Bracket",
                            "income-bracket",
                            list(INCOME_MAPPING.keys()),
                            "Middle",
                        ),

                        numeric_input(
                            "Financial Stress Score",
                            "financial-stress",
                            0.0,
                            10.0,
                            0.1,
                            5.0,
                        ),

                        numeric_input(
                            "Family Support Score",
                            "family-support",
                            0.0,
                            10.0,
                            0.1,
                            5.0,
                        ),

                        numeric_input(
                            "Stress Level",
                            "stress-level",
                            0.0,
                            10.0,
                            0.1,
                            5.0,
                        ),

                        numeric_input(
                            "Anxiety Score",
                            "anxiety-score",
                            0.0,
                            10.0,
                            0.1,
                            5.0,
                        ),

                        numeric_input(
                            "Motivation Score",
                            "motivation-score",
                            0.0,
                            10.0,
                            0.1,
                            5.0,
                        ),

                        numeric_input(
                            "Peer Pressure Score",
                            "peer-pressure",
                            0.0,
                            9.8,
                            0.1,
                            5.0,
                        ),

                        dropdown_input(
                            "Counseling Access",
                            "counseling-access",
                            list(COUNSELING_MAPPING.keys()),
                            "No",
                        ),

                        dropdown_input(
                            "Burnout Level",
                            "burnout-level",
                            list(BURNOUT_MAPPING.keys()),
                            "Medium",
                        ),
                    ],
                    style={
                        "backgroundColor": "white",
                        "padding": "24px",
                        "borderRadius": "10px",
                        "boxShadow": "0 2px 8px rgba(15, 23, 42, 0.08)",
                    },
                ),
            ],
            style={
                "display": "grid",
                "gridTemplateColumns": "repeat(auto-fit, minmax(320px, 1fr))",
                "gap": "24px",
            },
        ),

        html.Button(
            "Predict Dropout Risk",
            id="predict-button",
            n_clicks=0,
            style={
                "width": "100%",
                "marginTop": "28px",
                "padding": "16px",
                "backgroundColor": "#12364B",
                "color": "white",
                "border": "none",
                "borderRadius": "8px",
                "fontSize": "17px",
                "fontWeight": "700",
                "cursor": "pointer",
            },
        ),

        html.Div(
            id="prediction-result",
            style={"marginTop": "24px"},
        ),
    ],
    style={
        "maxWidth": "1100px",
        "margin": "0 auto",
        "padding": "32px 20px 60px 20px",
        "fontFamily": "Arial, sans-serif",
        "backgroundColor": "#F5F8FA",
    },
)


# ---------------------------------------------------------
# PREDICTION CALLBACK
# ---------------------------------------------------------

@app.callback(
    Output("prediction-result", "children"),
    Input("predict-button", "n_clicks"),
    State("age", "value"),
    State("gender", "value"),
    State("year-of-study", "value"),
    State("department", "value"),
    State("residence-type", "value"),
    State("attendance", "value"),
    State("study-hours", "value"),
    State("previous-gpa", "value"),
    State("backlogs", "value"),
    State("sleep-hours", "value"),
    State("screen-time", "value"),
    State("exercise-frequency", "value"),
    State("social-activity", "value"),
    State("part-time-job", "value"),
    State("income-bracket", "value"),
    State("financial-stress", "value"),
    State("family-support", "value"),
    State("stress-level", "value"),
    State("anxiety-score", "value"),
    State("motivation-score", "value"),
    State("peer-pressure", "value"),
    State("counseling-access", "value"),
    State("burnout-level", "value"),
    prevent_initial_call=True,
)
def predict_dropout_risk(
    n_clicks,
    age,
    gender,
    year_of_study,
    department,
    residence_type,
    attendance,
    study_hours,
    previous_gpa,
    backlogs,
    sleep_hours,
    screen_time,
    exercise_frequency,
    social_activity,
    part_time_job,
    income_bracket,
    financial_stress,
    family_support,
    stress_level,
    anxiety_score,
    motivation_score,
    peer_pressure,
    counseling_access,
    burnout_level,
):
    values = [
        age,
        gender,
        year_of_study,
        department,
        residence_type,
        attendance,
        study_hours,
        previous_gpa,
        backlogs,
        sleep_hours,
        screen_time,
        exercise_frequency,
        social_activity,
        part_time_job,
        income_bracket,
        financial_stress,
        family_support,
        stress_level,
        anxiety_score,
        motivation_score,
        peer_pressure,
        counseling_access,
        burnout_level,
    ]

    if any(value is None for value in values):
        return html.Div(
            "Please complete all fields before making a prediction.",
            style={
                "backgroundColor": "#FFF7E7",
                "color": "#92400E",
                "padding": "18px",
                "borderRadius": "8px",
                "fontWeight": "600",
            },
        )

    # Recreate the engineered features from the capstone notebook.
    mental_health_index = (
        float(stress_level)
        + float(anxiety_score)
        + float(financial_stress)
    ) / 3

    academic_engagement_index = (
        (float(attendance) / 100)
        + (float(study_hours) / 8.1)
        + (float(previous_gpa) / 10)
    ) / 3

    # Feature order must match X.columns from model training.
    model_input = np.array(
        [
            [
                float(age),
                GENDER_MAPPING[gender],
                float(year_of_study),
                DEPARTMENT_MAPPING[department],
                RESIDENCE_MAPPING[residence_type],
                float(attendance),
                float(study_hours),
                float(previous_gpa),
                float(backlogs),
                float(sleep_hours),
                float(screen_time),
                float(exercise_frequency),
                float(social_activity),
                PART_TIME_JOB_MAPPING[part_time_job],
                INCOME_MAPPING[income_bracket],
                float(financial_stress),
                float(family_support),
                float(stress_level),
                float(anxiety_score),
                float(motivation_score),
                float(peer_pressure),
                COUNSELING_MAPPING[counseling_access],
                BURNOUT_MAPPING[burnout_level],
                mental_health_index,
                academic_engagement_index,
            ]
        ],
        dtype=float,
    )

    prediction = int(model.predict(model_input)[0])

    probability = None

    if hasattr(model, "predict_proba"):
        probability = float(model.predict_proba(model_input)[0][1])

    if prediction == 1:
        title = "High Dropout Risk"
        background_color = "#FDECEC"
        border_color = "#E97862"
        title_color = "#B42318"
        recommendation = (
            "The model indicates that the student may benefit from "
            "additional human review and appropriate support."
        )
    else:
        title = "Low Dropout Risk"
        background_color = "#E8F6F3"
        border_color = "#0B7A75"
        title_color = "#067647"
        recommendation = (
            "The model indicates a lower predicted risk. Continue routine "
            "monitoring and make support resources available."
        )

    result_children = [
        html.H2(
            title,
            style={
                "color": title_color,
                "marginTop": "0",
            },
        ),
        html.P(
            recommendation,
            style={
                "fontSize": "16px",
                "color": "#1F2937",
            },
        ),
    ]

    if probability is not None:
        result_children.append(
            html.P(
                f"Estimated high-risk probability: {probability:.1%}",
                style={
                    "fontWeight": "700",
                    "color": "#12364B",
                },
            )
        )

    result_children.extend(
        [
            html.Hr(),
            html.P(
                (
                    "The result is produced by the capstone Logistic "
                    "Regression model. It is intended for decision support "
                    "and requires human review."
                ),
                style={
                    "fontSize": "13px",
                    "color": "#667085",
                    "marginBottom": "0",
                },
            ),
        ]
    )

    return html.Div(
        result_children,
        style={
            "backgroundColor": background_color,
            "borderLeft": f"6px solid {border_color}",
            "padding": "22px",
            "borderRadius": "8px",
        },
    )


# ---------------------------------------------------------
# LOCAL APPLICATION START
# ---------------------------------------------------------

if __name__ == "__main__":
    app.run(debug=True)
