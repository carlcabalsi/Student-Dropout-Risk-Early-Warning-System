# Deployment Guide

## Student Dropout Risk Early Warning System

This application deploys the trained Logistic Regression model through Dash.

## Repository Requirements

The following files must exist:

```text
src/app.py
models/dropout_model.pkl
requirements.txt
```

## Install Dependencies

From the repository root, run:

```bash
python -m pip install -r requirements.txt
```

## Run the Application Locally

From the repository root, run:

```bash
python src/app.py
```

## Open the Application

After starting the application, open the local address shown in Terminal.

The default Dash address is normally:

```text
http://127.0.0.1:8050
```

## Stop the Application

In Terminal, press:

```text
Control + C
```

## Model Inputs

The application collects academic, lifestyle, financial, mental-health, and support-related student indicators.

It automatically calculates:

- Mental Health Index
- Academic Engagement Index

The final input contains the same 25 features used to train the Logistic Regression model.

## Responsible Use

This application is a decision-support prototype. Predictions should be reviewed by qualified staff and should not be used as the sole basis for decisions affecting individual students.
