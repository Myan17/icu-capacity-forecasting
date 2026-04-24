# Lambda container image for the Forecast function.
# Uses the official AWS Lambda Python 3.11 base image (no size limit vs zip).
# Prophet + statsmodels + scipy are too large for a standard Lambda zip layer.

FROM public.ecr.aws/lambda/python:3.11

# Install OS-level dependencies required by Prophet / statsmodels
RUN dnf install -y gcc g++ && dnf clean all

# Copy and install Python dependencies first (layer-cache friendly)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the ML package (models, config, pipelines, features, evaluation)
COPY ../../ml /var/task/ml

# Copy the shared helpers (DynamoDB key builders + Pydantic models)
COPY ../../shared /var/task/shared

# Copy the Lambda handler
COPY handler.py /var/task/handler.py

CMD ["handler.lambda_handler"]
