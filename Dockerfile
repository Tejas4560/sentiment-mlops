# AWS Lambda base image — includes the Lambda Runtime Interface Client
# This is what makes Lambda know how to call our handler function
FROM public.ecr.aws/lambda/python:3.12

# Install app dependencies
COPY app/requirements.txt ./app/requirements.txt
RUN pip install --no-cache-dir -r app/requirements.txt

# Install model dependencies (CPU-only torch)
COPY model/requirements.txt ./model/requirements.txt
RUN pip install --no-cache-dir torch==2.4.1+cpu --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir transformers==4.44.2 numpy==1.26.4

# Copy source code and saved model artifact
COPY model/ ./model/
COPY app/ ./app/

# Point Lambda to the Mangum handler (not uvicorn)
CMD ["app.main.handler"]
