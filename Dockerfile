# Build Stage :  [ Step 1 - 3 ]
FROM python:3.10-slim AS build 

# Install Dependencies 

# hadolint ignore=DL3015 ignore=DL3008
RUN apt-get update && apt-get -y install libpq-dev gcc
# halolint ignore=DL3045
COPY ./requirements.txt requirements.txt
RUN pip3 install --no-cache-dir -r requirements.txt

# Runtime Stage [ Step 4 - 5 ]
FROM python:3.10-slim AS runtime
# Copying Dependencies from build stage  
COPY --from=build /usr/local/lib/python3.10/site-packages /usr/local/lib/python3.10/site-packages
COPY --from=build /usr/local/bin/streamlit /usr/local/bin/streamlit
ENV PYTHONPATH=/usr/lib/python3.10/site-packages

# Copying Source Code
WORKDIR /app
COPY . /app

# Expose Port
EXPOSE 8501

HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

ENTRYPOINT ["streamlit", "run", "streamlit.py"]
