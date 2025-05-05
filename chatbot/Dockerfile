# Dockerfile for Chatbot Front-End
FROM python:3.10-slim
WORKDIR /app
# Install dependencies
COPY pyproject.toml poetry.lock* /app/
RUN pip install poetry && \
    poetry config virtualenvs.create false && \
    poetry install --no-dev
# Copy source code
COPY . /app
# Expose port
EXPOSE 8000
# Start the app
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
