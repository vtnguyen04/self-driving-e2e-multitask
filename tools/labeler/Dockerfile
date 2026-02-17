FROM python:3.10-slim

WORKDIR /app

RUN pip install uv

COPY pyproject.toml uv.lock ./

RUN uv pip sync --system pyproject.toml

COPY . .

EXPOSE 8000

WORKDIR /app/tools/labeler

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
