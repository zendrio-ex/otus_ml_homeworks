FROM python:3.8

COPY pyproject.toml poetry.lock load_model.py settings.json /app/

WORKDIR /app/

RUN pip install poetry
RUN poetry config virtualenvs.create false
RUN poetry install --without dev

COPY src ./src

RUN python load_model.py

EXPOSE 8000

CMD ["poetry", "run", "uvicorn", "src.main:app", "--host", "0.0.0.0"]