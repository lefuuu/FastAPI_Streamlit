FROM python:3.10

WORKDIR /code

COPY ./requirements.txt /code/requirements.txt

RUN pip install --no-cache-dir -r /code/requirements.txt

COPY . . 
EXPOSE 8501

ENTRYPOINT ["streamlit", "run"]

CMD ["streamlit_app.py", "--server.port=8501"]