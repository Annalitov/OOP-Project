FROM python:3.10

RUN mkdir -p /Users/litann/Desktop/OOP/neironka
WORKDIR /Users/litann/Desktop/OOP/neironka

COPY . /Users/litann/Desktop/OOP/neironka
RUN pip3 --default-timeout=1000 install -r requirements.txt

CMD ["python3", "neironka.py"]