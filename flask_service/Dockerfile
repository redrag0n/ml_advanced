FROM python:3.8.5
COPY requirements.txt requirements.txt
RUN pip3 install --no-cache-dir -r requirements.txt
RUN pip3 install gdown
ENV FLASK_APP=run.py
RUN mkdir /flask_service
COPY . /flask_service
RUN mkdir models
WORKDIR /flask_service/models
RUN gdown -O heart.pkl https://drive.google.com/uc?id=1iRawkm79oFkHdWMe9UsqF7q-wUN7_ND0
RUN gdown -O heart_info.json https://drive.google.com/uc?id=1t-HKDbPHv9_nKaX_L69eHZaP5Zu_NQz8
RUN cat heart.pkl
WORKDIR /flask_service
EXPOSE 5000
RUN chmod +x run.sh
ENTRYPOINT ["./run.sh"]