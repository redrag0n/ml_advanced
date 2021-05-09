FROM python:3
ENV FLASK_APP=run.py
RUN mkdir /flask_service
COPY . /flask_service
WORKDIR /flask_service
RUN pip3 install --no-cache-dir -r requirements.txt
EXPOSE 5000
RUN chmod +x run.sh
ENTRYPOINT ["./run.sh"]