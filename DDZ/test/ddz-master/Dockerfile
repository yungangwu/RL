FROM python:3.8.7-slim

RUN mkdir -p /var/log/ddz \
  && chown nobody /var/log/ddz \
  && apt-get update \
  && apt-get install -y gcc \
  && rm -rf /var/lib/apt/lists/*

# Expose the port uWSGI will listen on
EXPOSE 3000

VOLUME [ "/var/log/ddz" ]

ENV UWSGI_CHEAPER=1
ENV UWSGI_PROCESSES=10
ENV UWSGI_THREADS=2

WORKDIR /work

COPY requirements.txt /work

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

USER nobody

# run the command to start uWSGI
ENTRYPOINT ["uwsgi", "--ini", "uwsgi.ini"]
