FROM python:3.8.8-slim

RUN mkdir -p /var/log/wuziqi \
  && chown nobody /var/log/wuziqi \
  && apt-get update \
  && apt-get install -y gcc \
  && rm -rf /var/lib/apt/lists/*

WORKDIR /work

COPY . .

COPY requirements.txt /work

RUN pip install --no-cache-dir -r requirements.txt

# RUN pip install --no-cache-dir -r requirements.txt --default-timeout=100 alphabet -i http://pypi.douban.com/simple/ --trusted-host pypi.douban.com

USER nobody

# run the command to start main
CMD ["python","./app/run_test.py"]
