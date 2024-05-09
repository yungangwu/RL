## What is DDZ ?

基于深度学习的斗地主出牌预测逻辑服务

## Technical details

- **Nginx** is a web server, it takes care of the HTTP connections and also can serve static files directly and more efficiently.

- **uWSGI** is an application server, that's what runs your Python code and it talks with Nginx.

- **Your Python code** has the actual **Flask** web application, and is run by uWSGI.

## Prerequisites

Make sure you have installed all of the following prerequisites on your machine

- [WSL2](https://docs.microsoft.com/en-us/windows/wsl/install-win10#update-to-wsl-2)
- [Docker](https://docs.docker.com/engine/install/ubuntu/)
- Python3.8+

## Python 代码规范

https://docs.shiyou.kingsoft.com/docs/standards/code/python


## Learning

- https://github.com/tiangolo/uwsgi-nginx-flask-docker
- https://github.com/tiangolo/uwsgi-nginx-docker
- https://www.youtube.com/watch?v=dVEjSmKFUVI
- https://github.com/keras-team/keras/issues/2397
