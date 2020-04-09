FROM python:3.7
ADD requirements.txt requirements.txt
RUN pip install -r requirements.txt
COPY . ./
WORKDIR ./
EXPOSE 5000
CMD ["python3", "./encode.py", "LogFile/failed-login.log"]
CMD ["python3", "./authenticate.py", "LogFileDfs/encoded"]

