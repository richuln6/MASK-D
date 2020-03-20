
FROM python:3.7
ADD requirements.txt /home/richul/Documents/EnhancingMailServerSecurity/requirements.txt
RUN pip install -r /home/richul/Documents/EnhancingMailServerSecurity/requirements.txt
COPY . /home/richul/Documents/EnhancingMailServerSecurity/
WORKDIR /home/richul/Documents/EnhancingMailServerSecurity/
EXPOSE 5000
CMD ["python3", "./encode.py", "LogFile/failed-login.log"]
CMD ["python3", "./authenticate.py", "LogFileDfs/encoded"]
