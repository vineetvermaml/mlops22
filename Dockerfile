#FROM ubuntu:latest
FROM python:3.8.1
COPY ./* /exp/
RUN pip install --upgrade pip
RUN pip3 install --no-cache-dir -r /exp/requirements.txt
WORKDIR /exp
# CMD ["python", "./plot_digits_classification.py"]
# CMD ["python","./question03.py"]
CMD ["python3","./plot_graphs.py","--clf_name","=","dtree","--random_state","=","42"]

EXPOSE 5000
