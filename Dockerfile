FROM python:3.7-buster

WORKDIR /app

COPY . .

RUN pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cpu

RUN pip3 install -r requirements.txt

EXPOSE 8080

CMD ["python3", "main.py"]