FROM python:3.12

WORKDIR /app

COPY ../pyproject.toml ./
RUN pip install --no-cache-dir -e .

# If u prefer using requirements.txt:
# COPY requirements.txt ./
# RUN pip install --no-cache-dir -r requirements.txt

RUN apt-get update && apt-get install -y \
    tesseract-ocr \
    tesseract-ocr-slk \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

ENV TESSERACT_CMD=/usr/bin/tesseract

COPY . .

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
