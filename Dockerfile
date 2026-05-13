FROM python:3.9-slim

WORKDIR /app

# Gerekli sistem kütüphanelerini kur
RUN apt-get update && apt-get install -y gcc

# Requirements dosyasını kopyala
COPY ./backend/requirements.txt /app/requirements.txt

# Önce PyTorch'un CPU versiyonunu kur (Daha az yer kaplar ve sunucuyu yormaz)
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu
# Kalan tüm kütüphaneleri kur
RUN pip install --no-cache-dir -r /app/requirements.txt

# Backend dosyalarını kopyala
COPY ./backend /app/backend
COPY ./frontend /app/frontend

WORKDIR /app/backend

# Hugging Face Spaces varsayılan olarak 7860 portunu kullanır
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]
