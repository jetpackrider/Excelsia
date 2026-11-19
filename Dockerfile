# The Discord bot. Small on purpose: no PyTorch, just numpy.
FROM python:3.12-slim
ENV PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
WORKDIR /app
COPY bot/requirements.txt bot/requirements.txt
RUN pip install --no-cache-dir -r bot/requirements.txt
COPY excelsia/__init__.py excelsia/tokenizer.py excelsia/mood.py excelsia/lite.py excelsia/
COPY data/vocab.json data/vocab.json
COPY bot/ bot/
CMD ["python", "-m", "bot.bot"]
