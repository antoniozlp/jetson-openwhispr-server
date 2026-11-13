# The prebuilt jetson-containers image already has faster-whisper, CTranslate2
# (CUDA) and PyAV built for JetPack 6 / L4T R36.x. Only the web server is added.
FROM dustynv/faster-whisper:r36.4.0-cu128-24.04

# The image points pip at the jetson-ai-lab index, which doesn't carry these;
# they are pure-Python or have aarch64 wheels on PyPI.
RUN pip install --no-cache-dir --index-url https://pypi.org/simple \
    "fastapi>=0.115,<1" "uvicorn>=0.30,<1" "python-multipart>=0.0.9"

WORKDIR /app
COPY server/ /app/

EXPOSE 8178
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8178"]
