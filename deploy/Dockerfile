# CPU (or CUDA, with a CUDA base image) container that runs AZChess as a Lichess bot.
#   docker build -f deploy/Dockerfile -t azchess-bot .
#   docker run -e LICHESS_BOT_TOKEN=... azchess-bot
FROM python:3.12-slim
RUN apt-get update && apt-get install -y --no-install-recommends g++ git && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY pyproject.toml setup.py README.md ./
COPY csrc csrc
COPY azchess azchess
RUN python -m venv .venv && .venv/bin/pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
    && AZCHESS_NATIVE=0 .venv/bin/pip install --no-cache-dir -e . && .venv/bin/python -c "import azchess._core"
COPY weights/azchess.pt weights/azchess.pt
COPY deploy deploy
RUN git clone --depth 1 https://github.com/lichess-bot-devs/lichess-bot.git deploy/lichess-bot/lichess-bot \
    && python -m venv deploy/lichess-bot/lichess-bot/.venv \
    && deploy/lichess-bot/lichess-bot/.venv/bin/pip install --no-cache-dir -r deploy/lichess-bot/lichess-bot/requirements.txt \
    && cp deploy/lichess-bot/config.yml.example deploy/lichess-bot/lichess-bot/config.yml
WORKDIR /app/deploy/lichess-bot/lichess-bot
CMD sed -i "s|^token: .*|token: \"${LICHESS_BOT_TOKEN}\"|" config.yml && .venv/bin/python lichess-bot.py
