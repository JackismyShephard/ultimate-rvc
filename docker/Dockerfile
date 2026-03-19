FROM ubuntu:24.04

SHELL ["/bin/bash", "-exo", "pipefail", "-c"]
ENV DEBIAN_FRONTEND=noninteractive

WORKDIR /app
COPY . .

RUN --mount=type=tmpfs,target=/var/lib/apt/lists \
	--mount=type=tmpfs,target=/var/cache/apt/archives \
	apt-get update && \
	apt-get install -y --no-install-recommends sudo wget curl && \
	./urvc docker-bootstrap

EXPOSE 7860

CMD ["./urvc", "run", "--listen", "--listen-host", "0.0.0.0"]
