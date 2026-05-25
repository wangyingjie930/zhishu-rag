FROM pgvector/pgvector:pg16

ARG SCWS_VERSION=1.2.3
ARG ZHPARSER_VERSION=v2.3
ARG PG_SEARCH_VERSION=0.23.4

RUN set -eux; \
    apt-get update; \
    apt-get install -y --no-install-recommends \
      build-essential \
      ca-certificates \
      git \
      postgresql-server-dev-16 \
      wget; \
    . /etc/os-release; \
    arch="$(dpkg --print-architecture)"; \
    wget -q -O /tmp/pg_search.deb \
      "https://github.com/paradedb/paradedb/releases/download/v${PG_SEARCH_VERSION}/postgresql-16-pg-search_${PG_SEARCH_VERSION}-1PARADEDB-${VERSION_CODENAME}_${arch}.deb"; \
    apt-get install -y --no-install-recommends /tmp/pg_search.deb; \
    wget -q -O /tmp/scws.tar.bz2 "http://www.xunsearch.com/scws/down/scws-${SCWS_VERSION}.tar.bz2"; \
    mkdir -p /tmp/scws; \
    tar -xjf /tmp/scws.tar.bz2 -C /tmp/scws --strip-components=1; \
    cd /tmp/scws; \
    ./configure; \
    make -j"$(nproc)"; \
    make install; \
    ldconfig; \
    git clone --depth 1 --branch "${ZHPARSER_VERSION}" https://github.com/amutu/zhparser.git /tmp/zhparser; \
    cd /tmp/zhparser; \
    make -j"$(nproc)"; \
    make install; \
    rm -rf /tmp/pg_search.deb /tmp/scws /tmp/scws.tar.bz2 /tmp/zhparser; \
    apt-get purge -y --auto-remove build-essential git postgresql-server-dev-16 wget; \
    rm -rf /var/lib/apt/lists/*
