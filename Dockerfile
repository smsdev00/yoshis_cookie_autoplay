# syntax=docker/dockerfile:1

FROM ubuntu:24.04 AS bsnes-builder

ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ca-certificates \
        g++ \
        git \
        libao-dev \
        libgtk-3-dev \
        libgtksourceview-3.0-dev \
        libopenal-dev \
        libsdl2-dev \
        make \
        pkg-config \
    && rm -rf /var/lib/apt/lists/*

ARG BSNES_REF=7d5aa1e656b9171524d01b1b22917197d8121cb4

RUN git init /src/bsnes \
    && git -C /src/bsnes remote add origin https://github.com/bsnes-emu/bsnes.git \
    && git -C /src/bsnes fetch --depth 1 origin "${BSNES_REF}" \
    && git -C /src/bsnes checkout --detach FETCH_HEAD \
    && make -j"$(nproc)" -C /src/bsnes/bsnes local=false


FROM ubuntu:24.04

ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        libao4 \
        libgomp1 \
        libgtk-3-0 \
        libgtksourceview-3.0-1 \
        libopenal1 \
        libsdl2-2.0-0 \
        libx11-6 \
        libxext6 \
        libxrender1 \
        mesa-utils \
        procps \
        tini \
        x11-utils \
        xvfb \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /opt/bsnes

COPY --from=bsnes-builder /src/bsnes/bsnes/out/bsnes /usr/local/bin/bsnes
COPY --from=bsnes-builder /src/bsnes/bsnes/Database ./Database
COPY --from=bsnes-builder /src/bsnes/shaders ./Shaders
COPY --from=bsnes-builder /src/bsnes/GPLv3.txt ./GPLv3.txt
COPY docker/headless-entrypoint.sh /usr/local/bin/headless-entrypoint

RUN chmod 0755 /usr/local/bin/bsnes /usr/local/bin/headless-entrypoint \
    && mkdir -p /rom /runtime /data/config /data/share

ENV DISPLAY=:99 \
    XDG_CONFIG_HOME=/data/config \
    XDG_DATA_HOME=/data/share \
    LIBGL_ALWAYS_SOFTWARE=1 \
    ALSOFT_DRIVERS=null \
    SDL_AUDIODRIVER=dummy

ENTRYPOINT ["/usr/bin/tini", "--", "/usr/local/bin/headless-entrypoint"]
CMD ["bsnes", "/rom/game.zip"]
