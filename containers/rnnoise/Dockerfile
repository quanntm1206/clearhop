ARG BASE_IMAGE=ubuntu@sha256:4fbb8e6a8395de5a7550b33509421a2bafbc0aab6c06ba2cef9ebffbc7092d90
FROM ${BASE_IMAGE}

ARG AUTOCONF_VERSION
ARG AUTOMAKE_VERSION
ARG BUILD_ESSENTIAL_VERSION
ARG CA_CERTIFICATES_VERSION
ARG CURL_VERSION
ARG LIBTOOL_VERSION
ARG PKG_CONFIG_VERSION
ARG APT_SNAPSHOT_URL

RUN set -eux; \
    sed -i \
      -e "s#http://archive.ubuntu.com/ubuntu/#${APT_SNAPSHOT_URL}#g" \
      -e "s#http://security.ubuntu.com/ubuntu/#${APT_SNAPSHOT_URL}#g" \
      /etc/apt/sources.list.d/ubuntu.sources; \
    test "$(grep -c "${APT_SNAPSHOT_URL}" /etc/apt/sources.list.d/ubuntu.sources)" = 2; \
    apt-get -o Acquire::https::Verify-Peer=false -o Acquire::Check-Valid-Until=false update; \
    for spec in \
      "autoconf=${AUTOCONF_VERSION}" \
      "automake=${AUTOMAKE_VERSION}" \
      "build-essential=${BUILD_ESSENTIAL_VERSION}" \
      "ca-certificates=${CA_CERTIFICATES_VERSION}" \
      "curl=${CURL_VERSION}" \
      "libtool=${LIBTOOL_VERSION}" \
      "pkg-config=${PKG_CONFIG_VERSION}"; do \
        package="${spec%%=*}"; expected="${spec#*=}"; \
        candidate="$(apt-cache policy "${package}" | awk '/Candidate:/{print $2}')"; \
        test "${candidate}" = "${expected}"; \
    done; \
    apt-get -o Acquire::https::Verify-Peer=false -o Acquire::Check-Valid-Until=false install -y --no-install-recommends \
      "autoconf=${AUTOCONF_VERSION}" \
      "automake=${AUTOMAKE_VERSION}" \
      "build-essential=${BUILD_ESSENTIAL_VERSION}" \
      "ca-certificates=${CA_CERTIFICATES_VERSION}" \
      "curl=${CURL_VERSION}" \
      "libtool=${LIBTOOL_VERSION}" \
      "pkg-config=${PKG_CONFIG_VERSION}"; \
    rm -rf /var/lib/apt/lists/*

ARG RNNOISE_COMMIT
ARG RNNOISE_ARCHIVE_URL
ARG RNNOISE_ARCHIVE_SHA256
ARG RNNOISE_MODEL_URL
ARG RNNOISE_MODEL_SHA256
RUN set -eux; \
    curl --fail --location --proto '=https' --tlsv1.2 "${RNNOISE_ARCHIVE_URL}" --output /tmp/rnnoise.tar.gz; \
    echo "${RNNOISE_ARCHIVE_SHA256}  /tmp/rnnoise.tar.gz" | sha256sum -c -; \
    mkdir /src; \
    tar -xzf /tmp/rnnoise.tar.gz --strip-components=1 -C /src; \
    rm /tmp/rnnoise.tar.gz; \
    test "$(cat /src/model_version)" = "${RNNOISE_MODEL_SHA256}"; \
    curl --fail --location --proto '=https' --tlsv1.2 "${RNNOISE_MODEL_URL}" \
      --output "/src/rnnoise_data-${RNNOISE_MODEL_SHA256}.tar.gz"; \
    echo "${RNNOISE_MODEL_SHA256}  /src/rnnoise_data-${RNNOISE_MODEL_SHA256}.tar.gz" | sha256sum -c -; \
    printf '%s\n' "${RNNOISE_COMMIT}" > /src/pinned_commit.txt

RUN set -eux; \
    cd /src; \
    ./autogen.sh; \
    ./configure --disable-shared; \
    make -j2; \
    mkdir /out; \
    cp examples/rnnoise_demo COPYING README pinned_commit.txt /out/; \
    sha256sum examples/rnnoise_demo src/rnnoise_data.c src/rnnoise_data.h > /out/SHA256SUMS; \
    dpkg-query -W -f='${Package}=${Version}\n' \
      autoconf automake build-essential ca-certificates curl libtool pkg-config \
      > /out/toolchain_packages.txt
