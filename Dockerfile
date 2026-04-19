# syntax=docker/dockerfile:1.7

# ---------- builder stage ----------
FROM golang:1.26-bookworm AS builder

WORKDIR /src

COPY go.mod go.sum ./
RUN go mod download

COPY main.go ./
COPY internal ./internal

RUN CGO_ENABLED=1 go build -ldflags="-s -w" -o /out/spam-detector .


# ---------- runtime stage ----------
FROM debian:bookworm-slim AS runtime

ARG ORT_VERSION=1.22.0
ARG TARGETARCH

RUN apt-get update \
 && apt-get install -y --no-install-recommends ca-certificates curl \
 && case "$TARGETARCH" in \
      amd64) ORT_ARCH=x64 ;; \
      arm64) ORT_ARCH=aarch64 ;; \
      *)     echo "unsupported arch: $TARGETARCH" && exit 1 ;; \
    esac \
 && curl -fsSL -o /tmp/ort.tgz \
      "https://github.com/microsoft/onnxruntime/releases/download/v${ORT_VERSION}/onnxruntime-linux-${ORT_ARCH}-${ORT_VERSION}.tgz" \
 && tar -xzf /tmp/ort.tgz -C /tmp \
 && cp /tmp/onnxruntime-linux-*/lib/libonnxruntime.so* /usr/lib/ \
 && rm -rf /tmp/ort.tgz /tmp/onnxruntime-linux-* \
 && apt-get purge -y curl \
 && apt-get autoremove -y \
 && rm -rf /var/lib/apt/lists/*

COPY --from=builder /out/spam-detector /usr/local/bin/spam-detector

RUN useradd --system --uid 1000 --create-home app
USER app
WORKDIR /app

EXPOSE 8080

ENTRYPOINT ["/usr/local/bin/spam-detector"]
CMD ["--addr", ":8080", \
     "--model", "/app/artifacts/spam_detector.onnx", \
     "--vocab", "/app/artifacts/vocab.json", \
     "--ort-lib", "/usr/lib/libonnxruntime.so"]
