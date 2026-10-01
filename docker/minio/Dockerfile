# MinIO now distributes the community server as source; the old image was
# unavailable during this deployment. Build the pinned official release.
# Compile the official RELEASE.2024-05-28T17-19-04Z source at its fixed commit.
# Source: https://github.com/minio/minio/commit/f79a4ef4d0dc3e6562cad0d1d1db674bc8c75531
FROM golang:1.22.12-alpine3.21@sha256:1699c10032ca2582ec89a24a1312d986a3f094aed3d5c1147b19880afe40e052 AS builder
WORKDIR /src
ADD https://codeload.github.com/minio/minio/tar.gz/f79a4ef4d0dc3e6562cad0d1d1db674bc8c75531 /tmp/minio-source.tar.gz
RUN tar -xzf /tmp/minio-source.tar.gz --strip-components=1 -C /src
RUN apk add --no-cache git
ENV CGO_ENABLED=0 GOTOOLCHAIN=local GOPROXY="https://proxy.golang.org|direct"
# Go verifies downloaded modules against the release's go.sum. Preserve successful
# downloads and compilation across a transient network failure.
RUN --mount=type=cache,target=/go/pkg/mod --mount=type=cache,target=/root/.cache/go-build \
    go build -p 2 -trimpath -ldflags="-s -w" -o /out/minio .

FROM alpine:3.21.3@sha256:a8560b36e8b8210634f77d9f7f9efd7ffa463e380b75e2e74aff4511df3ef88c
COPY --from=builder /out/minio /usr/local/bin/minio
COPY --from=builder /src/LICENSE /licenses/MinIO-LICENSE
EXPOSE 9000 9001
ENTRYPOINT ["/usr/local/bin/minio"]
