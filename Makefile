IMAGE       ?= spam-detector
TAG         ?= $(shell git describe --tags --always --dirty 2>/dev/null || echo dev)
REGISTRY    ?=
PORT        ?= 8080
ARTIFACTS   ?= $(CURDIR)/training/artifacts
PLATFORMS   ?= linux/amd64,linux/arm64

FULL_IMAGE := $(if $(REGISTRY),$(REGISTRY)/,)$(IMAGE):$(TAG)

.PHONY: all help docker-build docker-buildx docker-run docker-push docker-clean fmt test

all: docker-build

## docker-build: 로컬 플랫폼용 Docker 이미지 빌드
docker-build:
	docker build -t $(FULL_IMAGE) -t $(IMAGE):latest .

## docker-buildx: 멀티 아키텍처(linux/amd64, linux/arm64) 이미지 빌드 및 push
##               사용 전 `docker buildx create --use` 필요
docker-buildx:
	@test -n "$(REGISTRY)" || { echo "ERROR: REGISTRY 변수를 지정하세요 (예: REGISTRY=ghcr.io/you)"; exit 1; }
	docker buildx build \
		--platform $(PLATFORMS) \
		-t $(FULL_IMAGE) \
		--push \
		.

## docker-run: 이미지를 실행한다 (학습 산출물을 /app/artifacts 에 마운트)
##             ARTIFACTS 환경변수로 경로 지정 가능 (기본: training/artifacts)
docker-run:
	@test -f $(ARTIFACTS)/spam_detector.onnx || { \
		echo "ERROR: $(ARTIFACTS)/spam_detector.onnx 가 없습니다."; \
		echo "       먼저 training/ 디렉토리에서 학습을 수행하세요."; \
		exit 1; \
	}
	docker run --rm -it \
		-p $(PORT):8080 \
		-v $(ARTIFACTS):/app/artifacts:ro \
		--name $(IMAGE) \
		$(IMAGE):latest

## docker-push: 이미지를 레지스트리에 푸시
docker-push:
	@test -n "$(REGISTRY)" || { echo "ERROR: REGISTRY 변수를 지정하세요"; exit 1; }
	docker push $(FULL_IMAGE)

## docker-clean: 로컬 이미지 삭제
docker-clean:
	-docker rmi $(FULL_IMAGE) $(IMAGE):latest

## fmt: Go 소스 포매팅
fmt:
	go fmt ./...

## test: Go 테스트 실행
test:
	go test ./...

## help: 사용 가능한 타깃 목록
help:
	@echo "사용 가능한 타깃:"
	@awk '/^## /{sub(/^## /,""); print "  " $$0}' $(MAKEFILE_LIST)
	@echo
	@echo "환경 변수:"
	@echo "  IMAGE     = $(IMAGE)"
	@echo "  TAG       = $(TAG)"
	@echo "  REGISTRY  = $(REGISTRY)"
	@echo "  PORT      = $(PORT)"
	@echo "  ARTIFACTS = $(ARTIFACTS)"
