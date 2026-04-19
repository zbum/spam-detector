# spam-detector

채팅/댓글 스팸 분류기. Google [Magika](https://github.com/google/magika)의 철학을 따라 **작은 Char-CNN 모델 + ONNX Runtime 으로 CPU만 써서 ms 단위 추론**을 목표로 한다.

- **학습**: Python (PyTorch) 에서 문자 단위 CNN 학습 → ONNX export
- **서빙**: Go + `onnxruntime_go` 로 REST API 제공
- **강점**: 우회 문자열(`ㅅ.ㅍ.ㅏ.ㅁ`), 이모지, 신조어, 오타, 다국어 혼용에 강한 문자 단위 모델

> 설계 원리와 ONNX Runtime 내부 동작이 궁금하다면 → **[TRAINING.md](TRAINING.md)** 참조

## 디렉토리 구조

```
spam-detector/
├── main.go                      # REST API 서버 엔트리
├── internal/
│   ├── detector/                # ONNX 세션 래퍼
│   ├── server/                  # HTTP 핸들러
│   └── tokenizer/               # Python 토크나이저의 Go 포팅
└── training/
    ├── config.yaml              # 하이퍼파라미터/경로
    ├── tokenizer.py             # 문자 단위 토크나이저
    ├── model.py                 # Char-CNN 정의
    ├── dataset.py               # CSV → Dataset
    ├── train.py                 # 학습 스크립트
    ├── export_onnx.py           # .pt → .onnx
    ├── verify_onnx.py           # PyTorch ↔ ONNX 일치 검증
    ├── make_sample_data.py      # 합성 데이터 생성
    ├── requirements.txt
    ├── data/                    # (git ignored) 학습 데이터
    └── artifacts/               # (git ignored) 학습 산출물
```

---

## 1. 학습 (Training)

> 학습 파이프라인의 **왜** — Char-CNN 선택 이유, 토크나이저 정규화 규칙, class weight, early stopping 등 설계 근거는 [TRAINING.md](TRAINING.md) 를 참조.

### 1.1 사전 준비

```bash
cd training
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

> **Python 버전**: 3.12 권장. 3.14 에서도 동작은 하지만 PyTorch 휠이 늦게 올라와 문제가 생길 수 있다.

### 1.2 학습 데이터 준비

학습에는 3개의 CSV 가 필요하다. 경로는 `training/config.yaml` 의 `data` 섹션에서 변경 가능.

| 파일 | 기본 경로 | 용도 |
|------|----------|------|
| 훈련 | `training/data/train.csv` | 모델 학습 |
| 검증 | `training/data/val.csv` | Early stopping / best checkpoint 선정 |
| 평가 | `training/data/test.csv` | 최종 성능 보고 |

**CSV 스키마**

```csv
text,label
안녕하세요 회의 몇 시였죠?,0
★초대박★ 무료이벤트 당첨! http://bit.ly/xxx,1
```

- `text`: 분류 대상 문자열 (길이 제한 없음, `max_length` 로 잘림)
- `label`: `0` = ham(정상), `1` = spam
- URL 과 전화번호는 토크나이저가 자동으로 `<url>`, `<phone>` 으로 정규화

**스모크 테스트용 합성 데이터 생성**

```bash
cd training
python make_sample_data.py
# → data/{train,val,test}.csv 생성
```

> 실제 운영용 학습은 이 합성 데이터가 아니라 **실제 채팅 로그**로 교체해야 한다. 템플릿만으로 학습된 모델은 실제 분포와 차이가 크다.

### 1.3 학습 설정 수정

`training/config.yaml` 주요 항목.

| 항목 | 기본값 | 설명 |
|------|--------|------|
| `tokenizer.max_length` | 200 | 입력 문자열 최대 길이 (채팅은 보통 짧음) |
| `tokenizer.min_freq` | 2 | 어휘에 포함되는 최소 출현 횟수 |
| `model.embedding_dim` | 64 | 문자 임베딩 차원 |
| `model.conv_channels` | 128 | Conv 출력 채널 |
| `model.kernel_sizes` | `[3, 4, 5]` | Multi-kernel CNN 필터 크기 |
| `model.dropout` | 0.3 | Dropout 비율 |
| `train.batch_size` | 128 | 배치 사이즈 |
| `train.epochs` | 15 | 최대 에폭 |
| `train.lr` | 0.001 | 학습률 |
| `train.early_stop_patience` | 3 | 검증 F1 개선 없을 때 허용 에폭 |
| `train.device` | `auto` | `auto` / `cpu` / `cuda` / `mps` |

### 1.4 학습 실행

```bash
cd training
source .venv/bin/activate
python train.py --config config.yaml
```

산출물은 `training/artifacts/` 에 저장된다.

| 파일 | 내용 |
|------|------|
| `vocab.json` | 토크나이저 어휘 (Go 측에서도 사용) |
| `model.pt` | 가장 좋은 검증 F1 의 PyTorch 체크포인트 |
| `metrics.json` | 검증/테스트 F1 및 분류 리포트 |

### 1.5 ONNX 변환

```bash
cd training
python export_onnx.py --config config.yaml
# → artifacts/spam_detector.onnx
```

### 1.6 일치성 검증

PyTorch 와 ONNX 출력이 일치하는지 + 샘플 문장의 추론 결과 확인.

```bash
python verify_onnx.py --config config.yaml
```

예시 출력:

```
[info] max |pytorch - onnx| = 7.15e-07
  [HAM ] p(spam)=0.3810  오늘 회의 몇 시였죠?
  [SPAM] p(spam)=0.9927  ★초대박★ 무료 이벤트 당첨! http://bit.ly/abcd
  [HAM ] p(spam)=0.1064  내일 점심 같이 먹어요
  [SPAM] p(spam)=0.9856  즉시대출 010-1234-5678 무심사 당일입금
  [SPAM] p(spam)=0.9917  ㅅ.ㅍ.ㅏ.ㅁ 아닙니다 진짜 고수익 부업
```

오차가 `1e-4` 를 넘으면 어서션이 터진다. 이때는 `export_onnx.py` 의 `opset_version` 이나 `dynamo` 옵션을 재검토한다.

---

## 2. Docker 빌드

> **Docker 전용 워크플로우**: `onnxruntime_go` 는 CGO 바인딩이고 런타임에 libonnxruntime 공유 라이브러리가 필요하다. 플랫폼별 수동 설치를 피하기 위해 **빌드와 실행은 Docker 안에서만** 수행한다.
>
> ONNX Runtime 이 어떻게 CPU 만으로 빠른 추론을 해주는지(MLAS, Operator fusion, Execution Provider) 는 [TRAINING.md § 7](TRAINING.md#7-onnx-runtime-의-동작-원리) 를 참고.

### 2.1 사전 준비

- Docker 20.10+ (buildx 사용 시 24+)
- 학습 산출물(`training/artifacts/{spam_detector.onnx,vocab.json}`)이 준비되어 있어야 한다

### 2.2 이미지 빌드

```bash
make docker-build
# 내부적으로: docker build -t spam-detector:<git-tag> -t spam-detector:latest .
```

주요 환경 변수:

| 변수 | 기본값 | 설명 |
|------|--------|------|
| `IMAGE` | `spam-detector` | 이미지 이름 |
| `TAG` | `git describe` 결과 또는 `dev` | 이미지 태그 |
| `REGISTRY` | (없음) | 레지스트리 prefix (예: `registry.manty.co.kr`) |

`Dockerfile` 구조:

1. **builder 스테이지**: `golang:1.26-bookworm` 에서 CGO 로 정적-유사 바이너리 생성
2. **runtime 스테이지**: `debian:bookworm-slim` + ONNX Runtime (GitHub Release `v${ORT_VERSION}` 에서 다운로드) + 바이너리
3. 비-root 유저(`app`, UID 1000) 로 실행

### 2.3 멀티 아키텍처 빌드 (linux/amd64 + linux/arm64)

```bash
docker buildx create --use   # 최초 1회
REGISTRY=ghcr.io/myorg make docker-buildx
```

---

## 3. 실행

### 3.1 이미지 실행

```bash
make docker-run
# 내부적으로: docker run --rm -it \
#     -p 8080:8080 \
#     -v $(pwd)/training/artifacts:/app/artifacts:ro \
#     --name spam-detector spam-detector:latest
```

학습 산출물이 없으면 실행 전 에러로 알려준다.

### 3.2 변수로 제어

| 변수 | 기본값 | 설명 |
|------|--------|------|
| `PORT` | `8080` | 호스트 노출 포트 |
| `ARTIFACTS` | `$(pwd)/training/artifacts` | 호스트의 모델 디렉토리 (컨테이너의 `/app/artifacts` 에 마운트) |

예시: 포트 변경 + 외부 경로의 모델 사용

```bash
PORT=9000 ARTIFACTS=/opt/models make docker-run
```

### 3.3 직접 `docker run` 사용

Makefile 없이 실행하려면:

```bash
docker run --rm -it \
  -p 8080:8080 \
  -v $(pwd)/training/artifacts:/app/artifacts:ro \
  spam-detector:latest
```

컨테이너는 `SIGTERM` 에 그레이스풀 셧다운한다 (`docker stop` 으로 정상 종료 가능).

---

## 4. API 테스트

### 4.1 Health Check

```bash
curl -s http://localhost:8080/healthz
# {"status":"ok"}
```

### 4.2 단일 메시지 분류

**Request**

```bash
curl -s -X POST http://localhost:8080/classify \
  -H 'Content-Type: application/json' \
  -d '{"text":"오늘 점심 뭐 먹지?"}'
```

**Response**

```json
{"isSpam":false,"pSpam":0.0117,"pHam":0.9883}
```

| 필드 | 타입 | 설명 |
|------|------|------|
| `isSpam` | boolean | `pSpam > pHam` 여부 |
| `pSpam` | float | 스팸 확률 (0~1) |
| `pHam` | float | 정상 확률 (0~1) |

### 4.3 배치 분류

**Request**

```bash
curl -s -X POST http://localhost:8080/classify/batch \
  -H 'Content-Type: application/json' \
  -d '{"texts":["내일 회의 있어요","즉시대출 010-1234-5678","ㅅ.ㅍ.ㅏ.ㅁ 아닙니다"]}'
```

**Response**

```json
{
  "results": [
    {"text":"내일 회의 있어요","isSpam":false,"pSpam":0.2481,"pHam":0.7519},
    {"text":"즉시대출 010-1234-5678","isSpam":true,"pSpam":0.8591,"pHam":0.1409},
    {"text":"ㅅ.ㅍ.ㅏ.ㅁ 아닙니다","isSpam":true,"pSpam":0.9087,"pHam":0.0913}
  ]
}
```

배치 최대 크기는 128. 초과 시 400.

### 4.4 에러 응답

| 상황 | HTTP | Body |
|------|------|------|
| `text` 누락 | 400 | `{"error":"'text' is required"}` |
| 잘못된 JSON | 400 | `{"error":"..."}` |
| `texts` 비어 있음 | 400 | `{"error":"'texts' must contain at least one message"}` |
| 배치 초과 | 400 | `{"error":"batch size exceeds limit"}` |
| 추론 실패 | 500 | `{"error":"..."}` |

---

## 5. End-to-end 흐름 요약

```bash
# 1) 학습
cd training && source .venv/bin/activate
python make_sample_data.py          # 또는 실제 data/*.csv 로 교체
python train.py
python export_onnx.py
python verify_onnx.py

# 2) Docker 이미지 빌드
cd ..
make docker-build

# 3) 실행 (학습 산출물을 /app/artifacts 에 마운트)
make docker-run

# 4) 호출
curl -s -X POST localhost:8080/classify \
  -H 'Content-Type: application/json' \
  -d '{"text":"테스트 메시지"}'
```

---

## 6. 모델 재학습 루틴

프로덕션 데이터가 쌓인 후 주기적인 재학습 권장 순서:

1. 새 로그 수집 → 라벨링 → `training/data/*.csv` 갱신
2. `python train.py` 로 새 체크포인트 생성
3. `python export_onnx.py` → `python verify_onnx.py` 로 ONNX 검증
4. `metrics.json` 의 F1 비교 → 기존 대비 개선 시 `artifacts/` 덮어쓰기
5. `docker stop spam-detector && make docker-run` 으로 재시작 (현재는 모델 Hot-reload 미지원)
   - 이미지 자체는 변경할 필요 없음: 모델은 볼륨 마운트이므로 재빌드 없이 반영됨
