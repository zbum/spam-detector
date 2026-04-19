# Training & ONNX Runtime Deep Dive

이 문서는 `spam-detector` 의 학습 파이프라인이 **왜 그렇게 설계되었는지**, 그리고 **ONNX Runtime 이 어떤 원리로 CPU만 써서 빠른 추론을 제공하는지** 를 정리한다. 사용 방법은 [README.md](README.md) 를, 이 문서는 내부 원리와 설계 의도를 설명한다.

## 목차

1. [전체 흐름](#1-전체-흐름)
2. [왜 Char-CNN인가 — Magika 의 철학](#2-왜-char-cnn인가--magika-의-철학)
3. [토크나이저 원리](#3-토크나이저-원리)
4. [모델 아키텍처](#4-모델-아키텍처)
5. [학습 루프](#5-학습-루프)
6. [ONNX 란 무엇인가](#6-onnx-란-무엇인가)
7. [ONNX Runtime 의 동작 원리](#7-onnx-runtime-의-동작-원리)
8. [PyTorch → ONNX Export](#8-pytorch--onnx-export)
9. [Go 바인딩 `onnxruntime_go`](#9-go-바인딩-onnxruntime_go)
10. [재학습 & 배포 루틴](#10-재학습--배포-루틴)

---

## 1. 전체 흐름

```
┌──────────────────┐     ┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│  CSV data        │ ──▶ │ CharTokenizer │ ──▶ │ Char-CNN(PT) │ ──▶ │ ONNX export  │
│  text,label      │     │  vocab.json   │     │  model.pt    │     │  .onnx       │
└──────────────────┘     └──────────────┘     └──────────────┘     └──────┬───────┘
                                                                          ▼
                                                         ┌─────────────────────────┐
                                                         │ ONNX Runtime (Go)        │
                                                         │  - Graph optimization   │
                                                         │  - CPU EP               │
                                                         │  - < 5ms/req            │
                                                         └─────────────────────────┘
```

학습(Python)과 서빙(Go)은 **ONNX 파일과 vocab.json 을 경유**해서만 의사소통한다. 이 둘 사이에 런타임 의존성이 전혀 없다는 점이 설계 핵심이다.

---

## 2. 왜 Char-CNN인가 — Magika 의 철학

[Google Magika](https://github.com/google/magika) 는 파일 타입을 맞추는 도구인데, **바이트 시퀀스에 작은 신경망을 돌려서 CPU 만으로 ms 단위**에 해결한다. 핵심 아이디어는 세 가지:

1. **작은 모델**: 수십만~수백만 파라미터면 충분
2. **Raw 입력**: 바이트(또는 문자)를 그대로 먹임 → 휴리스틱 최소화
3. **CPU EP + ONNX**: GPU/Python 런타임 없이 어디서나 동작

채팅/댓글 스팸 분류도 이 철학이 그대로 적용된다.

| 대안 | 이유로 배제 / 선택 |
|------|-------------------|
| Naive Bayes | 빠르지만 우회 문자열(`ㅅ.ㅍ.ㅏ.ㅁ`)/이모지/신조어에 취약 |
| BERT/Transformer | 정확도 높지만 모델 크기(수백 MB)·지연시간이 채팅 처리에 과함 |
| 형태소 분석기 + 단어 CNN | 채팅은 비문/신조어가 많아 형태소 분석기 정확도가 흔들림 |
| **Char-CNN (선택)** | 신조어/우회문자/다국어 혼용에 강하고, 모델 크기 ~수십만 파라미터 |

---

## 3. 토크나이저 원리

`training/tokenizer.py` 의 `CharTokenizer` 가 수행하는 일.

### 3.1 정규화

```python
def normalize(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)       # 전각 → 반각, 호환 문자 통일
    text = _URL_RE.sub(" <url> ", text)              # URL 을 특수 토큰으로
    text = _PHONE_RE.sub(" <phone> ", text)          # 전화번호도 치환
    text = _WHITESPACE_RE.sub(" ", text).strip()     # 공백 정규화
    return text
```

**왜 URL/전화번호를 치환하는가?**

모델이 `http://bit.ly/xYz123` 을 통째로 외우면 새로운 짧은URL/도메인이 오면 못 잡는다. `<url>` 이라는 단일 토큰으로 묶으면 "어떤 형태든 URL이 있다는 사실" 만 학습한다. 전화번호도 마찬가지.

**왜 NFKC 인가?**

`①②③`, `ｓｐａｍ`, `㈜` 같은 호환 문자를 표준형으로 바꿔서 같은 의미를 같은 벡터로 매핑한다. 스팸은 이런 호환문자 남용이 잦다.

### 3.2 Vocabulary 구성

```python
def fit(self, texts, min_freq=2):
    # 1) 모든 문자의 빈도를 센다
    counter = Counter()
    for raw in texts:
        for tok in self._iter_tokens(normalize(raw)):
            counter[tok] += 1

    # 2) 특수 토큰을 먼저 배치
    self.char_to_id = {tok: idx for idx, tok in enumerate(self.SPECIAL_TOKENS)}
    # ["<pad>", "<unk>", "<url>", "<phone>"] → 0, 1, 2, 3

    # 3) 빈도 ≥ min_freq 인 문자에 ID 부여 (빈도 내림차순)
    for char, freq in counter.most_common():
        if freq < min_freq: break
        self.char_to_id[char] = next_id
        next_id += 1
```

- **`<pad>` (ID 0)**: 가변 길이를 `max_length` 로 맞추기 위한 패딩
- **`<unk>` (ID 1)**: vocab 에 없는 문자 (테스트 시 등장)
- **`<url>`, `<phone>`**: 위에서 설명

`min_freq=2` 로 한 번만 나온 노이즈 문자는 `<unk>` 로 처리한다.

### 3.3 Encoding

`encode()` 는 항상 **고정 길이** 배열을 반환한다.

```python
def encode(self, text):
    ids = [self.char_to_id.get(tok, UNK) for tok in self._iter_tokens(normalize(text))]
    ids = ids[:max_length]                          # 자르기
    ids.extend([PAD] * (max_length - len(ids)))     # 짧으면 패딩
    return ids  # shape: [max_length]
```

고정 길이를 쓰는 이유:
- ONNX 모델은 정적 shape 이 더 효율적 (그래프 최적화)
- Go 측에서도 사전에 할당한 입력 텐서를 **재사용** 가능 (GC 부담 감소)

---

## 4. 모델 아키텍처

`training/model.py`.

```
input_ids [B, L]
  └─ Embedding(V, E)   →  [B, L, E]
       └─ transpose    →  [B, E, L]
            ├─ Conv1d(k=3, C) → ReLU → MaxPool1d → [B, C]
            ├─ Conv1d(k=4, C) → ReLU → MaxPool1d → [B, C]
            └─ Conv1d(k=5, C) → ReLU → MaxPool1d → [B, C]
                 └─ Concat    →  [B, 3C]
                      └─ Dropout
                           └─ Linear(3C, 2)  →  logits [B, 2]
```

| 설계 결정 | 이유 |
|----------|------|
| **Multi-kernel (3/4/5)** | 3-gram/4-gram/5-gram 문자 패턴을 동시 캡처 — 우회 문자열은 짧은 n-gram 이 결정적 |
| **Conv1D + ReLU** | O(L·C) 연산으로 CPU 에서 vectorized — CNN 은 ONNX Runtime 의 최적화된 커널(`MlasConv`) 사용 |
| **Global Max Pooling** | 입력 길이 변화에 강인. 각 채널의 "가장 강한 신호"만 남김 |
| **Dropout 0.3** | 작은 모델에서 과적합 완화 |
| **Linear → Softmax** | 2-class 분류 (Softmax 는 후처리에서 수행) |

모델 크기는 `vocab_size × 64 + 3 × (64 × 128 × kernel) + (384 × 2)` 정도로, 어휘가 2천 자일 때 **약 160K 파라미터(0.6MB)**.

---

## 5. 학습 루프

`training/train.py` 의 핵심.

### 5.1 Class-weighted CrossEntropy

채팅 데이터는 ham 이 spam 보다 훨씬 많다. 단순 CE 를 쓰면 모델이 "전부 ham" 이라고 예측하는 걸 선호하게 된다.

```python
weights = [total / (2 * count_ham), total / (2 * count_spam)]
criterion = nn.CrossEntropyLoss(weight=weights)
```

각 클래스의 역빈도로 가중치를 주면 손실이 **클래스별 평균** 처럼 계산된다.

### 5.2 Early Stopping

```python
if val_f1 > best_f1:
    best_f1 = val_f1
    torch.save(model.state_dict(), ckpt)
    bad_epochs = 0
else:
    bad_epochs += 1
    if bad_epochs >= patience: break
```

검증 F1(spam 기준) 이 `patience` 에폭 동안 개선되지 않으면 중단한다. 과적합 방지 + 시간 절약.

### 5.3 Optimizer

- **AdamW**: 일반 Adam + weight decay 를 제대로 분리한 변형. 작은 CNN + 짧은 학습에 무난
- **weight_decay 1e-4**: 파라미터가 과하게 커지는 것 억제
- **lr 1e-3**: Char-CNN 기본값. 필요하면 `config.yaml` 에서 조정

### 5.4 메트릭

- Binary F1 (spam 기준) 을 주 지표로 사용
- Classification report (precision/recall/f1 per class) 출력으로 모델이 어느 클래스에 치우치는지 파악

---

## 6. ONNX 란 무엇인가

### 6.1 ONNX = 신경망의 표준 IR

**ONNX (Open Neural Network Exchange)** 는 딥러닝 모델을 **프레임워크 독립적으로** 표현하는 파일 포맷이다. Protocol Buffers 로 직렬화된다.

```
┌──────────────────────────────────────────────────┐
│  ONNX Model (model.onnx)                          │
├──────────────────────────────────────────────────┤
│  - opset_version: 20                              │
│  - graph:                                         │
│    - inputs:  [{name: "input_ids", shape: [B,L]}] │
│    - outputs: [{name: "logits",    shape: [B,2]}] │
│    - nodes: [                                     │
│        Gather(embedding_weight, input_ids),       │
│        Transpose(…),                              │
│        Conv(…), Relu(…), GlobalMaxPool(…),        │
│        Concat(…), Dropout(…), Gemm(…)             │
│      ]                                            │
│    - initializers: [embedding_weight, conv_w, …]  │
└──────────────────────────────────────────────────┘
```

핵심 구성 요소:

| 요소 | 역할 |
|------|------|
| **Operator (Op)** | `Conv`, `Relu`, `Gemm` 등 표준화된 연산 |
| **Opset version** | 어떤 연산 세트를 쓰는지 (20 = 2024년 표준) |
| **Graph** | DAG 로 표현된 연산 순서 |
| **Initializer** | 학습된 가중치 (텐서 상수) |
| **IO 스펙** | 입력/출력 이름, 타입, shape |

### 6.2 왜 필요한가

- PyTorch 로 학습하고 TensorFlow 에서 서빙, Go 에서 추론 등 **학습/서빙 언어 분리**가 가능
- 파일 하나로 가중치 + 그래프가 모두 포함 → 배포 단순화
- 프레임워크-독립 **최적화/양자화** 도구 체인 (ONNX Runtime, TensorRT, OpenVINO 등)

---

## 7. ONNX Runtime 의 동작 원리

**ONNX Runtime (ORT)** 은 ONNX 모델을 실제로 실행하는 Microsoft의 크로스 플랫폼 엔진이다. Magika 도 이걸 쓴다.

### 7.1 실행 흐름

```
InferenceSession(model.onnx)
  ├─ [1] Protobuf parse → in-memory IR
  ├─ [2] Graph Optimization:
  │       • Constant folding (상수 계산 결과를 아예 노드로 대체)
  │       • Operator fusion (Conv+Relu → FusedConv)
  │       • Memory planning (동일 버퍼 재사용)
  │       • Layout optimization (NCHW/NHWC 선택)
  ├─ [3] Provider assignment:
  │       • 각 노드를 어느 "Execution Provider" 가 실행할지 결정
  │       • CPU EP / CUDA EP / CoreML EP / OpenVINO EP …
  └─ [4] Kernel binding:
          • CPU EP는 MLAS(Microsoft Linear Algebra Subprogram)
          • AVX2/AVX-512/ARM NEON 같은 SIMD 활용
```

그 후 `session.Run(inputs)` 시:

```
session.Run()
  ├─ Input binding (tensor allocation/copy)
  ├─ Execute optimized graph node-by-node
  └─ Output binding → 호출자 버퍼에 결과 기록
```

### 7.2 왜 빠른가 (CPU 에서)

1. **MLAS 커널**: 손수 튜닝된 BLAS 수준의 Conv/GEMM 구현. PyTorch eager 모드의 Python 오버헤드를 완전히 제거.
2. **Operator Fusion**: `Conv → Bias → ReLU` 를 하나의 커널로 합쳐 메모리 왕복을 줄임
3. **정적 그래프**: Python interpreter/AutoGrad 엔진이 없음 → 호출당 수십 µs 단위의 고정 오버헤드
4. **Memory planning**: 추론 중 사용할 중간 텐서의 수명을 미리 분석해 버퍼를 재사용 → 할당/해제 비용 0

결과적으로 같은 모델을 PyTorch eager 로 돌릴 때보다 CPU에서 **3-10배 빠르다**.

### 7.3 Execution Provider 개념

ORT 는 "어떤 연산을 어떤 하드웨어로 실행할지" 를 **Execution Provider** 추상화로 분리한다.

| EP | 용도 |
|----|------|
| `CPUExecutionProvider` | 기본. 어디서나 동작 |
| `CUDAExecutionProvider` | NVIDIA GPU |
| `CoreMLExecutionProvider` | Apple Silicon 의 Neural Engine |
| `OpenVINOExecutionProvider` | Intel CPU/iGPU 전용 최적화 |
| `TensorRTExecutionProvider` | NVIDIA TensorRT 로 컴파일 |

우리 서비스는 **CPU 만** 사용한다 (`providers=["CPUExecutionProvider"]`). 스팸 분류는 0.6MB 모델이고 채팅 1건이면 GPU 이전 비용이 추론 자체보다 크기 때문.

### 7.4 Opset / 버전 관리

Opset 은 "ONNX 가 정의한 연산의 버전 집합" 이다. 예: `Conv-11`, `Conv-22` 는 동일 이름이지만 다른 파라미터 규약을 가진다. 모델 export 시 opset 을 고정하면 재현성이 보장된다.

이 프로젝트는 `opset_version=20` 을 사용 (`onnxruntime >= 1.17` 이 완전히 지원).

---

## 8. PyTorch → ONNX Export

`training/export_onnx.py`.

```python
torch.onnx.export(
    model,                                          # nn.Module
    (dummy_input,),                                 # 예제 입력 (shape/dtype 힌트)
    "spam_detector.onnx",
    input_names=["input_ids"],
    output_names=["logits"],
    dynamic_axes={                                  # 어떤 축이 가변인지
        "input_ids": {0: "batch"},
        "logits":    {0: "batch"},
    },
    opset_version=20,
    dynamo=False,                                   # 레거시 TorchScript 기반 exporter
)
```

### 8.1 내부적으로 일어나는 일

1. **Tracing**: PyTorch 가 `model(dummy_input)` 을 한 번 실행하면서 사용된 연산을 **TorchScript 그래프** 로 기록
2. **Lowering**: 각 PyTorch 연산을 대응되는 ONNX Op 로 매핑 (`nn.Conv1d` → `Conv`, `nn.ReLU` → `Relu` 등)
3. **Weight serialization**: `state_dict` 의 가중치를 ONNX initializer 로 저장
4. **Shape inference**: 전체 그래프의 중간 shape 을 계산해서 검증

### 8.2 `dynamo=False` 로 고정한 이유

PyTorch 2.5+ 의 기본 exporter 는 `torch.export` (Dynamo) 기반인데, 이 경로는 opset 다운그레이드 변환기가 아직 불안정하다 (버전 변환 실패 → `No initializer or constant input` 에러). 레거시 TorchScript exporter 는 안정적이며 우리 모델 규모에선 충분하다.

### 8.3 `dynamic_axes` 의 의미

```python
dynamic_axes={"input_ids": {0: "batch"}}
```

- 0번 축(배치 차원)은 **실행 시 임의 크기** 가능하다고 명시
- 나머지(1번 축 = 시퀀스 길이)는 고정(`max_length`)
- 배치 크기가 다른 여러 요청을 같은 세션으로 처리 가능

### 8.4 검증 (`verify_onnx.py`)

PyTorch 와 ONNX 가 수치적으로 동일한지 확인.

```python
max_diff = abs(pytorch_logits - onnx_logits).max()
assert math.isclose(max_diff, 0, abs_tol=1e-4)
```

실제로 `7e-7` 수준 오차만 나온다 (Conv 의 누적 부동소수점 오차).

---

## 9. Go 바인딩 `onnxruntime_go`

`github.com/yalue/onnxruntime_go` 는 ORT 를 Go 에서 쓰기 위한 **CGO 래퍼** 다.

### 9.1 런타임 로딩

```go
ort.SetSharedLibraryPath("/opt/homebrew/lib/libonnxruntime.dylib")
ort.InitializeEnvironment()
```

내부적으로 `dlopen` → `dlsym` 으로 `OrtGetApiBase` 를 찾고 ORT C API 포인터를 받아둔다. 따라서 **빌드 시점에는 헤더만 필요**하고, 실제 라이브러리는 **런타임에** 시스템에서 로드된다.

### 9.2 텐서 할당 재사용

`internal/detector/detector.go` 에서는 세션 생성 시 입출력 텐서를 **미리 할당** 한다.

```go
inputTensor, _  := ort.NewEmptyTensor[int64]  (ort.NewShape(1, maxLen))
outputTensor, _ := ort.NewEmptyTensor[float32](ort.NewShape(1, 2))
session, _ := ort.NewAdvancedSession(onnxPath,
    []string{"input_ids"}, []string{"logits"},
    []ort.Value{inputTensor}, []ort.Value{outputTensor}, nil)
```

`Classify()` 는 이 버퍼를 재사용한다 — 메시지마다 새 텐서를 만들지 않으므로 GC 압박이 0 에 가깝다.

```go
copy(d.input.GetData(), ids)
d.session.Run()
logits := d.output.GetData()  // 길이 2의 float32
```

### 9.3 동시성

한 ORT 세션 + 공유 텐서 구조는 goroutine-safe 가 아니다. 두 가지 선택지:

1. **Mutex 직렬화** (현재 구현): 간단, 저부하에 적합
2. **세션 풀**: 고부하 대비 여러 세션을 만들어 각 고루틴에 나눠줌

우리는 1을 택했다. CPU 추론은 건당 1–5 ms 라 단일 코어로도 수백 RPS 는 충분.

---

## 10. 재학습 & 배포 루틴

```
[1] 데이터 수집         → 라벨링 후 training/data/{train,val,test}.csv
[2] python train.py     → artifacts/model.pt + metrics.json
[3] python export_onnx  → artifacts/spam_detector.onnx
[4] python verify_onnx  → PyTorch ↔ ONNX 일치 확인 + 샘플 추론
[5] metrics.json 비교   → 기존보다 나으면 배포 진행
[6] docker 재시작       → 모델은 볼륨 마운트라 이미지 재빌드 불필요
       make docker-run  (컨테이너는 학습 산출물 디렉토리를 /app/artifacts 에 마운트)
```

### 모델 호환성 체크리스트

재학습할 때 **반드시 같이 갱신되어야 하는 것**:

| 항목 | 이유 |
|------|------|
| `vocab.json` | 토큰→ID 매핑이 바뀌면 Go 측 인코딩도 바뀜 |
| `max_length` 일관성 | Go 측 텐서 shape 은 `vocab.max_length` 에서 자동으로 읽음 (코드 수정 불필요) |
| `model.num_classes` | 현재 2 고정. 늘리면 Go 응답 스키마도 확장 필요 |

### 관측 팁

- 실제 트래픽에서 `pSpam` 분포를 로깅해두면 **임계값** 을 데이터 기반으로 조정 가능 (기본은 0.5)
- False positive 가 문제라면 `pSpam > 0.7` 로 boundary 를 올리고, recall 이 중요하면 낮춘다
- 주기적으로 모델이 확신 없는(`0.4 < pSpam < 0.6`) 샘플을 human-in-the-loop 으로 라벨링해 학습셋에 추가
