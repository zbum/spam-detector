# Training Pipeline

Char-CNN 기반 채팅/댓글 스팸 분류 모델 학습 파이프라인.
Magika의 철학을 따라 **CPU만으로 ms 단위 추론**이 가능하도록 작은 ONNX 모델을 산출한다.

> 이 문서는 **실행 가이드**다. 토크나이저/모델/ONNX Runtime 의 설계 원리는 상위 [../TRAINING.md](../TRAINING.md) 를 참조.
> 전체 프로젝트(서빙/Docker 포함) 개요는 [../README.md](../README.md).

## 구조

```
training/
├── config.yaml           # 모든 하이퍼파라미터
├── tokenizer.py          # 문자 단위 토크나이저 (URL/전화번호 정규화 포함)
├── model.py              # Char-CNN
├── dataset.py            # CSV → PyTorch Dataset
├── train.py              # 학습 루프 + 검증 + 테스트
├── export_onnx.py        # .pt → .onnx 변환
├── verify_onnx.py        # PyTorch vs ONNX 출력 일치 확인
└── make_sample_data.py   # 합성 샘플 CSV 생성 (스모크 테스트용)
```

학습 산출물(체크포인트, 어휘, ONNX, 메트릭)은 `training/artifacts/` 에 저장되며 Git에서 무시된다.

## 사용법

```bash
cd training
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 1) 합성 데이터로 파이프라인 스모크 테스트
python make_sample_data.py

# 2) 학습
python train.py --config config.yaml

# 3) ONNX 변환
python export_onnx.py --config config.yaml

# 4) 일치성 검증
python verify_onnx.py --config config.yaml
```

## 실제 데이터 형식

`data/train.csv`, `data/val.csv`, `data/test.csv`:

| text | label |
|------|-------|
| 안녕하세요 회의 몇 시였죠? | 0 |
| ★초대박★ 무료이벤트 당첨! http://bit.ly/xxx | 1 |

- `label`: 0 = ham(정상), 1 = spam
- 문자열 길이 제한 없음 (`config.yaml` 의 `max_length` 로 잘림)
- URL과 전화번호는 토크나이저가 자동으로 `<url>`, `<phone>` 으로 정규화

## 추후 Go 측 연동

`artifacts/spam_detector.onnx` 와 `artifacts/vocab.json` 두 파일만 Go 런타임으로 복사하면 된다.
Go 측 추론 구현은 다음 단계에서 `onnxruntime_go` 로 붙인다.
