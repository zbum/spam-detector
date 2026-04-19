"""Generate synthetic chat/comment spam data for pipeline smoke tests.

Real training should replace these CSVs with actual labeled chat logs.
This exists so `python train.py` can run end-to-end on day one.
"""
from __future__ import annotations

import csv
import random
from pathlib import Path


HAM_TEMPLATES = [
    "오늘 점심 뭐 먹지?",
    "회의 15분 뒤에 시작합니다",
    "링크 고마워요 확인해볼게요",
    "ㅋㅋㅋ 그러게요",
    "넵 확인했습니다",
    "주말에 시간 괜찮으세요?",
    "혹시 그 문서 공유 가능하실까요",
    "오 이거 괜찮네요",
    "내일 오전에 전화드릴게요",
    "영화 재밌었어요?",
    "수고하셨습니다 내일 봬요",
    "아 그거 저도 궁금했어요",
    "자료 잘 받았습니다 감사합니다",
    "좋은 아침입니다",
    "커피 한잔 하실래요",
]

SPAM_TEMPLATES = [
    "★초대박★ 무료 이벤트 당첨! http://bit.ly/abcd 지금 클릭",
    "[광고] 즉시대출 010-1234-5678 무심사 당일입금",
    "💰 비트코인 하루 50% 수익 보장 ▶ https://scam.example",
    "신규가입시 50만원 즉시지급! 카톡ID: easymoney99",
    "ㅅ.ㅍ.ㅏ.ㅁ 아닙니다 진짜 고수익 부업 문의주세요",
    "성인 무료채팅 지금 접속 www.fake-site.co",
    "VVIP 전용 스포츠 픽 적중률 98% 문의 @telegramID",
    "당첨을 축하드립니다 상품 수령 링크 http://phish.example",
    "대출 한도 1억까지 가능 연락처 남겨주세요",
    "[★놓치면후회★] 오늘까지만 80% 할인 지금 클릭!!!",
    "재택으로 월300 가능 카톡 문의 money_kr",
    "불법 아닙니다 안전합니다 010.9999.0000",
    "무통장 즉시 입금 보장 ㄱr입문의",
    "지금 가입만해도 5만원 ㅈㅣ급 링크 타고 오세요",
    "고수익 알바 모집 나이무관 성별무관 연락주세요 t.me/xxx",
    "오늘 한국 들어가요. line 으로 연락해요."
]


def synth(templates: list[str], n: int, noise_p: float = 0.2) -> list[str]:
    out = []
    for _ in range(n):
        base = random.choice(templates)
        if random.random() < noise_p:
            pos = random.randint(0, len(base))
            noise = random.choice("!!..~~😂💯🔥✨⭐️ㅎㅎㅋㅋ")
            base = base[:pos] + noise + base[pos:]
        out.append(base)
    return out


def write_csv(path: Path, rows: list[tuple[str, int]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["text", "label"])
        writer.writerows(rows)


def main() -> None:
    random.seed(0)
    out_dir = Path("data")

    def build(n_ham: int, n_spam: int) -> list[tuple[str, int]]:
        rows = [(t, 0) for t in synth(HAM_TEMPLATES, n_ham)] + \
               [(t, 1) for t in synth(SPAM_TEMPLATES, n_spam)]
        random.shuffle(rows)
        return rows

    write_csv(out_dir / "train.csv", build(1500, 1500))
    write_csv(out_dir / "val.csv",   build(300, 300))
    write_csv(out_dir / "test.csv",  build(300, 300))
    print(f"[info] wrote synthetic CSVs under {out_dir.resolve()}")


if __name__ == "__main__":
    main()
