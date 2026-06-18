"""
자체 검증셋에 대한 Balanced Accuracy 계산 (대회 지표 모사).

Balanced Accuracy = (Acc_ambiguous + Acc_disambiguated) / 2
  → ambiguous/disambiguated 정확도를 *각각* 출력해 어느 쪽이 약한지 진단한다.
    (둘 중 하나만 높으면 최종 점수가 깎이므로 둘 다 모니터링)

입력 CSV 필요 컬럼: answer(예측), gold(정답), label_type(ambiguous|disambiguated)
"""

import argparse
import logging

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def acc(df):
    """정규화 비교 정확도."""
    norm = lambda s: "".join(str(s).split()).strip("'\"").lower()
    if len(df) == 0:
        return 0.0
    hit = sum(norm(p) == norm(g) for p, g in zip(df["answer"], df["gold"]))
    return hit / len(df)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred", required=True, help="answer/gold/label_type 포함 CSV")
    args = parser.parse_args()

    df = pd.read_csv(args.pred)
    amb = df[df["label_type"] == "ambiguous"]
    dis = df[df["label_type"] == "disambiguated"]
    acc_amb, acc_dis = acc(amb), acc(dis)
    balanced = (acc_amb + acc_dis) / 2

    logger.info("ambiguous     정확도: %.4f (n=%d)", acc_amb, len(amb))
    logger.info("disambiguated 정확도: %.4f (n=%d)", acc_dis, len(dis))
    logger.info("== Balanced Accuracy: %.4f ==", balanced)


if __name__ == "__main__":
    main()
