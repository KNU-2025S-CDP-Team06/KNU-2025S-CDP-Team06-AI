import pandas as pd

def check_if_holiday(input_dict: dict) -> list[int]:
    """
    최근 14일간의 매출을 기반으로,
    2주 연속 매출이 0인 요일들을 휴일로 판단하여 리스트로 반환.

    return: [0, 1, ..., 6] 중 휴일 요일 리스트 (0 = 월, ..., 6 = 일)
    """
    weekday_to_revs = {i: [] for i in range(7)}  # 월~일

    base_date = pd.to_datetime(input_dict["date"])  # 기준일 t

    for i in range(1, 15):
        key = f"rev_t-{i}"
        if key in input_dict:
            past_date = base_date - pd.Timedelta(days=i)
            weekday = past_date.weekday()
            weekday_to_revs[weekday].append(input_dict[key])

    # 2개 이상 존재하며 모두 0이면 휴일로 간주
    holidays = [weekday for weekday, revs in weekday_to_revs.items() if len(revs) >= 2 and all(r == 0 for r in revs)]
    return holidays