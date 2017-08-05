#!/usr/bin/env python
# -*- coding: utf-8 -*-
import pandas as pd

def run():
    df = pd.read_csv("../scoreboard/openai_gym_scoreboard-2017-08-05.csv", parse_dates=["timestamp"], encoding="utf-8")
    df["date"] = df["timestamp"].dt.strftime("%Y-%m-%d")
    print df
    text  = u"| 問題 | ユーザー名 | スコア | 提出時刻 |\n"
    text += u"| ---- | ---------- | ------ | -------- |\n"
    for ldx, row in df.iterrows():
        text += u"| %(env_id)s | %(username)s | [%(score)s](%(url)s) | %(date)s |\n" % row
    print text

if __name__ == "__main__":
    run()
