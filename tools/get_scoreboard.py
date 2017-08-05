import logging 
logging.basicConfig(format='%(asctime)-15s %(message)s', level=logging.DEBUG)
logger = logging.getLogger(__name__)

import pandas as pd
import requests
import lxml.html
from gym.envs.registration import registry

BASE_URL = "https://gym.openai.com"

def run(fpath):
    df_all = None
    for env_id in registry.env_specs.keys():
        logger.info("Downloading: `env_id=%s`" % env_id)
        r = requests.get("%s/envs/%s" % (BASE_URL, env_id))
        if r.status_code == 200:
            records = parse(html=r.text)
            if len(records) > 0:
                df = pd.DataFrame(records)
                df["env_id"] = env_id
                df_all = pd.concat([df_all, df])
                df_all.to_csv(fpath, index=False, encoding="utf-8")
        else:
            logger.info("[%s Error] Download failed for `env_id=%s`" % (r.status_code, env_id))

def parse(html):
    records = []
    root = lxml.html.fromstring(html) # .getroot()
    for tr in root.xpath("//tr[@class='success']"):
        record = {}
        for i, td in enumerate(tr.xpath("./td")):
            if (i == 0):
                if ("writeup" in td.xpath("normalize-space(./span)")):
                    a = td.xpath("./a")[0]
                    record["url"] = "%s%s" % (BASE_URL, a.get("href"))
                    record["username"] = a.text.strip().replace("'s algorithm", "")
            elif (i == 1):
                record["score"] = td.xpath("./a/text()")[0].strip()
            elif (i == 2):
                record["timestamp"] = td.xpath("./a/span[@class='timestamp']")[0].get("title")
        if "username" in record:
            logger.debug(record)
            records.append(record)
    return records

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--fpath', type=str, default="/tmp/openai_gym_scoreboard.csv")
    args = parser.parse_args()
    run(args.fpath)
