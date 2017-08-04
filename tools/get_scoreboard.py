import lxml.html

def run():
    root = lxml.html.parse("/tmp/CartPole-v0.html").getroot()
    for tr in root.xpath("//tr[@class='success']"):
        record = {}
        for i, td in enumerate(tr.xpath("./td")):
            if (i == 0) and ("writeup" in td.xpath("normalize-space(./span)")):
                a = td.xpath("./a")[0]
                record["url"] = a.get("href")
                record["username"] = a.text.strip().replace("'s algorithm", "")
            if (i == 1):
                record["score"] = td.xpath("./a/text()")[0].strip()
        if "username" in record:
            print record

if __name__ == "__main__":
    run()
