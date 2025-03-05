import requests
import re

import requests
import re


def remove_html_tags(text):
    """
    移除 HTML 标签，保留主要文本内容。
    """
    text = re.sub(r'<script.*?>.*?</script>', '', text, flags=re.DOTALL)  # 移除 script 标签
    text = re.sub(r'<style.*?>.*?</style>', '', text, flags=re.DOTALL)  # 移除 style 标签
    text = re.sub(r'<[^>]+>', ' ', text)  # 移除所有 HTML 标签
    text = re.sub(r'\s+', ' ', text).strip()  # 去除多余空白
    return text


def fetch_text_from_url(http_url, proxy=None):
    """
    从指定的 HTTP URL 通过代理获取网页内容，并提取主要文本。
    """
    proxies = {"http": proxy, "https": proxy} if proxy else None
    response = requests.get(http_url, proxies=proxies)
    response.raise_for_status()

    text = remove_html_tags(response.text)
    return text


def gs(arg1) -> dict:
    proxy_url = "http://10.144.1.10:8080"
    result_str = ""
    for go in arg1[0]['organic_results']:
        # result_str += fetch_text_from_url(go['link'], proxy=proxy_url)
        result_str += go['title'] + go['snippet']
        break
    return {
        "result": result_str,
    }


if __name__ == '__main__':
    google_o = [
        {
            "organic_results": [
                {
                    "title": "《哪吒2》突破115亿元，冲击全球TOP10！正面PK《美国队长4》",
                    "link": "https://www.stcn.com/article/detail/1528640.html",
                    "snippet": "数据显示，《哪吒2》累计票房已经突破115亿元，距离进入全球票房影史榜前十只差5亿元。 据猫眼平台数据，截至2月16日早上10时，《哪吒之魔童闹海》总票房（含 ..."
                },
                {
                    "title": "又刷新！《哪吒2》总票房超115亿！",
                    "link": "http://www.news.cn/fortune/20250216/b8d273a4c9da4434ad7d210f4491a35a/c.html",
                    "snippet": "又刷新！《哪吒2》总票房超115亿！ ... 据网络平台数据（灯塔专业版），《哪吒之魔童闹海》总票房（含点映及预售）已超115亿元。转发祝贺，期待《哪吒2》再创历史！"
                },
                {
                    "title": "再刷纪录！《哪吒2》总票房超115亿元",
                    "link": "http://www.ce.cn/xwzx/gnsz/gdxw/202502/16/t20250216_39292817.shtml",
                    "snippet": "再刷纪录！《哪吒2》总票房超115亿元 ... 成为首部进入全球票房榜前11亚洲电影！ 此前《哪吒之魔童闹海》已 ..."
                },
                {
                    "title": "北美首日票房超2200万《哪吒2》距全球票房前10仅差不到3亿 ...",
                    "link": "https://finance.sina.com.cn/tech/discovery/2025-02-16/doc-inekshpe4468646.shtml",
                    "snippet": "快科技2月16日消息，据灯塔专业版全球影史票房榜实时数据，《哪吒之魔童闹海》总票房（含点映、预售及海外票房）已超117.64亿元，距进入全球影史票房榜前10 ..."
                },
                {
                    "title": "《哪吒2》导演饺子登顶中国导演票房榜",
                    "link": "https://www.zaobao.com.sg/realtime/china/story20250215-5881929",
                    "snippet": "据猫眼专业版最新预测数据，《哪吒2》的总票房将达到160亿元，届时《哪吒2》将成为全球动画电影第一名，并在全球电影票房榜中排名第五，仅次于《泰坦尼克号》 ..."
                },
                {
                    "title": "冲刺全球影史票房榜前十！《哪吒2》票房破110亿元",
                    "link": "http://www.xinhuanet.com/fortune/20250215/4de6405c45a24698815f58f395c271f6/c.html",
                    "snippet": "冲刺全球影史票房榜前十！《哪吒2》票房破110亿元-据网络平台数据（灯塔专业版），《哪吒之魔童闹海》总票房（含点映及预售）已超110亿元，超过《速度与 ..."
                },
                {
                    "title": "又刷新！《哪吒2》总票房超115亿",
                    "link": "https://www.stdaily.com/web/gdxw/2025-02/16/content_297482.html",
                    "snippet": "据网络平台数据（灯塔专业版），《哪吒之魔童闹海》总票房（含点映及预售）已超115亿元。转发祝贺，期待《哪吒2》再创历史！ 责任编辑：冷媚. 相关稿件：."
                },
                {
                    "title": "超越《复联》，《哪吒2》进入全球票房榜前11名",
                    "link": "https://www.guancha.cn/internation/2025_02_15_765210.shtml",
                    "snippet": "截至目前《哪吒之魔童闹海》累计票房（含预售及海外票房）已超110.26亿，超《复仇者联盟》票房成绩，成为首部进入全球票房榜前11亚洲电影！ 此前《哪吒之魔童 ..."
                },
                {
                    "title": "超110亿！哪吒2进入全球票房榜第11位",
                    "link": "https://www.stdaily.com/web/gdxw/2025-02/15/content_297423.html",
                    "snippet": "哪吒2进入全球票房榜第11位 2025-02-15 14:38:22 来源: 人民日报客户端 据猫眼专业版数据，《哪吒之魔童闹海》累计票房（含预售及海外票房）超110.26亿，超《 ..."
                }
            ]
        }
    ]
    output = gs(google_o)
    print(output)
