"""Sample 'xiaoyu': a cross-border e-commerce seller talking to an assistant in Chinese (assistant style).

Small; exists to exercise the character-trigram path (unicode61 cannot segment Chinese) and CJK rule tagging.
"""
from evalset.schema import Q, Sample, session

SESSIONS = (
    session("2025-04-10 12:00", [
        ("user", "你好，我叫小雨，在杭州做跨境电商运营，主要卖宠物用品到欧洲。", "zh-intro"),
        ("assistant", "你好小雨！很高兴认识你。需要我帮你做什么？"),
        ("user", "先记住一条规则：以后给我写的所有文案都用简体中文，不要夹杂英文缩写。", "zh-rule-chinese"),
        ("assistant", "好的，以后所有文案都用简体中文，不用英文缩写。"),
        ("user", "我们的主力产品是一款自动喂食器，型号叫 PF-200，在德国站定价 59 欧元。", "zh-price-59"),
        ("assistant", "记下了：PF-200 自动喂食器，德国站 59 欧元。"),
    ]),
    session("2025-05-22 09:30", [
        ("user", "德国站竞争太激烈了，PF-200 降价到 49 欧元，从今天开始。", "zh-price-49"),
        ("assistant", "好的，PF-200 德国站售价更新为 49 欧元。"),
        ("user", "另外我养了一只柯基，叫年糕，两岁了。", "zh-dog"),
        ("assistant", "年糕这个名字很可爱。"),
        ("user", "六月我要去柏林参加宠物用品展，展会是 6 月 18 日到 20 日。", "zh-expo"),
        ("assistant", "好的，6 月 18 日到 20 日柏林宠物用品展，祝顺利。"),
    ]),
    session("2025-07-03 20:15", [
        ("user", "柏林展会回来了，签了一个波兰的分销商，首批订单 2000 台。", "zh-order"),
        ("assistant", "恭喜！首批 2000 台是个很好的开始。"),
        ("user", "前天刚把样品寄给他们，定金是三周前付的。", "zh-sample-ship"),
        ("assistant", "好的，记下了：样品前天寄出，定金三周前已付。"),
        ("user", "以后别再推荐猫砂类产品了，我们决定只做狗用品。", "zh-rule-dog-only"),
        ("assistant", "明白，以后只讨论狗用品，不再推荐猫砂类产品。"),
    ]),
)

QUESTIONS = (
    Q("zh-fr-01", "fact_recall/single_fact", "小雨养的柯基叫什么名字？", ["zh-dog"], answer="年糕"),
    Q("zh-fr-02", "fact_recall/single_fact", "波兰分销商的首批订单有多少台？", ["zh-order"], answer="2000 台"),
    Q("zh-go-01", "governance/new_value_overwrite", "PF-200 在德国站现在卖多少钱？", ["zh-price-49"], trap=["zh-price-59"], answer="49 欧元"),
    Q("zh-ex-01", "execution/rule_application", "给小雨写一段新品文案时要遵守什么规则？", ["zh-rule-chinese"], answer="全部用简体中文，不夹英文缩写"),
    Q("zh-te-01", "temporal/dates_intervals", "柏林宠物用品展是什么时候？", ["zh-expo"], answer="2025 年 6 月 18 日到 20 日"),
    Q("zh-pe-01", "personalization/preference_recall", "小雨的店以后还做猫用品吗？", ["zh-rule-dog-only"], answer="不做，只做狗用品"),
    # v0.4 additions (2026-10-08): relative-time resolution in Chinese
    Q("zh-te-02", "temporal/dates_intervals", "小雨是哪一天把样品寄给波兰分销商的？", ["zh-sample-ship"], answer="2025 年 7 月 1 日（7 月 3 日的前天）"),
    Q("zh-te-03", "temporal/dates_intervals", "波兰分销商的定金大约是什么时候付的？", ["zh-sample-ship"], answer="2025 年 6 月中旬，约 6 月 12 日（7 月 3 日的三周前）"),
)

SAMPLE = Sample(name="xiaoyu", style="assistant", sessions=SESSIONS, questions=QUESTIONS,
                description="Chinese-language seller chat, 3 sessions Apr-Jul 2025; price change, two rules, a dog.")
