"""Create a paragraph-aligned complete Chinese translation of the German thesis.

The script keeps all Markdown control markers, equations, code spans, the English
abstract, and bibliography entries intact. German prose is translated block by
block with a local Marian model. A JSONL alignment manifest makes completeness
auditable without relying on PDF page counts.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import sys
from typing import Iterable

import torch
from transformers import AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoTokenizer


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "thesis" / "manuscript.md"
CONDENSED_ZH = ROOT / "thesis" / "manuscript_zh.md"
DEFAULT_OUTPUT = ROOT / "thesis" / "manuscript_zh_full.md"
DEFAULT_MODEL = ROOT / "tmp" / "models" / "Qwen2.5-3B-Instruct"
DEFAULT_MANIFEST = ROOT / "tmp" / "translation" / "full_translation_alignment.jsonl"
DEFAULT_SUMMARY = ROOT / "tmp" / "translation" / "full_translation_audit.json"


COVER_BLOCKS = {
    "Masterarbeit im Studiengang Data Science & Artificial Intelligence":
        "数据科学与人工智能专业硕士论文（中文全译本）",
    "Fachhochschule Wedel": "韦德尔应用科学大学",
    (
        "Vorgelegt von: Yue Cao  \n"
        "Matrikelnummer: [Matrikelnummer ergänzen]  \n"
        "Anschrift: [Anschrift ergänzen]  \n"
        "E-Mail: [E-Mail-Adresse ergänzen]  \n"
        "Studiensemester: [Studiensemester ergänzen]  \n"
        "Erstbetreuung: [Erstbetreuung ergänzen]  \n"
        "Zweitbetreuung/Praxisbetreuung: [Zweitbetreuung ergänzen]  \n"
        "Abgabedatum: [Abgabedatum ergänzen]"
    ): (
        "作者：Yue Cao  \n"
        "学号：____________________________  \n"
        "地址：____________________________  \n"
        "电子邮箱：____________________________  \n"
        "学期：____________________________  \n"
        "第一导师：____________________________  \n"
        "第二导师/企业导师：____________________________  \n"
        "提交日期：____________________________"
    ),
}


ABSTRACT_TRANSLATIONS = [
    (
        "推荐系统通过从大规模目录中生成个性化排序列表来缓解信息过载。许多应用主要使用准确性指标评价系统质量。"
        "然而，只面向准确性的优化可能产生同质化列表，无法充分呈现用户已有的多重兴趣，并把可见性集中到少量热门对象上。"
        "因此，本文研究三种可在模型训练后使用的重排序方法：当排序准确性的相对损失受到明确限制时，"
        "这些方法能够在多大程度上提高 Top-K 推荐的多样性。"
    ),
    (
        "本研究使用 MovieLens 20M 数据集、按时间顺序划分的训练/验证/测试集，以及采用贝叶斯个性化排序训练的矩阵分解模型作为共同的相关性基础。"
        "评分不低于 4 星被定义为正偏好，评分不高于 2 星被定义为显式负反馈。为避免信息泄漏，迭代式 5-Core 仅依据按时间划分的训练部分确定。"
        "处理后的数据集包含 134,703 名用户、11,851 部电影和 9,952,928 条正交互；其中训练、验证和测试交互分别为 7,908,519、939,551 和 1,104,858 条。"
        "系统首先为每名用户确定评分最高、此前既没有正面观察也没有负面观察的 100 个候选对象，随后生成长度为 10 的推荐列表。"
    ),
    (
        "比较的方法包括最大边际相关性（MMR）、迁移到用户兴趣方面的 xQuAD 变体，以及校准重排序。控制参数 λ 只在验证集上选择。"
        "对于 1%、3%、5% 和 10% 的可接受相对 NDCG@10 损失，分别选择仍满足约束且多样性最高的运行点；测试集在最终评价之前保持完全不使用。"
        "主分析采用三个训练随机种子、用户级配对 Bootstrap 置信区间、符号检验、Holm 校正和标准化效应量。"
        "此外还检查候选集大小为 50 和 200、输出长度为 5 和 20、不同用户分组以及 Tag Genome 表示。"
    ),
    (
        "在主实验中，MMR 在 5% 验证预算下选择 λ=0.40，并在测试集上使基于类型的列表内多样性平均提高 16.30%，同时使 NDCG@10 降低 3.80%。"
        "平均目录覆盖率从 48.28% 上升到 49.74%，长尾占比从 1.68% 上升到 2.04%。xQuAD 和校准重排序对传统列表内多样性的改变较弱，"
        "但分别改善了用户兴趣方面覆盖和推荐列表与历史兴趣画像的一致程度。因此不存在普遍最优的方法：MMR 最适合提高成对列表多样性，"
        "xQuAD 最适合提高方面覆盖，校准方法最适合按比例呈现用户兴趣。Tag Genome 敏感性分析支持这一总体结论，同时也表明测得的效果取决于所采用的特征表示。"
    ),
    (
        "本文提供了一个可复现的实验设计、一条基于预算的选择规则，以及对各项目标指标的差异化解释。结果证明，在准确性损失受到控制时可以获得实质性的多样性收益。"
        "结果同时表明，离线多样性、校准、目录分布和实际用户价值不能被视为同一概念。因此，对于生产环境，本文建议采用针对具体方法的目标指标、分组监控，并在此后进行在线测试。"
    ),
    "关键词：推荐系统，多样性，重排序，MMR，xQuAD，校准，贝叶斯个性化排序，MovieLens 20M",
]


MANUAL_BLOCKS = {
    "Die leitende Forschungsfrage lautet:": "主要研究问题如下：",
    (
        "RQ1: Wie verändert diversitätsorientiertes Re-Ranking die Genauigkeit und die Diversität von "
        "Top-K-Empfehlungen, wenn der relative NDCG-Verlust auf 1 %, 3 %, 5 % oder 10 % begrenzt wird?"
    ): (
        "RQ1：当 NDCG 的相对损失被限制在 1%、3%、5% 或 10% 时，"
        "面向多样性的重排序会如何改变 Top-K 推荐的准确性和多样性？"
    ),
    "Diese Frage wird durch vier Teilfragen präzisiert:": "该问题通过以下四个子问题进一步具体化：",
    (
        "RQ2: Unterscheiden sich MMR, xQuAD und kalibriertes Re-Ranking hinsichtlich paarweiser Diversität, "
        "Abdeckung von Nutzeraspekten und proportionaler Interessenrepräsentation?"
    ): "RQ2：MMR、xQuAD 和校准重排序在成对多样性、用户兴趣方面覆盖和兴趣比例呈现方面是否存在差异？",
    (
        "RQ3: Bleiben die beobachteten Trade-offs bei anderen Kandidatenmengen, Ausgabelängen, "
        "Trainingsseeds und Merkmalsrepräsentationen qualitativ stabil?"
    ): "RQ3：在改变候选集大小、输出长度、训练随机种子和特征表示后，观察到的权衡关系在定性上是否仍然稳定？",
    (
        "RQ4: Profitieren Nutzergruppen mit unterschiedlicher Aktivität und Profilbreite in vergleichbarer "
        "Weise von der Diversifizierung?"
    ): "RQ4：活跃度和画像宽度不同的用户群体能否以相近方式从多样化中受益？",
    (
        "RQ5: Gehen lokale Diversitätsgewinne mit Veränderungen der Katalogabdeckung, "
        "Expositionskonzentration und Long-Tail-Sichtbarkeit einher?"
    ): "RQ5：局部多样性收益是否伴随目录覆盖率、曝光集中度和长尾可见性的变化？",
    "Aus den Fragen werden folgende prüfbare Hypothesen abgeleitet:": "由这些问题推导出以下可检验假设：",
    (
        "H1: Für mindestens einen zulässigen Genauigkeitsverlust bis 5 % existiert ein Re-Ranking-Betriebspunkt, "
        "der ILD@10 gegenüber der unveränderten Relevanzrangliste erhöht."
    ): "H1：对于至少一个不超过 5% 的可接受准确性损失，存在一个重排序运行点，使 ILD@10 高于未改变的相关性排序基线。",
    (
        "H2: MMR erzielt bei gleicher Genauigkeitsnebenbedingung einen größeren Zuwachs der paarweisen ILD "
        "als xQuAD und kalibriertes Re-Ranking."
    ): "H2：在相同准确性约束下，MMR 带来的成对 ILD 增长高于 xQuAD 和校准重排序。",
    (
        "H3: xQuAD erhöht den nutzergewichteten Subtopic Recall stärker als die Baseline, während kalibriertes "
        "Re-Ranking die Kalibrierungsähnlichkeit am stärksten verbessert."
    ): "H3：xQuAD 使用户加权 Subtopic Recall 相对基线提高，而校准重排序对校准相似度的改善最大。",
    (
        "H4: Der relative Diversitätsgewinn von MMR ist bei unterschiedlichen Kandidatenmengen und "
        "Ausgabelängen positiv, seine Größe ist jedoch von N und K abhängig."
    ): "H4：在不同候选集大小和输出长度下，MMR 的相对多样性收益均为正，但其大小取决于 N 和 K。",
    (
        "H5: Die Diversifizierung verbessert Katalogabdeckung und Long-Tail-Anteil, reduziert aber nicht "
        "notwendigerweise die stark konzentrierte Gesamtexposition in gleichem Maß."
    ): "H5：多样化提高目录覆盖率和长尾占比，但不一定以相同幅度降低高度集中的总体曝光。",
    (
        "Die Hypothesen sind richtungsbezogen, werden aber nicht allein über p-Werte beurteilt. Entscheidend "
        "sind Effektgröße, Konfidenzintervall, Stabilität über Seeds und praktische Relevanz. Bei 134.703 "
        "Nutzenden können auch kleine Abweichungen statistisch auffällig werden. Eine methodisch saubere "
        "Interpretation muss deshalb zwischen statistischer Nachweisbarkeit und substanzieller Größe unterscheiden."
    ): (
        "这些假设具有方向性，但不只依据 p 值作出判断。关键还包括效应量、置信区间、随机种子之间的稳定性和实际相关性。"
        "在 134,703 名用户的样本中，即使很小的差异也可能在统计上显著。因此，方法上严谨的解释必须区分统计可检出性与实质性大小。"
    ),
    (
        "Ich versichere hiermit, dass ich die vorliegende Arbeit selbstständig und nur unter Benutzung der "
        "angegebenen Quellen und Hilfsmittel angefertigt habe. Wörtlich oder inhaltlich übernommene Stellen "
        "sind als solche kenntlich gemacht. Die Arbeit wurde in gleicher oder ähnlicher Form noch keiner "
        "anderen Prüfungsbehörde vorgelegt."
    ): (
        "本人声明，本论文由本人独立完成，仅使用文中列明的资料和工具。凡直接或间接引用的内容均已明确标注。"
        "本论文或内容实质相同的作品未曾以相同或类似形式提交其他考试机构。"
    ),
    "Ort, Datum: [Ort und Datum ergänzen]": "地点、日期：____________________________",
    "Unterschrift: ______________________________": "签名：____________________________",
    (
        "Für die Strukturierung, sprachliche Ausarbeitung, Quellcodeunterstützung und technische "
        "Dokumenterstellung wurde OpenAI Codex/ChatGPT als generatives KI-Werkzeug eingesetzt. Die "
        "experimentellen Kennzahlen stammen aus lokal ausgeführten, reproduzierbaren Programmläufen im "
        "angegebenen Repository. Auswahl, Bewertung, fachliche Prüfung und Verantwortung für sämtliche "
        "Inhalte verbleiben beim Verfasser. Vor Abgabe sind diese Erklärung, die konkrete Werkzeugbezeichnung "
        "und der verlangte Detaillierungsgrad mit der Prüfungsordnung und der Betreuung abzustimmen."
    ): (
        "在论文结构组织、语言表达、源代码支持和技术文档制作过程中使用了 OpenAI Codex/ChatGPT 作为生成式人工智能工具。"
        "实验指标来自所述代码仓库中在本地执行的可复现程序运行。所有内容的选择、评价、专业核验和最终责任均由作者承担。"
        "正式提交前，应依据考试规定并与导师确认本声明、具体工具名称以及要求披露的详细程度。"
    ),
}


# Confirmatory research questions and hypotheses from the final Exposé-aligned
# manuscript are translated manually to keep their direction and status exact.
MANUAL_BLOCKS.update({
    "Die im Exposé vom 20. August 2026 festgehaltene leitende Forschungsfrage wird unverändert als RQ0 übernommen:":
        "2026 年 8 月 20 日开题报告中确定的主要研究问题原样保留为 RQ0：",
    "RQ0: Wie unterscheiden sich diversitätsorientierte Re-Ranking-Verfahren hinsichtlich des Trade-offs zwischen Empfehlungsgenauigkeit und Diversität in Top-K-Listen?":
        "RQ0：面向多样性的重排序方法在 Top-K 列表的推荐准确性与多样性权衡方面有何差异？",
    "Sie wird durch fünf Teilfragen präzisiert:": "该问题通过以下五个子问题进一步具体化：",
    "RQ1: Welches Verfahren erzielt bei festen relativen Accuracy-Verlustbudgets von 1 %, 3 %, 5 % und 10 % den höchsten Diversitätsgewinn?":
        "RQ1：在相对准确性损失预算固定为 1%、3%、5% 和 10% 时，哪种方法取得最高的多样性收益？",
    "RQ2: Wie verändern Kandidatenpoolgröße N und Ausgabelänge K die Pareto-Front?":
        "RQ2：候选池大小 N 和输出长度 K 如何改变 Pareto 前沿？",
    "RQ3: Wie sensitiv sind die Ergebnisse gegenüber der Itemrepräsentation, insbesondere Genre-Vektoren gegenüber Tag-Genome-Vektoren?":
        "RQ3：结果对对象表示有多敏感，尤其是类型向量与 Tag Genome 向量之间的差异？",
    "RQ4: Unterscheiden sich die Verfahren für Nutzergruppen mit geringer, mittlerer und hoher historischer Interessenbreite?":
        "RQ4：对于历史兴趣宽度较低、中等和较高的用户群体，各方法是否存在差异？",
    "RQ5: Welche Verfahren verbessern nutzerbezogene Diversität, ohne dabei aggregierte Coverage, Kalibrierung oder Laufzeit unangemessen zu verschlechtern?":
        "RQ5：哪些方法能够改善用户层面的多样性，同时又不会不适当地恶化聚合覆盖率、校准或运行时间？",
    "Die folgenden fünf Hypothesen entsprechen ebenfalls dem Wortlaut und der inhaltlichen Richtung des Exposés. Sie bilden den konfirmatorischen Rahmen der Arbeit:":
        "以下五项假设同样与开题报告的原始措辞和内容方向一致，并构成本研究的验证性框架：",
    "H1: Alle drei Kernverfahren erhöhen gegenüber dem nicht re-rankten Baseline-Ranking mindestens eine Diversity-Metrik, typischerweise bei sinkender NDCG.":
        "H1：与未经重排序的基线排序相比，三种核心方法都会提高至少一项多样性指标，通常伴随 NDCG 下降。",
    "H2: xQuAD erzielt bei Genre-basierten Metriken stärkere Coverage-Gewinne als MMR, weil Genres explizit als Aspekte modelliert werden.":
        "H2：由于类型被显式建模为兴趣方面，xQuAD 在基于类型的指标上取得的覆盖收益强于 MMR。",
    "H3: Kalibriertes Re-Ranking erhält die Verteilung individueller Interessen besser als ein globaler, für alle Nutzenden identischer Diversity-Trade-off.":
        "H3：与对所有用户都相同的全局多样性权衡相比，校准重排序能够更好地保持个体兴趣分布。",
    "H4: Größere Kandidatenpools verschieben die Pareto-Front nach außen, weil mehr relevante und zugleich diverse Items verfügbar sind.":
        "H4：更大的候选池会使 Pareto 前沿向外移动，因为其中可用的相关且多样的对象更多。",
    "H5: Ein global optimales λ ist nicht für alle Nutzersegmente optimal; Nutzende mit breiteren historischen Interessen tolerieren oder benötigen andere Diversitätsgrade.":
        "H5：全局最优的 λ 并非对所有用户分群都最优；历史兴趣更广的用户能够容忍或需要不同程度的多样性。",
    "Ich erkläre hiermit an Eides Statt, dass ich die vorliegende Arbeit selbstständig und ohne Benutzung anderer als der angegebenen Hilfsmittel angefertigt habe; die aus fremden Quellen direkt oder indirekt übernommenen Gedanken sowie durch eine künstliche Intelligenz wie ChatGPT erstellte oder bearbeitete Inhalte sind als solche kenntlich gemacht.":
        "本人在此郑重声明：本论文由本人独立完成，除已注明的辅助工具外未使用其他工具；直接或间接取自外部来源的思想，以及由 ChatGPT 等人工智能生成或处理的内容，均已作出相应标注。",
    "Die Arbeit wurde bisher in gleicher oder ähnlicher Form keiner anderen Prüfungskommission vorgelegt und auch nicht veröffentlicht.":
        "本论文此前未以相同或类似形式提交给其他考试委员会，也未公开发表。",
    "Bei der Erstellung dieser Arbeit wurde OpenAI Codex/ChatGPT für Übersetzung, sprachliche Überarbeitung, Programmierung und Tests, Methoden- und Statistikprüfung, unterstützende Literaturrecherche, Strukturierung sowie Entwürfe ergänzender Textpassagen verwendet. Art und Kontrolle der Unterstützung sind in Anhang E dokumentiert. Experimentelle Kennzahlen stammen aus den angegebenen lokal ausgeführten Programmläufen. Die fachliche Auswahl, Quellenprüfung, Endredaktion und Verantwortung für den Inhalt verbleiben beim Verfasser.":
        "在撰写本论文期间，OpenAI Codex/ChatGPT 被用于翻译、语言修改、编程与测试、方法和统计检查、辅助文献检索、结构组织以及补充文本段落的草拟。相关支持的类型及其控制方式记录于附录 E。实验指标来自文中所列、在本地执行的程序运行。专业内容的选择、来源核查、最终编辑和内容责任仍由作者承担。",
})


POST_REPLACEMENTS = (
    ("推荐者系统", "推荐系统"),
    ("推荐器系统", "推荐系统"),
    ("建议系统", "推荐系统"),
    ("重新排序程序", "重排序器"),
    ("重新排序器", "重排序器"),
    ("再排序", "重排序"),
    ("再排名", "重排序"),
    ("列表内差异", "列表内多样性"),
    ("清单内多样性", "列表内多样性"),
    ("名单内多样性", "列表内多样性"),
    ("清单多样性", "列表多样性"),
    ("目录覆盖面", "目录覆盖率"),
    ("目录覆盖范围", "目录覆盖率"),
    ("准确度", "准确性"),
    ("精确度", "准确性"),
    ("精确性", "准确性"),
    ("校准重新排序", "校准重排序"),
    ("矩阵因子化", "矩阵分解"),
    ("矩阵分解法", "矩阵分解"),
    ("贝叶斯个人化排名", "贝叶斯个性化排序"),
    ("贝叶斯个性化排名", "贝叶斯个性化排序"),
    ("候选数量", "候选集大小"),
    ("候选人集合", "候选集"),
    ("用户方面", "用户兴趣方面"),
    ("业务点", "运行点"),
    ("操作点", "运行点"),
    ("运营点", "运行点"),
    ("长尾部分", "长尾占比"),
    ("长期尾部", "长尾"),
    ("种子", "随机种子"),
    ("验证量", "验证集"),
    ("测试量", "测试集"),
    ("培训集", "训练集"),
    ("培训", "训练"),
    ("电影镜头", "电影"),
    ("电影对象", "电影"),
    ("物件", "对象"),
    ("物体", "对象"),
    ("项目目录", "对象目录"),
    ("使用者", "用户"),
    ("排名列表", "排序列表"),
    ("排名精确性", "排序准确性"),
    ("排名准确性", "排序准确性"),
    ("候选者", "候选对象"),
    ("调参参数", "控制参数"),
    ("本诺特区间", "Bootstrap 置信区间"),
    ("霍尔姆校正", "Holm 校正"),
    ("日志基因组", "Tag Genome"),
    ("标签基因组", "Tag Genome"),
    ("目录覆盖度", "目录覆盖率"),
    ("中位目录覆盖率", "平均目录覆盖率"),
    ("长尾比例", "长尾占比"),
    ("积极互动", "正交互"),
    ("正面互动", "正交互"),
    ("relevance scores", "相关性分数"),
    ("重排序序序", "重排序"),
    ("重排序序", "重排序"),
    ("重排器", "重排序器"),
    ("Maximal Marginal Relevance", "最大边际相关性"),
    ("一对一对比距离", "成对距离"),
    ("一对一对异质性", "成对多样性"),
    ("一对一的", "成对的"),
    ("一对一", "成对"),
    ("Genre熵", "类型熵"),
    ("Genre数量", "类型数量"),
    ("训练随机随机种子", "训练随机种子"),
    ("用户活动", "用户活跃度"),
    ("戏剧", "剧情片"),
    ("互动", "交互"),
    ("可重复", "可复现"),
    ("较少的相关对象", "相关性较低的对象"),
    ("相关性小众对象", "相关的长尾对象"),
    ("的 相关性", "的相关性"),
)


PROTECTED_RE = re.compile(r"(`[^`]+`|\[\[FN\d+\]\])")
HEADING_RE = re.compile(r"^(#{1,3})\s+.+$")
FIGURE_RE = re.compile(r"^@@FIG:[^@]+@@$")
MARKER_RE = re.compile(r"^@@[A-Z_]+(?::[^@]+)?@@$")
NUMBER_RE = re.compile(
    r"(?:[-+−]?\d+(?:[.,]\d+)*(?:\s*%)?|[⁰¹²³⁴⁵⁶⁷⁸⁹]+)"
)
KEEP_RE = re.compile(
    r"(?:(?<![A-Za-z0-9])[-+−]?\d+(?:[.,]\d+)*(?:\s*%)?(?![A-Za-z0-9])|"
    r"[⁰¹²³⁴⁵⁶⁷⁸⁹]+)"
)


def blocks(text: str) -> list[str]:
    return [part.strip("\n") for part in re.split(r"\n\s*\n", text.strip())]


def heading_lines(text: str) -> list[str]:
    return [line.strip() for line in text.splitlines() if HEADING_RE.match(line.strip())]


def figure_lines(text: str) -> list[str]:
    return [line.strip() for line in text.splitlines() if FIGURE_RE.match(line.strip())]


def abbreviation_table(text: str) -> str:
    all_blocks = blocks(text)
    return next(block for block in all_blocks if block.startswith("| 缩略语/符号 |"))


def is_formula_block(text: str) -> bool:
    if "\n" in text:
        return False
    german_words = re.findall(r"\b[A-Za-zÄÖÜäöüß]{3,}\b", text)
    math_signals = sum(symbol in text for symbol in ("=", "Σ", "∈", "→", "||", "^", "_", "√"))
    return math_signals >= 1 and len(german_words) <= 4


def split_sentences(text: str) -> list[str]:
    pieces = re.split(r"(?<=[.!?])\s+(?=[A-ZÄÖÜ0-9„\"(])", text)
    return [piece.strip() for piece in pieces if piece.strip()]


def model_prefix(tokenizer) -> str:
    tokenizer_name = tokenizer.__class__.__name__
    if "Nllb" in tokenizer_name or "Qwen" in tokenizer_name:
        return ""
    return ">>zh_CN<< "


def split_for_model(text: str, tokenizer, token_limit: int = 430) -> list[str]:
    prefix = model_prefix(tokenizer)
    if len(tokenizer(prefix + text, add_special_tokens=True).input_ids) <= token_limit:
        return [text]

    sentences = split_sentences(text)
    if len(sentences) == 1:
        sentences = [part.strip() for part in re.split(r"(?<=[;:])\s+", text) if part.strip()]
    if len(sentences) == 1:
        words = text.split()
        chunks = []
        current = []
        for word in words:
            proposed = " ".join(current + [word])
            if current and len(tokenizer(prefix + proposed).input_ids) > token_limit:
                chunks.append(" ".join(current))
                current = [word]
            else:
                current.append(word)
        if current:
            chunks.append(" ".join(current))
        return chunks

    chunks = []
    current = ""
    for sentence in sentences:
        proposed = f"{current} {sentence}".strip()
        if current and len(tokenizer(prefix + proposed).input_ids) > token_limit:
            chunks.extend(split_for_model(current, tokenizer, token_limit))
            current = sentence
        else:
            current = proposed
    if current:
        chunks.extend(split_for_model(current, tokenizer, token_limit))
    return chunks


def post_edit(text: str) -> str:
    result = text.strip()
    result = re.sub(r"^(?:译文|翻译|中文译文)\s*[：:]\s*", "", result)
    for source, target in POST_REPLACEMENTS:
        result = result.replace(source, target)
    result = result.replace("％", "%")
    result = re.sub(r"\s+([，。；：！？])", r"\1", result)
    result = re.sub(r"([（])\s+", r"\1", result)
    result = re.sub(r"\s+([）])", r"\1", result)
    result = re.sub(r"\s{2,}", " ", result)
    return result


def protect_translation_text(text: str) -> tuple[str, dict[str, str]]:
    replacements: dict[str, str] = {}

    def alpha_index(value: int) -> str:
        result = ""
        while True:
            value, remainder = divmod(value, 26)
            result = chr(ord("A") + remainder) + result
            if value == 0:
                return result
            value -= 1

    def replace(match: re.Match[str]) -> str:
        key = f"ZXQNUM{alpha_index(len(replacements))}〔{match.group(0)}〕XQZ"
        replacements[key] = match.group(0)
        return key

    return KEEP_RE.sub(replace, text), replacements


def restore_translation_text(text: str, replacements: dict[str, str]) -> str:
    result = text
    for key, value in replacements.items():
        result = re.sub(re.escape(key), lambda _: value, result, flags=re.IGNORECASE)
        tag = key.split("〔", 1)[0]
        value_pattern = re.escape(value)
        value_pattern = value_pattern.replace(",", r"\s*[,，.]\s*")
        value_pattern = value_pattern.replace(r"\.", r"\s*[.,，]\s*")
        flexible_key = (
            rf"{re.escape(tag)}\s*[\[〔【（(]\s*{value_pattern}\s*"
            rf"[\]〕】）)]\s*XQZ"
        )
        result = re.sub(flexible_key, lambda _: value, result, flags=re.IGNORECASE)
        result = re.sub(
            rf"{re.escape(tag)}.*?XQZ",
            lambda _: value,
            result,
            flags=re.IGNORECASE,
        )
    leftovers = re.findall(r"ZXQNUM|XQZ", result, flags=re.IGNORECASE)
    if leftovers:
        print(
            f"[placeholder-warning] unresolved marker fragments={leftovers}",
            flush=True,
        )
    expected_numbers = normalized_numbers(" ".join(replacements.values()))
    actual_numbers = normalized_numbers(result)
    if actual_numbers != expected_numbers:
        print(
            "[numeric-warning] protected chunk needs block-level review: "
            f"expected={dict(expected_numbers)}, actual={dict(actual_numbers)}",
            flush=True,
        )
    return result


def translate_batches(
    texts: list[str], tokenizer, model, device, batch_size: int,
    forced_bos_token_id: int | None, is_causal: bool
) -> list[str]:
    translated = []
    prefix = model_prefix(tokenizer)
    system_prompt = (
        "你是一名专业的德中学术翻译。将用户提供的德文论文段落完整、准确地翻译为简体中文。"
        "使用正式、自然、清晰的学术中文，避免生硬直译。不得总结、删减、合并句子、解释或增加内容。"
        "必须保留所有数字、百分比、符号、缩写、公式、专有名词和引用标记。"
        "形如 ZXQNUMA〔0,5〕XQZ 的数字占位符必须逐字原样保留，位置和数量均不得改变。"
        "术语统一如下：Recommender-System=推荐系统；Re-Ranking=重排序；Diversität=多样性；"
        "Genauigkeit=准确性；Intra-List-Diversity=列表内多样性；Kalibrierung=校准；"
        "Kandidatenmenge=候选集；Betriebspunkt=运行点；Matrixfaktorisierung=矩阵分解；"
        "Bayesian Personalized Ranking=贝叶斯个性化排序；Maximal Marginal Relevance=最大边际相关性；"
        "Nutzende=用户；Objekt=对象；paarweise=成对；Genre=类型；Nutzeraspekt=用户兴趣方面；"
        "Profilbreite=画像宽度；Nutzeraktivität=用户活跃度；Relevanzscore=相关性分数；"
        "Merkmalsraum=特征空间；Katalogabdeckung=目录覆盖率；Long-Tail-Anteil=长尾占比；"
        "Bootstrap-Konfidenzintervall=Bootstrap 置信区间；Vorzeichentest=符号检验；"
        "Trainingseed=训练随机种子；mittlere=平均。MMR、xQuAD、NDCG、Recall、ILD、Tag Genome、"
        "Exposure-Gini、Candidate Recall、Subtopic Recall、Bootstrap 和 Holm 等名称保留原文。只输出中文译文。"
    )
    for start in range(0, len(texts), batch_size):
        raw_batch = texts[start:start + batch_size]
        protected_batch = []
        protected_maps = []
        for text in raw_batch:
            protected, mapping = protect_translation_text(text)
            protected_batch.append(protected)
            protected_maps.append(mapping)
        if is_causal:
            batch = [
                tokenizer.apply_chat_template(
                    [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": text},
                    ],
                    tokenize=False,
                    add_generation_prompt=True,
                )
                for text in protected_batch
            ]
        else:
            batch = [prefix + text for text in protected_batch]
        encoded = tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=False,
        ).to(device)
        with torch.inference_mode():
            if is_causal:
                generated = model.generate(
                    **encoded,
                    max_new_tokens=768,
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id,
                )
                prompt_length = encoded["input_ids"].shape[1]
                decoded = tokenizer.batch_decode(
                    generated[:, prompt_length:], skip_special_tokens=True
                )
            else:
                generation_args = {"max_new_tokens": 512}
                if forced_bos_token_id is not None:
                    # NLLB is used as an independent literal cross-check. Greedy
                    # decoding is substantially faster and avoids beam-memory
                    # spikes on the 8 GB evaluation GPU.
                    generation_args["num_beams"] = 1
                    generation_args["forced_bos_token_id"] = forced_bos_token_id
                else:
                    generation_args.update(num_beams=4, early_stopping=True)
                generated = model.generate(**encoded, **generation_args)
                decoded = tokenizer.batch_decode(generated, skip_special_tokens=True)
        translated.extend(
            restore_translation_text(value, mapping)
            for value, mapping in zip(decoded, protected_maps)
        )
        done = min(start + batch_size, len(texts))
        print(f"[translate] {done}/{len(texts)} chunks", flush=True)
    return [post_edit(value) for value in translated]


def normalized_numbers(text: str) -> Counter[str]:
    values = []
    superscript_digits = str.maketrans("⁰¹²³⁴⁵⁶⁷⁸⁹", "0123456789")
    for match in NUMBER_RE.findall(text):
        value = match.translate(superscript_digits).replace("−", "-").replace(" ", "")
        value = value.replace(".", "").replace(",", "")
        values.append(value)
    return Counter(values)


def sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def linewise_units(text: str) -> Iterable[tuple[str, bool]]:
    """Yield pieces and whether they require translation."""
    protected_parts = PROTECTED_RE.split(text)
    for part in protected_parts:
        if not part:
            continue
        if PROTECTED_RE.fullmatch(part):
            yield part, False
            continue
        if "\n" not in part:
            yield part, bool(part.strip())
            continue
        lines = part.splitlines(keepends=True)
        for line in lines:
            content = line.rstrip("\r\n")
            ending = line[len(content):]
            if not content.strip():
                yield line, False
                continue
            list_match = re.match(r"^(\s*(?:[-*]|\d+\.)\s+)(.*)$", content)
            if list_match:
                yield list_match.group(1), False
                yield list_match.group(2), True
                if ending:
                    yield ending, False
            else:
                yield content, True
                if ending:
                    yield ending, False


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument(
        "--reuse-output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Reuse finalized translations whose German source block is unchanged.",
    )
    parser.add_argument(
        "--reuse-manifest",
        type=Path,
        default=DEFAULT_MANIFEST,
        help="Alignment manifest corresponding to --reuse-output.",
    )
    parser.add_argument("--batch-size", type=int, default=12)
    parser.add_argument("--sample", type=int, default=0, help="Translate only the first N prose blocks")
    args = parser.parse_args()

    source_text = args.source.read_text(encoding="utf-8")
    condensed_text = CONDENSED_ZH.read_text(encoding="utf-8")
    source_blocks = blocks(source_text)
    # MANUAL_BLOCKS intentionally contains translations from earlier manuscript
    # revisions. Only entries still present in the current German source apply.

    source_headings = heading_lines(source_text)
    target_headings = heading_lines(condensed_text)
    if len(source_headings) != len(target_headings):
        raise ValueError(f"Heading count mismatch: {len(source_headings)} != {len(target_headings)}")
    source_figures = figure_lines(source_text)
    target_figures = figure_lines(condensed_text)
    if len(source_figures) != len(target_figures):
        raise ValueError(f"Figure count mismatch: {len(source_figures)} != {len(target_figures)}")
    zh_abbreviations = abbreviation_table(condensed_text)

    reuse_cache: dict[str, str] = {}
    if args.reuse_output.exists() and args.reuse_manifest.exists():
        previous_targets = blocks(args.reuse_output.read_text(encoding="utf-8"))
        previous_records = [
            json.loads(line)
            for line in args.reuse_manifest.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if len(previous_targets) == len(previous_records):
            for previous_target, previous_record in zip(previous_targets, previous_records):
                source_hash = previous_record.get("source_sha256")
                if source_hash:
                    reuse_cache.setdefault(source_hash, previous_target)
            print(f"[reuse] loaded {len(reuse_cache)} aligned translations", flush=True)
        else:
            print(
                "[reuse-warning] target/manifest length mismatch; translating without cache: "
                f"{len(previous_targets)} != {len(previous_records)}",
                flush=True,
            )

    print(f"[model] loading {args.model}", flush=True)
    model_name = args.model.name.lower()
    using_nllb = "nllb" in model_name
    using_qwen = "qwen" in model_name
    tokenizer_args = {"local_files_only": True}
    if using_nllb:
        tokenizer_args.update(src_lang="deu_Latn", tgt_lang="zho_Hans")
    tokenizer = AutoTokenizer.from_pretrained(args.model, **tokenizer_args)
    if using_qwen:
        tokenizer.padding_side = "left"
        model = AutoModelForCausalLM.from_pretrained(
            args.model,
            local_files_only=True,
            torch_dtype=torch.float16,
        )
    else:
        model = AutoModelForSeq2SeqLM.from_pretrained(args.model, local_files_only=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    print(f"[model] device={device}", flush=True)

    output_blocks: list[str | None] = []
    records: list[dict] = []
    units_to_translate: list[str] = []
    unit_targets: list[tuple[int, int]] = []
    pending_units: dict[int, list[tuple[str, bool]]] = {}

    heading_index = 0
    figure_index = 0
    abstract_index = 0
    in_german_abstract = False
    in_english_abstract = False
    in_bibliography = False
    translatable_seen = 0

    for block_index, source_block in enumerate(source_blocks):
        stripped = source_block.strip()
        record = {
            "block": block_index + 1,
            "source_sha256": sha256(source_block),
            "source_chars": len(source_block),
            "kind": "",
        }

        if HEADING_RE.fullmatch(stripped):
            translated_block = target_headings[heading_index]
            heading_index += 1
            output_blocks.append(translated_block)
            record.update(kind="heading", target_chars=len(translated_block), target_sha256=sha256(translated_block))
            records.append(record)
            if stripped == "# Zusammenfassung":
                in_german_abstract = True
            elif stripped == "# Abstract":
                in_german_abstract = False
                in_english_abstract = True
            elif stripped == "# Literaturverzeichnis":
                in_bibliography = True
            elif stripped.startswith("# Anhang "):
                in_bibliography = False
            continue

        if stripped == "@@PAGEBREAK@@" and in_english_abstract:
            in_english_abstract = False

        if source_block in COVER_BLOCKS:
            translated_block = COVER_BLOCKS[source_block]
            output_blocks.append(translated_block)
            record.update(kind="cover", target_chars=len(translated_block), target_sha256=sha256(translated_block))
            records.append(record)
            continue

        if source_block in MANUAL_BLOCKS:
            translated_block = MANUAL_BLOCKS[source_block]
            output_blocks.append(translated_block)
            record.update(
                kind="manual",
                target_chars=len(translated_block),
                target_sha256=sha256(translated_block),
            )
            records.append(record)
            continue

        if in_german_abstract:
            if abstract_index >= len(ABSTRACT_TRANSLATIONS):
                raise ValueError("German abstract contains more prose blocks than expected")
            translated_block = ABSTRACT_TRANSLATIONS[abstract_index]
            abstract_index += 1
            output_blocks.append(translated_block)
            record.update(
                kind="manual_abstract",
                target_chars=len(translated_block),
                target_sha256=sha256(translated_block),
            )
            records.append(record)
            continue

        if in_english_abstract or in_bibliography:
            output_blocks.append(source_block)
            record.update(kind="preserved", target_chars=len(source_block), target_sha256=sha256(source_block))
            records.append(record)
            continue

        if source_block.startswith("| Kürzel/Symbol |"):
            output_blocks.append(zh_abbreviations)
            record.update(kind="abbreviation_table", target_chars=len(zh_abbreviations), target_sha256=sha256(zh_abbreviations))
            records.append(record)
            continue

        if FIGURE_RE.fullmatch(stripped):
            translated_block = target_figures[figure_index]
            figure_index += 1
            output_blocks.append(translated_block)
            record.update(kind="figure_marker", target_chars=len(translated_block), target_sha256=sha256(translated_block))
            records.append(record)
            continue

        if MARKER_RE.fullmatch(stripped) or is_formula_block(stripped):
            output_blocks.append(source_block)
            record.update(kind="marker_or_formula", target_chars=len(source_block), target_sha256=sha256(source_block))
            records.append(record)
            continue

        cached_translation = reuse_cache.get(record["source_sha256"])
        if cached_translation is not None:
            output_blocks.append(cached_translation)
            record.update(
                kind="cached_translation",
                target_chars=len(cached_translation),
                target_sha256=sha256(cached_translation),
            )
            records.append(record)
            continue

        translatable_seen += 1
        if args.sample and translatable_seen > args.sample:
            output_blocks.append(source_block)
            record.update(kind="sample_untranslated", target_chars=len(source_block), target_sha256=sha256(source_block))
            records.append(record)
            continue

        output_index = len(output_blocks)
        output_blocks.append(None)
        pieces = list(linewise_units(source_block))
        pending_units[output_index] = pieces
        translated_slots = []
        for piece_index, (piece, needs_translation) in enumerate(pieces):
            if not needs_translation or not piece.strip():
                continue
            for chunk in split_for_model(piece, tokenizer):
                unit_targets.append((output_index, len(translated_slots)))
                units_to_translate.append(chunk)
                translated_slots.append(None)
        record.update(
            kind="translated",
            output_index=output_index,
            source_numbers=dict(normalized_numbers(source_block)),
            source_footnotes=PROTECTED_RE.findall(source_block),
        )
        record["translated_slots"] = translated_slots
        records.append(record)

    forced_bos_token_id = tokenizer.convert_tokens_to_ids("zho_Hans") if using_nllb else None
    translations = translate_batches(
        units_to_translate, tokenizer, model, device, args.batch_size,
        forced_bos_token_id, using_qwen
    )

    record_by_output = {
        record["output_index"]: record
        for record in records
        if record["kind"] == "translated"
    }
    for (output_index, slot), translation in zip(unit_targets, translations):
        record_by_output[output_index]["translated_slots"][slot] = translation

    for output_index, pieces in pending_units.items():
        record = record_by_output[output_index]
        translated_slots = iter(record.pop("translated_slots"))
        rendered_parts = []
        for piece, needs_translation in pieces:
            if not needs_translation or not piece.strip():
                rendered_parts.append(piece)
                continue
            chunk_count = len(split_for_model(piece, tokenizer))
            rendered_parts.append("".join(next(translated_slots) for _ in range(chunk_count)))
        translated_block = post_edit("".join(rendered_parts))
        output_blocks[output_index] = translated_block
        record.update(
            target_chars=len(translated_block),
            target_sha256=sha256(translated_block),
            target_numbers=dict(normalized_numbers(translated_block)),
            target_footnotes=PROTECTED_RE.findall(translated_block),
        )
        record["number_match"] = record["source_numbers"] == record["target_numbers"]
        record["footnote_match"] = record["source_footnotes"] == record["target_footnotes"]

    if any(block is None for block in output_blocks):
        raise RuntimeError("Not all translated blocks were reconstructed")
    if abstract_index != len(ABSTRACT_TRANSLATIONS):
        raise ValueError(
            f"German abstract block mismatch: {abstract_index} != {len(ABSTRACT_TRANSLATIONS)}"
        )
    output_text = "\n\n".join(output_blocks) + "\n"

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(output_text, encoding="utf-8")
    with args.manifest.open("w", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")

    translated_records = [record for record in records if record["kind"] == "translated"]
    summary = {
        "source": str(args.source),
        "output": str(args.output),
        "model": str(args.model),
        "device": str(device),
        "sample_limit": args.sample,
        "source_blocks": len(source_blocks),
        "output_blocks": len(blocks(output_text)),
        "source_headings": len(source_headings),
        "output_headings": len(heading_lines(output_text)),
        "source_figures": len(source_figures),
        "output_figures": len(figure_lines(output_text)),
        "source_tables": source_text.count("@@TABLE:"),
        "output_tables": output_text.count("@@TABLE:"),
        "source_footnotes": len(re.findall(r"\[\[FN\d+\]\]", source_text)),
        "output_footnotes": len(re.findall(r"\[\[FN\d+\]\]", output_text)),
        "translated_blocks": len(translated_records),
        "manual_blocks": sum(record["kind"] == "manual" for record in records),
        "number_mismatch_blocks": [
            record["block"] for record in translated_records if not record.get("number_match", True)
        ],
        "footnote_mismatch_blocks": [
            record["block"] for record in translated_records if not record.get("footnote_match", True)
        ],
        "source_chars": len(source_text),
        "output_chars": len(output_text),
    }
    args.summary.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except AttributeError:
        pass
    main()
