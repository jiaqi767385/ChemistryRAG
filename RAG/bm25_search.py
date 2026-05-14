import os
import re
from pathlib import Path

import jieba
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from rank_bm25 import BM25Okapi

"""
该模块利用 BM25 算法结合化学实体加权，对化学题目的来源章节进行检索评估。
主要用于验证 RAG 系统的检索准确率（Top-K Recall）。
"""

# 设置绘图风格与中文支持
sns.set_theme(style="whitegrid")
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


class ChemLogicAnalyzer:
    """化学逻辑分析器，用于语料检索与评估。

    该类负责加载化学文档语料库，通过分词和化学实体识别，利用加权的 BM25 算法
    计算题目与文档段落的相关性。

    Attributes:
        raw_texts (list[str]): 原始语料文本列表。
        metadata (list[dict]): 对应文本的元数据（如章节号）。
        chem_weight (float): 化学实体的得分权重倍率。
        chem_elements (dict): 元素符号与中文名称的映射表。
        chem_set (set): 预定义的化学实体词库，用于权重判断。
        tokenized_corpus (list[list[str]]): 分词后的语料库。
        bm25 (BM25Okapi): BM25 检索模型实例。
    """

    def __init__(self, corpus_dir, chem_weight=1.0):
        """初始化分析器并加载语料。

        Args:
            corpus_dir (str): 语料库所在目录路径。
            chem_weight (float, optional): 化学关键词的权重增强倍数。默认为 10.0。
        """
        self.raw_texts = []
        self.metadata = []
        self.chem_weight = chem_weight

        # 完整元素周期表
        self.chem_elements = {
            "氢": "H", "氦": "He", "锂": "Li", "铍": "Be", "硼": "B", "碳": "C", "氮": "N", "氧": "O", "氟": "F",
            "氖": "Ne",
            "钠": "Na", "镁": "Mg", "铝": "Al", "硅": "Si", "磷": "P", "硫": "S", "氯": "Cl", "氩": "Ar", "钾": "K",
            "钙": "Ca",
            "钪": "Sc", "钛": "Ti", "钒": "V", "铬": "Cr", "锰": "Mn", "铁": "Fe", "钴": "Co", "镍": "Ni", "铜": "Cu",
            "锌": "Zn",
            "镓": "Ga", "锗": "Ge", "砷": "As", "硒": "Se", "溴": "Br", "氪": "Kr", "铷": "Rb", "锶": "Sr", "钇": "Y",
            "锆": "Zr",
            "铌": "Nb", "钼": "Mo", "锝": "Tc", "钌": "Ru", "铑": "Rh", "钯": "Pd", "银": "Ag", "镉": "Cd", "铟": "In",
            "锡": "Sn",
            "锑": "Sb", "碲": "Te", "碘": "I", "氙": "Xe", "铯": "Cs", "钡": "Ba", "镧": "La", "铈": "Ce", "镨": "Pr",
            "钕": "Nd",
            "钷": "Pm", "钐": "Sm", "铕": "Eu", "Gd": "Gd", "钆": "Gd", "铽": "Tb", "镝": "Dy", "钬": "Ho", "铒": "Er",
            "铥": "Tm",
            "镱": "Yb", "镥": "Lu", "铪": "Hf", "钽": "Ta", "钨": "W", "铼": "Re", "锇": "Os", "铱": "Ir", "铂": "Pt",
            "金": "Au",
            "汞": "Hg", "铊": "Tl", "铅": "Pb", "铋": "Bi", "钋": "Po", "砹": "At", "氡": "Rn", "钫": "Fr", "镭": "Ra",
            "锕": "Ac",
            "钍": "Th", "镤": "Pa", "铀": "U", "镎": "Np", "钚": "Pu", "镅": "Am", "锔": "Cm", "锫": "Bk", "Cf": "Cf",
            "锎": "Cf",
            "锿": "Es", "镄": "Fm", "钔": "Md", "锘": "No", "铹": "Lr"
        }
        self.chem_set = set(self.chem_elements.keys()) | set(self.chem_elements.values())

        self._load_corpus(corpus_dir)

        if self.raw_texts:
            print(f"语料库加载完成: 包含 {len(self.raw_texts)} 条记录，化学权重倍率: {self.chem_weight}")
            self.tokenized_corpus = [self._smart_tokenizer(text) for text in self.raw_texts]
            self.bm25 = BM25Okapi(self.tokenized_corpus)

    def _smart_tokenizer(self, text):
        """针对化学文本的智能分词器。

        结合结巴分词与正则匹配，保留化学方程式、分子式及希腊字母。

        Args:
            text (str): 待分词的原始文本。

        Returns:
            list[str]: 提取出的关键词和化学实体列表。
        """
        # 匹配希腊字母、分子式、离子符号等
        chem_pattern = re.compile(r'[ΔΔα-ωΑ-Ω][\w\d^/_]*|[A-Z][a-z]?\d*[\d\w^+-]*')
        words = jieba.lcut(text)
        formulas = chem_pattern.findall(text)
        # 过滤单字中文（通常为虚词），保留化学式
        tokens = [w for w in words if len(w) > 1] + formulas
        return tokens

    def _is_chem_entity(self, token):
        """判断一个词是否为定义的化学实体。

        Args:
            token (str): 待检测的词汇。

        Returns:
            bool: 如果是化学元素符号或名称则返回 True。
        """
        return token in self.chem_set

    def get_weighted_bm25_scores(self, query_tokens):
        """计算加权后的 BM25 相关性分数。

        Args:
            query_tokens (list[str]): 查询句的分词列表。

        Returns:
            np.ndarray: 语料库中每条记录对应的得分数组。
        """
        aggregate_scores = np.zeros(len(self.raw_texts))
        for token in query_tokens:
            # 获取该词在所有文档中的 BM25 原始分数
            token_scores_list = self.bm25.get_batch_scores([token], list(range(len(self.raw_texts))))
            token_scores = np.array(token_scores_list)
            # 应用化学权重
            weight = self.chem_weight if self._is_chem_entity(token) else 1.0
            aggregate_scores += token_scores * weight
        return aggregate_scores

    def run_evaluation(self, questions, k=5):
        """执行评估流程并生成可视化报告。

        判定准则：只要 Top-K 检索出的章节列表中包含题目预期的章节，即视为命中（Recall@K）。

        Args:
            questions (list[dict]): 题目列表，每个 dict 包含 'text'、'expected' 和 'id'。
            k (int, optional): 评估 Top-K 召回。默认为 5。
        """
        results = []
        for q in questions:
            query_tokens = self._smart_tokenizer(q['text'])
            final_scores = self.get_weighted_bm25_scores(query_tokens)

            # 获取得分最高的前 K 个候选索引
            top_k_indices = np.argsort(final_scores)[::-1][:k]
            top_k_chapters = [self.metadata[idx]["chapter"] for idx in top_k_indices]

            expected_chap = q['expected']
            is_hit = 1 if expected_chap in top_k_chapters else 0

            results.append({
                "id": q['id'],
                "expected": expected_chap,
                "top_k_list": top_k_chapters,
                "is_correct": is_hit,
                "score": final_scores[top_k_indices[0]] if len(top_k_indices) > 0 else 0
            })

        df = pd.DataFrame(results)
        self._generate_topk_report(df, k)
        self._visualize(df, k)

    def _load_corpus(self, directory_path):
        """私有方法：遍历目录并加载 Markdown 格式语料。

        Args:
            directory_path (str): 语料库路径。
        """
        base_path = Path(directory_path)
        for file_path in base_path.rglob('*.md'):
            try:
                # 从文件名提取章节号
                chapter_match = re.search(r'\d+', file_path.name)
                chapter_num = int(chapter_match.group()) if chapter_match else -1
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    if not content.strip(): continue
                    # 按标题级别拆分段落
                    segments = re.split(r'\n(?=#{1,3} )', content)
                    for seg in segments:
                        text = seg.strip()
                        # 过滤过短的噪声文本
                        if len(text) > 30:
                            self.raw_texts.append(text)
                            self.metadata.append({"chapter": chapter_num})
            except Exception:
                continue

    def parse_questions(self, file_paths):
        """解析题目文件。

        预期题目格式为: "章节号-序号 题目内容"

        Args:
            file_paths (list[str]): 题目文件路径列表。

        Returns:
            list[dict]: 包含解析后题目信息的字典列表。
        """
        all_questions = []
        q_pattern = re.compile(r'^(\d+)-\d+\s*(.*)', re.S)
        for fp in file_paths:
            if not os.path.exists(fp): continue
            with open(fp, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line: continue
                    match = q_pattern.match(line)
                    if match:
                        all_questions.append({
                            "expected": int(match.group(1)),
                            "id": line.split()[0],
                            "text": match.group(2)
                        })
        return all_questions

    def _generate_topk_report(self, df, k):
        """打印评估统计报告。

        Args:
            df (pd.DataFrame): 评估结果数据框。
            k (int): Top-K 参数值。
        """
        print(f"\n" + "=" * 35)
        print(f"Top-{k} 检索召回报告 (化学权重: x{self.chem_weight})")
        print(f"总体召回率 (Recall@{k}): {df['is_correct'].mean():.2%}")
        print("判定准则: 预期章节存在于 Top-K 列表中即为正确")
        print("=" * 35)

        stats = df.groupby('expected').agg(
            total=('id', 'count'),
            hit_count=('is_correct', 'sum'),
            recall_at_k=('is_correct', 'mean')
        )
        print(stats.to_string(formatters={'recall_at_k': '{:,.2%}'.format}))

    def _visualize(self, df, k):
        """绘制各章节召回率柱状图。

        Args:
            df (pd.DataFrame): 评估结果数据框。
            k (int): Top-K 参数值。
        """
        plt.figure(figsize=(14, 7))
        data_plot = df.groupby('expected')['is_correct'].mean().reset_index()
        sns.barplot(
            data=data_plot,
            x='expected',
            y='is_correct',
            hue='expected',
            palette='viridis',
            legend=False
        )
        plt.axhline(df['is_correct'].mean(), color='red', linestyle='--',
                    label=f'平均 Recall@{k}: {df["is_correct"].mean():.2%}')
        plt.title(f"各章节 Top-{k} 检索命中率 (Recall) chem_weight={self.chem_weight}", fontsize=16)
        plt.xlabel("章节号")
        plt.ylabel(f"命中率 (Recall@{k})")
        plt.ylim(0, 1.1)
        plt.legend()
        plt.show()

    def find_optimal_weight(self, questions, weight_range=range(0, 11), k=5):
        """遍历不同权重寻找最优解并绘制折线图。

        Args:
            questions (list[dict]): 解析后的题目列表。
            weight_range (range/list): 权重的遍历范围。
            k (int): 评估 Top-K 召回率。

        Returns:
            float: 表现最好的权重值。
        """
        weights = list(weight_range)
        recall_scores = []

        print(f"\n开始搜索最优 chem_weight (Range: {weights[0]}-{weights[-1]}, k={k})...")

        for w in weights:
            self.chem_weight = w
            results = []
            for q in questions:
                query_tokens = self._smart_tokenizer(q['text'])
                final_scores = self.get_weighted_bm25_scores(query_tokens)
                top_k_indices = np.argsort(final_scores)[::-1][:k]
                top_k_chapters = [self.metadata[idx]["chapter"] for idx in top_k_indices]
                results.append(1 if q['expected'] in top_k_chapters else 0)

            avg_recall = np.mean(results)
            recall_scores.append(avg_recall)
            print(f"Weight: {w} -> Overall Recall@{k}: {avg_recall:.2%}")

        # 找到最优权重
        best_idx = np.argmax(recall_scores)
        best_weight = weights[best_idx]
        best_recall = recall_scores[best_idx]

        # 绘制折线图
        self._plot_weight_trend(weights, recall_scores, best_weight, best_recall, k)

        return best_weight

    def _plot_weight_trend(self, weights, scores, best_w, best_r, k):
        """私有辅助方法：绘制权重趋势折线图。"""
        plt.figure(figsize=(10, 6))
        plt.plot(weights, scores, marker='o', linestyle='-', color='#2c7fb8', linewidth=2, markersize=8)

        # 标注最优顶点
        plt.annotate(f'Optimal Weight: {best_w}\nMax Recall: {best_r:.2%}',
                     xy=(best_w, best_r),
                     xytext=(best_w, best_r - 0.05),
                     ha='center',
                     arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=8),
                     bbox=dict(boxstyle="round,pad=0.3", fc="yellow", alpha=0.3))

        plt.title(f"Impact of Chemical Weight on Retrieval Recall@{k}", fontsize=14)
        plt.xlabel("Chemical Entity Weight Multiplier")
        plt.ylabel(f"Average Recall@{k}")
        plt.xticks(weights)
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    # 路径动态定位
    current_script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_script_dir)

    CORPUS_DIR = os.path.join(project_root, "corpus_chunks", "inorganic")
    QUESTION_FILES = [
        os.path.join(project_root, "problems", "无机化学_无图片版_定量计算题_question.md"),
        os.path.join(project_root, "problems", "无机化学_无图片版_化学推断题_question.md")
    ]

    # 初始化
    analyzer = ChemLogicAnalyzer(CORPUS_DIR)

    # 解析题目
    questions = analyzer.parse_questions(QUESTION_FILES)

    if questions:
        # 寻找最优权重并自动画折线图
        best_w = analyzer.find_optimal_weight(questions, weight_range=range(0, 11), k=5)

        # 设置为最优权重，生成报告和章节柱状图
        print(f"\n>>> 使用最优权重 {best_w} 重新生成详细评估报告...")
        analyzer.chem_weight = best_w
        analyzer.run_evaluation(questions, k=5)
    else:
        print("未找到有效题目，请检查路径。")