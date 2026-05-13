import os
import re
import jieba
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from rank_bm25 import BM25Okapi

# 设置绘图风格与中文支持
sns.set_theme(style="whitegrid")
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


class ChemLogicAnalyzer:
    def __init__(self, corpus_dir, chem_weight=10.0):
        self.raw_texts = []
        self.metadata = []
        self.chem_weight = chem_weight

        # 完整元素周期表
        self.chem_elements = {
            'H': '氢', 'He': '氦', 'Li': '锂', 'Be': '铍', 'B': '硼', 'C': '碳', 'N': '氮', 'O': '氧', 'F': '氟',
            'Ne': '氖',
            'Na': '钠', 'Mg': '镁', 'Al': '铝', 'Si': '硅', 'P': '磷', 'S': '硫', 'Cl': '氯', 'Ar': '氩', 'K': '钾',
            'Ca': '钙',
            'Sc': '钪', 'Ti': '钛', 'V': '钒', 'Cr': '铬', 'Mn': '锰', 'Fe': '铁', 'Co': '钴', 'Ni': '镍', 'Cu': '铜',
            'Zn': '锌',
            'Ga': '镓', 'Ge': '锗', 'As': '砷', 'Se': '硒', 'Br': '溴', 'Kr': '氪', 'Rb': '铷', 'Sr': '锶', 'Y': '钇',
            'Zr': '锆',
            'Nb': '铌', 'Mo': '钼', 'Tc': '锝', 'Ru': '钌', 'Rh': '铑', 'Pd': '钯', 'Ag': '银', 'Cd': '镉', 'In': '铟',
            'Sn': '锡',
            'Sb': '锑', 'Te': '碲', 'I': '碘', 'Xe': '氙', 'Cs': '铯', 'Ba': '钡', 'La': '镧', 'Ce': '铈', 'Pr': '镨',
            'Nd': '钕',
            'Pm': '钷', 'Sm': '钐', 'Eu': '铕', 'Gd': '钆', 'Tb': '铽', 'Dy': '镝', 'Ho': '钬', 'Er': '铒', 'Tm': '铥',
            'Yb': '镱',
            'Lu': '镥', 'Hf': '铪', 'Ta': '钽', 'W': '钨', 'Re': '铼', 'Os': '锇', 'Ir': '铱', 'Pt': '铂', 'Au': '金',
            'Hg': '汞',
            'Tl': '铊', 'Pb': '铅', 'Bi': '铋', 'Po': '钋', 'At': '砹', 'Rn': '氡'
        }
        self.chem_set = set(self.chem_elements.keys()) | set(self.chem_elements.values())

        self._load_corpus(corpus_dir)

        if self.raw_texts:
            print(f"语料库加载完成: 包含 {len(self.raw_texts)} 条记录，化学权重倍率: {self.chem_weight}")
            self.tokenized_corpus = [self._smart_tokenizer(text) for text in self.raw_texts]
            self.bm25 = BM25Okapi(self.tokenized_corpus)

    def _smart_tokenizer(self, text):
        chem_pattern = re.compile(r'[ΔΔα-ωΑ-Ω][\w\d^/_]*|[A-Z][a-z]?\d*[\d\w^+-]*')
        words = jieba.lcut(text)
        formulas = chem_pattern.findall(text)
        tokens = [w for w in words if len(w) > 1] + formulas
        return tokens

    def _is_chem_entity(self, token):
        return token in self.chem_set

    def get_weighted_bm25_scores(self, query_tokens):
        aggregate_scores = np.zeros(len(self.raw_texts))
        for token in query_tokens:
            token_scores_list = self.bm25.get_batch_scores([token], list(range(len(self.raw_texts))))
            token_scores = np.array(token_scores_list)
            weight = self.chem_weight if self._is_chem_entity(token) else 1.0
            aggregate_scores += token_scores * weight
        return aggregate_scores

    def run_evaluation(self, questions, k=5):
        """
        执行评估：Top-K 逻辑。只要 Top-K 列表中包含预期章节，即视为命中。
        """
        results = []
        for q in questions:
            query_tokens = self._smart_tokenizer(q['text'])
            final_scores = self.get_weighted_bm25_scores(query_tokens)

            # 获取得分最高的前 K 个候选索引
            top_k_indices = np.argsort(final_scores)[::-1][:k]
            top_k_chapters = [self.metadata[idx]["chapter"] for idx in top_k_indices]

            expected_chap = q['expected']

            # --- 修改核心逻辑：Top-K 命中判定 ---
            is_hit = 1 if expected_chap in top_k_chapters else 0

            results.append({
                "id": q['id'],
                "expected": expected_chap,
                "top_k_list": top_k_chapters,
                "is_correct": is_hit,  # 现在的正确定义改为：是否在 Top-K 内
                "score": final_scores[top_k_indices[0]]
            })

        df = pd.DataFrame(results)
        self._generate_topk_report(df, k)
        self._visualize(df, k)

    def _load_corpus(self, directory_path):
        base_path = Path(directory_path)
        for file_path in base_path.rglob('*.md'):
            try:
                chapter_match = re.search(r'\d+', file_path.name)
                chapter_num = int(chapter_match.group()) if chapter_match else -1
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    if not content.strip(): continue
                    segments = re.split(r'\n(?=#{1,3} )', content)
                    for seg in segments:
                        text = seg.strip()
                        if len(text) > 30:
                            self.raw_texts.append(text)
                            self.metadata.append({"chapter": chapter_num})
            except Exception:
                continue

    def parse_questions(self, file_paths):
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
        plt.title(f"各章节 Top-{k} 检索命中率 (Recall)", fontsize=16)
        plt.xlabel("章节号")
        plt.ylabel(f"命中率 (Recall@{k})")
        plt.ylim(0, 1.1)
        plt.legend()
        plt.show()


if __name__ == "__main__":
    root_path = os.getcwd()
    CORPUS_DIR = os.path.join(root_path, "ChemistryRAG/corpus_chunks/inorganic")
    QUESTION_FILES = [
        os.path.join(root_path, "ChemistryRAG/problems/无机化学_无图片版_化学推断题_question.md"),
        os.path.join(root_path, "无机化学_无图片版_定量计算题_question.md")
       ]

    # 设置权重为 10.0，查看 Top-5 的表现
    analyzer = ChemLogicAnalyzer(CORPUS_DIR, chem_weight=0)
    questions = analyzer.parse_questions(QUESTION_FILES)
    if questions:
        analyzer.run_evaluation(questions, k=7)
