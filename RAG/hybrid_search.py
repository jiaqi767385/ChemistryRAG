import os
import re
import jieba
import numpy as np
import pandas as pd
import matplotlib
import seaborn as sns

# 强制使用 Agg 后端以兼容服务器环境
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
import faiss


class HybridChemAnalyzer:
    def __init__(self, corpus_dir, model_name='BAAI/bge-small-zh-v1.5'):
        self.raw_texts = []
        self.metadata = []
        self.bm25 = None
        self.all_embeddings = None
        self.encoder = None

        os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

        abs_corpus_path = os.path.abspath(corpus_dir)
        print(f"正在扫描语料目录: {abs_corpus_path}")
        self._load_corpus(abs_corpus_path)

        if not self.raw_texts:
            print(f"❌ 错误: 未找到语料文件")
            return

        print(f"语料加载完成: {len(self.raw_texts)} 条记录。正在初始化模型...")
        self.encoder = SentenceTransformer(model_name)

        # 构建索引
        self.tokenized_corpus = [self._smart_tokenizer(text) for text in self.raw_texts]
        self.bm25 = BM25Okapi(self.tokenized_corpus)

        embeddings = self.encoder.encode(self.raw_texts, show_progress_bar=True, normalize_embeddings=True)
        self.all_embeddings = np.array(embeddings).astype('float32')

    def _smart_tokenizer(self, text):
        chem_pattern = re.compile(r'[Δα-ωΑ-Ω][\w\d^/_]*|[A-Z][a-z]?\d*[\d\w^+-]*')
        words = jieba.lcut(text)
        formulas = chem_pattern.findall(text)
        return [w for w in words if len(w) > 1] + formulas

    def _normalize(self, scores):
        s_min, s_max = np.min(scores), np.max(scores)
        if s_max == s_min: return np.zeros_like(scores)
        return (scores - s_min) / (s_max - s_min)

    def get_hybrid_ranks(self, query, alpha):
        """
        优化后的混合检索：
        alpha = 1.0: 纯 BM25，跳过向量编码
        alpha = 0.0: 纯 Vector，跳过 BM25 计算
        0.0 < alpha < 1.0: 加权融合
        """

        # 场景 A: 纯 BM25 (效率最高，无需模型推理)
        if alpha >= 1.0:
            query_tokens = self._smart_tokenizer(query)
            bm25_scores = np.array(self.bm25.get_scores(query_tokens))
            return np.argsort(bm25_scores)[::-1]

        # 场景 B: 纯 Vector (跳过关键词分词和 BM25 评分)
        if alpha <= 0.0:
            query_vec = self.encoder.encode([query], normalize_embeddings=True).astype('float32')
            # 如果 query_vec 是二维的 (1, dim)，通过 flatten 转为一维
            vec_scores = np.dot(self.all_embeddings, query_vec.T).flatten()
            return np.argsort(vec_scores)[::-1]

        # 场景 C: 混合模式 (正常计算并归一化融合)
        # 1. BM25 分数
        query_tokens = self._smart_tokenizer(query)
        bm25_scores = np.array(self.bm25.get_scores(query_tokens))

        # 2. 向量分数
        query_vec = self.encoder.encode([query], normalize_embeddings=True).astype('float32')
        vec_scores = np.dot(self.all_embeddings, query_vec.T).flatten()

        # 3. 归一化并融合
        norm_bm25 = self._normalize(bm25_scores)
        norm_vec = self._normalize(vec_scores)
        combined = alpha * norm_bm25 + (1.0 - alpha) * norm_vec

        return np.argsort(combined)[::-1]

    def _load_corpus(self, directory_path):
        base_path = Path(directory_path)
        for file_path in base_path.rglob('*.md'):
            with open(file_path, 'r', encoding='utf-8') as f:
                segments = re.split(r'\n(?=#{1,3} )', f.read())
                for seg in segments:
                    if len(seg.strip()) > 5:
                        self.raw_texts.append(seg.strip())
                        match = re.search(r'\d+', file_path.name)
                        self.metadata.append({"chapter": int(match.group()) if match else -1})

    def evaluate(self, questions, k_val, alpha):
        correct = 0
        for q in questions:
            ranks = self.get_hybrid_ranks(q['text'], alpha)
            top_k_chapters = [self.metadata[idx]["chapter"] for idx in ranks[:k_val]]
            if q['expected'] in top_k_chapters:
                correct += 1
        return correct / len(questions)

    def save_chapter_bar_chart(self, questions, k_val, alpha, label):
        """
        按照用户要求的格式进行绘图：
        1. 英文标注 2. 渐变色 3. 格式化 Alpha
        """
        results = []
        for q in questions:
            ranks = self.get_hybrid_ranks(q['text'], alpha)
            is_hit = 1 if q['expected'] in [self.metadata[i]["chapter"] for i in ranks[:k_val]] else 0
            results.append({"chapter": q['expected'], "is_correct": is_hit})

        df = pd.DataFrame(results)
        stats = df.groupby('chapter')['is_correct'].mean().reset_index()
        avg_recall = df['is_correct'].mean()

        # 设置绘图风格
        sns.set_theme(style="whitegrid")
        plt.figure(figsize=(16, 8))

        # 绘制渐变色柱状图
        ax = sns.barplot(
            data=stats,
            x='chapter',
            y='is_correct',
            palette='viridis',
            edgecolor='black',
            linewidth=0.8,
            hue='chapter',
            legend=False
        )

        # 添加平均线
        ax.axhline(avg_recall, color='red', linestyle='--', linewidth=2,
                   label=f'Avg Recall@{k_val}: {avg_recall:.2%}')

        # 设置标题和标签 (Alpha 保留一位小数)
        plt.title(f"Detailed Recall@{k_val} Per Chapter (Alpha={alpha:.1f})", fontsize=18, pad=20)
        plt.xlabel("Chapter Number", fontsize=14)
        plt.ylabel("Recall Rate", fontsize=14)

        plt.ylim(0, 1.1)
        plt.xticks(rotation=0)
        plt.legend(loc='upper right', fontsize=12)

        plt.tight_layout()
        plt.savefig(f"detail_k{k_val}_{label}.png", dpi=300)
        plt.close()


def parse_questions(file_path):
    questions = []
    if not os.path.exists(file_path): return []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            match = re.match(r'^(\d+)-\d+\s*(.*)', line.strip())
            if match:
                questions.append({"expected": int(match.group(1)), "text": match.group(2)})
    return questions


if __name__ == "__main__":
    root_path = os.getcwd()
    CORPUS_DIR = os.path.join(root_path, "ChemistryRAG/corpus_chunks/statistical_mechanical")
    QUESTION_FILES = os.path.join(root_path, "ChemistryRAG/problems/statistical_mechanical_problems.md")

    analyzer = HybridChemAnalyzer(CORPUS_DIR)
    questions = parse_questions(QUESTION_FILES)

    if not analyzer.raw_texts or not questions:
        print("❌ 语料库或问题文件加载失败。")
    else:
        # --- 阶段 1: 寻找最优 Alpha (K=5) ---
        print("\n--- Phase 1: Optimizing Alpha (K=5) ---")
        alphas = np.linspace(0.1, 0.9, 9)
        alpha_recalls = []

        for a in alphas:
            rec = analyzer.evaluate(questions, k_val=5, alpha=a)
            alpha_recalls.append(rec)
            print(f"Alpha: {a:.1f} | Recall@5: {rec:.2%}")

        best_alpha = alphas[np.argmax(alpha_recalls)]
        print(f"\n✨ Best Alpha Found: {best_alpha:.1f}")

        # 绘制 Alpha 寻优折线图
        plt.figure(figsize=(10, 6))
        plt.plot(alphas, alpha_recalls, marker='o', color='royalblue', linewidth=2)
        plt.title(f"Alpha Optimization (K=5, Best={best_alpha:.1f})", fontsize=14)
        plt.xlabel("Alpha (BM25 Weight Weight)", fontsize=12)
        plt.ylabel("Overall Recall@5", fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.savefig("alpha_optimization.png", dpi=300)
        plt.close()

        # --- 阶段 2: 测试不同 K ---
        print(f"\n--- Phase 2: Testing K-Values (Fixed Alpha={best_alpha:.1f}) ---")
        for k in [3, 5, 7]:
            recall = analyzer.evaluate(questions, k_val=k, alpha=best_alpha)
            print(f"K={k} | Total Recall: {recall:.2%}")
            # 调用更新后的绘图函数
            analyzer.save_chapter_bar_chart(questions, k, best_alpha, "best_alpha")

        print("\n✅ Analysis complete. Check the directory for .png files.")