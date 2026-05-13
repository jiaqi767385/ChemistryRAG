import os
import re
import jieba
import numpy as np
import pandas as pd
import matplotlib
import seaborn as sns

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer


class WaterfallOptimizer:
    def __init__(self, corpus_dir, model_name='BAAI/bge-small-zh-v1.5'):
        self.raw_texts = []
        self.metadata = []
        self.bm25 = None
        self.encoder = None
        self.all_embeddings = None

        os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
        self._load_corpus(os.path.abspath(corpus_dir))

        if self.raw_texts:
            print(f"Successfully loaded {len(self.raw_texts)} documents.")
            # 初始化检索组件
            tokenized_corpus = [self._smart_tokenizer(text) for text in self.raw_texts]
            self.bm25 = BM25Okapi(tokenized_corpus)
            self.encoder = SentenceTransformer(model_name)
            self.all_embeddings = self.encoder.encode(self.raw_texts, normalize_embeddings=True)

    def _smart_tokenizer(self, text):
        chem_pattern = re.compile(r'[Δα-ωΑ-Ω][\w\d^/_]*|[A-Z][a-z]?\d*[\d\w^+-]*')
        return jieba.lcut(text) + chem_pattern.findall(text)

    def get_waterfall_ranks(self, query, threshold):
        query_tokens = self._smart_tokenizer(query)
        bm25_scores = np.array(self.bm25.get_scores(query_tokens))
        max_bm25 = np.max(bm25_scores) if len(bm25_scores) > 0 else 0

        if max_bm25 >= threshold:
            return np.argsort(bm25_scores)[::-1]
        else:
            query_vec = self.encoder.encode([query], normalize_embeddings=True)
            vec_scores = np.dot(self.all_embeddings, query_vec.T).flatten()
            return np.argsort(vec_scores)[::-1]

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

    def evaluate(self, questions, k_val, threshold):
        correct = 0
        for q in questions:
            ranks = self.get_waterfall_ranks(q['text'], threshold)
            if q['expected'] in [self.metadata[i]["chapter"] for i in ranks[:k_val]]:
                correct += 1
        return correct / len(questions)

    def save_bar_chart(self, questions, k_val, threshold, label):
        results = []
        for q in questions:
            ranks = self.get_waterfall_ranks(q['text'], threshold)
            is_hit = 1 if q['expected'] in [self.metadata[i]["chapter"] for i in ranks[:k_val]] else 0
            results.append({"chapter": q['expected'], "is_correct": is_hit})

        df = pd.DataFrame(results)
        stats = df.groupby('chapter')['is_correct'].mean().reset_index()
        avg_recall = df['is_correct'].mean()

        sns.set_theme(style="whitegrid")
        plt.figure(figsize=(16, 8))
        ax = sns.barplot(data=stats, x='chapter', y='is_correct', palette='viridis', edgecolor='black', hue='chapter',
                         legend=False)
        ax.axhline(avg_recall, color='red', linestyle='--', linewidth=2, label=f'Avg Recall: {avg_recall:.2%}')

        plt.title(f"Waterfall Recall@{k_val} (Optimal Threshold={threshold:.1f})", fontsize=18, pad=20)
        plt.xlabel("Chapter Number", fontsize=14)
        plt.ylabel("Recall Rate", fontsize=14)
        plt.ylim(0, 1.1)
        plt.legend(loc='upper right')
        plt.tight_layout()
        plt.savefig(f"waterfall_detail_k{k_val}_{label}.png", dpi=300)
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

    optimizer = WaterfallOptimizer(CORPUS_DIR)
    questions = parse_questions(QUESTION_FILES)

    if optimizer.raw_texts and questions:
        # --- 阶段 1: 阈值寻优 (K=5) ---
        print("\n--- Phase 1: Threshold Optimization (K=5) ---")
        thresholds = np.linspace(0, 50, 11)  # 0, 5, 10, ..., 50 (共11个样本)
        th_recalls = []

        for t in thresholds:
            rec = optimizer.evaluate(questions, k_val=5, threshold=t)
            th_recalls.append(rec)
            print(f"Testing Threshold: {t:4.1f} | Overall Recall@5: {rec:.2%}")

        best_th = thresholds[np.argmax(th_recalls)]
        print(f"\n✨ Best Threshold Found: {best_th:.1f}")

        # 绘制阈值折线图
        plt.figure(figsize=(10, 6))
        plt.plot(thresholds, th_recalls, marker='s', color='darkorange', linewidth=2, markersize=8)
        plt.title("Recall@5 vs. BM25 Threshold", fontsize=14)
        plt.xlabel("Threshold (BM25 Score)", fontsize=12)
        plt.ylabel("Overall Recall Rate", fontsize=12)
        plt.grid(True, linestyle=':', alpha=0.6)
        plt.savefig("threshold_optimization_line.png", dpi=300)
        plt.close()

        # --- 阶段 2: 在最优阈值下测试不同 K ---
        print(f"\n--- Phase 2: Evaluating K-Values at Threshold {best_th:.1f} ---")
        for k in [3, 5, 7]:
            optimizer.save_bar_chart(questions, k, best_th, "optimal")

        print("\n✅ All tasks completed. Line chart and bar charts generated.")