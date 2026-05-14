1. bm25_search.py

实验策略：遍历无机化学评测习题集中的1203道题目，在corpus chunks语料库中检索TOP K最匹配的chunk，并将匹配到的chuck与预期chuck进行逻辑对齐。

策略重点：

  1.1 特征工程算法: 维护一个完整的元素周期表给所有化学元素增加权重(e.g. 铁/Fe)。
  
  1.2 TOP-K匹配策略：BM25算法会针对query question对于语料库所有文本进行匹配度打分，只要TOP-K匹配结果列表中有query question预期的正确章节则判断为匹配正确。

  <img width="1000" height="600" alt="bm25_chem_weight_optimization" src="https://github.com/user-attachments/assets/ecefc7ac-af7f-4b24-aee4-8e260dcfaaf7" />

  <img width="1400" height="700" alt="bm25_optimal_chem_weight_top3" src="https://github.com/user-attachments/assets/d1a7f86b-8a34-4c93-b1b6-7af6acabd830" />

  <img width="1400" height="700" alt="bm25_optimal_chem_weight_top5" src="https://github.com/user-attachments/assets/20ecc3cb-074c-4530-a7a1-789eb3ab7c67" />

  <img width="1400" height="700" alt="bm25_optimal_chem_weight_top7" src="https://github.com/user-attachments/assets/a46d8b53-d8ed-4d04-93a4-a89c9dc08a84" />
   
2. hybrid_search.py
3. waterfall_search.py
