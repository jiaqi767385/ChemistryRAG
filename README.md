1. bm25_search.py

实验策略：遍历无机化学评测习题集中的1203道题目，在corpus chunks语料库中检索TOP K最匹配的chunk，并将匹配到的chuck与预期chuck进行逻辑对齐。

策略重点：

  1.1 特征工程算法: 维护一个完整的元素周期表给所有化学元素增加权重(e.g. 铁/Fe)。
  
  1.2 TOP-K匹配策略：BM25算法会针对query question对于语料库所有文本进行匹配度打分，只要TOP-K匹配结果列表中有query question预期的正确章节则判断为匹配正确。





   
3. hybrid_search.py
4. waterfall_search.py
