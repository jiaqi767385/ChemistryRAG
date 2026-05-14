1. bm25_search.py

实验策略：遍历无机化学评测习题集中的1203道题目，在corpus chunks语料库中检索TOP K最匹配的chunk，并将匹配到的chuck与预期chuck进行逻辑对齐。
策略重点：
  1.1 特征工程算法: 维护一个完整的元素周期表给所有化学元素增加权重(e.g. 铁/Fe)。
  1.2 TOP-K匹配策略：BM25算法会针对query question对于语料库所有文本进行匹配度打分，只要TOP-K匹配结果列表中有query question预期的正确章节则判断为匹配正确。

<img width="1000" height="600" alt="Figure_1" src="https://github.com/user-attachments/assets/632b6569-da56-4543-a24c-55415dba7001" />
<img width="1400" height="700" alt="Figure_5" src="https://github.com/user-attachments/assets/ec0f0f50-bef2-4023-9de6-fc48affdeea7" />
<img width="1400" height="700" alt="Figure_2" src="https://github.com/user-attachments/assets/6a66fd2f-9a56-4684-85c2-67a798bbac18" />
<img width="1400" height="700" alt="Figure_3" src="https://github.com/user-attachments/assets/ec17d2c0-debf-4532-841f-86fe7eee7a4f" />



   
3. hybrid_search.py
4. waterfall_search.py
