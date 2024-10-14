# Grammar_Correction: 尝试使用监督式微调探索开源大语言模型进行中文语法纠错

# 介绍

欢迎来到 Grammar_Correction 的代码仓库！

这是一个基于 [NLPCC 2023 共享任务1](http://tcci.ccf.org.cn/conference/2023/taskdata.php) 的中文文本纠错小项目。我们尝试使用该任务提供的数据集来评估我们的模型，希望能为中文语法纠错的研究做出一点贡献。

目前我们已经发布了以下内容:
* 用于训练的1000条[数据](./pseudo_data/instruction.json)，其中65%由ChatGPT生成，剩下的是我们辛苦标注的。
* 用于训练和推理的代码（希望对你有帮助）。


# 💭 概述
Grammar_Correction 是我们尝试探索开源大语言模型在中文语法纠错方面潜力的一个小项目。我们主要是利用了ChatGPT生成的数据和人工标注的混合数据集。对于那些有明显线索的语法错误，我们想出了一个小技巧，就是给ChatGPT一些提示来引导它生成不太合语法的句子。至于那些没有明显线索的语法错误，我们只能从网上收集一些不太合语法的句子，然后自己动手修改。另外，我们还试了一个叫"错误不变增强"的方法，希望能提高模型纠正中文语法错误的能力。


# 📚 混合数据集的构建
-
这个表格列出了中文母语者常见的六种主要语法错误类型，我们把它们分成两类：有线索(w/)和无线索(w/o)。有趣的是，这些错误的句子看起来很流畅，完全符合中国人的说话习惯。但是，它们其实不太符合中文语法规范，这就让纠错变得更有挑战性了。我们分别用ChatGPT生成的数据和人工标注的数据来处理这两类语法错误。

## ChatGPT生成的数据
有线索的语法错误其实挺好玩的，因为只要看到特定的线索就能发现和纠正。比如，**超过**和**大约**一起用就会显得有点啰嗦，**原因是**和**造成的**一起用会让句子结构有点乱，**促进**和**步伐**一起用听起来就怪怪的。反过来想，我们是不是也可以在正确的句子里加入这些线索，就能造出不太合语法的句子呢？我们就试着从网上找了一些[这样的线索](https://wenku.baidu.com/view/1ce351635727a5e9846a610e?aggId=e4e228d30166f5335a8102d276a20029bc646366&fr=catalogMain_text_ernie_recall_v1%3Awk_recommend_main_graph&_wkts_=1686039387317&bdQuery=%E5%86%97%E4%BD%99%E7%97%85%E5%8F%A5%E7%BB%83%E4%B9%A0)，然后让ChatGPT按照我们的想法生成一些不太合语法的句子。

## 人工标注的数据
对于那些没有明显线索的语法错误，我们只能自己动手丰衣足食了。我们从一些公开网站[1](https://wenku.baidu.com/view/1ce351635727a5e9846a610e?aggId=e4e228d30166f5335a8102d276a20029bc646366&fr=catalogMain_text_ernie_recall_v1%3Awk_recommend_main_graph&_wkts_=1686039387317&bdQuery=%E5%86%97%E4%BD%99%E7%97%85%E5%8F%A5%E7%BB%83%E4%B9%A0) [2](https://baijiahao.baidu.com/s?id=1675817725570818147&wfr=spider&for=pc) [3](https://easylearn.baidu.com/edu-page/tiangong/exercisedetail?id=174470eef8c75fbfc77db25d&from=search-duoti_pc-xiti_Detail_pc) [4](http://bj.xdf.cn/zhongkao/chuer/zhidao/134300.html) [5](http://bj.xdf.cn/zhongkao/chuer/zhidao/134299.html) [6](https://www.yueyeche.com.cn/zhjx/202207/19911.html) [7](https://mp.weixin.qq.com/s?__biz=MzI0NzE5NDI2MA==&mid=2652204429&idx=2&sn=6db3a396e1f1da2a56185917e8459d71&chksm=f2527a76c525f3600808e041222a6a78a49817314ad69603ab48129d31492a60b6920c8ac736&scene=27) [8](https://mp.weixin.qq.com/s?__biz=MzUzMDQ2MTM4OQ==&mid=2247557713&idx=4&sn=50caf0d739fd625a277e0d88fd97e1e8&chksm=fa52c5f3cd254ce57609af3da2a21e6fd0c7cdbb45d6a41cb3168c0e7e57b23b825508433d6e&scene=27) [9](https://wenku.baidu.com/view/5c9798cd961ea76e58fafab069dc5022aaea46f2.html?fr=aladdin664466&ind=3&_wkts_=1686039743632&bdQuery=%E5%8F%A5%E5%BC%8F%E6%9D%82%E7%B3%85) [10](https://zhuanlan.zhihu.com/p/479275444) [11](https://www.zszzs.com/wendang/qitafanwen/54091.html) [12](https://mp.weixin.qq.com/s?__biz=MzU4NTc3MzkwMw==&mid=2247500319&idx=3&sn=6ba362341e8f5543a8bb815e3a1657bd&chksm=fd87e43fcaf06d29a7486e45fa98215710987154fe9fcd58df33a4abf676699be2d44c293646&scene=27) [13](https://baijiahao.baidu.com/s?id=1742587369710610978&wfr=spider&for=pc) [14](https://mp.weixin.qq.com/s/DQnlXE_bKrSmTUVqTesqIg) [15](https://baijiahao.baidu.com/s?id=1617092703098480309&wfr=spider&for=pc) [16](https://www.renrendoc.com/paper/208183328.html) 收集了一些数据，然后自己动手标注。虽然有点辛苦，但是感觉挺有意思的。

## 错误不变增强
我们发现，中国人说话时的语法错误通常很微妙，很少会在人名、地名这些地方出错。所以我们想了个小办法，就是在平行数据中把这些命名实体换成类似的（我们用了[同义词](https://github.com/chatopera/Synonyms)这个工具）。


# 🚀 训练

我们使用了[ChatGLM-6B](https://github.com/THUDM/ChatGLM-6B)作为我们的基础模型，并采用了[P-Tuning v2](https://github.com/THUDM/P-tuning-v2)方法进行微调。这种方法不仅能有效地利用大型预训练模型的知识，还能在有限的计算资源下实现快速微调。

我们的训练过程如下：

1. 数据准备：我们使用了上述混合数据集，包括ChatGPT生成的数据和人工标注的数据。

2. 模型配置：我们使用了ChatGLM-6B模型，并应用了P-Tuning v2方法。

3. 训练参数：
   - 学习率：2e-5
   - 批次大小：64
   - 训练轮数：1
   - 最大序列长度：256

4. 训练过程：我们使用了单个GPU进行训练，整个过程大约持续了2小时。

5. 模型评估：我们使用了NLPCC 2023共享任务1提供的验证集进行模型评估。

通过这个训练过程，我们希望能够让模型更好地理解和纠正中文语法错误。
