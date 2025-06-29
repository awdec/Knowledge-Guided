#          Medicine-Knowledge-Guided

## 项目介绍：

​	利用文本（Text）+ 图像（Image）+ 知识图谱（KG）三模态信息，对样本进行分类，验证在多模态融合下，使用 K 引导图谱嵌入算法对分类任务的效果提升。以及对比不同的图谱嵌入算法（TranSE 和 RotaTE）对模型效果的影响。

​	数据集共使用了 397 份 NACC 和 439 份 ADNI。分别使用了朴素三模态方案、TransE K 引导三模态方案、RotatE K 引导三模态方案对 NACC 和 ADNI 进行测试。

​	**提取特征：**

   将性别、婚姻状态、族裔、教育程度等类别特征转为数值型。使用相同的编码器处理 train 和 test，避免数据偏差。提取 EHR（电子病历）特征：以 EHR_ 开头的列。

​	提取 Demographic 特征：之前编码过的类别列。

**构建数据集：**

  把数据从 DataFrame 转换为 PyTorch 张量，以供模型训练使用。使用 DataLoader 构建批量训练与测试数据集。

**定义多模态融合模型：**

  使用 KGMultiModalTransformer 接收 EHR、图像特征、知识图谱嵌入、Demographic 将多个模态数据融合。

**K** **引导：**

   通过 RotatEextract 和 get_embeddings 函数提取每个样本对应的知识图谱嵌入。将知识图谱嵌入与临床和图像数据特征拼接，输入到多模态 Transformer 模型中。模型训练和预测时，会被引导去利用来自知识图谱的结构化语义信息，从而提升预测的表现和鲁棒性。

**TransE**：

   TransE（Translation Embedding）是一个用于知识图谱（Knowledge Graph）嵌入的模型，它的核心思想是将实体（Entities）和关系（Relations）嵌入到同一个低维向量空间中，并通过简单的线性运算来建模实体之间的关系。 

   在知识图谱中，知识通常表示为三元组 $(h,r,t)$：h：头实体（head entity）、r：关系（relation）、t：尾实体（tail entity）TransE 模型的假设是如果三元组  是成立的，那么在向量空间中，头实体向量加上关系向量应该接近尾实体向量。

   TransE 的优点是简单直观，易于实现。参数少，训练效率高。适合建模一对一关系（如国家-首都）。同时 TransE 的局限性，不能很好地处理一对多、多对一、多对多的关系。例如，对于“国家-语言”这种一对多关系，TransE 无法很好地区分多个尾实体。线性假设太强，不能建模复杂关系（如对称关系、反对称关系等）。所有实体和关系都在同一个空间中，限制了表达能力。

**RotatE**：

   RotatE 是一种用于知识图谱嵌入的模型，它是对 TransE 类方法的一种重要改进，用于更有效地建模实体之间复杂的关系类型，尤其是对称、反对称、可逆和组合性关系。

   RotatE 的基本假设是：给定一个三元组 $(h,r,t)$，RotatE 把实体 h 和 t 映射到复数空间中的向量，把关系 r 表示为一个复数单位模长的旋转向量，使得：t ≈ h·r，其中  是复数向量，r 是模长为1的复数向量。这种做法等价于在复数平面中对每个维度上的实体向量 h 做一个旋转，从而得到 t。

   RotatE 的优点，能够建模多种复杂关系（对称、反对称、可逆、传递等）。比 TransE 更有表达能力，用复数旋转替代向量平移，结构更清晰，具有更强的理论基础。

**训练模型：**

  优化器：Adam，适用于复杂深度模型。

  损失函数：交叉熵函数（适用于多分类）。

  前向传播 → 计算损失 → 反向传播 → 参数更新，

  进行 300 轮训练。

**模型评估：**

​	计算Accuracy、Recall、F1 Score、Precision、AUC-ROC。

​	绘制 Loss 曲线，ROC 曲线。

## 项目结构：

codeADNI/code：ADNI 模型

codeNACC/code：NACC 模型

## 数据集：

kaggle：https://www.kaggle.com/datasets/largerice16pro/medicine-knowledge-guided

## 联系方式：

QQ：1120571672

邮箱：1120571672@qq.com