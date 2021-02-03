# Incremental Learning

+ **Incremental On-line Learning: A Review and Comparison of State of the Art Algorithms**
> 增量学习中假设当前模型的构建仅基于前一个模型和最近多次的样本，文中采用了八种具有代表性的方法实现增量学习，分别采用离线和在线两种环境下进行对比测试模型的预测准确率和复杂度，且分别采用全部训练样本和部分训练样本进行超参数优化，对比发现不同超参数优化的方法获得结果相近，即超参数优化对样本数目具有鲁棒性。当出现概念漂移时，大部分算法的结果有明显的下滑，采用遗忘机制可以较好地适应概念漂移。