def kennard_stone_split(X, n_train):
    """
    使用Kennard-Stone算法将数据集划分为训练集和测试集。
    
    参数：
    X: 特征矩阵，每行表示一个样本，每列表示一个特征。
    n_train: 训练集的大小。
    
    返回值：
    train_indices: 训练集样本的索引。
    test_indices: 测试集样本的索引。
    """
    n_samples = X.shape[0]
    
    # 计算距离矩阵
    distances = np.zeros((n_samples, n_samples))
    for i in range(n_samples):
        for j in range(i+1, n_samples):
            distances[i, j] = np.linalg.norm(X[i] - X[j])
            distances[j, i] = distances[i, j]
    
    # 初始化训练集和测试集
    train_indices = []
    test_indices = []
    
    # 随机选择第一个训练样本
    train_index = np.random.randint(0, n_samples)
    train_indices.append(train_index)
    
    # 选择剩余样本中与第一个训练样本距离最远的样本作为第一个测试样本
    distances_to_train = distances[train_index, :]
    test_index = np.argmax(distances_to_train)
    test_indices.append(test_index)
    
    # 从剩余样本中选择训练样本和测试样本，直到满足训练集大小的要求
    while len(train_indices) < n_train:
        distances_to_train = np.min(distances[test_indices, :], axis=0)
        train_index = np.argmax(distances_to_train)
        train_indices.append(train_index)
        
        distances_to_train = distances[train_index, :]
        test_index = np.argmax(distances_to_train)
        test_indices.append(test_index)
    
    return train_indices, test_indices
