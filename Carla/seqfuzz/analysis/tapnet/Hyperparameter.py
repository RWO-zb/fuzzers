class Hyperparameter:
   
    # [修改] 预训练权重对应 Step=33, Dimension=17
    Step = 33 
    Dimension = 17 
    
    nclass = 2 
    bench_noCrash = []
    bench_crash = []

# 初始化 (保持在类外部)
Hyperparameter.bench_noCrash = [[0] * Hyperparameter.Step for _ in range(Hyperparameter.Dimension)]
Hyperparameter.bench_crash = [[1] * Hyperparameter.Step for _ in range(Hyperparameter.Dimension)]