import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, roc_auc_score

class Tradaboost(object):##针对二分类设计的tradaboost
    def __init__(self,N=None,base_estimator=None,threshold:float=None,topN:bool = None,score=roc_auc_score):    
        self.N=N
        self.threshold=threshold
        self.topN = topN
        self.base_estimator=base_estimator
        self.score=score
        self.weights = []
        self.estimators= []
        self.beta_all = np.zeros([1, self.N])
            
    # 权重的标准化，其实差别不大，在前面的例子中也有用到过
    def _calculate_weights(self,weights):      
        total = np.sum(weights)      
        return np.asarray(weights / total, order='C')      
          

          
    #计算目标域上的错误率     
    def _calculate_error_rate(self,y_true, y_pred, weight):      
        total = np.sum(weight)      
        return np.sum(weight[:, 0] / total * np.abs(y_true - y_pred))      
          
    #根据逻辑回归输出的score的得到标签，注意这里不能用predict直接输出标签      
   
          
    def fit(self,source,target,source_label,target_label,early_stopping_rounds):#注意，输入要转为numpy格式的
        '''
        Source: 待迁移数据
        target: 目标数据

        return :
        更新类中的权重数据
        '''
        
        source_shape=source.shape[0]
        target_shape=target.shape[0]
        trans_data = np.concatenate((source, target), axis=0)      
        trans_label = np.concatenate((source_label,target_label), axis=0)      
        weights_source = np.ones([source_shape, 1])/source_shape      
        weights_target = np.ones([target_shape, 1])/target_shape
        weights = np.concatenate((weights_source, weights_target), axis=0)
        
        # 输出统计信息:
        N_positive = sum((target_label>0).astype(int))
        print(f'''Statistic info:
        Size of the source data : {source_shape}
        Size of the target data : {target_shape}
        Target Npositive:{N_positive}
	''')
        # 根据公式初始化参数，具体可见原文
        
        bata = 1 / (1 + np.sqrt(2 * np.log(source_shape / self.N)))    
        result_label = np.ones([source_shape+target_shape, self.N])    

        trans_data = np.asarray(trans_data, order='C')     #行优先 
        trans_label = np.asarray(trans_label, order='C')     
        
        score=0
        flag=0
        
        for i in range(self.N):      
            P = self._calculate_weights(weights)      #权重的标准化
            self.base_estimator.fit(trans_data,trans_label,P*100)#这里xgb有bug，，如果权重系数太小貌似是被忽略掉了？
            self.estimators.append(self.base_estimator)
            y_preds=self.base_estimator.predict_proba(trans_data)[:,1] #全量数据的预测
            result_label[:, i]=y_preds #保存全量数据的预测结果用于后面的各个模型的评价
             

            #注意，仅仅计算在目标域上的错误率 ，
            if self.threshold is not None or  self.topN is True:
                y_target_pred=self.base_estimator.predict_proba(target)[:,1]#目标域的预测
                if self.topN is True:
                    # 使用topN 进行过滤
                    rank = pd.DataFrame(y_target_pred,columns=['score'])
                    rank['rank'] = rank.score.rank(ascending=False)
                    rank['Y'] = 0
                    rank.loc[rank['rank']<N_positive,'Y'] = 1
                    error_rate = self._calculate_error_rate(target_label, rank['Y'].values,  \
                                              weights[source_shape:source_shape + target_shape, :])  
                    
                elif self.threshold is not None:
                    # 使用阈值进行过滤
                    error_rate = self._calculate_error_rate(target_label, (y_target_pred>self.threshold).astype(int),  \
                                              weights[source_shape:source_shape + target_shape, :])  
            else:
                # 直接使用predict进行处理
                y_target_pred=self.base_estimator.predict(target)#目标域的预测
                error_rate = self._calculate_error_rate(target_label, y_target_pred,  \
                                              weights[source_shape:source_shape + target_shape, :])  
            #根据不同的判断阈值来对二分类的标签进行判断，对于不均衡的数据集合很有效，比如100：1的数据集，不设置class_wegiht
            #的情况下需要将正负样本的阈值提高到99%.
            
            # 防止过拟合     
            if error_rate > 0.5:      
                print (f'Error_rate is {error_rate}, scale it !')
                error_rate = 0.5      
            if error_rate == 0:      
                N = i      
                break       

            self.beta_all[0, i] = error_rate / (1 - error_rate)      

            # 调整目标域样本权重      
            for j in range(target_shape):      
                weights[source_shape + j] = weights[source_shape + j] * \
                np.power(self.beta_all[0, i],(-np.abs(result_label[source_shape + j, i] - target_label[j])))

                
            # 调整源域样本权重      
            for j in range(source_shape):      
                weights[j] = weights[j] * np.power(bata,np.abs(result_label[j, i] - source_label[j]))
                
            # 只在目标域上进行 AUC 的评估
            tp=self.score(target_label,y_target_pred)
            print('The '+str(i)+' rounds score is '+str(tp),f"Error Rate is {error_rate}")
            if tp > score :      
                score = tp      
                best_round = i  
                flag=0
                self.best_round=best_round
                self.best_score=score
                self.weights = weights
                print('Get a valid weight , updating ...')
            else:
                flag+=1

            if flag >=early_stopping_rounds:  
                print('early stop!')
                break  
    
        
    # 预测代码 -- proba 类别
    def predict_prob(self, x_test):
        result = np.ones([x_test.shape[0], self.N + 1])
        predict = []

        i = 0
        for estimator in self.estimators:
            # 修改，这里采用proba作为预测器，并使用输出为1的作为结果
            y_pred = estimator.predict_proba(x_test)[:,1]
            result[:, i] = y_pred
            i += 1

        for i in range(x_test.shape[0]):
            left = np.sum(result[i, int(np.ceil(self.N / 2)): self.N] *
                         np.log(1 / self.beta_all[0, int(np.ceil(self.N / 2)):self.N]) )

            right = 0.5 * np.sum( np.log(1 / self.beta_all[0, int(np.ceil(self.N / 2)): self.N]) )
            predict.append([left, right])
        return predict
