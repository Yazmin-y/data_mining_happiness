import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import font_manager
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from xgboost import XGBClassifier
from KNN import KNN
import time
import graphviz

sns.set()
font_manager.fontManager.addfont('./SimHei.ttf')
plt.rcParams['font.sans-serif'] = 'SimHei'
plt.rcParams['axes.unicode_minus'] = False


def formDict(x):
    dictionary = {}
    x1 = x.split(';')
    for i in x1:
        if '=' in i:
            v, k = i.split('=')
            dictionary[int(v)] = k.strip()
    return dictionary


def plot_feature_importance(model):
    p = plt.barh(range(len(model.feature_importances_)), model.feature_importances_, tick_label=columns_cn)
    plt.bar_label(p)
    plt.xlabel("Feature importance")
    plt.ylabel("Feature")
    plt.title('决策树特征重要性')
    plt.tight_layout()
    plt.show()


def plot_feature_importance_2(model1, model2):
    height = 0.5
    y = np.arange(len(model1.feature_importances_))
    axes1 = plt.subplot(1, 1, 1)
    axes1.barh(y - height / 2, model1.feature_importances_, label='未调参', height=height, tick_label=columns_cn)
    axes1.barh(y + height / 2, model2.feature_importances_, label='调参后', height=height)
    axes1.legend()
    axes1.set_xlabel("Feature importance")
    axes1.set_ylabel("Feature")
    axes1.set_title('调参前后特征重要性对比')
    plt.tight_layout()
    plt.show()


def Tuning(cv_params, other_params, x_train_array, y_train_):
    model2 = XGBClassifier(**other_params)
    optimized_GBM = GridSearchCV(estimator=model2,
                                 param_grid=cv_params,
                                 scoring='accuracy',
                                 cv=5,
                                 n_jobs=-1)
    optimized_GBM.fit(x_train_array, y_train_)
    evalute_result = optimized_GBM.cv_results_['mean_test_score']
    # print('每轮迭代运行结果:{0}'.format(evalute_result))
    print('参数的最佳取值：{0}'.format(optimized_GBM.best_params_))
    print('最佳模型得分:{0}'.format(optimized_GBM.best_score_))
    return optimized_GBM


train_data = pd.read_csv("happiness_train_abbr.csv", index_col='id')
# 查看数据整体情况
print(train_data.head(5))
print('data description:')
print(train_data.describe())  # 数据统计性描述
print('data info:')
print(train_data.info())  # work_status等存在异常
print(pd.value_counts(train_data.work_status))
print(train_data.work_status.unique())  # 存在大量空值

# 空值处理
train_data.drop(['work_status', 'work_yr', 'work_type', 'work_manage'], axis=1, inplace=True)
train_data = train_data.replace(-8, np.nan).dropna(how='any')  # 去除负值
print('data description after preprocessing:')
print(train_data.happiness.describe())

# 数据可视化
vis_data = train_data.copy()
happiness_dict = {1: '非常不幸福', 2: '比较不幸福', 3: '说不清楚', 4: '比较幸福', 5: '非常幸福'}
happiness = vis_data.happiness.map(lambda x: happiness_dict[x])
h_cnt = happiness.value_counts()
h_cnt.name = '幸福程度'
h_cnt.plot.pie(figsize=(6, 6), autopct='%.2f')
plt.tight_layout()
# f = plt.gcf()  # 获取当前图像
# f.savefig(r'./DataImage/pieChart.png')
# f.clear()
plt.show()

dicts = pd.read_excel('happiness_index.xlsx')
cla_item = [1, 2, 6, 8, 9, 10, 11, 13, 17, 19, 20, 21, 22, 23, 24, 26, 29, 31, 32, 33, 34, 35, 36]  # 要做分类的列
reg_item = [7, 12, 14, 15, 16, 18, 25, 27, 28, 30]  # 要做回归的列
columns_cn = ['', '类型', '省份', '城市', '区县', '问卷时间', '性别', '出生年', '民族', '信教', '宗教活动',
              '教育程度', '收入', '政治面貌', '房屋面积', '身高', '体重', '健康程度', '健康影响', '感到沮丧', '户口',
              '社交', '休息', '学习充电', '感到公平', '处于等级', '工作经历', '家庭收入', '家人数量', '家境状况', '房产数量', '小汽车', '婚姻状况', '与同龄相比地位',
              '与三年前比地位', '与大众看法', '收入合理情况']  # 各列的中文名
i_cla_cn = vis_data.columns[cla_item]
i_reg_cn = vis_data.columns[reg_item]
vis_data['幸福程度'] = happiness

for i, v in enumerate(i_cla_cn):
    d = dicts[dicts['变量名'] == v]['取值含义'].values[0]
    col_cn = columns_cn[cla_item[i]]
    choose_cn = formDict(d)
    to_cn = vis_data[v].map(lambda x: choose_cn[x])
    vis_data[col_cn] = to_cn.values.tolist()

for i, v in enumerate(i_reg_cn):
    col_cn = columns_cn[reg_item[i]]
    vis_data[col_cn] = vis_data[v].values.tolist()

vis_data.drop(vis_data.columns[range(1, 37)], axis=1, inplace=True)

for icol in range(2, 35):
    cols = vis_data.columns
    colp = cols[icol]
    if columns_cn.index(colp) in cla_item:
        print(colp + '的幸福分直方分布图')
        g1 = sns.FacetGrid(vis_data, col=colp, col_wrap=4, sharey=False, sharex=False)
        g1 = g1.map(plt.hist, cols[0])
        g1.set(ylabel='频数', xlabel='幸福分')
        hs = vis_data.groupby(colp)[cols[0]].mean().sort_values().round(3)
        hs.name = '平均幸福度'
        plt.tight_layout()
        # f = plt.gcf()  # 获取当前图像
        # f.savefig(r'./DataImage/{}.png'.format(icol))
        # f.clear()
        plt.show()
        hs.plot(kind='bar', legend=True)
        plt.tight_layout()
        # f = plt.gcf()  # 获取当前图像
        # f.savefig(r'./DataImage/mean{}.png'.format(icol))
        # f.clear()
        plt.show()
    else:
        print(colp + '的幸福分盒图')
        g2 = sns.boxplot(x=cols[1], y=colp, data=vis_data)
        pltrange = (vis_data[colp].quantile(0.05), vis_data[colp].quantile(0.9))
        plt.ylim(pltrange)
        plt.tight_layout()
        # f = plt.gcf()  # 获取当前图像
        # f.savefig(r'./DataImage/box{}.png'.format(icol))
        # f.clear()
        plt.show()

# 分类任务
train_data.drop(['survey_time'], axis=1, inplace=True)
columns_cn.remove('问卷时间')
columns_cn.remove('')
train_X, test_X, train_y, test_y = train_test_split(train_data.iloc[:, 1:], train_data['happiness'], test_size=0.2)

print('Before PCA, the shape of training data: ' + str(train_X.shape))
ss = StandardScaler()
pca = PCA(n_components=0.99, copy=True)
pca_trainX = pca.fit_transform(train_X)
pca_testX = pca.fit_transform(test_X)
pca_trainX = ss.fit_transform(pca_trainX)
pca_testX = ss.fit_transform(pca_testX)
print('After PCA, the shape of training data: ' + str(pca_trainX.shape))

times = []
time2 = []

# 决策树
start = time.perf_counter_ns()
decision_tree = DecisionTreeClassifier(criterion="entropy", class_weight='balanced', splitter='random')
decision_tree.fit(train_X, train_y)
y_pred1 = decision_tree.predict(test_X)
# decision_tree.fit(pca_trainX, train_y)
# y_pred1 = decision_tree.predict(pca_testX)
end = time.perf_counter_ns()
t = end - start
times.append(t)
print('采用决策树进行分类')
print('用时：%.2f ns' % t)
print('accuracy = %.2f' % accuracy_score(test_y, y_pred1))
print(classification_report(test_y, y_pred1, zero_division=1))

confusion_matrix_result = confusion_matrix(test_y, y_pred1)
print('The confusion matrix result of Decision Tree:\n', confusion_matrix_result)
plt.figure(figsize=(8, 6))
sns.heatmap(confusion_matrix_result, annot=True, cmap='Blues')
plt.xlabel('Predicted labels')
plt.ylabel('True labels')
plt.title('决策树混淆矩阵热力图')
plt.show()

# dot_data = tree.export_graphviz(decision_tree, feature_names=columns_cn, class_names=['非常不幸福', '比较不幸福', '说不清楚', '比较幸福', '非常幸福'], filled=True, rounded=True)
# graph = graphviz.Source(dot_data)
# graph.view()

plot_feature_importance(decision_tree)

# SVM
start = time.perf_counter_ns()
svc = SVC(kernel='rbf', decision_function_shape='ovo')
# svc.fit(train_X, train_y)
# y_pred2 = svc.predict(test_X)
svc.fit(pca_trainX, train_y)
y_pred2 = svc.predict(pca_testX)

end = time.perf_counter_ns()
t = end - start
times.append(t)
print('采用RBF kernel SVM进行分类')
print('用时：%.2f ns' % t)
print('accuracy = %.2f' % accuracy_score(test_y, y_pred2))
print(classification_report(test_y, y_pred2, zero_division=1))

confusion_matrix_result = confusion_matrix(test_y, y_pred2)
print('The confusion matrix result of SVM:\n', confusion_matrix_result)
plt.figure(figsize=(8, 6))
sns.heatmap(confusion_matrix_result, annot=True, cmap='Blues')
plt.xlabel('Predicted labels')
plt.ylabel('True labels')
plt.title('SVM混淆矩阵热力图')
plt.show()

# 逻辑回归
start = time.perf_counter_ns()
logisitic = LogisticRegression(max_iter=5000, solver="newton-cg", multi_class='multinomial')
# logisitic.fit(train_X, train_y)
# y_pred3 = logisitic.predict(test_X)
logisitic.fit(pca_trainX, train_y)
y_pred3 = logisitic.predict(pca_testX)
end = time.perf_counter_ns()
t = end - start
times.append(t)
print('采用logisitic回归模型进行分类')
print('用时：%.2f ns' % t)
print('accuracy = %.2f' % accuracy_score(test_y, y_pred3))
print(classification_report(test_y, y_pred3, zero_division=1))

confusion_matrix_result = confusion_matrix(test_y, y_pred3)
print('The confusion matrix result of Logistic Regression:\n', confusion_matrix_result)
plt.figure(figsize=(8, 6))
sns.heatmap(confusion_matrix_result, annot=True, cmap='Blues')
plt.xlabel('Predicted labels')
plt.ylabel('True labels')
plt.title('逻辑回归混淆矩阵热力图')
plt.show()

# KNN
start = time.perf_counter_ns()
y_pred4 = KNN(pca_trainX, train_y, pca_testX, k=1)
end = time.perf_counter_ns()
t = end - start
times.append(t)
print('采用KNN进行分类')
print('用时：%.2f ns' % t)
print('accuracy = %.2f' % accuracy_score(test_y, y_pred4))
print(classification_report(test_y, y_pred4, zero_division=1))

confusion_matrix_result = confusion_matrix(test_y, y_pred4)
print('The confusion matrix result of KNN:\n', confusion_matrix_result)
plt.figure(figsize=(8, 6))
sns.heatmap(confusion_matrix_result, annot=True, cmap='Blues')
plt.xlabel('Predicted labels')
plt.ylabel('True labels')
plt.title('KNN混淆矩阵热力图')
plt.show()

# XGBoost
start = time.perf_counter_ns()
train_y = np.int64(train_y - 1)
test_y = np.int64(test_y - 1)
xgbc = XGBClassifier(objective='multi:softmax', use_label_encoder=False)
xgbc.fit(train_X, train_y)
y_pred5 = xgbc.predict(test_X)
end = time.perf_counter_ns()
t = end - start
times.append(t)
time2.append(t)
print('采用XGBoost进行分类')
print('用时：%.2f ns' % t)
print('accuracy = %.2f' % accuracy_score(test_y, y_pred5))
print(classification_report(test_y, y_pred5, zero_division=1))

confusion_matrix_result = confusion_matrix(test_y, y_pred5)
print('The confusion matrix result of Decision Tree:\n', confusion_matrix_result)
plt.figure(figsize=(8, 6))
sns.heatmap(confusion_matrix_result, annot=True, cmap='Blues')
plt.xlabel('Predicted labels')
plt.ylabel('True labels')
plt.title('XGBoost混淆矩阵热力图')
plt.show()

plot_feature_importance(xgbc)

# 时间对比可视化
name = ['决策树', 'SVM', '逻辑回归', 'KNN', 'XGBoost']
p = plt.barh(range(len(times)), times, tick_label=name)
plt.bar_label(p, label_type='edge')
plt.xlabel('time/ns')
plt.ylabel('算法')
plt.show()

# 对XGBoost调参
other_params = {
    'use_label_encoder': False,
    'booster': 'gbtree',
    'objective': 'multi:softmax',
    'num_class': 5,
    'learning_rate': 0.05,
    'max_depth': 5,
    'min_child_weight': 5,
    'colsample_bytree': 0.8,
    'subsample': 0.6,
    'reg_alpha': 0.5
}
cv_params = {
    # 'learning_rate': [0.01, 0.02, 0.05, 0.1, 0.15],
    # 'max_depth': [2, 3, 4, 5],
    # 'min_child_weight': [0, 2, 5, 10, 20],
    # 'subsample': [0.6, 0.7, 0.8, 0.85, 0.95],
    # 'colsample_bytree': [0.5, 0.6, 0.7, 0.8, 0.9],
    'reg_alpha': [0, 0.25, 0.5, 0.75, 1]
}
# opt = Tuning(cv_params, other_params, train_X, train_y)

# 调参后XGBoost
start = time.perf_counter_ns()
xgbc2 = XGBClassifier(**other_params)
xgbc2.fit(train_X, train_y)
y_pred5 = xgbc2.predict(test_X)
end = time.perf_counter_ns()
t = end - start
time2.append(t)
print('采用调参后的XGBoost进行分类')
print('用时：%.2f ns' % t)
print('accuracy = %.2f' % accuracy_score(test_y, y_pred5))
print(classification_report(test_y, y_pred5, zero_division=1))

confusion_matrix_result = confusion_matrix(test_y, y_pred5)
print('The confusion matrix result of Decision Tree:\n', confusion_matrix_result)
plt.figure(figsize=(8, 6))
sns.heatmap(confusion_matrix_result, annot=True, cmap='Blues')
plt.xlabel('Predicted labels')
plt.ylabel('True labels')
plt.title('XGBoost混淆矩阵热力图')
plt.show()

# 时间对比可视化
name = ['调参前', '调参后']
p = plt.barh(range(len(time2)), time2, tick_label=name, height=0.5)
plt.bar_label(p, label_type='edge')
plt.xlabel('time/ns')
plt.ylabel('XGBoost调参前后')
plt.show()

plot_feature_importance_2(xgbc, xgbc2)
