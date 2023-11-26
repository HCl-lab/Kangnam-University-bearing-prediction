import numpy as np
from sklearn.metrics import classification_report, accuracy_score
from .ml_models import models

def adjust_labels_for_stage_one(y):
    """ 将非0类标签修改为1 """
    return np.array([0 if label == 0 else 1 for label in y])


def adjust_labels_for_stage_two(y):
    """ 移除0类标签，保留1、2、3类标签 """
    return y[y != 0]


class TrainerMl:
    def __init__(self, x_train, x_test, y_train, y_test, models):
        # 四分类模型
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test
        # 一阶段：区分0类和非0类
        self.x_train_stage_one = x_train
        self.y_train_stage_one = adjust_labels_for_stage_one(y_train)
        self.x_test_stage_one = x_test
        self.y_test_stage_one = adjust_labels_for_stage_one(y_test)
        # 二阶段：在非0类中区分1、2、3类
        self.x_train_stage_two = x_train[y_train != 0]
        self.y_train_stage_two = adjust_labels_for_stage_two(y_train)
        self.x_test_stage_two = x_test[y_test != 0]
        self.y_test_stage_two = adjust_labels_for_stage_two(y_test)
        # 初始化
        self.models = models

    def train_once(self):
        max_accuracy = 0
        best_model = None
        model_name = None
        # 打印model训练集各类别样本数量
        print("model训练集各类别样本数量:")
        print("0类: {0}".format(len(self.y_train[self.y_train == 0])))
        print("1类: {0}".format(len(self.y_train[self.y_train == 1])))
        print("2类: {0}".format(len(self.y_train[self.y_train == 2])))
        print("3类: {0}".format(len(self.y_train[self.y_train == 3])))
        for name, model in models.items():
            print(f"训练模型: {name}")
            model.fit(self.x_train, self.y_train)

            # 预测
            y_pred = model.predict(self.x_test)

            # 计算精度
            accuracy = accuracy_score(self.y_test, y_pred)
            print(f"准确度: {accuracy}")

            # 打印分类报告（包括查准率、召回率和F1分数）
            # report = classification_report(self.y_test, y_pred)
            # print("分类报告:")
            # print(report)
            if max_accuracy < accuracy:
                max_accuracy = accuracy
                best_model = model
                model_name = name
            else:
                continue

        return best_model, model_name

    def train_stage_one(self):
        max_accuracy = 0
        best_model_one_stage = None
        model_name = None
        # 打印model1训练集各类别样本数量
        print("model1训练集各类别样本数量:")
        print("0类: {0}".format(len(self.y_train_stage_one[self.y_train_stage_one == 0])))
        print("非0类: {0}".format(len(self.y_train_stage_one[self.y_train_stage_one == 1])))
        for name, model in models.items():
            print(f"训练模型: {name}")
            model.fit(self.x_train_stage_one, self.y_train_stage_one)

            # 预测
            y_pred_stage_one = model.predict(self.x_test_stage_one)

            # 计算精度
            accuracy = accuracy_score(self.y_test_stage_one, y_pred_stage_one)
            print(f"准确度: {accuracy}")

            # 打印分类报告（包括查准率、召回率和F1分数）
            # report = classification_report(self.y_test_stage_one, y_pred_stage_one)
            # print("分类报告:")
            # print(report)
            if max_accuracy < accuracy:
                max_accuracy = accuracy
                best_model_one_stage = model
                model_name = name
            else:
                continue

        return best_model_one_stage, model_name

    def train_stage_two(self):
        max_accuracy = 0
        best_model_two_stage = None
        model_name = None
        # 打印model2训练集各类别样本数量
        print("model2训练集各类别样本数量:")
        print("1类: {0}".format(len(self.y_train_stage_two[self.y_train_stage_two == 1])))
        print("2类: {0}".format(len(self.y_train_stage_two[self.y_train_stage_two == 2])))
        print("3类: {0}".format(len(self.y_train_stage_two[self.y_train_stage_two == 3])))

        for name, model in models.items():
            print(f"训练模型: {name}")
            model.fit(self.x_train_stage_two, self.y_train_stage_two)

            # 预测
            y_pred_stage_two = model.predict(self.x_test_stage_two)

            # 计算精度
            accuracy = accuracy_score(self.y_test_stage_two, y_pred_stage_two)
            print(f"准确度: {accuracy}")

            # 打印分类报告（包括查准率、召回率和F1分数）
            # report = classification_report(self.y_test_stage_two, y_pred_stage_two)
            # print("分类报告:")
            # print(report)
            if max_accuracy < accuracy:
                max_accuracy = accuracy
                best_model_two_stage = model
                model_name = name
            else:
                continue

        return best_model_two_stage, model_name

    def val(self, x_test, y_test, **kwargs):
        # 从kwargs中提取模型
        model = kwargs.get('model')
        model1 = kwargs.get('model1')
        model2 = kwargs.get('model2')

        # 检查并执行model的预测
        if model:
            y_pred = model.predict(x_test)

            # 计算精度和打印报告
            accuracy = accuracy_score(y_test, y_pred)
            print(f"单模型准确度: {accuracy}")
            report = classification_report(y_test, y_pred)
            print("单模型分类报告:")
            print(report)

        # 检查并执行model1和model2的预测
        if model1 and model2:
            # 使用model1对测试集所有样本进行初步预测
            y_pred_initial = model1.predict(x_test)

            # 初始化最终的预测结果数组
            y_pred_final = np.zeros_like(y_test)

            # 将model1预测为0类的结果直接赋值给最终结果
            y_pred_final[y_pred_initial == 0] = 0

            # 对于model1预测为1类的样本，使用model2进行再次预测
            # 这里需要先提取model1预测为1类的样本
            x_test_model2 = x_test[y_pred_initial == 1]

            # 如果有model1预测为1类的样本存在，才进行model2的预测
            if len(x_test_model2) > 0:
                # 使用model2进行预测
                y_pred_model2 = model2.predict(x_test_model2)

                # 将model2的预测结果映射回最终结果
                # 这里的映射需要将model2的预测结果(0, 1, 2)映射为(1, 2, 3)
                y_pred_final[y_pred_initial == 1] = y_pred_model2 + 1

            # 计算精度和打印报告
            accuracy = accuracy_score(y_test, y_pred_final)
            print(f"联合模型准确度: {accuracy}")
            report = classification_report(y_test, y_pred_final)
            print("联合模型分类报告:")
            print(report)

        # 如果没有传入任何模型，则抛出错误
        if not model and not (model1 and model2):
            raise ValueError("没有正确提供模型")


