from .dl_models import *
from .dataset import *
from .utils import *
from sklearn.metrics import classification_report, accuracy_score
import os


# 实例化模型，设置loss，优化器等
def train(args, seed, x_train, x_test, y_train, y_test, num_classes):
    seed_everything(seed)

    if num_classes == 4:
        model = models[args.model](num_classes=num_classes).to(args.device)
    elif num_classes == 2:
        model = models[args.model1](num_classes=num_classes).to(args.device)
    elif num_classes == 3:
        model = models[args.model2](num_classes=num_classes).to(args.device)
    else:
        raise ValueError("num_classes参数错误")

    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    train_loss = []
    valid_loss = []
    train_epochs_loss = []
    valid_epochs_loss = []

    early_stopping = EarlyStopping(patience=args.patience, verbose=True)

    # 开始训练以及调整lr
    for epoch in range(args.epochs):
        model.train()
        train_dataloader, valid_dataloader = create_dataloader(x_train, x_test, y_train, y_test, args)
        train_epoch_loss = []
        for idx, (data_x, data_y) in enumerate(train_dataloader, 0):
            # 将data_y转换为独热向量编码
            data_y = np.array(data_y)
            data_y = np.eye(num_classes)[data_y]
            data_y = torch.from_numpy(data_y)

            data_x = data_x.to(torch.float32).to(args.device)
            data_y = data_y.to(torch.float32).to(args.device)
            data_x = data_x.unsqueeze(-1)
            outputs = model(data_x)
            optimizer.zero_grad()
            loss = criterion(data_y, outputs)
            loss.backward()
            optimizer.step()
            train_epoch_loss.append(loss.item())
            train_loss.append(loss.item())
            if idx % (len(train_dataloader) // 2) == 0:
                print("epoch={}/{},{}/{}of train, loss={}".format(
                    epoch, args.epochs, idx, len(train_dataloader), loss.item()))
        train_epochs_loss.append(np.average(train_epoch_loss))

        # =====================valid============================
        model.eval()
        valid_epoch_loss = []
        for idx, (data_x, data_y) in enumerate(valid_dataloader, 0):
            # 将data_y转换为独热向量编码
            data_y = np.array(data_y)
            data_y = np.eye(num_classes)[data_y]
            data_y = torch.from_numpy(data_y)

            data_x = data_x.to(torch.float32).to(args.device)
            data_y = data_y.to(torch.float32).to(args.device)
            data_x = data_x.unsqueeze(-1)
            outputs = model(data_x)
            loss = criterion(data_y, outputs)
            valid_epoch_loss.append(loss.item())
            valid_loss.append(loss.item())
        valid_epochs_loss.append(np.average(valid_epoch_loss))
        # ==================early stopping======================
        # 检查路径是否存在
        if not os.path.exists(args.ckpt_save):
            os.makedirs(args.ckpt_save)
        early_stopping(valid_epochs_loss[-1], model=model, path=args.ckpt_save)
        if early_stopping.early_stop:
            print("Early stopping")
            break
        # ====================adjust lr========================
        lr_adjust = {
            2: 5e-5, 4: 1e-5, 6: 5e-6, 8: 1e-6,
            10: 5e-7, 15: 1e-7, 20: 5e-8
        }
        if epoch in lr_adjust.keys():
            lr = lr_adjust[epoch]
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
            print('Updating learning rate to {}'.format(lr))

    if args.plot:
        plot(train_loss, train_epochs_loss, valid_epochs_loss)

    return model


def predict(model, x, device):
    model.eval()  # Set the model to evaluation mode
    predictions = []
    dataloader = create_predict_dataloader(x)

    with torch.no_grad():  # No need to track gradients for prediction
        for data in dataloader:
            data = data.to(torch.float32).to(device)  # Move data to the specified device
            data = data.unsqueeze(-1)
            outputs = model(data)  # Get the model output for the given data
            predicted = outputs.data  # Extract the data from the model output
            predicted = torch.argmax(predicted, dim=1)
            predictions.extend(predicted.cpu().numpy())  # Move the predictions to CPU and convert to numpy array
    return np.array(predictions)


class TrainerDl:
    def __init__(self, args, x_train, x_test, y_train, y_test, seed):
        # 初始化
        self.seed = seed
        self.args = args
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

    def train_once(self):
        # 打印model训练集各类别样本数量
        print("model训练集各类别样本数量:")
        print("0类: {0}".format(len(self.y_train[self.y_train == 0])))
        print("1类: {0}".format(len(self.y_train[self.y_train == 1])))
        print("2类: {0}".format(len(self.y_train[self.y_train == 2])))
        print("3类: {0}".format(len(self.y_train[self.y_train == 3])))
        print(f"训练模型: {self.args.model}")
        # 训练
        model = train(self.args,  self.seed, self.x_train, self.x_test, self.y_train, self.y_test, 4)
        # 预测
        y_pred = predict(model, self.x_test, device=self.args.device)

        # 计算精度
        accuracy = accuracy_score(self.y_test, y_pred)
        print(f"准确度: {accuracy}")

        # 打印分类报告（包括查准率、召回率和F1分数）
        # report = classification_report(self.y_test, y_pred)
        # print("分类报告:")
        # print(report)
        return model, self.args.model

    def train_stage_one(self):
        # 打印model1训练集各类别样本数量
        print("model1训练集各类别样本数量:")
        print("0类: {0}".format(len(self.y_train_stage_one[self.y_train_stage_one == 0])))
        print("非0类: {0}".format(len(self.y_train_stage_one[self.y_train_stage_one == 1])))
        print(f"训练模型: {self.args.model}")
        # 训练
        model1 = train(self.args, self.seed, self.x_train_stage_one, self.x_test_stage_one, self.y_train_stage_one,
                       self.y_test_stage_one, 2)
        # 预测
        y_pred_stage_one = predict(model1, self.x_test_stage_one, device=self.args.device)

        # 计算精度
        accuracy = accuracy_score(self.y_test_stage_one, y_pred_stage_one)
        print(f"准确度: {accuracy}")

        # 打印分类报告（包括查准率、召回率和F1分数）
        # report = classification_report(self.y_test_stage_one, y_pred_stage_one)
        # print("分类报告:")
        # print(report)
        return model1, self.args.model1

    def train_stage_two(self):
        # 打印model2训练集各类别样本数量
        print("model2训练集各类别样本数量:")
        print("1类: {0}".format(len(self.y_train_stage_two[self.y_train_stage_two == 0])))
        print("2类: {0}".format(len(self.y_train_stage_two[self.y_train_stage_two == 1])))
        print("3类: {0}".format(len(self.y_train_stage_two[self.y_train_stage_two == 2])))
        print(f"训练模型: {self.args.model}")
        # 训练
        model2 = train(self.args, self.seed, self.x_train_stage_two, self.x_test_stage_two, self.y_train_stage_two,
                       self.y_test_stage_two, 3)
        # 预测
        y_pred_stage_two = predict(model2, self.x_test_stage_two, device=self.args.device)

        # 计算精度
        accuracy = accuracy_score(self.y_test_stage_two, y_pred_stage_two)
        print(f"准确度: {accuracy}")

        # 打印分类报告（包括查准率、召回率和F1分数）
        # report = classification_report(self.y_test_stage_two, y_pred_stage_two)
        # print("分类报告:")
        # print(report)
        return model2, self.args.model2

    def val(self, x_test, y_test, **kwargs):
        # 从kwargs中提取模型
        model = kwargs.get('model')
        model1 = kwargs.get('model1')
        model2 = kwargs.get('model2')

        # 检查并执行model的预测
        if model:
            y_pred = predict(model, x_test, device=self.args.device)

            # 计算精度和打印报告
            accuracy = accuracy_score(y_test, y_pred)
            print(f"单模型准确度: {accuracy}")
            report = classification_report(y_test, y_pred)
            print("单模型分类报告:")
            print(report)

        # 检查并执行model1和model2的预测
        if model1 and model2:
            # 使用model1对测试集所有样本进行初步预测
            y_pred_initial = predict(model1, x_test, device=self.args.device)

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
                y_pred_model2 = predict(model2, x_test_model2, device=self.args.device)

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

