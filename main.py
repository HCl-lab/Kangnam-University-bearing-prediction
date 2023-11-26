from data_process import load_data
from machine_leaning.ml_trainer import TrainerMl
from machine_leaning.ml_models import models

import torch
from deep_learning.dl_trainer import TrainerDl

class Argparse:
    pass


def ml_train(path, seed):
    # 加载数据集
    print('----------Loading data...----------')
    x_train, x_test, y_train, y_test = load_data(path, seed)

    # 打印数据集信息
    print('----------Data information:----------')
    print("x_train:{0}".format(x_train.shape))
    print("x_test:{0}".format(x_test.shape))
    print("y_train:{0}".format(y_train.shape))
    print("y_test:{0}".format(y_test.shape))

    print('----------Training:----------')

    # 训练模型
    trainer = TrainerMl(x_train, x_test, y_train, y_test, models)
    print('----------Training once----------')
    best_model, best_model_name = trainer.train_once()
    print('----------Training stage one----------')
    best_model_one_stage, best_model_one_stage_name = trainer.train_stage_one()
    print('----------Training stage two----------')
    best_model_two_stage, best_model_two_stage_name = trainer.train_stage_two()

    # 验证模型
    print("|-----Once time model:{0}-----|".format(best_model_name))
    print("|---First model: {0}，Second model：{1}---|".format(best_model_one_stage_name, best_model_two_stage_name))
    print('----------Validating...----------')
    trainer.val(x_test, y_test, model=best_model, model1=best_model_one_stage, model2=best_model_two_stage)


def dl_train(path, seed):
    # 加载数据集
    print('----------Loading data...----------')
    x_train, x_test, y_train, y_test = load_data(path, seed)
    # 打印数据集信息
    print('----------Data information:----------')
    print("x_train:{0}".format(x_train.shape))
    print("x_test:{0}".format(x_test.shape))
    print("y_train:{0}".format(y_train.shape))
    print("y_test:{0}".format(y_test.shape))

    print('----------Training:----------')

    # 训练模型
    trainer = TrainerDl(args, x_train, x_test, y_train, y_test, seed)
    print('----------Training once----------')
    model, model_name = trainer.train_once()
    print('----------Training stage one----------')
    model_one_stage, model_one_stage_name = trainer.train_stage_one()
    print('----------Training stage two----------')
    model_two_stage, model_two_stage_name = trainer.train_stage_two()

    # 验证模型
    print("|-----Once time model:{0}-----|".format(model_name))
    print("|---First model: {0}，Second model：{1}---|".format(model_one_stage_name, model_two_stage_name))
    print('----------Validating...----------')
    trainer.val(x_test, y_test, model=model, model1=model_one_stage, model2=model_two_stage)


if __name__ == '__main__':
    train_set = 0  # 0: Machine Learning, 1: Deep Learning
    # ========机器学习参数========
    path = 'data'
    seed = 0
    # ========深度学习参数========
    args = Argparse()
    args.epochs, args.learning_rate, args.patience = [2, 0.001, 10]
    args.device, = [torch.device("cuda:0" if torch.cuda.is_available() else "cpu"), ]
    args.batch_size_train, args.batch_size_val = [16, 16]
    args.plot = True
    args.model, args.model1, args.model2 = ['Transformer', 'Transformer', 'Transformer']
    args.ckpt_save = 'deep_learning/checkpoint'

    if train_set == 0:
        ml_train(path, seed)

    if train_set == 1:
        dl_train(path, seed)










