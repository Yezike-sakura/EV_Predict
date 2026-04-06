import os
from tqdm import tqdm
import torch
import numpy as np
import utils
import pandas as pd
import math


def training(args, net, optim, loss_func, train_loader, valid_loader, fold):
    # valid_loss = 1000
    net.train()

    # 【纯UI修改】：把 tqdm 赋给一个变量，方便后面追加显示内容
    epoch_iterator = tqdm(range(args.epoch), desc='Training')

    # 👇【大一统架构升级】：根据参数决定是否挂载调度器
    scheduler = None
    if args.lradj == 'step':
        # 专门为 AGCRN 准备的阶梯衰减
        scheduler = torch.optim.lr_scheduler.StepLR(optim, step_size=20, gamma=0.5)
    elif args.lradj == 'plateau':
        #  策略一：动态监测衰减
        # 只要验证集 Loss 连续 5 轮 (patience=5) 不下降，就将学习率砍半 (factor=0.5)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optim, mode='min', factor=0.5, patience=5)

    elif args.lradj == 'cosine':
        #  策略二：带预热的余弦退火
        warmup_epochs = 5  # 前 5 轮用于预热

        def lr_lambda(epoch):
            if epoch < warmup_epochs:
                # 预热期：学习率从 0 线性爬升到 100%
                return (epoch + 1) / warmup_epochs
            else:
                # 余弦退火期：按照余弦曲线平滑下降到 0
                return 0.5 * (1 + math.cos(math.pi * (epoch - warmup_epochs) / (args.epoch - warmup_epochs)))

        # 使用 LambdaLR 将自定义的数学函数注入优化器
        scheduler = torch.optim.lr_scheduler.LambdaLR(optim, lr_lambda=lr_lambda)

    elif args.lradj == 'type1':
        # 预留给未来可能缝合的原作者旧代码
        pass

    # 【新增UI追踪器】：在训练开始前，记录初始的学习率
    previous_lr = optim.param_groups[0]['lr']
    # 👇【新增架构防御】：初始化全局最优验证集 Loss，初始设为正无穷
    best_valid_loss_avg = float('inf')
    # 👇【新增 1】：初始化早停耐心计数器
    patience = args.patience
    patience_counter = 0


    for epoch in epoch_iterator:
        train_loss_total = 0.0  # 【纯UI修改】：用于累加本轮训练 loss

        for j, data in enumerate(train_loader):
            '''
            occupancy = (batch, seq, node)
            add_tensor = (batch, seq, node)
            label = (batch, node)
            '''
            net.train()
            extra_feat = 'None'
            if args.add_feat != 'None':
                occupancy, label, extra_feat = data
            else:
                occupancy, label = data

            optim.zero_grad()
            predict = net(occupancy, extra_feat)

            # 🎯 彻底消灭原作者的 Broadcasting 维度爆炸 Bug
            loss = loss_func(predict.view_as(label), label)

            # if predict.shape != label.shape:
            #     loss = loss_func(predict.unsqueeze(-1), label)
            # else:
            #     loss = loss_func(predict, label)

            loss.backward() #反向传播
            optim.step()    #w，b更新

            train_loss_total += loss.item()  # 【纯UI修改】：单纯记录 loss 值用于展示

        # 【纯UI修改】：计算本轮平均训练 loss 用于展示
        train_loss_avg = train_loss_total / len(train_loader)

        # validation
        net.eval()
        valid_loss_total = 0.0  # 【纯UI修改】：用于累加本轮验证 loss

        for j, data in enumerate(valid_loader):
            '''
            occupancy = (batch, seq, node)
         = (batch, seq, node)
            label = (batch, node)
            '''
            net.train()  # 严格保留原代码
            extra_feat = 'None'
            if args.add_feat != 'None':
                occupancy, label, extra_feat = data
            else:
                occupancy, label = data

            predict = net(occupancy, extra_feat)
            if predict.shape != label.shape:
                loss = loss_func(predict.unsqueeze(-1), label)
            else:
                loss = loss_func(predict, label)

            valid_loss_total += loss.item()  # 【纯UI修改】：单纯记录 loss 值用于展示


        # 【纯UI修改】：计算本轮平均验证 loss
        valid_loss_avg = valid_loss_total / len(valid_loader)

        # 👇【架构升级：全局最优保存策略】
        # 只有当本轮的期末整体平均成绩 (valid_loss_avg) 破纪录时，才执行保存！
        if valid_loss_avg < best_valid_loss_avg:
            best_valid_loss_avg = valid_loss_avg
            # 👇【新增 2】：有下降，计数器清零
            patience_counter = 0
            output_dir = '../checkpoints/'
            os.makedirs(output_dir, exist_ok=True)
            path = (output_dir + args.model + '_' +
                    'feat-' + args.feat + '_' +
                    'pred_len-' + str(args.pred_len) + '_' +
                    'fold-' + str(args.fold) + '_' +
                    'node-' + str(args.pred_type) + '_' +
                    'add_feat-' + str(args.add_feat) + '_' +
                    'epoch-' + str(args.epoch) + '.pth')
            torch.save(net.state_dict(), path)

            # 👇【纯UI修改】：保存最优模型时，使用 write 输出极其清晰的破纪录提示！
            epoch_iterator.write(
                f"🏆 [Epoch {epoch + 1}] 突破历史最优全局验证集平均 Loss: {best_valid_loss_avg:.4f}，模型已保存！")

        else:
            patience_counter += 1

        # 【触发刹车】：只在调度器存在时才更新学习率
        if scheduler is not None:
            if args.lradj == 'plateau':
                # 策略一极其特殊，它必须吃进 valid_loss_avg 作为判断依据
                scheduler.step(valid_loss_avg)
            else:
                # step 和 cosine 策略只需按轮次往前走即可
                scheduler.step()


        # 【纯UI修改】：获取当前学习率
        current_lr = optim.param_groups[0]['lr']

        # 1. 输出本轮完整的 Loss 记录，保留历史轨迹
        epoch_iterator.write(
            f" [Epoch {epoch + 1}/{args.epoch}] Train_Loss_Avg: {train_loss_avg:.4f} | Val_Loss_Avg: {valid_loss_avg:.4f}")

        # 2. 检测学习率是否发生衰减，只有在变化时才主动弹窗提醒
        if current_lr != previous_lr:
            epoch_iterator.write(
                f" [Epoch {epoch + 1}] 引擎触发降速！学习率发生改变: {previous_lr:.6f} -> {current_lr:.6f}")
            previous_lr = current_lr  # 更新追踪器

        # 3. 进度条尾部只保留最清爽的核心指标，防止终端太乱
        epoch_iterator.set_postfix({
            'Train': f'{train_loss_avg:.4f}',
            'Val': f'{valid_loss_avg:.4f}'
        })

# 👇【新增 4】：触发早停机制，跳出整个 epoch 循环
        if patience_counter >= patience:
            tqdm.write(f"🛑 [Early Stopping] 验证集连续 {patience} 轮未改善，提前终止训练！")
            break
# def training(args, net, optim, loss_func, train_loader, valid_loader, fold):
#         valid_loss = 1000
#         net.train()
#         for _ in tqdm(range(args.epoch), desc='Training'):
#             for j, data in enumerate(train_loader):
#                 '''
#                 occupancy = (batch, seq, node)
#                 add_tensor = (batch, seq, node)
#                 label = (batch, node)
#                 '''
#                 net.train()
#                 extra_feat = 'None'
#                 if args.add_feat != 'None':
#                     occupancy, label, extra_feat = data
#                 else:
#                     occupancy, label = data
#
#                 optim.zero_grad()
#                 predict = net(occupancy, extra_feat)
#                 if predict.shape != label.shape:
#                     loss = loss_func(predict.unsqueeze(-1), label)
#                 else:
#                     loss = loss_func(predict, label)
#                 loss.backward()
#                 optim.step()
#
#             # validation
#             net.eval()
#             for j, data in enumerate(valid_loader):
#                 '''
#                 occupancy = (batch, seq, node)
#              = (batch, seq, node)
#                 label = (batch, node)
#                 '''
#                 net.train()
#                 extra_feat = 'None'
#                 if args.add_feat != 'None':
#                     occupancy, label, extra_feat = data
#                 else:
#                     occupancy, label = data
#
#                 predict = net(occupancy,extra_feat)
#                 if predict.shape != label.shape:
#                     loss = loss_func(predict.unsqueeze(-1), label)
#                 else:
#                     loss = loss_func(predict, label)
#                 if loss.item() < valid_loss:
#                     valid_loss = loss.item()
#                     output_dir = '../checkpoints/'
#                     os.makedirs(output_dir, exist_ok=True)
#                     path = (output_dir + args.model + '_' +
#                             'feat-' + args.feat + '_' +
#                             'pred_len-' + str(args.pred_len) + '_' +
#                             'fold-' + str(args.fold) + '_' +
#                             'node-' + str(args.pred_type) + '_' +
#                             'add_feat-' + str(args.add_feat) + '_' +
#                             'epoch-' + str(args.epoch) + '.pth')
#                     torch.save(net.state_dict(), path)

def test(args, test_loader, occ,net,scaler='None'):
    # ----init---
    result_list = []
    predict_list = np.zeros([1, occ.shape[1]])
    label_list = np.zeros([1, occ.shape[1]])
    if args.pred_type != 'region':
        predict_list = np.zeros([1,1])
        label_list = np.zeros([1,1])
    # ----init---
    if not args.stat_model:
        output_dir = '../checkpoints/'
        os.makedirs(output_dir,exist_ok=True)
        path = (output_dir + args.model + '_' +
                'feat-' + args.feat + '_' +
                'pred_len-' + str(args.pred_len) + '_' +
                'fold-' + str(args.fold) + '_' +
                'node-' + str(args.pred_type) + '_' +
                'add_feat-' + str(args.add_feat) + '_' +
                'epoch-' + str(args.epoch) + '.pth')
        state_dict = torch.load(path,weights_only=True)
        net.load_state_dict(state_dict)
        net.eval()
        for j, data in enumerate(test_loader):
            extra_feat = 'None'
            if args.add_feat != 'None':
                occupancy, label, extra_feat = data
            else:
                occupancy, label = data
            with torch.no_grad():
                predict = net(occupancy, extra_feat)
                if predict.shape != label.shape:
                    predict = predict.unsqueeze(-1)
                predict = predict.cpu().detach().numpy()
            label = label.cpu().detach().numpy()

    else:
        train_valid_occ,test_occ = test_loader
        predict = net.predict(train_valid_occ,test_occ)
        label = test_occ


    predict_list = np.concatenate((predict_list, predict), axis=0)
    label_list = np.concatenate((label_list, label), axis=0)
    if scaler != 'None':
        predict_list = scaler.inverse_transform(predict_list)
        label_list = scaler.inverse_transform(label_list)

    output_no_noise = utils.metrics(test_pre=predict_list[1:], test_real=label_list[1:],args=args)
    result_list.append(output_no_noise)

    # Adding model name, pre_l and metrics and so on to DataFrame
    result_df = pd.DataFrame(result_list, columns=['MSE', 'RMSE', 'MAPE', 'RAE', 'MAE'])
    result_df['model_name'] = args.model
    result_df['pred_len'] = args.pred_len
    result_df['fold'] = args.fold 

    # Save the results in a CSV file
    output_dir = '../result' + '/' + 'main_exp' + '/' + 'region'
    os.makedirs(output_dir, exist_ok=True)
    csv_file = output_dir + '/' + f'results.csv'

    # Append the result if the file exists, otherwise create a new file
    if os.path.exists(csv_file):
        result_df.to_csv(csv_file, mode='a', header=False, index=False, encoding='gbk')
    else:
        result_df.to_csv(csv_file, index=False, encoding='gbk')


