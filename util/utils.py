import torch
import torchvision
import matplotlib.pyplot as plt
import numpy as np
import os

result_save_ind = 0
threshold = 0.5

def eval_metrics(loader, model, device="cuda", multiple_outputs=False): #用于评估模型在某个数据加载器上常见指标，包括准确度、精确度、召回率、假阳性率、F1分数、dice系数和IOU系数
    model.eval()  #将模型设置为验证模式
    eps = 1e-7  #防止在计算过程中出现除0的错误

    TP_total = 0
    FP_total = 0
    TN_total = 0
    FN_total = 0

    with torch.no_grad(): #禁止梯度计算，不需要进行反向传播
        for x, y,_ in loader: #x是输入数据，y是目标标签
            x = x.to(device)
            y = y.to(device).unsqueeze(1)#二值标签形状通常是（b,h,w）增加一个维度确保标签与输入图像的形状一致

            if multiple_outputs == True: #true表示模型有多个输出结果
                final_output = model(x)[result_save_ind] #[result_save_ind] 用于索引模型输出的某一部分
                preds_probability = torch.sigmoid(final_output)#final_output通常是原始未经过激活的分数，经过激活函数，转化为概率
                preds = (preds_probability > threshold).float() #对概率值进行二值化处理，两者相比较得到的是布尔值（true or false）,再通过.float处理转化为浮点数张量
            else:  #模型的输出只有一种结果
                preds_probability = torch.sigmoid(model(x))
                preds = (preds_probability > threshold).float()

            confusion_matirx = preds / y  #预测值与真值标签之间进行比较，通过元素级除法得到每个位置的结果

            TP =  torch.sum(confusion_matirx == 1).item()
            FP = torch.sum(confusion_matirx == float('inf')).item()
            TN = torch.sum(torch.isnan(confusion_matirx)).item()
            FN = torch.sum(confusion_matirx == 0).item()


            TP_total += TP
            FP_total += FP
            TN_total += TN
            FN_total += FN


    accuracy = (TP_total + TN_total) / (TP_total + FP_total + TN_total + FN_total + eps)  #在所有预测的比例中，被预测正确的比例
    precision = (TP_total) / (TP_total+FP_total + eps)#当模型预测为正类，实际上有多少是正类的比例
    recall = (TP_total) / (TP_total+FN_total + eps) # TP rate #在实际为正类的样本中，模型能被正确预测为正类的比例
    FP_rate = FP_total / (FP_total+TN_total + eps)#假阳性率，实际上是负类的样本中，被模型错误的预测为正类的比例
    f1_score = 2* (precision*recall)/(precision+recall+eps)
    dice_score = 2*TP_total / (2*TP_total+FP_total+FN_total + eps) # will be the same as f1 score
    IOU_score = TP_total / (TP_total + FP_total + FN_total + eps)

    print(f'Global Accuracy : {accuracy} / Precision : {precision} / Recall : {recall} / FPR : {FP_rate} / F1 score : {f1_score}')
    print(f'Dice Score {dice_score} / IOU score {IOU_score}')

def eval_OIS(loader, model, device="cuda", multiple_outputs=False):#通过不同的阈值计算每个阈值下的模型性能，并返回最佳的F1分数和相应的阈值
    best_OIS_lst = []  #用来存储每个批次数据中的最佳OIS（F1分数）
    best_thres_lst = [] #用来存储每个批次数据中的最佳阈值
    thres_list = [i for i in np.arange(0, 1, step=0.01)] #设置0-1之间，以0.01步长的多个阈值
    eps = 1e-7

    model.eval()

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device).unsqueeze(1)

            best_thres = 0
            best_OIS = 0
            for thres in thres_list:
                if multiple_outputs == True:
                    final_output = model(x)[result_save_ind]
                    preds_probability = torch.sigmoid(final_output)
                    preds = (preds_probability > threshold).float()
                else:
                    preds_probability = torch.sigmoid(model(x))
                    preds = (preds_probability > threshold).float()


                confusion_matirx = preds / y

                TP = torch.sum(confusion_matirx == 1).item()
                FP = torch.sum(confusion_matirx == float('inf')).item()
                TN = torch.sum(torch.isnan(confusion_matirx)).item()
                FN = torch.sum(confusion_matirx == 0).item()

                precision = (TP) / (TP+FP+eps)
                recall = (TP) / (TP+FN+eps) # TP rate
                f1_score = 2* (precision*recall)/(precision+recall+eps)

                if f1_score > best_OIS:
                    best_OIS = f1_score
                    best_thres = thres

            best_thres_lst.append(best_thres)
            best_OIS_lst.append(best_OIS)

    mean_OIS = np.mean(best_OIS_lst)
    mean_thres = np.mean(best_thres_lst)

    print(f'OIS F1 Score : {mean_OIS} / with the mean threshod : {mean_thres}')

    return mean_OIS, mean_thres

def eval_ODS(loader, model, device="cuda", multiple_outputs=False):  #根据不同的阈值来评估模型的表现，找到最佳的阈值
    model.eval()

    best_ODS = 0
    best_thres = 0
    thres_list = [i for i in np.arange(0, 1, step=0.01)]
    eps = 1e-7


    with torch.no_grad():
        for thres in thres_list:
            TP_total = 0
            FP_total = 0
            TN_total = 0
            FN_total = 0
            for x, y in loader:
                x = x.to(device)
                y = y.to(device).unsqueeze(1)

                if multiple_outputs == True:
                    final_output = model(x)[result_save_ind]
                    preds_probability = torch.sigmoid(final_output)
                    preds = (preds_probability > threshold).float()
                else:
                    preds_probability = torch.sigmoid(model(x))
                    preds = (preds_probability > threshold).float()

                confusion_matirx = preds / y

                TP =  torch.sum(confusion_matirx == 1).item()
                FP = torch.sum(confusion_matirx == float('inf')).item()
                TN = torch.sum(torch.isnan(confusion_matirx)).item()
                FN = torch.sum(confusion_matirx == 0).item()

                TP_total += TP
                FP_total += FP
                TN_total += TN
                FN_total += FN

            precision = (TP_total) / (TP_total+FP_total+eps)
            recall = (TP_total) / (TP_total+FN_total+eps) # TP rate
            f1_score = 2* (precision*recall)/(precision+recall+eps)
            if f1_score > best_ODS:
                best_ODS = f1_score
                best_thres = thres

    print(f'ODS F1 Score : {best_ODS} / with the threshod : {best_thres}')

    return best_ODS, best_thres

def save_predictions_as_imgs(loader, model, folder="saved_images/", device="cuda", multiple_outputs=False, threshold=0.5, result_save_ind=0):
    os.makedirs(folder, exist_ok=True)  # ✅ 确保目录存在
    model.eval()

    for idx, (x, y, filename) in enumerate(loader):  # ✅ 解包图像名
        x = x.to(device=device)
        with torch.no_grad():
            if multiple_outputs:
                final_output = model(x)[result_save_ind]
                preds_probability = torch.sigmoid(final_output)
                preds = (preds_probability > threshold).float()
            else:
                preds_probability = torch.sigmoid(model(x))
                preds = (preds_probability > threshold).float()

        # 提取图像文件名，不带扩展名
        base_name = os.path.splitext(filename[0])[0]
        pred_path = os.path.join(folder, f"{base_name}_pred.png")
        label_path = os.path.join(folder, f"{base_name}_gt.png")

        torchvision.utils.save_image(preds, pred_path)
        torchvision.utils.save_image(y.unsqueeze(1), label_path)

    model.train()



def loss_plot(train_loss, val_loss): #用于绘制训练损失和验证损失的图形，并将其保存为图像文件
    if len(train_loss) != len(val_loss):
        print('The number of losses are different')
    else:
        labels = [i for i in range(1, len(train_loss)+1)]
        plt.plot(train_loss)
        plt.plot(val_loss)
        # plt.xticks(range(0, len(train_loss), 10), labels[::9])
        ticks = [i for i in range(0, len(train_loss), 10)]  # Ticks at every 10th index
        tick_labels = [labels[i-1] for i in ticks]  # Corresponding labels for the ticks
        plt.xticks(ticks, tick_labels)
        plt.gca().get_xticklabels()[0].set_visible(False)
        plt.xlabel('Epoch', fontsize=17)
        plt.ylabel('Loss', fontsize=17)
        plt.show()
        plt.savefig('loss_output.png')