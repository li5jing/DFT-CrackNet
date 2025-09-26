from pytorch_lightning.callbacks import EarlyStopping, Callback, ModelCheckpoint
import os

class MyPrintingCallBack(Callback): #自定义的回调类，用在训练开始和结束时打印信息
    def __init__(self):
        super(MyPrintingCallBack, self).__init__()

    def on_train_start(self, trainer, pl_module):
        print("Start Training")

    def on_train_end(self, trainer, pl_module):
        print("Training is done")

checkpoint_callback = ModelCheckpoint(    #回调用于在训练过程中根据某些监控指标（如验证损失）保存模型的检查点
    dirpath=os.path.join(os.getcwd(), 'checkpoints', 'v7_BCEDICE0_2_final'),  #指定检查点保存的目录，意味着检查点文件将保存在当前工作目录下的 checkpoints/v7_BCEDICE0_2_final 文件夹中。
    filename='v7-epoch{epoch:02d}-val_loss{val_loss:.4f}', #定义检查点的文件命名规则，包含了epoch数和验证集的损失值
    verbose=True, #在保存检查点时输出详细信息
    save_last=True, #保存训练过程中的最后一个检查点
    save_top_k=5, #保存验证集损失最小的前五个检查点
    monitor='val_loss', #监控的指标是验证集的损失值，训练的过程中会根据val_loss的变化来决定是否保存检查点
    mode='min' #选择验证损失最小的检查点进行保存，如果监测的指标是准确率，此处改为max
)

early_stopping = EarlyStopping(  #用于提前中止训练
    monitor='val_loss', #监控验证集损失值
    patience=30, #如果验证损失在10个epoch内没有改善，则停止训练
    verbose=True,
    mode='min'
)
