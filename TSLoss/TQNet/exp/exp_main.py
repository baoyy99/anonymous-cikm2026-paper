from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from models import Informer, Autoformer, Transformer, DLinear, Linear, NLinear, PatchTST, SegRNN, CycleNet, \
    iTransformer, TimeXer, TQNet, TQDLinear, TQPatchTST, TQiTransformer
from utils.tools import EarlyStopping, adjust_learning_rate, visual, test_params_flop
from Loss_Metric.metric_factory import plot_all_metrics_single,Metrics
from Loss_Metric.loss_factory import get_loss_function,Residual_vis 
import numpy as np
import torch
import torch.nn as nn
from torch import optim
from torch.optim import lr_scheduler

import os
import time

import warnings
import matplotlib.pyplot as plt
import numpy as np

warnings.filterwarnings('ignore')

# ========== 新增：日志配置 ==========
import logging  # 导入logging模块
from logging.handlers import RotatingFileHandler  # 可选：防止log文件过大
# 初始化日志
# TQNet
os.makedirs("./TQNet/TSout", exist_ok=True)
def init_logger(log_file_path):
    
    # 日志格式：时间 - 日志级别 - 内容
    log_format = '%(asctime)s - %(levelname)s - %(message)s'
    # 配置日志
    logging.basicConfig(
        level=logging.INFO,  # 日志级别（INFO：普通信息，DEBUG：调试信息，ERROR：错误）
        format=log_format,
        handlers=[
            # 1. 输出到log文件
            RotatingFileHandler(  # 可选：用RotatingFileHandler自动切割大文件
                log_file_path,
                maxBytes=10*1024*1024,  # 单个log文件最大10MB
                backupCount=5,  # 最多保留5个旧log文件
                encoding='utf-8'
            ),
            # # 2. 同时输出到终端
            # logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

# 初始化日志（指定log文件保存路径，比如项目根目录下的training.log）TQNet
logger = init_logger(log_file_path='./TQNet/TSout/training.log')

# ========== 日志配置结束 ==========

class Exp_Main(Exp_Basic):
    def __init__(self, args):
        super(Exp_Main, self).__init__(args)

    def _build_model(self):
        model_dict = {
            'Autoformer': Autoformer,
            'Transformer': Transformer,
            'Informer': Informer,
            'DLinear': DLinear,
            'NLinear': NLinear,
            'Linear': Linear,
            'PatchTST': PatchTST,
            'SegRNN': SegRNN,
            'CycleNet': CycleNet,
            'iTransformer': iTransformer,
            'TimeXer': TimeXer,
            'TQNet': TQNet,
            'TQDLinear': TQDLinear,
            'TQPatchTST': TQPatchTST,
            'TQiTransformer': TQiTransformer
        }
        model = model_dict[self.args.model].Model(self.args).float()

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    # def _select_criterion(self):
    #     criterion = nn.MSELoss()
    #     return criterion
       # 我的调整后的损失函数选择方法，增加日志打印功能
    def _select_criterion(self):
        criterion = get_loss_function(self.args)
        logger.info(f"🔹 loss function: {criterion.__class__.__name__}")
        return criterion

    def vali(self, vali_data, vali_loader, criterion):
        total_loss = []
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark, batch_cycle) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)
                batch_cycle = batch_cycle.int().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if any(substr in self.args.model for substr in {'CycleNet', 'TQ'}):
                            outputs = self.model(batch_x, batch_cycle)
                        elif any(substr in self.args.model for substr in
                                 {'Linear', 'MLP', 'SegRNN', 'TST'}):
                            outputs = self.model(batch_x)
                        else:
                            if self.args.output_attention:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                            else:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    if any(substr in self.args.model for substr in {'CycleNet', 'TQ'}):
                        outputs = self.model(batch_x, batch_cycle)
                    elif any(substr in self.args.model for substr in {'Linear', 'MLP', 'SegRNN', 'TST'}):
                        outputs = self.model(batch_x)
                    else:
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)

                pred = outputs.detach()
                true = batch_y.detach()

                # loss = criterion(pred, true)
                # 参照原来PS Loss的逻辑，在计算验证的损失的时候依旧使用MSE损失
                if self.args.loss == 'psloss' or self.args.loss == 'psloss_taq': 
                    # print(self.model.output_proj.parameters)
                    mse_loss = nn.MSELoss()
                    loss = mse_loss(pred, true)
                else   :
                    loss = criterion(pred, true)

                total_loss.append(loss.item())
        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss

    def train(self, setting):
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')

        #========== checkpoints输出目录调整 ==========
        # path = os.path.join(self.args.checkpoints, setting)
        path = os.path.join('./TQNet/TSout/checkpoints', setting)
        #========== checkpoints输出目录调整结束 ==========
        if not os.path.exists(path):
            os.makedirs(path)

        # ========== 增加每个 epoch 开始的日志打印 ==========
        # logger.info(self.args.model_id)
        logger.info( setting)
        logger.info("loss: {} dataset: {} model: {}".format(self.args.loss, self.args.data_path, self.args.model))
        logger.info("begin training")
        # ========== 日志配置结束 ==========

        time_now = time.time()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()

        scheduler = lr_scheduler.OneCycleLR(optimizer=model_optim,
                                            steps_per_epoch=train_steps,
                                            pct_start=self.args.pct_start,
                                            epochs=self.args.train_epochs,
                                            max_lr=self.args.learning_rate)

        for epoch in range(self.args.train_epochs):
            #============= 增加第10个epoch时的测试逻辑 ==============
            # 第10个epoch时执行test逻辑
            if epoch == 10:
                logger.info("===== Epoch 10: Running test with current checkpoint =====")
                # 保存当前模型为临时checkpoint
                temp_ckpt_path = os.path.join(path, 'temp_epoch10_checkpoint.pth')
                torch.save(self.model.state_dict(), temp_ckpt_path)
                # 加载临时checkpoint并执行test
                self.model.load_state_dict(torch.load(temp_ckpt_path))
                self.test(setting, test=1)
                # 删除临时checkpoint（可选，如需保留可注释）
                os.remove(temp_ckpt_path)
                logger.info("===== Epoch 10: Test completed, resume training =====")
                # 模型切回训练模式
                self.model.train()
            #============= 增加第10个epoch时的测试逻辑 ==============                   

            iter_count = 0
            train_loss = []

            self.model.train()
            epoch_time = time.time()
            # max_memory = 0
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark, batch_cycle) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()
                batch_x = batch_x.float().to(self.device)

                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)
                batch_cycle = batch_cycle.int().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if any(substr in self.args.model for substr in {'CycleNet', 'TQ'}):
                            outputs = self.model(batch_x, batch_cycle)
                        elif any(substr in self.args.model for substr in
                                 {'Linear', 'MLP', 'SegRNN', 'TST'}):
                            outputs = self.model(batch_x)
                        else:
                            if self.args.output_attention:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                            else:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                        f_dim = -1 if self.args.features == 'MS' else 0
                        outputs = outputs[:, -self.args.pred_len:, f_dim:]
                        batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                        loss = criterion(outputs, batch_y)
                        train_loss.append(loss.item())
                else:
                    if any(substr in self.args.model for substr in {'CycleNet', 'TQ'}):
                        outputs = self.model(batch_x, batch_cycle)
                    elif any(substr in self.args.model for substr in {'Linear', 'MLP', 'SegRNN', 'TST'}):
                        outputs = self.model(batch_x)
                    else:
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]

                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark, batch_y)
                    # print(outputs.shape,batch_y.shape)
                    f_dim = -1 if self.args.features == 'MS' else 0
                    outputs = outputs[:, -self.args.pred_len:, f_dim:]
                    batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                    if self.args.loss == 'psloss' or self.args.loss == 'psloss_taq':
                        loss = criterion(outputs, batch_y,self.model)
                    else   :
                        loss = criterion(outputs, batch_y)
                    
                    # loss = criterion(outputs, batch_y)
                    train_loss.append(loss.item())

                if (i + 1) % 600 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                    # ========== 格式化训练迭代日志，制表符分隔+固定宽度对齐 ==========
                    log_msg = (
                        "Ititers: {iters:5d}\t | "     #加｜分割
                        "Loss: {loss:.7f}\t | "
                        "Speed: {speed:.4f}s/iter\t | "
                        "Left Time: {left:.4f}s"
                    ).format(
                        iters=i + 1,
                        loss=loss.item(),
                        speed=speed,
                        left=left_time
                    )
                    logger.info(log_msg)
                    # ========== 日志配置结束 ==========

                if self.args.use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    loss.backward()
                    model_optim.step()

                # current_memory = torch.cuda.max_memory_allocated() / 1024 ** 2
                # max_memory = max(max_memory, current_memory)

                if self.args.lradj == 'TST':
                    adjust_learning_rate(model_optim, scheduler, epoch + 1, self.args, printout=False)
                    scheduler.step()

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            vali_loss = self.vali(vali_data, vali_loader, criterion)
            test_loss = self.vali(test_data, test_loader, criterion)

            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss, test_loss))
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                # ========== 增加早停的日志打印 ==========
                logger.info("Early stopping")
                # ========== 日志配置结束 ==========
                break
            # ========== 增加每个 epoch 结束后的日志打印 ==========
            logger.info("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            logger.info("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(epoch + 1, train_steps, train_loss, vali_loss, test_loss))
            # ========== 日志配置结束 ==========

            if self.args.lradj != 'TST':
                adjust_learning_rate(model_optim, scheduler, epoch + 1, self.args)
            else:
                print('Updating learning rate to {}'.format(scheduler.get_last_lr()[0]))

        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))

        # print(f"Max Memory (MB): {max_memory}")

        return self.model

    def test(self, setting, test=0):
        # 每个模型创建自己的Metrics实例，状态完全独立
        metrics = Metrics()
        test_data, test_loader = self._get_data(flag='test')

        if test:
            print('loading model')
            # ========== 增加加载模型的日志打印 ==========
            logger.info("loading model")
            # ========== 日志配置结束 ==========
            #========== checkpoints检索目录调整 ==========
            # self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth')))
            self.model.load_state_dict(torch.load(os.path.join('./TQNet/TSout/checkpoints/' + setting, 'checkpoint.pth')))
            # self.model.load_state_dict(torch.load(os.path.join('./TQNet/TSout0427/checkpoints/' + setting, 'checkpoint.pth')))
            #========== checkpoints检索目录调整结束 ==========
        preds = []
        trues = []
        inputx = []
        # ========== test_results输出目录调整 ==========
        # folder_path = './test_results/' + setting + '/'
        folder_path = './TQNet/TSout/test_results/' + setting + '/'
        # ========== test_results输出目录调整结束 ==========
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        ############### 按batch平均
        ############### 按batch平均
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark, batch_cycle) in enumerate(test_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)
                batch_cycle = batch_cycle.int().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if any(substr in self.args.model for substr in {'CycleNet', 'TQ'}):
                            outputs = self.model(batch_x, batch_cycle)
                        elif any(substr in self.args.model for substr in
                                 {'Linear', 'MLP', 'SegRNN', 'TST'}):
                            outputs = self.model(batch_x)
                        else:
                            if self.args.output_attention:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                            else:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    if any(substr in self.args.model for substr in {'CycleNet', 'TQ'}):
                        outputs = self.model(batch_x, batch_cycle)
                    elif any(substr in self.args.model for substr in {'Linear', 'MLP', 'SegRNN', 'TST'}):
                        outputs = self.model(batch_x)
                    else:
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]

                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                f_dim = -1 if self.args.features == 'MS' else 0
                # print(outputs.shape,batch_y.shape)
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                
                outputs = outputs.detach().cpu().numpy()
                batch_y = batch_y.detach().cpu().numpy()

                pred = outputs  # outputs.detach().cpu().numpy()  # .squeeze()
                true = batch_y  # batch_y.detach().cpu().numpy()  # .squeeze()
                ############### 按batch平均
                metrics(pred, true)
                ############### 按batch平均

                # inputx.append(batch_x.detach().cpu().numpy())
                if i % 10 == 0:
                    input = batch_x.detach().cpu().numpy()

                    gt = true[0, :, -1]
                    pd = pred[0, :, -1]

                    # visual(gt, pd, os.path.join(folder_path, str(i) + '.pdf'))
                    # # np.savetxt(os.path.join(folder_path, str(i) + '.txt'), pd)
                    # # np.savetxt(os.path.join(folder_path, str(i) + 'true.txt'), gt)
                    # # ====================
                    Residual_vis(
                        pred=pd, 
                        true=gt, 
                        loss_type=self.args.loss.lower(), 
                        batch_idx=i,
                        save_dir_base=folder_path
                    )
                # ==========================================================
        # 一次性获取完整的最终误差分布画像
        final_profile = metrics.finalize()
        (
            tar, mae, mse, equal_ratio, overest_ratio, overest_sum_ratio,
            underest_ratio, underest_sum_ratio, peak_acc, valley_acc, peak_valley_mae,
            window_pos_avg_loss, p100, p80
        ) = final_profile

        if self.args.test_flop:
            test_params_flop(self.model, (batch_x.shape[1], batch_x.shape[2]))
            exit()


        # result save
        # ========== results输出目录调整 ==========
        # folder_path = './results/' + setting + '/'
        folder_path = './TQNet/TSout/results/' + setting + '/'
        # ========== results输出目录调整结束 ==========
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        # ========== 增加最终测试结果的日志打印 ==========
        timestamp = time.strftime("%Y-%m-%d %H:%M", time.localtime())   
        logger.info('mse:{:.3f}, mae:{:.3f}'.format(mse, mae))
        logger.info('P(1.0): {:.3f}, P(0.8): {:.3f}'.format(p100, p80))
        logger.info('TAR: {:.3f}'.format(tar))
        logger.info('ER: {:.3f}%, OR: {:.3f}%, OSR: {:.3f}%, UR: {:.3f}%, USR: {:.3f}%'.format(
            equal_ratio, overest_ratio, overest_sum_ratio, underest_ratio, underest_sum_ratio
        ))
        logger.info('PA: {:.3f}, VA: {:.3f}'.format(peak_acc, valley_acc))
        logger.info('PVMAE: {:.3f}'.format(peak_valley_mae))
        logger.info(window_pos_avg_loss)
            # 在你的test函数末尾添加
        save_path="./TQNet/TSout/metrics/"
        if not os.path.exists(save_path):
            os.makedirs(save_path)
            
        plot_all_metrics_single(
            mse=mse,
            mae=mae,
            P1=p100,
            P08=p80,
            tar=tar,
            er=equal_ratio,
            or_=overest_ratio,
            osr=overest_sum_ratio,
            ur=underest_ratio,
            usr=underest_sum_ratio,
            pa=peak_acc,
            va=valley_acc,
            pvmae=peak_valley_mae,
            winpa=window_pos_avg_loss,
            save_path=f'{save_path}{setting}.pdf'
        )   
         # ========== 日志配置结束 ==========

        # print('original:mse:{}, mae:{}'.format(mse, mae))
        f = open("./TQNet/TSout/result_sum.txt", 'a')
        # ========== 增加当前时间戳（格式化：年-月-日 时:分）
        f.write(timestamp + "  \n")
        # ========== 日志配置结束 ==========
        f.write(setting + "  \n")
        f.write('mse:{}, mae:{}'.format(mse, mae))
        f.write('\n')
        f.write('P(1.0):{:.3f}, P(0.8):{:.3f}'.format(p100, p80))
        f.write('\n')
        f.write('TAR: {:.3f}'.format(tar))
        f.write('\n')
        f.write('ER: {:.3f}%, OR: {:.3f}%, OSR: {:.3f}%, UR: {:.3f}%, USR: {:.3f}%'.format(
            equal_ratio, overest_ratio, overest_sum_ratio, underest_ratio, underest_sum_ratio
        ))
        f.write('\n')
        f.write('PA: {:.3f}, VA: {:.3f}'.format(peak_acc, valley_acc))
        f.write('\n')
        f.write('PVMAE: {:.3f}'.format(peak_valley_mae))
        f.write('\n')
        # f.write(' '.join([f"{x:.4f}" for x in window_pos_avg_loss]) + '\n')
        f.close()

        # np.save(folder_path + 'metrics.npy', np.array([mae, mse, rmse, mape, mspe,rse, corr]))
        # np.save(folder_path + 'pred.npy', preds)
        # np.save(folder_path + 'true.npy', trues)
        # np.save(folder_path + 'x.npy', inputx)
        return