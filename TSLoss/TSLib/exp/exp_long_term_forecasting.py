from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from utils.tools import EarlyStopping, adjust_learning_rate, visual
from utils.metrics import metric
from Loss_Metric.metric_factory import plot_all_metrics_single,Metrics
from Loss_Metric.loss_factory import get_loss_function,Residual_vis 
import torch
import torch.nn as nn
from torch import optim
import os
import time
import warnings
import numpy as np
from utils.dtw_metric import dtw, accelerated_dtw
from utils.augmentation import run_augmentation, run_augmentation_single 

# ========== 新增：日志配置 ==========
import logging  # 导入logging模块
from logging.handlers import RotatingFileHandler  # 可选：防止log文件过大
# 初始化日志
os.makedirs("./TSLib/TSout", exist_ok=True)
def init_logger(log_file_path='training.log'):
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

# 初始化日志（指定log文件保存路径，比如项目根目录下的training.log）
logger = init_logger(log_file_path='./TSLib/TSout/training.log')
# ========== 日志配置结束 ==========

warnings.filterwarnings('ignore')


class Exp_Long_Term_Forecast(Exp_Basic):
    def __init__(self, args):
        super(Exp_Long_Term_Forecast, self).__init__(args)

    def _build_model(self):
        model = self.model_dict[self.args.model].Model(self.args).float()

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim
#调整后的损失函数选择方法，增加日志打印功能
    def _select_criterion(self):
        criterion = get_loss_function(self.args)
        logger.info(f"🔹 loss function: {criterion.__class__.__name__}")
        return criterion
    
    def vali(self, vali_data, vali_loader, criterion):
        total_loss = []
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)

                pred = outputs.detach()
                true = batch_y.detach()

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
        print(f"data load start")
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')

        #========== checkpoints输出目录调整 ==========
        # path = os.path.join(self.args.checkpoints, setting)
        path = os.path.join('./TSLib/TSout/checkpoints', setting)
        #========== checkpoints输出目录调整结束 ==========

        if not os.path.exists(path):
            os.makedirs(path)
         # ========== 增加每个 epoch 开始的日志打印 ==========
        # logger.info(self.args.model_id)
        logger.info("loss: {} dataset: {} model: {}".format(self.args.loss, self.args.data_path, self.args.model))
        logger.info( setting)
        logger.info("begin training")
        # ========== 日志配置结束 ==========
        time_now = time.time()
 
        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []

            self.model.train()
            epoch_time = time.time()

            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                        f_dim = -1 if self.args.features == 'MS' else 0
                        outputs = outputs[:, -self.args.pred_len:, f_dim:]
                        batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                        loss = criterion(outputs, batch_y)
                        train_loss.append(loss.item())
                else:#进入模型开始训练
                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                    f_dim = -1 if self.args.features == 'MS' else 0
                    outputs = outputs[:, -self.args.pred_len:, f_dim:]
                    batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                    #[batch, pred_len, feature]
                    if self.args.loss == 'psloss' or self.args.loss == 'psloss_taq':
                        loss = criterion(outputs, batch_y,self.model)
                    else   :
                        loss = criterion(outputs, batch_y)
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
                    #计算梯度，反向传播
                    loss.backward()
                    #根据梯度更新参数，梯度下降
                    model_optim.step()

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

            adjust_learning_rate(model_optim, epoch + 1, self.args)

        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))

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
            self.model.load_state_dict(torch.load(os.path.join('./TSLib/TSout/checkpoints/' + setting, 'checkpoint.pth')))
            
            # self.model.load_state_dict(torch.load(os.path.join('./TSLib/TSout/checkpoints/mse_0.55_0.0_1.0_ftexchange_rate.csv_sliTransformer_llcustom_plM_dm96_nh48_el96_dl128_df8_expand2_dc1_fc128_eb2_dt4_3_timeF_True', 'checkpoint.pth')))
            #========== checkpoints检索目录调整结束 ==========
        preds = []
        trues = []
        # ========== test_results输出目录调整 ==========
        # folder_path = './test_results/' + setting + '/'
        folder_path = './TSLib/TSout/test_results/' + setting + '/'
        # ========== test_results输出目录调整结束 ==========

        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        ############### 按batch平均
        ############### 按batch平均
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, :]
                batch_y = batch_y[:, -self.args.pred_len:, :].to(self.device)

                
                outputs = outputs.detach().cpu().numpy()
                batch_y = batch_y.detach().cpu().numpy()
                if test_data.scale and self.args.inverse:
                    shape = batch_y.shape
                    if outputs.shape[-1] != batch_y.shape[-1]:
                        outputs = np.tile(outputs, [1, 1, int(batch_y.shape[-1] / outputs.shape[-1])])
                    outputs = test_data.inverse_transform(outputs.reshape(shape[0] * shape[1], -1)).reshape(shape)
                    batch_y = test_data.inverse_transform(batch_y.reshape(shape[0] * shape[1], -1)).reshape(shape)

                outputs = outputs[:, :, f_dim:]
                batch_y = batch_y[:, :, f_dim:]

                pred = outputs
                true = batch_y
                ############### 按batch平均
                metrics(pred, true)
                ############### 按batch平均

                if i % 20 == 0:
                    input = batch_x.detach().cpu().numpy()
                    if test_data.scale and self.args.inverse:
                        shape = input.shape
                        input = test_data.inverse_transform(input.reshape(shape[0] * shape[1], -1)).reshape(shape)
                    gt = np.concatenate((input[0, :, -1], true[0, :, -1]), axis=0)
                    pd = np.concatenate((input[0, :, -1], pred[0, :, -1]), axis=0)
                    # visual(gt, pd, os.path.join(folder_path, str(i) + '.pdf'))
                    # # ========== 新增的 4线可视化,有pall ema==========
                    Residual_vis(
                        pred=pd, 
                        true=gt, 
                        loss_type=self.args.loss.lower(), 
                        batch_idx=i,
                        save_dir_base=folder_path
                    )
                # ==========================================================
        ############### 按batch平均
        final_profile = metrics.finalize()
        (
            tar, mae, mse, equal_ratio, overest_ratio, overest_sum_ratio,
            underest_ratio, underest_sum_ratio, peak_acc, valley_acc, peak_valley_mae,
            window_pos_avg_loss, p100, p80
        ) = final_profile
        ############### 按batch平均
        
        if self.args.use_dtw:
            dtw_list = []
            manhattan_distance = lambda x, y: np.abs(x - y)
            for i in range(preds.shape[0]):
                x = preds[i].reshape(-1, 1)
                y = trues[i].reshape(-1, 1)
                if i % 100 == 0:
                    print("calculating dtw iter:", i)
                    # ========== 增加 计算dtw 的日志打印，几乎不会使用 ==========
                    logger.info('calculating dtw iter: {}'.format(i))
                    # ========== 日志配置结束 ==========
                d, _, _, _ = accelerated_dtw(x, y, dist=manhattan_distance)
                dtw_list.append(d)
            dtw = np.array(dtw_list).mean()
        else:
            dtw = 'Not calculated'
        # print('mse:{}, mae:{}, dtw:{}'.format(mse, mae, dtw))

        # result save
        # ========== results输出目录调整 ==========
        # folder_path = './results/' + setting + '/'
        folder_path = './TSLib/TSout/results/' + setting + '/'
        # ========== results输出目录调整结束 ==========

        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        
        # ========== 增加最终测试结果的日志打印 ==========
        timestamp = time.strftime("%Y-%m-%d %H:%M", time.localtime())   
        # logger.info(timestamp)
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
        # plot_all_metrics_single(
        #     mse=mse,
        #     mae=mae,
        #     P1=p100,
        #     P08=p80,
        #     tar=tar,
        #     er=equal_ratio,
        #     or_=overest_ratio,
        #     osr=overest_sum_ratio,
        #     ur=underest_ratio,
        #     usr=underest_sum_ratio,
        #     pa=peak_acc,
        #     va=valley_acc,
        #     pvmae=peak_valley_mae,
        #     winpa=window_pos_avg_loss,
        #     save_path=f'{save_path}{setting}.pdf'
        # )   
         # ========== 日志配置结束 ==========


        f = open("./TSLib/TSout/result_sum.txt", 'a')
        # ========== 增加当前时间戳（格式化：年-月-日 时:分）
        f.write(timestamp + "  \n")
        # ========== 日志配置结束 ==========
        f.write(setting + "  \n")
        f.write(" \n")
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
        f.write(' '.join([f"{x:.4f}" for x in window_pos_avg_loss]) + '\n')
        f.close()

        # np.save(folder_path + 'metrics.npy', np.array([mae, mse]))
        # np.save(folder_path + 'pred.npy', preds)
        # np.save(folder_path + 'true.npy', trues)

        return
