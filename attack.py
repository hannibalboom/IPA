import argparse
import torch
from pipeline.modules.early_stop import EarlyStop
from pipeline.modules.pgd_optimizer import PGDOptimizer
from pipeline.modules.scheduler import ReduceLROnPlateau
from pipeline.pipeline import Pipeline
from tools.config import Config
from tools.debug_utils import time_block
from tools.logger import Logger


def parse_args():
    parser = argparse.ArgumentParser(description='Attack Pipeline')
    # params of evaluate
    parser.add_argument(
        "--config",
        "-f",
        dest="cfg",
        default="./data_share/config/attack.yaml",
        help="The config file path.",
        required=False,
        type=str)
    return parser.parse_args()


def perform_gradient_accumulation(pgd, scheduler, step_loss_list, pipeline, i, gradient_accumulation_steps):
    """
    执行梯度累积和参数更新的逻辑
    :param pgd: 优化器
    :param scheduler: 学习率调度器
    :param step_loss_list: 步骤损失列表
    :param pipeline: 管道对象
    :param i: 当前迭代次数
    :param gradient_accumulation_steps: 梯度累积步数
    """
    if (i + 1) % gradient_accumulation_steps == 0:
        # update patch
        pgd.step(step_type="softmax", step_loss_list=step_loss_list)
        scheduler.step(pipeline.loss.success_rate)
        # 清零梯度
        pgd.zero_grad()
        step_loss_list = []
    return step_loss_list


def main(args):
    cfg = Config(args.cfg)
    logger = Logger(cfg.cfg_logger)
    logger.broadcast_logger(cfg.cfg_all, exclude=[])

    pipeline = Pipeline(cfg)

    es = EarlyStop(max_step=180, eps=1e-3)
    train_flag = True
    epoch = 0
    step = 0
    step_loss_list = []

    pgd = PGDOptimizer(params={"adv_texture_hls": pipeline.stickers.adv_texture_hls,
                               "ori_texture_hls": pipeline.stickers.ori_texture_hls,
                               "mask": pipeline.stickers.mask},
                       alpha=cfg.cfg_attack["optimizer"]["alpha"],
                       clip_min=cfg.cfg_attack["optimizer"]["clip_min"],
                       clip_max=cfg.cfg_attack["optimizer"]["clip_max"],
                       device=cfg.cfg_global["device"])
    scheduler = ReduceLROnPlateau(optimizer=pgd,
                                  mode='max',
                                  factor=0.1,
                                  patience=60,
                                  threshold=1e-3,
                                  threshold_mode='rel',
                                  cooldown=10,
                                  min_lr=0,
                                  eps=1e-8,
                                  verbose=False)

    # 设置梯度累积步数
    gradient_accumulation_steps = 4

    while train_flag:
        # Sync visualization epoch
        pipeline.visualization.epoch = epoch
        # Init for new epoch
        _epoch_loss = 0
        dataset = pipeline.dataset.train_data
        for i, sample in enumerate(dataset):
            # Sync visualization step
            pipeline.visualization.step = step
            try:
                with time_block("Forward & Backward & Step"):
                    loss = pipeline.forward(sample)
                    if loss is not None:
                        # 梯度累积：将损失除以累积步数
                        scaled_loss = loss / gradient_accumulation_steps
                        scaled_loss.backward()
                    pgd.record()
            except KeyboardInterrupt:
                print("Stop Attack Manually!")
                logger.close_logger()

            with torch.no_grad():
                if loss is None:
                    loss = torch.tensor(0.0)
                _step_loss = loss.clone().cpu().item() * -1 * gradient_accumulation_steps
                _epoch_loss += _step_loss
                step_loss_list.append(_step_loss)
                # Visualization Pipeline
                with time_block("Vis"):
                    pipeline.visualization.vis(scenario_index=sample.scenario_index,
                                               scenario=pipeline.scenario,
                                               renderer=pipeline.renderer,
                                               stickers=pipeline.stickers,
                                               smoke=pipeline.smoke,
                                               defense=pipeline.defense)
                # print to terminal
                print(f"epoch: {epoch:04d}   step: {step:04d}   step_loss: {_step_loss:.10f}")
                # clear and prepare for the next step
                step += 1

            # 执行梯度累积和参数更新
            step_loss_list = perform_gradient_accumulation(pgd, scheduler, step_loss_list, pipeline, i,
                                                           gradient_accumulation_steps)

            # 释放不再使用的中间变量并清空 CUDA 缓存
            del loss, scaled_loss
            torch.cuda.empty_cache()

        # 如果最后一个 batch 不足梯度累积步数，仍然更新参数
        if len(dataset) % gradient_accumulation_steps != 0:
            pgd.step(step_type="softmax", step_loss_list=step_loss_list)
            scheduler.step(pipeline.loss.success_rate)
            pgd.zero_grad()
            step_loss_list = []

        # print to terminal
        print("==============================================================")
        print(f"epoch: {epoch:04d}   epoch_loss: {_epoch_loss:.10f}   mean_loss: {_epoch_loss / len(dataset):.10f}")
        print("==============================================================\n")

        # save patch
        pipeline.visualization.save_texture(_epoch_loss, pipeline.stickers)
        # check if you can stop training
        train_flag = es.step(-1 * pipeline.loss.success_rate)

        # save patch before exit
        if not train_flag:
            pipeline.visualization.save_texture(_epoch_loss, pipeline.stickers)

        # clear and prepare for the next epoch
        step_loss_list = []
        epoch += 1

    # ensure all metrics and code are logged before exiting
    logger.close_logger()


if __name__ == '__main__':
    args = parse_args()
    main(args)