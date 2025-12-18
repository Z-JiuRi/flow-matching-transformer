"""主程序"""

import argparse
from omegaconf import OmegaConf

import warnings
warnings.filterwarnings("ignore")

from core import Trainer, Tester, Inferencer
from utils.visualize import setup_global_fonts



def main():
    parser = argparse.ArgumentParser(description="Flow-Matching Conditional Generation")
    parser.add_argument("--config", type=str, default="configs/config.yaml", help="配置文件路径")
    parser.add_argument("--mode", type=str, choices=["train", "test", "inference"], default="train", help="运行模式")
    parser.add_argument("--resume", type=str, default=None, help="恢复训练的检查点路径")
    
    # 推理参数
    parser.add_argument("--checkpoint", type=str, default=None, help="测试/推理使用的检查点路径")
    parser.add_argument("--condition", type=str, default=None, help="条件向量文件路径")
    parser.add_argument("--layer", type=int, default=0, help="Layer索引 (0或1)")
    parser.add_argument("--output", type=str, default=None, help="输出文件路径")
    
    args = parser.parse_args()
    
    # 加载配置
    cfg = OmegaConf.load(args.config)
    setup_global_fonts()
    # 运行
    if args.mode == "train":
        trainer = Trainer(config=cfg, resume=args.resume)
        trainer.train()
    
    elif args.mode == "test":
        tester = Tester(config=cfg, checkpoint_path=args.checkpoint)
        tester.test()
    
    elif args.mode == "inference":
        if args.condition is None:
            raise ValueError("必须指定 --condition 参数")
        if args.output is None:
            raise ValueError("必须指定 --output 参数")        
        inferencer = Inferencer(config=cfg, checkpoint_path=args.checkpoint)
        inferencer.run(
            condition_path=args.condition,
            layer=args.layer,
            output_path=args.output,
        )


if __name__ == "__main__":
    main()
