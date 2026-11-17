#!/usr/bin/env python
"""
00_verify_env.py - 环境验证脚本（开题第一周必须运行）

验证内容：
1. CUDA 是否可用、GPU 型号与显存
2. PyTorch 版本与 CUDA 编译版本
3. sm_120 (Blackwell) 架构支持
4. bitsandbytes 4-bit 量化是否真正生效（防止 Windows 上静默 no-op）
5. BF16 计算支持

用法：
    conda activate lora-project
    python scripts/00_verify_env.py
"""
import sys
from pathlib import Path

# 将项目根目录加入 sys.path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def check_cuda():
    print("=" * 60)
    print("[1/5] CUDA 环境检查")
    print("=" * 60)
    try:
        import torch
    except ImportError:
        print("[FAIL] PyTorch 未安装")
        return None

    print(f"PyTorch 版本      : {torch.__version__}")
    print(f"CUDA 编译版本     : {torch.version.cuda}")
    print(f"CUDA 是否可用     : {torch.cuda.is_available()}")

    if not torch.cuda.is_available():
        print("[FAIL] CUDA 不可用，请检查驱动与 PyTorch CUDA 轮子")
        return None

    print(f"GPU 型号          : {torch.cuda.get_device_name(0)}")
    cap = torch.cuda.get_device_capability(0)
    print(f"计算能力          : sm_{cap[0]}{cap[1]}")
    total, _ = torch.cuda.get_device_properties(0).total_memory, None
    print(f"显存总量          : {total / 1024**3:.2f} GB")

    arch_list = torch.cuda.get_arch_list()
    print(f"PyTorch 支持架构  : {arch_list}")
    if 'sm_120' in arch_list:
        print("[OK] sm_120 (Blackwell) 架构已支持")
    else:
        print("[WARN] 未列出 sm_120，但驱动可能通过 PTX 兼容运行")

    if torch.cuda.is_bf16_supported():
        print("[OK] BF16 计算受支持")
    else:
        print("[WARN] BF16 不受支持，配置中需改用 fp16")
    return torch


def check_bitsandbytes(torch):
    print()
    print("=" * 60)
    print("[2/5] bitsandbytes 4-bit 量化生效验证（关键步骤）")
    print("=" * 60)
    try:
        import bitsandbytes as bnb
        from bitsandbytes.nn import Linear4bit
    except ImportError as e:
        print(f"[FAIL] bitsandbytes 未安装: {e}")
        return False

    print(f"bitsandbytes 版本 : {bnb.__version__}")

    # 创建一个小型线性层并量化
    layer = torch.nn.Linear(32, 32).to('cuda')
    qlayer = Linear4bit(
        layer.in_features,
        layer.out_features,
        quant_type='nf4',
        compute_dtype=torch.bfloat16,
    ).to('cuda')

    # 验证量化确实生效
    weight = qlayer.weight
    print(f"量化后权重类型    : {type(weight).__name__}")
    print(f"量化状态          : {getattr(weight, 'quant_state', None) is not None}")

    if type(weight).__name__ == 'Params4bit' and getattr(weight, 'quant_state', None) is not None:
        print("[OK] bitsandbytes NF4 4-bit 量化正常工作")
        result = True
    else:
        print("[FAIL] 量化未生效！可能原因：")
        print("       - Windows 上 pip 安装的 wheel 与 CUDA 版本不匹配")
        print("       - 尝试: conda install -c conda-forge bitsandbytes")
        print("       - 或从源码编译: pip install bitsandbytes --no-binary :all:")
        result = False

    # FP4 检查
    try:
        qlayer_fp4 = Linear4bit(
            32, 32, quant_type='fp4', compute_dtype=torch.bfloat16
        ).to('cuda')
        if getattr(qlayer_fp4.weight, 'quant_state', None) is not None:
            print("[OK] FP4 量化类型可用")
        else:
            print("[WARN] FP4 量化类型异常")
    except Exception as e:
        print(f"[WARN] FP4 检查跳过: {e}")

    # 8-bit 优化器检查
    try:
        import bitsandbytes.optim as bnb_optim
        opt = bnb_optim.PagedAdamW8bit(layer.parameters())
        print("[OK] PagedAdamW8bit 优化器可用")
        del opt
    except Exception as e:
        print(f"[WARN] PagedAdamW8bit 检查失败: {e}")

    return result


def check_transformers_stack():
    print()
    print("=" * 60)
    print("[3/5] Transformers / PEFT / TRL 库版本检查")
    print("=" * 60)
    ok = True
    for name in ['transformers', 'peft', 'trl', 'datasets', 'accelerate']:
        try:
            mod = __import__(name)
            print(f"[OK] {name:15s} {getattr(mod, '__version__', 'unknown')}")
        except ImportError:
            print(f"[FAIL] {name} 未安装")
            ok = False
    return ok


def check_local_assets():
    print()
    print("=" * 60)
    print("[4/5] 本地资源文件检查")
    print("=" * 60)
    ok = True
    required = [
        ROOT / 'configs' / 'experiment.yaml',
        ROOT / 'models' / 'Qwen2.5-1.5B-Instruct' / 'config.json',
        ROOT / 'models' / 'Qwen2.5-1.5B-Instruct' / 'model.safetensors',
        ROOT / 'models' / 'Qwen2.5-1.5B-Instruct' / 'tokenizer.json',
        ROOT / 'data' / 'raw' / 'ag_news' / 'data' / 'train-00000-of-00001.parquet',
        ROOT / 'data' / 'raw' / 'ag_news' / 'data' / 'test-00000-of-00001.parquet',
    ]
    for p in required:
        if p.exists():
            size = p.stat().st_size
            print(f"[OK] {p.relative_to(ROOT)} ({size / 1024**2:.1f} MB)")
        else:
            print(f"[FAIL] 缺失: {p.relative_to(ROOT)}")
            ok = False
    return ok


def check_transformers_load(torch):
    print()
    print("=" * 60)
    print("[5/5] 实际加载模型 + 4-bit 量化冒烟测试")
    print("=" * 60)
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        model_path = str(ROOT / 'models' / 'Qwen2.5-1.5B-Instruct')
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type='nf4',
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            quantization_config=bnb_config,
            device_map='auto',
            local_files_only=True,
        )
        tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
        mem = torch.cuda.max_memory_allocated() / 1024**3
        print(f"[OK] 模型加载成功，峰值显存约 {mem:.2f} GB")
        print(f"[OK] 词表大小: {len(tokenizer)}")
        del model
        torch.cuda.empty_cache()
        return True
    except Exception as e:
        print(f"[FAIL] 模型加载失败: {e}")
        return False


def main():
    print("\n" + "=" * 60)
    print("CIE6053 Final Project - 环境验证")
    print("=" * 60 + "\n")

    torch = check_cuda()
    if torch is None:
        print("\n[结论] 环境未就绪，请先修复 CUDA/PyTorch")
        sys.exit(1)

    bnb_ok = check_bitsandbytes(torch)
    libs_ok = check_transformers_stack()
    assets_ok = check_local_assets()
    load_ok = check_transformers_load(torch)

    print()
    print("=" * 60)
    print("验证汇总")
    print("=" * 60)
    results = {
        'bitsandbytes 量化': bnb_ok,
        '依赖库': libs_ok,
        '本地资源': assets_ok,
        '模型加载': load_ok,
    }
    for k, v in results.items():
        print(f"  {'[PASS]' if v else '[FAIL]'} {k}")

    if all(results.values()):
        print("\n[结论] 环境完全就绪，可以开始实验。")
    else:
        print("\n[结论] 存在未通过项，请按上方提示修复后重新运行。")
        sys.exit(1)


if __name__ == '__main__':
    main()
