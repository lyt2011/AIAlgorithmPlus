import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
import time
import os
import sys
import numpy as np
from tqdm import tqdm
import warnings

# 忽略不必要的警告
warnings.filterwarnings('ignore', category=UserWarning, module='torchvision')

# ==================== FIX: 修复 autocast API 导入 ====================
try:
    # PyTorch 2.0+ 新API
    from torch.amp import autocast, GradScaler

    USE_NEW_AMP = True
except ImportError:
    # 旧版本回退
    from torch.cuda.amp import autocast, GradScaler

    USE_NEW_AMP = False


# ==================== 革命性配置区 ====================
class Config2026:
    def __init__(self):
        # FIX: Windows检测
        is_windows = sys.platform == 'win32'

        self.development_mode = 'COLAB' in os.environ or 'KAGGLE' in os.environ or 'PYCHARM' in os.environ

        self.batch_size = 512 if torch.cuda.is_available() else 64
        self.epochs = 50 if not self.is_development() else 5
        # FIX: Windows强制num_workers=0避免多进程错误，Linux/Mac使用min(4, cpu_count)
        self.num_workers = 0 if is_windows else min(4, os.cpu_count() or 2)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        # FIX: CPU环境下禁用混合精度（CPU不支持）
        self.mixed_precision = torch.cuda.is_available()
        self.compile_model = hasattr(torch, 'compile') and torch.cuda.is_available()
        self.gradient_accumulation = 4 if torch.cuda.is_available() else 1
        # FIX: Windows CPU禁用pin_memory
        self.pin_memory = torch.cuda.is_available() and not is_windows
        # FIX: Windows禁用persistent_workers
        self.persistent_workers = False
        self.prefetch_factor = 2
        self.quick_test = False
        self.parallel_experiments = False
        self.experiment_modules = ['softmax', 'sasp', 'sassp', 'autoformula']

    def is_development(self):
        return self.development_mode or not torch.cuda.is_available()

    def get_device_info(self):
        if torch.cuda.is_available():
            return {
                'name': torch.cuda.get_device_name(0),
                'capability': torch.cuda.get_device_capability(),
                'memory': f"{torch.cuda.get_device_properties(0).total_memory / 1024 ** 3:.1f}GB",
                'count': torch.cuda.device_count()
            }
        return {'name': 'CPU', 'cores': os.cpu_count()}


CONFIG_2026 = Config2026()

print(f"🚀 2026极速训练框架 | 设备: {CONFIG_2026.get_device_info()['name']}")
print(f"⚡ 混合精度: {'ON' if CONFIG_2026.mixed_precision else 'OFF'} | "
      f"模型编译: {'ON' if CONFIG_2026.compile_model else 'OFF'}")
print(f"Workers: {CONFIG_2026.num_workers} | 开发模式: {'ON' if CONFIG_2026.is_development() else 'OFF'}")
print("-" * 70)


# ==================== 1. 修复版概率模块 ====================

class SASP(nn.Module):
    """稳定自适应平方概率 (修复版)"""

    def __init__(self, alpha_init=1.0):
        super().__init__()
        self.alpha = nn.Parameter(torch.tensor(alpha_init, dtype=torch.float32))
        self.register_buffer('min_value', torch.tensor(1e-6))
        self.register_buffer('base_offset', torch.tensor(1.0))

    def forward(self, x):
        alpha_safe = F.softplus(self.alpha) + 0.5
        x_squared = x.pow(2) + self.base_offset
        powered = torch.pow(x_squared, torch.clamp(alpha_safe, 0.5, 3.0))
        prob = powered / (powered.sum(dim=-1, keepdim=True) + self.min_value)
        return prob


class SASSP(nn.Module):
    """符号自适应稳定概率 (修复版)"""

    def __init__(self, alpha_init=1.0, beta_init=0.1):
        super().__init__()
        self.alpha = nn.Parameter(torch.tensor(alpha_init, dtype=torch.float32))
        self.beta = nn.Parameter(torch.tensor(beta_init, dtype=torch.float32))
        self.register_buffer('min_value', torch.tensor(1e-6))

    def forward(self, x):
        alpha_safe = F.softplus(self.alpha) + 0.5
        beta_safe = F.softplus(self.beta) + 0.01

        sign = torch.sign(x)
        abs_x = torch.abs(x) + beta_safe

        pos_mask = (sign >= 0).float()
        neg_mask = (sign < 0).float()

        pos_part = torch.pow(abs_x * pos_mask + 1.0, alpha_safe) * pos_mask
        neg_part = torch.pow(abs_x * neg_mask + 1.0, -alpha_safe) * neg_mask

        power = pos_part + neg_part + self.min_value
        prob = power / (torch.sum(power, dim=-1, keepdim=True) + self.min_value)
        return prob


class AutoFormulaProb(nn.Module):
    """自动公式生成器 (修复版)"""

    def __init__(self, hidden_dim=32):
        super().__init__()
        self.formula_learner = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.Softplus(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )
        self.temperature = nn.Parameter(torch.tensor(1.0))
        self.min_temp = 0.1

    def forward(self, x):
        x_reshaped = x.unsqueeze(-1)
        transformed = self.formula_learner(x_reshaped).squeeze(-1)
        temp_safe = torch.clamp(self.temperature, self.min_temp, 5.0)
        prob = F.softmax(transformed / temp_safe, dim=-1)
        return prob


# ==================== 2. 修复版模型定义 ====================

class CustomResNet18(nn.Module):
    def __init__(self, num_classes=10, prob_module='softmax'):
        super().__init__()
        self.backbone = torchvision.models.resnet18(weights=None, num_classes=num_classes)

        self.backbone.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.backbone.maxpool = nn.Identity()

        self.prob_module_type = prob_module
        self.prob_layer = self._create_prob_layer(prob_module)

        self.ema_alpha = 0.999
        self.register_buffer('running_mean', torch.zeros(num_classes))
        self.register_buffer('num_batches_tracked', torch.tensor(0, dtype=torch.long))

        self.grad_clip = 1.0
        self.training_step = 0

    def _create_prob_layer(self, prob_module):
        if prob_module == 'sasp':
            return SASP()
        elif prob_module == 'sassp':
            return SASSP()
        elif prob_module == 'autoformula':
            return AutoFormulaProb()
        else:
            return None

    def forward(self, x):
        logits = self.backbone(x)

        if self.training and torch.rand(1).item() < 0.05:
            return logits, F.softmax(logits, dim=-1)

        if self.prob_module_type in ['sasp', 'sassp', 'autoformula'] and self.prob_layer is not None:
            prob = self.prob_layer(logits)
        else:
            prob = F.softmax(logits, dim=-1)

        if self.training:
            self.training_step += 1
            if self.training_step % 100 == 0:
                with torch.no_grad():
                    current_mean = prob.mean(0)
                    if self.num_batches_tracked == 0:
                        self.running_mean.copy_(current_mean)
                    else:
                        self.running_mean.mul_(self.ema_alpha).add_(current_mean, alpha=1 - self.ema_alpha)
                    self.num_batches_tracked += 1

        return logits, prob

    def compile_model(self):
        if CONFIG_2026.compile_model:
            print(f"🔥 启用Torch编译优化...")
            try:
                return torch.compile(self, mode='default')
            except Exception as e:
                print(f"⚠️ 编译失败: {e}, 使用原生模式")
                return self
        return self


# ==================== 3. 修复版数据加载 ====================

def get_cifar10_loaders_optimized():
    """修复版数据加载器"""
    print(f"[⚡] 创建数据加载器 | Workers: {CONFIG_2026.num_workers}")

    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    print("[1/3] 📥 准备CIFAR-10数据集...")

    try:
        trainset = torchvision.datasets.CIFAR10(
            root='./data', train=True, download=True, transform=transform_train
        )
        testset = torchvision.datasets.CIFAR10(
            root='./data', train=False, download=True, transform=transform_test
        )
    except Exception as e:
        print(f"⚠️ 下载失败: {e}, 使用备用源...")
        torchvision.datasets.CIFAR10.url = 'https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz'
        trainset = torchvision.datasets.CIFAR10(
            root='./data', train=True, download=True, transform=transform_train
        )
        testset = torchvision.datasets.CIFAR10(
            root='./data', train=False, download=True, transform=transform_test
        )

    if CONFIG_2026.quick_test:
        print("  [⚡ 快速模式] 使用10%数据")
        train_size = len(trainset) // 10
        test_size = len(testset) // 10
        trainset, _ = random_split(trainset, [train_size, len(trainset) - train_size],
                                   generator=torch.Generator().manual_seed(42))
        testset, _ = random_split(testset, [test_size, len(testset) - test_size],
                                  generator=torch.Generator().manual_seed(42))

    # FIX: Windows优化 - 移除prefetch_factor当num_workers=0时
    train_kwargs = {
        'batch_size': CONFIG_2026.batch_size,
        'shuffle': True,
        'num_workers': CONFIG_2026.num_workers,
        'pin_memory': CONFIG_2026.pin_memory,
        'drop_last': True,
        'generator': torch.Generator().manual_seed(42)
    }
    test_kwargs = {
        'batch_size': CONFIG_2026.batch_size * 2,
        'shuffle': False,
        'num_workers': CONFIG_2026.num_workers,
        'pin_memory': CONFIG_2026.pin_memory,
    }

    # 只有当num_workers > 0时才添加这些参数
    if CONFIG_2026.num_workers > 0:
        train_kwargs['persistent_workers'] = CONFIG_2026.persistent_workers
        train_kwargs['prefetch_factor'] = CONFIG_2026.prefetch_factor
        test_kwargs['persistent_workers'] = CONFIG_2026.persistent_workers
        test_kwargs['prefetch_factor'] = CONFIG_2026.prefetch_factor

    trainloader = DataLoader(trainset, **train_kwargs)
    testloader = DataLoader(testset, **test_kwargs)

    print(f"[3/3] ✅ 加载完成!")
    print(f"  📊 训练: {len(trainset):,} | 测试: {len(testset):,}")
    print(f"  🔢 Batch: {CONFIG_2026.batch_size} | Workers: {CONFIG_2026.num_workers}")
    print("-" * 70)

    return trainloader, testloader


# ==================== 4. 修复版训练测试 ====================

def train_epoch_optimized(model, loader, optimizer, criterion, device, epoch, scaler=None):
    """修复版训练循环"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    progress_bar = tqdm(loader, desc=f'Epoch {epoch + 1}/{CONFIG_2026.epochs}',
                        leave=False, ncols=100)

    for batch_idx, (inputs, targets) in enumerate(progress_bar):
        # FIX: Windows CPU不使用non_blocking
        inputs = inputs.to(device, non_blocking=CONFIG_2026.pin_memory)
        targets = targets.to(device, non_blocking=CONFIG_2026.pin_memory)

        accum_steps = CONFIG_2026.gradient_accumulation
        is_accumulating = ((batch_idx + 1) % accum_steps != 0) and (batch_idx + 1 != len(loader))

        # FIX: 使用新的autocast API
        if USE_NEW_AMP and CONFIG_2026.mixed_precision:
            with autocast(device_type='cuda', enabled=True):
                logits, prob = model(inputs)

                if model.prob_module_type != 'softmax' and criterion == 'custom':
                    loss = F.nll_loss(torch.log(prob + 1e-6), targets)
                else:
                    loss = F.cross_entropy(logits, targets)

                loss = loss / accum_steps
        else:
            # CPU模式或无混合精度
            logits, prob = model(inputs)

            if model.prob_module_type != 'softmax' and criterion == 'custom':
                loss = F.nll_loss(torch.log(prob + 1e-6), targets)
            else:
                loss = F.cross_entropy(logits, targets)

            loss = loss / accum_steps

        if scaler:
            scaler.scale(loss).backward()
            if not is_accumulating:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), model.grad_clip)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
        else:
            loss.backward()
            if not is_accumulating:
                torch.nn.utils.clip_grad_norm_(model.parameters(), model.grad_clip)
                optimizer.step()
                optimizer.zero_grad()

        running_loss += loss.item() * accum_steps
        _, predicted = prob.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        current_acc = 100. * correct / total
        current_loss = running_loss / (batch_idx + 1)
        progress_bar.set_postfix({
            'loss': f'{current_loss:.4f}',
            'acc': f'{current_acc:.2f}%',
            'lr': f'{optimizer.param_groups[0]["lr"]:.6f}'
        })

    epoch_loss = running_loss / len(loader)
    epoch_acc = 100. * correct / total
    return epoch_loss, epoch_acc


def test_epoch_optimized(model, loader, device):
    """修复版测试循环"""
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, targets in tqdm(loader, desc='Testing', leave=False, ncols=80):
            # FIX: Windows CPU不使用non_blocking
            inputs = inputs.to(device, non_blocking=CONFIG_2026.pin_memory)
            targets = targets.to(device, non_blocking=CONFIG_2026.pin_memory)

            if USE_NEW_AMP and CONFIG_2026.mixed_precision:
                with autocast(device_type='cuda', enabled=True):
                    _, prob = model(inputs)
            else:
                _, prob = model(inputs)

            _, predicted = prob.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    acc = 100. * correct / total
    return acc


# ==================== 5. 修复版实验执行 ====================

class ExperimentManager:
    """修复版实验管理器"""

    def __init__(self):
        self.results = {}
        self.start_time = time.time()
        self._trainloader = None
        self._testloader = None

    def get_loaders(self):
        """FIX: 延迟加载并缓存数据加载器"""
        if self._trainloader is None:
            self._trainloader, self._testloader = get_cifar10_loaders_optimized()
        return self._trainloader, self._testloader

    def run_single_experiment(self, prob_module, device_id=0):
        """运行单个实验"""
        device = f'cuda:{device_id}' if torch.cuda.is_available() else 'cpu'
        if torch.cuda.is_available():
            torch.cuda.set_device(device_id)

        print(f"\n{'=' * 65}")
        print(f"🔥 开始训练 {prob_module.upper()} | 设备: {device}")
        print(f"{'=' * 65}")

        trainloader, testloader = self.get_loaders()

        model = CustomResNet18(num_classes=10, prob_module=prob_module).to(device)

        if CONFIG_2026.compile_model:
            model = model.compile_model()

        scaler = GradScaler() if CONFIG_2026.mixed_precision else None

        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=0.001,
            weight_decay=0.02,
            betas=(0.9, 0.999),
            eps=1e-8
        )

        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=10, T_mult=2
        )

        best_acc = 0
        epoch_times = []

        for epoch in range(CONFIG_2026.epochs):
            epoch_start = time.time()

            train_loss, train_acc = train_epoch_optimized(
                model, trainloader, optimizer, 'custom', device, epoch, scaler
            )

            test_acc = test_epoch_optimized(model, testloader, device)

            scheduler.step()

            if test_acc > best_acc:
                best_acc = test_acc
                best_state = {
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'best_acc': best_acc,
                }

            epoch_time = time.time() - epoch_start
            epoch_times.append(epoch_time)

            print(f"✅ Epoch {epoch + 1}/{CONFIG_2026.epochs} | "
                  f"Loss: {train_loss:.4f} | "
                  f"Train: {train_acc:.2f}% | "
                  f"Test: {test_acc:.2f}% | "
                  f"Best: {best_acc:.2f}% | "
                  f"Time: {epoch_time:.1f}s")

        total_time = time.time() - self.start_time
        avg_epoch_time = sum(epoch_times) / len(epoch_times)

        result = {
            'module': prob_module,
            'best_acc': best_acc,
            'total_time': total_time,
            'avg_epoch_time': avg_epoch_time,
            'device': device,
            'best_state': best_state
        }

        self.results[prob_module] = result

        print(f"\n🎉 {prob_module.upper()} 完成! 最佳: {best_acc:.2f}% | 耗时: {total_time:.1f}秒")
        return result

    def run_all_experiments(self):
        """顺序执行所有实验"""
        print(f"\n{'=' * 75}")
        print(f"🚀 实验集群启动 | 顺序执行模式")
        print(f"{'=' * 75}")

        for module in CONFIG_2026.experiment_modules:
            try:
                self.run_single_experiment(module, 0)
            except Exception as e:
                print(f"❌ 实验 {module} 失败: {str(e)}")
                import traceback
                traceback.print_exc()

        return self.results


# ==================== 6. 修复版主函数 ====================

def main_2026():
    """修复版主函数"""
    print("🎯 启动修复版训练框架...")

    if not torch.cuda.is_available():
        print("⚠️ 警告: 未检测到GPU! 训练速度将大幅降低")
        CONFIG_2026.epochs = min(CONFIG_2026.epochs, 5)

    pt_version = torch.__version__.split('+')[0]
    major, minor = map(int, pt_version.split('.')[:2])
    if major < 2:
        print(f"⚠️ 警告: PyTorch {pt_version} 版本过低，已禁用编译优化")
        CONFIG_2026.compile_model = False

    def set_seed(seed=42):
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        import random
        random.seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    set_seed(42)

    manager = ExperimentManager()
    start_time = time.time()
    results = manager.run_all_experiments()
    total_time = time.time() - start_time

    print(f"\n{'=' * 80}")
    print(f"📈 实验结果分析 | 总耗时: {total_time:.1f}秒")
    print(f"{'=' * 80}")

    if not results:
        print("❌ 无实验结果")
        return results

    baseline_acc = results.get('softmax', {}).get('best_acc', 0)

    print(f"{'方法':<15} | {'最佳准确率':<12} | {'总耗时':<10} | {'每轮平均':<10} | {'相对基线':<12}")
    print("-" * 80)

    for module in CONFIG_2026.experiment_modules:
        if module in results:
            data = results[module]
            rel_gain = data['best_acc'] - baseline_acc if module != 'softmax' and baseline_acc > 0 else 0
            marker = ' 🚀' if rel_gain > 1 else ' ✅' if rel_gain > 0 else ''

            print(f"{module:<15} | {data['best_acc']:<12.2f} | {data['total_time']:<10.1f} | "
                  f"{data['avg_epoch_time']:<10.2f} | {rel_gain:+.2f}%{marker}")

    print("-" * 80)

    if 'softmax' in results and len(results) > 1:
        best_module = max(
            [(m, d['best_acc']) for m, d in results.items()],
            key=lambda x: x[1]
        )
        print(f"\n🏆 最佳方法: {best_module[0].upper()} ({best_module[1]:.2f}%)")

    print(f"\n💡 技术状态:")
    print(f"   • 混合精度: {'✅' if CONFIG_2026.mixed_precision else '❌'}")
    print(f"   • Torch编译: {'✅' if CONFIG_2026.compile_model else '❌'}")
    print(f"   • 梯度累积: {CONFIG_2026.gradient_accumulation}x")
    print(f"   • 梯度裁剪: ✅ (修正位置)")

    return results


# ==================== 7. 程序入口 ====================

if __name__ == "__main__":
    try:
        results = main_2026()
    except KeyboardInterrupt:
        print("\n\n🛑 训练被用户中断")
    except Exception as e:
        print(f"\n❌ 错误: {str(e)}")
        import traceback

        traceback.print_exc()