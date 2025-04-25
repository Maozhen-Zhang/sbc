import torch
import torch.fft


class BatchLowPassFilter(torch.nn.Module):
    def __init__(self, args,filter_size=30):
        super().__init__()
        self.filter_size = filter_size
        self.args = args
        # 预先生滤波器掩码（与输入尺寸解耦）
        self.register_buffer("mask", None, persistent=False)

    def _create_mask(self, h, w):
        """动态创建与输入尺寸匹配的掩码"""
        cy, cx = h // 2, w // 2
        y = torch.arange(h, dtype=torch.float32, device=self.args.device)
        x = torch.arange(w, dtype=torch.float32, device=self.args.device)
        mask_y = (y >= cy - self.filter_size) & (y <= cy + self.filter_size)
        mask_x = (x >= cx - self.filter_size) & (x <= cx + self.filter_size)
        return mask_y[:, None] & mask_x[None, :]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        参数:
            x : 输入张量 [B, C, H, W]
        返回:
            滤波后的张量 [B, C, H, W]
        """
        B, C, H, W = x.shape

        # 动态生成掩码（兼容任意尺寸）
        if self.mask is None or self.mask.shape != (H, W):
            self.mask = self._create_mask(H, W).to(x.device)

        # 傅里叶变换（保持复数类型）
        x_fft = torch.fft.fft2(x, norm="ortho")
        x_fft_shifted = torch.fft.fftshift(x_fft, dim=(-2, -1))

        # 应用低通滤波（自动广播到批量和通道维度）
        filtered_fft = x_fft_shifted * self.mask[None, None, ...]

        # 逆变换
        x_ifft_shifted = torch.fft.ifftshift(filtered_fft, dim=(-2, -1))
        filtered = torch.fft.ifft2(x_ifft_shifted, norm="ortho").real

        # 数值稳定性处理
        filtered = torch.clamp(filtered, 0.0, 1.0) if x.dtype == torch.float32 else filtered
        return filtered.to(x.dtype)


# # 测试用例
# if __name__ == "__main__":
#     # 模拟输入（假设数值范围[0,1]）
#     batch = torch.rand(8, 3, 224, 224).cuda()  # 支持GPU
#
#     # 初始化滤波器
#     lowpass = BatchLowPassFilter(filter_size=30).to(self.args.device)
#
#     # 前向传播
#     filtered_batch = lowpass(batch)
#
#     print("输入形状:", batch.shape)
#     print("输出形状:", filtered_batch.shape)
#     print("数据类型保持:", filtered_batch.dtype == batch.dtype)