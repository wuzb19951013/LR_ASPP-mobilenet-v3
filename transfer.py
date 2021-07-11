import torch
from model import network

# 读入之前训练好的.pth模型
state = torch.load('G:/torch/SHM-netchange-v3/ckpt/human_matting/model/ckpt_lastest.pth',
                   map_location=lambda storage,
                   loc: storage
                   )
model = network.net()
model.load_state_dict(state['state_dict'], strict=True)

# example = torch.rand(1, 3, 320, 320).cuda()
# model.to(device)

# torch_out = torch.onnx.export(model,
#                               example,
#                               "new-mobilenetv2-128_S.onnx",
#                               verbose=True,
#                               export_params=True
#                               )

example = torch.rand(1, 3, 320, 320)

model = model.eval()

traced_script_module = torch.jit.trace(model, example)
output = traced_script_module(example)
print(traced_script_module)
# 导出trace后的模型
traced_script_module.save('SHM-modelv3-cat-front.pt')
