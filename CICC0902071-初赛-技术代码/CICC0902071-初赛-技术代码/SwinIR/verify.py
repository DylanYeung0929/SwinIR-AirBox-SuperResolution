import onnx
onnx_model = onnx.load("SwinIR.onnx")
onnx.checker.check_model(onnx_model)

# 使用ONNX Runtime测试推理
import numpy as np
import onnxruntime as ort

ort_session = ort.InferenceSession("SwinIR.onnx")
input_onnx = np.random.randn(1, 3, 48, 48).astype(np.float32)
ort_inputs = {ort_session.get_inputs()[0].name: input_onnx}
ort_outputs = ort_session.run(None, ort_inputs)

# 对比PyTorch输出
with torch.no_grad():
    torch_output = model(torch.from_numpy(input_onnx)).numpy()

np.testing.assert_allclose(torch_output, ort_outputs[0], rtol=1e-3, atol=1e-5)