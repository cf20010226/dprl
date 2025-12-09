import onnx
import sys
import os
import math
from collections import Counter


def get_tensor_shape(tensor):
    """提取张量形状，支持动态维度 (batch_size)"""
    shape = []
    for dim in tensor.type.tensor_type.shape.dim:
        if dim.dim_value > 0:
            shape.append(str(dim.dim_value))
        elif dim.dim_param:
            shape.append(dim.dim_param)  # 通常是 'batch_size'
        else:
            shape.append("?")
    return f"[{', '.join(shape)}]"


def get_node_summary(node):
    """获取节点的关键信息（如卷积核大小）"""
    info = ""
    if node.op_type == "Conv":
        for attr in node.attribute:
            if attr.name == "kernel_shape":
                info = f"k{attr.ints}"
    elif node.op_type == "Gemm":
        # 全连接层通常没有太多属性需要展示，除非想看权重形状（比较复杂）
        info = "(Dense)"
    return info


def print_header(title):
    print("\n" + "=" * 60)
    print(f" {title}")
    print("=" * 60)


def main():
    # 1. 自动寻找根目录下的 .onnx 文件
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    target_file = None

    # 优先找 policy.onnx，找不到则找任何 onnx 文件
    possible_files = [f for f in os.listdir(base_dir) if f.endswith(".onnx")]
    if "policy.onnx" in possible_files:
        target_file = os.path.join(base_dir, "policy.onnx")
    elif len(possible_files) > 0:
        target_file = os.path.join(base_dir, possible_files[0])

    if not target_file:
        print(f"Error: 在 {base_dir} 目录下没有找到 .onnx 文件！")
        return

    print(f"正在分析模型: {target_file} ...")
    model = onnx.load(target_file)
    graph = model.graph

    # ================= 1. 输入与输出 =================
    print_header("1. 输入与输出 (I/O Tensors)")
    print(f"| {'名称 (Name)':<22} | {'形状 (Shape)':<22} | {'类型 (Type)':<10}")
    print(f"|{'-' * 24}|{'-' * 24}|{'-' * 10}|")

    # 输入
    for input_tensor in graph.input:
        name = input_tensor.name
        shape = get_tensor_shape(input_tensor)
        print(f"| {name:<22} | {shape:<22} | Input 📥")

    # 输出
    print(f"|{'-' * 24}|{'-' * 24}|{'-' * 10}|")
    for output_tensor in graph.output:
        name = output_tensor.name
        shape = get_tensor_shape(output_tensor)
        print(f"| {name:<22} | {shape:<22} | Output 📤")

    # ================= 2. 算子统计 =================
    print_header("2. 算子统计 (Operator Statistics)")
    ops = [node.op_type for node in graph.node]
    op_counts = Counter(ops)

    # 排序：数量多的在前
    sorted_ops = sorted(op_counts.items(), key=lambda x: x[1], reverse=True)

    for op, count in sorted_ops:
        # 简单的进度条可视化
        bar = "█" * (count * 2)
        if len(bar) > 20: bar = bar[:20]  # 限制长度
        print(f" 🔹 {op:<16} : {count:<3} {bar}")

    # ================= 3. 核心计算逻辑 =================
    print_header("3. 核心计算逻辑 (Neural Flow)")
    print(" (仅展示关键计算节点，省略 Reshape/Transpose/Constant)\n")

    print(" START")
    print("   │")

    # 定义我们要关注的“主要层”
    major_layers = ["Conv", "Gemm", "MatMul", "Relu", "Tanh", "Softmax", "Flatten", "Concat", "LSTM", "GRU"]

    # 简单的流式打印
    # 注意：ONNX 的 node 列表通常已经是拓扑排序的，直接遍历即可展示大致流程
    branch_detected = False

    for i, node in enumerate(graph.node):
        if node.op_type not in major_layers:
            continue

        summary = get_node_summary(node)

        # 简单的分支可视化逻辑
        prefix = "   │──"

        # 如果是 Concat，通常意味着特征融合
        if node.op_type == "Concat":
            print("   ▼")
            print(" 🔗 [Concat (Feature Fusion)]")
            print("   │")
            branch_detected = True
            continue

        # 如果是 Flatten，通常意味着从 CNN 转入 MLP
        if node.op_type == "Flatten":
            print(f"{prefix} 🔽 {node.op_type}")
            continue

        # 打印节点
        print(f"{prefix} {node.op_type} {summary}")

        # 如果是激活函数，稍微缩进一点表示它属于上一层
        if node.op_type in ["Relu", "Tanh", "Softmax"]:
            # 实际上在文本流中，直接列出更清晰，或者你可以选择不缩进
            pass

    print("   │")
    print(" 🏁 [End of Graph]")


if __name__ == "__main__":
    main()