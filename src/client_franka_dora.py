import base64
import os
import time
import datetime
import numpy as np
import requests
import cv2
from dora import Node

# --- 1. 配置参数 ---
FRANKA_CONTROL_SERVER_URL = os.environ.get(
    "FRANKA_CONTROL_SERVER_URL", "http://127.0.0.2:5000/"
)
INFERENCE_SERVER_URL = os.environ.get(
    "INFERENCE_SERVER_URL", "http://172.16.17.77:8000/infer"
)
CONTROL_MODE = "eepose"
LOG_IMAGE_DIR = "./log_images_franka"

# --- 2. 工具函数 ---
def encode_image(img: np.ndarray) -> str:
    """将OpenCV图像编码为base64 PNG字符串"""
    _, buffer = cv2.imencode(".png", img)
    return base64.b64encode(buffer).decode("utf-8")

def setup_logging_directory():
    """创建日志目录（如果不存在）"""
    if not os.path.exists(LOG_IMAGE_DIR):
        os.makedirs(LOG_IMAGE_DIR)
        print(f"创建日志目录: {LOG_IMAGE_DIR}")

def get_franka_state(arm_url):
    """从控制服务器获取末端位姿和夹爪状态"""
    try:
        pose_res = requests.post(f"{arm_url}/getpos", timeout=5)
        pose = pose_res.json()["pose"]  # [x, y, z, qx, qy, qz, qw]
        
        gripper_res = requests.post(f"{arm_url}/get_gripper", timeout=5)
        gripper_width = gripper_res.json()["gripper"]
        gripper_state = 1 if gripper_width > 0.7 else 0  # 0-1归一化
        return pose + [gripper_state]
    except Exception as e:
        print(f"获取机械臂状态失败: {e}")
        return None

def control_franka_pose(arm_url, pose, gripper_state):
    """发送位姿和夹爪命令到控制服务器"""
    try:
        # 发送位姿命令
        requests.post(f"{arm_url}/pose", json={"arr": pose})
        # 发送夹爪命令
        gripper_pos = 270 if gripper_state > 0.5 else 30
        requests.post(f"{arm_url}/move_gripper", json={"gripper_pos": gripper_pos})
    except Exception as e:
        print(f"控制机械臂失败: {e}")

# --- 3. 主函数 ---
def main():
    # 初始化
    setup_logging_directory()
    arm_url = FRANKA_CONTROL_SERVER_URL
    print(f"机械臂API地址: {arm_url}")
    
    # 初始化机器人到初始位姿
    print("将机器人移动到初始位姿...")
    initial_pose = [0.305, 0.0, 0.481, 1.0, 0.0, 0.0, 0.0]
    control_franka_pose(arm_url, initial_pose, 1)  # 打开夹爪
    time.sleep(5)
    print("初始化完成，进入事件循环...")

    # 缓存输入数据
    cache = {
        "image_front": None,
        "image_wrist": None,
        "arm_data": None
    }

    node = Node()
    for event in node:
        # 处理输入事件
        if event["type"] == "INPUT":
            input_id = event["id"]
            value = event["value"]

            # 缓存前端相机图像
            if input_id == "image_front":
                # 假设输入为numpy数组 (BGR格式)
                cache["image_front"] = value
                print("收到前端相机图像")

            # 缓存腕部相机图像
            elif input_id == "image_wrist":
                cache["image_wrist"] = value
                print("收到腕部相机图像")

            # 缓存机械臂状态数据
            elif input_id == "arm_data":
                arm_data = value.to_pylist()
                ee_pose = arm_data[8:]
                gripper = arm_data[7]
                pose = (ee_pose + [gripper])
                cache["arm_data"] = pose
                print("收到机械臂状态数据")

            # 检查是否所有数据都已到位
            if all(v is not None for v in cache.values()):
                try:
                    process_cycle(arm_url, cache)
                except Exception as e:
                    print(f"处理周期出错: {e}")
                finally:
                    # 清空缓存，准备下一轮
                    cache = {k: None for k in cache}

        # 处理停止事件
        elif event["type"] == "INPUT" and event["id"] == "stop":
            print("收到停止指令，退出循环...")
            break

    # 退出清理
    print("Dora节点退出，清理资源...")

def process_cycle(arm_url, cache):
    """处理完整的控制周期"""
    print("\n" + "="*50)
    print("开始处理控制周期...")

    # 1. 解析缓存数据
    front_image = cache["image_front"]
    wrist_image = cache["image_wrist"]
    eef_pose_state = cache["arm_data"]

    print(f"当前机械臂状态: {np.round(eef_pose_state, 3)}")

    # 验证图像数据
    if front_image is None or wrist_image is None:
        print("警告: 图像数据不完整，跳过本轮")
        return

    # 2. 保存图像日志
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    cv2.imwrite(os.path.join(LOG_IMAGE_DIR, f"front_{timestamp}.png"), front_image)
    cv2.imwrite(os.path.join(LOG_IMAGE_DIR, f"wrist_{timestamp}.png"), wrist_image)
    print(f"已保存图像日志: {timestamp}")

    # 3. 编码图像并准备推理请求
    encoded_front = encode_image(front_image)
    encoded_wrist = encode_image(wrist_image)

    request_data = {
        "eef_pose": eef_pose_state,
        "instruction": "Pick up the orange and place it into the plate.",
        "images": {
            "cam_front": encoded_front,
            "cam_wrist": encoded_wrist,
        }
    }

    # 4. 发送请求到推理服务器
    print(f"发送请求到推理服务器: {INFERENCE_SERVER_URL}")
    try:
        response = requests.post(
            INFERENCE_SERVER_URL,
            json=request_data,
            timeout=60
        )
        response.raise_for_status()
        result = response.json()
        print("成功接收推理结果")
    except requests.exceptions.RequestException as e:
        print(f"推理服务器通信失败: {e}")
        return

    # 5. 解析并执行动作
    if CONTROL_MODE == "eepose":
        actions = result.get("eepose", [])
        if not actions:
            print("未收到动作指令，跳过")
            return

        for i, act in enumerate(actions):
            action = np.array(act, dtype=np.float32)
            if action.shape[0] != 8:
                print(f"动作维度错误 (预期8，实际{action.shape[0]})，跳过")
                continue

            target_pose = action[:7]
            target_gripper = action[7]
            print(f"执行动作 {i+1}/{len(actions)}: 位姿={np.round(target_pose, 3)}, 夹爪={target_gripper:.1f}")
            control_franka_pose(arm_url, target_pose.tolist(), target_gripper)
            time.sleep(0.05)  # 动作间隔

        print("动作序列执行完成")

    else:
        print(f"不支持的控制模式: {CONTROL_MODE}")

if __name__ == "__main__":
    main()
