import os
import time
import requests
import pyarrow as pa
from dora import Node


def get_arm_data(url):
    """统一获取机械臂关节、夹爪、位姿数据，返回字典格式"""
    arm_data = {
        "jointstate": None,
        "gripper": None, 
        "pose": None,
        "success": False
    }

    try:
        # 获取关节角度
        joint_resp = requests.post(f"{url}getq", timeout=0.1)
        if joint_resp.status_code == 200:
            arm_data["jointstate"] = joint_resp.json()["q"]
        # 获取夹爪距离
        gripper_resp = requests.post(f"{url}get_gripper", timeout=0.1)
        if gripper_resp.status_code == 200:
            arm_data["gripper"] = [gripper_resp.json()["gripper"]]

        # 获取末端位姿
        pose_resp = requests.post(f"{url}getpos_euler", timeout=0.1)
        if pose_resp.status_code == 200:
            pose = pose_resp.json()["pose"]
            arm_data["pose"] = pose

        # 标记数据是否完整
        if all(v is not None for v in [arm_data["jointstate"], arm_data.get("gripper_raw"), arm_data["pose"]]):
            arm_data["success"] = True

    except requests.exceptions.RequestException as e:
        print(f"机械臂API请求失败: {e}")

    return arm_data

def main():
    arm_url = os.getenv("url", "http://127.0.0.2:5000/")
    print(f"机械臂API地址: {arm_url}")

    node = Node()


    # 事件循环
    for event in node:
        if event["type"] == "INPUT" and event["id"] == "tick":
            arm_data = get_arm_data(arm_url)
            
            combined_list = (
                arm_data["jointstate"]
                + arm_data["gripper"]
                + arm_data["pose"]
            )
            node.send_output(
                "jointstate",
                pa.array(combined_list, type=pa.float32()),
                {"timestamp": time.time_ns()}
            )

        elif event["type"] == "INPUT" and event["id"] == "stop":
            print("收到停止指令，停止机械臂...")

    print("Dora节点退出，清理资源...")

if __name__ == "__main__":
    main()